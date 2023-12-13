import numpy as np
import argh

from agent_distribution import AgentDistribution
from gradient_estimation_beta import GradientEstimator
from expected_gradient_beta_naive import ExpectedGradientBetaNaive
from empirical_welfare_maximization import EmpiricalWelfareMaximization

from reporting import report_results
from utils import (
    compute_continuity_noise,
    fixed_point_interpolation_true_distribution,
    keep_theta_in_bounds,
    convert_to_polar_coordinates,
    convert_to_unit_vector,
)
from optimal_beta import optimal_beta_expected_policy_loss, expected_policy_loss

from data_gen import get_agent_distribution_and_losses_nels


def learn_model(
    agent_dist,
    nels,
    sigma,
    q,
    true_scores=None,
    learning_rate=0.05,
    max_iter=30,
    method="total_deriv",
    perturbation_s=0.1,
    perturbation_beta=0.1,
    beta_init=None,
    loss_type=None,
    month_attended_losses=None,
    eta_losses=None,
    socio_econ_losses=None,
):

    res_dic = {
        "params": [],
        "s_eqs": [],
        "emp_losses": [],
        "mag_pg": [],
        "mag_eg": [],
        "mag_mg": [],
    }
    if beta_init is None:
        beta = np.random.uniform(-1, 1, size=(agent_dist.d, 1))
        beta_norm = np.sqrt(np.sum(beta**2))
        beta /= beta_norm
    else:
        beta = beta_init
    for i in range(max_iter):
        true_scores = get_n_true_scores(
            agent_dist,
            nels,
            loss_type,
            month_attended_losses,
            eta_losses,
            socio_econ_losses,
        )
        #        import pdb
        #        pdb.set_trace()
        s_eq = agent_dist.quantile_fixed_point_true_distribution(beta, sigma, q)
        res_dic["params"].append(beta.copy())
        res_dic["s_eqs"].append(s_eq)

        #        if "naive" in method:
        #            grad_exp = ExpectedGradientBetaNaive(
        #                agent_dist, beta, s_eq, sigma, q, true_scores
        #            )
        #            grad_beta = grad_exp.expected_gradient_loss_beta()
        #            loss = grad_exp.empirical_loss()
        #        else:
        grad_est = GradientEstimator(
            agent_dist,
            beta,
            s_eq,
            sigma,
            q,
            true_scores,
            perturbation_s_size=perturbation_s,
            perturbation_beta_size=perturbation_beta,
        )
        if method == "total_deriv":
            dic = grad_est.compute_total_derivative()
            grad_beta = dic["total_deriv"]
            loss = dic["loss"]
        elif method == "partial_deriv_loss_beta":
            dic = grad_est.compute_partial_derivative()
            grad_beta = dic["partial_deriv_loss_beta"]
            loss = dic["loss"]
        else:
            assert False, "gradient type not valid"

        res_dic["emp_losses"].append(loss)
        res_dic["mag_mg"].append(
            np.sqrt(np.sum(dic["partial_deriv_loss_beta"] ** 2)).item()
        )
        res_dic["mag_pg"].append(np.sqrt(np.sum(dic["total_deriv"] ** 2)).item())
        res_dic["mag_eg"].append(
            np.sqrt(
                np.sum((dic["partial_deriv_loss_s"] * dic["partial_deriv_s_beta"]) ** 2)
            ).item()
        )

        print(
            "Loss: {}".format(loss),
            "Beta:{}".format(beta),
            "Gradient: {}".format(grad_beta * learning_rate),
        )
        beta -= grad_beta * learning_rate
        beta_norm = np.sqrt(np.sum(beta**2))
        beta /= beta_norm
        agent_dist.resample()

    return res_dic


def create_generic_agent_dist(n, n_types, d):
    etas = np.random.uniform(3.0, 8.0, n_types * d).reshape(n_types, d, 1)
    gammas = np.random.uniform(0.05, 5.0, n_types * d).reshape(n_types, d, 1)
    dic = {"etas": etas, "gammas": gammas}
    agent_dist = AgentDistribution(n=n, d=d, n_types=n_types, types=dic, prop=None)
    return agent_dist


def create_challenging_agent_dist(n, n_types, d):
    gaming_type_etas = np.random.uniform(3.0, 5.0, int(n_types * d / 2)).reshape(
        int(n_types / 2), d, 1
    )
    gaming_type_gamma_one = np.random.uniform(
        0.01, 0.02, int(n_types / 2) * int(d / 2)
    ).reshape(int(n_types / 2), int(d / 2), 1)
    gaming_type_gamma_two = np.random.uniform(
        10.0, 20.0, int(n_types * (d - int(d / 2)) / 2)
    ).reshape(int(n_types / 2), d - int(d / 2), 1)
    gaming_type_gammas = np.hstack((gaming_type_gamma_one, gaming_type_gamma_two))
    natural_type_etas = np.random.uniform(5.0, 7.0, int(n_types * d / 2)).reshape(
        int(n_types / 2), d, 1
    )
    natural_type_gammas = np.random.uniform(10.0, 20.0, int(n_types * d / 2)).reshape(
        int(n_types / 2), d, 1
    )
    etas = np.vstack((gaming_type_etas, natural_type_etas))
    gammas = np.vstack((gaming_type_gammas, natural_type_gammas))
    dic = {"etas": etas, "gammas": gammas}
    agent_dist = AgentDistribution(n=n, d=d, n_types=n_types, types=dic, prop=None)
    return agent_dist


def get_n_true_scores(
    agent_dist,
    nels,
    loss_type=None,
    month_attended_losses=None,
    eta_losses=None,
    socio_econ_losses=None,
):
    if nels:
        if loss_type == "etas":
            losses = eta_losses
        elif loss_type == "months_attended":
            losses = month_attended_losses
        elif loss_type == "socio_econ":
            losses = socio_econ_losses
        else:
            assert False
        true_scores = losses[agent_dist.n_agent_types].reshape(agent_dist.n, 1)
    else:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0
        etas = agent_dist.get_etas()
        true_scores = np.array(
            [-np.matmul(true_beta.T, eta).item() for eta in etas]
        ).reshape(agent_dist.n, 1)

    return true_scores


def get_true_scores(
    agent_dist,
    nels,
    loss_type=None,
    month_attended_losses=None,
    eta_losses=None,
    socio_econ_losses=None,
):
    if nels:
        if loss_type == "etas":
            true_scores = eta_losses
        elif loss_type == "months_attended":
            true_scores = month_attended_losses
        elif loss_type == "socio_econ":
            true_scores = socio_econ_losses
        else:
            assert False
        true_scores = true_scores.reshape(agent_dist.n_types, 1)
    else:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0
        etas = agent_dist.types["etas"]
        true_scores = np.array(
            [-np.matmul(true_beta.T, eta).item() for eta in etas]
        ).reshape(agent_dist.n_types, 1)

    return true_scores


@argh.arg("--nels", default=False)
@argh.arg("--n", default=100000)
@argh.arg("--n_types", default=1)
@argh.arg("--d", default=2)
@argh.arg("--perturbation_s", default=0.1)
@argh.arg("--perturbation_beta", default=0.1)
@argh.arg("--learning_rate", default=1.0)
@argh.arg("--max_iter", default=500)
@argh.arg("--method", default="total_deriv")
@argh.arg("--seed", default=0)
@argh.arg("--save_dir", default="results")
@argh.arg("--save", default="results")
@argh.arg("--loss_type", default=None)
def main(
    nels=False,
    n=100000,
    n_types=1,
    d=2,
    perturbation_s=0.1,
    perturbation_beta=0.1,
    learning_rate=1.0,
    max_iter=500,
    method="total_deriv",
    seed=0,
    save_dir="results",
    save="results",
    loss_type=None,
):
    np.random.seed(seed)
    q = 0.7
    prev_beta = np.ones(d) / np.sqrt(d)

    print("Data generation started...")
    if nels:
        print("Getting NELS data...")
        d = 9
        (
            agent_dist,
            _,
            _,
            month_attended_losses,
            eta_losses,
            socio_econ_losses,
            sigma,
        ) = get_agent_distribution_and_losses_nels(
            n, prev_beta, n_clusters=8, seed=seed
        )

        prev_beta = prev_beta.reshape(agent_dist.d, 1)

    else:
        print("Computing agent distribution...")
        agent_dist = create_generic_agent_dist(n, n_types, d)
        sigma = compute_continuity_noise(agent_dist) + 0.05
        month_attended_losses = None
        eta_losses = None
        socio_econ_losses = None
        loss_type = None

    if method == "ewm":
        true_scores = get_n_true_scores(
            agent_dist,
            nels,
            loss_type,
            month_attended_losses,
            eta_losses,
            socio_econ_losses,
        )
        ewm = EmpiricalWelfareMaximization(agent_dist, sigma, q, true_scores)
        final_beta = ewm.estimate_beta()
        s_eq = agent_dist.quantile_fixed_point_true_distribution(final_beta, sigma, q)

        final_loss = expected_policy_loss(
            agent_dist,
            final_beta.reshape(agent_dist.d, 1),
            s_eq,
            sigma,
            true_scores=get_true_scores(
                agent_dist,
                nels,
                loss_type,
                month_attended_losses,
                eta_losses,
                socio_econ_losses,
            ),
        )
        final_s = s_eq
        final_beta = list(final_beta.flatten())
        res_dic = None
    else:
        res_dic = learn_model(
            agent_dist,
            nels=nels,
            sigma=sigma,
            q=q,
            learning_rate=learning_rate,
            max_iter=max_iter,
            method=method,
            perturbation_s=perturbation_s,
            perturbation_beta=perturbation_beta,
            loss_type=loss_type,
            month_attended_losses=month_attended_losses,
            eta_losses=eta_losses,
            socio_econ_losses=socio_econ_losses,
        )

        final_loss = expected_policy_loss(
            agent_dist,
            res_dic["params"][-1].reshape(agent_dist.d, 1),
            res_dic["s_eqs"][-1],
            sigma,
            true_scores=get_true_scores(
                agent_dist,
                nels,
                loss_type,
                month_attended_losses,
                eta_losses,
                socio_econ_losses,
            ),
        )
        final_beta = list(res_dic["params"][-1].flatten())
        assert len(res_dic["params"]) == len(res_dic["emp_losses"])

    results = {
        "n": n,
        "d": d,
        "n_types": n_types,
        "sigma": sigma,
        "q": q,
        "seed": seed,
        #        "perturbation_s": perturbation_s,
        #        "perturbation_beta": perturbation_beta,
        "final_loss": final_loss,
        "final_beta": final_beta,
        "method": method,
    }
    print(results)
    report_results(save_dir, results, res_dic=res_dic, save=save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
