import numpy as np
import argh

from agent_distribution import AgentDistribution
from gradient_estimation import GradientEstimator
from expected_gradient import ExpectedGradient
from expected_gradient_naive import ExpectedGradientNaive
from reporting import report_results
from utils import (
    compute_continuity_noise,
    fixed_point_interpolation_true_distribution,
    keep_theta_in_bounds,
    convert_to_polar_coordinates,
    convert_to_unit_vector,
)
from optimal_beta import optimal_beta_expected_policy_loss, expected_policy_loss


def learn_model(
    agent_dist,
    sigma,
    q,
    #    f,
    true_beta=None,
    learning_rate=0.05,
    max_iter=30,
    gradient_type="total_deriv",
    perturbation_s=0.1,
    perturbation_theta=0.1,
):
    if true_beta is None:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0

    theta = convert_to_polar_coordinates(true_beta)
    thetas = []
    s_eqs = []
    emp_losses = []
    for i in range(max_iter):
        theta = keep_theta_in_bounds(theta)
        # Get equilibrium cutoff
        #        s_eq = f(theta.item())
        beta = convert_to_unit_vector(theta)
        s_eq = agent_dist.quantile_fixed_point_true_distribution(beta, sigma, q)
        thetas.append(np.array(list(theta)).reshape(len(theta), 1))
        s_eqs.append(s_eq)
        if "naive_expected" in gradient_type:
            grad_exp = ExpectedGradientNaive(agent_dist, 
                       theta, 
                       s_eq, 
                       sigma, 
                       q, 
                       true_beta)
            loss = grad_exp.expected_loss()
            grad_theta = grad_exp.expected_gradient_loss_theta()

        else:
            grad_est = GradientEstimator(
                agent_dist,
                theta,
                s_eq,
                sigma,
                q,
                true_beta,
                perturbation_s_size=perturbation_s,
                perturbation_theta_size=perturbation_theta,
            )
            dic = grad_est.compute_total_derivative()
            loss = dic["loss"]
            if gradient_type == "total_deriv":
                grad_theta = dic["total_deriv"]
            elif gradient_type == "partial_deriv_loss_theta":
                grad_theta = dic["partial_deriv_loss_theta"]

        emp_losses.append(loss)
        print(
            "Loss: {}".format(loss),
            "Theta:{}".format(theta),
            "Gradient: {}".format(grad_theta),
        )
        theta -= grad_theta * learning_rate
        agent_dist.resample()

    return thetas, s_eqs, emp_losses


def create_generic_agent_dist(n, n_types, d):
    etas = np.random.uniform(3.0, 8.0, n_types * d).reshape(n_types, d, 1)
    gammas = np.random.uniform(0.05, 2.0, n_types * d).reshape(n_types, d, 1)
    dic = {"etas": etas, "gammas": gammas}
    agent_dist = AgentDistribution(n=n, d=d, n_types=n_types, types=dic, prop=None)
    return agent_dist


def create_challenging_agent_dist(n, n_types, d):
    gaming_type_etas = np.random.uniform(3.0, 5.0, int(n_types * d / 2)).reshape(
        int(n_types / 2), d, 1
    )
    gaming_type_gamma_one = np.random.uniform(0.01, 0.02, int(n_types / 2)).reshape(
        int(n_types / 2), 1, 1
    )
    gaming_type_gamma_two = np.random.uniform(
        10.0, 20.0, int(n_types * (d - 1) / 2)
    ).reshape(int(n_types / 2), d - 1, 1)
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


@argh.arg("--n", default=100000)
@argh.arg("--n_types", default=1)
@argh.arg("--d", default=2)
@argh.arg("--perturbation_s", default=0.1)
@argh.arg("--perturbation_theta", default=0.1)
@argh.arg("--learning_rate", default=1.0)
@argh.arg("--max_iter", default=500)
@argh.arg("--gradient_type", default="total_deriv")
@argh.arg("--seed", default=0)
@argh.arg("--save", default="results")
def main(
    n=100000,
    n_types=1,
    d=2,
    perturbation_s=0.1,
    perturbation_theta=0.1,
    learning_rate=1.0,
    max_iter=500,
    gradient_type="total_deriv",
    seed=0,
    save="results",
):
    np.random.seed(seed)

    agent_dist = create_challenging_agent_dist(n, n_types, d)
    sigma = compute_continuity_noise(agent_dist) + 0.05
    q = 0.7
    thetas, s_eqs, emp_losses = learn_model(
        agent_dist,
        sigma,
        q,
        true_beta=None,
        learning_rate=learning_rate,
        max_iter=max_iter,
        gradient_type=gradient_type,
        perturbation_s=perturbation_s,
        perturbation_theta=perturbation_theta,
    )
    if d == 2:
        f = fixed_point_interpolation_true_distribution(
            agent_dist, sigma, q, plot=False, savefig=None
        )

        min_loss, opt_beta, _, _, _ = optimal_beta_expected_policy_loss(
            agent_dist, sigma, f, plot=False
        )
        opt_theta = convert_to_polar_coordinates(opt_beta).item()

        results = {
            "n": n,
            "d": d,
            "n_types": n_types,
            "sigma": sigma,
            "q": q,
            "seed": seed,
            "perturbation_s": perturbation_s,
            "perturbation_theta": perturbation_theta,
            "opt_loss": min_loss,
            "opt_theta": opt_theta,
            "final_loss": expected_policy_loss(
                agent_dist, convert_to_unit_vector(thetas[-1]), s_eqs[-1], sigma
            ),
            "final_theta": thetas[-1][0].item(),
            "gradient_type": gradient_type,
        }
    else:
        results = {
            "n": n,
            "d": d,
            "n_types": n_types,
            "sigma": sigma,
            "q": q,
            "seed": seed,
            "perturbation_s": perturbation_s,
            "perturbation_theta": perturbation_theta,
            "final_loss": expected_policy_loss(
                agent_dist, thetas[-1], s_eqs[-1], sigma
            ),
            "final_theta": thetas[-1],
            "gradient_type": gradient_type,
        }
    assert len(thetas) == len(emp_losses)
    print(results)
    report_results(results, thetas, emp_losses, save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
