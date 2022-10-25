import numpy as np
import argh

from agent_distribution import AgentDistribution
from gradient_estimation_beta import GradientEstimator
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
    sigma,
    q,
    true_scores,
    learning_rate=0.05,
    max_iter=30,
    gradient_type="total_deriv",
    perturbation_s=0.1,
    perturbation_beta=0.1,
    beta_init=None
):
   

    betas = []
    s_eqs = []
    emp_losses = []
    if beta_init is None:
        beta = np.ones((agent_dist.d, 1))
        beta_norm = np.sqrt(np.sum(beta ** 2))
        beta /= beta_norm
    else:
        beta = beta_init
    for i in range(max_iter):
        s_eq = agent_dist.quantile_fixed_point_true_distribution(beta, sigma, q)
        betas.append(beta.copy())
        s_eqs.append(s_eq)
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
        if gradient_type == "total_deriv":
            dic = grad_est.compute_total_derivative()
            grad_beta = dic["total_deriv"]
            loss = dic["loss"]
        elif gradient_type == "partial_deriv_loss_beta":
            dic = grad_est.compute_partial_derivative()
            grad_beta = dic["partial_deriv_loss_beta"]
            loss = dic["loss"]
        else:
            assert False, "gradient type not valid"

        emp_losses.append(loss)

        print(
            "Loss: {}".format(loss),
            "Beta:{}".format(beta),
            "Gradient: {}".format(grad_beta * learning_rate),
        )
        beta -= grad_beta * learning_rate
        beta_norm = np.sqrt(np.sum(beta ** 2))
        beta /= beta_norm

    return betas, s_eqs, emp_losses


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


@argh.arg("--nels", default=False)
@argh.arg("--n", default=100000)
@argh.arg("--n_types", default=1)
@argh.arg("--d", default=2)
@argh.arg("--perturbation_s", default=0.1)
@argh.arg("--perturbation_beta", default=0.1)
@argh.arg("--learning_rate", default=1.0)
@argh.arg("--max_iter", default=500)
@argh.arg("--gradient_type", default="total_deriv")
@argh.arg("--seed", default=0)
@argh.arg("--save", default="results")
def main(
    nels=False,
    n=100000,
    n_types=1,
    d=2,
    perturbation_s=0.1,
    perturbation_beta=0.1,
    learning_rate=1.0,
    max_iter=500,
    gradient_type="total_deriv",
    seed=0,
    save="results",
):
    np.random.seed(seed)
    q = 0.7
    
    if nels:
        d=9
        prev_beta = np.ones(d)/np.sqrt(d)
        agent_dist, _, _, losses, sigma = get_agent_distribution_and_losses_nels(n, prev_beta, n_clusters=5, seed=0)
        true_scores = losses[agent_dist.n_agent_types].reshape(agent_dist.n, 1) 
        beta = np.random.normal(size=(agent_dist.d, 1))
        beta_norm = np.sqrt(np.sum(beta ** 2))
        beta /= beta_norm
        betas, s_eqs, emp_losses = learn_model(agent_dist, 
                                               sigma, 
                                               q,
                                               true_scores=true_scores,
                                               learning_rate=learning_rate,
                                               max_iter=max_iter,
                                               gradient_type=gradient_type,
                                               perturbation_s=perturbation_s,
                                               perturbation_beta=perturbation_beta,
                                               beta_init=beta
                                              )

        final_loss = emp_losses[-1]
    
    else:
        
        agent_dist = create_generic_agent_dist(n, n_types, d)
        sigma = compute_continuity_noise(agent_dist) + 0.05
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0
        etas = agent_dist.get_etas()
        true_scores = np.array(
            [-np.matmul(self.true_beta.T, eta).item() for eta in etas]
        ).reshape(agent_dist.n, 1)

        betas, s_eqs, emp_losses = learn_model(
            agent_dist,
            sigma,
            q,
            true_scores=true_scores,
            learning_rate=learning_rate,
            max_iter=max_iter,
            gradient_type=gradient_type,
            perturbation_s=perturbation_s,
            perturbation_beta=perturbation_beta,
        )
        final_loss = expected_policy_loss(
            agent_dist, np.array(betas[-1]).reshape(d, 1), s_eqs[-1], sigma
        )
 
    results = {
        "n": n,
        "d": d,
        "n_types": n_types,
        "sigma": sigma,
        "q": q,
        "seed": seed,
        "perturbation_s": perturbation_s,
        "perturbation_beta": perturbation_beta,
        "final_loss": final_loss,
        "final_beta": list(betas[-1].flatten()),
        "gradient_type": gradient_type,
    }
    assert len(betas) == len(emp_losses)
    print(results)
    report_results(results, betas, emp_losses, save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
