import numpy as np
import argh
import matplotlib.pyplot as plt
from scipy.misc import derivative

from agent_distribution import AgentDistribution
from gradient_estimation import GradientEstimator
from utils import compute_continuity_noise, fixed_point_interpolation_true_distribution
from reparametrized_gradient import expected_total_derivative
from expected_gradient import ExpectedGradient
from metrics import mse


def compute_metrics_vs_perturbation_s(
    agent_dist, sigma, q, f, derivatives, true_beta=None, savefig=None
):
    perturbation_s_sizes = np.linspace(0.01, 0.20, 10)
    default_theta_size = 0.1
    results = [{"perturbation_s_size": perturb_s} for perturb_s in perturbation_s_sizes]
    derivs_exp_all = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for deriv in derivatives:
        for res in results:
            res[deriv] = []
    for theta in thetas:
        s = f(theta)
        grad_exp = ExpectedGradient(agent_dist, theta, s, sigma, true_beta)
        val = grad_exp.compute_total_derivative()
        derivs_exp_all.append(val)

        for perturb_s in perturbation_s_sizes:
            grad_est = GradientEstimator(
                agent_dist,
                theta,
                s,
                sigma,
                q,
                true_beta,
                perturbation_s_size=perturb_s,
                perturbation_theta_size=default_theta_size,
            )
            hat_val = grad_est.compute_total_derivative()
            for res in results:
                if res["perturbation_s_size"] == perturb_s:
                    for deriv in derivatives:
                        res[deriv].append(hat_val[deriv])
    final_results = []

    for perturb_s in perturbation_s_sizes:
        final_res = {}
        for deriv in derivatives:
            mse_key = deriv + "_mse"
            deriv_exp = [dic[deriv] for dic in derivs_exp_all]
            for res in results:
                if res["perturbation_s_size"] == perturb_s:
                    deriv_emp = res[deriv]
                    mse_est = mse(deriv_emp, deriv_exp)
                    final_res[mse_key] = mse_est
        final_results.append(final_res)

    fig, ax = plt.subplots(
        1, len(final_results[0].keys()), figsize=(12 * len(final_results[0]), 5)
    )

    for i, key in enumerate(final_results[0].keys()):
        ys = [dic[key] for dic in final_results]
        ax[i].plot(perturbation_s_sizes, ys)
        ax[i].set_xlabel("size of Perturbation to s")
        ax[i].set_ylabel(key)
        ax[i].set_title("size of Perturbation to s vs. {}".format(key))

    if savefig is not None:
        title = savefig.split("/")[-1]
        plt.suptitle(title)
        plt.savefig(savefig)
    plt.show()
    plt.close()


def compute_metrics_vs_perturbation_theta(
    agent_dist, sigma, q, f, derivatives, true_beta=None, savefig=None
):
    perturbation_theta_sizes = np.linspace(0.01, 0.20, 10)
    default_s_size = 0.1
    results = [
        {"perturbation_theta_size": perturb_s} for perturb_s in perturbation_theta_sizes
    ]
    derivs_exp_all = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for deriv in derivatives:
        for res in results:
            res[deriv] = []
    for theta in thetas:
        s = f(theta)
        grad_exp = ExpectedGradient(agent_dist, theta, s, sigma, true_beta)
        val = grad_exp.compute_total_derivative()
        derivs_exp_all.append(val)

        for perturb_theta in perturbation_theta_sizes:
            grad_est = GradientEstimator(
                agent_dist,
                theta,
                s,
                sigma,
                q,
                true_beta,
                perturbation_s_size=default_s_size,
                perturbation_theta_size=perturb_theta,
            )
            hat_val = grad_est.compute_total_derivative()
            for res in results:
                if res["perturbation_theta_size"] == perturb_theta:
                    for deriv in derivatives:
                        res[deriv].append(hat_val[deriv])
    final_results = []

    for perturb_theta in perturbation_theta_sizes:
        final_res = {}
        for deriv in derivatives:
            mse_key = deriv + "_mse"
            deriv_exp = [dic[deriv] for dic in derivs_exp_all]
            for res in results:
                if res["perturbation_theta_size"] == perturb_theta:
                    deriv_emp = res[deriv]
                    mse_est = mse(deriv_emp, deriv_exp)
                    final_res[mse_key] = mse_est
        final_results.append(final_res)

    fig, ax = plt.subplots(
        1, len(final_results[0].keys()), figsize=(12 * len(final_results[0]), 5)
    )

    for i, key in enumerate(final_results[0].keys()):
        ys = [dic[key] for dic in final_results]
        ax[i].plot(perturbation_theta_sizes, ys)
        ax[i].set_xlabel("size of Perturbation to theta")
        ax[i].set_ylabel(key)
        ax[i].set_title("size of Perturbation to theta vs. {}".format(key))

    if savefig is not None:
        title = savefig.split("/")[-1]
        plt.suptitle(title)
        plt.savefig(savefig)
    plt.show()
    plt.close()


@argh.arg("--n", default=100000)
@argh.arg("--perturbation_s", default=False)
@argh.arg("--perturbation_theta", default=False)
@argh.arg("--save", default="results")
@argh.arg("--total_deriv", default=False)
@argh.arg("--partial_deriv_loss_theta", default=False)
@argh.arg("--partial_deriv_pi_theta", default=False)
@argh.arg("--partial_deriv_pi_s", default=False)
@argh.arg("--partial_deriv_loss_s", default=False)
@argh.arg("--partial_deriv_s_theta", default=False)
@argh.arg("--density", default=False)
def main(
    n=100000,
    perturbation_s=True,
    perturbation_theta=True,
    save="results",
    total_deriv=False,
    partial_deriv_loss_theta=False,
    partial_deriv_pi_theta=False,
    partial_deriv_pi_s=False,
    partial_deriv_loss_s=False,
    partial_deriv_s_theta=False,
    density=False,
):
    np.random.seed(0)

    n_types = 1
    d = 2
    etas = np.random.uniform(3.0, 8.0, n_types * d).reshape(n_types, d, 1)
    gammas = np.random.uniform(0.05, 0.1, n_types * d).reshape(n_types, d, 1)
    dic = {"etas": etas, "gammas": gammas}
    agent_dist = AgentDistribution(n=n, d=d, n_types=n_types, types=dic, prop=None)
    #    sigma = compute_continuity_noise(agent_dist)
    true_beta = np.array([1.0, 0.0]).reshape(2, 1)

    derivatives_to_plot = []
    if total_deriv:
        derivatives_to_plot.append("total_deriv")
    if partial_deriv_loss_theta:
        derivatives_to_plot.append("partial_deriv_loss_theta")
    if partial_deriv_loss_s:
        derivatives_to_plot.append("partial_deriv_loss_s")
    if partial_deriv_pi_theta:
        derivatives_to_plot.append("partial_deriv_pi_theta")
    if partial_deriv_pi_s:
        derivatives_to_plot.append("partial_deriv_pi_s")
    if partial_deriv_s_theta:
        derivatives_to_plot.append("partial_deriv_s_theta")
    if density:
        derivatives_to_plot.append("density_estimate")
    sigma = 2.0
    q = 0.7
    f = fixed_point_interpolation_true_distribution(
        agent_dist, sigma, q, plot=False, savefig=None
    )
    print(derivatives_to_plot)
    if perturbation_s:
        print("perturbation s")
        compute_metrics_vs_perturbation_s(
            agent_dist,
            sigma,
            q,
            f,
            savefig="results/figures_vs_perturbation_s/{}".format(save),
            true_beta=true_beta,
            derivatives=derivatives_to_plot,
        )
    if perturbation_theta:
        print("perturbation theta")
        compute_metrics_vs_perturbation_theta(
            agent_dist,
            sigma,
            q,
            f,
            savefig="results/figures_vs_perturbation_theta/{}".format(save),
            true_beta=true_beta,
            derivatives=derivatives_to_plot,
        )


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()
