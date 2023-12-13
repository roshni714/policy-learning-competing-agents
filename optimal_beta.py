import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from agent_distribution import AgentDistribution
from utils import (
    compute_continuity_noise,
    compute_contraction_noise,
    compute_score_bounds,
    convert_to_unit_vector,
)


def empirical_policy_loss(agent_dist, beta, s, sigma, q, true_beta=None):
    """Method that returns the empirical policy loss incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters (N, 1) array
    s -- threshold (float)
    q -- quantile
    sigma -- standard deviation of the noise (float)
    true_beta -- (N, 1) array

    Returns:
    loss -- empirical policy loss
    """
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.0

    true_agent_types = agent_dist.n_agent_types
    etas = agent_dist.get_etas()
    true_scores = np.array(
        [np.matmul(true_beta.T, eta).item() for eta in etas]
    ).reshape(agent_dist.n, 1)

    br_dist = agent_dist.best_response_score_distribution(beta, s, sigma)
    n_br = br_dist[agent_dist.n_agent_types].reshape(agent_dist.n, 1)

    noisy_scores = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(
        agent_dist.n, 1
    )
    noisy_scores += n_br
    x = np.quantile(noisy_scores, q).item()
    loss = -np.mean(true_scores * (noisy_scores >= x))
    return loss


def optimal_beta_empirical_policy_loss(
    agent_dist, sigma, q, f, true_beta=None, plot=False, savefig=None
):
    """Method returns the model parameters that minimize the empirical policy loss.

    Keyword args:
    agent_dist -- AgentDistribution
    sigma -- standard deviation of noise distribution
    q -- quantile
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)
    true_beta -- optional ideal model
    plot -- optional plotting
    savefig -- path to save figure

    Returns:
    min_loss -- minimum loss (float)
    opt_beta -- beta that minimizes the loss (2, 1) array
    opt_s_beta -- optimal threshold (float)
    """
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    thetas = np.linspace(0.0, 2 * np.pi, 50)
    losses = []
    for theta in thetas:
        beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
        loss = empirical_policy_loss(
            agent_dist, beta, f(theta), sigma, q, true_beta=true_beta
        )
        losses.append(loss)

    idx = np.argmin(losses)
    min_loss = losses[idx]
    opt_beta = np.array([np.cos(thetas[idx]), np.sin(thetas[idx])]).reshape(2, 1)
    opt_s_beta = f(thetas[idx])
    if plot:
        plt.plot(thetas, losses)
        plt.xlabel("Theta (Represents Beta)")
        plt.ylabel("Empirical Loss")
        plt.title("Empirical Loss Incurred at Different Beta")
        if savefig:
            plt.savefig(savefig)
        plt.show()
        plt.close()

    return min_loss, opt_beta, opt_s_beta, thetas, losses


def expected_policy_loss(agent_dist, beta, s, sigma, true_scores=None):
    """Method that computes the expected policy loss of deploying a particular model.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters
    s -- threshold
    sigma -- standard deviation of noise distribution
    true_beta -- optional ideal model


    Returns:
    loss -- expected policy loss at beta
    """
    dim = agent_dist.d

    if true_scores is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.0
        true_scores = np.array(
            [-np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]
        ).reshape(agent_dist.n_types, 1)

    #    bounds = compute_score_bounds(beta)
    br_dist = agent_dist.best_response_score_distribution(beta, s, sigma)
    z = s - br_dist.reshape(agent_dist.n_types, 1)
    prob = 1 - norm.cdf(x=z, loc=0.0, scale=sigma)
    product = true_scores * prob * agent_dist.prop.reshape(agent_dist.n_types, 1)
    return np.sum(product).item()  # np.sum(product).item()


def optimal_beta_expected_policy_loss(
    agent_dist, sigma, f, true_beta=None, plot=False, savefig=None
):
    """Method returns the model parameters that minimize the expected policy loss.

    Keyword args:
    agent_dist -- AgentDistribution
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)
    true_beta -- optional ideal model
    plot -- optional plotting
    savefig -- path to save figure
    """
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    thetas = np.linspace(0.0, 2 * np.pi, 50)
    losses = []
    for theta in thetas:
        s_eq = f(theta)
        loss = expected_policy_loss(
            agent_dist,
            convert_to_unit_vector(np.array([theta]).reshape(1, 1)),
            s_eq,
            sigma,
            true_beta,
        )
        losses.append(loss)

    idx = np.argmin(losses)
    min_loss = losses[idx]
    opt_beta = np.array([np.cos(thetas[idx]), np.sin(thetas[idx])]).reshape(2, 1)
    opt_s_beta = f(thetas[idx])
    if plot:
        plt.plot(thetas, losses)
        plt.xlabel("Theta (Represents Beta)")
        plt.ylabel("Expected Loss")
        plt.title("Expected Loss Incurred at Different Beta")
        if savefig:
            plt.savefig(savefig)
        plt.show()
        plt.close()

    return min_loss, opt_beta, opt_s_beta, thetas, losses
