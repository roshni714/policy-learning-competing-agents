import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli, norm, gaussian_kde
from scipy.misc import derivative
import tqdm

from utils import compute_score_bounds, convert_to_unit_vector


def plot_total_derivative(
    agent_dist,
    sigma,
    q,
    f,
    perturbation_s_size=0.1,
    perturbation_theta_size=0.1,
    true_beta=None,
    savefig=None,
):
    deriv = []
    deriv_emp = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in thetas:
        s = f(theta)
        val = expected_total_derivative(agent_dist, theta, s, sigma, true_beta)
        hat_val = empirical_total_derivative(
            agent_dist,
            theta,
            s,
            sigma,
            q,
            true_beta,
            perturbation_s_size=perturbation_s_size,
            perturbation_theta_size=perturbation_theta_size,
        )
        deriv.append(val)
        deriv_emp.append(hat_val)
    plt.plot(thetas, deriv_emp, label="empirical")
    plt.plot(thetas, deriv, label="expectation")
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("total deriv dL/dtheta")
    if savefig is not None:
        title = savefig.split("/")[-1]
        plt.title("{} : total deriv dL/dtheta vs. theta ".format(title))
        plt.savefig(savefig)
    else:
        plt.title("total deriv dL/dtheta vs theta")
    plt.show()
    plt.close()


def empirical_gradient_loss_s(
    agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.05
):
    """Method that returns the empirical gradient of loss wrt to s incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters (N, 1) array
    s -- threshold (float)
    sigma -- standard deviation of noise distribution (float)
    q -- quantile
    true_beta -- (N, 1) array

    Returns:
    gamma_loss_s -- empirical gradient dL/ds
    """
    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.0

    perturbations = (
        2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) - 1
    ) * perturbation_size
    scores = []

    bounds = compute_score_bounds(beta)
    interpolators = []

    for agent in agent_dist.agents:
        interpolators.append(agent.br_score_function_s(beta, sigma))

    for i in range(agent_dist.n):
        s_perturbed = np.clip(s + perturbations[i], a_min=bounds[0], a_max=bounds[1])
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](s_perturbed)
        scores.append(br_score.item())

    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = scores + noise
    perturbed_noisy_scores = noisy_scores - perturbations

    # Compute loss
    treatments = perturbed_noisy_scores >= np.quantile(perturbed_noisy_scores, q)
    loss_vector = treatments * np.array(
        [-np.matmul(true_beta.T, eta).item() for eta in agent_dist.get_etas()]
    ).reshape(agent_dist.n, 1)

    Q = np.matmul(perturbations.T, perturbations)
    gamma_loss_s = np.linalg.solve(Q, np.matmul(perturbations.T, loss_vector))
    return gamma_loss_s.item()


def expected_gradient_loss_s(agent_dist, theta, s, sigma, true_beta=None):
    """Method computes partial loss/partial s.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)

    Returns:
    d_pi_d_s -- expected gradient wrt to s of policy loss 

    """
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    br_dist, grad_s_dist = agent_dist.br_gradient_s_distribution(beta, s, sigma)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.0

    z = s - np.array([np.matmul(beta.T, x) for x in br_dist]).reshape(len(br_dist), 1)

    prob = norm.pdf(z, loc=0.0, scale=sigma)
    vec = np.array(
        [1 - np.matmul(beta.T, grad_s_dist[i]).item() for i in range(len(br_dist))]
    ).reshape(agent_dist.n_types, 1)
    true_scores = np.array(
        [np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]
    ).reshape(agent_dist.n_types, 1)
    res = prob * vec * true_scores * agent_dist.prop.reshape(agent_dist.n_types, 1)

    d_loss_d_s = np.sum(res)
    return d_loss_d_s.item()


def expected_total_derivative(agent_dist, theta, s, sigma, true_beta=None):
    gamma_loss_s = expected_gradient_loss_s(agent_dist, theta, s, sigma, true_beta)
    gamma_loss_theta = expected_gradient_loss_theta(
        agent_dist, theta, s, sigma, true_beta
    )
    gamma_s_theta = expected_gradient_s_theta(agent_dist, theta, s, sigma)
    total_derivative = (gamma_loss_s * gamma_s_theta) + gamma_loss_theta
    return total_derivative


def empirical_total_derivative(
    agent_dist,
    theta,
    s,
    sigma,
    q,
    perturbation_s_size,
    perturbation_theta_size,
    true_beta=None,
):
    gamma_loss_s = empirical_gradient_loss_s(
        agent_dist,
        theta,
        s,
        sigma,
        q,
        perturbation_size=perturbation_s_size,
        true_beta=true_beta,
    )
    gamma_loss_theta = empirical_gradient_loss_theta(
        agent_dist,
        theta,
        s,
        sigma,
        q,
        perturbation_size=perturbation_theta_size,
        true_beta=true_beta,
    )
    gamma_s_theta = empirical_gradient_s_theta(
        agent_dist,
        theta,
        s,
        sigma,
        perturbation_s_size=perturbation_s,
        perturbation_theta_size=perturbation_theta_size,
    )
    total_derivative = (gamma_loss_s * gamma_s_theta) + gamma_loss_theta
    return total_derivative


def plot_grad_s_theta(agent_dist, sigma, f, savefig=None):
    deriv = []
    deriv_emp = []
    dx = 0.001
    thetas = np.linspace(-np.pi + dx, np.pi - dx, 25)
    for theta in thetas:
        s = f(theta)
        val = expected_gradient_s_theta(agent_dist, theta, s, sigma)
        hat_val = empirical_gradient_s_theta(agent_dist, theta, s, sigma)
        deriv.append(val)
        deriv_emp.append(hat_val)
    plt.plot(
        thetas,
        np.array([derivative(f, theta, dx) for theta in thetas]),
        label="true derivative (numerical approx)",
    )
    plt.plot(thetas, deriv, label="our estimator (expectation version)")
    plt.plot(thetas, deriv_emp, label="our estimator (empirical version)")
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("ds/dtheta")
    plt.title("ds/dtheta vs. theta ")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()


def empirical_gradient_pi_s(agent_dist, theta, s, sigma, r, perturbation_size=0.05):
    """Method that returns the empirical gradient of pi wrt to s incurred given an agent distribution and model and threshold.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of noise distribution (float)
    r -- value that CDF should be evaluated at

    Returns:
    gamma_pi_s -- empirical gradient dL/ds
    """
    beta = convert_to_unit_vector(theta)
    perturbations = (
        2 * bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(agent_dist.n, 1) - 1
    ) * perturbation_size
    scores = []

    bounds = compute_score_bounds(beta)
    interpolators = []

    for agent in agent_dist.agents:
        interpolators.append(agent.br_score_function_s(beta, sigma))

    for i in range(agent_dist.n):
        s_perturbed = np.clip(s + perturbations[i], a_min=bounds[0], a_max=bounds[1])
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](s_perturbed)
        scores.append((br_score - perturbations[i]).item())

    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = scores + noise

    indicators = noisy_scores <= r

    Q = np.matmul(perturbations.T, perturbations)
    gamma_pi_s = np.linalg.solve(Q, np.matmul(perturbations.T, indicators))
    return gamma_pi_s.item()


def expected_gradient_pi_s(agent_dist, theta, s, sigma, r):
    """Method computes partial pi/partial s.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters
    sigma -- standard deviation of noise distribution
    f -- function that maps arctan(beta[1]/beta[0]) -> s_beta (fixed point)

    Returns:
    d_pi_d_s -- expected gradient wrt to s of policy loss 

    """
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)
    br_dist, grad_s_dist = agent_dist.br_gradient_s_distribution(beta, s, sigma)
    z = r - np.array([np.matmul(beta.T, x) for x in br_dist]).reshape(len(br_dist), 1)

    prob = norm.pdf(z, loc=0.0, scale=sigma)
    vec = np.array(
        [-np.matmul(beta.T, grad_s_dist[i]).item() for i in range(len(br_dist))]
    ).reshape(agent_dist.n_types, 1)
    res = prob * vec * agent_dist.prop.reshape(agent_dist.n_types, 1)

    d_pi_d_s = np.sum(res)
    return d_pi_d_s.item()


def plot_grad_pi_s(agent_dist, sigma, f, savefig=None):
    emp_grad_s = []
    grad_s = []
    theta = np.pi / 4
    beta = convert_to_unit_vector(theta)
    s_beta = f(theta)
    bounds = compute_score_bounds(beta)
    rs = np.linspace(bounds[0], bounds[1], 50)
    for r in rs:
        emp_grad = empirical_gradient_pi_s(agent_dist, theta, s_beta, sigma, r)
        grad = expected_gradient_pi_s(agent_dist, theta, s_beta, sigma, r)
        emp_grad_s.append(emp_grad)
        grad_s.append(grad)
    plt.plot(rs, emp_grad_s, label="empirical")
    plt.plot(rs, grad_s, label="expected")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("dpi(r)/ds")
    plt.title("r vs. dpi(r)/ds")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()


def plot_grad_pi_theta(agent_dist, sigma, f, savefig=None):
    emp_grad_theta = []
    grad_theta = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in thetas:
        beta = convert_to_unit_vector(theta)
        s_beta = f(theta)
        bounds = compute_score_bounds(beta)
        emp_grad = empirical_gradient_pi_theta(agent_dist, theta, s_beta, sigma, s_beta)
        grad = expected_gradient_pi_theta(agent_dist, theta, s_beta, sigma, s_beta)
        emp_grad_theta.append(emp_grad)
        grad_theta.append(grad)
    plt.plot(thetas, emp_grad_theta, label="empirical")
    plt.plot(thetas, grad_theta, label="expected")
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("dpi(s_theta)/dtheta")
    plt.title("theta vs. dpi(s_theta)/dtheta")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()


def expected_gradient_s_theta(agent_dist, theta, s, sigma):
    """Method that computes expected gradient ds/dtheta

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise

    Returns:
    ds_dtheta - expected gradient
    """
    beta = convert_to_unit_vector(theta)
    r = s
    density = agent_dist.best_response_pdf(beta, s, sigma, r)
    pi_theta = expected_gradient_pi_theta(agent_dist, theta, s, sigma, r)
    pi_s = expected_gradient_pi_s(agent_dist, theta, s, sigma, r)
    val = -(1 / (pi_s + density)) * pi_theta
    return val


def empirical_gradient_s_theta(
    agent_dist, theta, s, sigma, perturbation_s_size, perturbation_theta_size
):
    """
    Empirical gradient ds/dtheta

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)

    Returns:
    ds_dtheta -- gradient estimate
    """
    beta = convert_to_unit_vector(theta)
    r = s

    hat_pi_theta = empirical_gradient_pi_theta(
        agent_dist, theta, s, sigma, r, perturbation_size=perturbation_theta_size
    )
    hat_pi_s = empirical_gradient_pi_s(
        agent_dist, theta, s, sigma, r, perturbation_size=perturbation_s_size
    )
    hat_density = empirical_density(
        agent_dist,
        theta,
        s,
        sigma,
        r,
        perturbation_s_size=perturbation_s_size,
        perturbation_theta_size=perturbation_theta_size,
    )

    val = -(1 / (hat_pi_s + hat_density)) * hat_pi_theta
    return val


def empirical_density(
    agent_dist, theta, s, sigma, r, perturbation_s_size, perturbation_theta_size
):

    """
    Empirical density p_beta,s(r).

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)


    Returns:
    density_estimate -- density estimate p_beta,s(r)
    """
    perturbations_theta = (
        2 * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d - 1)) - 1
    ) * perturbation_theta_size
    perturbations_s = (
        2 * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d - 1)) - 1
    ) * perturbation_s_size

    scores = []

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)

    interpolators_theta = []
    for agent in agent_dist.agents:
        f, _ = agent.br_score_function_beta(s, sigma)
        interpolators_theta.append(f)

    theta = np.arctan2(beta[1], beta[0])
    for i in range(int(agent_dist.n / 2)):
        theta_perturbed = theta + perturbations_theta[i]
        if theta_perturbed < -np.pi:
            theta_perturbed += 2 * np.pi
        if theta_perturbed > np.pi:
            theta_perturbed -= 2 * np.pi
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators_theta[agent_type](theta_perturbed)
        scores.append(br_score.item())

    interpolators_s = []
    for agent in agent_dist.agents:
        interpolators_s.append(agent.br_score_function_s(beta, sigma))

    for i in range(int(agent_dist.n / 2), agent_dist.n):
        s_perturbed = np.clip(s + perturbations_s[i], a_min=bounds[0], a_max=bounds[1])
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators_s[agent_type](s_perturbed)
        scores.append((br_score - perturbations_s[i]).item())

    scores = np.array(scores)
    noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n)
    noisy_scores = scores + noise
    kde = gaussian_kde(noisy_scores)
    return kde(r).item()


def expected_gradient_pi_theta(agent_dist, theta, s, sigma, r):
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)

    br_dist, grad_theta_dist = agent_dist.br_gradient_theta_distribution(
        theta, s, sigma
    )

    z = r - np.array([np.matmul(beta.T, x) for x in br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0.0, scale=sigma)
    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    first_term = np.array(
        [
            np.matmul(grad_theta_dist[i].T, beta).item()
            for i in range(len(grad_theta_dist))
        ]
    ).reshape(agent_dist.n_types, 1)
    second_term = np.array(
        [
            np.matmul(br_dist[i].T, dbeta_dtheta).item()
            for i in range(len(grad_theta_dist))
        ]
    ).reshape(agent_dist.n_types, 1)
    total = first_term + second_term
    res = -prob * total * agent_dist.prop.reshape(len(agent_dist.prop), 1)
    grad_pi_theta = np.sum(res)
    return grad_pi_theta.item()


def empirical_gradient_pi_theta(agent_dist, theta, s, sigma, r, perturbation_size=0.1):
    """Method that returns the empirical gradient of pi wrt to beta incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    beta -- model parameters (N, 1) array
    s -- threshold (float)
    sigma -- standard deviation of the noise (float)

    Returns:
    gamma_pi_beta - gradient of pi at beta
    """
    perturbations = (
        2
        * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d - 1)).reshape(
            agent_dist.n, agent_dist.d - 1, 1
        )
        - 1
    ) * perturbation_size
    scores = []

    beta = convert_to_unit_vector(theta)
    bounds = compute_score_bounds(beta)

    interpolators = []
    for agent in agent_dist.agents:
        f, _ = agent.br_score_function_beta(s, sigma)
        interpolators.append(f)

    for i in range(agent_dist.n):
        theta_perturbed = theta + perturbations[i]
        if theta_perturbed < -np.pi:
            theta_perturbed += 2 * np.pi
        if theta_perturbed > np.pi:
            theta_perturbed -= 2 * np.pi
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](theta_perturbed)
        scores.append(br_score.item())

    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = scores + noise
    indicators = noisy_scores <= r

    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d - 1)
    Q = np.matmul(perturbations.T, perturbations)
    gamma_pi_theta = np.linalg.solve(Q, np.matmul(perturbations.T, indicators))
    return gamma_pi_theta.item()


def expected_gradient_loss_theta(agent_dist, theta, s, sigma, true_beta=None):
    """Method computes partial L(theta)/partial theta.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters
    s -- threshold
    sigma -- standard deviation of noise distribution
    true_beta -- optional ideal model


    Returns:
    d_l_d_beta -- expected gradient wrt to beta of policy loss at beta

    """
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    beta = convert_to_unit_vector(theta)
    if true_beta is None:
        true_beta = np.zeros((agent_dist.d, 1))
        true_beta[0] = 1.0
    bounds = compute_score_bounds(beta)

    true_scores = np.array(
        [np.matmul(true_beta.T, agent.eta).item() for agent in agent_dist.agents]
    ).reshape(agent_dist.n_types, 1)

    br_dist, grad_theta_dist = agent_dist.br_gradient_theta_distribution(
        theta, s, sigma
    )
    z = s - np.array([np.matmul(beta.T, x) for x in br_dist]).reshape(len(br_dist), 1)
    prob = norm.pdf(z, loc=0.0, scale=sigma)

    dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
    first_term = np.array(
        [
            np.matmul(grad_theta_dist[i].T, beta).item()
            for i in range(len(grad_theta_dist))
        ]
    ).reshape(agent_dist.n_types, 1)
    second_term = np.array(
        [
            np.matmul(br_dist[i].T, dbeta_dtheta).item()
            for i in range(len(grad_theta_dist))
        ]
    ).reshape(agent_dist.n_types, 1)
    total = first_term + second_term

    res = -prob * total * true_scores * agent_dist.prop.reshape(agent_dist.n_types, 1)
    dl_dtheta = np.sum(res).item()

    return dl_dtheta


def empirical_gradient_loss_theta(
    agent_dist, theta, s, sigma, q, true_beta=None, perturbation_size=0.1
):
    """Method that returns the empirical gradient of loss wrt to theta incurred given an agent distribution and model and threshold.
    Assumes that there is an model true_beta when applied to the agents' hidden eta features
    optimally selects the top agents.

    Keyword args:
    agent_dist -- AgentDistribution
    theta -- model parameters (D-1, 1) array
    s -- threshold (float)
    q -- quantile
    sigma -- standard deviation of the noise (float)
    true_beta -- (D, 1) array

    Returns:
    loss -- empirical policy loss
    """
    beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
    if true_beta is None:
        true_beta = np.zeros(beta.shape)
        true_beta[0] = 1.0

    perturbations = (
        2
        * bernoulli.rvs(p=0.5, size=agent_dist.n * (agent_dist.d - 1)).reshape(
            agent_dist.n, agent_dist.d - 1, 1
        )
        - 1
    ) * perturbation_size
    scores = []

    bounds = compute_score_bounds(beta)

    interpolators = []
    for agent in agent_dist.agents:
        f, _ = agent.br_score_function_beta(s, sigma)
        interpolators.append(f)

    for i in range(agent_dist.n):
        theta_perturbed = theta + perturbations[i]
        if theta_perturbed < -np.pi:
            theta_perturbed += 2 * np.pi
        if theta_perturbed > np.pi:
            theta_perturbed -= 2 * np.pi
        agent_type = agent_dist.n_agent_types[i]
        br_score = interpolators[agent_type](theta_perturbed)
        scores.append(br_score.item())

    scores = np.array(scores).reshape(agent_dist.n, 1)
    noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(agent_dist.n, 1)
    noisy_scores = scores + noise

    # Compute loss
    treatments = noisy_scores >= np.quantile(noisy_scores, q)
    loss_vector = treatments * np.array(
        [-np.matmul(true_beta.T, eta).item() for eta in agent_dist.get_etas()]
    ).reshape(agent_dist.n, 1)

    perturbations = perturbations.reshape(agent_dist.n, agent_dist.d - 1)
    Q = np.matmul(perturbations.T, perturbations)
    gamma_loss_theta = np.linalg.solve(Q, np.matmul(perturbations.T, loss_vector))
    return gamma_loss_theta.item()


def plot_grad_loss_theta(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_theta = []
    emp_grad_theta = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in tqdm.tqdm(thetas):
        s_beta = f(theta)
        grad = expected_gradient_loss_theta(agent_dist, theta, s_beta, sigma, true_beta)
        grad_theta.append(grad)
        emp_grad = empirical_gradient_loss_theta(
            agent_dist, theta, s_beta, sigma, q, true_beta
        )
        emp_grad_theta.append(emp_grad)

    plt.plot(thetas, emp_grad_theta, label="empirical")
    plt.plot(thetas, grad_theta, label="expected")
    plt.xlabel("Theta (Corresponds to Beta)")
    plt.ylabel("dL/dtheta")
    plt.title("Theta vs. dL/dtheta")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()


def plot_grad_loss_s(agent_dist, sigma, q, f, true_beta=None, savefig=None):
    grad_s = []
    emp_grad_s = []
    thetas = np.linspace(-np.pi, np.pi, 50)
    for theta in thetas:
        s_beta = f(theta)
        d_l_d_s = expected_gradient_loss_s(agent_dist, theta, s_beta, sigma, true_beta)
        emp_grad = empirical_gradient_loss_s(
            agent_dist, theta, s_beta, sigma, q, true_beta
        )
        emp_grad_s.append(emp_grad)
        grad_s.append(d_l_d_s)
    plt.plot(thetas, emp_grad_s, label="empirical")
    plt.plot(thetas, grad_s, label="expected")
    plt.legend()
    plt.xlabel("Theta (Corresponds to Beta)")
    plt.ylabel("dL/ds")
    plt.title("Beta vs. dL/ds")
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    plt.close()
