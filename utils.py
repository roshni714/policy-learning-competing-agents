import numpy as np
import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def keep_theta_in_bounds(theta):

    if theta[-1] > 2 * np.pi:
        theta[-1] -= 2 * np.pi
    elif theta[-1] < 0:
        theta[-1] += 2 * np.pi

    for i in range(theta.shape[0] - 1):
        theta[i] = np.clip(theta[i], 0.0, np.pi)

    return theta


def convert_to_polar_coordinates(beta):
    """Method that converts a D-dimensional unit vector to polar coordinates (D-1 - dimensional.)

    Keyword args:
    beta -- (D, 1) dimensional unit vector

    Returns:
    theta -- (D-1, 1) vector of angles
    """
    #    np.testing.assert_almost_equal(np.sqrt(np.sum(beta ** 2)).item(), 1.)
    d = beta.shape[0]
    theta = np.zeros(shape=(d - 1, 1))
    for i in range(d - 1):

        if beta[i] != 0.0 and np.all(beta[i + 1 :] == 0):
            if beta[i] > 0:
                theta[i] = 0
            else:
                theta[i] = np.pi
        elif np.all(beta[i:] == 0):
            theta[i] = 0
        else:
            if i < d - 2:
                rem_beta = beta[i:]
                norm = np.sqrt(np.sum(rem_beta**2))
                psi = np.arccos(beta[i] / norm)
                theta[i] = psi

            if i == d - 2:
                rem_beta = beta[i:]
                norm = np.sqrt(np.sum(rem_beta**2))
                psi = np.arccos(beta[i] / norm)
                if beta[-1] >= 0:
                    angle = psi
                else:
                    angle = 2 * np.pi - psi
                theta[i] = angle

    return theta


def convert_to_unit_vector(theta):
    """Method that converts a polar coordinates to Euclidean unit vector.

    Keyword args:
    theta -- (D-1, 1) vector of angles

    Returns:
    beta -- (D, 1)  unit vector
    """
    assert (
        theta[-1] >= 0 and theta[-1] <= 2 * np.pi
    ), "last theta coordinate must be between 0 and 2*pi"
    assert np.all(theta[:-1] >= 0), "first through n-1 theta coordinates must be > 0"
    assert np.all(
        theta[:-1] < np.pi
    ), "first through n-1 theta coordinates must be < pi"

    d_minus_one = theta.shape[0]

    beta = []
    d = d_minus_one + 1
    beta = np.zeros(shape=(d, 1))
    for i in range(d):
        if i == 0:
            beta[i] = np.cos(theta[i])
        if i >= 1 and i <= d - 3 and 1 <= d - 3:
            beta[i] = np.cos(theta[i]) * np.prod(np.sin(theta[: i - 1]))
        if i == d - 2:
            beta[i] = np.cos(theta[-1]) * np.prod(np.sin(theta[:-1]))
        if i == d - 1:
            beta[i] = np.sin(theta[-1]) * np.prod(np.sin(theta[:-1]))

    beta = np.array(beta).reshape(d, 1)
    return beta


def compute_continuity_noise(agent_dist):
    """Method that returns the standard deviation of the noise distribution for ensuring continuity.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    gammas = [agent.gamma for agent in agent_dist.agents]

    min_eigenvalue = np.min(gammas)
    return np.sqrt(1 / (2 * min_eigenvalue * (np.sqrt(2 * np.pi * np.e)))) + 0.001


def compute_continuity_noise_gammas(gammas):
    """Method that returns the standard deviation of the noise distribution for ensuring continuity.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    min_eigenvalue = np.min(gammas)
    return np.sqrt(1 / (2 * min_eigenvalue * (np.sqrt(2 * np.pi * np.e)))) + 0.001


def compute_contraction_noise(agent_dist):
    """Method that returns the standard deviation of the noise distribution for ensuring contraction.

    Keyword args:
    agent_dist -- AgentDistribution
    """
    gammas = [agent.gamma for agent in agent_dist.agents]

    min_eigenvalue = np.min(gammas)
    return np.sqrt(1 / (min_eigenvalue * (np.sqrt(2 * np.pi * np.e)))) + 0.001


def compute_score_bounds(beta, sigma):
    """Method that returns bounds on the highest and lowest possible scores that an agent can achieve.
    Assumes that agents take actions in [0, 1]^2

    Keyword arguments:
    beta -- model parameters (D,1) array
    sigma -- standard deviation of noise distribution (float)

    Returns:
    min_score -- lower score bound
    max_score -- uper score bound
    """
    #    assert beta.shape[0] == 2, "Method does not work for beta with dim {}".format(
    #        beta.shape[0]
    #    )
    max_score = 0.0
    min_score = 0.0
    for i in range(len(beta)):
        if beta[i] > 0:
            max_score += beta[i] * 10
        if beta[i] < 0:
            min_score += beta[i] * 10

    min_score -= 4 * sigma
    max_score += 4 * sigma
    return min_score, max_score


def spherical_coordinates(beta):
    assert beta.shape[0] == 2, "Method does not work for beta with dim {}".format(
        beta.shape[0]
    )

    return np.arctan2(beta[1], beta[0])


def fixed_point_interpolation_true_distribution(
    agent_dist, sigma, q, plot=False, savefig=None
):
    """Method that returns a function that maps model parameters to the fixed point it induces.

    The function is estimated by doing a linear interpolation of the fixed points from theta
    (a 1-dimensional parametrization of beta). theta -> beta = [cos (theta),  sin(theta)]
    The function maps theta -> s_beta.

    Keyword args:
    agent_dist -- AgentDistribution
    sigma -- standard deviation of the noise distribution (float)
    q -- quantile (float)
    plot -- optional plotting argument
    savefig -- path to save figure

    Returns:
    f -- interp1d object that maps theta to s_beta
    """
    dim = agent_dist.d
    assert dim == 2, "Method does not work for dimension {}".format(dim)

    thetas = np.linspace(0.0, 2 * np.pi, 100)
    fixed_points = []

    # compute beta and fixed point for each theta
    for theta in tqdm.tqdm(thetas):
        beta = convert_to_unit_vector(np.array([theta]).reshape(1, 1))
        fp = agent_dist.quantile_fixed_point_true_distribution(beta, sigma, q)
        fixed_points.append(fp)

    f = interp1d(thetas, fixed_points, kind="linear")

    if plot:
        plt.plot(thetas, fixed_points, label="actual")
        plt.plot(thetas, f(thetas), label="interpolation")
        plt.xlabel("Thetas (corresponds to different Beta)")
        plt.ylabel("s_beta")
        plt.title("Location of Fixed Points: s_beta vs. beta")
        plt.legend()
        if savefig is not None:
            plt.savefig(savefig)
        plt.show()
        plt.close()

    return f
