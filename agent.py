from scipy.stats import norm
from scipy.optimize import newton
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

from utils import compute_score_bounds


class Agent:
    """This is a class for representing a strategic agent.

    Keyword arguments:
    eta -- natural ability of agent (D, 1) array
    gamma -- gaming ability of agent (D, 1) array
    """

    def __init__(self, eta, gamma):
        self.eta = eta
        self.gamma = gamma

    def best_response(self, beta, s, sigma, x0=None):
        """Method for computing an agent's best response given a particular model and threshold under a noise assumption.

        Keyword arguments:
        beta -- model parameters (D, 1) array
        s -- threshold (float)
        sigma -- standard deviation of the noise distribution (float)
        x0 -- initial guess of best response (D, 1) array

        Returns:
        br -- agent best response (D,1) array
        """
        bounds = compute_score_bounds(beta, sigma)
        if s < bounds[0] and s > bounds[1]:
            print("CAUTION: s out of bounds")
        #        assert (
        #            s >= bounds[0] and s <= bounds[1]
        #        ), "cannot compute best response for s out of score bounds"
        try:
            if x0 is None:
                x0 = self.eta.flatten()
            else:
                x0 = x0.flatten()
            res = newton(
                Agent._func_derivative_utility(beta, s, self.eta, self.gamma, sigma),
                x0=x0,
                maxiter=10000,
                full_output=True,
            )
            if all(res.converged):
                val = res.root
            else:
                val = np.array([0.0, 0.0]).reshape(2, 1)
        except:
            val = np.array([0.0, 0.0]).reshape(2, 1)
            print(
                "Failed to compute best response for agent with eta={}, gamma={} under beta={}, s={}, sigma={}.".format(
                    self.eta, self.gamma, beta, s, sigma
                )
            )
        # val = np.clip(val, a_min=0.0, a_max=1.0)
        br = val.reshape(beta.shape)
        return br

    def plot_best_response_score(self, beta, sigma):
        """Method for plotting agent's best response score (without noise) vs. perceived threshold s.

        Keyword arguments:
        beta -- model parameters (D, 1) array
        sigma -- standard deviation of noise distribution (float)
        """
        bounds = compute_score_bounds(beta, sigma)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        br = [
            np.matmul(np.transpose(beta), self.best_response(beta, s, sigma)).item()
            for s in thresholds
        ]
        plt.xlabel("Threshold")
        plt.ylabel("Score of Best Response")
        plt.title("Score of Agent's Best Response vs. Threshold")
        plt.plot(thresholds, br)
        plt.show()
        plt.close()

    @staticmethod
    def _func_derivative_utility(beta, s, eta, gamma, sigma):
        """Method that returns a function that compute the derivative of an agent's utility function.
        Note that scipy's implementation of Newton's Method expects arrays of shape (D,) so the function below
        takes input of shape (D,). The input is reshaped to a (D,1) vector in the method and the output is again
        reshaped to be of shape (D,)

        Keyword arguments:
        beta -- model parameters (D, 1) array
        s -- perceived threshold (float)
        eta -- agent's raw features (D, 1) array
        gamma -- agent's gaming ability (D, 1) array
        sigma -- standard deviation of noise distribution (float)

        Returns:
        f -- function that computes the derivative of agent's utility (Python function that takes (D,) array as input and outputs (1,) array)
        """

        def f(x):
            d = x.reshape(eta.shape) - eta
            cost_of_gaming = -2 * np.matmul(np.transpose(d), np.diag(gamma.flatten()))
            score = np.matmul(np.transpose(beta), x).item()
            allocation = norm.pdf(s - score, loc=0.0, scale=sigma) * np.transpose(beta)
            val = cost_of_gaming + allocation
            return (cost_of_gaming + allocation).flatten()

        return f

    def br_score_function_s(self, beta, sigma):
        """Method that returns an interpolator of the agent's best response score (without noise) as a function of perceived threshold s.

        Keyword arguments:
        beta -- model parameters (D, 1) array
        sigma -- standard deviation of noise distribution (float)

        Returns:
        f -- interpolator function (Python function that takes (1,) array as input and outputs (1,) array)
        """

        bounds = compute_score_bounds(beta, sigma)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        br = [
            np.matmul(np.transpose(beta), self.best_response(beta, s, sigma)).item()
            for s in thresholds
        ]

        f = interp1d(thresholds, br, kind="linear")
        return f

    def br_score_function_beta(self, s, sigma):
        """Method that returns an interpolator of the agent's best response score (without noise) as a function of the model parameters
        in polar coordinates. CAUTION: Assumes that the model is given by 1-dimensional polar coordinate theta.

        Keyword arguments:
        s -- perceived threshold (float)
        sigma -- standard deviation of the noise distribution

        Returns:
        f -- interpolator function (Python function that takes (1,) array as input and outputs (1,) array)
        """
        thetas = np.linspace(-np.pi, np.pi, 100)
        br = []
        valid_theta = []
        for theta in thetas:
            beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
            bounds = compute_score_bounds(beta, sigma)
            if s >= bounds[0] and s <= bounds[1]:
                br.append(np.matmul(beta.T, self.best_response(beta, s, sigma)).item())
                valid_theta.append(theta)

        f = interp1d(valid_theta, br, kind="linear")
        return f, valid_theta

    def br_gradient_s(self, beta, s, sigma):
        """
        Computes of the gradient of the best response wrt to s.
        Keyword arguments:
        beta -- model parameters (D, 1)
        s -- threshold (float)
        sigma -- standard deviation of the noise distribution (float)
        Returns:
        best_response -- (D, 1) array
        deriv_s -- (D, 1) array
        """
        bounds = compute_score_bounds(beta, sigma)
        assert s >= bounds[0] and s <= bounds[1]
        G = np.diag(self.gamma.flatten())
        best_response = self.best_response(beta, s, sigma)
        arg = s - np.matmul(beta.T, best_response)
        prob_prime = -(arg / (sigma ** 2)) * norm.pdf(arg, loc=0, scale=sigma)
        rank_one_mat = np.matmul(beta, beta.T)
        deriv_s = np.matmul(
            np.linalg.inv(2 * G + prob_prime * rank_one_mat) * prob_prime, beta
        )
        return best_response, deriv_s

    def br_gradient_theta(self, theta, s, sigma):
        """
        Computes of the gradient of the best response wrt to theta.
        Keyword arguments:
        beta -- model parameters (D, 1)
        s -- threshold (float)
        sigma -- standard deviation of the noise distribution (float)
        Returns:
        best_response -- (D, 1) array
        deriv_s -- (D, 1) array
        """

        beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
        bounds = compute_score_bounds(beta, sigma)
        assert s >= bounds[0] and s <= bounds[1]
        best_response = self.best_response(beta, s, sigma)
        arg = s - np.matmul(beta.T, best_response)
        prob = norm.pdf(arg, loc=0, scale=sigma)
        prob_prime = -(arg / (sigma ** 2)) * norm.pdf(arg, loc=0, scale=sigma)
        rank_one_mat = np.matmul(beta, beta.T)
        dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
        G = np.diag(self.gamma.flatten())
        first = 2 * G + prob_prime * rank_one_mat
        inv_mat = np.linalg.inv(first)
        second = (
            -(prob_prime * np.matmul(best_response.T, dbeta_dtheta)).item() * beta
            + prob_prime * dbeta_dtheta
        )
        grad_theta = np.matmul(inv_mat, second)
        return best_response, grad_theta


if __name__ == "__main__":
    eta = np.array([0.5, 0.5]).reshape(2, 1)
    gamma = np.array([1.0, 2.0]).reshape(2, 1)
    agent = Agent(eta, gamma)
    beta = np.array([np.sin(np.pi / 4), np.cos(np.pi / 4)]).reshape(2, 1)
    sigma = 0.35
    s = 0.5
    print(agent.best_response(beta, s, sigma))
