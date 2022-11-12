from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class ExpectedGradientBetaNaive:
    def __init__(
        self, agent_dist, theta, s, sigma, true_scores,
    ):
        self.agent_dist = agent_dist
        self.sigma = sigma
        self.theta = theta
        self.s = s
        self.beta = convert_to_unit_vector(self.theta)
        if true_beta is None:
            true_beta = np.zeros((self.agent_dist.d, 1))
            true_beta[0] = 1.0
        self.true_scores = true_scores
        self.bounds = compute_score_bounds(self.beta, self.sigma)

        self.br_dist = self.agent_dist.best_response_distribution(
            self.beta, self.s, self.sigma
        )

    def expected_gradient_loss_beta(self):
        dim = self.agent_dist.d

        z = self.s - np.array(
            [np.matmul(self.beta.T, x) for x in self.br_dist]
        ).reshape(len(self.br_dist), 1)
        prob = norm.pdf(z, loc=0.0, scale=self.sigma)
        second_term = np.array(self.br_dist).reshape(self.agent_dist.n_types, dim, 1)

        res = (
            prob
            * second_term
            * self.true_scores
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        dl_dbeta = np.sum(res, axis=0).item()

        assert dl_dbeta.shape[0] == dim
        assert dl_dbeta.shape[1] == 1

        return dl_dtheta

    def empirical_loss(self):
        """TODO"""

    def expected_loss(self):
        z = self.s - np.array(
            [np.matmul(self.beta.T, x) for x in self.br_dist]
        ).reshape(len(self.br_dist), 1)
        prob = 1 - norm.cdf(x=z, loc=0.0, scale=self.sigma)
        product = (
            self.true_scores
            * prob
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )

        return np.sum(product).item()
