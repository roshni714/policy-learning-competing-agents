from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class ExpectedGradientNaive:
    def __init__(
        self, agent_dist, theta, s, sigma, true_beta,
    ):
        self.agent_dist = agent_dist
        self.sigma = sigma
        self.theta = theta
        self.s = s
        self.beta = convert_to_unit_vector(self.theta)
        if true_beta is None:
            true_beta = np.zeros((self.agent_dist.d, 1))
            true_beta[0] = 1.0
        self.true_beta = true_beta
        self.bounds = compute_score_bounds(self.beta)

        self.true_scores = np.array(
            [
                -np.matmul(true_beta.T, agent.eta).item()
                for agent in self.agent_dist.agents
            ]
        ).reshape(self.agent_dist.n_types, 1)
        (
            self.br_dist,
            self.grad_theta_dist,
        ) = self.agent_dist.br_gradient_theta_distribution(
            self.theta, self.s, self.sigma
        )
        _, self.grad_s_dist = self.agent_dist.br_gradient_s_distribution(
            self.beta, self.s, self.sigma
        )

    def expected_gradient_loss_theta(self):
        dim = self.agent_dist.d
        assert dim == 2, "Method does not work for dimension {}".format(dim)

        z = self.s - np.array(
            [np.matmul(self.beta.T, x) for x in self.br_dist]
        ).reshape(len(self.br_dist), 1)
        prob = norm.pdf(z, loc=0.0, scale=self.sigma)

        first_term = np.array(
            [
                np.matmul(self.grad_theta_dist[i].T, self.beta).item()
                for i in range(len(self.grad_theta_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)
        
        res = (
            prob
            * total
            * first_term
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        dl_dtheta = np.sum(res).item()
        return dl_dtheta

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


