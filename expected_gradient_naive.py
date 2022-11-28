from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class ExpectedGradientNaive:
    def __init__(
        self, agent_dist, theta, s, sigma, q, true_beta,
    ):
        self.q = q
        self.agent_dist = agent_dist
        self.sigma = sigma
        self.theta = theta
        self.s = s
        self.beta = convert_to_unit_vector(self.theta)
        if true_beta is None:
            true_beta = np.zeros((self.agent_dist.d, 1))
            true_beta[0] = 1.0
        self.true_beta = true_beta
        self.bounds = compute_score_bounds(self.beta, self.sigma)

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
        self.dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
        self.noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )

    def expected_gradient_loss_theta(self):
        dim = self.agent_dist.d
        assert dim == 2, "Method does not work for dimension {}".format(dim)

        z = self.s - np.array(
            [np.matmul(self.beta.T, x) for x in self.br_dist]
        ).reshape(len(self.br_dist), 1)
        prob = norm.pdf(z, loc=0.0, scale=self.sigma)
        second_term = np.array(
            [
                np.matmul(self.br_dist[i].T, self.dbeta_dtheta).item()
                for i in range(len(self.grad_theta_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)

        res = (
            prob
            * second_term
            * self.true_scores
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        dV_dr = -np.sum(
            prob
            * self.true_scores
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        density = np.sum(
            prob * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        dr_dtheta = (
            np.sum(
                prob
                * second_term
                * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
            )
            / density
        )
        capacity_deriv_theta = (dV_dr * dr_dtheta).item()

        dl_dtheta = np.sum(res).item()

        return dl_dtheta + capacity_deriv_theta

    def empirical_loss(self):
        br_star_scores = np.array(
            [np.matmul(self.beta.T, br).item() for br in self.br_dist]
        )
        types = self.agent_dist.n_agent_types.astype(int)
        n_br = np.array(br_star_scores[self.agent_dist.n_agent_types]).reshape(
            self.agent_dist.n, 1
        )
        scores = n_br + self.noise
        cutoff = np.quantile(scores, self.q)
        treatments = scores > cutoff
        n_true_scores = self.true_scores[self.agent_dist.n_agent_types]
        losses = n_true_scores * treatments

        return np.mean(losses).item()

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
