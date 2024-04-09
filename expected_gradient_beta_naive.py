from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class ExpectedGradientBetaNaive:
    def __init__(
        self,
        agent_dist,
        beta,
        s,
        sigma,
        q,
        true_scores,
    ):
        self.agent_dist = agent_dist
        self.sigma = sigma
        self.s = s
        self.beta = beta
        self.q = q
        self.true_scores = true_scores
        self.bounds = compute_score_bounds(self.beta, self.sigma)
        self.br_dist = self.agent_dist.best_response_distribution(
            self.beta, self.s, self.sigma
        )
        self.noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )

    def expected_gradient_loss_beta(self):
        dim = self.agent_dist.d

        z = self.s - np.array(
            [np.matmul(self.beta.T, x) for x in self.br_dist]
        ).reshape(len(self.br_dist), 1, 1)

        prob = norm.pdf(z, loc=0.0, scale=self.sigma)
        second_term = np.array(self.br_dist).reshape(self.agent_dist.n_types, dim, 1)

        res = (
            prob
            * second_term
            * self.true_scores.reshape(self.agent_dist.n_types, 1, 1)
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1, 1)
        )
        dl_dbeta = np.sum(res, axis=0)

        assert dl_dbeta.shape[0] == dim
        assert dl_dbeta.shape[1] == 1

        dV_dr = np.sum(
            -(
                prob
                * self.true_scores.reshape(self.agent_dist.n_types, 1, 1)
                * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1, 1)
            ),
            axis=0,
        )
        density = np.sum(
            prob * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1, 1), axis=0
        )
        dr_dbeta = (
            np.sum(
                prob
                * second_term
                * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1, 1),
                axis=0,
            )
            / density
        )

        capacity_deriv_beta = dV_dr * dr_dbeta

        return dl_dbeta + capacity_deriv_beta

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
