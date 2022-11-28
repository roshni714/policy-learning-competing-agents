from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds
from sklearn.linear_model import LinearRegression


class EmpiricalWelfareMaximization:
    def __init__(
        self, agent_dist, sigma, q, true_scores,
    ):
        self.q = q
        self.agent_dist = agent_dist
        self.sigma = sigma

        self.true_scores = true_scores
        self.eta_dist = agent_dist.get_etas()

        self.noise_d_dim = norm.rvs(
            loc=0.0, scale=sigma, size=agent_dist.n * agent_dist.d
        ).reshape(agent_dist.n, agent_dist.d, 1)

        self.noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )

        self.treatment_assignment = bernoulli.rvs(p=0.5, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )

    def estimate_beta(self):
        coefs = []
        intercepts = []
        for treatment in [0.0, 1.0]:
            idx = self.treatment_assignment == treatment
            outcomes = -self.true_scores[idx]
            X = self.eta_dist[idx.flatten(), :, :]
            noise = self.noise_d_dim[idx.flatten(), :, :]
            noisy_X = X + noise
            reg = LinearRegression().fit(
                noisy_X.reshape(len(noisy_X), self.agent_dist.d), outcomes
            )
            coefs.append(reg.coef_)
            intercepts.append(reg.intercept_)

        cate = coefs[1] - coefs[0]
        cate_norm = np.sqrt(np.sum(cate ** 2))
        beta_naive = cate / cate_norm

        return beta_naive

    def empirical_loss(self, beta, s):
        br_dist = self.agent_dist.best_response_distribution(beta, s, self.sigma)
        br_star_scores = np.array([np.matmul(beta.T, br).item() for br in br_dist])
        types = self.agent_dist.n_agent_types.astype(int)
        n_br = np.array(br_star_scores[self.agent_dist.n_agent_types]).reshape(
            self.agent_dist.n, 1
        )
        scores = n_br + self.noise
        cutoff = np.quantile(scores, self.q)
        treatments = scores > cutoff
        losses = self.true_scores * treatments

        return np.mean(losses).item()
