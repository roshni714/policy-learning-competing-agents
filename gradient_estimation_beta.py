from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds, keep_theta_in_bounds
import itertools


class GradientEstimator:
    def __init__(
        self,
        agent_dist,
        beta,
        s,
        sigma,
        q,
        true_scores,
        perturbation_s_size,
        perturbation_beta_size,
    ):
        self.agent_dist = agent_dist
        self.sigma = sigma

        self.perturbations_s_idx = np.random.choice(
            list(range(0, 2**1)), size=agent_dist.n
        )
        self.perturbations_beta_idx = np.random.choice(
            list(range(0, 2**agent_dist.d)), size=agent_dist.n
        )

        self.noise = norm.rvs(loc=0.0, scale=sigma, size=agent_dist.n).reshape(
            agent_dist.n, 1
        )
        self.beta = beta
        self.s = s
        self.q = q
        self.true_scores = true_scores
        self.perturbation_s_size = perturbation_s_size
        self.perturbation_beta_size = perturbation_beta_size

        self.p_betas = np.array(
            list(itertools.product([-1.0, 1.0], repeat=self.beta.shape[0]))
        )
        self.p_ss = np.array(list(itertools.product([-1.0, 1.0], repeat=1)))

        self.perturbations_s = self.p_ss[self.perturbations_s_idx].reshape(
            agent_dist.n, 1
        )
        self.perturbations_beta = self.p_betas[self.perturbations_beta_idx].reshape(
            agent_dist.n, agent_dist.d
        )

    def get_best_responses(self):
        best_responses = {i: {} for i in range(self.agent_dist.n_types)}
        for agent_type in range(self.agent_dist.n_types):
            for i in range(len(self.p_betas)):
                for j in range(len(self.p_ss)):
                    beta_perturbed = self.beta + (
                        self.p_betas[i].reshape(self.beta.shape)
                        * self.perturbation_beta_size
                    )
                    s_perturbed = (
                        self.s + (self.p_ss[j] * self.perturbation_s_size).item()
                    )
                    if len(best_responses[agent_type]) == 0:
                        br = self.agent_dist.agents[agent_type].best_response(
                            beta_perturbed, s_perturbed, self.sigma
                        )
                    else:
                        br_prev = best_responses[agent_type][(0, 0)]
                        br = self.agent_dist.agents[agent_type].best_response(
                            beta_perturbed, s_perturbed, self.sigma, x0=br_prev
                        )
                    best_responses[agent_type][(i, j)] = br

        return best_responses

    def get_best_responses_partial(self):
        best_responses = {i: {} for i in range(self.agent_dist.n_types)}
        for agent_type in range(self.agent_dist.n_types):
            for i in range(len(self.p_betas)):
                beta_perturbed = self.beta + (
                    self.p_betas[i].reshape(self.beta.shape)
                    * self.perturbation_beta_size
                )
                if len(best_responses[agent_type]) == 0:
                    br = self.agent_dist.agents[agent_type].best_response(
                        beta_perturbed, self.s, self.sigma
                    )
                else:
                    br_prev = best_responses[agent_type][0]
                    br = self.agent_dist.agents[agent_type].best_response(
                        beta_perturbed, self.s, self.sigma, x0=br_prev
                    )
                best_responses[agent_type][i] = br

        return best_responses

    def get_scores(self, best_responses):
        unperturbed_scores = []
        beta_perturbed_scores = []
        scores = []
        for k in range(self.agent_dist.n):
            agent_type = self.agent_dist.n_agent_types[k]
            i = self.perturbations_beta_idx[k]
            j = self.perturbations_s_idx[k]
            p_beta = self.p_betas[i]
            p_s = self.p_ss[j]
            br = best_responses[agent_type][(i, j)]
            beta_perturbed = self.beta + (
                np.array(p_beta).reshape(self.beta.shape) * self.perturbation_beta_size
            )
            scores.append(
                np.matmul(beta_perturbed.T, br).item()
                - (p_s.item() * self.perturbation_s_size)
            )
            beta_perturbed_scores.append(np.matmul(beta_perturbed.T, br).item())
            unperturbed_scores.append(np.matmul(self.beta.T, br).item())
        scores = np.array(scores).reshape(self.agent_dist.n, 1)
        #        unperturbed_scores = np.array(unperturbed_scores).reshape(self.agent_dist.n, 1)
        scores = self.noise + scores
        unperturbed_scores = self.noise + np.array(unperturbed_scores).reshape(
            self.agent_dist.n, 1
        )
        beta_perturbed_scores = self.noise + np.array(beta_perturbed_scores).reshape(
            self.agent_dist.n, 1
        )
        return scores, unperturbed_scores, beta_perturbed_scores

    def get_scores_partial(self, best_responses):
        beta_perturbed_scores = []
        for k in range(self.agent_dist.n):
            agent_type = self.agent_dist.n_agent_types[k]
            i = self.perturbations_beta_idx[k]
            p_beta = self.p_betas[i]
            br = best_responses[agent_type][i]
            beta_perturbed = self.beta + (
                np.array(p_beta).reshape(self.beta.shape) * self.perturbation_beta_size
            )
            beta_perturbed_scores.append(np.matmul(beta_perturbed.T, br).item())
        #        unperturbed_scores = np.array(unperturbed_scores).reshape(self.agent_dist.n, 1)
        beta_perturbed_scores = self.noise + np.array(beta_perturbed_scores).reshape(
            self.agent_dist.n, 1
        )
        return beta_perturbed_scores

    def compute_gradients_partial(self, beta_perturbed_scores, cutoff):
        p_beta = self.perturbations_beta * self.perturbation_beta_size
        Q_beta = np.matmul(p_beta.T, p_beta)
        treatments = beta_perturbed_scores > cutoff
        loss_vector = treatments * self.true_scores

        gamma_loss_beta = np.linalg.solve(
            Q_beta, np.matmul(p_beta.T, loss_vector)
        ).reshape(self.agent_dist.d, 1)

        return gamma_loss_beta, loss_vector

    def compute_gradients(
        self, scores, unperturbed_scores, beta_perturbed_scores, cutoff
    ):
        p_s = self.perturbations_s * self.perturbation_s_size
        p_beta = self.perturbations_beta * self.perturbation_beta_size
        Q_s = np.matmul(p_s.T, p_s)
        p_beta = p_beta.reshape(self.agent_dist.n, self.agent_dist.d)
        Q_beta = np.matmul(p_beta.T, p_beta)

        # Compute loss
        treatments = scores > cutoff
        loss_vector = treatments * self.true_scores

        gamma_loss_s = np.linalg.solve(Q_s, np.matmul(p_s.T, loss_vector)).reshape(1, 1)
        gamma_loss_beta = np.linalg.solve(
            Q_beta, np.matmul(p_beta.T, loss_vector)
        ).reshape(self.agent_dist.d, 1)

        indicators_beta = beta_perturbed_scores > cutoff
        indicators_s = unperturbed_scores > cutoff
        gamma_pi_s = -np.linalg.solve(Q_s, np.matmul(p_s.T, indicators_s)).reshape(1, 1)
        gamma_pi_beta = -np.linalg.solve(
            Q_beta, np.matmul(p_beta.T, indicators_beta)
        ).reshape(self.agent_dist.d, 1)

        return gamma_loss_s, gamma_loss_beta, gamma_pi_s, gamma_pi_beta, loss_vector

    def compute_density(self, scores, cutoff):
        kde = gaussian_kde(scores.flatten())
        return kde(cutoff).reshape(1, 1)

    def compute_total_derivative(self):
        br = self.get_best_responses()
        scores, unperturbed_scores, beta_perturbed_scores = self.get_scores(br)
        cutoff = np.quantile(scores, self.q).item()

        (
            gamma_loss_s,
            gamma_loss_beta,
            gamma_pi_s,
            gamma_pi_beta,
            loss_vector,
        ) = self.compute_gradients(
            scores, unperturbed_scores, beta_perturbed_scores, cutoff
        )
        density_estimate = self.compute_density(unperturbed_scores, cutoff)

        gamma_s_beta = -(1 / (density_estimate + gamma_pi_s)) * gamma_pi_beta
        total_deriv = (gamma_loss_s * gamma_s_beta) + gamma_loss_beta

        assert total_deriv.shape == (self.agent_dist.d, 1)

        dic = {
            "total_deriv": total_deriv,
            "partial_deriv_loss_s": gamma_loss_s,
            "partial_deriv_loss_beta": gamma_loss_beta,
            "partial_deriv_pi_s": gamma_pi_s,
            "partial_deriv_pi_beta": gamma_pi_beta,
            "partial_deriv_s_beta": gamma_s_beta,
            "density_estimate": density_estimate,
            "loss": loss_vector.mean().item(),
        }
        return dic

    def compute_partial_derivative(self):
        br = self.get_best_responses_partial()
        beta_perturbed_scores = self.get_scores_partial(br)
        cutoff = np.quantile(beta_perturbed_scores, self.q).item()

        gamma_loss_beta, loss_vector = self.compute_gradients_partial(
            beta_perturbed_scores, cutoff
        )

        dic = {
            "partial_deriv_loss_beta": gamma_loss_beta,
            "loss": loss_vector.mean().item(),
        }

        return dic
