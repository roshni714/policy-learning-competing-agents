from scipy.stats import bernoulli, norm, gaussian_kde
import numpy as np
from utils import convert_to_unit_vector, compute_score_bounds


class ExpectedGradient:
    def __init__(
        self,
        agent_dist,
        theta,
        s,
        sigma,
        true_beta,
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
        self.bounds = compute_score_bounds(self.beta, self.sigma)

        self.true_scores = np.array(
            [
                -np.matmul(true_beta.T, agent.eta).item()
                for agent in self.agent_dist.agents
            ]
        ).reshape(self.agent_dist.n_types, 1)
        self.dbeta_dtheta = np.array([-np.sin(theta), np.cos(theta)]).reshape(2, 1)
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
        second_term = np.array(
            [
                np.matmul(self.br_dist[i].T, self.dbeta_dtheta).item()
                for i in range(len(self.grad_theta_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)
        total = first_term + second_term

        res = (
            prob
            * total
            * self.true_scores
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        dl_dtheta = np.sum(res).item()
        return dl_dtheta

    def expected_gradient_pi_theta(self, r):
        dim = self.agent_dist.d
        assert dim == 2, "Method does not work for dimension {}".format(dim)

        z = r - np.array([np.matmul(self.beta.T, x) for x in self.br_dist]).reshape(
            len(self.br_dist), 1
        )
        prob = norm.pdf(z, loc=0.0, scale=self.sigma)

        first_term = np.array(
            [
                np.matmul(self.grad_theta_dist[i].T, self.beta).item()
                for i in range(len(self.grad_theta_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)
        second_term = np.array(
            [
                np.matmul(self.br_dist[i].T, self.dbeta_dtheta).item()
                for i in range(len(self.grad_theta_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)
        total = first_term + second_term

        res = -prob * total * self.agent_dist.prop.reshape(len(self.agent_dist.prop), 1)
        grad_pi_theta = np.sum(res)
        return grad_pi_theta.item()

    def expected_gradient_loss_s(self):
        dim = self.agent_dist.d
        assert dim == 2, "Method does not work for dimension {}".format(dim)

        z = self.s - np.array(
            [np.matmul(self.beta.T, x) for x in self.br_dist]
        ).reshape(len(self.br_dist), 1)
        prob = norm.pdf(z, loc=0.0, scale=self.sigma)

        vec = np.array(
            [
                1 - np.matmul(self.beta.T, self.grad_s_dist[i]).item()
                for i in range(len(self.br_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)

        res = (
            -prob
            * vec
            * self.true_scores
            * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        )
        d_loss_d_s = np.sum(res)
        return d_loss_d_s.item()

    def expected_gradient_pi_s(self, r):
        z = r - np.array([np.matmul(self.beta.T, x) for x in self.br_dist]).reshape(
            len(self.br_dist), 1
        )
        prob = norm.pdf(z, loc=0.0, scale=self.sigma)
        vec = np.array(
            [
                -np.matmul(self.beta.T, self.grad_s_dist[i]).item()
                for i in range(len(self.br_dist))
            ]
        ).reshape(self.agent_dist.n_types, 1)
        res = prob * vec * self.agent_dist.prop.reshape(self.agent_dist.n_types, 1)
        d_pi_d_s = np.sum(res)
        return d_pi_d_s.item()

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

    def compute_total_derivative(self):
        gamma_loss_s = self.expected_gradient_loss_s()
        gamma_loss_theta = self.expected_gradient_loss_theta()
        gamma_pi_s = self.expected_gradient_pi_s(self.s)
        gamma_pi_theta = self.expected_gradient_pi_theta(self.s)
        density_estimate = self.agent_dist.best_response_pdf(
            self.beta, self.s, self.sigma, self.s
        )

        gamma_s_theta = -(1 / (gamma_pi_s + density_estimate)) * gamma_pi_theta
        total_deriv = (gamma_loss_s * gamma_s_theta) + gamma_loss_theta
        dic = {
            "total_deriv": total_deriv,
            "partial_deriv_loss_s": gamma_loss_s,
            "partial_deriv_loss_theta": gamma_loss_theta,
            "partial_deriv_pi_s": gamma_pi_s,
            "partial_deriv_pi_theta": gamma_pi_theta,
            "partial_deriv_s_theta": gamma_s_theta,
            "density_estimate": density_estimate,
            "loss": self.expected_loss(),
        }
        return dic
