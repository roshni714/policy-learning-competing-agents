from scipy.stats import norm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils import compute_score_bounds
from agent import Agent


class AgentDistribution:
    """This is a class for representing a distribution over a finite number of agents.
    
    Keyword arguments:
    n -- number of agents in distribution (float)
    d -- dimension of agent (float)
    n_types -- number of agent types (float)
    types -- optional argument: a dictionary of agent types of the form 
        {etas: (n_types, D, 1), gammas: (n_types, D, 1)}
    prop -- optional argument: proportion of population with each type (D,1) array, by default this is uniform.
    """

    def __init__(self, n=1000, d=2, n_types=50, types=None, prop=None):
        self.n = n
        self.d = d
        self.n_types = n_types
        self.types = types
        self.prop = prop
        if types is None:
            # Generate n_types agent types randomly
            etas = np.random.uniform(0.4, 0.6, size=n_types * d).reshape(n_types, d, 1)
            gammas = np.ones((n_types, d, 1)) * 8
        #            gammas = np.random.uniform(1.0, 2.0, size=n_types * d).reshape(
        else:
            etas = types["etas"]
            gammas = types["gammas"]
        if not prop:
            self.prop = np.ones(n_types) * (1 / n_types)
        else:
            self.prop = prop
        np.testing.assert_allclose(np.sum(self.prop), 1.0)
        self.n_agent_types = np.random.choice(
            list(range(self.n_types)), self.n, p=self.prop
        )

        # Create representative agents
        self.agents = []
        for i in range(n_types):
            self.agents.append(Agent(etas[i], gammas[i]))

    def get_etas(self):
        """Method that returns the etas for all agents in the distribution.

        Returns:
        etas -- (N, D, 1) array
        """
        etas = []
        for i in range(self.n):
            # get type of ith agent
            agent_type = self.n_agent_types[i]
            # get agent that has  type agent_type
            agent = self.agents[agent_type]
            # get eta
            etas.append(agent.eta)
        etas = np.array(etas).reshape(self.n, self.d, 1)
        return etas

    def get_gammas(self):
        """Method that returns the gammas for all agents in the distribution

        Returns:
        gammas -- (N, D, 1) array
        """
        gammas = []
        for i in range(self.n):
            # get type of ith agent
            agent_type = self.n_agent_types[i]
            # get agent that has  type agent_type
            agent = self.agents[agent_type]
            # get eta
            gammas.append(agent.gamma)
        gammas = np.array(gammas).reshape(self.n, self.d, 1)
        return gammas

    def best_response_distribution(self, beta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold.
        
        Keyword arguments:
        beta -- model parameters (D,1) array
        s -- threshold (float)
        sigma -- standard deviation of noise distribution (float)
        
        Returns:
        br -- a list of np.arrays
        """
        br = []
        for agent in self.agents:
            br.append(agent.best_response(beta, s, sigma))
        return br

    def br_gradient_theta_distribution(self, theta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold and the gradient wrt to theta.
        
        Keyword arguments:
        theta -- model parameters (D,1) array
        s -- threshold (float)
        sigma -- standard deviation of noise distribution (float)
        
        Returns
        br -- a list of np.arrays of dimension (D, 1)
        grad -- a list of np.arrays of dimension (D, 1)
        """
        br = []
        grad = []
        for agent in self.agents:
            b, j = agent.br_gradient_theta(theta, s, sigma)
            grad.append(j)
            br.append(b)
        return br, grad

    def br_gradient_s_distribution(self, beta, s, sigma):
        """This is a method that returns the best response of each agent type to a model and threshold and the derivative wrt to s.
        
        Keyword arguments:
        beta -- model parameters
        s -- threshold
        sigma -- standard deviation of noise distribution
        
        Returns:
        br -- a list of np.arrays of dimension (D, 1)
        deriv_s -- a list of np.arrays of dimension (D, 1)
        """
        br = []
        deriv_s = []
        for agent in self.agents:
            b, d = agent.br_gradient_s(beta, s, sigma)
            deriv_s.append(d)
            br.append(b)
        return br, deriv_s

    def best_response_score_distribution(self, beta, s, sigma):
        """This is a method that returns the score of the best response of each agent type to a model and threshold.
        
        Keyword arguments:
        beta -- model parameters (D, 1) array
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        
        Returns:
        br_score_dist -- a (n_types,) dimensional array
        """
        br_score_dist = [
            np.matmul(np.transpose(beta), x).item()
            for x in self.best_response_distribution(beta, s, sigma)
        ]
        return np.array(br_score_dist)

    def best_response_noisy_score_distribution(self, beta, s, sigma):
        """This is a method that returns the distribution over agent scores after noise has been added
        
        Keyword arguments:
        beta -- model parameters (D,1) array
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        
        Returns:
        br_dist -- a (N, 1) dimensional array
        """
        bounds = compute_score_bounds(beta, sigma)
        noisy_scores = norm.rvs(loc=0.0, scale=sigma, size=self.n)
        br_dist = self.best_response_score_distribution(beta, s, sigma)

        n_br = br_dist[self.n_agent_types]
        noisy_scores += n_br
        #        noisy_scores = np.clip(noisy_scores, a_min=bounds[0], a_max=bounds[1])
        return noisy_scores.reshape(self.n, 1)

    def quantile_best_response(self, beta, s, sigma, q):
        """The method returns the qth quantile of the noisy score distribution.
        
        Keyword arguments:
        beta -- model parameters (D,1) array
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        
        Returns:
        q_quantile -- qth quantile of the noisy score distribution (float)
        """
        noisy_scores = self.best_response_noisy_score_distribution(beta, s, sigma)
        q_quantile = np.quantile(noisy_scores, q)
        return q_quantile.item()

    def plot_quantile_best_response(self, beta, sigma, q):
        """This method plots the quantile of the noisy score distribution vs. thresholds.
        
        Keyword arguments:
        beta -- model parameters (D,1) array
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        """
        bounds = compute_score_bounds(beta, sigma)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        quantile_br = [
            self.quantile_best_response(beta, s, sigma, q) for s in thresholds
        ]

        plt.plot(thresholds, quantile_br)
        plt.xlabel("Thresholds")
        plt.ylabel("Quantile BR")
        plt.title("Quantile BR vs. Threshold")

    def quantile_mapping_vary_s(self, beta, sigma, q):
        """This method returns the quantile mapping function q(beta, s) for fixed beta.
        
        Keyword arguments:
        beta -- model parameters (Nx1)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)

        Returns:
        q_function: interpolator function (Python function that maps threshold (1,) array to q(beta, s) (1,) array)
        """
        bounds = compute_score_bounds(beta, sigma)
        thresholds = np.linspace(bounds[0], bounds[1], 50)
        quantile_map = []

        for s in tqdm.tqdm(thresholds):
            cdf_vals = []
            for r in thresholds:
                cdf_vals.append(self.best_response_cdf(beta, s, sigma, r))
            inverse_cdf_s = interp1d(cdf_vals, thresholds, kind="linear")
            quantile_map.append(inverse_cdf_s(q))
        q_function = interp1d(thresholds, quantile_map)
        return q_function

    def quantile_mapping_vary_beta(self, s, sigma, q):
        """This method returns the quantile mapping function q(beta, s) for fixed s.
        CAUTION: This method assumes that model is represented by 1-dimensional polar coordinate theta.
        
        Keyword arguments:
        s -- threshold (float)
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)

        Returns:
        q_function: interpolator function (Python function that maps theta. (1,) array to q(beta, s) (1,) array)
        valid_theta: a list of theta values such that include s in the range of their scores.
        """
        thetas = np.linspace(-np.pi, np.pi, 50)
        quantile_map = []
        valid_theta = []
        for theta in tqdm.tqdm(thetas):
            cdf_vals = []
            beta = np.array([np.cos(theta), np.sin(theta)]).reshape(2, 1)
            bounds = compute_score_bounds(beta, sigma)
            score_range = np.linspace(bounds[0], bounds[1], 50)
            if s >= bounds[0] and s <= bounds[1]:
                for r in score_range:
                    cdf_vals.append(self.best_response_cdf(beta, s, sigma, r))
                inverse_cdf_theta = interp1d(cdf_vals, score_range, kind="linear")
                plt.plot(cdf_vals, score_range)
                plt.xlabel("q")
                plt.ylabel("F^-1(q)")
                quantile_map.append(inverse_cdf_theta(q))
                valid_theta.append(theta)
        q = interp1d(valid_theta, quantile_map, kind="linear")
        return q, valid_theta

    def quantile_fixed_point_true_distribution(self, beta, sigma, q):
        """Finds the fixed point of quantile mapping (with fixed beta) via binary search.
        Very important function!! Do not delete! Returns s_star such that s = q(beta, s).

        Keyword arguments:
        beta -- model parameters (D,1) array
        sigma -- standard deviation of noise distribution (float)
        q -- quantile (float)
        
        Returns:
        s_star -- fixed point (float)
        """

        def compute_fs_s(s):
            cdf_val = 0.0
            for i, agent in enumerate(self.agents):
                cdf_val += (
                    norm.cdf(
                        s - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
                        loc=0.0,
                        scale=sigma,
                    )
                    * self.prop[i]
                )
            return cdf_val.item()

        bounds = compute_score_bounds(beta, sigma)
        l = bounds[0]
        r = bounds[1]
        curr = np.array([(l + r) / 2])
        val = compute_fs_s(curr)
        count = 0
        while abs(val - q) > 1e-10:
            if val > q:
                r = curr
            if val < q:
                l = curr

            curr = (l + r) / 2
            val = compute_fs_s(curr)
            count += 1
            if count > 30:
                break

        s_star = curr.item()
        return s_star

    def quantile_mapping_true_distribution(self, beta, s, sigma, q):
        """Computes q(beta, s) for any choice of beta and s.

        Keyword arguments:
        beta -- model parameters (D,1) array
        sigma -- standard deviation of noise distribution (float)
        q -- quantile (float)
        
        Returns:
        curr -- value of q(beta, s) (float)
        """

        def compute_fs_curr(curr):
            return self.best_response_cdf(beta, s, sigma, curr)

        #        def compute_fs_curr(curr):
        #            cdf_val = 0.0
        #            for i, agent in enumerate(self.agents):
        #                cdf_val += (
        #                    norm.cdf(
        #                        curr - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
        #                        loc=0.0,
        #                        scale=sigma,
        #                    )
        #                    * self.prop[i]
        #                )
        #            return cdf_val.item()

        bounds = compute_score_bounds(beta, sigma)
        l = bounds[0]
        r = bounds[1]
        curr = np.array([(l + r) / 2])
        val = compute_fs_curr(curr)
        count = 0
        while abs(val - q) > 1e-10:
            if val > q:
                r = curr
            if val < q:
                l = curr

            curr = (l + r) / 2
            val = compute_fs_curr(curr)
            count += 1
            if count > 30:
                break

        return curr.item()

    def best_response_pdf(self, beta, s, sigma, r):
        """Computes value of PDF of noisy best response score distribution at r for any choice of beta and s.

        Keyword arguments:
        beta -- model parameters (D,1) array
        s -- perceived threshold (float)
        sigma -- standard deviation of noise distribution (float)
        r -- value at which to evaluate PDF (float)

        Returns:
        pdf_val -- value of PDF of best response score distribution at r (float)
        """

        bounds = compute_score_bounds(beta, sigma)
        if s < bounds[0]:
            return 0.0
        if s > bounds[1]:
            return 0.0

        pdf_val = 0.0

        for i, agent in enumerate(self.agents):
            pdf_val += (
                norm.pdf(
                    r - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
                    loc=0.0,
                    scale=sigma,
                )
                * self.prop[i]
            )
        return pdf_val.item()

    def best_response_cdf(self, beta, s, sigma, r):
        """Computes value of CDF of noisy best response score distribution at r for any choice of beta and s.

        Keyword arguments:
        beta -- model parameters (D,1) array
        s -- perceived threshold (float)
        sigma -- standard deviation of noise distribution (float)
        r -- value at which to evaluate CDF (float)

        Returns:
        cdf_val -- value of CDF of best response score distribution at r (float)
        """

        bounds = compute_score_bounds(beta, sigma)
        if s < bounds[0]:
            return 0.0
        if s > bounds[1]:
            return 1.0

        cdf_val = 0.0
        for i, agent in enumerate(self.agents):
            cdf_val += (
                norm.cdf(
                    r - np.matmul(beta.T, agent.best_response(beta, s, sigma)),
                    loc=0.0,
                    scale=sigma,
                )
                * self.prop[i]
            )
        return cdf_val.item()

    def quantile_fixed_point_iteration(
        self, beta, sigma, q, maxiter=200, s0=0.5, plot=False
    ):
        """This method computes the iterates of stochastic fixed point iteration with the random quantile operator.
        Note that the iterates are random variables and will not converge to deterministic fixed point.
        
        Keyword arguments:
        beta -- model parameters (D,1) array
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        maxiter -- number of iterations of FPI (int)
        s0 -- initial threshold (float)
        plot -- plotting function (bool)

        Returns:
        all_s -- a list of all the iterates of the stochastic fixed point iteration.
        """
        bounds = compute_score_bounds(beta, sigma)
        thresholds = np.linspace(bounds[0], bounds[1], 50)

        all_s = [s0]
        s = s0
        for k in range(maxiter):
            new_s = self.quantile_best_response(beta, s, sigma, q)
            all_s.append(new_s)
            s = new_s
            self.resample()
            if plot and k % 50 == 0 and k > 0:
                plt.plot(list(range(len(all_s))), all_s, label="n={}".format(self.n))
                plt.xlabel("Iteration " + r"$t$")
                plt.ylabel("Threshold " + r"$s_t$")
                plt.legend()
                plt.show()
                plt.close()
        return all_s

    def quantile_fixed_point_iteration_true_distribution(
        self, beta, sigma, q, maxiter=200, s0=0.5, plot=False
    ):
        """ 
        This method computes the iterates of fixed point iteration with the population quantile operator. 
        by fixed point iteration. When the population quantile operator is a contraction, this will converge 
        to the deterministic fixed point. Otherwise, it is not guaranteed to converge.
        
        Keyword arguments:
        beta -- model parameters (D,1) array
        sigma -- standard deviation of noise distribution(float)
        q -- quantile between 0 and 1 (float)
        maxiter -- number of iterations of FPI (int)
        s0 -- initial threshold (float)
        plot -- plotting function (bool)

        Returns:
        all_s -- a list of all the iterates of the fixed point iteration
        """
        bounds = compute_score_bounds(beta, sigma)
        thresholds = np.linspace(bounds[0], bounds[1], 50)

        all_s = [s0]
        s = s0
        for k in range(maxiter):
            new_s = self.quantile_mapping_true_distribution(beta, s, sigma, q)
            all_s.append(new_s)
            s = new_s
            if plot and k % 50 == 0 and k > 0:
                plt.plot(list(range(len(all_s))), all_s, label="Population")
                plt.ylim(min(all_s) - 0.05, max(all_s) + 0.05)
                plt.xlabel("Iteration " + r"$t$")
                plt.ylabel("Threshold " + r"$s_t$")
                plt.legend()
                plt.show()
                plt.close()
        return all_s

    def resample(self):
        """
        This method generates a new empirical distribution by resampling the types of the n agents.
        """
        self.n_agent_types = np.random.choice(
            list(range(self.n_types)), self.n, p=self.prop
        )


if __name__ == "__main__":

    agent_dist = AgentDistribution()
    etas = agent_dist.get_etas()
    gammas = agent_dist.get_gammas()
    etas2 = agent_dist.get_etas()
    print(etas.shape)
    print(gammas.shape)
