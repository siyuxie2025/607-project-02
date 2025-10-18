import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import seaborn as sns
from methods import RiskAwareBandit, OLSBandit
from tqdm import tqdm
from generators import NormalGenerator, TGenerator, UniformGenerator

class SimulationStudy:
    """
    Orchestrate simulation studies for assessing bandit algorithms.
    This class manages running multiple simulation replications to estimate
    statistical properties. It handles:
    - Running individual scenarios (one generator + one bandit algorithm)
    - Running factorial designs (all combinations of generators and bandit algorithms)
    - Computing Monte Carlo confidence intervals for estimates
    - Storing and reporting results"""
    def __init__(self, n_sim, K, d, T, q, h, tau, err_generator, context_generator, beta_low=0.0, beta_high=1.0, random_seed=None):
        self.n_sim = n_sim
        self.K = K
        self.d = d
        self.T = T
        self.q = q
        self.h = h
        self.tau = tau
        self.err_generator = err_generator
        self.context_generator = context_generator
        self.random_seed = random_seed

        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
            np.random.seed(random_seed)
        else:
            self.rng = np.random.default_rng()

        self.q_err = np.quantile(self.err_generator.generate(2000, rng=self.rng), self.tau)

        # Generate real beta and alpha values once for all simulations
        self.beta_real_value = UniformGenerator(low=beta_low, high=beta_high).generate((self.K, self.d), rng=self.rng)
        self.alpha_real_value = UniformGenerator(low=beta_low, high=beta_high).generate(self.K, rng=self.rng)

        # Store results
        self.results = None


    def run_one_scenario(self):
        """Run one scenario to compute cumulative regret for both bandit algorithms.    
        """
        RAB = RiskAwareBandit(q = self.q, h = self.h, tau = self.tau, 
                              d = self.d, K = self.K, 
                              beta_real_value = self.beta_real_value, 
                              alpha_real_value = self.alpha_real_value)
        OLSB = OLSBandit(q = self.q, h = self.h, d = self.d, K = self.K,
                         beta_real_value= self.beta_real_value)

        diff = 0
        RWD = []
        RWD_OLS = []
        opt_RWD = []
        opt_RWD_OLS = []

        for t in tqdm(range(1,self.T+1), desc="Running one scenario"):
            rwd, RAB, rwd_OLS, OLSB, opt_rwd, opt_rwd_OLS, diff = self._run_one_timestep(RAB, OLSB, t, diff)
            RWD.append(rwd)
            RWD_OLS.append(rwd_OLS)
            opt_RWD.append(opt_rwd)
            opt_RWD_OLS.append(opt_rwd_OLS)
        
        regret_RAB = np.cumsum(opt_RWD) - np.cumsum(RWD)
        regret_OLSB = np.cumsum(opt_RWD_OLS) - np.cumsum(RWD_OLS)

        return regret_RAB, regret_OLSB, diff


    def _run_one_timestep(self, RAB, OLSB, t, diff, err_generator=None):
        """Run one timestep of the simulation for both bandit algorithms.
        If err_generator is not provided, use the one from initialization.
        """
        if err_generator is None:
            err_generator = self.err_generator

        x = self.context_generator.generate(1, rng=self.rng)[0]
        a = RAB.choose_a(t, x)
        a_OLS = OLSB.choose_a(t, x)

        diff += (a != a_OLS)

        # Risk Aware Bandit update
        rwd = np.dot(self.beta_real_value, x)[a] + self.alpha_real_value[a]
        rwd_noisy = rwd + (0.5 * x[-1] + 1) * (err_generator.generate(1, rng=self.rng) - self.q_err)
        RAB.update_beta(rwd_noisy, t)

        # OLS Bandit update
        rwd_OLS = np.dot(self.beta_real_value, x)[a_OLS] + self.alpha_real_value[a_OLS]
        rwd_OLS_noisy = rwd_OLS + (0.5 * x[-1] + 1) * (err_generator.generate(1, rng=self.rng) - self.q_err)
        OLSB.update_beta(rwd_OLS_noisy, t)

        # Optimal rewards (same for both)
        opt_rwd = np.amax(np.dot(self.beta_real_value, x) + self.alpha_real_value)  # optimal reward

        return rwd, RAB, rwd_OLS, OLSB, opt_rwd, diff

    

    def run_simulation(self):
        cumulated_regret_RiskAware = []
        cumulated_regret_OLS = []
        num_diff = []

        for _ in tqdm(range(self.n_sim), desc="Running simulations"):
            regret_RAB, regret_OLSB, diff = self.run_one_scenario()

            cumulated_regret_RiskAware.append(regret_RAB)
            cumulated_regret_OLS.append(regret_OLSB)
            num_diff.append(diff/self.T)

        self.results = {
            "cumulated_regret_RiskAware": np.array(cumulated_regret_RiskAware),
            "cumulated_regret_OLS": np.array(cumulated_regret_OLS),
            "num_diff": np.array(num_diff)
        }

        return self.results

    def plot_results(self, results = None, figsize = (10,6), use_ci = True, ci_level = 0.95):
        """Plot simulation results with confidence intervals or min/max ranges.
        """
        steps = np.arange(1,self.T+1)

        if results is None:
            if self.results is None:
                raise ValueError("No results to plot. Please run the simulation first.")
            results = self.results

        cumulated_regret_RiskAware = results["cumulated_regret_RiskAware"]
        cumulated_regret_OLS = results["cumulated_regret_OLS"]
        num_diff = results["num_diff"]

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))

        # Left plot: Cumulative Regret
        ax1 = axes[0]
        mean_risk_aware = np.mean(cumulated_regret_RiskAware, axis=0)
        mean_ols = np.mean(cumulated_regret_OLS, axis=0)

        if use_ci:
            # Compute standard errors
            se_risk_aware = np.std(cumulated_regret_RiskAware, axis=0, ddof=1) / np.sqrt(self.n_sim)
            se_ols = np.std(cumulated_regret_OLS, axis=0, ddof=1) / np.sqrt(self.n_sim)

            t_crit = stats.t.ppf((1 + ci_level) / 2, df=self.n_sim - 1)

            # Confidence intervals
            lower_risk_aware = mean_risk_aware - t_crit * se_risk_aware
            upper_risk_aware = mean_risk_aware + t_crit * se_risk_aware
            lower_ols = mean_ols - t_crit * se_ols
            upper_ols = mean_ols + t_crit * se_ols

            ci_label = f"{int(ci_level*100)}% CI"
        else:
            # Min/max ranges
            lower_risk_aware = np.min(cumulated_regret_RiskAware, axis=0)
            upper_risk_aware = np.max(cumulated_regret_RiskAware, axis=0)
            lower_ols = np.min(cumulated_regret_OLS, axis=0)
            upper_ols = np.max(cumulated_regret_OLS, axis=0)

            ci_label = "Min/Max Range"

        # Plot Risk Aware Bandit
        ax1.fill_between(steps, lower_risk_aware, upper_risk_aware, 
                         color='red', alpha=0.2, label=f'Risk Aware {ci_label}')
        ax1.plot(steps, mean_risk_aware, 'r-', 
                label='Risk Aware Bandit (Mean)', linewidth=2)

        # Plot OLS Bandit
        ax1.fill_between(steps, lower_ols, upper_ols, 
                         color='blue', alpha=0.2, label=f'OLS {ci_label}')
        ax1.plot(steps, mean_ols, 'b-', 
                label='OLS Bandit (Mean)', linewidth=2)

        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel(f'Cumulative Regret', fontsize=12)
        ax1.legend(loc='best')
        ax1.set_title(f'Cumulative Regret, d={self.d}, K={self.K}, tau={self.tau}', fontsize=14)

        # Plot the number of different actions
        ax2 = axes[1]
        ax2.plot(num_diff, 'g-', linewidth=1.5)
        ax2.axhline(y=np.mean(num_diff), color='orange', linestyle='--', label=f'Mean: {np.mean(num_diff):.3f}')
        ax2.set_xlabel('Simulation', fontsize=12)
        ax2.set_ylabel('Proportion of Different Actions', fontsize=12)
        ax2.set_title('Action Disagreement between Algorithms', fontsize=14)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'simulation_results_d{self.d}_K{self.K}_tau{self.tau}.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.show()

        # Print summary statistics
        print(f"\n=== Simulation Summary (n_sim={self.n_sim}) ===")
        print(f"Final Median Regret - Risk Aware: {np.median(cumulated_regret_RiskAware[:,-1]):.2f}")
        print(f"Final Median Regret - OLS: {np.median(cumulated_regret_OLS[:,-1]):.2f}")
        print(f"Mean Action Disagreement: {np.mean(num_diff):.2%}")
        print(f"Std Action Disagreement: {np.std(num_diff):.2%}")


## usage example:
if __name__ == "__main__":
    n_sim = 50
    K = 2
    d = 10
    T = 10
    q = 0.1
    h = 0.5
    tau = 0.5

    RANDOM_SEED = 1010

    err_generator = TGenerator(df=3)
    context_generator = NormalGenerator(mean=0.0, std=1.0)

    study = SimulationStudy(n_sim=n_sim, K=K, d=d, T=T, q=q, h=h, tau=tau, random_seed=RANDOM_SEED,
                                err_generator=err_generator,
                                context_generator=context_generator)

    results = study.run_simulation()
    study.plot_results(results=results, use_ci=True, ci_level=0.95)