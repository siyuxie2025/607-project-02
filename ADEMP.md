
# ADEMP: Simulation Study for Risk-Aware vs OLS Bandit Algorithms

In this project, I conduct the simulation study based on a previous research project on quantile contextual bandit algorithm. 

**Primary Question:**
1. Does risk-aware bandit have a lower cumulative regret than ordinary least square update when the error is heavy-tailed?
2. Does Risk-aware Bandit have a lower beta estimation MSE than OLS bandit?
3. How does the decisions differ from different algorithms?

**Hypotheses:**
- H1: The Risk-Aware Bandit achieves lower cumulative regret than OLS in non-Gaussian, heavy-tailed noise environments.
- H2: The Risk-Aware Bandit yields more stable β-estimates and lower estimation error.


**Data Generating Processes**:
1. contextual vector $X_t$ : Truncated Normal distribution 
$$
x_t \sim \text{TruncNorm}(0, 1)
$$
2. Reward model : 
$$
r_t(a) = x_t^\top \beta_a + \alpha_a + \epsilon_t
$$
where:
- $\beta_a \in \mathbb{R}^d$ and $\alpha_a$ are arm-specific parameters.
- Errors $\epsilon_t$ are generated using a scaled *Student-t* distribution (via `TGenerator`), with varying degrees of freedom (df) and scale.


**Estimands:**
1. $\beta$ in each step, we evaluate via mean-squared error of all $K$ arms.
2. The expected cumulative regret $\sum_{t=1}^T R_t$
3. Proportion of different actions


**Methods:**
Here, I compare the quantile appraoch with ordinary least square approach in [Online Decision Making with High-Dimensional Covariates](https://pubsonline.informs.org/doi/abs/10.1287/opre.2019.1902). 


**Performance measures:**
| Measure | Description | Computation |
|:--|:--|:--|
| **Variance** | Across-simulation variability of β-estimates | Var(β̂) |
| **MSE** | Bias² + Variance | $E[(\hat{\beta}-\beta)^2]$ |
| **Cumulative Regret** | Performance gap to oracle policy | $R_T = \sum (r^*_t - r_t)$ |
| **Action Disagreement** | Proportion of rounds where RAB ≠ OLS action | $ \frac{1}{T} \sum 1(a_{OLS} \neq a_{RAB})$ |

## Simulation Design Matrix

| Generator | K | d | T | τ | n_sim | Notes |
|:--|:--:|:--:|:--:|:--:|:--:|:--|
| T(df in [1.5, 2.25, 3, 5, 10], scaler=0.7) | 2 | 10 | 1000 | 0.5 | 50 | different error distribution |
| TruncNorm | 2 | 10 | 1000 | 0.5 | 50 | bounded contexts |
| T(df=2.25) | 4 | 10 | 1000 | 0.5 | 50 | multi-arm test |
| T(df=2.25) | 2 | 20 | 1000 | 0.5 | 50 | high-dimensional |

