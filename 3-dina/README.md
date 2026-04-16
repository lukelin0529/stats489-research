# DINA Q-Matrix Estimation: Method Comparison Study

A Monte Carlo simulation study comparing methods for estimating the Q-matrix in the DINA diagnostic classification model. Replicates and extends Oka & Okada (2023, *Psychometrika*) by benchmarking Variational Bayes (VB) against Gibbs sampling, EM+Lasso, and Integer Programming (IP) approaches on identical simulated datasets.

---

## Project Structure

```
3-dina/
├── README.md                        # This file
├── dina_utils.py                    # Shared: data generation, recovery metrics
├── generate_datasets.py             # Generates simulation datasets (run once)
├── dina_vb_estimator.py             # VB method (Oka & Okada 2023)
├── dina_gibbs_estimator.py          # Gibbs sampler (Chung 2019)
├── dina_lasso_estimator.py          # EM+Lasso (Chen et al. 2015)
├── run_simulation.py                # Standalone VB/Gibbs/Lasso runner
├── run_all_methods.py               # Integrated runner (VB + IP-20 + IP-50)
├── final_results.csv                # Simulation output
└── qem-dina/                        # IP method package (EM + MILP via Gurobi)
```

**External dependency:** The IP method uses the `qem-dina` package (in `qem-dina/`). To run `run_all_methods.py`, the package must be accessible via `PYTHONPATH`:

```bash
export PYTHONPATH=./qem-dina/src:$PYTHONPATH
```

---

## Simulation Design

All methods are compared on **identical** simulated datasets to ensure fair comparison. Data is generated once via `generate_datasets.py` and saved to `simulation_datasets.pkl`.

**Data generating process** (Oka & Okada 2023, Equation 1):
- K = 3 attributes, J = 10 items
- True Q-matrix from paper (Table 1)
- Attributes generated via multivariate normal + thresholding:
  `alpha_ik = 1 if Z_ik >= Phi^{-1}(k/(K+1))`
- Slip and guess: s = g = 0.2
- Correlation: rho = 0
- Sample sizes: N ∈ {250, 500, 1000}
- Replications: 100 per condition

**Recovery metrics** (computed after best column permutation):
- **eMRR** (element-wise mean recovery rate): fraction of correctly estimated Q-matrix cells
- **mMRR** (matrix-wise mean recovery rate): count of perfect Q-matrix recoveries out of 100

---

## Method Details

### Variational Bayes (VB) — Oka & Okada (2023)
Stochastic VB with mini-batch sampling. Translated from the authors' Julia code.
- nrun = 10 parallel chains
- niter = 550 iterations per chain, nburn = 50
- batchsize = 200 (N=250) or 300 (N≥500)
- Parallelized via `joblib` on 7 cores

### Gibbs Sampler — Chung (2019)
MCMC-based Bayesian estimation with Dirichlet/Beta conjugate priors.
- 3 chains × 10,000 iterations × 5,000 burn-in
- Samples Q per-item conditional on other parameters
- Translated from `DINA_QestMCMC.jl`

### EM + Lasso — Chen et al. (2015)
Penalized maximum likelihood with L1 regularization.
- 10 parallel paths with different lambda values
- Adaptive lambda update until row sparsity achieved
- Translated from `DINA_Lasso.jl`

### Integer Programming (IP) — Professor's method
EM with MIP-based M-step using Gurobi solver.
- Multistart configurations: 20 and 50
- SAT/MaxSAT sampler for diverse Q initializations
- Constraints: col_lb=[1]*K, row_lb=[1]*J (each skill measured, each item has ≥1 skill)
- Requires `gurobipy` with academic license and `python-sat`

---

## How to Run

### 1. Install dependencies
```bash
pip3 install numpy scipy joblib gurobipy python-sat
```

### 2. Generate datasets (once)
```bash
python3 generate_datasets.py
```
Creates `simulation_datasets.pkl` with 100 datasets per N value.

### 3. Run full comparison
```bash
export PYTHONPATH=./qem-dina/src:$PYTHONPATH
python3 run_all_methods.py
```

For quick testing, edit line 190 of `run_all_methods.py` to set `n_rep = 10`.

---

## Results Summary (100 replications, K=3, J=10, rho=0)

| Method | N=250 |  | N=500 |  | N=1000 |  |
|---|---|---|---|---|---|---|
|  | mMRR | eMRR | mMRR | eMRR | mMRR | eMRR |
| **VB (paper)** | 41/100 | 97.0% | 80/100 | 99.2% | 91/100 | 99.7% |
| **VB (ours)** | 34/100 | 96.4% | 75/100 | 99.1% | 96/100 | 99.8% |
| **IP-20** | 0/100 | 83.6% | 0/100 | 81.7% | 0/100 | 81.4% |
| **IP-50** | 0/100 | 85.1% | 0/100 | ~83% | 0/100 | ~84% |

**Key finding:** VB's stochastic mini-batch optimization substantially outperforms the deterministic EM+MIP approach. The IP method's flat recovery across N suggests it gets stuck in local optima that more data cannot fix.

Gibbs and EM+Lasso results available in `run_simulation.py` output.

---

## References

- Oka, M., & Okada, K. (2023). Scalable Bayesian approach for the DINA Q-matrix estimation combining stochastic optimization and variational inference. *Psychometrika*, 88, 302–331.
- Chung, M. T. (2019). A Gibbs sampling algorithm that estimates the Q-matrix for the DINA model. *Journal of Mathematical Psychology*, 93, 102275.
- Chen, Y., Liu, J., Xu, G., & Ying, Z. (2015). Statistical analysis of Q-matrix based diagnostic classification models. *Journal of the American Statistical Association*, 110, 850–866.

---

## Author

Luke Lin — University of Michigan
