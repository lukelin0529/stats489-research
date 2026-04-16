# Sensitivity Analysis for DIF Detection

## Overview

This project investigates whether **Rosenbaum sensitivity analysis** can reliably distinguish true DIF items from items that appear biased due to hidden confounding. We simulate item response data under a latent confounding model and compare sensitivity (critical lambda) across four item types and three confounding scenarios.

### Item Types

| Type | DIF (γ) | Hidden Confounder Effect (δ) | Interpretation |
|------|---------|-------------------------------|----------------|
| A    | 0       | 0                             | No DIF, no confounding (anchor) |
| B    | ≠ 0     | 0                             | True DIF only |
| C    | 0       | ≠ 0                           | Confounding only (false positive risk) |
| D    | ≠ 0     | ≠ 0                           | Both DIF and confounding |

### Scenarios

- **Scenario 1:** No hidden education effect (baseline)
- **Scenario 2:** Moderate hidden confounder (α_U = 0.7, δ_C = 0.5)
- **Scenario 3:** Strong hidden confounder (α_U = 1.2, δ_C = 0.7)

### Core Question

Does sensitivity analysis (critical λ) correctly assign high robustness to Type B items and low robustness to Type C/D items, across different ability score estimates?

---

## Files

| File | Description |
|------|-------------|
| `DIF_simu.R` | Data-generating model: simulates θ, U, Z, Y under A/B/C/D item structure |
| `DIF_sens.R` | Rosenbaum `binarysens` wrapper: pair-matches on θ, computes critical λ |
| `App_utils.R` | Matching utilities (rank-based Mahalanobis distance, caliper, balance plots) |
| `Run_sen.R` | SLURM array job runner — runs one replicate per task ID, saves `.rds` |
| `example.R` | Quick start: single-run DIF analysis with `difR`, lordif, MH, MIMIC, Wald |
| `example_full.R` | Full pipeline: all 9 DIF methods + sensitivity analysis in one script |
| `submit_sen.sbatch` | HPC submission script (Great Lakes, 50 array jobs × 24 h) |

---

## Reproducing the Simulation

### Local (single replicate)

```r
source("DIF_simu.R")
source("DIF_sens.R")

cfg <- make_scenario_3(N = 2000, J_A = 100, J_B = 10, J_C = 10, J_D = 10)
sim <- simulate_abcd(cfg)

# Estimate theta via 2PL
library(mirt)
mod <- mirt(as.data.frame(sim$Y), model = 1, itemtype = "2PL")
theta_hat <- as.numeric(fscores(mod, method = "EAP")[, 1])

# Run sensitivity for all items
sens_results <- lapply(1:ncol(sim$Y), function(j) {
  binarysens_for_item(j, Y = sim$Y, Z = sim$Z, theta_vec = theta_hat)
})

crit_lambdas <- sapply(sens_results, critical_lambda)
```

### HPC (50 replicates)

```bash
sbatch submit_sen.sbatch
```

Results are saved to `results/sens_run_<id>.rds`.

---

## Dependencies

```r
library(mirt)
library(difR)
library(lordif)
library(optmatch)
library(rbounds)
library(dplyr)
library(lattice)
```

---

## Author

Luke Lin — University of Michigan
