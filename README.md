# STATS 489 Research — Luke Lin, University of Michigan

This repository contains three interconnected projects on measurement fairness and psychometric modeling, developed for STATS 489.

---

## Projects

### [1. Sensitivity Analysis for DIF Detection](./1-sensitivity-analysis/)

A simulation study evaluating the robustness of Rosenbaum sensitivity analysis when applied to Differential Item Functioning (DIF) detection under hidden confounding. Four item types (A–D) are simulated across three confounding scenarios, and sensitivity (critical lambda) is compared using true vs. estimated ability scores.

**Language:** R  
**Key methods:** 2PL IRT, Logistic Regression DIF, Rosenbaum `binarysens`, optimal pair matching

---

### [2. PISA DIF Analysis (US Sample)](./2-pisa-dif-analysis/)

An empirical DIF analysis on PISA 2022 reading items (US sample) across 12 socioeconomic, demographic, and psychological grouping variables. Multiple DIF detection methods are applied and cross-validated, with Rosenbaum sensitivity analysis used to assess robustness of flagged items.

**Language:** R  
**Key methods:** Logistic Regression DIF, Mantel–Haenszel, Wald Test (IRT), BH correction, Rosenbaum sensitivity analysis  
**Data:** PISA 2022 (available via [OECD](https://www.oecd.org/pisa/data/))

---

### [3. DINA Model with Constrained Q-Matrix Estimation](./3-dina/)

Implementation and simulation study of the DINA (Deterministic Input, Noisy And gate) model with constrained Q-matrix estimation. Compares multiple estimation approaches: Gibbs sampling, Variational Bayes, LASSO, and a Mixed-Integer Linear Program (MILP) via Gurobi. Includes a Python package (`qem-dina`) for EM-based estimation.

**Language:** Python  
**Key methods:** DINA model, EM algorithm, MILP (Gurobi), Gibbs sampler, Variational Bayes, LASSO

---

## Author

Luke Lin — University of Michigan
