"""
Generate simulation datasets matching Oka & Okada (2023) exactly.

Run this ONCE, then all methods (VB, Gibbs, Lasso, IP) load the same data.

Data generating process (from paper Section 3.1):
- Attributes: MVN + threshold, even when rho=0
  alpha_ik = 1 if Z_ik >= Phi^{-1}(k/(K+1))
  For K=3: mastery rates ~75%, 50%, 25%
- Responses: DINA model with s=0.2, g=0.2
- 100 replications per condition
"""

import numpy as np
from scipy.stats import norm
import pickle
import os


def simulate_DINA_data_paper(Q, N, slip=0.2, guess=0.2, rho=0.0, rng=None):
    """
    Generate DINA data matching Oka & Okada (2023) Equation 1.
    
    This is the EXACT data generating process from the paper:
    1. Generate Z ~ MVN(0, Sigma) where Sigma_kk'= rho for k!=k', 1 for k=k'
    2. alpha_ik = 1 if Z_ik >= Phi^{-1}(k/(K+1))
    3. eta_ij = prod_k alpha_ik^{q_jk}  (DINA AND-gate)
    4. P(X_ij=1) = (1-s_j) if eta_ij=1, else g_j
    5. X_ij ~ Bernoulli(P)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    J, K = Q.shape
    
    # Step 1: Generate correlated normal variates
    Sigma = np.full((K, K), rho)
    np.fill_diagonal(Sigma, 1.0)
    Z = rng.multivariate_normal(np.zeros(K), Sigma, size=N)
    
    # Step 2: Threshold to get binary attributes (Equation 1)
    alpha = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        threshold = norm.ppf((k + 1) / (K + 1))
        alpha[:, k] = (Z[:, k] >= threshold).astype(np.float64)
    
    # Step 3: Compute ideal responses (DINA AND-gate)
    natt = Q.sum(axis=1)  # (J,) number of required attributes per item
    eta = (alpha @ Q.T == natt[np.newaxis, :]).astype(np.float64)  # (N, J)
    
    # Step 4-5: Generate responses via inverse transform sampling
    prob = np.where(eta == 1, 1.0 - slip, guess)  # (N, J)
    U = rng.random(size=(N, J))
    Y = (prob >= U).astype(np.float64)
    
    return Y


# ---- True Q-matrix (Oka & Okada K=3, J=10) ----
Q_true_K3J10 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=np.float64)


if __name__ == "__main__":
    K = 3
    J = 10
    n_rep = 100
    N_values = [250, 500, 1000]
    rho = 0.0
    slip = 0.2
    guess = 0.2
    
    print("=" * 60)
    print("  Generating simulation datasets")
    print(f"  Matching Oka & Okada (2023) Section 3.1")
    print(f"  K={K}, J={J}, rho={rho}, s={slip}, g={guess}")
    print(f"  N = {N_values}, {n_rep} replications each")
    print("=" * 60)
    
    print("\nTrue Q-matrix:")
    print(Q_true_K3J10.astype(int))
    print(f"Column sums: {Q_true_K3J10.sum(axis=0).astype(int)}")
    
    # Use fixed master seed for reproducibility
    master_rng = np.random.default_rng(2023)
    
    datasets = {}
    datasets["Q_true"] = Q_true_K3J10
    datasets["K"] = K
    datasets["J"] = J
    datasets["rho"] = rho
    datasets["slip"] = slip
    datasets["guess"] = guess
    datasets["n_rep"] = n_rep
    
    for N in N_values:
        print(f"\nGenerating N={N}...")
        data_list = []
        for rep in range(n_rep):
            Y = simulate_DINA_data_paper(
                Q_true_K3J10, N, slip=slip, guess=guess, rho=rho,
                rng=np.random.default_rng(master_rng.integers(0, 2**31))
            )
            data_list.append(Y)
            if (rep + 1) % 25 == 0:
                print(f"  Rep {rep+1}/{n_rep}")
        
        datasets[f"data_N{N}"] = data_list
    
    # Save
    output_file = "simulation_datasets.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(datasets, f)
    
    print(f"\nDatasets saved to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1e6:.1f} MB")
    
    # Verify
    print("\nVerification:")
    for N in N_values:
        data = datasets[f"data_N{N}"]
        print(f"  N={N}: {len(data)} datasets, each shape {data[0].shape}, "
              f"mean response rate = {np.mean([d.mean() for d in data]):.3f}")
