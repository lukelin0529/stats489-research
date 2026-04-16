"""
DINA Q-Matrix Simulation — IP Method (Professor's DINAEM)
Runs multistart=20 and multistart=50 on the same data as VB/Gibbs/Lasso.

Usage:
    export PYTHONPATH=~/Desktop/dina-qip/src:$PYTHONPATH
    cd ~/Desktop/dina_python
    python3 run_ip_simulation.py

Requires: gurobipy, python-sat, dina package (via PYTHONPATH)
"""

import sys, os
import numpy as np
import time
import csv

# Add dina package to path
sys.path.insert(0, os.path.expanduser("~/Desktop/dina-qip/src"))

from dina.config import EMConfig, QMipConfig, QSatConfig
from dina.em import DINAEM
from dina_utils import simulate_DINA_data, get_Q_true_K3J10, compute_recovery


def run_ip_method(Q_true, N, n_rep=100, multistart=20, seed_base=4649):
    """
    Run the professor's IP method on simulated data.
    """
    J, K = Q_true.shape

    # EM config
    em_cfg = EMConfig(
        multistart=multistart,
        max_iter=200,
        tol=1e-6,
        verbose=False,  # quiet for simulation
    )

    # Q estimation (MIP) config
    col_lb = [1] * K
    q_cfg = QMipConfig(
        include_identity=True,
        include_distinctness=False,     # changed! true Q has duplicates
        lexi=True,
        lexi_ascending=False,
        lexi_strict=False,             # changed! relaxed
        hierarchy_edges=None,
        time_limit=60.0,
        mip_gap=1e-4,
        log_to_console=False,
        col_lb=col_lb,
        row_lb=[1] * J,
    )

    # Q init sampling (SAT) config
    q_sat_cfg = QSatConfig.from_qmip(
        q_cfg,
        frac_assumption_sat=0.75,
        frac_sparse_maxsat=0.25,
        frac_dense_maxsat=0.00,
        min_hamming_frac=0.05,
        assumption_frac=0.25,
        tries_per_start=40,
        solver_name="g4",
    )

    em = DINAEM()
    s_true = np.full(J, 0.2)
    g_true = np.full(J, 0.2)

    recoveries = []
    times_list = []

    rng = np.random.RandomState(seed_base)
    rep_seeds = rng.choice(range(1000, 100000), n_rep, replace=False)

    print(f"\n{'='*60}")
    print(f"  IP (multistart={multistart}) — N={N}, {n_rep} replications")
    print(f"{'='*60}")

    for rep in range(n_rep):
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Rep {rep+1}/{n_rep}")

        # Generate data using OUR method (matching VB/Gibbs/Lasso)
        np.random.seed(rep_seeds[rep])
        sim = simulate_DINA_data(Q_true, N)
        R = sim["Y"]

        # Run IP
# Run IP with retry
        t0 = time.time()
        recovery = 0.0
        for attempt in range(3):
            try:
                out = em.fit(
                    R, K,
                    em_cfg=em_cfg,
                    q_cfg=q_cfg,
                    q_sat_cfg=q_sat_cfg,
                    seed=int(rep_seeds[rep]),
                )
                Q_hat = out["Q"].astype(np.float64)
                recovery = compute_recovery(Q_true, Q_hat, K)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"    Rep {rep+1} failed after 3 attempts: {e}")
        elapsed = time.time() - t0
        time.sleep(2)  # longer delay between reps# avoid WLS rate limiting
        recoveries.append(recovery)
        times_list.append(elapsed)

    eMRR = np.mean(recoveries)
    mMRR = np.sum(np.array(recoveries) == 1.0)
    mean_time = np.mean(times_list)

    print(f"\n  Results: eMRR={eMRR*100:.2f}%  mMRR={mMRR}/{n_rep}  time={mean_time:.1f}s")

    return {
        "eMRR": eMRR, "mMRR": mMRR, "mean_time": mean_time,
        "all_recovery": recoveries, "all_times": times_list
    }


if __name__ == "__main__":
    Q_true = get_Q_true_K3J10()
    K = 3
    N_values = [250, 500, 1000]
    n_rep = 10  # Change to 10 for quick test

    print("=" * 70)
    print("  DINA Q-Matrix Simulation — IP Method")
    print(f"  K={K}, J={Q_true.shape[0]}, rho=0, {n_rep} replications")
    print("=" * 70)

    all_results = {}

    # ---- IP with multistart=20 ----
    for N in N_values:
        r = run_ip_method(Q_true, N, n_rep=n_rep, multistart=20,
                          seed_base=7649 + N)
        all_results[("IP-20", N)] = r

    # ---- IP with multistart=50 ----
    for N in N_values:
        r = run_ip_method(Q_true, N, n_rep=n_rep, multistart=50,
                          seed_base=8649 + N)
        all_results[("IP-50", N)] = r

    # ---- Print comparison table ----
    print("\n" + "=" * 75)
    print(f"  Table: IP Results — K={K}, J={Q_true.shape[0]}, rho=0, {n_rep} reps")
    print("=" * 75)
    print()
    header = f"{'Method':<12}"
    for N in N_values:
        header += f"  {'mMRR':>7} {'eMRR':>7} {'T(s)':>6}"
    print(header)
    print("-" * 75)

    for method in ["IP-20", "IP-50"]:
        row = f"{method:<12}"
        for N in N_values:
            r = all_results[(method, N)]
            row += f"  {int(r['mMRR']):>3}/{n_rep:<3} {r['eMRR']*100:>6.1f}% {r['mean_time']:>5.1f}"
        print(row)
    print("-" * 75)

    # ---- Save to CSV ----
    with open("simulation_results_IP.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "N", "K", "J", "rho", "mMRR", "eMRR", "mean_time"])
        for (method, N), r in all_results.items():
            writer.writerow([method, N, K, Q_true.shape[0], 0.0,
                             int(r["mMRR"]), f"{r['eMRR']:.6f}", f"{r['mean_time']:.2f}"])
    print("\nResults saved to simulation_results_IP.csv")
