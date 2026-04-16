"""
IP-only simulation — run overnight with delays to avoid WLS rate limiting.
VB results already collected separately.

Usage:
    export PYTHONPATH=~/Desktop/dina-qip/src:$PYTHONPATH
    cd ~/Desktop/dina_python
    python3 run_ip_only.py
"""

import sys, os
import numpy as np
import time
import pickle
import csv
from itertools import permutations

sys.path.insert(0, os.path.expanduser("~/Desktop/dina-qip/src"))


# =====================================================================
# UTILITIES
# =====================================================================

def asbinary(m, K):
    bits = np.zeros(K, dtype=np.float64)
    for k in range(K - 1, -1, -1):
        bits[k] = m % 2
        m = m // 2
    return bits

def simA(K):
    L = 2 ** K
    A = np.zeros((L, K), dtype=np.float64)
    for m in range(1, L):
        A[m, :] = asbinary(m, K)
    return A

def delta(D):
    return 1.0 - np.mean(np.abs(D))

def reorder_Q(C, A, K):
    best_val = -np.inf
    best_A = A.copy()
    for p in permutations(range(K)):
        Ap = A[:, list(p)]
        v = np.sum(C * Ap)
        if v > best_val:
            best_val = v
            best_A = Ap.copy()
    return best_A

def compute_recovery(Qtrue, Qest, K):
    C = np.where(Qtrue == 1.0, 0.8, 0.2)
    Qest_reordered = reorder_Q(C, Qest, K)
    Qest_round = np.round(Qest_reordered)
    return delta(Qtrue - Qest_round)


# =====================================================================
# IP ESTIMATOR — manual multistart with generous delays
# =====================================================================

def estimate_IP(Y, K, Qtrue, multistart=20, seed=1234):
    from dina.config import EMConfig, QMipConfig
    from dina.em import DINAEM
    J = Qtrue.shape[0]
    em_cfg = EMConfig(multistart=1, max_iter=200, tol=1e-6, verbose=False)
    q_cfg = QMipConfig(
        include_identity=False,
        include_distinctness=False,
        lexi=False,
        time_limit=60.0,
        mip_gap=1e-4,
        log_to_console=False,
        col_lb=[1] * K,
        row_lb=[1] * J,
    )
    em = DINAEM()
    rng = np.random.RandomState(seed)
    best_ll = -np.inf
    best_Q = None
    for start in range(multistart):
        start_seed = int(rng.randint(1000, 100000))
        for attempt in range(3):
            try:
                out = em.fit(Y, K, em_cfg=em_cfg, q_cfg=q_cfg, seed=start_seed)
                if out["loglik"] > best_ll:
                    best_ll = out["loglik"]
                    best_Q = out["Q"].astype(np.float64)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"      Start {start+1} attempt {attempt+1} failed, retrying in 10s...")
                    time.sleep(10)
                else:
                    print(f"      Start {start+1} failed after 3 attempts: {e}")
    if best_Q is None:
        return None
    return compute_recovery(Qtrue, best_Q, K)


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    with open("simulation_datasets.pkl", "rb") as f:
        datasets = pickle.load(f)

    Qtrue = datasets["Q_true"]
    K = datasets["K"]
    N_values = [250, 500, 1000]
    n_rep = 100

    print("=" * 70)
    print("  IP-Only Simulation (overnight run)")
    print(f"  K={K}, J={Qtrue.shape[0]}, rho=0, {n_rep} reps")
    print(f"  10s delay between reps to avoid WLS rate limiting")
    print("=" * 70)

    all_results = {}

    for ms_label, ms_val in [("IP-20", 20), ("IP-50", 50)]:
        for N in N_values:
            data_list = datasets[f"data_N{N}"]
            recoveries = []
            times_list = []
            failed = 0
            rng = np.random.RandomState(42 + N)
            seeds = rng.choice(range(1000, 100000), n_rep, replace=False)

            print(f"\n{'='*50}")
            print(f"  {ms_label} — N={N}")
            print(f"{'='*50}")

            for rep in range(n_rep):
                if (rep + 1) % 5 == 0 or rep == 0:
                    print(f"  Rep {rep+1}/{n_rep}  "
                          f"(so far: eMRR={np.mean(recoveries)*100:.1f}% "
                          f"mMRR={sum(r==1.0 for r in recoveries)}/{len(recoveries)} "
                          f"failed={failed})" if recoveries else f"  Rep {rep+1}/{n_rep}")

                Y = data_list[rep]
                t0 = time.time()
                rec = estimate_IP(Y, K, Qtrue, multistart=ms_val,
                                   seed=int(seeds[rep]))
                elapsed = time.time() - t0

                if rec is None:
                    failed += 1
                    print(f"    Rep {rep+1} FAILED — sleeping 30s before next rep")
                    time.sleep(30)
                else:
                    recoveries.append(rec)
                    times_list.append(elapsed)
                    # Delay between reps to avoid rate limiting
                    time.sleep(10)

            n_success = len(recoveries)
            if n_success > 0:
                eMRR = np.mean(recoveries)
                mMRR = int(np.sum(np.array(recoveries) == 1.0))
                mean_time = np.mean(times_list)
            else:
                eMRR, mMRR, mean_time = 0, 0, 0

            print(f"\n  RESULT: eMRR={eMRR*100:.2f}%  mMRR={mMRR}/{n_success}  "
                  f"time={mean_time:.1f}s  failed={failed}/{n_rep}")

            all_results[(ms_label, N)] = {
                "eMRR": eMRR, "mMRR": mMRR, "mean_time": mean_time,
                "n_success": n_success, "failed": failed,
                "recoveries": recoveries
            }

    # ---- Print table ----
    print("\n" + "=" * 80)
    print(f"  IP Results — K={K}, J={Qtrue.shape[0]}, rho=0")
    print("=" * 80)
    print()
    print(f"{'Method':<12}  {'N=250':^22}  {'N=500':^22}  {'N=1000':^22}")
    print(f"{'':12}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>5}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>5}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>5}")
    print("-" * 80)
    for method in ["IP-20", "IP-50"]:
        row = f"{method:<12}"
        for N in N_values:
            r = all_results[(method, N)]
            ns = r["n_success"]
            row += f"  {r['mMRR']:>3}/{ns:<3} {r['eMRR']*100:>6.1f}% {r['mean_time']:>5.1f}"
        print(row)
    print("-" * 80)
    print("\nVB results (from earlier run):")
    print("VB          34/100  96.4%  20.7   75/100  99.1% 185.2   96/100  99.8%  26.5")

    # ---- Save ----
    with open("ip_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    with open("ip_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "N", "mMRR", "n_success", "failed", "eMRR", "mean_time"])
        for (method, N), r in all_results.items():
            writer.writerow([method, N, r["mMRR"], r["n_success"], r["failed"],
                             f"{r['eMRR']:.6f}", f"{r['mean_time']:.2f}"])
    print("\nResults saved to ip_results.csv and ip_results.pkl")
