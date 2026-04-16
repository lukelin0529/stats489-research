"""
DINA Q-Matrix Simulation — All Methods on Same Data
Loads pre-generated datasets from generate_datasets.py.

Usage:
    1. python3 generate_datasets.py          (run once)
    2. export PYTHONPATH=~/Desktop/dina-qip/src:$PYTHONPATH
    3. python3 run_all_methods.py             (run this)
"""

import sys, os
import numpy as np
import time
import pickle
import csv
from itertools import permutations
from scipy.special import digamma, betaln, gammaln
from scipy.stats import norm

# Add professor's IP package
sys.path.insert(0, os.path.expanduser("~/Desktop/dina-qip/src"))


# =====================================================================
# SHARED UTILITIES
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
# VB ESTIMATOR (Oka & Okada 2023)
# =====================================================================

def dina_vb(X, Q, max_it=1000, epsilon=1e-5):
    N, J = X.shape
    K = Q.shape[1]
    L = 2 ** K
    A = simA(K)
    natt = Q.sum(axis=1, keepdims=True)
    cc = Q @ A.T
    eta_jl = (cc == natt).astype(np.float64)
    a_s, b_s = 1.5, 2.0
    a_g, b_g = 1.5, 2.0
    delta_0 = np.ones(L)
    r_il = np.full((N, L), 1.0 / L)
    onemX = 1.0 - X
    r_il_eta = r_il @ eta_jl.T
    ELBO = -np.inf
    for m in range(max_it):
        delta_ast = r_il.sum(axis=0) + delta_0
        a_s_ast = (r_il_eta * onemX).sum(axis=0) + a_s
        b_s_ast = (r_il_eta * X).sum(axis=0) + b_s
        a_g_ast = ((1.0 - r_il_eta) * X).sum(axis=0) + a_g
        b_g_ast = ((1.0 - r_il_eta) * onemX).sum(axis=0) + b_g
        E_log_s = digamma(a_s_ast) - digamma(a_s_ast + b_s_ast)
        E_log_1_s = digamma(b_s_ast) - digamma(a_s_ast + b_s_ast)
        E_log_g = digamma(a_g_ast) - digamma(a_g_ast + b_g_ast)
        E_log_1_g = digamma(b_g_ast) - digamma(a_g_ast + b_g_ast)
        E_log_pi = digamma(delta_ast) - digamma(delta_ast.sum())
        lrho = (X * (E_log_1_s - E_log_g)[np.newaxis, :] +
                onemX * (E_log_s - E_log_1_g)[np.newaxis, :]) @ eta_jl + \
               E_log_pi[np.newaxis, :]
        lrho_max = lrho.max(axis=1, keepdims=True)
        tmp = np.exp(lrho - lrho_max)
        r_il = tmp / tmp.sum(axis=1, keepdims=True)
        r_il_eta = r_il @ eta_jl.T
        ll = np.sum(r_il_eta * (X * E_log_1_s[np.newaxis, :] + onemX * E_log_s[np.newaxis, :])) + \
             np.sum((1.0 - r_il_eta) * (X * E_log_g[np.newaxis, :] + onemX * E_log_1_g[np.newaxis, :]))
        kl_pi = np.sum((delta_ast - delta_0) * E_log_pi) - \
                np.sum(gammaln(delta_ast)) + gammaln(delta_ast.sum()) + \
                np.sum(gammaln(delta_0)) - gammaln(delta_0.sum())
        kl_s = np.sum((a_s_ast - a_s) * digamma(a_s_ast) + (b_s_ast - b_s) * digamma(b_s_ast) -
                       (a_s_ast + b_s_ast - a_s - b_s) * digamma(a_s_ast + b_s_ast) -
                       (betaln(a_s_ast, b_s_ast) - betaln(a_s, b_s)))
        kl_g = np.sum((a_g_ast - a_g) * digamma(a_g_ast) + (b_g_ast - b_g) * digamma(b_g_ast) -
                       (a_g_ast + b_g_ast - a_g - b_g) * digamma(a_g_ast + b_g_ast) -
                       (betaln(a_g_ast, b_g_ast) - betaln(a_g, b_g)))
        r_safe = np.maximum(r_il, 1e-300)
        entropy = -np.sum(r_safe * np.log(r_safe))
        class_ll = np.sum(r_il * E_log_pi[np.newaxis, :])
        new_ELBO = ll + class_ll + entropy - kl_pi - kl_s - kl_g
        if abs(new_ELBO - ELBO) < epsilon:
            ELBO = new_ELBO
            break
        ELBO = new_ELBO
    att_pat_est = A[r_il.argmax(axis=1), :]
    return {"s_est": a_s_ast / (a_s_ast + b_s_ast),
            "g_est": a_g_ast / (a_g_ast + b_g_ast),
            "att_pat_est": att_pat_est, "ELBO": ELBO}


def update_Q_item(allQ, g_j, s_j, Y_j, alpha, gamma_j):
    natt = allQ.sum(axis=1, keepdims=True)
    eta_hi = (allQ @ alpha.T == natt).astype(np.float64)
    g_j = np.clip(g_j, 1e-10, 1.0 - 1e-10)
    s_j = np.clip(s_j, 1e-10, 1.0 - 1e-10)
    Y_mat = Y_j[np.newaxis, :]
    ll = np.sum(eta_hi * (Y_mat * np.log(1.0 - s_j) + (1.0 - Y_mat) * np.log(s_j)) +
                (1.0 - eta_hi) * (Y_mat * np.log(g_j) + (1.0 - Y_mat) * np.log(1.0 - g_j)), axis=1)
    gamma_j = np.clip(gamma_j, 1e-10, 1.0 - 1e-10)
    log_prior = allQ @ np.log(gamma_j) + (1.0 - allQ) @ np.log(1.0 - gamma_j)
    pm = ll + log_prior
    pm = np.exp(pm - pm.max())
    pm = pm / pm.sum()
    return allQ[pm.argmax(), :]


def vb_single_run(X, K, niter=550, batchsize=200, seed=1234):
    rng = np.random.RandomState(seed)
    N, J = X.shape
    A = simA(K)
    allQ = A[1:, :]
    H = allQ.shape[0]
    Q = allQ[rng.choice(H, J, replace=True), :]
    Qsamp = np.zeros((J, K, niter))
    ELBOs = np.zeros(niter)
    for t in range(niter):
        idx = rng.choice(N, min(batchsize, N), replace=True)
        Xs = X[idx, :]
        vb = dina_vb(Xs, Q)
        gam = (Q + 1.0) / 3.0
        Q_new = np.zeros((J, K))
        for j in range(J):
            Q_new[j, :] = update_Q_item(allQ, vb["g_est"][j], vb["s_est"][j],
                                         Xs[:, j], vb["att_pat_est"], gam[j, :])
        Q = Q_new
        Qsamp[:, :, t] = Q
        ELBOs[t] = vb["ELBO"]
    return {"ELBO": ELBOs, "Qsample": Qsamp}


def estimate_VB(Y, K, Qtrue, seed=1234):
    from joblib import Parallel, delayed
    rng = np.random.RandomState(seed)
    nrun = 10
    niter = 550
    nburn = 50
    N = Y.shape[0]
    batchsize = 200 if N <= 250 else 300
    run_seeds = rng.choice(range(1000, 10000), nrun, replace=False)
    results = Parallel(n_jobs=7)(
        delayed(vb_single_run)(Y, K, niter, batchsize, int(s)) for s in run_seeds)
    ELBOs = np.column_stack([r["ELBO"] for r in results])
    Qsample = np.stack([r["Qsample"] for r in results], axis=3)
    best_run = np.argmax(ELBOs[nburn:, :].mean(axis=0))
    Qout = Qsample[:, :, nburn:, best_run]
    qest = Qout.mean(axis=2)
    return compute_recovery(Qtrue, qest, K)


# =====================================================================
# IP ESTIMATOR (Professor's method — manual multistart)
# =====================================================================

def estimate_IP(Y, K, Qtrue, multistart=20, seed=1234):
    from dina.config import EMConfig, QMipConfig, QSatConfig
    from dina.em import DINAEM
    J = Qtrue.shape[0]
    em_cfg = EMConfig(multistart=multistart, max_iter=200, tol=1e-6, verbose=False)
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
    q_sat_cfg = QSatConfig.from_qmip(
        q_cfg,
        frac_assumption_sat=0.75,
        frac_sparse_maxsat=0.25,
        min_hamming_frac=0.05,
        assumption_frac=0.25,
        tries_per_start=40,
        solver_name="g4",
    )
    em = DINAEM()
    for attempt in range(3):
        try:
            out = em.fit(Y, K, em_cfg=em_cfg, q_cfg=q_cfg,
                         q_sat_cfg=q_sat_cfg, seed=seed)
            Q_hat = out["Q"].astype(np.float64)
            return compute_recovery(Qtrue, Q_hat, K)
        except Exception:
            time.sleep(5)
    return None


# =====================================================================
# SIMULATION RUNNER
# =====================================================================

def run_method(method_name, estimate_fn, datasets, Qtrue, K, N, n_rep, **kwargs):
    data_list = datasets[f"data_N{N}"]
    recoveries = []
    times_list = []
    print(f"\n  {method_name} — N={N}")
    rng = np.random.RandomState(42 + N)
    seeds = rng.choice(range(1000, 100000), n_rep, replace=False)
    for rep in range(n_rep):
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"    Rep {rep+1}/{n_rep}")
        Y = data_list[rep]
        t0 = time.time()
        rec = estimate_fn(Y, K, Qtrue, seed=int(seeds[rep]), **kwargs)
        elapsed = time.time() - t0
        recoveries.append(rec)
        times_list.append(elapsed)
    eMRR = np.mean(recoveries)
    mMRR = int(np.sum(np.array(recoveries) == 1.0))
    mean_time = np.mean(times_list)
    print(f"    eMRR={eMRR*100:.2f}%  mMRR={mMRR}/{n_rep}  time={mean_time:.1f}s")
    return {"eMRR": eMRR, "mMRR": mMRR, "mean_time": mean_time}


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    # Load pre-generated datasets
    with open("simulation_datasets.pkl", "rb") as f:
        datasets = pickle.load(f)

    Qtrue = datasets["Q_true"]
    K = datasets["K"]
    N_values = [250, 500, 1000]
    n_rep = datasets["n_rep"]  # 100

    # Change to 10 for quick test
    n_rep = 100

    print("=" * 80)
    print("  DINA Q-Matrix Simulation — All Methods on Same Data")
    print(f"  K={K}, J={Qtrue.shape[0]}, rho={datasets['rho']}, {n_rep} replications")
    print(f"  Data: Oka & Okada (2023) DGP with threshold-based attributes")
    print("=" * 80)

    all_results = {}

    # ---- VB ----
    print("\n" + "=" * 40)
    print("  METHOD: VB (Oka & Okada 2023)")
    print("=" * 40)
    for N in N_values:
        all_results[("VB", N)] = run_method("VB", estimate_VB, datasets, Qtrue, K, N, n_rep)

    # ---- IP-20 ----
    print("\n" + "=" * 40)
    print("  METHOD: IP (multistart=20)")
    print("=" * 40)
    for N in N_values:
        all_results[("IP-20", N)] = run_method("IP-20", estimate_IP, datasets, Qtrue, K, N, n_rep, multistart=20)

    # ---- IP-50 ----
    print("\n" + "=" * 40)
    print("  METHOD: IP (multistart=50)")
    print("=" * 40)
    for N in N_values:
        all_results[("IP-50", N)] = run_method("IP-50", estimate_IP, datasets, Qtrue, K, N, n_rep, multistart=50)

    # ---- Print table ----
    print("\n" + "=" * 80)
    print(f"  Table: Q-Matrix Recovery — K={K}, J={Qtrue.shape[0]}, rho=0, {n_rep} reps")
    print("=" * 80)
    print()
    print(f"{'Method':<12}  {'N=250':^20}  {'N=500':^20}  {'N=1000':^20}")
    print(f"{'':12}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>5}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>5}  {'mMRR':>7} {'eMRR':>7} {'T(s)':>5}")
    print("-" * 80)
    for method in ["VB", "IP-20", "IP-50"]:
        row = f"{method:<12}"
        for N in N_values:
            r = all_results[(method, N)]
            row += f"  {r['mMRR']:>3}/{n_rep:<3} {r['eMRR']*100:>6.1f}% {r['mean_time']:>5.1f}"
        print(row)
    print("-" * 80)
    print("\nPaper (VB): 41/100 97.0%  —     80/100 99.2%  —     91/100 99.7%  —")

    # ---- Save ----
    with open("final_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "N", "K", "J", "rho", "mMRR", "eMRR", "mean_time"])
        for (method, N), r in all_results.items():
            writer.writerow([method, N, K, Qtrue.shape[0], 0.0,
                             r["mMRR"], f"{r['eMRR']:.6f}", f"{r['mean_time']:.2f}"])
    print("\nResults saved to final_results.csv")
