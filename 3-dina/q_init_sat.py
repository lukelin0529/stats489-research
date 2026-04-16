# ========================= src/qem_dina/q_init_sat.py =========================
from __future__ import annotations
from typing import Optional, Sequence, List, Tuple, Dict, Any, Iterable
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
import copy

def _as_bounds_vec(x: Optional[Sequence[Optional[int]]], L: int, default=None):
    if x is None:
        return [default] * L
    if len(x) != L:
        raise ValueError(f"Expected length {L}, got {len(x)}")
    return list(x)

def _xor_equiv(cnf: CNF, v_y: int, v_a: int, v_b: int):
    # y <-> (a XOR b)
    cnf.extend([
        [-v_y,  v_a,  v_b],
        [-v_y, -v_a, -v_b],
        [ v_y,  v_a, -v_b],
        [ v_y, -v_a,  v_b],
    ])

def _add_lex_leq_columns(
    cnf: CNF, vpool: IDPool, var_of, J: int, c_left: int, c_right: int,
    order: Sequence[int], strict: bool
):
    """
    Enforce column[c_left] <=_lex column[c_right] w.r.t. row order `order`.
    Standard comparator using prefix-equal auxiliaries.
    Bits are x[r, c] ∈ {0,1}. Lex: at first row where they differ, require 0<=1.
    If `strict`, also forbid complete equality: add big OR over XORs.
    """
    t_rows = list(order)
    if len(t_rows) == 0:
        return
    # eq_t: prefix equal up to t-1 (1-based for clarity)
    eq_vars = []
    for t, r in enumerate(t_rows, start=1):
        vL = var_of(r, c_left)
        vR = var_of(r, c_right)
        if t == 1:
            # (¬(1st-diff in wrong dir)): (xL=1 & xR=0) forbidden at first position
            # Encode: NOT( vL & ¬vR )  =>  (¬vL ∨ vR)
            cnf.append([-vL, vR])
            # create eq_1 <-> (xL == xR)
            eq = vpool.id(('eq', c_left, c_right, t))
            # eq -> (vL -> vR) & (vR -> vL)
            cnf.extend([[-eq, -vL, vR], [-eq, -vR, vL]])
            # (vL == vR) -> eq  i.e., (vL∨vR∨-eq) & (-vL∨-vR∨-eq)
            cnf.extend([[vL, vR, -eq], [-vL, -vR, -eq]])
            eq_vars.append(eq)
        else:
            prev = eq_vars[-1]
            # When prefix equal so far (prev==1), apply same constraint here:
            # prev -> (¬(vL & ¬vR))  =>  (¬prev ∨ ¬vL ∨ vR)
            cnf.append([-prev, -vL, vR])
            # define eq_t <-> (prev & (vL == vR))
            eq = vpool.id(('eq', c_left, c_right, t))
            # eq -> prev
            cnf.append([-eq, prev])
            # eq -> (vL==vR)
            cnf.extend([[-eq, -vL, vR], [-eq, -vR, vL]])
            # (prev & (vL==vR)) -> eq
            cnf.extend([[-prev, -vL, vR, -eq], [-prev, -vR, vL, -eq], [prev, -eq]])  # last ensures prev is true when eq is true
            eq_vars.append(eq)

    if strict:
        # Forbid full equality: OR of XORs across rows (at least one row differs)
        y_lits = []
        for r in t_rows:
            y = vpool.id(('lexxor', r, c_left, c_right))
            _xor_equiv(cnf, y, var_of(r, c_left), var_of(r, c_right))
            y_lits.append(y)
        cnf.append(y_lits)

def _apply_hierarchy(cnf: CNF, var_of, J: int, edges: List[Tuple[int, int]], transitive: bool):
    """
    Each edge is (parent, child) meaning parent prerequisite for child:
    enforce q[j,parent] >= q[j,child] for all rows j.
    If transitive: take transitive closure before adding constraints.
    """
    if not edges:
        return
    # Build adjacency and closure if needed
    K = max([max(u, v) for (u, v) in edges] + [0]) + 1
    reach = [[False]*K for _ in range(K)]
    for u, v in edges:
        reach[u][v] = True
    if transitive:
        for k in range(K):
            reach[k][k] = reach[k][k] or False
        # Floyd–Warshall for reachability
        for m in range(K):
            for i in range(K):
                if reach[i][m]:
                    rim = reach[i]
                    rmm = reach[m]
                    for j in range(K):
                        rim[j] = rim[j] or (rmm[j])
    # Add implications parent >= child
    for u in range(K):
        for v in range(K):
            if reach[u][v]:
                # for all rows j: q[j,u] - q[j,v] >= 0   i.e.,  q[j,v] -> q[j,u]
                for j in range(J):
                    cnf.append([-var_of(j, v), var_of(j, u)])

def build_Q_sat_from_cfg(
    J: int,
    K: int,
    *,
    include_identity: bool = False,
    identity_allowed_cols: Optional[Sequence[bool]] = None,
    include_distinctness: bool = False,
    distinctness_use_indicators: bool = False,
    col_lb: Optional[Sequence[Optional[int]]] = None,
    col_up: Optional[Sequence[Optional[int]]] = None,
    row_lb: Optional[Sequence[Optional[int]]] = None,
    row_up: Optional[Sequence[Optional[int]]] = None,
    lexi: bool = False,
    lexi_ascending: bool = False,
    lexi_strict: bool = False,
    lexi_row_order: Optional[Sequence[int]] = None,
    hierarchy_edges: Optional[List[Tuple[int,int]]] = None,
    hierarchy_transitive: bool = False,
    enc_type: EncType = EncType.seqcounter,
) -> tuple[CNF, IDPool, Dict[str, Any]]:
    """
    Build CNF for Q ∈ {0,1}^{J×K} under QMipConfig-like options.
    Identity rows (if included) are physically *pinned at the top*; remaining rows are free.
    Column/row bounds and distinctness apply over the *full* matrix (bounds
    are adjusted by the identity contribution).
    """
    if include_identity:
        if J < 1:
            raise ValueError("J must be >= 1.")
        if identity_allowed_cols is None:
            # full identity over all K columns
            id_cols_mask = [True] * K
        else:
            if len(identity_allowed_cols) != K:
                raise ValueError("identity_allowed_cols must have length K.")
            id_cols_mask = list(map(bool, identity_allowed_cols))
        nI = sum(id_cols_mask)
        if nI > J:
            raise ValueError(f"Not enough rows to host identity rows: need {nI}, have {J}.")
    else:
        id_cols_mask = [False]*K
        nI = 0

    vpool = IDPool()
    cnf = CNF()

    # var(j,k) for all rows/cols
    def var_of(j: int, k: int) -> int:
        return vpool.id(('x', j, k))

    # 0) Pin identity rows (if any)
    # Place them at rows r=0..nI-1, aligning to the True columns in id_cols_mask in order.
    id_cols_list = [k for k, ok in enumerate(id_cols_mask) if ok]
    for i, k in enumerate(id_cols_list):
        r = i  # row index used for identity
        # q[r,k] = 1; q[r,other] = 0
        cnf.append([ var_of(r, k) ])
        for kk in range(K):
            if kk == k:
                continue
            cnf.append([ -var_of(r, kk) ])

    # Helpers: bounds after accounting for identity rows
    col_lb_vec = _as_bounds_vec(col_lb, K, None)
    col_up_vec = _as_bounds_vec(col_up, K, None)
    row_lb_vec = _as_bounds_vec(row_lb, J, None)
    row_up_vec = _as_bounds_vec(row_up, J, None)

    # 1) Row cardinality constraints (for *all* rows). If a row is an identity row,
    # these are already satisfied by pinning; adding at-most/at-least again is fine.
    for r in range(J):
        lits = [var_of(r, c) for c in range(K)]
        lb = row_lb_vec[r]
        ub = row_up_vec[r]
        if lb is not None and lb >= 0:
            cnf.extend(CardEnc.atleast(lits=lits, bound=lb, vpool=vpool, encoding=enc_type).clauses)
        if ub is not None and ub >= 0:
            cnf.extend(CardEnc.atmost(lits=lits, bound=ub, vpool=vpool, encoding=enc_type).clauses)

    # 2) Column cardinalities, adjusted for identity contribution
    # identity provides +1 to column k if k in id_cols_mask (and exactly at the pinned row).
    for c in range(K):
        base = [var_of(r, c) for r in range(J)]
        lb = col_lb_vec[c]
        ub = col_up_vec[c]
        addI = 1 if id_cols_mask[c] else 0
        if lb is not None and lb >= 0:
            # enforce sum >= lb, but we already pinned addI ones; still using the same literals is fine.
            cnf.extend(CardEnc.atleast(lits=base, bound=lb, vpool=vpool, encoding=enc_type).clauses)
        if ub is not None and ub >= 0:
            cnf.extend(CardEnc.atmost(lits=base, bound=ub, vpool=vpool, encoding=enc_type).clauses)

    nI = len(id_cols_list)          # number of identity rows you pinned
    rows_for_distinctness = range(nI, J)   # only rows below the identity block

    # 3) Distinctness across columns *in Q* (ignore identity rows)
    if include_distinctness:
        # If Q* has zero rows, columns cannot be distinct unless K<=1.
        # Adding an empty clause makes CNF UNSAT (PySAT accepts []).
        if J - nI == 0 and K > 1:
            cnf.append([])

        for c1 in range(K):
            for c2 in range(c1 + 1, K):
                y_lits = []
                for r in rows_for_distinctness:
                    y = vpool.id(('y', r, c1, c2))
                    _xor_equiv(cnf, y, var_of(r, c1), var_of(r, c2))
                    y_lits.append(y)
                cnf.append(y_lits)  # at least one *non-identity* row differs


    # 4) Hierarchy (parent -> child means parent≥child)
    if hierarchy_edges:
        _apply_hierarchy(cnf, var_of, J, list(hierarchy_edges), hierarchy_transitive)

    # 5) Lexicographic ordering of columns (break symmetry)
    if lexi:
        order = list(range(J)) if lexi_row_order is None else list(lexi_row_order)
        if any((r < 0 or r >= J) for r in order):
            raise ValueError("lexi_row_order contains out-of-range row indices.")
        # if descending, we can flip the sense by swapping columns
        # Here we implement ascending (col c <=_lex col c+1). For descending, we invert by
        # swapping arguments.
        for c1 in range(K - 1):
            c2 = c1 + 1
            if lexi_ascending:
                _add_lex_leq_columns(cnf, vpool, var_of, J, c1, c2, order, strict=lexi_strict)
            else:
                _add_lex_leq_columns(cnf, vpool, var_of, J, c2, c1, order, strict=lexi_strict)

    meta = {
        'var_of': var_of,
        'shape': (J, K),
        'id_cols_mask': id_cols_mask,
        'n_identity_rows': nI,
    }
    return cnf, vpool, meta

def sample_Q_init_with_sat(
    J: int,
    K: int,
    q_cfg: Any,
    *,
    solver_name: str = 'g4',
    enc_type: EncType = EncType.seqcounter,
    max_solutions: Optional[int] = 1,
    randomize_polarity: bool = True,
) -> np.ndarray:
    """
    Build CNF from q_cfg and return one feasible Q as uint8 array (J,K).
    Raises RuntimeError if unsat.
    """
    # Pull fields from QMipConfig
    cnf, vpool, meta = build_Q_sat_from_cfg(
        J, K,
        include_identity=getattr(q_cfg, 'include_identity', False),
        identity_allowed_cols=getattr(q_cfg, 'identity_allowed_cols', None),
        include_distinctness=getattr(q_cfg, 'include_distinctness', False),
        distinctness_use_indicators=getattr(q_cfg, 'distinctness_use_indicators', False),
        col_lb=getattr(q_cfg, 'col_lb', None),
        col_up=getattr(q_cfg, 'col_up', None),
        row_lb=getattr(q_cfg, 'row_lb', None),
        row_up=getattr(q_cfg, 'row_up', None),
        lexi=getattr(q_cfg, 'lexi', False),
        lexi_ascending=getattr(q_cfg, 'lexi_ascending', False),
        lexi_strict=getattr(q_cfg, 'lexi_strict', False),
        lexi_row_order=getattr(q_cfg, 'lexi_row_order', None),
        hierarchy_edges=getattr(q_cfg, 'hierarchy_edges', None),
        hierarchy_transitive=getattr(q_cfg, 'hierarchy_transitive', True),
        enc_type=enc_type,
    )
    J_, K_ = meta['shape']
    assert (J_, K_) == (J, K)

    # Optional random polarity to diversify initial solutions
    if randomize_polarity:
        import random
        rng = random.Random(0)
        for cl in cnf.clauses:
            rng.shuffle(cl)

    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as S:
        if not S.solve():
            raise RuntimeError("SAT initializer: constraints are UNSAT; cannot build Q_init.")
        model = set(S.get_model())

    # Build Q from positive assignments of x-vars
    Q = np.zeros((J, K), dtype=np.uint8)
    var_of = meta['var_of']
    for r in range(J):
        for c in range(K):
            v = var_of(r, c)
            Q[r, c] = 1 if v in model else 0
    return Q




def enumerate_Q_inits_with_sat(
    J: int,
    K: int,
    q_cfg: Any,
    *,
    n_starts: int,
    solver_name: str = 'g4',
    enc_type: EncType = EncType.seqcounter,
) -> Iterable[np.ndarray]:
    """
    All-SAT style enumerator: yields up to `n_starts` distinct feasible Q's
    under the same constraints implied by `q_cfg`.
    """
    cnf, vpool, meta = build_Q_sat_from_cfg(
        J, K,
        include_identity=getattr(q_cfg, 'include_identity', False),
        identity_allowed_cols=getattr(q_cfg, 'identity_allowed_cols', None),
        include_distinctness=getattr(q_cfg, 'include_distinctness', False),
        distinctness_use_indicators=getattr(q_cfg, 'distinctness_use_indicators', False),
        col_lb=getattr(q_cfg, 'col_lb', None),
        col_up=getattr(q_cfg, 'col_up', None),
        row_lb=getattr(q_cfg, 'row_lb', None),
        row_up=getattr(q_cfg, 'row_up', None),
        lexi=getattr(q_cfg, 'lexi', False),
        lexi_ascending=getattr(q_cfg, 'lexi_ascending', False),
        lexi_strict=getattr(q_cfg, 'lexi_strict', False),
        lexi_row_order=getattr(q_cfg, 'lexi_row_order', None),
        hierarchy_edges=getattr(q_cfg, 'hierarchy_edges', None),
        hierarchy_transitive=getattr(q_cfg, 'hierarchy_transitive', False),
        enc_type=enc_type,
    )
    var_of = meta['var_of']
    x_vars = [vpool.id(('x', r, c)) for r in range(J) for c in range(K)]

    yielded = 0
    with Solver(name=solver_name, bootstrap_with=cnf.clauses) as S:
        while yielded < n_starts and S.solve():
            model = set(S.get_model())
            Q = np.zeros((J, K), dtype=np.uint8)
            for r in range(J):
                for c in range(K):
                    v = var_of(r, c)
                    Q[r, c] = 1 if v in model else 0
            yield Q
            # block this exact assignment of decision vars
            block = [(-v if v in model else v) for v in x_vars]
            S.add_clause(block)
            yielded += 1
            
            
            
            
def sample_random_Q_inits_with_sat(
    J: int,
    K: int,
    q_cfg: Any,
    *,
    n_starts: int,
    pool_max: int = 128,
    solver_name: str = 'g4',
    enc_type: EncType = EncType.seqcounter,
    relax_lexi_for_init: bool = False,
    seed: int = 0,
) -> List[np.ndarray]:
    """
    Randomly sample `n_starts` feasible Q's:
      1) enumerate up to `pool_max` feasible Q's (respecting q_cfg),
      2) uniformly sample without replacement.
    If fewer than `n_starts` exist, returns as many as available (could be 0).
    Optionally relax lexi_strict for initialization diversity only.
    """
    rng = np.random.default_rng(seed)

    q_cfg_eff = q_cfg
    if relax_lexi_for_init and getattr(q_cfg, "lexi", False):
        q_cfg_eff = copy.copy(q_cfg)
        q_cfg_eff.lexi_strict = False

    # 1) build pool
    pool: List[np.ndarray] = []
    try:
        for Q in enumerate_Q_inits_with_sat(
            J, K, q_cfg_eff, n_starts=pool_max,
            solver_name=solver_name, enc_type=enc_type
        ):
            pool.append(Q)
            if len(pool) >= pool_max:
                break
    except Exception:
        # fallback to a single sample
        try:
            pool.append(sample_Q_init_with_sat(
                J, K, q_cfg_eff, solver_name=solver_name, enc_type=enc_type
            ))
        except Exception:
            return []

    if not pool:
        return []

    # 2) random sample without replacement
    idx = rng.permutation(len(pool))[:n_starts]
    return [pool[i] for i in idx]