# -*- coding: utf-8 -*-
"""Core matching/statistical helpers for Prism analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# Patsy parses long additive formulas as a left-deep tree; beyond this threshold
# formula parsing can hit Python's recursion limit.
GLMM_MAX_FORMULA_FIXED_EFFECTS = 100


def _fit_binomial_bayes_mixed_glmm(
    df: pd.DataFrame,
    outcome_col: str,
    physician_col: str,
    fixed_effects: list[str],
) -> tuple[BinomialBayesMixedGLM, object]:
    """Fit a binomial Bayes mixed GLM with physician random intercepts."""
    if len(fixed_effects) <= GLMM_MAX_FORMULA_FIXED_EFFECTS:
        formula = f"{outcome_col} ~ {' + '.join(fixed_effects)}"
        vc_formulas = {"physician_re": f"0 + C({physician_col})"}
        model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas=vc_formulas, data=df)
        return model, model.fit_vb()

    endog = df[outcome_col].to_numpy(dtype=int)
    exog = np.column_stack([
        np.ones(len(df), dtype=float),
        df[fixed_effects].to_numpy(dtype=float),
    ])
    fep_names = ["Intercept"] + [str(c) for c in fixed_effects]
    phys_dummies = pd.get_dummies(df[physician_col].astype(str), drop_first=False)
    exog_vc = phys_dummies.to_numpy(dtype=float)
    ident = np.zeros(exog_vc.shape[1], dtype=int)
    vc_names = [str(c) for c in phys_dummies.columns]
    model = BinomialBayesMixedGLM(
        endog,
        exog,
        exog_vc,
        ident,
        fep_names=fep_names,
        vc_names=vc_names,
    )
    return model, model.fit_vb()


def _glmm_fixed_effect_weights(result) -> dict[str, dict[str, float | None]]:
    """
    Posterior mean (log-odds weight) and posterior SD for each GLMM fixed effect,
    from the variational Bayes summary (Type M rows only; excludes variance parameters).
    """
    try:
        tables = result.summary().tables
        if not tables:
            return {}
        summary_df = tables[0]
        if not isinstance(summary_df, pd.DataFrame) or "Type" not in summary_df.columns:
            return {}
        fe = summary_df[summary_df["Type"] == "M"]
        if fe.empty:
            return {}
        mean_col = "Post. Mean" if "Post. Mean" in fe.columns else None
        sd_col = "Post. SD" if "Post. SD" in fe.columns else None
        if mean_col is None:
            mean_col = next((c for c in fe.columns if "Mean" in str(c)), None)
        out: dict[str, dict[str, float | None]] = {}
        for name in fe.index:
            entry: dict[str, float | None] = {}
            if mean_col and mean_col in fe.columns:
                v = fe.loc[name, mean_col]
                entry["post_mean"] = None if pd.isna(v) else float(v)
            if sd_col and sd_col in fe.columns:
                v = fe.loc[name, sd_col]
                entry["post_sd"] = None if pd.isna(v) else float(v)
            out[str(name)] = entry
        return out
    except Exception:
        return {}


def genetic_objective_function(weights, X, Y):
    """
    Objective function for genetic matching: minimise covariate imbalance after matching.

    Used by differential_evolution to find variable weights such that treated (Y=1)
    and control (Y=0) patients, when matched by weighted Euclidean distance, have
    similar covariate distributions. Balance is measured by the maximum absolute
    standardised mean difference (SMD) across covariates; the optimizer minimises
    this worst-case imbalance.

    Arguments:
        weights: 1D array of variable weights (one per column of X). Will be
            normalised to sum to 1 and applied as sqrt(weights) to scale columns.
        X: 2D array of covariates (n_samples, n_features). Should be standardised
            (e.g. z-score) before calling.
        Y: 1D array of binary outcome (0/1). Treated = 1, control = 0.

    Returns:
        float: Maximum absolute SMD across covariates after matching (lower is better).
        Returns 1e9 if either group is empty (infeasible).
    """
    # Guard: empty or non-finite inputs make matching undefined; return worst penalty.
    if X.size == 0 or Y.size == 0 or X.shape[1] == 0:
        return 1e9
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        return 1e9

    # Step 1: Normalise weights to a simplex (sum=1) and avoid zeros so that
    # no dimension is collapsed; use abs so the optimizer can use negative bounds.
    weights = np.asarray(weights, dtype=float).ravel()
    if len(weights) != X.shape[1]:
        return 1e9
    weights = np.abs(weights) + 1e-9
    weights = weights / np.sum(weights)

    # Step 2: Build weighted covariate space for distance: each column scaled by
    # sqrt(weight). Euclidean distance in this space is weighted Euclidean in original.
    X_weighted = X * np.sqrt(weights)

    # Step 3: Split indices by treatment status for treated-control matching.
    treated_idx = np.where(Y == 1)[0]
    control_idx = np.where(Y == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        return 1e9

    X_treated_weighted = X_weighted[treated_idx]
    X_control_weighted = X_weighted[control_idx]

    # Step 4: Greedy bipartite matching — for each treated unit, find the control
    # with smallest weighted Euclidean distance. This defines "matched" pairs for
    # balance evaluation.
    dists = cdist(X_treated_weighted, X_control_weighted, metric='euclidean')
    nearest_control_indices = np.argmin(dists, axis=1)

    # Step 5: Covariate balance on ORIGINAL (unweighted) X: compare mean of
    # treated vs mean of their matched controls. SMD = |mean_treated - mean_matched_control|
    # per variable; we return the max SMD so the optimizer minimises worst imbalance.
    X_treated_original = X[treated_idx]
    X_matched_controls_original = X[control_idx[nearest_control_indices]]

    smd = np.abs(np.mean(X_treated_original, axis=0) - np.mean(X_matched_controls_original, axis=0))
    return np.max(smd)


def _greedy_nearest_neighbor_pairs(
    local_dists: np.ndarray,
    *,
    caliper_threshold: float | None = None,
) -> list[tuple[int, int]]:
    """
    Greedy one-to-one nearest-neighbour pairing (without replacement).

    Arguments:
        local_dists: square distance matrix (diagonal should be inf).
        caliper_threshold: optional maximum allowed distance for a pair. When
            provided, candidate pairs whose distance strictly exceeds the
            threshold are refused, leaving the corresponding patient unpaired
            (Rosenbaum & Rubin 1985 caliper convention). When ``None`` (the
            default), the function behaves exactly as without a caliper, which
            preserves bit-for-bit backward compatibility.

    Returns:
        List of index pairs (i, j) with i != j and each index used at most once.
        For odd group sizes, or when the caliper is binding, some indices may
        remain unpaired.
    """
    d = np.asarray(local_dists, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        return []

    n = d.shape[0]
    used = np.zeros(n, dtype=bool)
    pairs: list[tuple[int, int]] = []

    for i in range(n):
        if used[i]:
            continue
        candidates = np.where(~used)[0]
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            continue
        cand_d = d[i, candidates]
        if not np.any(np.isfinite(cand_d)):
            continue
        argmin = int(np.argmin(cand_d))
        best_distance = float(cand_d[argmin])
        if caliper_threshold is not None and best_distance > caliper_threshold:
            continue
        j = int(candidates[argmin])
        used[i] = True
        used[j] = True
        pairs.append((int(i), j))
    return pairs


def _discordance_rate_from_pairs(y_local: np.ndarray, pairs: list[tuple[int, int]]) -> float:
    """Discordance rate = discordant pairs / total pairs; NaN when no pair."""
    if len(pairs) == 0:
        return np.nan
    y = np.asarray(y_local)
    disc = [float(y[i] != y[j]) for i, j in pairs]
    return float(np.mean(disc))


def _optimal_one_to_one_pairs(
    local_dists: np.ndarray,
    *,
    caliper_threshold: float | None = None,
) -> list[tuple[int, int]]:
    """
    Approximate minimum-weight 1-to-1 pairing without replacement.

    Solves the Hungarian assignment problem on the symmetric distance matrix
    (self-pairing disabled), then deduplicates undirected edges and selects
    unique nodes greedily by lowest weight. For odd group sizes, one node
    remains unpaired.

    Compared to greedy nearest-neighbour pairing, this jointly considers all
    assignment decisions and typically yields lower total intra-pair distance.

    Arguments:
        local_dists: square distance matrix.
        caliper_threshold: optional maximum allowed distance for a pair. When
            provided, candidate edges with distance strictly exceeding the
            threshold are filtered out before final selection (Rosenbaum &
            Rubin 1985 caliper convention). When ``None`` (the default), the
            function behaves exactly as without a caliper, preserving
            bit-for-bit backward compatibility.
    """
    d = np.asarray(local_dists, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        return []

    work = d.copy()
    np.fill_diagonal(work, np.inf)
    if not np.all(np.isfinite(work)):
        work = np.where(np.isfinite(work), work, 1e12)

    try:
        row_ind, col_ind = linear_sum_assignment(work)
    except ValueError:
        return []

    candidates: list[tuple[float, tuple[int, int]]] = []
    for r, c in zip(row_ind, col_ind):
        if r == c:
            continue
        w = float(work[int(r), int(c)])
        if not np.isfinite(w) or w >= 1e12:
            continue
        if caliper_threshold is not None and w > caliper_threshold:
            continue
        i, j = (int(r), int(c)) if r < c else (int(c), int(r))
        candidates.append((w, (i, j)))

    candidates.sort(key=lambda item: item[0])
    used: set[int] = set()
    seen_edges: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for _w, (i, j) in candidates:
        if (i, j) in seen_edges:
            continue
        seen_edges.add((i, j))
        if i in used or j in used:
            continue
        pairs.append((i, j))
        used.add(i)
        used.add(j)
    return pairs


def _per_physician_caliper_threshold(
    local_dists: np.ndarray,
    percentile: float,
) -> float:
    """Compute the caliper threshold for a single physician's distance matrix.

    The threshold is defined as the q-th percentile of the off-diagonal pairwise
    distances within the panel (the upper-triangular entries above the main
    diagonal). The diagonal itself is excluded because it is conventionally set
    to ``+inf`` to disable self-pairing, and would otherwise contaminate the
    quantile.

    Arguments:
        local_dists: square distance matrix with non-finite diagonal.
        percentile: a value in ``[0, 100]``. ``100`` is equivalent to no
            caliper (returns ``+inf``); ``0`` returns the minimum off-diagonal
            distance.

    Returns:
        The percentile-based threshold as a float. Returns ``+inf`` when the
        matrix is too small to define a pair, when no finite off-diagonal
        distance exists, or when ``percentile == 100``.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(
            f"caliper percentile must be in [0, 100]; got {percentile!r}."
        )
    if percentile == 100.0:
        return float("inf")
    d = np.asarray(local_dists, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        return float("inf")
    n = d.shape[0]
    iu = np.triu_indices(n, k=1)
    upper = d[iu]
    finite = upper[np.isfinite(upper)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, percentile))
