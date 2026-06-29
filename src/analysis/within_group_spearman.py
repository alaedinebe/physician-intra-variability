# -*- coding: utf-8 -*-
"""Within-group rank-preservation analysis.

The article reports an overall Spearman rank correlation between each blind
method and the ground truth, computed across the full panel of 20 physicians
(Table 4 of the manuscript). Because the synthetic generator partitions
physicians into 5 well-separated heterogeneity groups (Deterministic,
Near-deterministic, Moderate, Substantial, Coin-flip), this overall rho is
dominated by the *between-group* signal which is easy to recover. The
clinically interesting question is whether methods preserve the ranking
*within* a group of 4 physicians who share the same theoretical discordance
profile.

This module provides:

  - :func:`score2_group_assignment` -- map physician ids to the canonical
    SCORE2 fixture group (1..5) according to ``synthetic_generator``.
  - :func:`compute_within_group_spearman` -- per-(group, method) Spearman rho
    against the ground truth for one experiment.
  - :func:`build_cross_experiment_long_table` -- reshape per-experiment rho
    DataFrames to the long format consumed by the existing experiment-level
    bootstrap.

Statistical caveat (documented honestly): with n=4 physicians per group and
identical theoretical discordance within a group, the rho is computed on a
small, partially Monte-Carlo-driven ranking. Reported values should therefore
be interpreted as *complementary* to the overall rho rather than as a
substitute. Aggregating across many experiments with the experiment-level
bootstrap recovers statistical power.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


__all__ = [
    "MANUAL_COL",
    "MIN_PHYSICIANS_FOR_RHO",
    "SCORE2_GROUP_BOUNDARIES",
    "bootstrap_within_group_rhos",
    "build_cross_experiment_long_table",
    "compute_global_rank_correlations",
    "compute_within_group_spearman",
    "score2_group_assignment",
]


MANUAL_COL = "discordance_rate_manual"
"""Column name of the ground-truth discordance in ``intra_physician_variability``."""

MIN_PHYSICIANS_FOR_RHO = 3
"""Minimum number of non-NaN physicians required to compute a meaningful Spearman rho.

With n < 3 the correlation is either undefined (n=1), or deterministically +-1
(n=2) and conveys no information beyond the signs of the values.
"""

SCORE2_GROUP_BOUNDARIES: tuple[tuple[int, int], ...] = (
    (1, 4),
    (5, 8),
    (9, 12),
    (13, 16),
    (17, 20),
)
"""Inclusive (lo, hi) physician index ranges for the 5 SCORE2 fixture groups.

Mirrors ``SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS`` in
``synthetic_generator.py``: P1..P4 -> group 1, ..., P17..P20 -> group 5.
"""

_PID_RE = re.compile(r"^P(\d+)$")


# -----------------------------------------------------------------------------
# Group assignment
# -----------------------------------------------------------------------------


def score2_group_assignment(physician_ids: Iterable[object]) -> dict[str, int]:
    """Map ``"P<i>"``-style physician ids to their SCORE2 group index (1..5).

    Args:
        physician_ids: iterable of identifiers as produced by the SCORE2
            synthetic generator (``"P1"``, ``"P2"``, ..., ``"P20"``).

    Returns:
        Dictionary ``{physician_id: group_id}`` for every recognised id.
        Identifiers that do not match the ``"P<int>"`` pattern, or whose
        index does not fall in any of the five group ranges, are silently
        omitted (they will be excluded from later rho computations).
    """
    out: dict[str, int] = {}
    for raw in physician_ids:
        if raw is None:
            continue
        text = str(raw).strip()
        match = _PID_RE.match(text)
        if not match:
            continue
        idx = int(match.group(1))
        for group_id, (lo, hi) in enumerate(SCORE2_GROUP_BOUNDARIES, start=1):
            if lo <= idx <= hi:
                out[text] = group_id
                break
    return out


# -----------------------------------------------------------------------------
# Per-experiment Spearman computation
# -----------------------------------------------------------------------------


def compute_within_group_spearman(
    intra_df: pd.DataFrame,
    *,
    group_assignment: dict[str, int],
    method_cols: Optional[Sequence[str]] = None,
    manual_col: str = MANUAL_COL,
    physician_col: str = "physician",
) -> pd.DataFrame:
    """Compute per-(group, method) Spearman rho against the ground truth.

    Args:
        intra_df: per-physician DataFrame as produced by the analysis pipeline.
            It must contain the ``physician_col`` and ``manual_col`` columns,
            plus at least one ``method_col``.
        group_assignment: dictionary mapping physician id to integer group id
            (typically returned by :func:`score2_group_assignment`).
        method_cols: explicit list of method columns to evaluate. When ``None``
            (the default), every column starting with ``discordance_rate_`` is
            used (excluding ``manual_col``), plus the ``overdispersion_local``
            and ``ensemble_matching`` columns when present so that the GLMM and
            the ensemble are also reported.
        manual_col: ground-truth column.
        physician_col: physician identifier column.

    Returns:
        Tidy DataFrame with one row per (group, method) and columns:

            - ``group``: integer group id.
            - ``method``: column name of the evaluated method.
            - ``n_physicians``: number of non-NaN physicians retained.
            - ``spearman_rho``: Spearman rank correlation. ``NaN`` when fewer
              than :data:`MIN_PHYSICIANS_FOR_RHO` physicians remain or when
              either side is constant (``degenerate == True``).
            - ``p_value``: Spearman two-sided p-value (``NaN`` when ``rho`` is
              ``NaN``). Kept under this legacy name for backward compatibility.
            - ``kendall_tau_b``: Kendall τ_b rank correlation
              (``scipy.stats.kendalltau`` with the tie-corrected ``b`` variant).
              Reported alongside Spearman because τ_b is less inflated by
              well-separated structures and has a direct interpretation as the
              probability of concordance minus probability of discordance.
            - ``kendall_p_value``: τ_b two-sided p-value.
            - ``degenerate``: ``True`` when the manual or method values are
              all-equal among the retained physicians (groups 1 and 5 are
              expected to fall here often by construction; reported with a
              caveat in the article).
    """
    if intra_df is None or intra_df.empty:
        return _empty_long_df()
    if manual_col not in intra_df.columns or physician_col not in intra_df.columns:
        return _empty_long_df()

    df = intra_df.copy()
    df["__group__"] = df[physician_col].map(group_assignment)
    df = df.dropna(subset=["__group__"])
    if df.empty:
        return _empty_long_df()
    df["__group__"] = df["__group__"].astype(int)

    columns = list(method_cols) if method_cols is not None else _default_method_cols(df, manual_col)
    if not columns:
        return _empty_long_df()

    rows: list[dict[str, object]] = []
    for group_id, sub in df.groupby("__group__", sort=True):
        manual = pd.to_numeric(sub[manual_col], errors="coerce")
        for method_col in columns:
            if method_col not in sub.columns:
                continue
            method_values = pd.to_numeric(sub[method_col], errors="coerce")
            mask = manual.notna() & method_values.notna()
            n_kept = int(mask.sum())
            if n_kept < MIN_PHYSICIANS_FOR_RHO:
                rows.append(
                    {
                        "group": int(group_id),
                        "method": method_col,
                        "n_physicians": n_kept,
                        "spearman_rho": float("nan"),
                        "p_value": float("nan"),
                        "kendall_tau_b": float("nan"),
                        "kendall_p_value": float("nan"),
                        "degenerate": False,
                    }
                )
                continue
            x = manual[mask].to_numpy()
            y = method_values[mask].to_numpy()
            if _is_constant(x) or _is_constant(y):
                rows.append(
                    {
                        "group": int(group_id),
                        "method": method_col,
                        "n_physicians": n_kept,
                        "spearman_rho": float("nan"),
                        "p_value": float("nan"),
                        "kendall_tau_b": float("nan"),
                        "kendall_p_value": float("nan"),
                        "degenerate": True,
                    }
                )
                continue
            res_spear = spearmanr(x, y)
            res_kend = kendalltau(x, y, variant="b")
            rows.append(
                {
                    "group": int(group_id),
                    "method": method_col,
                    "n_physicians": n_kept,
                    "spearman_rho": _safe_float(res_spear.statistic),
                    "p_value": _safe_float(res_spear.pvalue),
                    "kendall_tau_b": _safe_float(res_kend.statistic),
                    "kendall_p_value": _safe_float(res_kend.pvalue),
                    "degenerate": False,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["group", "method"])
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Cross-experiment reshape
# -----------------------------------------------------------------------------


def build_cross_experiment_long_table(
    per_experiment_rows: Iterable[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Reshape per-experiment rho/tau tables to the long format expected by
    the cross-experiment bootstrap aggregator.

    Args:
        per_experiment_rows: iterable of ``(strategy_name, rho_df)`` pairs,
            where ``rho_df`` is the output of
            :func:`compute_within_group_spearman` for one experiment.

    Returns:
        DataFrame with columns ``strategy_name``, ``method``, ``group``,
        ``spearman_rho``, ``kendall_tau_b``. Rows with NaN values are kept so
        that the bootstrap can drop them per metric.
    """
    blocks: list[pd.DataFrame] = []
    for strategy_name, rho_df in per_experiment_rows:
        if rho_df is None or rho_df.empty:
            continue
        sub = rho_df.copy()
        sub["strategy_name"] = strategy_name
        cols = ["strategy_name", "group", "method", "spearman_rho"]
        if "kendall_tau_b" in sub.columns:
            cols.append("kendall_tau_b")
        blocks.append(sub[cols])
    if not blocks:
        return pd.DataFrame(
            columns=["strategy_name", "group", "method", "spearman_rho", "kendall_tau_b"]
        )
    return pd.concat(blocks, ignore_index=True)


# -----------------------------------------------------------------------------
# Global (all-physicians) rank correlations
# -----------------------------------------------------------------------------


def compute_global_rank_correlations(
    intra_df: pd.DataFrame,
    *,
    method_cols: Optional[Sequence[str]] = None,
    manual_col: str = MANUAL_COL,
    physician_col: str = "physician",
) -> pd.DataFrame:
    """Compute Spearman ρ AND Kendall τ_b on the **full** panel of physicians.

    Unlike :func:`compute_within_group_spearman` which evaluates rank
    preservation within each homogeneity group of size ``n=4``, this function
    evaluates the *global* rank correlation across all physicians (``n=J``).
    It corresponds to the global ρ reported in Table 4 of the article,
    augmented with τ_b per the second reviewer comment.

    The Kendall τ_b column is the "less inflated" reading: well-separated
    structures and outlier rank differences inflate Spearman ρ more than
    they inflate τ_b, so τ_b gives a more conservative summary of how well
    a method tracks the ground-truth ranking.

    Args:
        intra_df: per-physician analysis table (one row per physician).
        method_cols: explicit list of method columns. When ``None``, all
            ``discordance_rate_*`` columns (excluding ``manual_col``), plus
            ``overdispersion_local`` and ``ensemble_matching`` when present.
        manual_col: ground-truth column.
        physician_col: physician identifier column (used to count J).

    Returns:
        Tidy DataFrame with one row per method and columns:

            - ``method``
            - ``n_physicians``: number of non-NaN physicians retained.
            - ``spearman_rho``, ``p_value_spearman``
            - ``kendall_tau_b``, ``p_value_kendall``
            - ``degenerate``: ``True`` when ground truth or method is constant
              across the retained physicians.
    """
    if intra_df is None or intra_df.empty:
        return _empty_global_df()
    if manual_col not in intra_df.columns:
        return _empty_global_df()

    columns = list(method_cols) if method_cols is not None else _default_method_cols(
        intra_df, manual_col
    )
    if not columns:
        return _empty_global_df()

    manual = pd.to_numeric(intra_df[manual_col], errors="coerce")
    rows: list[dict[str, object]] = []
    for method_col in columns:
        if method_col not in intra_df.columns:
            continue
        method_values = pd.to_numeric(intra_df[method_col], errors="coerce")
        mask = manual.notna() & method_values.notna()
        n_kept = int(mask.sum())
        if n_kept < MIN_PHYSICIANS_FOR_RHO:
            rows.append(_empty_row(method_col, n_kept, degenerate=False))
            continue
        x = manual[mask].to_numpy()
        y = method_values[mask].to_numpy()
        if _is_constant(x) or _is_constant(y):
            rows.append(_empty_row(method_col, n_kept, degenerate=True))
            continue
        res_spear = spearmanr(x, y)
        res_kend = kendalltau(x, y, variant="b")
        rows.append(
            {
                "method": method_col,
                "n_physicians": n_kept,
                "spearman_rho": _safe_float(res_spear.statistic),
                "p_value_spearman": _safe_float(res_spear.pvalue),
                "kendall_tau_b": _safe_float(res_kend.statistic),
                "p_value_kendall": _safe_float(res_kend.pvalue),
                "degenerate": False,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _default_method_cols(df: pd.DataFrame, manual_col: str) -> list[str]:
    """Default ``method_cols`` for global correlations (mirrors the within-group helper)."""
    cols = [
        c for c in df.columns
        if c.startswith("discordance_rate_") and c != manual_col
    ]
    for extra in ("overdispersion_local", "ensemble_matching"):
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    return cols


def _empty_row(method_col: str, n_kept: int, *, degenerate: bool) -> dict[str, object]:
    return {
        "method": method_col,
        "n_physicians": n_kept,
        "spearman_rho": float("nan"),
        "p_value_spearman": float("nan"),
        "kendall_tau_b": float("nan"),
        "p_value_kendall": float("nan"),
        "degenerate": degenerate,
    }


def _empty_global_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "method",
            "n_physicians",
            "spearman_rho",
            "p_value_spearman",
            "kendall_tau_b",
            "p_value_kendall",
            "degenerate",
        ]
    )


# -----------------------------------------------------------------------------
# Cross-experiment bootstrap aggregation
# -----------------------------------------------------------------------------


def bootstrap_within_group_rhos(
    long_df: pd.DataFrame,
    *,
    group_col: str = "group",
    method_col: str = "method",
    value_col: str = "spearman_rho",
    experiment_col: str = "strategy_name",
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Bootstrap-aggregate per-experiment within-group rhos to mean +/- CI.

    Each ``(group, method)`` cell is bootstrapped *independently* across
    experiments. We deliberately do not reuse the paired experiment-level
    bootstrap from ``experiment_bootstrap.py`` here: in groups 1 and 5 the
    ground-truth discordance is essentially constant by construction, which
    makes their rho NaN in many experiments, and the paired bootstrap's
    ``dropna(how="any")`` step would discard most experiments and starve the
    other groups.

    Args:
        long_df: long-format DataFrame as produced by
            :func:`build_cross_experiment_long_table`, plus an
            ``experiment_col`` column (typically ``"strategy_name"``).
        group_col: stratification column (the physician group id).
        method_col: column holding the method label.
        value_col: column holding the rho to aggregate. NaN rows are dropped
            before computing the bootstrap statistics for that cell.
        experiment_col: experiment identifier column. Currently unused except
            for input validation, but documented so callers know what shape
            is expected.
        n_resamples: number of bootstrap resamples ``B``.
        seed: deterministic seed for reproducibility.
        confidence: two-sided confidence level (default 0.95).

    Returns:
        Tidy DataFrame with columns ``group``, ``method``, ``n_experiments``,
        ``mean_rho``, ``ci_low``, ``ci_high``. Cells with fewer than two finite
        rhos return ``mean_rho`` equal to the only available value (or NaN)
        and degenerate CI bounds.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}.")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive; got {n_resamples!r}.")
    if long_df is None or long_df.empty:
        return _empty_bootstrap_df()
    required = {group_col, method_col, value_col, experiment_col}
    missing = required - set(long_df.columns)
    if missing:
        raise KeyError(f"long_df is missing columns: {sorted(missing)}.")

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    q_low_p, q_high_p = alpha / 2.0, 1.0 - alpha / 2.0

    rows: list[dict[str, object]] = []
    for (group_id, method_name), sub in long_df.groupby([group_col, method_col], sort=True):
        rhos = pd.to_numeric(sub[value_col], errors="coerce").to_numpy()
        rhos = rhos[np.isfinite(rhos)]
        n = int(rhos.size)
        if n == 0:
            mean_rho = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
        elif n == 1:
            mean_rho = float(rhos[0])
            ci_low = mean_rho
            ci_high = mean_rho
        else:
            indices = rng.integers(low=0, high=n, size=(n_resamples, n))
            resampled = rhos[indices].mean(axis=1)
            mean_rho = float(rhos.mean())
            ci_low = float(np.quantile(resampled, q_low_p))
            ci_high = float(np.quantile(resampled, q_high_p))
        rows.append(
            {
                "group": int(group_id),
                "method": method_name,
                "n_experiments": n,
                "mean_rho": mean_rho,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["group", "method"])
        .reset_index(drop=True)
    )


def _empty_bootstrap_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["group", "method", "n_experiments", "mean_rho", "ci_low", "ci_high"]
    )


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _default_method_cols(df: pd.DataFrame, manual_col: str) -> list[str]:
    cols = [
        c for c in df.columns
        if c.startswith("discordance_rate_") and c != manual_col
    ]
    for extra in ("overdispersion_local", "ensemble_matching"):
        if extra in df.columns and extra not in cols:
            cols.append(extra)
    return cols


def _is_constant(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return True
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        finite = arr[np.isfinite(arr)]
        if finite.size <= 1:
            return True
        return float(np.nanmax(finite) - np.nanmin(finite)) == 0.0
    return float(np.max(arr) - np.min(arr)) == 0.0


def _safe_float(x: object) -> float:
    if x is None:
        return float("nan")
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _empty_long_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "group",
            "method",
            "n_physicians",
            "spearman_rho",
            "p_value",
            "kendall_tau_b",
            "kendall_p_value",
            "degenerate",
        ]
    )
