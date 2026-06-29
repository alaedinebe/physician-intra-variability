# -*- coding: utf-8 -*-
"""Experiment-level paired bootstrap for cross-experiment aggregates.

This module provides a post-hoc analysis layer on top of the per-experiment delta
values produced by ``main._extract_mean_delta_vs_manual``. It estimates the
sampling uncertainty of cross-experiment summaries (Tables 5/6/7 of the article)
by resampling experiments with replacement.

Why "paired" bootstrap?
-----------------------
At each bootstrap iteration, a single set of experiment indices is drawn and used
to compute every method's mean *simultaneously*. This preserves the
between-method correlation (experiments where one method is hard tend to be hard
for all methods), and yields strictly tighter confidence intervals on
inter-method differences than independent resampling would.

Statistical convention
----------------------
Confidence intervals follow the percentile method (Efron & Tibshirani, 1993).
Two-sided p-values for pairwise comparisons are reported as the percentile-based
achieved significance level: ``2 * min(P(d* <= 0), P(d* >= 0))`` over the
bootstrap distribution of the difference.

The module is purely numerical and writes nothing to disk; callers (e.g.
``main._save_cross_experiment_outputs``) are responsible for persistence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


__all__ = [
    "bootstrap_paired_experiment_means",
    "bootstrap_paired_pairwise_differences",
]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _pivot_per_experiment_matrix(
    detailed_df: pd.DataFrame,
    *,
    experiment_col: str,
    method_col: str,
    value_col: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Pivot the long delta table to a (n_experiments x n_methods) numeric matrix.

    Experiments missing at least one method's value are dropped. This keeps the
    paired bootstrap well-defined: every resampled experiment contributes the
    same set of methods.

    Args:
        detailed_df: Long-format DataFrame with one row per (experiment, method).
        experiment_col: Column name uniquely identifying an experiment.
        method_col: Column name holding the method label.
        value_col: Column name holding the numeric statistic to bootstrap.

    Returns:
        Tuple ``(matrix, experiments, methods)`` where ``matrix`` has shape
        ``(n_experiments, n_methods)``, with rows ordered as ``experiments`` and
        columns ordered as ``methods`` (sorted alphabetically for determinism).
    """
    pivoted = detailed_df.pivot_table(
        index=experiment_col,
        columns=method_col,
        values=value_col,
        aggfunc="first",
    )
    pivoted = pivoted.dropna(how="any")
    pivoted = pivoted.sort_index(axis=0).sort_index(axis=1)
    matrix = pivoted.to_numpy(dtype=float)
    return matrix, list(pivoted.index), list(pivoted.columns)


def _draw_paired_bootstrap_indices(
    n_experiments: int,
    *,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Draw a (n_resamples, n_experiments) matrix of experiment indices.

    Each row is one bootstrap resample. The same row is used for every method,
    which is what makes the bootstrap "paired".
    """
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=n_experiments, size=(n_resamples, n_experiments))


def _percentile_ci_bounds(confidence: float) -> tuple[float, float]:
    """Convert a confidence level (e.g. 0.95) to lower/upper quantile probabilities."""
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}.")
    alpha = 1.0 - confidence
    return alpha / 2.0, 1.0 - alpha / 2.0


def _validate_inputs(
    detailed_df: pd.DataFrame,
    *,
    experiment_col: str,
    method_col: str,
    value_col: str,
    n_resamples: int,
) -> None:
    if not isinstance(detailed_df, pd.DataFrame):
        raise TypeError("detailed_df must be a pandas DataFrame.")
    missing = {experiment_col, method_col, value_col} - set(detailed_df.columns)
    if missing:
        raise KeyError(f"detailed_df is missing required columns: {sorted(missing)}.")
    if n_resamples <= 0:
        raise ValueError(f"n_resamples must be positive; got {n_resamples!r}.")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def bootstrap_paired_experiment_means(
    detailed_df: pd.DataFrame,
    *,
    experiment_col: str = "strategy_name",
    method_col: str = "method",
    value_col: str = "mean_delta_vs_manual",
    group_col: Optional[str] = None,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute paired bootstrap CIs for the cross-experiment mean per method.

    Args:
        detailed_df: Long-format DataFrame with one row per (experiment, method),
            as produced by ``main._extract_mean_delta_vs_manual`` (and stored in
            ``mean_delta_vs_manual_per_experiment_and_method.csv``).
        experiment_col: Unique experiment identifier (default ``"strategy_name"``).
        method_col: Method label column (default ``"method"``).
        value_col: Numeric statistic to bootstrap (default
            ``"mean_delta_vs_manual"``).
        group_col: If provided, the bootstrap is run independently within each
            group (e.g. ``"window_size"`` for a Table 6 stratification). The
            output then carries one row per (group, method) with an extra column
            named after ``group_col``.
        n_resamples: Number of bootstrap resamples ``B``. Default 2000 yields
            stable percentile CIs for a 95% level.
        seed: Seed for the underlying ``numpy.random.Generator``. Same seed plus
            same input always yields identical CIs (regression-test guarantee).
        confidence: Two-sided confidence level (default 0.95).

    Returns:
        DataFrame with columns:

            - ``method``
            - ``n_experiments``: number of experiments retained for this method
              (after dropping experiments missing any method's value).
            - ``mean``: empirical mean across experiments (matches the existing
              ``mean_delta_vs_manual_aggregated_by_method.csv`` exactly).
            - ``ci_low``, ``ci_high``: percentile bootstrap CI bounds.
            - ``se``: bootstrap standard error (std of the resampled means).
            - the value of ``group_col`` (if provided).

        The DataFrame is sorted by ``mean`` descending (and, when grouped, by
        ``group_col`` then ``mean`` descending).
    """
    _validate_inputs(
        detailed_df,
        experiment_col=experiment_col,
        method_col=method_col,
        value_col=value_col,
        n_resamples=n_resamples,
    )

    if group_col is None:
        return _bootstrap_means_one_group(
            detailed_df,
            experiment_col=experiment_col,
            method_col=method_col,
            value_col=value_col,
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        )

    if group_col not in detailed_df.columns:
        raise KeyError(f"group_col {group_col!r} not in detailed_df columns.")

    group_results: list[pd.DataFrame] = []
    for group_value, sub_df in detailed_df.groupby(group_col, dropna=False, sort=True):
        if sub_df.empty:
            continue
        sub_result = _bootstrap_means_one_group(
            sub_df,
            experiment_col=experiment_col,
            method_col=method_col,
            value_col=value_col,
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        )
        if sub_result.empty:
            continue
        sub_result.insert(0, group_col, group_value)
        group_results.append(sub_result)

    if not group_results:
        return pd.DataFrame(
            columns=[group_col, "method", "n_experiments", "mean", "ci_low", "ci_high", "se"]
        )
    return pd.concat(group_results, ignore_index=True)


def _bootstrap_means_one_group(
    detailed_df: pd.DataFrame,
    *,
    experiment_col: str,
    method_col: str,
    value_col: str,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    matrix, _experiments, methods = _pivot_per_experiment_matrix(
        detailed_df,
        experiment_col=experiment_col,
        method_col=method_col,
        value_col=value_col,
    )
    n_experiments = matrix.shape[0]
    if n_experiments == 0 or len(methods) == 0:
        return pd.DataFrame(
            columns=["method", "n_experiments", "mean", "ci_low", "ci_high", "se"]
        )

    indices = _draw_paired_bootstrap_indices(
        n_experiments=n_experiments, n_resamples=n_resamples, seed=seed
    )
    resampled = matrix[indices].mean(axis=1)
    q_low, q_high = _percentile_ci_bounds(confidence)
    ci_low = np.quantile(resampled, q_low, axis=0)
    ci_high = np.quantile(resampled, q_high, axis=0)
    se = resampled.std(axis=0, ddof=1) if n_resamples > 1 else np.zeros(len(methods))
    means = matrix.mean(axis=0)

    out = pd.DataFrame(
        {
            "method": methods,
            "n_experiments": n_experiments,
            "mean": means,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "se": se,
        }
    )
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def bootstrap_paired_pairwise_differences(
    detailed_df: pd.DataFrame,
    *,
    experiment_col: str = "strategy_name",
    method_col: str = "method",
    value_col: str = "mean_delta_vs_manual",
    group_col: Optional[str] = None,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired bootstrap CIs and percentile p-values for inter-method differences.

    For every ordered pair of distinct methods ``(a, b)``, computes
    ``delta_observed = mean(a) - mean(b)`` along with a bootstrap distribution of
    that difference. Because the same experiment indices are reused across
    methods at each iteration, the bootstrap respects the within-experiment
    pairing structure and yields tight CIs on the difference even when both
    method-level CIs overlap.

    Args:
        detailed_df: see :func:`bootstrap_paired_experiment_means`.
        experiment_col, method_col, value_col, group_col, n_resamples, seed,
        confidence: see :func:`bootstrap_paired_experiment_means`.

    Returns:
        DataFrame with one row per ordered pair ``(method_a, method_b)``,
        ``method_a != method_b``, and columns:

            - ``method_a``, ``method_b``
            - ``n_experiments``
            - ``delta_observed``: ``mean(a) - mean(b)`` on the original sample.
            - ``ci_low``, ``ci_high``: percentile CI on the bootstrap
              distribution of the difference.
            - ``p_value``: percentile two-sided p-value
              ``2 * min(P(d* <= 0), P(d* >= 0))``.
            - the value of ``group_col`` (if provided).
    """
    _validate_inputs(
        detailed_df,
        experiment_col=experiment_col,
        method_col=method_col,
        value_col=value_col,
        n_resamples=n_resamples,
    )

    if group_col is None:
        return _bootstrap_pairwise_one_group(
            detailed_df,
            experiment_col=experiment_col,
            method_col=method_col,
            value_col=value_col,
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        )

    if group_col not in detailed_df.columns:
        raise KeyError(f"group_col {group_col!r} not in detailed_df columns.")

    group_results: list[pd.DataFrame] = []
    for group_value, sub_df in detailed_df.groupby(group_col, dropna=False, sort=True):
        if sub_df.empty:
            continue
        sub_result = _bootstrap_pairwise_one_group(
            sub_df,
            experiment_col=experiment_col,
            method_col=method_col,
            value_col=value_col,
            n_resamples=n_resamples,
            seed=seed,
            confidence=confidence,
        )
        if sub_result.empty:
            continue
        sub_result.insert(0, group_col, group_value)
        group_results.append(sub_result)

    if not group_results:
        return pd.DataFrame(
            columns=[
                group_col,
                "method_a",
                "method_b",
                "n_experiments",
                "delta_observed",
                "ci_low",
                "ci_high",
                "p_value",
            ]
        )
    return pd.concat(group_results, ignore_index=True)


def _bootstrap_pairwise_one_group(
    detailed_df: pd.DataFrame,
    *,
    experiment_col: str,
    method_col: str,
    value_col: str,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    matrix, _experiments, methods = _pivot_per_experiment_matrix(
        detailed_df,
        experiment_col=experiment_col,
        method_col=method_col,
        value_col=value_col,
    )
    n_experiments, n_methods = matrix.shape
    if n_experiments == 0 or n_methods < 2:
        return pd.DataFrame(
            columns=[
                "method_a",
                "method_b",
                "n_experiments",
                "delta_observed",
                "ci_low",
                "ci_high",
                "p_value",
            ]
        )

    indices = _draw_paired_bootstrap_indices(
        n_experiments=n_experiments, n_resamples=n_resamples, seed=seed
    )
    resampled_means = matrix[indices].mean(axis=1)
    observed_means = matrix.mean(axis=0)
    q_low, q_high = _percentile_ci_bounds(confidence)

    rows: list[dict[str, object]] = []
    for ia, name_a in enumerate(methods):
        for ib, name_b in enumerate(methods):
            if ia == ib:
                continue
            diff_resampled = resampled_means[:, ia] - resampled_means[:, ib]
            delta_obs = float(observed_means[ia] - observed_means[ib])
            ci_low = float(np.quantile(diff_resampled, q_low))
            ci_high = float(np.quantile(diff_resampled, q_high))
            p_two_sided = _percentile_two_sided_p_value(diff_resampled)
            rows.append(
                {
                    "method_a": name_a,
                    "method_b": name_b,
                    "n_experiments": int(n_experiments),
                    "delta_observed": delta_obs,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_two_sided,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["method_a", "method_b"]
    ).reset_index(drop=True)


def _percentile_two_sided_p_value(diff_resampled: np.ndarray) -> float:
    """Percentile-based two-sided p-value for the null H0: difference = 0.

    Defined as ``2 * min(P(d* <= 0), P(d* >= 0))`` evaluated on the bootstrap
    distribution. This is the achieved significance level reported by Efron &
    Tibshirani (1993, ch. 16) under the percentile convention.
    """
    n = diff_resampled.size
    if n == 0:
        return float("nan")
    p_le = float(np.mean(diff_resampled <= 0.0))
    p_ge = float(np.mean(diff_resampled >= 0.0))
    return min(1.0, 2.0 * min(p_le, p_ge))
