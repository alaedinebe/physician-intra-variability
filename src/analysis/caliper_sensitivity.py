# -*- coding: utf-8 -*-
"""Aggregation helpers for the caliper percentile sensitivity analysis.

This module contains *pure* functions that consume an already-computed
``intra_physician_variability`` DataFrame (as produced by the main analysis
pipeline) and summarise per-method bias and unpaired rates. The orchestration
(running the pipeline at multiple percentiles, generating cohorts) lives in
``scripts/caliper_sensitivity.py``.

The split exists so that the aggregation logic is unit-testable in isolation,
without requiring a full pipeline run.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


__all__ = [
    "summarise_caliper_run",
    "MANUAL_COL",
]


MANUAL_COL = "discordance_rate_manual"


def _method_columns(intra_df: pd.DataFrame) -> list[str]:
    """Return ``discordance_rate_<method>`` columns excluding the ground truth."""
    candidates = [
        c for c in intra_df.columns
        if c.startswith("discordance_rate_") and c != MANUAL_COL
    ]
    if "ensemble_matching" in intra_df.columns:
        candidates.append("ensemble_matching")
    return candidates


def summarise_caliper_run(
    intra_df: pd.DataFrame,
    *,
    scenario: str,
    percentile: float,
    caliper_enabled: bool,
) -> pd.DataFrame:
    """Aggregate one pipeline run at a fixed caliper percentile to a tidy table.

    Args:
        intra_df: Per-physician DataFrame as produced by the analysis pipeline,
            i.e. with at least the ``discordance_rate_manual`` column and one
            ``discordance_rate_<method>`` per blind method. When the caliper is
            active, the corresponding ``n_paired_<method>``,
            ``n_unpaired_<method>`` and ``caliper_threshold_<method>`` columns
            are also expected.
        scenario: Free-text label for the experimental scenario (e.g. ``"score2"``).
        percentile: Percentile used for this run (``100.0`` is conventionally
            the no-caliper baseline).
        caliper_enabled: Whether the caliper was actually active during the run.

    Returns:
        DataFrame with one row per blind method and columns:

            - ``scenario``, ``percentile``, ``caliper_enabled``
            - ``method``
            - ``mean_delta_vs_manual``
            - ``median_delta_vs_manual``
            - ``mean_unpaired_rate``: per-physician unpaired fraction averaged
              over physicians (NaN when columns are absent).
            - ``mean_caliper_threshold``: average per-physician threshold
              (NaN when the caliper was disabled).
            - ``n_physicians_kept``: physicians with a finite delta.
            - ``n_physicians_total``: rows in ``intra_df``.
    """
    if intra_df is None or intra_df.empty:
        return _empty_summary_df(scenario, percentile, caliper_enabled)
    if MANUAL_COL not in intra_df.columns:
        return _empty_summary_df(scenario, percentile, caliper_enabled)

    manual = pd.to_numeric(intra_df[MANUAL_COL], errors="coerce")
    rows: list[dict[str, object]] = []
    for method_col in _method_columns(intra_df):
        suffix = method_col.replace("discordance_rate_", "")
        method_values = pd.to_numeric(intra_df[method_col], errors="coerce")
        delta = method_values - manual
        unpaired_rate = _per_physician_unpaired_rate(intra_df, suffix)
        threshold = _per_physician_threshold(intra_df, suffix)
        rows.append(
            {
                "scenario": scenario,
                "percentile": float(percentile),
                "caliper_enabled": bool(caliper_enabled),
                "method": method_col,
                "mean_delta_vs_manual": _safe_float(delta.mean(skipna=True)),
                "median_delta_vs_manual": _safe_float(delta.median(skipna=True)),
                "mean_unpaired_rate": _safe_float(unpaired_rate.mean(skipna=True)),
                "mean_caliper_threshold": _safe_float(threshold.mean(skipna=True)),
                "n_physicians_kept": int(delta.notna().sum()),
                "n_physicians_total": int(len(intra_df)),
            }
        )
    return pd.DataFrame(rows)


def concat_summaries(summaries: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-percentile summaries and sort for stable display."""
    valid = [s for s in summaries if s is not None and not s.empty]
    if not valid:
        return pd.DataFrame()
    out = pd.concat(valid, ignore_index=True)
    out = out.sort_values(["scenario", "method", "percentile"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _per_physician_unpaired_rate(intra_df: pd.DataFrame, suffix: str) -> pd.Series:
    """Compute the per-physician unpaired fraction for one method.

    Returns a Series of NaNs when the bookkeeping columns are absent (e.g. for
    the ensemble row, or for legacy CSVs produced before the caliper feature).
    """
    n_paired_col = f"n_paired_{suffix}"
    n_unpaired_col = f"n_unpaired_{suffix}"
    if n_paired_col not in intra_df.columns or n_unpaired_col not in intra_df.columns:
        return pd.Series([np.nan] * len(intra_df), index=intra_df.index, dtype=float)
    n_paired = pd.to_numeric(intra_df[n_paired_col], errors="coerce")
    n_unpaired = pd.to_numeric(intra_df[n_unpaired_col], errors="coerce")
    total = n_paired + n_unpaired
    return n_unpaired.where(total > 0).divide(total.where(total > 0))


def _per_physician_threshold(intra_df: pd.DataFrame, suffix: str) -> pd.Series:
    col = f"caliper_threshold_{suffix}"
    if col not in intra_df.columns:
        return pd.Series([np.nan] * len(intra_df), index=intra_df.index, dtype=float)
    return pd.to_numeric(intra_df[col], errors="coerce")


def _safe_float(x: object) -> float:
    """Coerce a scalar to float, mapping pandas NA / NaN to ``float('nan')``."""
    if x is None:
        return float("nan")
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _empty_summary_df(scenario: str, percentile: float, caliper_enabled: bool) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scenario",
            "percentile",
            "caliper_enabled",
            "method",
            "mean_delta_vs_manual",
            "median_delta_vs_manual",
            "mean_unpaired_rate",
            "mean_caliper_threshold",
            "n_physicians_kept",
            "n_physicians_total",
        ]
    )
