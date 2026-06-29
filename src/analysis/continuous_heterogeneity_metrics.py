# -*- coding: utf-8 -*-
"""Discrimination metrics for the continuous-heterogeneity SCORE2 fixture.

When physicians are drawn on a continuous spectrum of ``p_high`` (rather than
clustered into 5 fixed groups), the relevant question is no longer whether the
methods preserve the *between-group* ranking, but whether they *discriminate
at the margin*: can a method correctly rank two physicians whose true expected
discordance differs by, say, 0.02?

This module produces the artefacts that answer that question:

  - per-method Spearman rho between method scores and the **true** expected
    discordance ``D* = 2 * p_high * (1 - p_high)`` (the design parameter, not
    the empirical Manual Pairing rate),
  - the same against the empirical Manual Pairing rate, for completeness,
  - a tidy table joining true ``p_high``, true ``D*``, empirical Manual rate
    and every method's estimate (one row per physician),
  - a faceted scatter plot of (true ``D*``, method estimate), with the y=x
    reference line, used to inspect discrimination visually.

The functions are pure and easy to unit-test in isolation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


__all__ = [
    "build_per_physician_table",
    "compute_continuous_heterogeneity_metrics",
    "save_scatter_grid",
]


MANUAL_COL = "discordance_rate_manual"


def build_per_physician_table(
    intra_df: pd.DataFrame,
    physician_params: Mapping[str, Mapping[str, float]],
    *,
    physician_col: str = "physician",
) -> pd.DataFrame:
    """Join the per-physician ``intra_physician_variability`` rows with the
    *true* design parameters of the continuous-heterogeneity fixture.

    Args:
        intra_df: DataFrame produced by the analysis pipeline (one row per
            physician, with ``discordance_rate_*`` columns and the GLMM
            ``overdispersion_local``).
        physician_params: mapping ``physician_id -> {"p_high", "p_low",
            "expected_discordance"}`` returned by
            :func:`generate_score2_continuous_heterogeneity`.
        physician_col: physician identifier column.

    Returns:
        A DataFrame combining the original ``intra_df`` columns and three
        added columns: ``p_high``, ``p_low``, ``expected_discordance``.
        Physicians missing from ``physician_params`` (rare; e.g. real-data
        cohort accidentally routed here) get ``NaN`` parameters.
    """
    if intra_df is None or intra_df.empty:
        return _empty_per_physician_df()
    if physician_col not in intra_df.columns:
        return _empty_per_physician_df()

    augmented = intra_df.copy()
    augmented["p_high"] = augmented[physician_col].map(
        lambda pid: physician_params.get(pid, {}).get("p_high", float("nan"))
    )
    augmented["p_low"] = augmented[physician_col].map(
        lambda pid: physician_params.get(pid, {}).get("p_low", float("nan"))
    )
    augmented["expected_discordance"] = augmented[physician_col].map(
        lambda pid: physician_params.get(pid, {}).get(
            "expected_discordance", float("nan")
        )
    )
    return augmented.reset_index(drop=True)


def compute_continuous_heterogeneity_metrics(
    intra_df: pd.DataFrame,
    physician_params: Mapping[str, Mapping[str, float]],
    *,
    method_cols: Optional[Sequence[str]] = None,
    physician_col: str = "physician",
    manual_col: str = MANUAL_COL,
) -> pd.DataFrame:
    """Per-method Spearman rho against the **true** expected discordance.

    Two reference axes are reported per method to disentangle "discrimination
    against the design" from "agreement with the empirical Manual rate":

      - ``rho_vs_true``: Spearman rho between the method scores and the true
        ``D* = 2 * p_high * (1 - p_high)`` (the design quantity).
      - ``rho_vs_manual``: Spearman rho between the method scores and the
        empirical Manual Pairing rate (the existing Table 4 quantity, kept
        here for direct comparison).

    Per the second reviewer comment ("replace the global rho with Kendall
    tau_b ... to give a less inflated reading"), Kendall τ_b is also
    reported alongside Spearman rho for the comparison vs. the *true* D*
    (where the ground truth is unambiguous). It is intentionally NOT reported
    against the empirical Manual rate to keep the table compact and aligned
    with the reviewer's request to apply tau_b to the ground-truth axis only.

    Args:
        intra_df: per-physician analysis table.
        physician_params: per-physician design parameters (see
            :func:`build_per_physician_table`).
        method_cols: explicit list of method columns to evaluate. When
            ``None``, every column starting with ``discordance_rate_`` is
            used (excluding ``manual_col``), plus ``overdispersion_local``
            and ``ensemble_matching`` when present.
        physician_col, manual_col: column-name overrides.

    Returns:
        Tidy DataFrame with one row per method and columns:

            - ``method``
            - ``n_physicians`` (after dropping NaNs in either side)
            - ``rho_vs_true``, ``p_value_vs_true``
            - ``tau_b_vs_true``, ``kendall_p_value_vs_true``
            - ``rho_vs_manual``, ``p_value_vs_manual``

        Returns an empty DataFrame (with the right columns) when the inputs
        are insufficient.
    """
    augmented = build_per_physician_table(
        intra_df, physician_params, physician_col=physician_col
    )
    if augmented.empty:
        return _empty_metrics_df()

    candidates = list(method_cols) if method_cols is not None else _default_method_cols(
        augmented, manual_col
    )
    if not candidates:
        return _empty_metrics_df()

    rows: list[dict[str, object]] = []
    for method_col in candidates:
        if method_col not in augmented.columns:
            continue
        method_values = pd.to_numeric(augmented[method_col], errors="coerce")
        true_d = pd.to_numeric(augmented["expected_discordance"], errors="coerce")
        manual = pd.to_numeric(augmented[manual_col], errors="coerce") if manual_col in augmented.columns else pd.Series([], dtype=float)

        rho_true, p_true, n_true = _safe_spearman(true_d, method_values)
        tau_true, p_tau_true, _ = _safe_kendall(true_d, method_values)
        if not manual.empty:
            rho_manual, p_manual, _ = _safe_spearman(manual, method_values)
        else:
            rho_manual, p_manual = float("nan"), float("nan")

        rows.append(
            {
                "method": method_col,
                "n_physicians": n_true,
                "rho_vs_true": rho_true,
                "p_value_vs_true": p_true,
                "tau_b_vs_true": tau_true,
                "kendall_p_value_vs_true": p_tau_true,
                "rho_vs_manual": rho_manual,
                "p_value_vs_manual": p_manual,
            }
        )

    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def save_scatter_grid(
    augmented_df: pd.DataFrame,
    *,
    method_cols: Sequence[str],
    method_display_names: Mapping[str, str],
    output_path: Path,
    n_columns: int = 3,
    title: str = "Discrimination at the margin: method estimate vs true D*",
) -> None:
    """Render a faceted scatter plot of (true D*, method estimate) per method.

    Each panel:
        - x-axis: true expected discordance ``D* = 2 * p_high * (1 - p_high)``.
        - y-axis: method's per-physician score.
        - one dot per physician, with a y=x reference line for perfect
          discrimination, plus the Spearman rho printed in the corner.

    The figure is written to ``output_path``. The function does no validation;
    callers should ensure ``augmented_df`` contains the needed columns.
    """
    methods_present = [m for m in method_cols if m in augmented_df.columns]
    if not methods_present:
        return
    n = len(methods_present)
    n_cols = max(1, min(n_columns, n))
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), squeeze=False
    )
    flat_axes = axes.flatten()
    x = pd.to_numeric(augmented_df["expected_discordance"], errors="coerce")
    for i, method in enumerate(methods_present):
        ax = flat_axes[i]
        y = pd.to_numeric(augmented_df[method], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.any():
            ax.scatter(x[mask], y[mask], s=22, alpha=0.7, edgecolor="black", linewidth=0.4)
            lo = float(min(x[mask].min(), y[mask].min()))
            hi = float(max(x[mask].max(), y[mask].max()))
            pad = 0.03 * max(1e-6, hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", linestyle="--", linewidth=0.8)
            rho, _, _ = _safe_spearman(x[mask], y[mask])
            if not np.isnan(rho):
                ax.text(
                    0.04, 0.95, f"ρ = {rho:.3f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=10, fontweight="bold",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "lightgrey"},
                )
        ax.set_title(method_display_names.get(method, method), fontsize=10)
        ax.set_xlabel("True D* = 2·p_high·(1-p_high)", fontsize=9)
        ax.set_ylabel("Method estimate", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def _safe_spearman(
    x: "pd.Series[float]", y: "pd.Series[float]",
) -> tuple[float, float, int]:
    return _safe_rank_correlation(x, y, kind="spearman")


def _safe_kendall(
    x: "pd.Series[float]", y: "pd.Series[float]",
) -> tuple[float, float, int]:
    return _safe_rank_correlation(x, y, kind="kendall")


def _safe_rank_correlation(
    x: "pd.Series[float]",
    y: "pd.Series[float]",
    *,
    kind: str,
) -> tuple[float, float, int]:
    """Compute Spearman rho or Kendall tau_b on aligned, non-NaN pairs.

    Returns ``(stat, p_value, n_kept)``. Returns NaN statistics when fewer
    than 3 valid pairs remain or when either side is constant.
    """
    x_arr = pd.to_numeric(x, errors="coerce")
    y_arr = pd.to_numeric(y, errors="coerce")
    mask = x_arr.notna() & y_arr.notna()
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    xv = x_arr[mask].to_numpy()
    yv = y_arr[mask].to_numpy()
    if _is_constant(xv) or _is_constant(yv):
        return float("nan"), float("nan"), n
    if kind == "spearman":
        res = spearmanr(xv, yv)
    elif kind == "kendall":
        res = kendalltau(xv, yv, variant="b")
    else:
        raise ValueError(f"Unknown rank correlation kind: {kind!r}.")
    return float(res.statistic), float(res.pvalue), n


def _is_constant(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return True
    return float(np.max(arr) - np.min(arr)) == 0.0


def _empty_per_physician_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["physician", "p_high", "p_low", "expected_discordance"]
    )


def _empty_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "method",
            "n_physicians",
            "rho_vs_true",
            "p_value_vs_true",
            "tau_b_vs_true",
            "kendall_p_value_vs_true",
            "rho_vs_manual",
            "p_value_vs_manual",
        ]
    )
