# -*- coding: utf-8 -*-
"""Plotting helpers for Prism analysis visualisations."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _physician_marker_shape(physician_id: object) -> str:
    """
    Marker shape for figure points by synthetic physician id (see synthetic_generator:
    professional_id = f"P{i}"). P1–P4: square; P5–P8: triangle; any other id: circle.
    Matching is case-insensitive on the canonical P + digit form only.
    """
    s = str(physician_id).strip().upper()
    if s in {"P1", "P2", "P3", "P4"}:
        return "s"
    if s in {"P5", "P6", "P7", "P8"}:
        return "^"
    return "o"


def _natural_physician_hue_order(values) -> list:
    """Stable sort: P1, P2, ..., P9, P10, ... then other strings."""
    uniq = list(dict.fromkeys(values))

    def _key(pid: object) -> tuple:
        s = str(pid).strip().upper()
        if len(s) >= 2 and s[0] == "P" and s[1:].isdigit():
            return (0, int(s[1:]))
        return (1, s)

    return sorted(uniq, key=_key)


def _scatter_hue_physician_categorical(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    *,
    order: list[str],
    hue_order: Optional[list] = None,
    palette: str = "husl",
    size: float = 8.0,
    linewidth: float = 1.0,
    jitter: float = 0.0,
    rng_seed: int = 42,
) -> None:
    """
    Categorical-axis scatter, hue-colored by physician, marker by P1–P4 / P5–P8 / other.
    Default jitter=0 keeps points on the category center (vertical alignment); set jitter>0 for horizontal spread.
    """
    if data.empty or hue_col not in data.columns:
        return
    plot_df = data.dropna(subset=[x_col, y_col, hue_col]).copy()
    if plot_df.empty:
        return
    if hue_order is None:
        hue_order = _natural_physician_hue_order(plot_df[hue_col].tolist())
    n_hue = len(hue_order)
    if n_hue == 0:
        return
    colors = sns.color_palette(palette, n_colors=n_hue)
    color_map = {h: colors[i] for i, h in enumerate(hue_order)}
    x_index = {cat: i for i, cat in enumerate(order)}
    rng = np.random.default_rng(rng_seed) if jitter > 0 else None
    point_s = float(size) ** 2 * 1.15
    for _, row in plot_df.iterrows():
        cat = row[x_col]
        if cat in x_index:
            xi = x_index[cat]
        else:
            xi = None
            for i, c in enumerate(order):
                if pd.isna(cat) and pd.isna(c):
                    xi = i
                    break
                if cat == c:
                    xi = i
                    break
                try:
                    if pd.notna(cat) and pd.notna(c) and float(cat) == float(c):
                        xi = i
                        break
                except (TypeError, ValueError):
                    continue
        if xi is None:
            continue
        phy = row[hue_col]
        col = color_map.get(phy)
        if col is None:
            col = sns.color_palette(palette, n_colors=1)[0]
        if jitter <= 0 or rng is None:
            x0 = float(xi)
        else:
            x0 = float(xi) + float(rng.uniform(-jitter, jitter))
        mk = _physician_marker_shape(phy)
        ax.scatter(
            x0,
            row[y_col],
            s=point_s,
            marker=mk,
            facecolors=col,
            edgecolors="black",
            linewidths=linewidth,
            zorder=5,
        )
