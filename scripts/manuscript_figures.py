"""Figures of the revised manuscript, built from the benchmark outputs.

    python scripts/manuscript_figures.py           # writes results/manuscript/figures/

Each figure is written as PNG (300 dpi) and PDF. Method colours are fixed
across figures (one hue per method, in the same order everywhere) and every
method also has its own marker, so that figures remain readable in grayscale
and for colour-vision deficiencies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAIN = ROOT / "results" / "revision_main"
GRIDS = ROOT / "results" / "revision_grids" / "tables"
OUT = ROOT / "results" / "manuscript" / "figures"

METHODS = ["euclidean", "mahalanobis", "lpa_guided", "genetic_mahalanobis",
           "learned_weights", "mutual_information", "rf_proximity", "glmm"]
LABEL = {"euclidean": "Euclidean", "mahalanobis": "Mahalanobis", "lpa_guided": "LPA-guided",
         "genetic_mahalanobis": "Genetic Mahalanobis", "learned_weights": "Learned Weights",
         "mutual_information": "Mutual Information", "rf_proximity": "RF Proximity", "glmm": "GLMM"}
# Validated categorical palette (8 slots, fixed order) and one marker per method.
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
MARKER = dict(zip(METHODS, ["o", "s", "^", "D", "v", "P", "X", "*"]))
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#3b3a37"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written", name)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=INK)


def load_long(cols: list[str]) -> pd.DataFrame:
    d = pd.read_csv(MAIN / "benchmark_long.csv.gz", usecols=cols + ["caliper_percentile"], low_memory=False)
    keep = d["caliper_percentile"].isna() | np.isclose(d["caliper_percentile"].astype(float), 25.0)
    return d[keep]


# ----------------------------------------------------------------------------- Figure 2

def figure2(long: pd.DataFrame) -> None:
    """SCORE2 five-group experiment."""
    s = long[(long.family == "score2") & (long.design == "in_sample")]
    ref = s.drop_duplicates("physician").sort_values("physician_group")
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM), gridspec_kw={"width_ratios": [1.15, 1.25, 0.9]})
    groups = np.arange(1, 6)
    d_star = ref.groupby("physician_group")["D_star"].first().to_numpy()
    d_tilde = ref.groupby("physician_group")["D_tilde"].mean().to_numpy()

    ax = axes[0]
    ax.plot(groups, d_star, color=REF, lw=1.2, ls="--", zorder=5, label="Theoretical D⋆")
    ax.plot(groups, d_tilde, color=REF, lw=0, marker="o", mfc="white", mec=REF, ms=5, zorder=6,
            label="Manual-pairing reference")
    for m in METHODS[:-1]:
        g = s[s.method == m].groupby("physician_group")["score"].mean()
        ax.plot(groups, g.to_numpy(), color=COLOR[m], lw=1.2, marker=MARKER[m], ms=3.8, label=LABEL[m])
    ax.set_xticks(groups)
    ax.set_xticklabels([f"{k}\n({p:.2f})" for k, p in zip(groups, [1.0, 0.9, 0.8, 0.7, 0.5])])
    ax.set_xlabel("Physician group (p_high)")
    ax.set_ylabel("Mean discordance")
    ax.set_ylim(-0.02, 0.58)
    ax.set_title("Estimated discordance by group", loc="left", color=INK)
    panel_letter(ax, "A")

    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, m in enumerate(METHODS[:-1]):
        g = s[s.method == m]
        e = (g["score"] - g["D_tilde"]).to_numpy()
        x = i + rng.uniform(-0.22, 0.22, len(e))
        ax.scatter(x, e, s=9, color=COLOR[m], marker=MARKER[m], lw=0, alpha=0.85)
        ax.plot([i - 0.32, i + 0.32], [e.mean()] * 2, color=INK, lw=1.4)
    ax.axhline(0, color=REF, lw=0.9, ls="--")
    ax.set_xticks(range(7))
    ax.set_xticklabels([LABEL[m] for m in METHODS[:-1]], rotation=40, ha="right")
    ax.set_ylabel("Error per physician")
    ax.set_ylim(-0.1, 0.33)
    ax.grid(axis="x", visible=False)
    ax.set_title("Physician-level error (bar: mean)", loc="left", color=INK)
    panel_letter(ax, "B")

    ax = axes[2]
    g = s[s.method == "glmm"].merge(ref[["physician", "physician_group"]], on="physician", suffixes=("", "_r"))
    for k in groups:
        v = g[g.physician_group == k]["score"].to_numpy()
        ax.scatter(np.full(len(v), k) + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=COLOR["glmm"],
                   marker=MARKER["glmm"], lw=0)
    ax.set_xticks(groups)
    ax.set_xlabel("Physician group")
    ax.set_ylabel("GLMM score")
    ax.set_title("GLMM score by group", loc="left", color=INK)
    panel_letter(ax, "C")

    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17), handlelength=2.2)
    fig.tight_layout(w_pad=1.6)
    save(fig, "Figure_2_score2")


# ----------------------------------------------------------------------------- Figure 3

def figure3(long: pd.DataFrame) -> None:
    """Continuous heterogeneity: Monte Carlo distribution of rank discrimination and one cohort."""
    from scipy.stats import spearmanr

    exp = pd.read_csv(MAIN / "tables" / "experiment_metrics_vs_D_tilde.csv")
    c = exp[exp.family == "continuous"]
    fig = plt.figure(figsize=(174 * MM, 165 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1], hspace=0.62, wspace=0.38)
    ax = fig.add_subplot(gs[0, :])
    order = ["learned_weights", "glmm", "rf_proximity", "mutual_information",
             "euclidean", "mahalanobis", "lpa_guided"]
    rng = np.random.default_rng(1)
    for i, m in enumerate(order[::-1]):
        v = c[(c.method == m) & (c.design == "in_sample")]["spearman"].to_numpy()
        ax.scatter(v, i + rng.uniform(-0.28, 0.28, len(v)), s=3, color=COLOR[m], lw=0, alpha=0.6)
        q = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        ax.plot([q[0], q[4]], [i, i], color=INK, lw=0.8)
        ax.plot([q[1], q[3]], [i, i], color=INK, lw=3.0, solid_capstyle="butt", alpha=0.55)
        ax.plot([q[2], q[2]], [i - 0.32, i + 0.32], color=INK, lw=1.6)
        o = c[(c.method == m) & (c.design == "oracle_eligible")]["spearman"]
        if len(o):
            ax.plot(np.median(o), i, marker="D", ms=4, mfc="white", mec=INK, mew=0.8, zorder=6)
        ax.text(1.01, i, f"{q[2]:.2f}", va="center", ha="left", fontsize=6.8, color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[m] for m in order[::-1]])
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Spearman correlation between estimated and true physician ranking (200 cohorts)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Rank discrimination across 200 cohorts (right: median)", loc="left", color=INK)
    ax.plot([], [], marker="D", ls="", mfc="white", mec=INK, label="Median when the true eligible stratum is supplied")
    ax.legend(loc="upper left", fontsize=6.5)
    panel_letter(ax, "A")

    lw = c[(c.method == "learned_weights") & (c.design == "in_sample")]
    rep = lw.iloc[(lw["spearman"] - lw["spearman"].median()).abs().argsort().iloc[0]]["experiment_id"]
    s = long[(long.experiment_id == rep) & (long.design == "in_sample")]
    panels = ["euclidean", "mahalanobis", "lpa_guided", "learned_weights", "mutual_information", "rf_proximity", "glmm"]
    for k, m in enumerate(panels):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        g = s[s.method == m]
        ax.scatter(g["D_star"], g["score"], s=8, color=COLOR[m], marker=MARKER[m], lw=0)
        if m != "glmm":
            ax.plot([0, 0.5], [0, 0.5], color=REF, lw=0.8, ls="--")
            ax.set_ylim(-0.02, 0.55)
            ax.set_ylabel("Estimated discordance" if k % 4 == 0 else "")
        else:
            ax.set_ylabel("GLMM score")
        rho = spearmanr(g["score"], g["D_star"]).statistic
        ax.text(0.04, 0.96, f"ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
        ax.set_xlim(-0.02, 0.52)
        ax.set_xlabel("True discordance D⋆")
        ax.set_title(LABEL[m], loc="left", color=INK, fontsize=7.5)
        if k == 0:
            panel_letter(ax, "B")
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    ax.text(0, 0.9, f"B: one representative cohort\n({rep}, 20,000 patients,\n50 physicians), chosen as the\n"
                    "cohort whose Learned Weights\ncorrelation is closest to the\nmedian. Dashed line: identity.",
            va="top", fontsize=6.5, color=INK2, transform=ax.transAxes)
    save(fig, "Figure_3_continuous")


# ----------------------------------------------------------------------------- Figure 4

def figure4() -> None:
    """Panel-size grid: median Spearman correlation by patients per physician."""
    g = pd.read_csv(GRIDS / "grid_panel_monte_carlo.csv")
    g = g[(g.design == "in_sample") & (g.metric == "spearman") & (g.patients_per_physician != 90)]
    fig, axes = plt.subplots(1, 3, figsize=(174 * MM, 60 * MM), sharey=True)
    methods = ["glmm", "learned_weights", "rf_proximity", "mutual_information", "euclidean", "mahalanobis"]
    offs = dict(zip(methods, np.linspace(-0.06, 0.06, len(methods))))
    for ax, J in zip(axes, [20, 50, 100]):
        for m in methods:
            r = g[(g.n_physicians == J) & (g.method == m)].sort_values("patients_per_physician")
            x = r["patients_per_physician"].to_numpy() * np.exp(offs[m])
            ax.errorbar(x, r["median"], yerr=[r["median"] - r["p2_5"], r["p97_5"] - r["median"]], color=COLOR[m],
                        lw=1.1, elinewidth=0.6, capsize=0, marker=MARKER[m], ms=3.8, label=LABEL[m])
        ax.axhline(0.7, color=REF, lw=0.7, ls=":")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 400, 800])
        ax.set_xticklabels(["50", "100", "200", "400", "800"])
        ax.minorticks_off()
        ax.set_xlabel("Patients per physician")
        ax.set_title(f"{J} physicians", loc="left", color=INK)
    axes[0].set_ylabel("Spearman correlation with D⋆\n(median, 2.5th–97.5th percentiles)")
    axes[0].text(55, 0.715, "0.7", fontsize=6.3, color=INK2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "Figure_4_panel_size")


# ----------------------------------------------------------------------------- Appendix figures

def score2_glmm_residuals():
    from src.benchmark.estimators import Cohort, GLMMEstimator
    from src.benchmark.runner import FEATURES, OUTCOME_COL, PHYSICIAN_COL, ExperimentSpec, generate_experiment
    from src.synthetic_generator import THRESHOLD_STRATEGY_TARGET_JOINT

    spec = ExperimentSpec("score2", "score2", seed=142, n_patients=10000, n_physicians=20,
                          hba1c_law="lognormal", covariate_model="literature")
    exp = generate_experiment(spec, THRESHOLD_STRATEGY_TARGET_JOINT)
    cohort = Cohort.from_frame(exp.df, FEATURES, OUTCOME_COL, PHYSICIAN_COL)
    est = GLMMEstimator().fit(cohort, np.arange(cohort.n))
    est.prepare(cohort)
    p = est.predict_proba(np.arange(cohort.n))
    r = (cohort.y - p) / np.sqrt(p * (1 - p))
    return cohort.y, p, r


def figureA1(diag: dict) -> None:
    y, p, r = score2_glmm_residuals()
    stored = diag["residuals"]["by_outcome"]
    # Consistency with the diagnostics of the benchmark run.
    assert abs(r[y == 0].mean() - stored["y0"]["mean"]) < 1e-3 and abs(r[y == 1].mean() - stored["y1"]["mean"]) < 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 58 * MM))
    ax = axes[0]
    bins = np.linspace(-4, 6, 81)
    ax.hist(r[y == 0], bins=bins, color="#2a78d6", alpha=0.85, label=f"No prescription (y = 0, n = {(y == 0).sum():,})")
    ax.hist(r[y == 1], bins=bins, color="#eb6834", alpha=0.85, label=f"Prescription (y = 1, n = {(y == 1).sum():,})")
    ax.set_xlabel("Pearson residual")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, 1350)
    ax.legend(loc="upper right", fontsize=6.3)
    ax.set_title("Residuals by observed outcome", loc="left", color=INK)
    panel_letter(ax, "A")
    ax = axes[1]
    b = pd.DataFrame(diag["residuals"]["by_fitted_bin"])
    ax.axhline(0, color=REF, lw=0.8, ls="--")
    ax.plot(b["fitted"], b["mean_residual"], color=COLOR["glmm"], marker="o", ms=3.5, lw=1.1)
    ax.set_xlabel(f"Fitted probability ({len(b)} classes of {int(b['n'].iloc[0])} patients)")
    ax.set_ylabel("Mean Pearson residual")
    ax.set_title("Mean residual by class of fitted probability", loc="left", color=INK)
    panel_letter(ax, "B")
    fig.tight_layout(w_pad=2)
    save(fig, "Figure_A1_glmm_residuals")


def figureB1(diag: dict) -> None:
    cal = diag["calibration"]
    lo = pd.DataFrame(cal["curve_lowess"])
    bn = pd.DataFrame(cal["curve_binned"])
    fig, ax = plt.subplots(figsize=(88 * MM, 80 * MM))
    ax.plot([0, 1], [0, 1], color=REF, lw=0.8, ls="--", label="Perfect calibration")
    ax.fill_between(lo["predicted"], lo["band_low"], lo["band_high"], color="#86b6ef", alpha=0.5, lw=0,
                    label="95% bootstrap band (2,000 resamples)")
    ax.plot(lo["predicted"], lo["observed"], color="#1c5cab", lw=1.4, label="Lowess calibration curve")
    ax.errorbar(bn["predicted"], bn["observed"], yerr=[bn["observed"] - bn["observed_low"],
                                                       bn["observed_high"] - bn["observed"]],
                fmt="o", ms=3, color=INK, elinewidth=0.6, capsize=0, label="Observed proportion (Wilson 95% CI)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability of prescription")
    ax.set_ylabel("Observed frequency")
    ax.legend(loc="upper left", fontsize=6.2)
    txt = (f"Calibration slope {cal['calibration_slope']:.3f} "
           f"[{cal['calibration_slope_ci95'][0]:.3f}, {cal['calibration_slope_ci95'][1]:.3f}]\n"
           f"ICI {cal['ici']:.3f}, E90 {cal['e90']:.3f}, Brier {cal['brier']:.3f}")
    ax.text(0.98, 0.03, txt, ha="right", va="bottom", transform=ax.transAxes, fontsize=6.3, color=INK2)
    save(fig, "Figure_B1_glmm_calibration")


def figureC1() -> None:
    g = pd.read_csv(GRIDS / "grid_cohort_panel_monte_carlo.csv")
    g = g[(g.design == "in_sample") & (g.metric == "bias") & (g.n_patients_cohort != 9000)]
    methods = ["euclidean", "mahalanobis", "learned_weights", "mutual_information", "rf_proximity"]
    Js = [5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, 5, figsize=(174 * MM, 56 * MM), sharey=True)
    offs = dict(zip(methods, np.linspace(-0.05, 0.05, len(methods))))
    for ax, J in zip(axes, Js):
        for m in methods:
            r = g[(g.n_physicians == J) & (g.method == m)].sort_values("n_patients_cohort")
            x = r["n_patients_cohort"].to_numpy() / 1000 * np.exp(offs[m])
            ax.errorbar(x, r["median"], yerr=[r["median"] - r["p2_5"], r["p97_5"] - r["median"]], color=COLOR[m],
                        lw=1.0, elinewidth=0.6, marker=MARKER[m], ms=3.4, label=LABEL[m])
        ax.axhline(0, color=REF, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 30])
        ax.set_xticklabels(["5", "10", "20", "30"])
        ax.minorticks_off()
        ax.set_title(f"{J} physicians", loc="left", color=INK)
        ax.set_xlabel("Patients (thousands)")
    axes[0].set_ylabel("Bias against the manual-pairing\nreference (median, 2.5th–97.5th pct.)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=0.6)
    save(fig, "Figure_C1_cohort_panel_grid")


def main() -> int:
    long = load_long(["family", "experiment_id", "physician", "physician_group", "method", "design",
                      "score", "D_star", "D_tilde"])
    figure2(long)
    figure3(long)
    figure4()
    diag = json.loads((MAIN / "fitted_parameters" / "score2.json").read_text())["fitted"]["glmm_diagnostics"]
    figureA1(diag)
    figureB1(diag)
    figureC1()
    return 0


if __name__ == "__main__":
    sys.exit(main())
