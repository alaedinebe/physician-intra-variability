"""Table S12 and Figure S1 of scenario E2 (physician-specific SCORE2 thresholds).

    python -m src.benchmark.metrics --run results/revision_physician_thresholds
    python scripts/physician_thresholds_report.py [--run results/revision_physician_thresholds]

Reads ``<run>/tables/E2_*.csv`` and writes ``<run>/report/``:
``Table_S12.csv`` (formatted cells, ready to paste), ``Table_S12_long.csv``
(unformatted values), ``Figure_S1_physician_thresholds.png/.pdf`` and
``key_numbers_E2.json`` (values quoted in the text). Uncertainty is Monte
Carlo: median over the replicates, with the minimum and maximum (few replicates).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["euclidean", "mahalanobis", "lpa_guided", "genetic_mahalanobis",
           "learned_weights", "mutual_information", "rf_proximity"]
LABEL = {"euclidean": "Euclidean", "mahalanobis": "Mahalanobis", "lpa_guided": "LPA-guided",
         "genetic_mahalanobis": "Genetic Mahalanobis", "learned_weights": "Learned Weights",
         "mutual_information": "Mutual Information", "rf_proximity": "RF Proximity",
         "glmm": "GLMM (additive)", "gbm_offset": "Gradient boosting + physician intercept",
         "glmm_oracle": "GLMM + true eligibility (oracle)"}


def f3(x: float, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    if abs(x) < 5e-4:
        return "0.000"
    return (f"{x:+.3f}" if signed else f"{x:.3f}").replace("-", "−")


def error_vs_epsilon(run: Path) -> pd.DataFrame:
    """Does a physician's error depend on how far their thresholds are from the common ones?

    For each physician j, epsilon_j = log(kappa_j) / sigma is the standardised threshold shift
    (shared across the values of sigma within a replicate). The paired change in error,
    (estimate - manual-pairing reference) at sigma minus the same quantity at sigma = 0 on the
    same cohort, is regressed on epsilon_j (signed: raised thresholds against lowered ones, which
    cancel in the mean bias) and the paired change in absolute error on |epsilon_j| (penalty
    for any departure from the consensus). In-sample design. Slopes are per unit of epsilon
    (one standard deviation of the log threshold multiplier). Pooled OLS with replicate
    intercepts and its 95% CI (physicians are independent draws of epsilon), plus the slope of
    each replicate.
    """
    cols = ["replicate", "physician", "threshold_log_sd", "threshold_multiplier", "method",
            "design", "score_type", "score", "D_tilde"]
    d = pd.read_csv(run / "benchmark_long.csv.gz", usecols=cols)
    d = d[(d.design == "in_sample") & (d.score_type == "discordance") & d.method.isin(METHODS)
          & d.score.notna() & d.D_tilde.notna()].copy()
    d["err"] = d.score - d.D_tilde
    key = ["replicate", "physician", "method"]
    base = d[d.threshold_log_sd == 0][key + ["err"]].rename(columns={"err": "err0"})

    def ols(y: np.ndarray, x: np.ndarray, rep: np.ndarray) -> tuple[float, float, float]:
        dummies = pd.get_dummies(rep).to_numpy(dtype=float)
        X = np.column_stack([x, dummies])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(y) - X.shape[1]
        se = float(np.sqrt(resid @ resid / dof * np.linalg.inv(X.T @ X)[0, 0]))
        return float(beta[0]), float(beta[0] - 1.96 * se), float(beta[0] + 1.96 * se)

    rows = []
    for sd in sorted(s for s in d.threshold_log_sd.unique() if s > 0):
        x = d[d.threshold_log_sd == sd].merge(base, on=key)
        x["eps"] = np.log(x.threshold_multiplier) / sd
        x["d_err"] = x.err - x.err0
        x["d_abs_err"] = x.err.abs() - x.err0.abs()
        for m, g in x.groupby("method"):
            for name, y, v in (("signed", g.d_err, g.eps), ("absolute", g.d_abs_err, g.eps.abs())):
                b, lo, hi = ols(y.to_numpy(), v.to_numpy(), g.replicate.to_numpy())
                per_rep = [float(np.polyfit(v[g.replicate == r], y[g.replicate == r], 1)[0])
                           for r in sorted(g.replicate.unique())]
                rows.append(dict(method=m, sd=sd, test=name, n=len(g), slope=b, ci_low=lo, ci_high=hi,
                                 rep_min=min(per_rep), rep_max=max(per_rep),
                                 n_rep_negative=sum(s < 0 for s in per_rep), n_rep=len(per_rep),
                                 mean_change=float(y.mean()),
                                 r2=float(np.corrcoef(v, y)[0, 1] ** 2)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "results" / "revision_physician_thresholds")
    args = ap.parse_args()
    tdir, out = args.run / "tables", args.run / "report"
    out.mkdir(parents=True, exist_ok=True)
    exp = pd.read_csv(tdir / "E2_experiment_metrics.csv")
    dec = pd.read_csv(tdir / "E2_decomposition.csv")
    elig = pd.read_csv(tdir / "E2_eligibility.csv")
    sds = sorted(exp["threshold_log_sd"].unique())
    n_rep = int(exp["replicate"].nunique())

    def stat(method: str, design: str, metric: str, sd: float) -> tuple[float, float, float]:
        v = exp[(exp.method == method) & (exp.design == design) & (exp.threshold_log_sd == sd)][metric].dropna()
        return (float(v.median()), float(v.min()), float(v.max())) if len(v) else (np.nan, np.nan, np.nan)

    eps = error_vs_epsilon(args.run)
    eps.to_csv(out / "E2_error_vs_epsilon.csv", index=False)

    rows, long = [], []
    for m in METHODS + ["glmm", "gbm_offset"]:
        if m not in set(exp.method):
            continue
        rec = {"Method": LABEL[m]}
        for sd in sds:
            tag = f"σ = {sd:g}"
            if m in METHODS:
                b = stat(m, "in_sample", "bias", sd)
                mae = stat(m, "in_sample", "mae", sd)
                ob = stat(m, "oracle_eligible", "bias", sd)
                mix = stat(m, "in_sample", "frac_pairs_mixed", sd)
                rec[f"Bias, {tag}"] = f"{f3(b[0], True)} ({f3(b[1], True)} to {f3(b[2], True)})"
                rec[f"MAE, {tag}"] = f3(mae[0])
                rec[f"Oracle-eligible bias, {tag}"] = f3(ob[0], True)
                rec[f"Mixed pairs, {tag}"] = f"{100 * mix[0]:.1f}%" if np.isfinite(mix[0]) else "NA"
                for name, v in (("bias", b), ("mae", mae), ("oracle_bias", ob), ("mixed", mix)):
                    long.append(dict(method=m, sd=sd, metric=name, median=v[0], min=v[1], max=v[2]))
            if m in METHODS and sd > 0:
                e = eps[(eps.method == m) & (eps.sd == sd) & (eps.test == "signed")]
                if len(e):
                    e = e.iloc[0]
                    rec[f"Error slope on ε, {tag}"] = (f"{f3(e.slope, True)} ({f3(e.ci_low, True)} "
                                                      f"to {f3(e.ci_high, True)})")
            s = stat(m, "in_sample", "spearman", sd)
            rec[f"Spearman, {tag}"] = f3(s[0])
            long.append(dict(method=m, sd=sd, metric="spearman", median=s[0], min=s[1], max=s[2]))
        rows.append(rec)
    table = pd.DataFrame(rows).fillna("NA")
    table.to_csv(out / "Table_S12.csv", index=False)
    long_df = pd.DataFrame(long)
    long_df.to_csv(out / "Table_S12_long.csv", index=False)

    # Decomposition (comparability vs estimation) and eligibility, for the text.
    keys: dict = {"n_replicates": n_rep, "threshold_log_sds": sds,
                  "eligibility": elig.round(4).to_dict(orient="records")}
    d = dec[dec.metric.isin(["total_bias", "comparability_bias", "estimation_bias"])]
    keys["decomposition_median"] = {f"{r.method}|sd={r.threshold_log_sd:g}|{r.metric}": round(r.median, 4)
                                    for r in d.itertuples()}
    for metric in ("bias", "mae", "spearman", "oracle_bias", "mixed"):
        sub = long_df[long_df.metric == metric]
        keys[f"{metric}_median_by_sd"] = {f"{r.method}|sd={r.sd:g}": round(r.median, 4) for r in sub.itertuples()}
    # Paired change with the common-threshold cohort (same replicate), in-sample.
    ch_path = tdir / "E2_change_vs_common_thresholds.csv"
    if ch_path.exists():
        ch = pd.read_csv(ch_path)
        ch = ch[ch.design == "in_sample"]
        keys["change_vs_sd0_median"] = {f"{r.method}|sd={r.threshold_log_sd:g}|{r.metric}": round(r.median, 4)
                                        for r in ch.itertuples()}
    keys["error_vs_epsilon"] = {f"{r.method}|sd={r.sd:g}|{r.test}": {
        "slope": round(r.slope, 4), "ci": [round(r.ci_low, 4), round(r.ci_high, 4)],
        "replicates": [round(r.rep_min, 4), round(r.rep_max, 4)],
        "n_rep_negative": int(r.n_rep_negative), "mean_change": round(r.mean_change, 4),
        "r2": round(r.r2, 3)} for r in eps.itertuples()}
    (out / "key_numbers_E2.json").write_text(json.dumps(keys, indent=1, default=float), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    for m in METHODS:
        sub = long_df[(long_df.method == m) & (long_df.metric == "bias")]
        if sub.empty:
            continue
        axes[0].errorbar(sub.sd, sub["median"], yerr=[sub["median"] - sub["min"], sub["max"] - sub["median"]],
                         marker="o", ms=3, lw=1, capsize=2, label=LABEL[m])
        sub = long_df[(long_df.method == m) & (long_df.metric == "mixed")]
        axes[1].plot(sub.sd, 100 * sub["median"], marker="o", ms=3, lw=1)
    axes[0].axhline(0, color="grey", lw=0.6, ls="--")
    for ax in axes[:2]:
        ax.set_xlabel("Spread σ of physician thresholds")
        ax.set_xticks(sds)
    axes[0].set_ylabel("Bias (median over replicates)")
    axes[1].set_ylabel("Mixed pairs (%)")
    # (C) Slope of the paired change in error on epsilon_j, with 95% CI.
    sig = eps[eps.test == "signed"]
    pos = np.arange(len(METHODS))
    for k, (sd, mk) in enumerate(((s, mk) for s, mk in zip(sorted(sig.sd.unique()), ("o", "s")))):
        sub = sig[sig.sd == sd].set_index("method").reindex(METHODS)
        y = pos + (k - 0.5) * 0.25
        axes[2].errorbar(sub.slope, y, xerr=[sub.slope - sub.ci_low, sub.ci_high - sub.slope],
                         fmt=mk, ms=3, lw=1, capsize=2, color="black" if k else "grey", label=f"σ = {sd:g}")
    axes[2].axvline(0, color="grey", lw=0.6, ls="--")
    axes[2].set_yticks(pos)
    axes[2].set_yticklabels([LABEL[m] for m in METHODS], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Change in error per unit of ε")
    axes[2].legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
    for ax, letter in zip(axes, "ABC"):
        ax.set_title(letter, loc="left", fontweight="bold", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Figure_S1_physician_thresholds.{ext}", dpi=300)
    print(table.to_string(index=False))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
