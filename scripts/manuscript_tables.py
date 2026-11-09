"""Tables of the revised manuscript, built from the benchmark outputs.

    python scripts/manuscript_tables.py            # writes results/manuscript/tables/

Inputs: ``results/revision_main`` (benchmark_long.csv.gz and tables/),
``results/revision_grids/tables`` and ``results/revision_glmm_continuous/tables``,
produced by ``python -m src.benchmark.metrics``. Each table is written as a CSV
of formatted cells (ready to paste) and collected in ``tables.json``.
Numbers: three decimals; brackets are 95% percentile bootstrap intervals
(B = 2,000, seed 42) unless stated otherwise; parentheses after a median are
the 2.5th and 97.5th percentiles over Monte Carlo replicates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "revision_main"
GRIDS = ROOT / "results" / "revision_grids" / "tables"
GLMMC = ROOT / "results" / "revision_glmm_continuous" / "tables"
OUT = ROOT / "results" / "manuscript" / "tables"

METHODS = ["euclidean", "mahalanobis", "lpa_guided", "genetic_mahalanobis",
           "learned_weights", "mutual_information", "rf_proximity"]
LABEL = {
    "euclidean": "Euclidean", "mahalanobis": "Mahalanobis", "lpa_guided": "LPA-guided",
    "genetic_mahalanobis": "Genetic Mahalanobis", "genetic_mahalanobis_legacy": "Genetic Mahalanobis (original)",
    "learned_weights": "Learned Weights", "mutual_information": "Mutual Information",
    "rf_proximity": "RF Proximity", "glmm": "GLMM (additive)", "glmm_oracle": "GLMM + true eligibility (oracle)",
    "glmm_interactions": "GLMM + pairwise interactions", "glmm_splines": "GLMM + natural splines",
    "gbm_offset": "Gradient boosting + physician intercept", "manual_pairing": "Manual-pairing reference",
    "theoretical": "Theoretical D⋆",
}
B, SEED = 2000, 42
UNSUPERVISED = {"euclidean", "mahalanobis", "lpa_guided"}


def f3(x: float, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    if abs(x) < 5e-4:
        return "0.000"
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-", "−")


def ci(m: float, lo: float, hi: float, signed: bool = False) -> str:
    return f"{f3(m, signed)} [{f3(lo, signed)}, {f3(hi, signed)}]"


def mc(med: float, lo: float, hi: float, signed: bool = False) -> str:
    return f"{f3(med, signed)} ({f3(lo, signed)} to {f3(hi, signed)})"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def boot_mean(values: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (B, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tab(name: str) -> pd.DataFrame:
    return pd.read_csv(MAIN / "tables" / f"{name}.csv")


def pick(t: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(t), dtype=bool)
    for k, v in kw.items():
        m &= (t[k].astype(str) == str(v)).to_numpy()
    r = t[m]
    if len(r) != 1:
        raise KeyError(f"{kw}: {len(r)} rows")
    return r.iloc[0]


# ----------------------------------------------------------------------------- loading

def load_long() -> pd.DataFrame:
    cols = ["family", "experiment_id", "physician", "physician_group", "method", "design", "caliper_percentile",
            "score_type", "score", "D_star", "D_tilde"]
    d = pd.read_csv(MAIN / "benchmark_long.csv.gz", usecols=cols, low_memory=False)
    keep = d["caliper_percentile"].isna() | np.isclose(d["caliper_percentile"].astype(float), 25.0)
    return d[keep]


# ----------------------------------------------------------------------------- main tables

def table5_score2(long: pd.DataFrame) -> pd.DataFrame:
    """Mean discordance by physician group (SCORE2), with bias, MAE and Spearman."""
    g = tab("table4_score2_groups")
    from scipy.stats import spearmanr

    rows = []
    for m in ["theoretical", "manual_pairing"] + METHODS:
        rec = {"Method": LABEL[m]}
        for k in range(1, 6):
            r = pick(g, method=m, physician_group=k)
            rec[f"Group {k}"] = f3(r["mean"]) if m == "theoretical" else ci(r["mean"], r["ci_low"], r["ci_high"])
        if m in METHODS:
            s = long[(long.family == "score2") & (long.design == "in_sample") & (long.method == m)]
            e = (s["score"] - s["D_tilde"]).to_numpy()
            rec["Bias"] = ci(*boot_mean(e), signed=True)
            rec["MAE"] = ci(*boot_mean(np.abs(e)))
            rec["Spearman"] = f"{spearmanr(s['score'], s['D_star']).statistic:.3f}"
        else:
            rec["Bias"] = rec["MAE"] = rec["Spearman"] = "NA"
        rows.append(rec)
    return pd.DataFrame(rows)


def table6_progressive() -> pd.DataFrame:
    t = tab("table5_progressive_vs_D_tilde")
    rows = []
    for m in METHODS:
        b = pick(t, method=m, metric="bias")
        rows.append({
            "Method": LABEL[m],
            "Bias": ci(b["mean"], b["ci_low"], b["ci_high"], signed=True),
            "Experiments with positive bias": pct(b["share_positive"]),
            "MAE": ci(*pick(t, method=m, metric="mae")[["mean", "ci_low", "ci_high"]]),
            "RMSE": ci(*pick(t, method=m, metric="rmse")[["mean", "ci_low", "ci_high"]]),
            "Calibration slope": ci(*pick(t, method=m, metric="calibration_slope")[["mean", "ci_low", "ci_high"]]),
        })
    return pd.DataFrame(rows)


def table7_window() -> pd.DataFrame:
    t = tab("table6_by_window")
    bins = ["1", "2-3", "4-6", "7-9"]
    n = {b: int(pick(t, method="euclidean", metric="bias", window_bin=b)["n_experiments"]) for b in bins}
    rows = []
    for metric, title, signed in (("bias", "A. Bias", True), ("mae", "B. MAE", False),
                                  ("calibration_slope", "C. Calibration slope", False)):
        rows.append({"Method": title, **{f"w = {b.replace('-', '–')} (n = {n[b]})": "" for b in bins}})
        for m in METHODS:
            rec = {"Method": LABEL[m]}
            for b in bins:
                r = pick(t, method=m, metric=metric, window_bin=b)
                rec[f"w = {b.replace('-', '–')} (n = {n[b]})"] = ci(r["mean"], r["ci_low"], r["ci_high"], signed)
            rows.append(rec)
    return pd.DataFrame(rows)


def table8_pass() -> pd.DataFrame:
    t = tab("table7_by_pass")
    rows = []
    for m in METHODS:
        p1, p2 = pick(t, method=m, metric="bias", pass_index=1), pick(t, method=m, metric="bias", pass_index=2)
        rows.append({"Method": LABEL[m], "Pass 1 (n = 45)": ci(p1["mean"], p1["ci_low"], p1["ci_high"], True),
                     "Pass 2 (n = 45)": ci(p2["mean"], p2["ci_low"], p2["ci_high"], True),
                     "Difference (pass 2 − pass 1)": f3(p2["mean"] - p1["mean"], True)})
    return pd.DataFrame(rows)


def table9_glmm() -> pd.DataFrame:
    # Progressive column on the 90 progressive experiments only: the stored
    # glmm_specifications table pools SCORE2 with them (91 experiments).
    sys.path.insert(0, str(ROOT))
    from src.benchmark.metrics import glmm_specifications
    exp_all = tab("experiment_metrics_vs_D_tilde")
    five = glmm_specifications(exp_all[exp_all["family"] == "progressive"])
    cont = pd.read_csv(GLMMC / "glmm_specifications_continuous.csv")
    paired = pd.read_csv(GLMMC / "glmm_specifications_continuous_paired.csv")
    exp = tab("experiment_metrics_vs_D_tilde")
    rows = []
    for m in ["glmm", "glmm_oracle", "glmm_splines", "glmm_interactions", "gbm_offset"]:
        s2 = exp[(exp.family == "score2") & (exp.method == m) & (exp.design == "in_sample")]["spearman"]
        pr = pick(five, design="in_sample", method=m, metric="spearman")
        c = pick(cont, design="in_sample", method=m, metric="spearman")
        k = pick(cont, design="kfold", method=m, metric="spearman")
        if m == "glmm":
            diff = "NA"
        else:
            p = pick(paired, method_a="glmm", method_b=m, metric="spearman", design="in_sample")
            # stored as glmm minus variant: report variant minus additive
            diff = mc(-p["median_difference"], -p["interval_high"], -p["interval_low"], signed=True)
        rows.append({
            "Specification": LABEL[m],
            "SCORE2 (five groups)": f"{float(s2.iloc[0]):.3f}",
            "Progressive rules, mean (n = 90)": ci(pr["mean"], pr["ci_low"], pr["ci_high"]),
            "Continuous heterogeneity, median (200 cohorts)": mc(c["median"], c["p2_5"], c["p97_5"]),
            "Difference with the additive GLMM (200 cohorts)": diff,
            "Cross-fitted, median (10 cohorts)": f3(k["median"]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- supplementary tables

def tableS2_crossfitting() -> pd.DataFrame:
    tb, tm = tab("cross_fitting_bias"), tab("cross_fitting_mae")
    contrasts = [("pool_effect", "Pairing pool (fold-restricted − in-sample)"),
                 ("leakage_effect", "Outcome leakage (k-fold − fold-restricted)"),
                 ("lopo_vs_in_sample", "Leave-one-physician-out − in-sample"),
                 ("external_vs_in_sample", "External − in-sample")]
    rows = []
    for metric, t, title in (("bias", tb, "A. Change in mean bias"), ("mae", tm, "B. Change in MAE")):
        rows.append({"Method": title, **{lab: "" for _, lab in contrasts}})
        for m in METHODS:
            rec = {"Method": LABEL[m]}
            for c, lab in contrasts:
                r = t[(t.method == m) & (t.contrast == c)]
                if r.empty:
                    rec[lab] = "not computed"
                elif c != "pool_effect" and m in UNSUPERVISED:
                    rec[lab] = "0 by construction"
                else:
                    r = r.iloc[0]
                    n = int(r["n_experiments"])
                    rec[lab] = (f"{f3(r['mean'], True)} (n = 1)" if n == 1 else
                                f"{ci(r['mean'], r['ci_low'], r['ci_high'], True)} (n = {n})")
            rows.append(rec)
    return pd.DataFrame(rows)


def tableS3_decomposition() -> pd.DataFrame:
    t = tab("decomposition_summary")
    cov = tab("coverage")
    rows = []
    for m in METHODS:
        rec = {"Method": LABEL[m]}
        for comp, lab in (("total", "Total"), ("comparability", "Comparability"), ("estimation", "Estimation")):
            r = pick(t, family="progressive", method=m, metric=f"{comp}_bias")
            rec[f"{lab} error, bias"] = ci(r["mean"], r["ci_low"], r["ci_high"], True)
        r = pick(t, family="progressive", method=m, metric="comparability_mae")
        rec["Comparability error, MAE"] = ci(r["mean"], r["ci_low"], r["ci_high"])
        c = pick(cov, family="progressive", design="in_sample", method=m)
        rec["Mixed pairs"] = f"{100 * c['frac_pairs_mixed']:.1f}%"
        rows.append(rec)
    return pd.DataFrame(rows)


def tableS4_coverage() -> pd.DataFrame:
    cov, cal = tab("coverage"), tab("caliper_sensitivity")
    rows = []
    for m in METHODS + ["genetic_mahalanobis_legacy"]:
        c = cov[(cov.family == "progressive") & (cov.design == "in_sample") & (cov.method == m)]
        rec = {"Method": LABEL[m]}
        if c.empty:
            rec.update({"Retention": "NA", "Pairs per physician (median)": "NA",
                        "Physicians with < 10 pairs": "NA", "Within-pair SMD, mean": "NA"})
        else:
            c = c.iloc[0]
            rec.update({"Retention": f"{100 * c['retention']:.1f}%",
                        "Pairs per physician (median)": f"{c['n_pairs_median']:.0f}",
                        "Physicians with < 10 pairs": f"{100 * c['unusable_physician_rate']:.1f}%",
                        "Within-pair SMD, mean": f"{c['smd_within_mean']:.2f}"})
        for p in (10, 25, 50, 100):
            r = cal[(cal.method == m) & (cal.design == "in_sample") & np.isclose(cal.caliper_percentile, p)]
            lab = "no caliper" if p == 100 else f"caliper P{p}"
            rec[f"Bias / MAE, {lab}"] = "NA" if r.empty else \
                f"{f3(r.iloc[0]['bias'], True)} / {f3(r.iloc[0]['mae'])}"
        rows.append(rec)
    return pd.DataFrame(rows)


def tableS5_latent() -> pd.DataFrame:
    t = tab("latent_E1")
    t = t[(t.design == "in_sample") & (t.score_type == "discordance")]
    rows = []
    for g in sorted(t.latent_gamma.unique()):
        s = t[t.latent_gamma == g]
        rec = {"γ": f"{g:g}", "Observable discordance": f3(s["D_observable"].iloc[0]),
               "Intrinsic inconsistency": f3(s["D_intrinsic"].iloc[0])}
        for m in METHODS:
            r = s[s.method == m]
            rec[LABEL[m]] = "NA" if r.empty else \
                f"{f3(r.iloc[0]['bias_vs_observable'], True)} / {f3(r.iloc[0]['bias_vs_intrinsic'], True)}"
        rows.append(rec)
    return pd.DataFrame(rows)


def tableS6_window_confounding() -> pd.DataFrame:
    t = tab("window_confounding_bias")
    rows = []
    for m in METHODS:
        u, a = pick(t, method=m, model="unadjusted"), pick(t, method=m, model="adjusted")
        rows.append({"Method": LABEL[m],
                     "Window size, unadjusted": ci(u["slope_window"], u["ci_low"], u["ci_high"], True),
                     "Window size, adjusted for eligibility": ci(a["slope_window"], a["ci_low"], a["ci_high"], True),
                     "Realised eligibility (adjusted model)": ci(a["slope_eligibility"], a["eligibility_ci_low"],
                                                                 a["eligibility_ci_high"], True)})
    return pd.DataFrame(rows)


def tableS7_glmm_diagnostics() -> pd.DataFrame:
    d = tab("glmm_diagnostics")
    names = {"score2": "SCORE2 (1)", "progressive": "Progressive rules (90)",
             "sensitivity_independent": "Sensitivity: original covariate model (1)",
             "sensitivity_diabetes_hba1c": "Sensitivity: diabetic HbA1c mixture (1)"}
    rows = []
    for fam, lab in names.items():
        s = d[d.family == fam]
        if s.empty:
            continue
        med = lambda c: float(s[c].median())  # noqa: E731
        slope_excl = float(((s["calibration_slope_ci_low"] > 1) | (s["calibration_slope_ci_high"] < 1)).mean())
        rows.append({
            "Experiments": lab,
            "Calibration-in-the-large": f3(med("calibration_in_the_large"), True),
            "Calibration slope": f3(med("calibration_slope")),
            "Slope interval excluding 1": f"{int(round(slope_excl * len(s)))} of {len(s)}",
            "ICI": f3(med("ici")), "E90": f3(med("e90")), "Brier": f3(med("brier")),
            "Spiegelhalter p < 0.05": f"{int((s['spiegelhalter_p'] < 0.05).sum())} of {len(s)}",
            "PPC coverage: rate": f"{med('coverage_95_rate'):.2f}",
            "PPC coverage: all-pairs discordance": f"{med('coverage_95_discordance'):.2f}",
            "PPC coverage: matched-pair discordance": f"{med('coverage_95_matched_pair_discordance'):.2f}",
            "Ranking after recalibration (Spearman)": f"{med('spearman_original_vs_recalibrated'):.2f}",
        })
    return pd.DataFrame(rows)


def tableS_sensitivity(long: pd.DataFrame, family: str) -> pd.DataFrame:
    from scipy.stats import spearmanr

    s = long[(long.family == family) & (long.design == "in_sample")]
    ref = s.drop_duplicates("physician")
    rows = [{"Method": LABEL["manual_pairing"],
             **{f"Group {k}": f3(ref[ref.physician_group == k]["D_tilde"].mean()) for k in range(1, 6)},
             "Bias": "NA", "MAE": "NA", "Spearman": "NA"}]
    for m in METHODS + ["genetic_mahalanobis_legacy", "glmm"]:
        r = s[s.method == m]
        if r.empty:
            continue
        rec = {"Method": LABEL[m]}
        if m == "glmm":
            rec.update({f"Group {k}": f"{r[r.physician_group == k]['score'].mean():.2f}" for k in range(1, 6)})
            rec.update(Bias="NA", MAE="NA")
        else:
            e = (r["score"] - r["D_tilde"]).to_numpy()
            rec.update({f"Group {k}": f3(r[r.physician_group == k]["score"].mean()) for k in range(1, 6)})
            rec.update(Bias=ci(*boot_mean(e), signed=True), MAE=ci(*boot_mean(np.abs(e))))
        rec["Spearman"] = f"{spearmanr(r['score'], r['D_star']).statistic:.3f}"
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    long = load_long()
    tables = {
        "Table 5": table5_score2(long),
        "Table 6": table6_progressive(),
        "Table 7": table7_window(),
        "Table 8": table8_pass(),
        "Table 9": table9_glmm(),
        "Table S2": tableS2_crossfitting(),
        "Table S3": tableS3_decomposition(),
        "Table S4": tableS4_coverage(),
        "Table S5": tableS5_latent(),
        "Table S6": tableS6_window_confounding(),
        "Table S7": tableS7_glmm_diagnostics(),
        "Table S8": tableS_sensitivity(long, "sensitivity_independent"),
        "Table S9": tableS_sensitivity(long, "sensitivity_diabetes_hba1c"),
    }
    payload = {}
    for name, t in tables.items():
        t.to_csv(OUT / f"{name.replace(' ', '_')}.csv", index=False)
        payload[name] = {"columns": list(t.columns), "rows": t.astype(str).values.tolist()}
        print(f"{name}: {t.shape}")
    (OUT / "tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
