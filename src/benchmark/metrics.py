# -*- coding: utf-8 -*-
"""Post-processing of the benchmark table into the tables of the manuscript.

Usage (from the repository root):

    python -m src.benchmark.metrics --run results/revision_main

reads ``<run>/benchmark_long.csv.gz`` and writes CSV tables to
``<run>/tables/``, plus ``<run>/tables/key_numbers.json`` with the values
quoted in the text. Every table is a pure function of the long table, so
the analysis is reproducible from the archived output alone.

Definitions (physician-level error e = score - reference, discordance-rate
methods only; the GLMM score is not on the discordance scale):

- mean bias: mean of e over the physicians of an experiment;
- MAE: mean of |e|; RMSE: sqrt(mean e^2); median and maximum of |e|;
- calibration intercept and slope: ordinary least squares of the score on the
  reference across the physicians of one experiment (score = a + b reference),
  so that b < 1 means compression of the estimated range;
- rank discrimination: Spearman correlation between the score and the
  theoretical discordance D_star (all methods, GLMM included).

The primary reference is the empirical manual pairing D_tilde, as in the
formula of Reviewer 2; D_star is the secondary reference. The noise floor is
the error of D_tilde itself with respect to D_star.

Experiment-level metrics are summarised across experiments by their mean
with a paired percentile bootstrap over experiments (B = 2000, seed 42): one
set of resampled experiments is shared by all methods. For Monte Carlo
replications (continuous heterogeneity), the distribution over replicates is
reported directly (median, IQR, 2.5 and 97.5 percentiles, Monte Carlo
standard error of the mean).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

LOGGER = logging.getLogger("benchmark.metrics")

MAIN_CALIPER = 25.0
N_BOOT = 2000
BOOT_SEED = 42
MIN_PAIRS_USABLE = 10
REFERENCES = ("D_tilde", "D_star")
EXPERIMENT_KEYS = ["family", "experiment_id"]
PRIMARY_FAMILIES = ("score2", "progressive")

METHOD_ORDER = [
    "euclidean", "mahalanobis", "lpa_guided", "genetic_mahalanobis", "genetic_mahalanobis_legacy",
    "learned_weights", "mutual_information", "rf_proximity", "glmm",
    "glmm_oracle", "glmm_interactions", "glmm_splines", "gbm_offset",
]
MODEL_METHODS = ("glmm", "glmm_oracle", "glmm_interactions", "glmm_splines", "gbm_offset")


# -----------------------------------------------------------------------------
# Loading and selection
# -----------------------------------------------------------------------------


def load(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "benchmark_long.csv.gz"
    df = pd.read_csv(path, low_memory=False)
    for col, default in (("caliper_percentile", MAIN_CALIPER), ("D_star_intrinsic", np.nan),
                         ("latent_gamma", np.nan), ("covariate_model", "literature")):
        if col not in df.columns:
            df[col] = default
    LOGGER.info("Loaded %d rows from %s", len(df), path)
    return df


def main_caliper(df: pd.DataFrame) -> pd.DataFrame:
    """Rows computed with the main caliper (the GLMM has no caliper)."""
    keep = df["caliper_percentile"].isna() | np.isclose(df["caliper_percentile"].astype(float), MAIN_CALIPER)
    return df[keep]


def discordance_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["score_type"] == "discordance"]


def _order_methods(table: pd.DataFrame, col: str = "method") -> pd.DataFrame:
    """Sort rows by the canonical method order, keeping the original order otherwise."""
    if table.empty or col not in table.columns:
        return table
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    rank = table[col].map(order).fillna(len(order))
    return table.iloc[np.argsort(rank.to_numpy(), kind="stable")].reset_index(drop=True)


# -----------------------------------------------------------------------------
# Experiment-level metrics
# -----------------------------------------------------------------------------


def _calibration(score: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(score) & np.isfinite(ref)
    if ok.sum() < 3 or np.var(ref[ok]) <= 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(ref[ok], score[ok], 1)
    return float(intercept), float(slope)


def _rank(score: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(score) & np.isfinite(truth)
    if ok.sum() < 3 or np.unique(truth[ok]).size < 2 or np.unique(score[ok]).size < 2:
        return float("nan"), float("nan")
    return float(spearmanr(score[ok], truth[ok]).statistic), float(kendalltau(score[ok], truth[ok]).statistic)


def experiment_metrics(df: pd.DataFrame, reference: str = "D_tilde",
                       truth_for_rank: str = "D_star") -> pd.DataFrame:
    """One row per (experiment, method, design, caliper) with error, calibration and coverage metrics."""
    keys = EXPERIMENT_KEYS + ["method", "supervised", "design", "computed_as", "caliper_percentile", "score_type"]
    first_cols = ["window_size", "pass_index", "realised_eligibility", "n_binary_rule_variables",
                  "latent_gamma", "replicate", "covariate_model", "hba1c_law", "calibration_attainable"]
    rows = []
    for key, g in df.groupby(keys, dropna=False, sort=False):
        rec = dict(zip(keys, key))
        for c in first_cols:
            if c in g.columns:
                rec[c] = g[c].iloc[0]
        score = g["score"].to_numpy(dtype=float)
        rec["n_physicians"] = int(np.isfinite(score).sum())
        rec["spearman"], rec["kendall"] = _rank(score, g[truth_for_rank].to_numpy(dtype=float))
        if rec["score_type"] == "discordance":
            ref = g[reference].to_numpy(dtype=float)
            e = score - ref
            e = e[np.isfinite(e)]
            rec.update(
                bias=float(np.mean(e)) if e.size else np.nan,
                mae=float(np.mean(np.abs(e))) if e.size else np.nan,
                rmse=float(np.sqrt(np.mean(e ** 2))) if e.size else np.nan,
                median_ae=float(np.median(np.abs(e))) if e.size else np.nan,
                max_ae=float(np.max(np.abs(e))) if e.size else np.nan,
            )
            rec["calibration_intercept"], rec["calibration_slope"] = _calibration(score, ref)
            rec["retention"] = float(g["retention"].mean())
            rec["n_pairs_median"] = float(g["n_pairs"].median())
            rec["unusable_physician_rate"] = float((g["n_pairs"] < MIN_PAIRS_USABLE).mean())
            for c in ("frac_pairs_both_eligible", "frac_pairs_mixed", "frac_pairs_both_ineligible",
                      "smd_within_max", "smd_within_mean", "pair_distance_median", "caliper_threshold"):
                rec[c] = float(g[c].mean())
        rec["fit_seconds"] = float(g["fit_seconds"].iloc[0])
        rec["estimate_seconds"] = float(g["estimate_seconds"].iloc[0])
        rec["n_fits"] = int(g["n_fits"].iloc[0])
        rec["reference"] = reference
        rows.append(rec)
    return pd.DataFrame(rows)


def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Error of the manual pairing D_tilde with respect to D_star, per experiment (method-independent)."""
    phys = df.drop_duplicates(EXPERIMENT_KEYS + ["physician"])
    rows = []
    for key, g in phys.groupby(EXPERIMENT_KEYS, sort=False):
        e = (g["D_tilde"] - g["D_star"]).to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        rows.append(dict(zip(EXPERIMENT_KEYS, key), bias=float(np.mean(e)), mae=float(np.mean(np.abs(e))),
                         rmse=float(np.sqrt(np.mean(e ** 2))), max_ae=float(np.max(np.abs(e))),
                         median_eligible_per_physician=float(g["n_eligible"].median()),
                         window_size=g["window_size"].iloc[0]))
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Summaries across experiments
# -----------------------------------------------------------------------------


def paired_bootstrap_summary(exp: pd.DataFrame, metrics: Iterable[str], by: Iterable[str] = (),
                             n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Mean of each metric per method, with paired percentile bootstrap CIs over experiments.

    Within each stratum of ``by``, one set of resampled experiment indices is
    shared by all methods, which preserves the between-method correlation.
    """
    metrics = list(metrics)
    by = list(by)
    out = []
    strata = [((), exp)] if not by else exp.groupby(by, dropna=False, sort=True)
    for skey, sdf in strata:
        skey = skey if isinstance(skey, tuple) else (skey,)
        experiments = sdf["experiment_id"].unique()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(experiments), size=(n_boot, len(experiments)))
        for method, mdf in sdf.groupby("method", sort=False):
            for metric in metrics:
                series = mdf.set_index("experiment_id")[metric].reindex(experiments).to_numpy(dtype=float)
                if not np.isfinite(series).any():
                    continue
                boots = np.nanmean(series[idx], axis=1)
                rec = dict(zip(by, skey))
                rec.update(method=method, metric=metric, n_experiments=int(np.isfinite(series).sum()),
                           mean=float(np.nanmean(series)), median=float(np.nanmedian(series)),
                           ci_low=float(np.nanpercentile(boots, 2.5)), ci_high=float(np.nanpercentile(boots, 97.5)),
                           share_positive=float(np.nanmean(series > 0)))
                out.append(rec)
    return _order_methods(pd.DataFrame(out))


def monte_carlo_summary(exp: pd.DataFrame, metrics: Iterable[str], by: Iterable[str] = ()) -> pd.DataFrame:
    """Distribution of experiment-level metrics over Monte Carlo replicates."""
    out = []
    for key, g in exp.groupby(list(by) + ["method"], dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            v = g[metric].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            rec = dict(zip(list(by) + ["method"], key))
            rec.update(metric=metric, n_replicates=int(v.size), median=float(np.median(v)),
                       q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)),
                       p2_5=float(np.quantile(v, 0.025)), p97_5=float(np.quantile(v, 0.975)),
                       mean=float(np.mean(v)), mcse=float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan)
            out.append(rec)
    return _order_methods(pd.DataFrame(out))


def paired_method_differences(exp: pd.DataFrame, metric: str, monte_carlo: bool = False,
                              n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Pairwise differences between methods on a metric, paired by experiment."""
    wide = exp.pivot_table(index="experiment_id", columns="method", values=metric, aggfunc="first")
    methods = [m for m in METHOD_ORDER if m in wide.columns]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(n_boot, len(wide)))
    out = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            d = (wide[a] - wide[b]).to_numpy(dtype=float)
            if not np.isfinite(d).any():
                continue
            if monte_carlo:
                v = d[np.isfinite(d)]
                lo, hi = np.quantile(v, [0.025, 0.975])
            else:
                boots = np.nanmean(d[idx], axis=1)
                lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            out.append(dict(method_a=a, method_b=b, metric=metric, mean_difference=float(np.nanmean(d)),
                            median_difference=float(np.nanmedian(d)), interval_low=float(lo), interval_high=float(hi),
                            interval="monte_carlo_2.5_97.5" if monte_carlo else "paired_bootstrap_95",
                            n=int(np.isfinite(d).sum())))
    return pd.DataFrame(out)


# -----------------------------------------------------------------------------
# Specific analyses
# -----------------------------------------------------------------------------


def physician_group_table(df: pd.DataFrame, family: str = "score2", design: str = "in_sample",
                          n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    """Analogue of Table 4: mean score per physician group with bootstrap CIs over physicians."""
    d = df[(df["family"] == family) & (df["design"] == design) & df["physician_group"].notna()]
    rng = np.random.default_rng(seed)
    out = []
    truth = d.drop_duplicates("physician")
    for label, frame, col in [("manual_pairing", truth, "D_tilde"), ("theoretical", truth, "D_star")] + \
            [(m, g, "score") for m, g in d.groupby("method", sort=False)]:
        for grp, gg in frame.groupby("physician_group"):
            v = gg[col].to_numpy(dtype=float)
            boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
            out.append(dict(method=label, physician_group=int(grp), mean=float(v.mean()),
                            ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5))))
    return pd.DataFrame(out)


def decomposition(df: pd.DataFrame, reference: str = "D_star") -> pd.DataFrame:
    """Error decomposition of Reviewer 2 comment 1, per experiment and method.

    D_hat - D_star = (D_hat - D_hat_oracle) + (D_hat_oracle - D_star): the first
    term is the comparability-recovery error, the second the within-stratum
    estimation error. Signed means and mean absolute values are reported.
    """
    d = discordance_rows(main_caliper(df))
    cols = EXPERIMENT_KEYS + ["method", "physician"]
    ins = d[d["design"] == "in_sample"].set_index(cols)
    ora = d[d["design"] == "oracle_eligible"].set_index(cols)["score"].rename("oracle")
    j = ins.join(ora, how="inner")
    j["total"] = j["score"] - j[reference]
    j["comparability"] = j["score"] - j["oracle"]
    j["estimation"] = j["oracle"] - j[reference]
    rows = []
    for key, g in j.reset_index().groupby(EXPERIMENT_KEYS + ["method"], sort=False):
        rec = dict(zip(EXPERIMENT_KEYS + ["method"], key))
        rec["window_size"] = g["window_size"].iloc[0]
        for c in ("total", "comparability", "estimation"):
            rec[f"{c}_bias"] = float(g[c].mean())
            rec[f"{c}_mae"] = float(g[c].abs().mean())
        rec["frac_pairs_both_eligible"] = float(g["frac_pairs_both_eligible"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def ineligible_stratum(df: pd.DataFrame) -> pd.DataFrame:
    """Discordance among ineligible patients against its closed-form target."""
    d = discordance_rows(df[df["design"] == "oracle_ineligible"])
    d = main_caliper(d).assign(err=lambda x: x["score"] - x["D_tilde_ineligible"],
                               err_star=lambda x: x["score"] - x["D_star_ineligible"])
    return d.groupby(EXPERIMENT_KEYS + ["method"], sort=False).agg(
        bias=("err", "mean"), mae=("err", lambda v: v.abs().mean()),
        bias_vs_star=("err_star", "mean")).reset_index()


def stratum_weighted_target(df: pd.DataFrame) -> pd.DataFrame:
    """Marginal target weighting the two strata by the realised eligibility of each physician.

    It is what an analyst would estimate without any stratification, and is
    compared with the in-sample estimate of each method.
    """
    d = discordance_rows(main_caliper(df[df["design"] == "in_sample"])).copy()
    w = d["n_eligible"] / d["n_patients"]
    d["D_weighted"] = w * d["D_tilde"] + (1 - w) * d["D_tilde_ineligible"]
    d["err"] = d["score"] - d["D_weighted"]
    return d.groupby(EXPERIMENT_KEYS + ["method"], sort=False).agg(
        bias_vs_weighted=("err", "mean"), mae_vs_weighted=("err", lambda v: v.abs().mean())).reset_index()


def cross_fitting_effects(exp: pd.DataFrame, metric: str = "bias") -> pd.DataFrame:
    """Effect of the candidate pool and of outcome leakage, paired by experiment.

    pool effect    = fold_restricted - in_sample (smaller pairing pool, same fit)
    leakage effect = kfold - fold_restricted     (held-out fit, same pairing pool)
    transfer       = lopo - in_sample and external - in_sample
    """
    wide = exp.pivot_table(index=["experiment_id", "method"], columns="design", values=metric, aggfunc="first")
    out = []
    contrasts = {"pool_effect": ("fold_restricted", "in_sample"), "leakage_effect": ("kfold", "fold_restricted"),
                 "lopo_vs_in_sample": ("lopo", "in_sample"), "external_vs_in_sample": ("external", "in_sample")}
    rng = np.random.default_rng(BOOT_SEED)
    for method, g in wide.groupby(level="method", sort=False):
        for name, (a, b) in contrasts.items():
            if a not in g.columns or b not in g.columns:
                continue
            d = (g[a] - g[b]).to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            if not d.size:
                continue
            boots = d[rng.integers(0, d.size, size=(N_BOOT, d.size))].mean(axis=1)
            out.append(dict(method=method, contrast=name, metric=metric, n_experiments=int(d.size),
                            mean=float(d.mean()), ci_low=float(np.percentile(boots, 2.5)),
                            ci_high=float(np.percentile(boots, 97.5))))
    return _order_methods(pd.DataFrame(out))


def window_confounding(exp: pd.DataFrame, metric: str = "bias") -> pd.DataFrame:
    """Regression of an experiment-level metric on window size, with and without realised eligibility."""
    import statsmodels.formula.api as smf

    d = exp[(exp["family"] == "progressive") & (exp["design"] == "in_sample")].dropna(subset=[metric])
    out = []
    for method, g in d.groupby("method", sort=False):
        if g["window_size"].nunique() < 2:
            continue
        for label, formula in (("unadjusted", f"{metric} ~ window_size"),
                               ("adjusted", f"{metric} ~ window_size + realised_eligibility")):
            fit = smf.ols(formula, data=g).fit(cov_type="HC3")
            ci = fit.conf_int().loc["window_size"]
            rec = dict(method=method, metric=metric, model=label, n_experiments=int(fit.nobs),
                       slope_window=float(fit.params["window_size"]), ci_low=float(ci[0]), ci_high=float(ci[1]))
            if label == "adjusted":
                ce = fit.conf_int().loc["realised_eligibility"]
                rec.update(slope_eligibility=float(fit.params["realised_eligibility"]),
                           eligibility_ci_low=float(ce[0]), eligibility_ci_high=float(ce[1]))
            out.append(rec)
    return _order_methods(pd.DataFrame(out))


def window_by_eligibility(exp: pd.DataFrame, metric: str = "bias") -> pd.DataFrame:
    """Table 6 stratified by window-size bin and realised-eligibility tertile."""
    d = exp[(exp["family"] == "progressive") & (exp["design"] == "in_sample")].copy()
    if d.empty:
        return d
    d["window_bin"] = pd.cut(d["window_size"], [0, 1, 3, 6, 9], labels=["1", "2-3", "4-6", "7-9"])
    d["eligibility_tertile"] = pd.qcut(d["realised_eligibility"].rank(method="first"), 3, labels=["low", "mid", "high"])
    return d.groupby(["method", "window_bin", "eligibility_tertile"], observed=True).agg(
        n_experiments=(metric, "size"), mean=(metric, "mean")).reset_index()


def caliper_sensitivity(df: pd.DataFrame, reference: str = "D_tilde") -> pd.DataFrame:
    d = discordance_rows(df[df["design"].isin(["in_sample", "oracle_eligible"])])
    exp_ids = d.loc[~np.isclose(d["caliper_percentile"].astype(float), MAIN_CALIPER), "experiment_id"].unique()
    d = d[d["experiment_id"].isin(exp_ids)]
    if d.empty:
        return d
    exp = experiment_metrics(d, reference)
    return exp.groupby(["method", "design", "caliper_percentile"]).agg(
        n_experiments=("bias", "size"), bias=("bias", "mean"), mae=("mae", "mean"),
        retention=("retention", "mean"), unusable_physician_rate=("unusable_physician_rate", "mean"),
        smd_within_mean=("smd_within_mean", "mean")).reset_index()


def coverage_table(exp: pd.DataFrame) -> pd.DataFrame:
    d = exp[(exp["score_type"] == "discordance")]
    cols = ["retention", "n_pairs_median", "unusable_physician_rate", "frac_pairs_both_eligible",
            "frac_pairs_mixed", "frac_pairs_both_ineligible", "smd_within_max", "smd_within_mean",
            "pair_distance_median"]
    return _order_methods(d.groupby(["family", "design", "method"])[cols].mean().reset_index())


def runtime_table(exp: pd.DataFrame) -> pd.DataFrame:
    """Wall-clock time of the in-sample fit and estimation per experiment."""
    d = exp[(exp["design"] == "in_sample") & exp["family"].isin(PRIMARY_FAMILIES)]
    t = d.assign(total_seconds=d["fit_seconds"] + d["estimate_seconds"])
    return _order_methods(t.groupby("method").agg(
        n_experiments=("total_seconds", "size"), median_fit_seconds=("fit_seconds", "median"),
        median_estimate_seconds=("estimate_seconds", "median"), median_total_seconds=("total_seconds", "median"),
        p90_total_seconds=("total_seconds", lambda v: float(np.quantile(v, 0.9)))).reset_index())


def latent_table(df: pd.DataFrame) -> pd.DataFrame:
    """Scenario E1: estimates against the observable and the intrinsic ground truths."""
    d = main_caliper(df[(df["family"] == "latent") & (df["physician_group"] > 1)])
    if d.empty:
        return d
    out = []
    for (gamma, method, design), g in d.groupby(["latent_gamma", "method", "design"], sort=True):
        rec = dict(latent_gamma=gamma, method=method, design=design, score_type=g["score_type"].iloc[0],
                   mean_score=float(g["score"].mean()), D_observable=float(g["D_star"].mean()),
                   D_intrinsic=float(g["D_star_intrinsic"].mean()), D_tilde=float(g["D_tilde"].mean()),
                   spearman_vs_observable=_rank(g["score"].to_numpy(float), g["D_star"].to_numpy(float))[0],
                   spearman_vs_intrinsic=_rank(g["score"].to_numpy(float), g["D_star_intrinsic"].to_numpy(float))[0])
        if rec["score_type"] == "discordance":
            rec["bias_vs_observable"] = float((g["score"] - g["D_star"]).mean())
            rec["bias_vs_intrinsic"] = float((g["score"] - g["D_star_intrinsic"]).mean())
        out.append(rec)
    return _order_methods(pd.DataFrame(out))


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def glmm_specifications(exp: pd.DataFrame) -> pd.DataFrame:
    """Ranking accuracy of the GLMM mean-structure specifications (Reviewer 2 comment 7).

    Spearman and Kendall correlations with D_star per design, and the paired
    difference of each specification with the baseline GLMM (same experiment,
    same design), with paired bootstrap CIs over experiments.
    """
    sub = exp[exp["method"].isin(MODEL_METHODS) & exp["family"].isin(PRIMARY_FAMILIES)]
    if sub.empty:
        return pd.DataFrame()
    level = paired_bootstrap_summary(sub, ["spearman", "kendall"], by=["design"])
    base = sub[sub["method"] == "glmm"].set_index(["experiment_id", "design"])[["spearman", "kendall"]]
    diff = sub[sub["method"] != "glmm"].join(base, on=["experiment_id", "design"], rsuffix="_glmm")
    diff = diff.assign(spearman_minus_glmm=diff["spearman"] - diff["spearman_glmm"],
                       kendall_minus_glmm=diff["kendall"] - diff["kendall_glmm"])
    delta = paired_bootstrap_summary(diff, ["spearman_minus_glmm", "kendall_minus_glmm"], by=["design"])
    return pd.concat([level, delta], ignore_index=True)


def glmm_diagnostics_tables(run_dir: Path, families: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """Calibration, residual and posterior predictive summaries from fitted_parameters/*.json."""
    rows, ppc_rows, curve_rows, lowess_rows = [], [], [], []
    for path in sorted((Path(run_dir) / "fitted_parameters").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        diag = (payload.get("fitted") or {}).get("glmm_diagnostics")
        if not diag:
            continue
        exp_id = path.stem
        family = (families or {}).get(exp_id)
        cal = {}
        for k, v in diag["calibration"].items():
            if k.endswith("_ci95") and isinstance(v, list) and len(v) == 2:
                cal[k.replace("_ci95", "_ci_low")], cal[k.replace("_ci95", "_ci_high")] = v
            elif not isinstance(v, (list, dict)):
                cal[k] = v
        res = diag["residuals"]["by_outcome"]
        ppc = diag["posterior_predictive"]
        rows.append({
            "experiment_id": exp_id, "family": family, **cal,
            "pearson_mean_y0": res["y0"]["mean"], "pearson_mean_y1": res["y1"]["mean"],
            **{k: v for k, v in ppc.items() if k.startswith("coverage_95_")},
            **diag["recalibration"],
        })
        for phys, stats in ppc["by_physician"].items():
            for stat, rec in stats.items():
                ppc_rows.append({"experiment_id": exp_id, "family": family, "physician": phys,
                                 "statistic": stat, **rec})
        for pt in diag["calibration"]["curve_binned"]:
            curve_rows.append({"experiment_id": exp_id, "family": family, **pt})
        for pt in diag["calibration"].get("curve_lowess", []):
            lowess_rows.append({"experiment_id": exp_id, "family": family, **pt})
    out = {"glmm_diagnostics": pd.DataFrame(rows),
           "glmm_posterior_predictive": pd.DataFrame(ppc_rows),
           "glmm_calibration_curves": pd.DataFrame(curve_rows),
           "glmm_calibration_lowess": pd.DataFrame(lowess_rows)}
    diag_df = out["glmm_diagnostics"]
    if not diag_df.empty:
        num = diag_df.select_dtypes("number").columns
        out["glmm_diagnostics_summary"] = (diag_df.groupby("family")[list(num)]
                                           .agg(["mean", "median", "min", "max"]).reset_index())
        out["glmm_diagnostics_summary"].columns = ["_".join(c).rstrip("_") for c in
                                                   out["glmm_diagnostics_summary"].columns.to_flat_index()]
    return out


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    main = main_caliper(df)
    exp_tilde = experiment_metrics(main, "D_tilde")
    exp_star = experiment_metrics(main, "D_star")
    err_metrics = ["bias", "mae", "rmse", "median_ae", "max_ae", "calibration_intercept", "calibration_slope"]
    tables: dict[str, pd.DataFrame] = {
        "experiment_metrics_vs_D_tilde": exp_tilde,
        "experiment_metrics_vs_D_star": exp_star,
        "noise_floor": noise_floor(main),
    }
    prim = exp_tilde[exp_tilde["family"].isin(PRIMARY_FAMILIES) & (exp_tilde["design"] == "in_sample")]
    prog = prim[prim["family"] == "progressive"]
    tables["table5_progressive_vs_D_tilde"] = paired_bootstrap_summary(prog, err_metrics + ["spearman"])
    tables["table5_progressive_vs_D_star"] = paired_bootstrap_summary(
        exp_star[(exp_star["family"] == "progressive") & (exp_star["design"] == "in_sample")], err_metrics)
    if not prog.empty:
        binned = prog.assign(window_bin=pd.cut(prog["window_size"], [0, 1, 3, 6, 9],
                                               labels=["1", "2-3", "4-6", "7-9"]).astype(str))
        tables["table6_by_window"] = paired_bootstrap_summary(binned, ["bias", "mae", "calibration_slope"], by=["window_bin"])
        tables["table7_by_pass"] = paired_bootstrap_summary(prog, ["bias", "mae"], by=["pass_index"])
        tables["window_confounding_bias"] = window_confounding(exp_tilde, "bias")
        tables["window_confounding_mae"] = window_confounding(exp_tilde, "mae")
        tables["window_by_eligibility"] = window_by_eligibility(exp_tilde, "bias")
    if (df["family"] == "score2").any():
        tables["table4_score2_groups"] = physician_group_table(main)
        tables["score2_metrics"] = exp_tilde[exp_tilde["family"] == "score2"]
    tables["decomposition"] = decomposition(df)
    tables["decomposition_summary"] = paired_bootstrap_summary(
        tables["decomposition"].assign(design="in_sample"),
        ["total_bias", "comparability_bias", "estimation_bias", "total_mae", "comparability_mae", "estimation_mae"],
        by=["family"])
    tables["ineligible_stratum"] = ineligible_stratum(df)
    tables["stratum_weighted_target"] = stratum_weighted_target(df)
    for metric in ("bias", "mae"):
        # Unsupervised methods are kept: their leakage effect is 0 by construction (a check).
        tables[f"cross_fitting_{metric}"] = cross_fitting_effects(exp_tilde, metric)
    tables["cross_fitting_designs_summary"] = paired_bootstrap_summary(
        exp_tilde[exp_tilde["family"].isin(PRIMARY_FAMILIES)], ["bias", "mae", "spearman"], by=["design"])
    tables["coverage"] = coverage_table(exp_tilde)
    tables["caliper_sensitivity"] = caliper_sensitivity(df)
    tables["runtime"] = runtime_table(exp_tilde)
    tables["glmm_specifications"] = glmm_specifications(exp_tilde)
    cont = exp_star[(exp_star["family"] == "continuous")]
    if not cont.empty:
        tables["continuous_monte_carlo"] = monte_carlo_summary(
            cont, ["spearman", "kendall", "bias", "mae", "calibration_slope"], by=["design"])
        tables["continuous_paired_spearman"] = paired_method_differences(
            cont[cont["design"] == "in_sample"], "spearman", monte_carlo=True)
    sens = exp_tilde[exp_tilde["family"].str.startswith("sensitivity")]
    if not sens.empty:
        tables["sensitivity_metrics"] = sens
    lat = latent_table(df)
    if not lat.empty:
        tables["latent_E1"] = lat
    return tables


def key_numbers(tables: dict[str, pd.DataFrame]) -> dict:
    """Values quoted in the text, gathered in one place."""
    out: dict = {}
    t5 = tables.get("table5_progressive_vs_D_tilde")
    if t5 is not None and not t5.empty:
        for metric in ("bias", "mae", "calibration_slope"):
            sub = t5[t5["metric"] == metric]
            out[f"progressive_{metric}"] = {r.method: [round(r.mean, 4), round(r.ci_low, 4), round(r.ci_high, 4)]
                                            for r in sub.itertuples()}
    nf = tables.get("noise_floor")
    if nf is not None and not nf.empty:
        out["noise_floor_mae_by_family"] = nf.groupby("family")["mae"].mean().round(4).to_dict()
    cmc = tables.get("continuous_monte_carlo")
    if cmc is not None and not cmc.empty:
        sub = cmc[(cmc["metric"] == "spearman") & (cmc["design"] == "in_sample")]
        out["continuous_spearman_median_p2.5_p97.5"] = {
            r.method: [round(r.median, 3), round(r.p2_5, 3), round(r.p97_5, 3)] for r in sub.itertuples()}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing benchmark_long.csv.gz.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load(args.run)
    tables = build_tables(df)
    tables.update(glmm_diagnostics_tables(
        args.run, df.drop_duplicates("experiment_id").set_index("experiment_id")["family"].to_dict()))
    out_dir = args.run / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table is None or table.empty:
            LOGGER.info("  %-38s (empty, skipped)", name)
            continue
        table.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")
        LOGGER.info("  %-38s %6d rows", name, len(table))
    with (out_dir / "key_numbers.json").open("w", encoding="utf-8") as fh:
        json.dump(key_numbers(tables), fh, indent=2)
    LOGGER.info("Tables written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
