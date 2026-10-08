# -*- coding: utf-8 -*-
"""Calibration and residual diagnostics of the GLMM (Reviewer 2 comments 8 and 9).

``glmm_diagnostics`` is applied to the baseline GLMM fitted on all rows of a
cohort (in-sample design). It returns a JSON-serialisable dictionary with:

- calibration: calibration-in-the-large (intercept of a logistic model of the
  outcome with the logit of the fitted probability as offset), calibration
  slope (coefficient of that logit), integrated calibration index with E50
  and E90 (Austin and Steyerberg 2019: absolute difference between a lowess
  calibration curve and the identity, averaged over patients), Brier score
  with its Murphy decomposition (reliability, resolution, uncertainty, ten
  equal-frequency bins) and Spiegelhalter's z;
- curves for figures: binned observed and predicted frequencies (Wilson 95%
  intervals per bin) and the lowess curve with a pointwise 95% percentile
  bootstrap band (B = 2000 resamples of patients); the same resamples give
  percentile intervals for the calibration intercept, slope and ICI. In the
  resamples the lowess uses linear interpolation between points closer than
  0.005 on the probability scale (``delta``), which changes the curve by less
  than 1e-4 and makes the bootstrap affordable; the point estimates use the
  exact lowess;
- residuals: Pearson residual summaries stratified by outcome (a Bernoulli
  Pearson residual takes two values for a given fitted probability, so its
  distribution is a two-component mixture by construction), residuals by
  bins of fitted probability, and per-physician residual summaries;
- an approximate posterior predictive check: replicated outcomes are drawn
  from the variational posterior (fixed effects and physician intercepts
  from independent normals with the variational means and SDs), and the
  observed physician-level prescription rate and all-pairs discordance are
  compared with their predictive distributions. Because the physician
  intercept absorbs the physician's prescribing level, the rate check passes
  almost by construction; the informative statistic is the discordance among
  covariate-matched pairs (each patient and its nearest neighbour within the
  physician panel, standardised covariates), which the model must reproduce
  from its mean structure;
- the effect of recalibration (intercept and slope) on the physician ranking.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy.stats import norm, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.benchmark.estimators import GLMM_EPSILON, Cohort, GLMMEstimator, zscore

N_PPC_DRAWS = 200
LOWESS_FRAC = 0.75
N_BINS_CURVE = 20
N_BINS_BRIER = 10
N_BOOT_CALIBRATION = 2000
LOWESS_DELTA_BOOT = 0.005
N_GRID_CURVE = 50


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, GLMM_EPSILON, 1 - GLMM_EPSILON)
    return np.log(p / (1 - p))


def _wilson(k: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / den
    return float(centre - half), float(centre + half)


def _logistic_intercept_slope(y: np.ndarray, lp: np.ndarray) -> tuple[float, float, float]:
    """Calibration-in-the-large (offset model) and logistic recalibration intercept and slope."""
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return float(citl.params[0]), float(fit.params[0]), float(fit.params[1])


def calibration_bootstrap(y: np.ndarray, p: np.ndarray, grid: np.ndarray, rng: np.random.Generator,
                          n_boot: int = N_BOOT_CALIBRATION) -> dict[str, Any]:
    """Percentile bootstrap (resampling patients) of the lowess curve, CITL, slope and ICI."""
    n = len(y)
    lp = _logit(p)
    curves = np.empty((n_boot, grid.size))
    citl = np.empty(n_boot)
    slope = np.empty(n_boot)
    ici = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        fit = lowess(yb, pb, frac=LOWESS_FRAC, it=0, delta=LOWESS_DELTA_BOOT, return_sorted=True)
        xs, fs = fit[:, 0], np.clip(fit[:, 1], 0, 1)
        curves[b] = np.interp(grid, xs, fs)
        ici[b] = np.mean(np.abs(np.interp(pb, xs, fs) - pb))
        try:
            citl[b], _, slope[b] = _logistic_intercept_slope(yb, lp[idx])
        except Exception:
            citl[b], slope[b] = np.nan, np.nan

    def _ci(v: np.ndarray) -> list[float]:
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {
        "n_boot": n_boot, "resampling_unit": "patient",
        "calibration_in_the_large_ci95": _ci(citl), "calibration_slope_ci95": _ci(slope), "ici_ci95": _ci(ici),
        "band_low": np.nanpercentile(curves, 2.5, axis=0), "band_high": np.nanpercentile(curves, 97.5, axis=0),
    }


def calibration_measures(y: np.ndarray, p: np.ndarray, rng: Optional[np.random.Generator] = None,
                         n_boot: int = N_BOOT_CALIBRATION) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), GLMM_EPSILON, 1 - GLMM_EPSILON)
    y = np.asarray(y, dtype=float)
    lp = _logit(p)
    citl, intercept, slope = _logistic_intercept_slope(y, lp)
    smooth = lowess(y, p, frac=LOWESS_FRAC, it=0, return_sorted=False)
    diff = np.abs(np.clip(smooth, 0, 1) - p)
    brier = float(np.mean((p - y) ** 2))
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, N_BINS_BRIER)
    ybar = y.mean()
    reliability = sum(len(b) * (p[b].mean() - y[b].mean()) ** 2 for b in bins) / len(y)
    resolution = sum(len(b) * (y[b].mean() - ybar) ** 2 for b in bins) / len(y)
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = float(num / den) if den > 0 else float("nan")
    curve_bins = np.array_split(order, N_BINS_CURVE)
    grid_idx = order[np.linspace(0, len(order) - 1, N_GRID_CURVE).astype(int)]
    grid = p[grid_idx]
    boot = calibration_bootstrap(y, p, grid, rng if rng is not None else np.random.default_rng(0), n_boot) \
        if n_boot > 0 else None
    curve_lowess = []
    for k, i in enumerate(grid_idx):
        rec = {"predicted": float(p[i]), "observed": float(np.clip(smooth[i], 0, 1))}
        if boot is not None:
            rec.update(band_low=float(boot["band_low"][k]), band_high=float(boot["band_high"][k]))
        curve_lowess.append(rec)
    binned = []
    for b in curve_bins:
        lo, hi = _wilson(float(y[b].sum()), int(len(b)))
        binned.append({"predicted": float(p[b].mean()), "observed": float(y[b].mean()), "n": int(len(b)),
                       "observed_low": lo, "observed_high": hi})
    out_boot = {k: v for k, v in (boot or {}).items() if k not in ("band_low", "band_high")}
    return {
        "calibration_in_the_large": citl,
        "calibration_slope": slope,
        "calibration_intercept_with_slope": intercept,
        **out_boot,
        "ici": float(np.mean(diff)), "e50": float(np.median(diff)), "e90": float(np.quantile(diff, 0.9)),
        "emax": float(np.max(diff)),
        "brier": brier, "brier_reliability": float(reliability), "brier_resolution": float(resolution),
        "brier_uncertainty": float(ybar * (1 - ybar)),
        "spiegelhalter_z": z, "spiegelhalter_p": float(2 * norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
        "curve_binned": binned,
        "curve_lowess": curve_lowess,
        "_recalibration": (intercept, slope),
    }


def residual_summaries(y: np.ndarray, p: np.ndarray, physician: np.ndarray) -> dict[str, Any]:
    p = np.clip(p, GLMM_EPSILON, 1 - GLMM_EPSILON)
    r = (y - p) / np.sqrt(p * (1 - p))
    q = [0.05, 0.25, 0.5, 0.75, 0.95]

    def _summ(v: np.ndarray) -> dict[str, float]:
        return {"n": int(v.size), "mean": float(np.mean(v)), "sd": float(np.std(v)),
                **{f"q{int(100 * k):02d}": float(np.quantile(v, k)) for k in q}}

    order = np.argsort(p, kind="stable")
    return {
        "by_outcome": {"y0": _summ(r[y == 0]), "y1": _summ(r[y == 1])},
        "all": _summ(r),
        "by_fitted_bin": [{"fitted": float(p[b].mean()), "mean_residual": float(r[b].mean()),
                           "mean_squared_residual": float(np.mean(r[b] ** 2)), "n": int(len(b))}
                          for b in np.array_split(order, N_BINS_CURVE)],
        "by_physician": {str(ph): {"n": int((physician == ph).sum()), "mean": float(r[physician == ph].mean()),
                                   "mean_squared": float(np.mean(r[physician == ph] ** 2))}
                         for ph in np.unique(physician)},
    }


def _all_pairs_discordance(y: np.ndarray) -> float:
    n = len(y)
    k = float(np.sum(y))
    return float(2 * k * (n - k) / (n * (n - 1))) if n > 1 else float("nan")


def posterior_predictive_check(est: GLMMEstimator, cohort: Cohort, rng: np.random.Generator,
                               n_draws: int = N_PPC_DRAWS) -> dict[str, Any]:
    idx = np.arange(cohort.n)
    exog = est._exog[idx]
    phys = cohort.physician[idx]
    labels = sorted(est.random_effects_)
    pos = {p: i for i, p in enumerate(labels)}
    re_mean = np.array([est.random_effects_[p] for p in labels])
    re_sd = np.array([est.random_effects_sd_[p] for p in labels])
    col = np.array([pos[p] for p in phys])
    z = zscore(cohort.X)
    nn: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in labels:
        ip = np.where(phys == p)[0]
        d = cdist(z[ip], z[ip])
        np.fill_diagonal(d, np.inf)
        nn[p] = (ip, ip[np.argmin(d, axis=1)])
    stats_rep = {p: {"rate": [], "discordance": [], "matched_pair_discordance": []} for p in labels}
    for _ in range(n_draws):
        beta = rng.normal(est.coef_, est.coef_sd_)
        u = rng.normal(re_mean, re_sd)
        prob = 1.0 / (1.0 + np.exp(-(exog @ beta + u[col])))
        y_rep = rng.binomial(1, prob)
        for p in labels:
            m = phys == p
            stats_rep[p]["rate"].append(y_rep[m].mean())
            stats_rep[p]["discordance"].append(_all_pairs_discordance(y_rep[m]))
            ia, ib = nn[p]
            stats_rep[p]["matched_pair_discordance"].append(float(np.mean(y_rep[ia] != y_rep[ib])))
    out: dict[str, Any] = {"n_draws": n_draws, "by_physician": {}}
    stats = ("rate", "discordance", "matched_pair_discordance")
    covered = {k: [] for k in stats}
    for p in labels:
        m = phys == p
        ia, ib = nn[p]
        obs = {"rate": float(cohort.y[m].mean()), "discordance": _all_pairs_discordance(cohort.y[m]),
               "matched_pair_discordance": float(np.mean(cohort.y[ia] != cohort.y[ib]))}
        rec = {}
        for stat in stats:
            rep = np.asarray(stats_rep[p][stat])
            lo, hi = np.quantile(rep, [0.025, 0.975])
            rec[stat] = {"observed": obs[stat], "predictive_mean": float(rep.mean()),
                         "predictive_low": float(lo), "predictive_high": float(hi),
                         "ppp": float(np.mean(rep >= obs[stat]))}
            covered[stat].append(lo <= obs[stat] <= hi)
        out["by_physician"][p] = rec
    for stat in stats:
        out[f"coverage_95_{stat}"] = float(np.mean(covered[stat]))
    return out


def glmm_diagnostics(est: GLMMEstimator, cohort: Cohort, truth: Optional[dict[str, float]] = None,
                     seed: int = 0, n_boot: int = N_BOOT_CALIBRATION) -> dict[str, Any]:
    """All diagnostics for a GLMM fitted and prepared on ``cohort`` (all rows)."""
    idx = np.arange(cohort.n)
    p = est.predict_proba(idx)
    y = cohort.y
    calib = calibration_measures(y, p, np.random.default_rng(seed + 1), n_boot)
    a, b = calib.pop("_recalibration")
    p_recal = 1.0 / (1.0 + np.exp(-(a + b * _logit(p))))

    def _scores(prob: np.ndarray) -> dict[str, float]:
        prob = np.clip(prob, GLMM_EPSILON, 1 - GLMM_EPSILON)
        r2 = (y - prob) ** 2 / (prob * (1 - prob))
        return {ph: float(r2[cohort.physician == ph].mean()) for ph in np.unique(cohort.physician)}

    s0, s1 = _scores(p), _scores(p_recal)
    phys = sorted(s0)
    recal = {"spearman_original_vs_recalibrated": float(spearmanr([s0[k] for k in phys], [s1[k] for k in phys]).statistic)}
    if truth:
        t = [truth[k] for k in phys]
        recal["spearman_original_vs_truth"] = float(spearmanr([s0[k] for k in phys], t).statistic)
        recal["spearman_recalibrated_vs_truth"] = float(spearmanr([s1[k] for k in phys], t).statistic)
    return {
        "calibration": calib,
        "residuals": residual_summaries(y, p, cohort.physician),
        "posterior_predictive": posterior_predictive_check(est, cohort, np.random.default_rng(seed)),
        "recalibration": recal,
    }
