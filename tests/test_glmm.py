"""Bayesian binomial GLMM, its Pearson-residual score, its variants and its diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from src.benchmark.estimators import GLMMEstimator, GradientBoostingOffsetEstimator, _pearson_scores
from src.benchmark.glmm_diagnostics import _wilson, calibration_measures, glmm_diagnostics


@pytest.fixture(scope="module")
def fitted_glmm(score2_cohort):
    est = GLMMEstimator().fit(score2_cohort, np.arange(score2_cohort.n))
    est.prepare(score2_cohort)
    return est


def test_pearson_score_is_the_mean_squared_pearson_residual(score2_cohort, fitted_glmm):
    idx = np.arange(score2_cohort.n)
    p = fitted_glmm.predict_proba(idx)
    out = fitted_glmm.estimate(score2_cohort, idx)
    assert {e.physician for e in out} == set(np.unique(score2_cohort.physician))
    for e in out:
        m = score2_cohort.physician == e.physician
        r = (score2_cohort.y[m] - p[m]) / np.sqrt(p[m] * (1 - p[m]))
        assert e.score == pytest.approx(np.mean(r ** 2), rel=1e-6)
        assert e.n_pairs == 0


def test_random_intercepts_are_estimated_for_every_physician(score2_cohort, fitted_glmm):
    assert set(fitted_glmm.random_effects_) == set(np.unique(score2_cohort.physician))
    assert len(fitted_glmm.coef_) == 1 + score2_cohort.X.shape[1]


def test_pearson_scores_skip_small_panels(score2_cohort):
    p = np.full(score2_cohort.n, 0.5)
    assert _pearson_scores(score2_cohort, np.arange(score2_cohort.n), p, min_patients=10_000) == []
    out = _pearson_scores(score2_cohort, np.arange(score2_cohort.n), p, min_patients=1)
    assert all(e.score == pytest.approx(1.0) for e in out)  # (y - 0.5)^2 / 0.25 = 1


def test_oracle_variant_requires_the_eligibility_indicator(score2_cohort):
    from src.benchmark.estimators import Cohort

    blind = Cohort(X=score2_cohort.X, y=score2_cohort.y, physician=score2_cohort.physician,
                   feature_names=score2_cohort.feature_names)
    with pytest.raises(ValueError):
        GLMMEstimator("oracle").fit(blind, np.arange(blind.n))
    with pytest.raises(ValueError):
        GLMMEstimator("unknown")


@pytest.mark.parametrize("variant", ["oracle", "splines", "interactions"])
def test_glmm_variants_fit_and_score(score2_cohort, variant):
    est = GLMMEstimator(variant).fit(score2_cohort, np.arange(score2_cohort.n))
    est.prepare(score2_cohort)
    scores = [e.score for e in est.estimate(score2_cohort, np.arange(score2_cohort.n))]
    assert len(scores) == len(np.unique(score2_cohort.physician))
    assert all(np.isfinite(s) and s > 0 for s in scores)
    if variant == "interactions":
        assert len(est.coef_) == 1 + 9 + 36  # 45 fixed effects plus the intercept


def test_oracle_glmm_ranks_the_five_groups(score2_cohort):
    from scipy.stats import spearmanr

    from src.synthetic_generator import five_group_physician_parameters

    est = GLMMEstimator("oracle").fit(score2_cohort, np.arange(score2_cohort.n))
    est.prepare(score2_cohort)
    out = est.estimate(score2_cohort, np.arange(score2_cohort.n))
    params = five_group_physician_parameters(10)
    d_star = [2 * params[e.physician]["p_high"] * (1 - params[e.physician]["p_high"]) for e in out]
    assert spearmanr([e.score for e in out], d_star).statistic > 0.8


def test_gradient_boosting_reference(score2_cohort):
    est = GradientBoostingOffsetEstimator().fit(score2_cohort, np.arange(score2_cohort.n))
    est.prepare(score2_cohort)
    out = est.estimate(score2_cohort, np.arange(score2_cohort.n))
    assert len(out) == len(np.unique(score2_cohort.physician))
    assert set(est.offsets_) == set(np.unique(score2_cohort.physician))


def test_wilson_interval():
    lo, hi = _wilson(50, 100)
    assert lo < 0.5 < hi and lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = _wilson(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.2
    assert all(np.isnan(_wilson(0, 0)))


def test_calibration_measures_on_well_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = rng.binomial(1, p)
    m = calibration_measures(y, p, n_boot=0)
    assert m["calibration_in_the_large"] == pytest.approx(0.0, abs=0.05)
    assert m["calibration_slope"] == pytest.approx(1.0, abs=0.05)
    assert m["ici"] < 0.02
    assert abs(m["spiegelhalter_z"]) < 3
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    # Murphy decomposition: Brier = reliability - resolution + uncertainty (binned approximation).
    assert m["brier"] == pytest.approx(m["brier_reliability"] - m["brier_resolution"] + m["brier_uncertainty"],
                                       abs=5e-3)


def test_calibration_measures_detect_overconfidence():
    rng = np.random.default_rng(1)
    p_true = rng.uniform(0.2, 0.8, 20_000)
    y = rng.binomial(1, p_true)
    lp = np.log(p_true / (1 - p_true))
    p_over = 1 / (1 + np.exp(-2 * lp))  # logits doubled: overconfident predictions
    m = calibration_measures(y, p_over, n_boot=0)
    assert m["calibration_slope"] == pytest.approx(0.5, abs=0.05)


def test_glmm_diagnostics_run_end_to_end(score2_cohort, fitted_glmm):
    out = glmm_diagnostics(fitted_glmm, score2_cohort, seed=0, n_boot=20)
    assert isinstance(out, dict) and out
