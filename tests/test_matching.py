"""Distances, pairing algorithm, caliper and matching-based discordance estimates."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from src.analysis.matching_core import _optimal_one_to_one_pairs, _per_physician_caliper_threshold
from src.benchmark.estimators import (
    Cohort,
    EuclideanEstimator,
    GeneticMahalanobisEstimator,
    LearnedWeightsEstimator,
    LPAGuidedEstimator,
    MahalanobisEstimator,
    MutualInformationEstimator,
    RandomForestProximityEstimator,
    _whitening_matrix,
    build_estimators,
    estimate_matching,
    zscore,
)


def _random_distance(n: int, seed: int = 0) -> np.ndarray:
    pts = np.random.default_rng(seed).normal(size=(n, 2))
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    return d


# ----------------------------------------------------------------------------- pairing


def test_one_to_one_pairs_use_each_patient_at_most_once():
    d = _random_distance(41)
    pairs = _optimal_one_to_one_pairs(d)
    used = [i for p in pairs for i in p]
    assert len(used) == len(set(used))
    assert all(i != j for i, j in pairs)
    assert len(pairs) <= 41 // 2


def test_one_to_one_pairs_recover_obvious_pairs():
    # Three well-separated couples of points.
    pts = np.array([[0, 0], [0, 0.1], [10, 0], [10, 0.1], [0, 10], [0.1, 10]], dtype=float)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    assert sorted(_optimal_one_to_one_pairs(d)) == [(0, 1), (2, 3), (4, 5)]


def test_caliper_threshold_is_the_percentile_of_off_diagonal_distances():
    d = _random_distance(30)
    iu = np.triu_indices(30, k=1)
    assert _per_physician_caliper_threshold(d, 25.0) == pytest.approx(np.percentile(d[iu], 25.0))
    assert np.isinf(_per_physician_caliper_threshold(d, 100.0))
    with pytest.raises(ValueError):
        _per_physician_caliper_threshold(d, 120.0)


def test_caliper_discards_distant_pairs():
    d = _random_distance(40)
    thr = _per_physician_caliper_threshold(d, 25.0)
    pairs = _optimal_one_to_one_pairs(d, caliper_threshold=thr)
    assert pairs and all(d[i, j] <= thr for i, j in pairs)
    assert len(pairs) <= len(_optimal_one_to_one_pairs(d))


# ----------------------------------------------------------------------------- distances


def test_euclidean_distance_is_scaled_by_the_number_of_covariates(toy_cohort):
    est = EuclideanEstimator()
    est.prepare(toy_cohort)
    idx = np.arange(10)
    Z = zscore(toy_cohort.X)
    expected = cdist(Z[idx], Z[idx]) / np.sqrt(Z.shape[1])
    assert np.allclose(est.distance_matrix(idx), expected)


def test_whitened_euclidean_distance_equals_mahalanobis(toy_cohort):
    maha = MahalanobisEstimator()
    maha.prepare(toy_cohort)
    idx = np.arange(25)
    Z = zscore(toy_cohort.X)
    keep, W = _whitening_matrix(Z)
    Zw = Z[:, keep] @ W
    assert np.allclose(np.cov(Zw, rowvar=False), np.eye(Zw.shape[1]), atol=1e-8)
    assert np.allclose(maha.distance_matrix(idx), cdist(Zw[idx], Zw[idx]), atol=1e-8)


def test_legacy_genetic_weights_cancel_exactly(toy_cohort):
    """First-submission variant: any weights give the Mahalanobis distance (Appendix E)."""
    maha = MahalanobisEstimator()
    maha.prepare(toy_cohort)
    idx = np.arange(30)
    for weights in ([1.0, 1.0, 1.0], [0.1, 3.0, 7.5]):
        legacy = GeneticMahalanobisEstimator(variant="legacy")
        legacy.weights_ = np.asarray(weights)
        legacy.prepare(toy_cohort)
        assert np.allclose(legacy.distance_matrix(idx), maha.distance_matrix(idx), atol=1e-8)


def test_genmatch_weights_change_the_distance(toy_cohort):
    idx = np.arange(30)
    est = GeneticMahalanobisEstimator(variant="genmatch")
    est.weights_ = np.ones(3) / np.sqrt(3)  # uniform weights: Mahalanobis up to a constant
    est.prepare(toy_cohort)
    maha = MahalanobisEstimator()
    maha.prepare(toy_cohort)
    ratio = est.distance_matrix(idx)[np.triu_indices(30, 1)] / maha.distance_matrix(idx)[np.triu_indices(30, 1)]
    assert np.allclose(ratio, ratio[0])
    est.weights_ = np.sqrt(np.array([0.9, 0.05, 0.05]))
    est.prepare(toy_cohort)
    ratio = est.distance_matrix(idx)[np.triu_indices(30, 1)] / maha.distance_matrix(idx)[np.triu_indices(30, 1)]
    assert not np.allclose(ratio, ratio[0])


def test_unknown_genetic_variant_is_rejected():
    with pytest.raises(ValueError):
        GeneticMahalanobisEstimator(variant="other")


def test_outcome_informed_weights_concentrate_on_the_rule_covariate(toy_cohort):
    # y depends only on the first covariate in the toy cohort.
    for est in (LearnedWeightsEstimator(), MutualInformationEstimator()):
        est.fit(toy_cohort, np.arange(toy_cohort.n))
        w = est.weights_
        assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
        assert np.argmax(w) == 0


def test_rf_proximity_dissimilarity_is_bounded(toy_cohort):
    est = RandomForestProximityEstimator().fit(toy_cohort, np.arange(toy_cohort.n))
    est.prepare(toy_cohort)
    d = est.distance_matrix(np.arange(20))
    off = d[~np.eye(20, dtype=bool)]
    assert np.all(off >= 0) and np.all(off <= 1)
    assert np.allclose(d, d.T)


def test_lpa_distance_combines_two_rescaled_terms(toy_cohort):
    est = LPAGuidedEstimator()
    est.prepare(toy_cohort)
    d = est.distance_matrix(np.arange(40))
    assert 2 <= est.n_components_ <= 10
    assert np.all(d >= 0) and d.max() <= 1.0 + 1e-12
    assert np.allclose(d, d.T)


# ----------------------------------------------------------------------------- estimation


def _brute_force_score(cohort: Cohort, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([cohort.y[i] != cohort.y[j] for i, j in zip(a, b)]))


@pytest.mark.parametrize("estimator", [EuclideanEstimator(), MahalanobisEstimator()])
def test_matching_estimate_is_the_discordance_of_within_physician_pairs(toy_cohort, estimator):
    estimator.prepare(toy_cohort)
    out = estimate_matching(estimator, toy_cohort, np.arange(toy_cohort.n))
    assert [e.physician for e in out] == ["A", "B"]
    for e in out:
        assert e.n_pairs == len(e.pair_index_a) > 0
        assert np.all(toy_cohort.physician[e.pair_index_a] == e.physician)
        assert np.all(toy_cohort.physician[e.pair_index_b] == e.physician)
        assert e.score == pytest.approx(_brute_force_score(toy_cohort, e.pair_index_a, e.pair_index_b))
        assert np.all(e.pair_distance <= e.caliper_threshold + 1e-12)
        assert e.n_patients_paired == 2 * e.n_pairs


def test_constant_outcome_gives_zero_discordance(toy_cohort):
    c = Cohort(X=toy_cohort.X, y=np.ones(toy_cohort.n, dtype=int), physician=toy_cohort.physician,
               feature_names=toy_cohort.feature_names)
    est = EuclideanEstimator()
    est.prepare(c)
    assert all(e.score == 0.0 for e in estimate_matching(est, c, np.arange(c.n)))


def test_candidate_pool_and_groups_restrict_the_pairs(toy_cohort):
    est = EuclideanEstimator()
    est.prepare(toy_cohort)
    pool = np.arange(0, toy_cohort.n, 2)  # even rows only (an oracle stratum, for instance)
    groups = np.arange(toy_cohort.n) % 3  # fold-like labels
    out = estimate_matching(est, toy_cohort, pool, groups=groups, min_patients=10)
    for e in out:
        assert np.all(np.isin(e.pair_index_a, pool)) and np.all(np.isin(e.pair_index_b, pool))
        assert np.all(groups[e.pair_index_a] == groups[e.pair_index_b])


def test_small_panels_are_skipped(toy_cohort):
    est = EuclideanEstimator()
    est.prepare(toy_cohort)
    assert estimate_matching(est, toy_cohort, np.arange(toy_cohort.n), min_patients=61) == []


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_genetic_matching_is_with_replacement(toy_cohort):
    est = GeneticMahalanobisEstimator(variant="genmatch")
    est.weights_ = np.ones(3) / np.sqrt(3)
    est.prepare(toy_cohort)
    out = estimate_matching(est, toy_cohort, np.arange(toy_cohort.n), caliper_percentile=None)
    for e in out:
        assert e.n_pairs == e.n_candidates  # every patient keeps its nearest neighbour without a caliper
        assert e.n_patients_paired == e.n_pairs


def test_estimators_never_need_the_eligibility_indicator(toy_cohort):
    assert toy_cohort.eligible is None
    for est in build_estimators(include_genetic=False):
        if est.kind != "matching":
            continue
        est.fit(toy_cohort, np.arange(toy_cohort.n))
        est.prepare(toy_cohort)
        assert estimate_matching(est, toy_cohort, np.arange(toy_cohort.n))


def test_genetic_weight_optimisation_runs_and_returns_normalised_weights(toy_cohort):
    est = GeneticMahalanobisEstimator(variant="genmatch", workers=1).fit(toy_cohort, np.arange(toy_cohort.n))
    assert est.raw_weights_.shape == (3,)
    assert np.all((est.raw_weights_ >= 0) & (est.raw_weights_ <= 10))
    assert np.sum(est.weights_ ** 2) == pytest.approx(1.0)
