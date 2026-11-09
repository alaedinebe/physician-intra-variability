"""Training designs: folds, cross-fitting, leave-one-physician-out, oracle strata."""
from __future__ import annotations

import numpy as np
import pytest

from src.benchmark.designs import DESIGNS, equivalent_design, run_design, stratified_folds, supports
from src.benchmark.estimators import Cohort, EuclideanEstimator, GLMMEstimator, LearnedWeightsEstimator


class SpyLearnedWeights(LearnedWeightsEstimator):
    """Records, at class level, the rows used for each fit (run_design works on a deep copy)."""

    fit_calls: list = []

    def fit(self, cohort, fit_idx):
        SpyLearnedWeights.fit_calls.append(np.asarray(fit_idx).copy())
        return super().fit(cohort, fit_idx)


def test_folds_are_balanced_within_physician_and_reproducible(score2_cohort):
    folds = stratified_folds(score2_cohort.physician, k=5, seed=1)
    for p in np.unique(score2_cohort.physician):
        counts = np.bincount(folds[score2_cohort.physician == p], minlength=5)
        assert counts.max() - counts.min() <= 1
    assert np.array_equal(folds, stratified_folds(score2_cohort.physician, k=5, seed=1))


def test_unsupervised_designs_collapse_by_construction():
    est = EuclideanEstimator()
    assert equivalent_design(est, "kfold") == "fold_restricted"
    assert equivalent_design(est, "lopo") == "in_sample"
    assert equivalent_design(est, "external") == "in_sample"
    assert equivalent_design(LearnedWeightsEstimator(), "kfold") == "kfold"


def test_glmm_supports_only_in_sample_and_kfold():
    glmm = GLMMEstimator()
    assert {d for d in DESIGNS if supports(glmm, d)} == {"in_sample", "kfold"}
    assert all(supports(EuclideanEstimator(), d) for d in DESIGNS)
    with pytest.raises(ValueError):
        run_design(glmm, Cohort(X=np.zeros((4, 1)), y=[0, 1, 0, 1], physician=list("aabb"),
                                feature_names=["x"]), "lopo")


def test_kfold_never_fits_on_the_held_out_fold(score2_cohort):
    folds = stratified_folds(score2_cohort.physician, k=5, seed=0)
    SpyLearnedWeights.fit_calls = []
    res = run_design(SpyLearnedWeights(), score2_cohort, "kfold", folds=folds)
    assert res.n_fits == 5 and len(SpyLearnedWeights.fit_calls) == 5
    held_out = []
    for idx in SpyLearnedWeights.fit_calls:
        missing = np.unique(folds[np.setdiff1d(np.arange(score2_cohort.n), idx)])
        assert missing.size == 1  # each fit excludes exactly one whole fold
        held_out.append(int(missing[0]))
        assert not np.any(folds[idx] == missing[0])
    assert sorted(held_out) == [0, 1, 2, 3, 4]


def test_kfold_pairs_are_formed_within_held_out_folds(score2_cohort):
    folds = stratified_folds(score2_cohort.physician, k=5, seed=0)
    res = run_design(LearnedWeightsEstimator(), score2_cohort, "kfold", folds=folds)
    assert {e.physician for e in res.estimates} == set(np.unique(score2_cohort.physician))
    for e in res.estimates:
        assert len(e.pair_index_a) > 0
        assert np.all(folds[e.pair_index_a] == folds[e.pair_index_b])
        y = score2_cohort.y
        assert e.score == pytest.approx(np.mean(y[e.pair_index_a] != y[e.pair_index_b]))


def test_fold_restricted_and_kfold_share_the_pairing_pool(score2_cohort):
    folds = stratified_folds(score2_cohort.physician, k=5, seed=0)
    res = run_design(EuclideanEstimator(), score2_cohort, "fold_restricted", folds=folds)
    for e in res.estimates:
        assert np.all(folds[e.pair_index_a] == folds[e.pair_index_b])
    assert res.n_fits == 1


def test_oracle_designs_pair_within_the_true_stratum(score2_cohort):
    m = score2_cohort.eligible
    elig = run_design(EuclideanEstimator(), score2_cohort, "oracle_eligible", eligible=m)
    inel = run_design(EuclideanEstimator(), score2_cohort, "oracle_ineligible", eligible=m)
    for e in elig.estimates:
        assert m[e.pair_index_a].all() and m[e.pair_index_b].all()
    for e in inel.estimates:
        assert (~m[e.pair_index_a]).all() and (~m[e.pair_index_b]).all()
    with pytest.raises(ValueError):
        run_design(EuclideanEstimator(), score2_cohort, "oracle_eligible")


def test_lopo_fits_without_the_evaluated_physician(score2_cohort):
    res = run_design(LearnedWeightsEstimator(), score2_cohort, "lopo")
    assert res.n_fits == len(np.unique(score2_cohort.physician))
    assert {e.physician for e in res.estimates} == set(np.unique(score2_cohort.physician))


def test_prefitted_estimator_is_reused_for_full_fit_designs(score2_cohort):
    base = run_design(LearnedWeightsEstimator(), score2_cohort, "in_sample")
    again = run_design(LearnedWeightsEstimator(), score2_cohort, "in_sample", prefitted=base.fitted)
    assert again.n_fits == 0
    assert [e.score for e in again.estimates] == [e.score for e in base.estimates]
