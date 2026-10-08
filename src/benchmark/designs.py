# -*- coding: utf-8 -*-
"""Evaluation designs: where an estimator learns, and where it forms pairs.

Designs (``DESIGNS``):

- ``in_sample``: learn on all rows, pair within each physician's full panel.
  This is the design of the first submission.
- ``fold_restricted``: learn on all rows, but pair only within cross-fitting
  folds. Control design: compared with ``in_sample`` it measures the effect of
  the smaller candidate pool; compared with ``kfold`` it isolates the effect
  of outcome leakage.
- ``kfold``: K-fold cross-fitting with folds stratified within physician.
  For fold k, learn on the other folds (all physicians pooled) and pair within
  fold k. Pairs are pooled over folds for each physician.
- ``lopo``: leave-one-physician-out. Learn on the other physicians, pair
  within the held-out physician's full panel.
- ``external``: learn on an independently simulated development cohort drawn
  from the same generating process, pair within the benchmark cohort.
- ``oracle_eligible`` / ``oracle_ineligible``: learn on all rows, pair only
  among truly eligible (resp. ineligible) patients. These are the oracle
  analyses of the estimand decomposition; the eligibility indicator restricts
  the candidate pool and is never seen by the estimator.

For unsupervised estimators ``fit`` ignores the outcome, so several designs
coincide by construction (``kfold`` = ``fold_restricted``; ``lopo`` and
``external`` = ``in_sample`` up to the development cohort used for
covariate-only transforms). ``equivalent_design`` returns the design that is
actually computed, so that the runner does not repeat identical work.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.benchmark.estimators import (
    DEFAULT_CALIPER_PERCENTILE,
    MIN_PATIENTS_PER_PHYSICIAN,
    Cohort,
    Estimator,
    PhysicianEstimate,
    estimate_matching,
)

DESIGNS = (
    "in_sample",
    "fold_restricted",
    "kfold",
    "lopo",
    "external",
    "oracle_eligible",
    "oracle_ineligible",
)
DEFAULT_K = 5
FULL_FIT_DESIGNS = frozenset({"in_sample", "fold_restricted", "oracle_eligible", "oracle_ineligible"})


@dataclass
class DesignResult:
    design: str
    estimates: list[PhysicianEstimate]
    fit_seconds: float
    estimate_seconds: float
    n_fits: int
    fitted: Optional[Estimator] = None


def stratified_folds(physician: np.ndarray, k: int = DEFAULT_K, seed: int = 0) -> np.ndarray:
    """Fold label per row, balanced within each physician (outcome not used)."""
    rng = np.random.default_rng(seed)
    folds = np.empty(len(physician), dtype=int)
    for p in np.unique(physician):
        idx = np.where(physician == p)[0]
        labels = np.arange(len(idx)) % k
        rng.shuffle(labels)
        folds[idx] = labels
    return folds


def equivalent_design(estimator: Estimator, design: str) -> str:
    """Design actually computed for ``design`` (identical by construction otherwise)."""
    if estimator.supervised:
        return design
    if design == "kfold":
        return "fold_restricted"
    if design in {"lopo", "external"}:
        return "in_sample"
    return design


def supports(estimator: Estimator, design: str) -> bool:
    """Whether a design is defined for an estimator.

    The GLMM score has no pairs, so ``fold_restricted`` and the oracle strata
    do not apply. Under ``lopo`` and ``external`` the evaluated physicians
    have no estimated random intercept, so their Pearson residuals would mix
    the physician's prescribing level with residual variability; these
    designs are not used for the GLMM.
    """
    if estimator.kind == "model":
        return design in {"in_sample", "kfold"}
    return design in DESIGNS


def _estimate(est: Estimator, cohort: Cohort, eval_idx: np.ndarray, groups: Optional[np.ndarray],
              caliper_percentile: Optional[float], min_patients: int) -> list[PhysicianEstimate]:
    if est.kind == "model":
        return est.estimate(cohort, eval_idx, min_patients=min_patients)  # type: ignore[attr-defined]
    return estimate_matching(est, cohort, eval_idx, groups=groups, caliper_percentile=caliper_percentile,
                             min_patients=min_patients)


def _pool_by_physician(parts: list[list[PhysicianEstimate]], y: np.ndarray, kind: str) -> list[PhysicianEstimate]:
    """Pool per-fold estimates of each physician (pairs pooled; GLMM residuals weighted by n)."""
    by_phys: dict[str, list[PhysicianEstimate]] = {}
    for part in parts:
        for e in part:
            by_phys.setdefault(e.physician, []).append(e)
    pooled: list[PhysicianEstimate] = []
    for phys, items in sorted(by_phys.items()):
        a = np.concatenate([e.pair_index_a for e in items])
        b = np.concatenate([e.pair_index_b for e in items])
        d = np.concatenate([e.pair_distance for e in items])
        n_cand = int(sum(e.n_candidates for e in items))
        if kind == "model":
            score = float(sum(e.score * e.n_candidates for e in items) / n_cand)
        else:
            score = float(np.mean(y[a] != y[b])) if len(a) else float("nan")
        thr = [e.caliper_threshold for e in items if np.isfinite(e.caliper_threshold)]
        pooled.append(PhysicianEstimate(
            physician=phys, score=score, n_candidates=n_cand, n_pairs=int(len(a)),
            n_patients_paired=int(sum(e.n_patients_paired for e in items)),
            caliper_threshold=float(np.mean(thr)) if thr else float("nan"),
            pair_index_a=a, pair_index_b=b, pair_distance=d,
        ))
    return pooled


def _keep_large_panels(estimates: list[PhysicianEstimate], cohort: Cohort,
                      min_patients: int = MIN_PATIENTS_PER_PHYSICIAN) -> list[PhysicianEstimate]:
    """Apply the minimum-panel rule to the physician's full panel, not to a fold."""
    sizes = {p: int(np.sum(cohort.physician == p)) for p in np.unique(cohort.physician)}
    return [e for e in estimates if sizes.get(e.physician, 0) >= min_patients]


def run_design(
    estimator: Estimator,
    cohort: Cohort,
    design: str,
    *,
    folds: Optional[np.ndarray] = None,
    eligible: Optional[np.ndarray] = None,
    development_cohort: Optional[Cohort] = None,
    caliper_percentile: Optional[float] = DEFAULT_CALIPER_PERCENTILE,
    prefitted: Optional[Estimator] = None,
) -> DesignResult:
    """Fit and evaluate one estimator under one design on one cohort.

    Args:
        folds: fold label per row (required for ``fold_restricted`` and ``kfold``).
        eligible: true eligibility per row (required for the oracle designs only;
            it restricts the candidate pool and is never passed to the estimator).
        development_cohort: independent cohort (required for ``external``).
        prefitted: estimator already fitted on all rows of ``cohort``. Designs
            that learn on all rows (``FULL_FIT_DESIGNS``) then reuse it instead
            of refitting; their ``fit_seconds`` is 0 and ``n_fits`` is 0.
    """
    if design not in DESIGNS:
        raise ValueError(f"Unknown design {design!r}.")
    if not supports(estimator, design):
        raise ValueError(f"Design {design!r} is not defined for estimator {estimator.name!r}.")
    reuse = prefitted is not None and design in FULL_FIT_DESIGNS
    est = copy.deepcopy(prefitted if reuse else estimator)
    all_idx = np.arange(cohort.n)
    fit_s = est_s = 0.0
    n_fits = 0

    def _fit(c: Cohort, idx: np.ndarray) -> None:
        nonlocal fit_s, n_fits
        t = time.perf_counter()
        est.fit(c, idx)
        est._n_fit_calls = getattr(est, "_n_fit_calls", 0) + 1
        fit_s += time.perf_counter() - t
        n_fits += 1

    def _prep_and_estimate(eval_idx: np.ndarray, groups: Optional[np.ndarray],
                           min_patients: int = MIN_PATIENTS_PER_PHYSICIAN) -> list[PhysicianEstimate]:
        nonlocal est_s
        t = time.perf_counter()
        token = (id(cohort), getattr(est, "_n_fit_calls", 0))
        if getattr(est, "_prepared_token", None) != token:
            est.prepare(cohort)
            est._prepared_token = token
        out = _estimate(est, cohort, eval_idx, groups, caliper_percentile, min_patients)
        est_s += time.perf_counter() - t
        return out

    if design in FULL_FIT_DESIGNS:
        if not reuse:
            _fit(cohort, all_idx)
        if design == "in_sample":
            estimates = _prep_and_estimate(all_idx, None)
        elif design == "fold_restricted":
            if folds is None:
                raise ValueError("fold_restricted requires folds.")
            estimates = _prep_and_estimate(all_idx, folds)
        else:
            if eligible is None:
                raise ValueError("Oracle designs require the eligibility indicator.")
            stratum = np.asarray(eligible, dtype=bool)
            if design == "oracle_ineligible":
                stratum = ~stratum
            estimates = _prep_and_estimate(all_idx[stratum], None)

    elif design == "kfold":
        if folds is None:
            raise ValueError("kfold requires folds.")
        parts = []
        for k in np.unique(folds):
            _fit(cohort, all_idx[folds != k])
            parts.append(_prep_and_estimate(all_idx[folds == k], None, min_patients=2))
        estimates = _keep_large_panels(_pool_by_physician(parts, cohort.y, est.kind), cohort)

    elif design == "lopo":
        parts = []
        for p in np.unique(cohort.physician):
            _fit(cohort, all_idx[cohort.physician != p])
            parts.append(_prep_and_estimate(all_idx[cohort.physician == p], None))
        estimates = _pool_by_physician(parts, cohort.y, est.kind)

    else:  # external
        if development_cohort is None:
            raise ValueError("external requires a development cohort.")
        _fit(development_cohort, np.arange(development_cohort.n))
        estimates = _prep_and_estimate(all_idx, None)

    return DesignResult(design=design, estimates=estimates, fit_seconds=fit_s,
                        estimate_seconds=est_s, n_fits=n_fits, fitted=est if design == "in_sample" else None)
