# -*- coding: utf-8 -*-
"""Discordance estimators with separated fitting and estimation steps.

Every estimator sees only what an analyst would see in observational data:
the covariates ``X``, the binary prescription ``y`` and the physician
identifier. The eligibility indicator is never passed to an estimator. The
oracle analyses of the benchmark restrict the *candidate pool* of patients
(see ``estimate_matching``), which is done by the design layer, not by the
estimator.

Two steps are kept separate so that cross-fitting is possible:

- ``fit(cohort, fit_idx)`` learns what depends on the outcome (feature
  weights, a random forest, a mixed model) from the training rows only;
- the estimation step builds per-physician distance matrices on evaluation
  rows, forms pairs under the caliper and computes the discordance rate.

Covariate standardisation uses the covariates of the evaluated cohort only
(never the outcome), as in the historical pipeline.

The numerical choices (random forest hyperparameters, caliper, pairing
heuristic, Gaussian mixture grid, differential evolution settings) reproduce
``src.analysis.analysis_pipeline`` so that in-sample results can be checked
for equivalence with the version of the code used for the first submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.analysis.matching_core import (
    _fit_binomial_bayes_mixed_glmm,
    _optimal_one_to_one_pairs,
    _per_physician_caliper_threshold,
    genetic_objective_function,
)

RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 8
MIN_PATIENTS_PER_PHYSICIAN = 30
DEFAULT_CALIPER_PERCENTILE = 25.0
LPA_K_MIN, LPA_K_MAX, LPA_N_INIT, LPA_REG_COVAR = 2, 10, 10, 1e-6
LPA_HYBRID_ALPHA = 0.5
DE_BOUNDS, DE_MAXITER, DE_POPSIZE, DE_TOL = (0.0, 10.0), 50, 30, 0.01
GLMM_EPSILON = 1e-10


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class Cohort:
    """Analyst-visible data of one simulated cohort."""

    X: np.ndarray
    y: np.ndarray
    physician: np.ndarray
    feature_names: list[str]

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        self.physician = np.asarray(self.physician).astype(str)
        if self.X.ndim != 2 or self.X.shape[1] != len(self.feature_names):
            raise ValueError("X must be (n, p) with one feature name per column.")
        if not (len(self.X) == len(self.y) == len(self.physician)):
            raise ValueError("X, y and physician must have the same number of rows.")
        if not set(np.unique(self.y)).issubset({0, 1}):
            raise ValueError("y must be binary (0/1).")

    @property
    def n(self) -> int:
        return int(len(self.y))

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        feature_names: list[str],
        outcome_col: str = "recommendation",
        physician_col: str = "professional_id",
    ) -> "Cohort":
        X = df[feature_names].astype(float).to_numpy()
        return cls(X=X, y=df[outcome_col].to_numpy(), physician=df[physician_col].to_numpy(),
                   feature_names=list(feature_names))


@dataclass
class PhysicianEstimate:
    """Result of one estimator for one physician in one evaluation design."""

    physician: str
    score: float
    n_candidates: int
    n_pairs: int
    n_patients_paired: int
    caliper_threshold: float
    pair_index_a: np.ndarray = field(repr=False)
    pair_index_b: np.ndarray = field(repr=False)
    pair_distance: np.ndarray = field(repr=False)


# -----------------------------------------------------------------------------
# Shared numerical helpers (identical to the historical pipeline)
# -----------------------------------------------------------------------------


def zscore(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score with the stability epsilon of the historical pipeline."""
    X = np.asarray(X, dtype=float)
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)


def _random_forest(n_train: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=max(5, int(0.01 * n_train)),
    )


def _normalised_positive(weights: np.ndarray, floor: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if (not np.all(np.isfinite(w))) or w.sum() <= 0:
        w = np.ones_like(w)
    w = np.maximum(w, floor)
    return w / w.sum()


def _whitening_matrix(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (kept columns, W) such that (Z[:, kept] @ W) has identity covariance.

    Uses the symmetric inverse square root of the covariance (pseudo-inverse for
    numerical robustness). Euclidean distance in the whitened space equals the
    Mahalanobis distance in the original space.
    """
    keep = np.std(Z, axis=0) > 1e-9
    cov = np.cov(Z[:, keep], rowvar=False)
    eigval, eigvec = np.linalg.eigh(np.atleast_2d(cov))
    inv_sqrt = np.where(eigval > 1e-12, 1.0 / np.sqrt(np.clip(eigval, 1e-12, None)), 0.0)
    return keep, (eigvec * inv_sqrt) @ eigvec.T


# -----------------------------------------------------------------------------
# Estimator interface
# -----------------------------------------------------------------------------


class Estimator:
    """Base class. Subclasses define ``fit`` and either a distance or a score."""

    name: str = "base"
    supervised: bool = False
    kind: str = "matching"  # "matching" or "model"
    pairing: str = "optimal"  # "optimal" (one-to-one) or "nn_replacement"

    def fit(self, cohort: Cohort, fit_idx: np.ndarray) -> "Estimator":
        """Learn outcome-dependent quantities from ``fit_idx`` rows of ``cohort``."""
        return self

    def prepare(self, cohort: Cohort) -> None:
        """Pre-compute transforms of an evaluation cohort (covariates only)."""
        raise NotImplementedError

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        """Pairwise distances among rows ``idx`` of the prepared cohort."""
        raise NotImplementedError

    def fitted_parameters(self) -> dict[str, Any]:
        """Learned quantities worth logging (weights, number of profiles...)."""
        return {}


class EuclideanEstimator(Estimator):
    name = "euclidean"

    def prepare(self, cohort: Cohort) -> None:
        p = cohort.X.shape[1]
        self._Z = zscore(cohort.X) * np.sqrt(np.ones(p) / p)

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        return cdist(self._Z[idx], self._Z[idx], metric="euclidean")


class MahalanobisEstimator(Estimator):
    name = "mahalanobis"

    def prepare(self, cohort: Cohort) -> None:
        Z = zscore(cohort.X)
        keep = np.std(cohort.X, axis=0) > 1e-9
        self._Z = Z[:, keep]
        self._VI = np.linalg.pinv(np.cov(self._Z, rowvar=False))

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        return cdist(self._Z[idx], self._Z[idx], metric="mahalanobis", VI=self._VI)


class LearnedWeightsEstimator(Estimator):
    """Random-forest Gini importances rescale the standardised covariate space."""

    name = "learned_weights"
    supervised = True

    def fit(self, cohort: Cohort, fit_idx: np.ndarray) -> "LearnedWeightsEstimator":
        Z = zscore(cohort.X)
        rf = _random_forest(len(fit_idx)).fit(Z[fit_idx], cohort.y[fit_idx])
        self.weights_ = _normalised_positive(rf.feature_importances_, 1e-6)
        return self

    def prepare(self, cohort: Cohort) -> None:
        self._Z = zscore(cohort.X) * np.sqrt(self.weights_)

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        return cdist(self._Z[idx], self._Z[idx], metric="euclidean")

    def fitted_parameters(self) -> dict[str, Any]:
        return {"weights": self.weights_.tolist()}


class MutualInformationEstimator(Estimator):
    """k-nearest-neighbour mutual information weights the standardised space."""

    name = "mutual_information"
    supervised = True

    def fit(self, cohort: Cohort, fit_idx: np.ndarray) -> "MutualInformationEstimator":
        Z = zscore(cohort.X)
        try:
            w = mutual_info_classif(Z[fit_idx], cohort.y[fit_idx], random_state=RANDOM_STATE)
        except ValueError:
            w = np.ones(Z.shape[1])
        self.weights_ = _normalised_positive(w, 1e-9)
        return self

    def prepare(self, cohort: Cohort) -> None:
        self._Z = zscore(cohort.X) * np.sqrt(self.weights_)

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        return cdist(self._Z[idx], self._Z[idx], metric="euclidean")

    def fitted_parameters(self) -> dict[str, Any]:
        return {"weights": self.weights_.tolist()}


class RandomForestProximityEstimator(Estimator):
    """Dissimilarity = 1 - fraction of trees in which two patients share a leaf.

    The forest is trained on ``fit_idx`` rows. Evaluation patients, including
    held-out ones under cross-fitting, are passed down the trained trees, so no
    held-out outcome enters the similarity.
    """

    name = "rf_proximity"
    supervised = True

    def fit(self, cohort: Cohort, fit_idx: np.ndarray) -> "RandomForestProximityEstimator":
        self._rf = _random_forest(len(fit_idx)).fit(cohort.X[fit_idx], cohort.y[fit_idx])
        return self

    def prepare(self, cohort: Cohort) -> None:
        self._leaves = self._rf.apply(cohort.X)

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        leaves = self._leaves[idx]
        prox = np.zeros((len(idx), len(idx)), dtype=float)
        for t in range(leaves.shape[1]):
            col = leaves[:, t]
            prox += col[:, None] == col
        prox /= leaves.shape[1]
        np.fill_diagonal(prox, -1.0)
        return 1.0 - prox


class LPAGuidedEstimator(Estimator):
    """Hybrid latent-clinical distance from a Gaussian mixture fitted on covariates.

    Unsupervised: the mixture uses covariates only, so ``fit`` ignores the
    outcome. It is fitted on the evaluated cohort in ``prepare``.
    """

    name = "lpa_guided"

    def prepare(self, cohort: Cohort) -> None:
        X = cohort.X
        Xs = np.asarray(RobustScaler().fit_transform(X), dtype=float)
        degenerate = np.where(np.std(Xs, axis=0) < 1e-12)[0]
        if degenerate.size:
            Xs[:, degenerate] = zscore(X[:, degenerate])
        best, best_bic, best_k = None, np.inf, None
        for k in range(LPA_K_MIN, min(LPA_K_MAX, len(X)) + 1):
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE,
                                 n_init=LPA_N_INIT, reg_covar=LPA_REG_COVAR)
            try:
                gm.fit(Xs)
            except ValueError:
                continue
            bic = float(gm.bic(Xs))
            if gm.converged_ and np.isfinite(bic) and bic < best_bic:
                best, best_bic, best_k = gm, bic, k
        if best is None:
            best = GaussianMixture(n_components=2, covariance_type="full", random_state=RANDOM_STATE,
                                   n_init=LPA_N_INIT, reg_covar=LPA_REG_COVAR).fit(Xs)
            best_k, best_bic = 2, float(best.bic(Xs))
        self.n_components_, self.bic_ = int(best_k), float(best_bic)
        self._Xs = Xs
        self._probs = np.asarray(best.predict_proba(Xs), dtype=float)

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        d_lat = cdist(self._probs[idx], self._probs[idx], metric="euclidean")
        d_x = cdist(self._Xs[idx], self._Xs[idx], metric="euclidean")
        max_l = float(np.max(d_lat)) if d_lat.size and np.max(d_lat) > 0 else 1.0
        max_x = float(np.max(d_x)) if d_x.size and np.max(d_x) > 0 else 1.0
        return LPA_HYBRID_ALPHA * (d_lat / max_l) + (1.0 - LPA_HYBRID_ALPHA) * (d_x / max_x)

    def fitted_parameters(self) -> dict[str, Any]:
        return {"n_components": getattr(self, "n_components_", None), "bic": getattr(self, "bic_", None)}


class GeneticMahalanobisEstimator(Estimator):
    """Genetic matching weights with nearest-neighbour matching with replacement.

    ``variant="legacy"`` reproduces the first submission: weights are applied
    to the standardised covariates and the Mahalanobis distance is then
    computed with the covariance of the weighted data. Because the
    Mahalanobis distance is invariant to a rescaling of the columns, the
    learned weights cancel exactly in this variant.

    ``variant="genmatch"`` follows Diamond and Sekhon (2013): covariates are
    whitened first (so that Euclidean distance equals the Mahalanobis
    distance), then weighted, both in the optimisation and in the final
    matching. The learned weights then change the distance.
    """

    name = "genetic_mahalanobis"
    supervised = True
    pairing = "nn_replacement"

    def __init__(self, variant: str = "genmatch", workers: int = -1) -> None:
        if variant not in {"legacy", "genmatch"}:
            raise ValueError(f"Unknown variant {variant!r}.")
        self.variant = variant
        self.workers = workers
        if variant == "legacy":
            self.name = "genetic_mahalanobis_legacy"

    def _space(self, X: np.ndarray) -> np.ndarray:
        Z = zscore(X)
        if self.variant == "legacy":
            return Z
        keep, W = _whitening_matrix(Z)
        return Z[:, keep] @ W

    def fit(self, cohort: Cohort, fit_idx: np.ndarray) -> "GeneticMahalanobisEstimator":
        S = self._space(cohort.X)
        Sf, yf = S[fit_idx], cohort.y[fit_idx]
        result = differential_evolution(
            genetic_objective_function, [DE_BOUNDS] * S.shape[1], args=(Sf, yf),
            maxiter=DE_MAXITER, popsize=DE_POPSIZE, tol=DE_TOL, workers=self.workers,
            updating="deferred" if self.workers != 1 else "immediate", seed=RANDOM_STATE,
        )
        self.raw_weights_ = np.asarray(result.x, dtype=float)
        if self.variant == "legacy":
            self.weights_ = np.sqrt(np.abs(self.raw_weights_) + 1e-6)
        else:
            self.weights_ = np.sqrt(_normalised_positive(np.abs(self.raw_weights_) + 1e-9, 1e-12))
        return self

    def prepare(self, cohort: Cohort) -> None:
        if self.variant == "legacy":
            Zw = zscore(cohort.X) * self.weights_
            keep = np.std(Zw, axis=0) > 1e-9
            self._Z = Zw[:, keep]
            self._VI = np.linalg.pinv(np.cov(self._Z, rowvar=False))
        else:
            self._Z = self._space(cohort.X) * self.weights_

    def distance_matrix(self, idx: np.ndarray) -> np.ndarray:
        if self.variant == "legacy":
            return cdist(self._Z[idx], self._Z[idx], metric="mahalanobis", VI=self._VI)
        return cdist(self._Z[idx], self._Z[idx], metric="euclidean")

    def fitted_parameters(self) -> dict[str, Any]:
        return {"raw_weights": self.raw_weights_.tolist(), "variant": self.variant}


# -----------------------------------------------------------------------------
# Matching-based estimation
# -----------------------------------------------------------------------------


def _pairs_optimal(d: np.ndarray, threshold: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    pairs = _optimal_one_to_one_pairs(d, caliper_threshold=threshold)
    if not pairs:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    a, b = zip(*pairs)
    return np.asarray(a, dtype=int), np.asarray(b, dtype=int)


def _pairs_nn_replacement(d: np.ndarray, threshold: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    nn = np.argmin(d, axis=1)
    a = np.arange(d.shape[0])
    keep = np.ones(d.shape[0], dtype=bool) if threshold is None else d[a, nn] <= threshold
    return a[keep], nn[keep]


def estimate_matching(
    estimator: Estimator,
    cohort: Cohort,
    eval_idx: np.ndarray,
    groups: Optional[np.ndarray] = None,
    caliper_percentile: Optional[float] = DEFAULT_CALIPER_PERCENTILE,
    min_patients: int = MIN_PATIENTS_PER_PHYSICIAN,
) -> list[PhysicianEstimate]:
    """Form pairs within each physician and compute discordance rates.

    Args:
        estimator: fitted estimator; ``prepare(cohort)`` must have been called.
        cohort: evaluated cohort.
        eval_idx: rows forming the candidate pool (all rows, a held-out fold,
            or an eligibility stratum for the oracle analyses).
        groups: optional label per row of ``cohort``; when given, pairs are
            formed only within (physician, group) cells, for example within
            cross-fitting folds. Pairs are pooled across cells.
        caliper_percentile: distance percentile of the within-cell pairwise
            distances used as caliper; ``None`` disables the caliper.
        min_patients: physicians with fewer candidates in ``eval_idx`` are
            skipped, as in the historical pipeline.
    """
    eval_idx = np.asarray(eval_idx, dtype=int)
    phys_eval = cohort.physician[eval_idx]
    pair_fn: Callable = _pairs_optimal if estimator.pairing == "optimal" else _pairs_nn_replacement
    out: list[PhysicianEstimate] = []
    for physician in np.unique(phys_eval):
        idx_p = eval_idx[phys_eval == physician]
        if len(idx_p) < min_patients:
            continue
        cells = [idx_p] if groups is None else [idx_p[groups[idx_p] == g] for g in np.unique(groups[idx_p])]
        a_all, b_all, d_all, thresholds = [], [], [], []
        for cell in cells:
            if len(cell) < 2:
                continue
            d = np.asarray(estimator.distance_matrix(cell), dtype=float)
            if not np.all(np.isfinite(d)):
                d = np.nan_to_num(d, nan=1e12, posinf=1e12, neginf=1e12)
            np.fill_diagonal(d, np.inf)
            thr = None if caliper_percentile is None else _per_physician_caliper_threshold(d, caliper_percentile)
            a, b = pair_fn(d, thr)
            a_all.append(cell[a]); b_all.append(cell[b]); d_all.append(d[a, b])
            thresholds.append(np.nan if thr is None else thr)
        a = np.concatenate(a_all) if a_all else np.empty(0, dtype=int)
        b = np.concatenate(b_all) if b_all else np.empty(0, dtype=int)
        dist = np.concatenate(d_all) if d_all else np.empty(0)
        score = float(np.mean(cohort.y[a] != cohort.y[b])) if len(a) else float("nan")
        n_paired = len(a) if estimator.pairing == "nn_replacement" else 2 * len(a)
        out.append(PhysicianEstimate(
            physician=str(physician), score=score, n_candidates=int(len(idx_p)), n_pairs=int(len(a)),
            n_patients_paired=int(n_paired),
            caliper_threshold=float(np.nanmean(thresholds)) if thresholds else float("nan"),
            pair_index_a=a, pair_index_b=b, pair_distance=dist,
        ))
    return out


# -----------------------------------------------------------------------------
# Bayesian binomial GLMM with Pearson-residual score
# -----------------------------------------------------------------------------


class GLMMEstimator(Estimator):
    """Physician random-intercept logistic model; score = mean squared Pearson residual.

    Fitted by mean-field variational Bayes on ``fit_idx`` rows. The physician
    random intercept of a physician absent from the training rows is set to 0.
    Continuous covariates are standardised on the evaluated cohort, as in the
    historical pipeline; binary covariates are kept as 0/1.
    """

    name = "glmm"
    supervised = True
    kind = "model"

    def _design(self, cohort: Cohort) -> pd.DataFrame:
        df = pd.DataFrame(cohort.X, columns=cohort.feature_names)
        cont = [c for c in cohort.feature_names if df[c].nunique() > 2]
        if cont:
            df[cont] = StandardScaler().fit_transform(df[cont])
        df["_y"] = cohort.y
        df["_phys"] = cohort.physician
        return df

    def fit(self, cohort: Cohort, fit_idx: np.ndarray) -> "GLMMEstimator":
        df = self._design(cohort).iloc[np.asarray(fit_idx, dtype=int)].reset_index(drop=True)
        model, result = _fit_binomial_bayes_mixed_glmm(df, "_y", "_phys", list(cohort.feature_names))
        names = list(model.fep_names)
        fe = np.asarray(result.fe_mean, dtype=float)
        self.intercept_ = float(fe[names.index("Intercept")]) if "Intercept" in names else 0.0
        self.coef_ = np.array([fe[names.index(c)] for c in cohort.feature_names])
        self.random_effects_ = {}
        for vc_name, value in zip(model.vc_names, np.asarray(result.vc_mean, dtype=float)):
            label = str(vc_name)
            if "[" in label and "]" in label:
                label = label.split("[")[-1].split("]")[0]
            if label.startswith("T."):
                label = label[2:]
            self.random_effects_[label] = float(value)
        missing = set(np.unique(df["_phys"])) - set(self.random_effects_)
        if missing:
            raise RuntimeError(f"GLMM random effects not mapped for physicians: {sorted(missing)[:5]}")
        return self

    def prepare(self, cohort: Cohort) -> None:
        self._df = self._design(cohort)
        self._features = list(cohort.feature_names)

    def estimate(self, cohort: Cohort, eval_idx: np.ndarray,
                 min_patients: int = MIN_PATIENTS_PER_PHYSICIAN) -> list[PhysicianEstimate]:
        eval_idx = np.asarray(eval_idx, dtype=int)
        df = self._df.iloc[eval_idx]
        lin = self.intercept_ + df[self._features].to_numpy(dtype=float) @ self.coef_
        u = df["_phys"].map(self.random_effects_).fillna(0.0).to_numpy(dtype=float)
        p = np.clip(1.0 / (1.0 + np.exp(-(lin + u))), GLMM_EPSILON, 1 - GLMM_EPSILON)
        r2 = (df["_y"].to_numpy() - p) ** 2 / np.clip(p * (1 - p), GLMM_EPSILON, None)
        out: list[PhysicianEstimate] = []
        phys = df["_phys"].to_numpy()
        empty_i, empty_f = np.empty(0, dtype=int), np.empty(0)
        for physician in np.unique(phys):
            mask = phys == physician
            if mask.sum() < min_patients:
                continue
            out.append(PhysicianEstimate(
                physician=str(physician), score=float(np.mean(r2[mask])), n_candidates=int(mask.sum()),
                n_pairs=0, n_patients_paired=0, caliper_threshold=float("nan"),
                pair_index_a=empty_i, pair_index_b=empty_i, pair_distance=empty_f,
            ))
        return out


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


def build_estimators(genetic_variant: str = "genmatch", include_genetic: bool = True,
                     genetic_workers: int = -1) -> list[Estimator]:
    """Instantiate the benchmark methods in their canonical order."""
    methods: list[Estimator] = [
        EuclideanEstimator(),
        MahalanobisEstimator(),
        LearnedWeightsEstimator(),
        RandomForestProximityEstimator(),
        MutualInformationEstimator(),
        LPAGuidedEstimator(),
    ]
    if include_genetic:
        methods.append(GeneticMahalanobisEstimator(variant=genetic_variant, workers=genetic_workers))
    methods.append(GLMMEstimator())
    return methods
