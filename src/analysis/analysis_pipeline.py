# -*- coding: utf-8 -*-
"""
Prism analysis module: quantification of intra-physician prescription variability.

This module implements a multi-method pipeline to measure inconsistency in medical
prescriptions across physicians. It combines a parametric mixed model (GLMM) with
several pairwise matching approaches to estimate how much physicians differ in
their decisions for clinically similar patients.

Methods (canonical order; display names used in figures and reports):
  1. Manual Pairing: Manual threshold matching; defines "comparable" patients by the
     clinical rule used for the generation of the dataset; discordance among comparable pairs.
     This is the positive control of the analysis, the ground truth, the gold standard.
  2. Euclidean Distance Pairing: Weighted Euclidean distance in covariate space with
     configurable weights; discordance by nearest neighbour.
  3. Mahalanobis Distance Pairing: Nearest-neighbour matching in covariate space;
     discordance = fraction of patients whose nearest neighbour has a different prescription.
  4. Learned Weights Euclidean Distance Pairing: RF feature importances as fixed weights
     for weighted Euclidean distance; discordance by nearest neighbour in that space.
  5. Genetic Matching: Optimises covariate weights via differential evolution to
     minimise imbalance (SMD), then computes discordance in the weighted space.
  6. Random Forest Proximity Pairing: Similarity = fraction of trees where two patients
     land in the same leaf; nearest neighbour by this similarity.
  7. Mutual Information: Feature weights from mutual information are used to build
     a weighted Euclidean distance space for nearest-neighbour pairing.
  8. Generalized Mixed Model: Bayesian binomial GLMM; extracts physician random effects
     and local overdispersion (OLRE) as a model-based variability metric.
  9. Ensemble Method: Aggregates the above (with GLMM projected to [0,1]) into a single
     score per physician.
"""
from __future__ import annotations

import os
import json
import logging
import textwrap
import warnings
import random
from pathlib import Path
from typing import Callable, Optional, Tuple
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from adjustText import adjust_text
from src.analysis.baseline_table import generate_baseline_table_one
from src.analysis.matching_core import (
    _discordance_rate_from_pairs,
    _fit_binomial_bayes_mixed_glmm,
    _glmm_fixed_effect_weights,
    _greedy_nearest_neighbor_pairs,
    _optimal_one_to_one_pairs,
    _per_physician_caliper_threshold,
    genetic_objective_function,
)
from src.analysis.plotting import _scatter_hue_physician_categorical
from src.synthetic_generator import (
    get_generation_rule_mask,
)

# -----------------------------------------------------------------------------
# Quick pipeline overview
# -----------------------------------------------------------------------------
# 1) run_glmm_analysis()
#    - Cleans and standardizes the data,
#    - Fits a binomial GLMM with physician random effects,
#    - Computes a local overdispersion score per physician (OLRE-like).
#
# 2) run_*_matching_analysis()
#    - Builds pairs of "similar" patients (per method),
#    - Measures, for each physician, decision discordance within those pairs.
#
# 3) run_ensemble_matching_analysis()
#    - Aggregates several scores into a single composite indicator.
#
# 4) plot_glmm_vs_matching_comparison()
#    - Visually compares methods (distribution, deltas, correlations).
# -----------------------------------------------------------------------------

# Canonical order and display names for all methods.
METHOD_PLOT_SPECS = [
    ("discordance_rate_manual", "Manual Pairing"),
    ("discordance_rate_euclidean", "Euclidean Distance Pairing"),
    ("discordance_rate_mahalanobis", "Mahalanobis Distance Pairing"),
    ("discordance_rate_learning", "Learned Weights Euclidean Distance Pairing"),
    ("discordance_rate_genetic", "Genetic Matching"),
    ("discordance_rate_genetic_mahalanobis", "Genetic Matching (Mahalanobis)"),
    ("discordance_rate_rf_matching", "Random Forest Proximity Pairing"),
    ("discordance_rate_lpa_guided_pairing", "LPA-guided Pairing"),
    ("discordance_rate_mutual_info", "Mutual Information"),
    ("overdispersion_local", "Generalized Mixed Model"),
    ("ensemble_matching", "Ensemble Method"),
]
METHOD_DISPLAY_NAMES = dict(METHOD_PLOT_SPECS)

# Subdirectory under results_dir / "plots" / ...
PLOT_SUBDIR_METHOD_COMPARISON = "method_comparison"


class Analysis:
    """
    Orchestrator for the full variability analysis pipeline.

    Runs GLMM-based overdispersion estimation and multiple discordance-by-matching
    methods, then aggregates results into a single per-physician table and comparison
    plots. All methods use the same outcome (e.g. recommendation 0/1), physician ID,
    and fixed-effect covariates; matching methods operate on the same complete-case
    subset produced by the GLMM step for consistency.

    Attributes:
        df: Patient-level DataFrame (copy); must contain outcome, physician ID, and
            fixed-effect columns.
        results_dir: Directory for GLMM report JSON, plots, and output CSV.
        config: Dict with outcome_col, physician_col, fixed_effects, min_patients_per_physician,
            and optional dataset_targets for reporting.
        results: Dict holding "glmm", "intra_physician_variability", etc., populated
            by run_full_analysis() or individual run_* methods.
    """

    def __init__(self, df: pd.DataFrame, results_dir: Optional[str] = None, LOGGER=None, config: Optional[dict] = None):
        """
        Initialize the analysis with data and configuration.

        Arguments:
            df: Raw patient-level DataFrame (outcome, physician, covariates).
            results_dir: Output directory for reports and plots; default "experiments".
            LOGGER: Logger instance (required).
            config: Override for column names and thresholds; see _default_config().
        """
        if LOGGER is None:
            raise ValueError("LOGGER must be provided.")
        self.df = df.copy()
        self.LOGGER = LOGGER
        self.results_dir = Path(results_dir) if results_dir else Path("experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or self._default_config()
        self.results = {}

    def _default_config(self) -> dict:
        """
        Default analysis configuration: column names and GLMM/matching parameters.

        Returns:
            dict: outcome_col, physician_col, fixed_effects list, min_patients_per_physician,
                and optional dataset_targets for expected variance/coefficients in reports.
        """
        return {
            "outcome_col": "recommendation",
            "physician_col": "professional_id",
            "fixed_effects": [
                "age",
                "is_male",
                "is_smoker",
                "ldl_cholesterol",
                "non_hdl_cholesterol",
                "hdl_cholesterol",
                "systolic_blood_pressure",
                "biomarker_hba1c_ngsp_blood",
                "estimated_glomerular_filtration_rate",
            ],
            "min_patients_per_physician": 30,
            "generation_strategy": "score2_five_groups_heter_patients",
            "dataset_targets": {},
            "enable_manual_pairing": True,
            "lpa_enabled": True,
            "lpa_n_components_min": 2,
            "lpa_n_components_max": 10,
            "lpa_covariance_type": "full",
            "lpa_random_state": 42,
            "lpa_n_init": 10,
            "lpa_robust_scaling": True,
            "lpa_distance_metric": "hybrid",
            "lpa_hybrid_alpha": 0.5,
            "lpa_pairing_strategy": "optimal",
            "pairing_strategy": "optimal",
            "caliper_enabled": False,
            "caliper_percentile": 25.0,
            "glmm_enabled": True,
        }

    def _glmm_enabled(self) -> bool:
        """Return whether the GLMM step should run (can be disabled via YAML config)."""
        return bool(self.config.get("glmm_enabled", True))

    def _resolve_pairing_fn(self) -> Callable[..., list[tuple[int, int]]]:
        """Resolve the pairing function for distance-based matching methods.

        Reads `self.config["pairing_strategy"]` ("optimal" by default) and returns
        either Hungarian-based optimal 1-to-1 matching or the legacy greedy
        nearest-neighbour pairing. Result is cached per Analysis instance to
        avoid re-emitting log lines on every call.

        Hungarian is order-invariant by construction (Pierre, May 2026 review)
        and is now the default for all distance-based methods (Euclidean,
        Mahalanobis, Learned Weights, RF Proximity, Mutual Information).
        Manual Pairing uses a boolean eligibility mask (no greedy step) and the
        genetic methods use a separate `argmin` mechanism by design, so neither
        is affected by this knob.
        """
        cached = getattr(self, "_cached_pairing_fn", None)
        if cached is not None:
            return cached
        raw = self.config.get("pairing_strategy", "optimal")
        strategy = str(raw).strip().lower()
        if strategy not in {"greedy", "optimal"}:
            self.LOGGER.warning(
                "Invalid pairing_strategy=%r; falling back to 'optimal'.", raw,
            )
            strategy = "optimal"
        fn: Callable[..., list[tuple[int, int]]] = (
            _optimal_one_to_one_pairs if strategy == "optimal"
            else _greedy_nearest_neighbor_pairs
        )
        self.LOGGER.info(
            "Pairing strategy for distance-based methods: %s (Hungarian=optimal, legacy=greedy).",
            strategy,
        )
        self._cached_pairing_fn = fn
        return fn

    # --------------------------------------------------------------------------
    # PIPELINE METHOD 1: Parametric modelling (GLMM + OLRE)
    # --------------------------------------------------------------------------

    def _prepare_analysis_cohort(self, log_prefix: str = "Analysis cohort") -> pd.DataFrame:
        """Clean and standardize the cohort shared by the GLMM and matching steps."""
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        fixed_effects = self.config["fixed_effects"]

        cols_needed = [outcome_col, physician_col] + fixed_effects
        missing_cols = [c for c in cols_needed if c not in self.df.columns]
        if missing_cols:
            error_msg = f"CRITICAL: Missing columns for analysis cohort: {missing_cols}"
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        n_complete_case = int(self.df[cols_needed].dropna(how="any").shape[0])
        df_clean = self.df[cols_needed].copy()
        df_clean = df_clean.dropna(subset=[outcome_col, physician_col])
        n_with_outcome_physician = len(df_clean)
        if n_complete_case < n_with_outcome_physician:
            self.LOGGER.info(
                "%s: %d rows with outcome and physician present; only %d complete cases "
                "(all covariates non-null). Median imputation of covariates to retain the sample.",
                log_prefix,
                n_with_outcome_physician,
                n_complete_case,
            )
        else:
            self.LOGGER.info(
                "%s: %d rows after outcome/physician filter (complete cases on all columns: %d).",
                log_prefix,
                n_with_outcome_physician,
                n_complete_case,
            )

        for c in fixed_effects:
            if c not in df_clean.columns:
                continue
            if pd.api.types.is_bool_dtype(df_clean[c]):
                df_clean[c] = df_clean[c].astype("float64")
            elif not pd.api.types.is_numeric_dtype(df_clean[c]):
                df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
            med = df_clean[c].median(skipna=True)
            n_na = int(df_clean[c].isna().sum())
            if n_na == 0:
                continue
            if pd.isna(med):
                df_clean[c] = df_clean[c].fillna(0.0)
                self.LOGGER.warning(
                    "%s: covariate '%s' — %d missing values, undefined median; imputing with 0.",
                    log_prefix,
                    c,
                    n_na,
                )
            else:
                df_clean[c] = df_clean[c].fillna(med)
                self.LOGGER.info(
                    "%s: covariate '%s' — %d missing values imputed with the median (%.6g).",
                    log_prefix,
                    c,
                    n_na,
                    float(med),
                )

        if len(df_clean) < 50:
            error_msg = (
                f"CRITICAL: Not enough complete data for analysis. Found {len(df_clean)} rows, minimum is 50."
            )
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        if pd.api.types.is_bool_dtype(df_clean[outcome_col]):
            df_clean[outcome_col] = df_clean[outcome_col].astype(int)
        else:
            outcome_num = pd.to_numeric(df_clean[outcome_col], errors="coerce")
            if outcome_num.isna().any():
                n_bad = int(outcome_num.isna().sum())
                bad_examples = df_clean.loc[outcome_num.isna(), outcome_col].astype(str).head(5).tolist()
                error_msg = (
                    f"CRITICAL: Outcome column '{outcome_col}' contains {n_bad} non-numeric values. "
                    f"Examples: {bad_examples}"
                )
                self.LOGGER.error(error_msg)
                raise ValueError(error_msg)
            unique_vals = sorted(pd.unique(outcome_num))
            unique_set = set(unique_vals)
            if unique_set.issubset({0, 1}):
                df_clean[outcome_col] = outcome_num.astype(int)
            elif len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                self.LOGGER.warning(
                    "Outcome column '%s' is binary but not encoded as {0,1}. Applying mapping: %s",
                    outcome_col,
                    mapping,
                )
                df_clean[outcome_col] = outcome_num.map(mapping).astype(int)
            else:
                error_msg = (
                    f"CRITICAL: Outcome column '{outcome_col}' must be binary. "
                    f"Found values: {unique_vals}"
                )
                self.LOGGER.error(error_msg)
                raise ValueError(error_msg)
        if df_clean[outcome_col].nunique() < 2:
            error_msg = f"CRITICAL: Outcome column '{outcome_col}' has a single class after cleaning."
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        for c in fixed_effects:
            if c in df_clean.columns and pd.api.types.is_bool_dtype(df_clean[c]):
                df_clean[c] = df_clean[c].astype(int)
        numeric_cols = [
            c for c in fixed_effects
            if c in df_clean.columns
            and pd.api.types.is_numeric_dtype(df_clean[c])
            and df_clean[c].nunique() > 2
        ]
        if numeric_cols:
            scaler = StandardScaler()
            df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

        if not fixed_effects:
            raise ValueError("CRITICAL: The 'fixed_effects' list is empty. At least one fixed effect is required.")

        return df_clean

    def _prepare_matching_cohort_only(self) -> None:
        """Prepare the analysis cohort without fitting a GLMM (matching only)."""
        self.LOGGER.info("=== PREPARING MATCHING COHORT (GLMM DISABLED) ===")
        if self._is_matching_cohort_ready():
            return
        try:
            df_clean = self._prepare_analysis_cohort(log_prefix="Matching cohort")
        except ValueError as exc:
            self.results["matching_cohort"] = {"error": str(exc)}
            self.LOGGER.error("Matching cohort preparation failed: %s", exc)
            return

        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        min_patients = self.config["min_patients_per_physician"]
        physician_metrics = []
        for physician, group in df_clean.groupby(physician_col):
            n_patients = len(group)
            if n_patients < min_patients:
                continue
            physician_metrics.append(
                {
                    "physician": physician,
                    "n_patients": n_patients,
                    "prescription_rate": round(group[outcome_col].mean(), 4),
                }
            )
        self.results["matching_cohort"] = {
            "df_clean": df_clean,
            "n_physicians": df_clean[physician_col].nunique(),
            "n_observations": len(df_clean),
        }
        self.results["intra_physician_variability"] = pd.DataFrame(physician_metrics)

    def run_glmm_analysis(self) -> None:
        """
        Fit a binomial GLMM, compute global and per-physician variability (OLRE), and save a master report.
        """
        self.LOGGER.info("=== STARTING GLMM & OLRE PIPELINE ===")

        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        fixed_effects = self.config["fixed_effects"]
        min_patients = self.config["min_patients_per_physician"]

        df_clean = self._prepare_analysis_cohort(log_prefix="GLMM")

        seed_value = 42
        np.random.seed(seed_value)
        random.seed(seed_value)

        # Variational Bayes binomial GLMM fitting.
        # Block 2: model fitting, after data preparation.
        try:
            model, result = _fit_binomial_bayes_mixed_glmm(
                df_clean, outcome_col, physician_col, fixed_effects,
            )
        except Exception as exc:
            error_msg = (
                f"CRITICAL: GLMM fitting failed ({len(fixed_effects)} fixed effects). "
                f"Error: {exc}"
            )
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg) from exc

        # End of variational Bayes binomial GLMM fitting.
        # ------------------------------------------------------------

        summary_text = str(result.summary())
        # Global weights derived from fixed-effect coefficients (used for reporting).
        fixed_effect_weights = _glmm_fixed_effect_weights(result)

        # Global physician random-effect variance (proxy for inter-physician heterogeneity).
        global_variance = 0.0
        if hasattr(result, 'vcp_mean') and result.vcp_mean is not None:
            vcp = np.array(result.vcp_mean).flatten()
            if len(vcp) > 0 and not np.isnan(vcp[0]):
                global_variance = float(np.abs(vcp[0]))

        # Patient-level predicted probabilities.
        if hasattr(result, "predict"):
            p_pred = result.predict()
        else:
            lin_pred = model.predict(result.params)
            p_pred = 1 / (1 + np.exp(-lin_pred))

        if hasattr(p_pred, "flatten"):
            p_pred = p_pred.flatten()
        elif isinstance(p_pred, (pd.Series, pd.DataFrame)):
            p_pred = p_pred.values.flatten()
        p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
        if p_pred.shape[0] != len(df_clean):
            error_msg = (
                f"CRITICAL: Predicted probability length mismatch: got {p_pred.shape[0]}, "
                f"expected {len(df_clean)}."
            )
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)
        if not np.all(np.isfinite(p_pred)):
            self.LOGGER.warning("Non-finite predicted probabilities detected; replacing with 0.5.")
            p_pred = np.where(np.isfinite(p_pred), p_pred, 0.5)
        p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)
        df_clean["_predicted_prob"] = p_pred

        # Extract individual physician random effects.
        # Block 3: extract global variance, predicted probabilities, and per-physician random effects.
        # Depending on summary() structure (which sometimes varies), try several paths.
        def _normalize_physician_label(raw_label: object) -> str:
            label = str(raw_label).strip()
            label = label.strip("'").strip('"')
            if label.startswith("T."):
                label = label[2:]
            if "[" in label and "]" in label:
                label = label.split("[")[-1].split("]")[0]
                label = label.strip("'").strip('"')
                if label.startswith("T."):
                    label = label[2:]
            return label

        physician_labels = [str(p) for p in pd.unique(df_clean[physician_col])]
        physician_bias = {p: 0.0 for p in physician_labels}
        try:
            # Path 1 (preferred): use the explicit vc_names <-> vc_mean mapping.
            vc_names = getattr(model, "vc_names", None)
            vc_mean = getattr(result, "vc_mean", None)
            mapped = 0
            if vc_names is not None and vc_mean is not None:
                vc_names_list = list(vc_names)
                vc_vals = np.asarray(vc_mean).flatten()
                if len(vc_names_list) == len(vc_vals):
                    for name, val in zip(vc_names_list, vc_vals):
                        key = _normalize_physician_label(name)
                        if key in physician_bias and np.isfinite(val):
                            physician_bias[key] = float(val)
                            mapped += 1
                else:
                    self.LOGGER.warning(
                        "Random-effect mapping mismatch: %d vc_names vs %d vc_mean values.",
                        len(vc_names_list),
                        len(vc_vals),
                    )

            # Path 2: parse the summary table if explicit mapping is incomplete.
            summary_tables = result.summary().tables
            if mapped < len(physician_bias) and len(summary_tables) > 1:
                summary_df = summary_tables[1]
                re_rows = [idx for idx in summary_df.index if 'physician_re' in idx]
                if re_rows:
                    post_mean_col = next((c for c in summary_df.columns if 'mean' in str(c).lower()), None)
                    for row_name in re_rows:
                        clean_name = _normalize_physician_label(row_name)
                        if post_mean_col is not None:
                            try:
                                val = float(summary_df.loc[row_name, post_mean_col])
                                if clean_name in physician_bias and np.isfinite(val):
                                    physician_bias[clean_name] = val
                            except (KeyError, TypeError, ValueError):
                                continue

            # Path 3 (last resort): if nothing could be aligned by name but the sizes
            # match, apply a heuristic zip while preserving appearance order.
            if all(v == 0.0 for v in physician_bias.values()):
                vcp = np.asarray(getattr(result, "vc_mean", np.array([]))).flatten()
                if len(vcp) == len(physician_labels):
                    self.LOGGER.warning(
                        "Using heuristic fallback for physician random effects (order-based zip)."
                    )
                    for p, v in zip(physician_labels, vcp):
                        if np.isfinite(v):
                            physician_bias[p] = float(v)
        except Exception as e:
            self.LOGGER.warning(f"Failed to parse physician random effects robustly, keeping defaults. Error: {e}")

        # Per-physician local metrics:
        # - observed prescription rate,
        # - estimated random intercept,
        # - local overdispersion via Pearson residuals.
        physician_metrics = []
        all_pearson_residuals: list[float] = []
        pearson_residuals_by_physician: dict[str, np.ndarray] = {}
        all_actual_values: list[float] = []
        all_fitted_values: list[float] = []
        actual_fitted_by_physician: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        epsilon = 1e-10
        grouped = df_clean.groupby(physician_col)
        for physician, group in grouped:
            n_patients = len(group)
            u_j = physician_bias.get(str(physician), 0.0)
            p_avg = group["_predicted_prob"].values
            p_avg_clipped = np.clip(p_avg, epsilon, 1 - epsilon)
            z_adjusted = np.log(p_avg_clipped / (1 - p_avg_clipped)) + u_j
            p_specific = 1 / (1 + np.exp(-z_adjusted))
            y_obs = group[outcome_col].values
            residuals = y_obs - p_specific
            variance_term = np.clip(p_specific * (1 - p_specific), epsilon, None)
            pearson_residuals = residuals / np.sqrt(variance_term)
            finite_af = np.isfinite(y_obs) & np.isfinite(p_specific)
            if np.any(finite_af):
                y_obs_finite = np.asarray(y_obs[finite_af], dtype=float)
                p_specific_finite = np.asarray(p_specific[finite_af], dtype=float)
                all_actual_values.extend(y_obs_finite.tolist())
                all_fitted_values.extend(p_specific_finite.tolist())
                actual_fitted_by_physician[str(physician)] = (
                    y_obs_finite.copy(),
                    p_specific_finite.copy(),
                )

            pearson_residuals = pearson_residuals[np.isfinite(pearson_residuals)]
            if pearson_residuals.size:
                all_pearson_residuals.extend(pearson_residuals.tolist())
                pearson_residuals_by_physician[str(physician)] = pearson_residuals.copy()
            if n_patients < min_patients:
                continue
            overdispersion = float(np.mean(pearson_residuals ** 2)) if pearson_residuals.size else np.nan
            physician_metrics.append(
                {
                    "physician": physician,
                    "n_patients": n_patients,
                    "prescription_rate": round(y_obs.mean(), 4),
                    "random_effect_bias": round(u_j, 4),
                    "overdispersion_local": round(overdispersion, 4),
                }
            )

        df_metrics = pd.DataFrame(physician_metrics)
        # Centralize GLMM outputs so downstream steps can reuse them
        # (matching + plots + final export).
        self.results["glmm"] = {
            "model": result,
            "summary": summary_text,
            "fixed_effect_weights": fixed_effect_weights,
            "variance_physician": global_variance,
            "df_with_residuals": df_clean,
            "n_physicians": df_clean[physician_col].nunique(),
            "n_observations": len(df_clean),
        }
        self.results["intra_physician_variability"] = df_metrics

        glmm_dir = self.results_dir / "glmm"
        glmm_dir.mkdir(parents=True, exist_ok=True)
        master_report = {
            "metadata": {
                "n_observations": len(df_clean),
                "n_physicians": df_clean[physician_col].nunique(),
                "fixed_effects": fixed_effects,
                "min_patients_threshold": min_patients,
            },
            "dataset_targets": self.config.get("dataset_targets", {}),
            "glmm_global_results": {
                "empirical_variance_absolute": global_variance,
                "model_summary_raw": summary_text,
                "fixed_effect_weights": fixed_effect_weights,
            },
            "physician_metrics": df_metrics.to_dict(orient="records"),
        }
        master_file_path = glmm_dir / "glmm_master_report.json"
        with open(master_file_path, "w", encoding="utf-8") as f:
            json.dump(master_report, f, indent=4, ensure_ascii=False)

        def _safe_physician_label(label: str) -> str:
            safe_label = "".join(
                ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
                for ch in str(label)
            )
            return safe_label or "unknown_physician"

        def _calibration_curve_points(
            actual_arr: np.ndarray,
            fitted_arr: np.ndarray,
            n_bins: int = 10,
            min_bin_size: int = 5,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if actual_arr.size == 0 or fitted_arr.size == 0:
                return np.array([]), np.array([]), np.array([])
            cal_df = pd.DataFrame({"actual": actual_arr, "fitted": fitted_arr}).dropna()
            if cal_df.empty:
                return np.array([]), np.array([]), np.array([])
            try:
                cal_df["bin"] = pd.qcut(cal_df["fitted"], q=min(n_bins, len(cal_df)), duplicates="drop")
            except ValueError:
                cal_df["bin"] = pd.cut(cal_df["fitted"], bins=min(n_bins, len(cal_df)), duplicates="drop")
            grouped_cal = (
                cal_df.groupby("bin", observed=False)
                .agg(mean_fitted=("fitted", "mean"), observed_rate=("actual", "mean"), n=("actual", "size"))
                .dropna()
            )
            grouped_cal = grouped_cal[grouped_cal["n"] >= min_bin_size]
            if grouped_cal.empty:
                grouped_cal = (
                    cal_df.groupby("bin", observed=False)
                    .agg(mean_fitted=("fitted", "mean"), observed_rate=("actual", "mean"), n=("actual", "size"))
                    .dropna()
                )
            if grouped_cal.empty:
                return np.array([]), np.array([]), np.array([])
            return (
                grouped_cal["mean_fitted"].to_numpy(dtype=float),
                grouped_cal["observed_rate"].to_numpy(dtype=float),
                grouped_cal["n"].to_numpy(dtype=float),
            )

        if all_pearson_residuals:
            pearson_residuals_arr = np.asarray(all_pearson_residuals, dtype=float)
            plt.figure(figsize=(10, 6))
            sns.histplot(pearson_residuals_arr, bins=50, stat="density", kde=True, color="#4C72B0", alpha=0.35)
            plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
            plt.title("Distribution of Pearson residuals (all observations)")
            plt.xlabel("Pearson residual")
            plt.ylabel("Density")
            plt.tight_layout()
            pearson_plot_path = glmm_dir / "pearson_residuals_distribution.png"
            plt.savefig(pearson_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.results["glmm"]["pearson_residuals_plot"] = str(pearson_plot_path)

        if pearson_residuals_by_physician:
            by_physician_dir = glmm_dir / "pearson_residuals_by_physician"
            by_physician_dir.mkdir(parents=True, exist_ok=True)
            by_physician_paths: dict[str, str] = {}
            for physician_label, residuals_arr in pearson_residuals_by_physician.items():
                safe_physician_label = _safe_physician_label(physician_label)
                plt.figure(figsize=(10, 6))
                sns.histplot(
                    residuals_arr,
                    bins=30,
                    stat="density",
                    kde=bool(residuals_arr.size > 1),
                    color="#55A868",
                    alpha=0.35,
                )
                plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
                plt.title(f"Pearson residuals - physician {physician_label} (n={residuals_arr.size})")
                plt.xlabel("Pearson residual")
                plt.ylabel("Density")
                plt.tight_layout()
                physician_plot_path = by_physician_dir / f"pearson_residuals_{safe_physician_label}.png"
                plt.savefig(physician_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                by_physician_paths[physician_label] = str(physician_plot_path)
            self.results["glmm"]["pearson_residuals_plots_by_physician"] = by_physician_paths

        if all_actual_values and all_fitted_values:
            actual_arr = np.asarray(all_actual_values, dtype=float)
            fitted_arr = np.asarray(all_fitted_values, dtype=float)
            x_jitter = np.random.uniform(-0.03, 0.03, size=actual_arr.size)
            plt.figure(figsize=(10, 6))
            plt.scatter(actual_arr + x_jitter, fitted_arr, alpha=0.25, s=14, color="#C44E52", edgecolors="none")
            plt.xlim(-0.1, 1.1)
            plt.ylim(-0.02, 1.02)
            plt.xticks([0, 1], ["0", "1"])
            plt.xlabel("Actual")
            plt.ylabel("Fitted")
            plt.title("Actual vs fitted probabilities (all observations)")
            plt.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            actual_fitted_plot_path = glmm_dir / "actual_vs_fitted.png"
            plt.savefig(actual_fitted_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.results["glmm"]["actual_vs_fitted_plot"] = str(actual_fitted_plot_path)

        if actual_fitted_by_physician:
            by_physician_af_dir = glmm_dir / "actual_vs_fitted_by_physician"
            by_physician_af_dir.mkdir(parents=True, exist_ok=True)
            by_physician_af_paths: dict[str, str] = {}
            for physician_label, (actual_arr, fitted_arr) in actual_fitted_by_physician.items():
                if actual_arr.size == 0 or fitted_arr.size == 0:
                    continue
                safe_physician_label = _safe_physician_label(physician_label)
                x_jitter = np.random.uniform(-0.03, 0.03, size=actual_arr.size)
                plt.figure(figsize=(10, 6))
                plt.scatter(actual_arr + x_jitter, fitted_arr, alpha=0.35, s=14, color="#8172B2", edgecolors="none")
                plt.xlim(-0.1, 1.1)
                plt.ylim(-0.02, 1.02)
                plt.xticks([0, 1], ["0", "1"])
                plt.xlabel("Actual")
                plt.ylabel("Fitted")
                plt.title(f"Actual vs fitted - physician {physician_label} (n={actual_arr.size})")
                plt.grid(axis="y", alpha=0.2)
                plt.tight_layout()
                physician_af_path = by_physician_af_dir / f"actual_vs_fitted_{safe_physician_label}.png"
                plt.savefig(physician_af_path, dpi=300, bbox_inches="tight")
                plt.close()
                by_physician_af_paths[physician_label] = str(physician_af_path)
            self.results["glmm"]["actual_vs_fitted_plots_by_physician"] = by_physician_af_paths

        if all_actual_values and all_fitted_values:
            actual_arr = np.asarray(all_actual_values, dtype=float)
            fitted_arr = np.asarray(all_fitted_values, dtype=float)
            x_curve, y_curve, n_curve = _calibration_curve_points(actual_arr, fitted_arr, n_bins=10, min_bin_size=10)
            if x_curve.size and y_curve.size:
                marker_size = np.clip(12 + np.sqrt(n_curve) * 2.5, 16, 70)
                plt.figure(figsize=(8, 8))
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Perfect calibration")
                plt.scatter(x_curve, y_curve, s=marker_size, alpha=0.85, color="#4C72B0")
                plt.plot(x_curve, y_curve, color="#4C72B0", linewidth=1.4)
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.xlabel("Mean fitted probability (bin)")
                plt.ylabel("Observed outcome rate (bin)")
                plt.title("Calibration curve (all observations)")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                calib_plot_path = glmm_dir / "calibration_curve.png"
                plt.savefig(calib_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                self.results["glmm"]["calibration_curve_plot"] = str(calib_plot_path)

        if actual_fitted_by_physician:
            by_physician_calib_dir = glmm_dir / "calibration_curve_by_physician"
            by_physician_calib_dir.mkdir(parents=True, exist_ok=True)
            by_physician_calib_paths: dict[str, str] = {}
            for physician_label, (actual_arr, fitted_arr) in actual_fitted_by_physician.items():
                x_curve, y_curve, n_curve = _calibration_curve_points(actual_arr, fitted_arr, n_bins=6, min_bin_size=5)
                if not (x_curve.size and y_curve.size):
                    continue
                safe_physician_label = _safe_physician_label(physician_label)
                marker_size = np.clip(14 + np.sqrt(n_curve) * 2.2, 16, 64)
                plt.figure(figsize=(8, 8))
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Perfect calibration")
                plt.scatter(x_curve, y_curve, s=marker_size, alpha=0.85, color="#55A868")
                plt.plot(x_curve, y_curve, color="#55A868", linewidth=1.4)
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.xlabel("Mean fitted probability (bin)")
                plt.ylabel("Observed outcome rate (bin)")
                plt.title(f"Calibration curve - physician {physician_label} (n={actual_arr.size})")
                plt.grid(alpha=0.2)
                plt.tight_layout()
                physician_calib_path = by_physician_calib_dir / f"calibration_curve_{safe_physician_label}.png"
                plt.savefig(physician_calib_path, dpi=300, bbox_inches="tight")
                plt.close()
                by_physician_calib_paths[physician_label] = str(physician_calib_path)
            self.results["glmm"]["calibration_curve_plots_by_physician"] = by_physician_calib_paths

    # --------------------------------------------------------------------------
    # PIPELINE METHODS 2–6: Matching-based discordance
    # --------------------------------------------------------------------------

    def _prepare_matching_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare shared inputs for all matching methods.

        Returns:
            X_all: numeric covariate matrix for matching.
            Y_all: binary outcome vector.
            physicians_all: physician ID vector.
            df_match: consolidated DataFrame (outcome, physician, covariates).
        """
        # Keep only rows from the analysis cohort (GLMM or matching-only) so all
        # methods work on the same population.
        df_source = self.df.loc[self._analysis_cohort_indices()].copy()
        # Target column names: clinical outcome and physician identifier.
        outcome_col, physician_col = self.config["outcome_col"], self.config["physician_col"]
        # Columns to EXCLUDE from matching:
        # - outcome_col and physician_col (not matching covariates),
        # - internal technical columns generated in the pipeline,
        # - patient identifiers that must not drive pairing.
        exclude_cols = [outcome_col, physician_col, "_predicted_prob", "_residual", "_obs_id", "member_pseudo_id", "person_id"]
        # Also exclude any "outcome_*" column to avoid introducing other outcomes
        # into the matching distance.
        exclude_cols = exclude_cols + [c for c in df_source.columns if str(c).startswith("outcome_")]
        # fixed_effects comes from config; cast to str to avoid type surprises.
        fixed_effects = [str(c) for c in self.config.get("fixed_effects", [])]
        
        # First choice: use covariates explicitly listed in fixed_effects,
        # but only if they exist, are not excluded, and are numeric/boolean.
        matching_cols = [
            c for c in fixed_effects
            if c in df_source.columns
            and c not in exclude_cols
            and (pd.api.types.is_numeric_dtype(df_source[c]) or pd.api.types.is_bool_dtype(df_source[c]))
        ]

        # Robust fallback: if fixed_effects yields nothing, take all allowed
        # numeric/boolean columns from the table.
        if len(matching_cols) == 0:
            matching_cols = [
                c for c in df_source.columns
                if c not in exclude_cols
                and (pd.api.types.is_numeric_dtype(df_source[c]) or pd.api.types.is_bool_dtype(df_source[c]))
            ]
            self.LOGGER.warning("WARNING: No explicit matching covariates in fixed_effects; using all numeric/boolean columns.")
            self.LOGGER.warning("WARNING: Columns considered: %s", ", ".join(matching_cols))

        # If there is still no matching variable, stop explicitly with a clear message.
        if len(matching_cols) == 0:
            raise ValueError(
                "No numeric or boolean covariate columns available for matching. "
                "Ensure the dataset has at least one matching covariate (e.g. fixed_effects) in addition to outcome and physician ID."
            )

        # Minimal working DataFrame: outcome, physician, and matching covariates.
        df_match = df_source[[outcome_col, physician_col] + matching_cols].copy()
        for col in matching_cols:
            # Booleans are cast to float to homogenize distance calculations.
            if pd.api.types.is_bool_dtype(df_match[col]):
                df_match[col] = df_match[col].astype(float)
            # Simple imputation: replace NaNs with the column median.
            # This avoids matching errors caused by missing values.
            if df_match[col].isna().sum() > 0:
                df_match[col] = df_match[col].fillna(df_match[col].median())
                self.LOGGER.warning("WARNING: %s column has %d missing values, filled with median.", col, df_match[col].isna().sum())

        # Output in the format expected by matching methods:
        # X_all = covariate matrix, Y_all = outcome, physicians_all = physician IDs.
        X_all = df_match[matching_cols].values
        Y_all = df_match[outcome_col].values
        physicians_all = df_match[physician_col].values
        return X_all, Y_all, physicians_all, df_match

    def _export_matched_pairs_to_csv(
        self,
        df_source: pd.DataFrame,
        ref_indices: np.ndarray,
        match_indices: np.ndarray,
        method_name: str,
        distances: np.ndarray | None = None,
    ) -> None:
        """
        Export the A/B pairs retained by a matching method.

        The CSV is mainly used for clinical review: compare two patients deemed
        "similar" and check decision consistency.

        If `distances` is provided and aligned in size with `ref_indices` /
        `match_indices`, a `pair_distance` column is inserted right after
        `physician_id`. The distance is expressed in the method's unit
        (e.g. weighted Euclidean for Euclidean, Mahalanobis for
        Mahalanobis, `1 - proximity` for Random Forest, etc.).
        """
        if len(ref_indices) == 0 or len(match_indices) == 0:
            return
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        exclude_for_export = {outcome_col, physician_col} | {c for c in df_source.columns if str(c).startswith("outcome_")}
        feature_cols = [c for c in df_source.columns if c not in exclude_for_export]
        df_A = df_source.iloc[ref_indices].reset_index(drop=True)
        df_B = df_source.iloc[match_indices].reset_index(drop=True)
        physician_id = df_A[physician_col].rename("physician_id")
        patient_A_outcome = df_A[outcome_col].rename("patient_A_outcome")
        patient_B_outcome = df_B[outcome_col].rename("patient_B_outcome")
        df_A_feat = df_A[feature_cols].add_suffix("_A")
        df_B_feat = df_B[feature_cols].add_suffix("_B")
        pieces: list[pd.Series | pd.DataFrame] = [physician_id]
        if distances is not None:
            distances_array = np.asarray(distances, dtype=float)
            if distances_array.size == len(ref_indices):
                pieces.append(pd.Series(distances_array, name="pair_distance"))
            else:
                self.LOGGER.warning(
                    "Skipping pair_distance column for %s: distances size %d != n_pairs %d.",
                    method_name, distances_array.size, len(ref_indices),
                )
        pieces.extend([patient_A_outcome, patient_B_outcome, df_A_feat, df_B_feat])
        paired = pd.concat(pieces, axis=1)
        out_dir = self.results_dir / "clinical_reviews"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{method_name}_pairs.csv"
        paired.to_csv(out_path, index=False)
        self.LOGGER.info("Clinical validation export: %s (%d pairs)", out_path, len(paired))

    def _is_glmm_ready(self) -> bool:
        """Return whether the GLMM step produced a usable result."""
        return "glmm" in self.results and "error" not in self.results["glmm"]

    def _is_matching_cohort_ready(self) -> bool:
        """Return whether an analysis cohort is ready (GLMM or matching only)."""
        if self._is_glmm_ready():
            return True
        matching_cohort = self.results.get("matching_cohort")
        return (
            isinstance(matching_cohort, dict)
            and "df_clean" in matching_cohort
            and "error" not in matching_cohort
        )

    def _analysis_cohort_indices(self) -> pd.Index:
        """Row indices retained for analysis (consistent across GLMM / matching-only)."""
        if self._is_glmm_ready():
            return self.results["glmm"]["df_with_residuals"].index
        return self.results["matching_cohort"]["df_clean"].index

    def _standardize_matrix(self, X: np.ndarray) -> np.ndarray:
        """Column-wise standardization (z-score) with a stability epsilon."""
        X = np.asarray(X, dtype=float)
        return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)

    def _merge_intra_physician_result(self, df_res: pd.DataFrame) -> None:
        """Merge a per-physician score into the main aggregated table."""
        if df_res.empty:
            return
        if "intra_physician_variability" not in self.results:
            self.results["intra_physician_variability"] = df_res.copy()
            return
        self.results["intra_physician_variability"] = pd.merge(
            self.results["intra_physician_variability"], df_res, on="physician", how="left"
        )

    def _distance_pairs_by_physician(
        self,
        y_all: np.ndarray,
        physicians_all: np.ndarray,
        metric_col: str,
        local_distance_builder: Callable[[np.ndarray], np.ndarray],
        clip_to_unit_interval: bool = False,
        pairing_fn: Callable[..., list[tuple[int, int]]] = _greedy_nearest_neighbor_pairs,
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Shared engine:
        - computes a local distance matrix per physician,
        - builds 1-to-1 pairs without replacement via `pairing_fn`,
        - computes an outcome discordance rate within pairs.

        Caliper (Rosenbaum & Rubin, 1985): if `caliper_enabled` is `True` in the
        configuration, a per-physician threshold is computed as the
        `caliper_percentile` of off-diagonal pairwise distances and passed to
        `pairing_fn`, which rejects pairs exceeding that threshold. Patients
        without an accepted match are reported via the `n_unpaired_<suffix>`
        columns.

        Returns:
            df_res: per-physician discordance + caliper bookkeeping.
            ref_flat, match_flat: flat patient indices of A / B in `df_source`.
            dist_flat: pairwise distance for each pair (same length as
                `ref_flat`), in the unit of the method's distance metric.

        Memory note: we deliberately avoid any global NxN matrix to limit
        footprint when N is large.
        """
        caliper_enabled = bool(self.config.get("caliper_enabled", False))
        caliper_percentile = float(self.config.get("caliper_percentile", 25.0))
        suffix = metric_col.replace("discordance_rate_", "", 1)
        n_paired_col = f"n_paired_{suffix}"
        n_unpaired_col = f"n_unpaired_{suffix}"
        caliper_threshold_col = f"caliper_threshold_{suffix}"

        all_ref_indices: list[int] = []
        all_match_indices: list[int] = []
        all_distances: list[float] = []
        discordance_results: list[dict[str, object]] = []
        for physician in np.unique(physicians_all):
            idx = np.where(physicians_all == physician)[0]
            if len(idx) < self.config["min_patients_per_physician"]:
                continue
            local_dists = np.asarray(local_distance_builder(idx), dtype=float)
            if local_dists.shape != (len(idx), len(idx)):
                self.LOGGER.warning(
                    "Invalid local distance shape for physician %s: expected (%d, %d), got %s. Skipping.",
                    physician, len(idx), len(idx), tuple(local_dists.shape)
                )
                continue
            if not np.all(np.isfinite(local_dists)):
                np.nan_to_num(local_dists, copy=False, nan=1e12, posinf=1e12, neginf=1e12)
            np.fill_diagonal(local_dists, np.inf)
            y_local = y_all[idx]
            n_total = int(len(idx))

            if caliper_enabled:
                threshold = _per_physician_caliper_threshold(local_dists, caliper_percentile)
                pairs = pairing_fn(local_dists, caliper_threshold=threshold)
            else:
                threshold = float("nan")
                pairs = pairing_fn(local_dists)

            n_paired = int(2 * len(pairs))
            n_unpaired = int(n_total - n_paired)
            if caliper_enabled and n_total > 0 and n_unpaired / n_total > 0.5:
                self.LOGGER.warning(
                    "Caliper q=%.0f: physician %s / %s has %d/%d (%.0f%%) patients unpaired; "
                    "discordance based on %d pairs only.",
                    caliper_percentile, physician, suffix,
                    n_unpaired, n_total, 100.0 * n_unpaired / n_total, len(pairs),
                )
            if pairs:
                all_ref_indices.extend([int(idx[i]) for i, _ in pairs])
                all_match_indices.extend([int(idx[j]) for _, j in pairs])
                all_distances.extend([float(local_dists[int(i), int(j)]) for i, j in pairs])
            score = _discordance_rate_from_pairs(y_local, pairs)
            if clip_to_unit_interval and np.isfinite(score):
                score = float(np.clip(score, 0.0, 1.0))
            discordance_results.append(
                {
                    "physician": physician,
                    metric_col: score,
                    n_paired_col: n_paired,
                    n_unpaired_col: n_unpaired,
                    caliper_threshold_col: threshold,
                }
            )
        df_res = pd.DataFrame(discordance_results)
        ref_flat = np.asarray(all_ref_indices, dtype=int)
        match_flat = np.asarray(all_match_indices, dtype=int)
        dist_flat = np.asarray(all_distances, dtype=float)
        return df_res, ref_flat, match_flat, dist_flat

    def _nearest_neighbor_argmin_by_physician(
        self,
        y_all: np.ndarray,
        physicians_all: np.ndarray,
        metric_col: str,
        local_distance_builder: Callable[[np.ndarray], np.ndarray],
        ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        "Row-wise argmin" variant (neighbor reuse allowed), used to preserve the
        historical behavior of genetic matching.

        Caliper (Rosenbaum & Rubin, 1985): if `caliper_enabled` is `True` in the
        configuration, the per-physician threshold (off-diagonal distance
        percentile `caliper_percentile`) filters out patients whose nearest
        neighbor lies beyond the threshold; those patients are excluded from the
        discordance calculation and counted via `n_unpaired_<suffix>`.
        """
        caliper_enabled = bool(self.config.get("caliper_enabled", False))
        caliper_percentile = float(self.config.get("caliper_percentile", 25.0))
        suffix = metric_col.replace("discordance_rate_", "", 1)
        n_paired_col = f"n_paired_{suffix}"
        n_unpaired_col = f"n_unpaired_{suffix}"
        caliper_threshold_col = f"caliper_threshold_{suffix}"

        all_ref_indices: list[np.ndarray] = []
        all_match_indices: list[np.ndarray] = []
        all_distances: list[np.ndarray] = []
        discordance_results: list[dict[str, object]] = []
        for physician in np.unique(physicians_all):
            idx = np.where(physicians_all == physician)[0]
            if len(idx) < self.config["min_patients_per_physician"]:
                continue
            local_dists = np.asarray(local_distance_builder(idx), dtype=float)
            if local_dists.shape != (len(idx), len(idx)):
                self.LOGGER.warning(
                    "Invalid local distance shape for physician %s: expected (%d, %d), got %s. Skipping.",
                    physician, len(idx), len(idx), tuple(local_dists.shape)
                )
                continue
            if not np.all(np.isfinite(local_dists)):
                np.nan_to_num(local_dists, copy=False, nan=1e12, posinf=1e12, neginf=1e12)
            np.fill_diagonal(local_dists, np.inf)
            y_local = y_all[idx]
            nn = np.argmin(local_dists, axis=1)
            n_total = int(len(idx))

            if caliper_enabled:
                threshold = _per_physician_caliper_threshold(local_dists, caliper_percentile)
                nn_distances = local_dists[np.arange(n_total), nn]
                kept_mask = nn_distances <= threshold
            else:
                threshold = float("nan")
                kept_mask = np.ones(n_total, dtype=bool)

            kept_idx = np.where(kept_mask)[0]
            n_paired = int(kept_mask.sum())
            n_unpaired = int(n_total - n_paired)
            if caliper_enabled and n_total > 0 and n_unpaired / n_total > 0.5:
                self.LOGGER.warning(
                    "Caliper q=%.0f: physician %s / %s has %d/%d (%.0f%%) patients unpaired; "
                    "discordance based on %d patients only.",
                    caliper_percentile, physician, suffix,
                    n_unpaired, n_total, 100.0 * n_unpaired / n_total, n_paired,
                )

            if n_paired > 0:
                all_ref_indices.append(idx[kept_idx])
                all_match_indices.append(idx[nn[kept_idx]])
                all_distances.append(local_dists[kept_idx, nn[kept_idx]].astype(float))
                disc = float(
                    np.sum(y_local[kept_idx] != y_local[nn[kept_idx]]) / n_paired
                )
            else:
                disc = float("nan")
            discordance_results.append(
                {
                    "physician": physician,
                    metric_col: disc,
                    n_paired_col: n_paired,
                    n_unpaired_col: n_unpaired,
                    caliper_threshold_col: threshold,
                }
            )
        if all_ref_indices:
            ref_flat = np.concatenate(all_ref_indices)
            match_flat = np.concatenate(all_match_indices)
            dist_flat = np.concatenate(all_distances)
        else:
            ref_flat = np.asarray([], dtype=int)
            match_flat = np.asarray([], dtype=int)
            dist_flat = np.asarray([], dtype=float)
        return pd.DataFrame(discordance_results), ref_flat, match_flat, dist_flat

    def run_discordance_matching_analysis(self) -> None:
        # Mahalanobis = Euclidean distance in a covariance-whitened space,
        # useful when covariates are correlated with each other.
        self.LOGGER.info("Starting Discordance Analysis (Mahalanobis Distance)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        X_std = self._standardize_matrix(X_all)
        cols_ok = np.std(X_all, axis=0) > 1e-9
        X_clean = X_std[:, cols_ok]
        if X_clean.shape[1] == 0:
            self.LOGGER.warning("Mahalanobis matching skipped: all covariates have zero variance.")
            return
        if X_clean.shape[1] != X_std.shape[1]:
            self.LOGGER.warning("Columns dropped because no variance from X_std: %s", np.setdiff1d(np.arange(X_std.shape[1]), np.arange(X_clean.shape[1])))

        # Pseudo-inverse for numerical robustness even with ill-conditioned covariance.
        cov_inv = np.linalg.pinv(np.cov(X_clean, rowvar=False))
        df_res, ref_flat, match_flat, dist_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_mahalanobis",
            local_distance_builder=lambda idx: cdist(
                X_clean[idx], X_clean[idx], metric="mahalanobis", VI=cov_inv
            ),
            pairing_fn=self._resolve_pairing_fn(),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "mahalanobis", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def run_genetic_matching_analysis(self) -> None:
        # More expensive method: learn optimal weights via differential evolution.
        self.LOGGER.info("Starting Discordance Analysis (Genetic Matching)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        X_std = self._standardize_matrix(X_all)
        if X_std.shape[1] == 0:
            self.LOGGER.warning("Genetic matching skipped: no covariate dimensions available.")
            return
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Genetic matching skipped: outcome has only one class (need both 0 and 1).")
            return
        effective_weights, gm_dir = self._optimize_and_export_genetic_weights(
            X_std, Y_all, df_match, "genetic_matching", "genetic_matching"
        )

        X_weighted_final = X_std * effective_weights
        # Block 4: compute comparable patient pairs.
        df_res, ref_flat, match_flat, dist_flat = self._nearest_neighbor_argmin_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_genetic",
            local_distance_builder=lambda idx: cdist(
                X_weighted_final[idx], X_weighted_final[idx], metric="euclidean"
            ),
        )

        self._export_genetic_reuse_stats(
            physicians_all=physicians_all,
            ref_flat=ref_flat,
            match_flat=match_flat,
            artifact_dir=gm_dir,
            results_key="genetic_matching",
        )

        # End of comparable patient-pair extraction.
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "genetic", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def run_genetic_mahalanobis_matching_analysis(self) -> None:
        # Variant: same optimized weights as genetic matching, but Mahalanobis distance.
        self.LOGGER.info("Starting Discordance Analysis (Genetic Matching + Mahalanobis)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        X_std = self._standardize_matrix(X_all)
        if X_std.shape[1] == 0:
            self.LOGGER.warning("Genetic Mahalanobis matching skipped: no covariate dimensions available.")
            return
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Genetic Mahalanobis matching skipped: outcome has only one class (need both 0 and 1).")
            return

        effective_weights, gm_dir = self._optimize_and_export_genetic_weights(
            X_std, Y_all, df_match, "genetic_matching_mahalanobis", "genetic_matching_mahalanobis"
        )
        X_weighted = X_std * effective_weights
        cols_ok = np.std(X_weighted, axis=0) > 1e-9
        X_clean = X_weighted[:, cols_ok]
        if X_clean.shape[1] == 0:
            self.LOGGER.warning("Genetic Mahalanobis matching skipped: all weighted covariates have zero variance.")
            self._export_genetic_reuse_stats(
                physicians_all=physicians_all,
                ref_flat=np.asarray([], dtype=int),
                match_flat=np.asarray([], dtype=int),
                artifact_dir=gm_dir,
                results_key="genetic_matching_mahalanobis",
            )
            return
        cov_inv = np.linalg.pinv(np.cov(X_clean, rowvar=False))
        df_res, ref_flat, match_flat, dist_flat = self._nearest_neighbor_argmin_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_genetic_mahalanobis",
            local_distance_builder=lambda idx: cdist(
                X_clean[idx], X_clean[idx], metric="mahalanobis", VI=cov_inv
            ),
        )
        self._export_genetic_reuse_stats(
            physicians_all=physicians_all,
            ref_flat=ref_flat,
            match_flat=match_flat,
            artifact_dir=gm_dir,
            results_key="genetic_matching_mahalanobis",
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "genetic_mahalanobis", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def _optimize_and_export_genetic_weights(
        self,
        X_std: np.ndarray,
        Y_all: np.ndarray,
        df_match: pd.DataFrame,
        artifact_subdir: str,
        results_key: str,
        ) -> tuple[np.ndarray, Path]:
        # Optimize covariate weights (bounds [0, 10]).
        result = differential_evolution(
            genetic_objective_function, [(0, 10)] * X_std.shape[1], args=(X_std, Y_all),
            maxiter=50, popsize=30, tol=0.01, workers=-1, seed=42
        )
        raw_weights = np.asarray(result.x, dtype=float).reshape(-1)
        if raw_weights.size != X_std.shape[1]:
            error_msg = (
                f"Genetic matching weight dimension mismatch: got {raw_weights.size} weights "
                f"for {X_std.shape[1]} covariates."
            )
            self.LOGGER.error(error_msg)
            raise ValueError(error_msg)

        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        feature_cols = [c for c in df_match.columns if c not in {outcome_col, physician_col}]
        if len(feature_cols) != raw_weights.size:
            self.LOGGER.warning(
                "Feature/weight mismatch while exporting genetic weights: %d features vs %d weights. "
                "Falling back to positional feature names.",
                len(feature_cols),
                raw_weights.size,
            )
            feature_cols = [f"feature_{i}" for i in range(raw_weights.size)]

        effective_weights = np.sqrt(np.abs(raw_weights) + 1e-6)
        df_weights = pd.DataFrame(
            {
                "feature": feature_cols,
                "raw_weight": raw_weights,
                "effective_weight": effective_weights,
            }
        )
        artifact_dir = self.results_dir / artifact_subdir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        gm_weights_path = artifact_dir / "gm_weights.csv"
        df_weights.to_csv(gm_weights_path, index=False)
        self.results[results_key] = {
            "weights": df_weights.copy(),
            "weights_file": str(gm_weights_path),
        }
        self.LOGGER.info("Genetic matching weights exported: %s", gm_weights_path)
        return effective_weights, artifact_dir

    def _export_genetic_reuse_stats(
        self,
        physicians_all: np.ndarray,
        ref_flat: np.ndarray,
        match_flat: np.ndarray,
        artifact_dir: Path,
        results_key: str,
        ) -> None:
        reuse_cols = [
            "physician",
            "n_reference_patients",
            "n_matched_rows",
            "n_unique_matched_patients",
            "n_reused_patients",
            "n_reused_assignments",
            "reuse_rate_assignments",
        ]
        if ref_flat.size and match_flat.size:
            reuse_rows: list[dict[str, object]] = []
            df_pairs = pd.DataFrame(
                {
                    "physician": physicians_all[ref_flat],
                    "ref_idx": ref_flat,
                    "match_idx": match_flat,
                }
            )
            for physician, grp in df_pairs.groupby("physician", sort=False):
                matched = pd.Series(grp["match_idx"].to_numpy(dtype=int)).value_counts()
                n_matched_rows = int(len(grp))
                n_unique_matched_patients = int(matched.size)
                n_reused_patients = int((matched >= 2).sum())
                n_reused_assignments = int((matched - 1).clip(lower=0).sum())
                reuse_rate_assignments = float(n_reused_assignments / n_matched_rows) if n_matched_rows > 0 else 0.0
                reuse_rows.append(
                    {
                        "physician": physician,
                        "n_reference_patients": n_matched_rows,
                        "n_matched_rows": n_matched_rows,
                        "n_unique_matched_patients": n_unique_matched_patients,
                        "n_reused_patients": n_reused_patients,
                        "n_reused_assignments": n_reused_assignments,
                        "reuse_rate_assignments": reuse_rate_assignments,
                    }
                )
            df_reuse = pd.DataFrame(reuse_rows, columns=reuse_cols)
            if not df_reuse.empty:
                df_reuse = df_reuse.sort_values(
                    "physician", kind="stable", key=lambda s: s.astype(str)
                ).reset_index(drop=True)
        else:
            df_reuse = pd.DataFrame(columns=reuse_cols)

        gm_reuse_path = artifact_dir / "patients_reused.csv"
        df_reuse.to_csv(gm_reuse_path, index=False)
        if results_key not in self.results or not isinstance(self.results[results_key], dict):
            self.results[results_key] = {}
        self.results[results_key]["reuse_stats"] = df_reuse.copy()
        self.results[results_key]["reuse_stats_file"] = str(gm_reuse_path)
        self.LOGGER.info("Genetic matching reuse stats exported: %s", gm_reuse_path)

    def run_manual_matching_analysis(self) -> None:
        # "Gold standard" based on the synthetic generation rule:
        # compare patients deemed comparable by design.
        self.LOGGER.info("Starting Discordance Analysis (Manual Clinical Thresholds)...")
        if not self.config.get("enable_manual_pairing", True):
            self.LOGGER.info("Manual matching disabled by config (enable_manual_pairing=false); skipping.")
            return
        if not self._is_matching_cohort_ready():
            return
        # Important: use the cohort-aligned source table (not df_match) so batch-specific
        # metadata columns such as generation_rule_mask remain available.
        df_source = self.df.loc[self._analysis_cohort_indices()].copy()
        outcome_col = self.config["outcome_col"]
        physician_col = self.config["physician_col"]
        if outcome_col not in df_source.columns or physician_col not in df_source.columns:
            self.LOGGER.warning(
                "Manual matching skipped: missing required columns outcome='%s' or physician='%s'.",
                outcome_col,
                physician_col,
            )
            return
        y_all = df_source[outcome_col].to_numpy()
        physicians_all = df_source[physician_col].to_numpy()
        generation_strategy = self.config.get("generation_strategy", "")
        try:
            eligible_mask = get_generation_rule_mask(df_source, generation_strategy)
        except ValueError as exc:
            self.LOGGER.warning(
                "Manual matching skipped: could not retrieve generation rule for strategy '%s' (%s).",
                generation_strategy, exc
            )
            return
        if len(eligible_mask) != len(df_source):
            self.LOGGER.warning(
                "Manual matching skipped: generation rule mask length mismatch (%d vs %d).",
                len(eligible_mask),
                len(df_source),
            )
            return
        # Boolean NxN matrix: True if both patients are in the eligible zone.
        global_match = eligible_mask[:, None] & eligible_mask[None, :]
        np.fill_diagonal(global_match, False)
        discordance_results = []
        for physician in np.unique(physicians_all):
            # Indices of patients followed by this physician.
            idx = np.where(physicians_all == physician)[0]
            # Skip physicians with too few patients (statistical stability).
            if len(idx) < self.config["min_patients_per_physician"]:
                self.LOGGER.warning("WARNING: Physician %s has less than %d patients, skipping.", physician, self.config["min_patients_per_physician"])
                continue
            # Matching submatrix restricted to this physician's patients.
            # local_match[i, j] = True if i and j (for this physician) are comparable
            # under the manual clinical rule.
            local_match = global_match[np.ix_(idx, idx)]
            # Outcomes for this physician's patients (binary vector).
            y_local = y_all[idx]
            # Number of valid pairs:
            # - local_match is symmetric (i,j) and (j,i),
            # - so divide by 2 to count each pair once.
            n_pairs_found = np.sum(local_match) / 2
            if n_pairs_found == 0:
                # No comparable pair found => undefined score (NaN).
                discordance_results.append({"physician": physician, "discordance_rate_manual": np.nan})
                self.LOGGER.warning("WARNING: No comparable pairs found for physician %s, discordance rate set to NaN.", physician)
            else:
                # Discordance = share of comparable pairs with different outcomes.
                # (y_local[:, None] != y_local) builds the NxN disagreement matrix.
                # Intersect with local_match to keep only valid pairs.
                # /2 for symmetry, then /n_pairs_found to obtain a proportion in [0,1].
                discordance_results.append({
                    "physician": physician,
                    "discordance_rate_manual": np.sum(local_match & (y_local[:, None] != y_local)) / 2 / n_pairs_found,
                })
        self._merge_intra_physician_result(pd.DataFrame(discordance_results))

    def run_random_forest_matching_analysis(self) -> None:
        # RF-proximity similarity: fraction of trees where two patients share the same leaf.
        self.LOGGER.info("Starting Discordance Analysis (Random Forest Matching)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Random Forest matching skipped: outcome has only one class.")
            return
        default_n_estimators = 300
        raw_rf_n_estimators = self.config.get("rf_n_estimators", default_n_estimators)
        rf_n_estimators = default_n_estimators
        try:
            parsed_n_estimators = float(raw_rf_n_estimators)
            if np.isfinite(parsed_n_estimators) and parsed_n_estimators > 0 and parsed_n_estimators.is_integer():
                rf_n_estimators = int(parsed_n_estimators)
            else:
                raise ValueError
        except (TypeError, ValueError):
            self.LOGGER.warning(
                "Invalid rf_n_estimators=%r in config; using default %d.",
                raw_rf_n_estimators,
                default_n_estimators,
            )
        self.LOGGER.info("Random Forest matching parameter: n_estimators=%d", rf_n_estimators)
        target_rf = np.asarray(Y_all, dtype=int)
        min_leaf = max(5, int(0.01 * len(df_match)))
        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=42, n_jobs=-1, max_depth=8, min_samples_leaf=min_leaf
        ).fit(X_all, target_rf)
        # leaf_indices[i, t] = leaf id of patient i in tree t.
        leaf_indices = rf.apply(X_all)
        n_trees = leaf_indices.shape[1]

        def _rf_local_dists(idx: np.ndarray) -> np.ndarray:
            local_leaves = leaf_indices[idx]
            local_prox = np.zeros((len(idx), len(idx)), dtype=float)
            for tree_idx in range(n_trees):
                leaves = local_leaves[:, tree_idx]
                local_prox += leaves[:, None] == leaves
            local_prox /= n_trees
            np.fill_diagonal(local_prox, -1.0)
            return 1.0 - local_prox

        df_res, ref_flat, match_flat, dist_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_rf_matching",
            local_distance_builder=_rf_local_dists,
            pairing_fn=self._resolve_pairing_fn(),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "rf_matching", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def run_learning_matching_analysis(self) -> None:
        # "Learning" variant: weight space by RF importances, then Euclidean distance.
        self.LOGGER.info("Starting Discordance Analysis (Learning Matching via RF weights)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Learning matching skipped: outcome has only one class.")
            return
        X_std = self._standardize_matrix(X_all)
        target_learning = np.asarray(Y_all, dtype=int)
        min_leaf = max(5, int(0.01 * len(df_match)))
        rf = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, max_depth=8, min_samples_leaf=min_leaf
        ).fit(X_std, target_learning)
        # Guardrails when importances are degenerate/non-finite.
        importances = np.array(rf.feature_importances_, dtype=float)
        if (not np.all(np.isfinite(importances))) or importances.sum() <= 0:
            importances = np.ones_like(importances, dtype=float)
        importances = np.maximum(importances, 1e-6)
        importances = importances / importances.sum()
        X_weighted = X_std * np.sqrt(importances)
        df_res, ref_flat, match_flat, dist_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_learning",
            local_distance_builder=lambda idx: cdist(
                X_weighted[idx], X_weighted[idx], metric="euclidean"
            ),
            pairing_fn=self._resolve_pairing_fn(),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "learning", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def run_euclidean_matching_analysis(self) -> None:
        # Euclidean distance with fixed uniform weighting.
        self.LOGGER.info("Starting Discordance Analysis (Euclidean matching)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Euclidean matching skipped: outcome has only one class.")
            return
        n_features = X_all.shape[1]
        weights = np.ones(n_features, dtype=float) / n_features
        X_std = self._standardize_matrix(X_all)
        X_weighted = X_std * np.sqrt(weights)
        df_res, ref_flat, match_flat, dist_flat = self._distance_pairs_by_physician(
            Y_all,
            physicians_all,
            "discordance_rate_euclidean",
            local_distance_builder=lambda idx: cdist(
                X_weighted[idx], X_weighted[idx], metric="euclidean"
            ),
            pairing_fn=self._resolve_pairing_fn(),
        )
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "euclidean", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def run_mutual_info_analysis(self) -> None:
        # Poidage des covariables par information mutuelle (feature -> outcome).
        self.LOGGER.info("Starting Heterogeneity Analysis (Mutual Information)...")
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, _ = self._prepare_matching_data()
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("Mutual information analysis skipped: outcome has only one class.")
            return
        X_std = self._standardize_matrix(X_all)
        try:
            mi_weights = mutual_info_classif(X_std, Y_all, random_state=42)
        except Exception as exc:
            self.LOGGER.warning("Mutual information weighting failed (%s); using equal weights.", exc)
            mi_weights = np.ones(X_std.shape[1], dtype=float)
        mi_weights = np.asarray(mi_weights, dtype=float).ravel()
        if mi_weights.size != X_std.shape[1] or (not np.all(np.isfinite(mi_weights))) or float(np.sum(mi_weights)) <= 0.0:
            mi_weights = np.ones(X_std.shape[1], dtype=float)
        mi_weights = np.maximum(mi_weights, 1e-9)
        mi_weights = mi_weights / np.sum(mi_weights)
        X_weighted = X_std * np.sqrt(mi_weights)
        df_res, _, _, _ = self._distance_pairs_by_physician(
            Y_all.astype(int),
            physicians_all,
            "discordance_rate_mutual_info",
            local_distance_builder=lambda idx: cdist(
                X_weighted[idx], X_weighted[idx], metric="euclidean"
            ),
            clip_to_unit_interval=True,
            pairing_fn=self._resolve_pairing_fn(),
        )
        self._merge_intra_physician_result(df_res)

    def run_lpa_guided_pairing_analysis(self) -> None:
        # LPA-guided pairing: global GMM on covariates, then 1-to-1 intra-physician pairing.
        # Improvements: robust scaling, multi-start n_init, wider K grid,
        # hybrid distance (latent + clinical), optimal matching (Hungarian).
        self.LOGGER.info("Starting Discordance Analysis (LPA-guided Pairing)...")
        if not self.config.get("lpa_enabled", True):
            self.LOGGER.info("LPA-guided pairing disabled by config (lpa_enabled=false); skipping.")
            return
        if not self._is_matching_cohort_ready():
            return
        X_all, Y_all, physicians_all, df_match = self._prepare_matching_data()
        if X_all.size == 0 or X_all.shape[1] == 0:
            self.LOGGER.warning("LPA-guided pairing skipped: no covariate dimensions available.")
            return
        if not np.all(np.isfinite(X_all)) or not np.all(np.isfinite(Y_all)):
            self.LOGGER.warning("LPA-guided pairing skipped: non-finite values detected in inputs.")
            return
        if len(np.unique(Y_all)) < 2:
            self.LOGGER.warning("LPA-guided pairing skipped: outcome has only one class.")
            return

        # 1) Preprocessing: optional robust scaling to limit the effect of outliers.
        use_robust = bool(self.config.get("lpa_robust_scaling", True))
        if use_robust:
            try:
                X_std = np.asarray(RobustScaler().fit_transform(X_all), dtype=float)
                # Zero-variance columns after RobustScaler (e.g. constant indicators):
                # fall back to z-score so they do not remain degenerate.
                col_std = np.std(X_std, axis=0)
                degenerate = np.where(col_std < 1e-12)[0]
                if degenerate.size:
                    fallback = self._standardize_matrix(X_all[:, degenerate])
                    X_std[:, degenerate] = fallback
            except Exception as exc:
                self.LOGGER.warning("Robust scaling failed (%s); falling back to z-score.", exc)
                X_std = self._standardize_matrix(X_all)
        else:
            X_std = self._standardize_matrix(X_all)
        if not np.all(np.isfinite(X_std)):
            self.LOGGER.warning("LPA-guided pairing skipped: standardized covariates contain non-finite values.")
            return

        n_samples = X_std.shape[0]
        if n_samples < 2:
            self.LOGGER.warning("LPA-guided pairing skipped: need at least 2 patients.")
            return

        # 2) Select the number of components by BIC on a wider grid + n_init.
        raw_min_k = self.config.get("lpa_n_components_min", 2)
        raw_max_k = self.config.get("lpa_n_components_max", 10)
        try:
            min_k = max(2, int(raw_min_k))
        except (TypeError, ValueError):
            min_k = 2
        try:
            max_k = max(min_k, int(raw_max_k))
        except (TypeError, ValueError):
            max_k = max(min_k, 10)
        max_k = min(max_k, n_samples)
        if min_k > max_k:
            self.LOGGER.warning(
                "LPA-guided pairing skipped: invalid n_components range after clipping (min=%d, max=%d).",
                min_k,
                max_k,
            )
            return

        valid_cov_types = {"full", "tied", "diag", "spherical"}
        cov_type = str(self.config.get("lpa_covariance_type", "full")).strip().lower()
        if cov_type not in valid_cov_types:
            self.LOGGER.warning(
                "Invalid lpa_covariance_type=%r; falling back to 'full'.",
                self.config.get("lpa_covariance_type"),
            )
            cov_type = "full"
        try:
            random_state = int(self.config.get("lpa_random_state", 42))
        except (TypeError, ValueError):
            random_state = 42
        try:
            n_init = max(1, int(self.config.get("lpa_n_init", 10)))
        except (TypeError, ValueError):
            n_init = 10

        best_model: GaussianMixture | None = None
        best_k: int | None = None
        best_bic = np.inf
        for k in range(min_k, max_k + 1):
            try:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=random_state,
                    n_init=n_init,
                    reg_covar=1e-6,
                )
                model.fit(X_std)
                converged = bool(getattr(model, "converged_", False))
                bic = float(model.bic(X_std))
            except Exception as exc:
                self.LOGGER.warning("LPA fit failed for k=%d (%s).", k, exc)
                continue
            self.LOGGER.info(
                "LPA candidate k=%d: converged=%s, bic=%.6f, n_init=%d",
                k, converged, bic, n_init,
            )
            if not converged:
                continue
            if not np.isfinite(bic):
                self.LOGGER.warning("LPA candidate k=%d skipped: non-finite BIC.", k)
                continue
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_k = k

        if best_model is None:
            fallback_k = 2
            if fallback_k > n_samples:
                self.LOGGER.warning("LPA-guided pairing skipped: fallback K=2 is infeasible for n=%d.", n_samples)
                return
            self.LOGGER.warning("No converged LPA candidate selected by BIC; trying fallback K=2.")
            try:
                fallback_model = GaussianMixture(
                    n_components=fallback_k,
                    covariance_type=cov_type,
                    random_state=random_state,
                    n_init=n_init,
                    reg_covar=1e-6,
                )
                fallback_model.fit(X_std)
            except Exception as exc:
                self.LOGGER.warning("LPA-guided pairing skipped: fallback K=2 fit failed (%s).", exc)
                return
            fallback_converged = bool(getattr(fallback_model, "converged_", False))
            self.LOGGER.info("LPA fallback K=2 convergence status: %s", fallback_converged)
            if not fallback_converged:
                self.LOGGER.warning("LPA-guided pairing skipped: fallback K=2 did not converge.")
                return
            best_model = fallback_model
            best_k = fallback_k
            best_bic = float(best_model.bic(X_std))

        self.LOGGER.info(
            "Selected LPA model: k=%d, covariance_type=%s, converged=%s, bic=%.6f, n_init=%d, robust_scaling=%s",
            int(best_k),
            cov_type,
            bool(getattr(best_model, "converged_", False)),
            float(best_bic),
            n_init,
            use_robust,
        )
        latent_probs = np.asarray(best_model.predict_proba(X_std), dtype=float)
        if latent_probs.shape[0] != n_samples or latent_probs.shape[1] < 2:
            self.LOGGER.warning(
                "LPA-guided pairing skipped: invalid latent probability shape %s.",
                tuple(latent_probs.shape),
            )
            return
        if not np.all(np.isfinite(latent_probs)):
            self.LOGGER.warning("LPA-guided pairing skipped: latent probabilities contain non-finite values.")
            return

        # 3) Latent distance: Euclidean on probabilities, or hybrid (latent + clinical).
        distance_metric = str(self.config.get("lpa_distance_metric", "hybrid")).strip().lower()
        if distance_metric not in {"euclidean_proba", "hybrid"}:
            self.LOGGER.warning(
                "Invalid lpa_distance_metric=%r; falling back to 'hybrid'.",
                distance_metric,
            )
            distance_metric = "hybrid"
        try:
            hybrid_alpha = float(self.config.get("lpa_hybrid_alpha", 0.5))
        except (TypeError, ValueError):
            hybrid_alpha = 0.5
        hybrid_alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))

        def _local_distance_builder(idx: np.ndarray) -> np.ndarray:
            d_latent = cdist(latent_probs[idx], latent_probs[idx], metric="euclidean")
            if distance_metric == "euclidean_proba":
                return d_latent
            d_x = cdist(X_std[idx], X_std[idx], metric="euclidean")
            # Per-physician normalization to make alpha comparable across differently scaled spaces.
            max_l = float(np.max(d_latent)) if d_latent.size else 1.0
            max_x = float(np.max(d_x)) if d_x.size else 1.0
            if max_l <= 0:
                max_l = 1.0
            if max_x <= 0:
                max_x = 1.0
            return hybrid_alpha * (d_latent / max_l) + (1.0 - hybrid_alpha) * (d_x / max_x)

        # 4) Optimal pairing (Hungarian) or legacy greedy pairing.
        pairing_strategy = str(self.config.get("lpa_pairing_strategy", "optimal")).strip().lower()
        if pairing_strategy not in {"greedy", "optimal"}:
            self.LOGGER.warning(
                "Invalid lpa_pairing_strategy=%r; falling back to 'optimal'.",
                pairing_strategy,
            )
            pairing_strategy = "optimal"
        pairing_fn = (
            _optimal_one_to_one_pairs if pairing_strategy == "optimal"
            else _greedy_nearest_neighbor_pairs
        )

        df_res, ref_flat, match_flat, dist_flat = self._distance_pairs_by_physician(
            Y_all.astype(int),
            physicians_all,
            "discordance_rate_lpa_guided_pairing",
            local_distance_builder=_local_distance_builder,
            clip_to_unit_interval=True,
            pairing_fn=pairing_fn,
        )
        self.results["lpa_guided_pairing"] = {
            "n_components": int(best_k),
            "bic": float(best_bic),
            "converged": bool(getattr(best_model, "converged_", False)),
            "n_init": int(n_init),
            "robust_scaling": bool(use_robust),
            "distance_metric": distance_metric,
            "hybrid_alpha": float(hybrid_alpha),
            "pairing_strategy": pairing_strategy,
        }
        if ref_flat.size and match_flat.size:
            self._export_matched_pairs_to_csv(
                df_match, ref_flat, match_flat, "lpa_guided_pairing", distances=dist_flat,
            )
        self._merge_intra_physician_result(df_res)

    def run_ensemble_matching_analysis(self) -> None:
        # Composite score: simple mean of available methods.
        # overdispersion_local is first projected onto [0,1] via 1-exp(-x).
        self.LOGGER.info("Computing Ensemble Matching score (including asymptotical GLMM)...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"].copy()
        if "overdispersion_local" in df.columns:
            olre = np.asarray(df["overdispersion_local"], dtype=float)
            olre = np.clip(np.where(np.isfinite(olre), olre, 0.0), 0.0, None)
            df["glmm_projected"] = 1.0 - np.exp(-olre)
        candidate_cols = [
            "glmm_projected",
            "discordance_rate_euclidean",
            "discordance_rate_mahalanobis",
            "discordance_rate_learning",
            "discordance_rate_genetic",
            "discordance_rate_genetic_mahalanobis",
            "discordance_rate_rf_matching",
            "discordance_rate_lpa_guided_pairing",
            "discordance_rate_mutual_info",
        ]
        cols = [c for c in candidate_cols if c in df.columns]
        if len(cols) == 0:
            self.LOGGER.warning("No matching columns available for Ensemble Matching; skipping.")
            return
        df["ensemble_matching"] = df[cols].mean(axis=1, skipna=True)
        if "glmm_projected" in df.columns:
            df.drop(columns=["glmm_projected"], inplace=True)
        self.results["intra_physician_variability"] = df

    def plot_glmm_vs_matching_comparison(self) -> None:
        # 4- or 5-panel figure:
        # 1) GLMM distribution,
        # 2) discordance-method distribution,
        # 3) delta vs manual method,
        # 4) Spearman correlation heatmap (overall, method vs method),
        # 5) within-group Spearman heatmap (method vs group), only if
        #    `self.results["within_group_spearman"]` is available (SCORE2 case).
        self.LOGGER.info("Generating Multi-Method comparison plots...")
        if "intra_physician_variability" not in self.results:
            return
        df = self.results["intra_physician_variability"]
        cols_map = METHOD_DISPLAY_NAMES
        existing_cols = [c for c in cols_map.keys() if c in df.columns]
        if len(existing_cols) < 2:
            return

        plot_dir = self.results_dir / "plots" / PLOT_SUBDIR_METHOD_COMPARISON
        plot_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")
        # Rank correlation (robust to non-Gaussian distributions).
        corr_matrix = df[existing_cols].corr(method='spearman').fillna(0.0)
        corr_matrix.index = [cols_map[c] for c in corr_matrix.index]
        corr_matrix.columns = [cols_map[c] for c in corr_matrix.columns]
        df_melt = df.copy()
        if "overdispersion_local" in df_melt.columns:
            df_melt["overdispersion_local"] = df_melt["overdispersion_local"].round(2)
        plot_cols = [col for col in [
            "discordance_rate_manual", "discordance_rate_euclidean", "discordance_rate_mahalanobis",
            "discordance_rate_learning", "discordance_rate_genetic", "discordance_rate_genetic_mahalanobis",
            "discordance_rate_rf_matching",
            "discordance_rate_lpa_guided_pairing", "discordance_rate_mutual_info",
            "overdispersion_local", "ensemble_matching",
        ] if col in df_melt.columns]
        df_long = pd.melt(df_melt, id_vars=["physician"], value_vars=plot_cols, var_name="Method", value_name="Score")
        method_names = cols_map
        df_long["Method"] = df_long["Method"].map(method_names)
        # Separate GLMM from matching methods to plot adapted axes.
        df_glmm = df_long[df_long["Method"] == "Generalized Mixed Model"]
        df_match = df_long[df_long["Method"] != "Generalized Mixed Model"]
        physicians_list = df_long["physician"].unique()
        within_group_df = self.results.get("within_group_spearman")
        has_within_group_panel = (
            within_group_df is not None and not within_group_df.empty
        )
        # Kendall τ_b is reported alongside Spearman ρ when available, in a
        # separate sibling heatmap. This addresses the second reviewer comment
        # ("replace the global ρ with Kendall τ_b ... to give a less inflated
        # reading"): we report both, the reader compares them directly.
        has_kendall_panel = (
            has_within_group_panel
            and within_group_df is not None
            and "kendall_tau_b" in within_group_df.columns
        )
        if has_kendall_panel:
            fig, axes = plt.subplots(
                1, 6, figsize=(36, 8),
                gridspec_kw={'width_ratios': [1.0, 2.4, 2.0, 1.2, 1.1, 1.1]},
            )
            ax_glmm, ax_match, ax_delta, ax_corr, ax_within, ax_kendall = axes
        elif has_within_group_panel:
            fig, axes = plt.subplots(
                1, 5, figsize=(33, 8),
                gridspec_kw={'width_ratios': [1.0, 2.4, 2.0, 1.2, 1.1]},
            )
            ax_glmm, ax_match, ax_delta, ax_corr, ax_within = axes
            ax_kendall = None
        else:
            fig, axes = plt.subplots(
                1, 4, figsize=(30, 8),
                gridspec_kw={'width_ratios': [1.0, 2.4, 2.0, 1.2]},
            )
            ax_glmm, ax_match, ax_delta, ax_corr = axes
            ax_within = None
            ax_kendall = None

        if not df_glmm.empty:
            if df_glmm["Score"].nunique() > 1:
                sns.violinplot(data=df_glmm, x="Method", y="Score", inner=None, color="lightgray", alpha=0.5, ax=ax_glmm)
            _scatter_hue_physician_categorical(
                ax_glmm, df_glmm, "Method", "Score", "physician",
                order=["Generalized Mixed Model"], hue_order=list(physicians_list), size=8.0, linewidth=1.0
            )
            texts_glmm = [ax_glmm.text(x=0, y=row["Score"], s=str(row["physician"]), fontsize=7, color='black', alpha=0.8)
                          for _, row in df_glmm.iterrows() if pd.notna(row["Score"])]
            # adjust_text can be verbose/noisy; suppress stdout + warnings.
            with open(os.devnull, 'w') as f, redirect_stdout(f), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjust_text(texts_glmm, ax=ax_glmm, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))
            if df_glmm["Score"].nunique() == 1:
                val = df_glmm["Score"].iloc[0]
                if pd.notna(val) and np.isfinite(val):
                    ax_glmm.set_ylim(max(0.0, float(val) - 1.0), float(val) + 1.0)
                else:
                    ax_glmm.set_ylim(0.0, 1.0)
        ax_glmm.set_title("Generalized Mixed Model", fontsize=14, fontweight='bold')
        ax_glmm.set_xlabel("")
        ax_glmm.set_ylabel("Raw Overdispersion (OLRE)", fontsize=12)
        ax_glmm.tick_params(axis='x', labelsize=12)

        if not df_match.empty:
            order_match = [method_names[c] for c in plot_cols if c != "overdispersion_local" and method_names.get(c) in df_match["Method"].values]
            if df_match["Score"].nunique() > 1:
                sns.violinplot(data=df_match, x="Method", y="Score", order=order_match, inner=None, color="lightgray", alpha=0.5, ax=ax_match)
            _scatter_hue_physician_categorical(
                ax_match, df_match, "Method", "Score", "physician",
                order=order_match, hue_order=list(physicians_list), size=8.0, linewidth=1.0
            )
            x_dict = {method: i for i, method in enumerate(order_match)}
            texts_match = [ax_match.text(x=x_dict.get(row["Method"], 0), y=row["Score"], s=str(row["physician"]), fontsize=7, color='black', alpha=0.8)
                           for _, row in df_match.iterrows() if pd.notna(row["Score"])]
            with open(os.devnull, 'w') as f, redirect_stdout(f), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjust_text(texts_match, ax=ax_match, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))
        ax_match.set_title("Pairwise Discordance Methods", fontsize=14, fontweight='bold')
        ax_match.set_xlabel("")
        ax_match.set_ylabel("Discordance Rate (auto-scaled, bounded to [0,1])", fontsize=12)
        ax_match.tick_params(axis='x', labelsize=9)
        plt.setp(ax_match.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")
        score_vals = pd.to_numeric(df_match["Score"], errors="coerce").to_numpy(dtype=float) if not df_match.empty else np.array([], dtype=float)
        finite_scores = score_vals[np.isfinite(score_vals)]
        if finite_scores.size > 0:
            observed_max = float(np.max(finite_scores))
            # Auto-scale to observed discordance while preserving valid [0,1] bounds.
            y_upper = max(0.1, min(1.0, observed_max + 0.05))
        else:
            y_upper = 1.0
        ax_match.set_ylim(0.0, y_upper)

        delta_cols = [c for c in plot_cols if c not in {"discordance_rate_manual", "overdispersion_local"} and c in df.columns]
        if "discordance_rate_manual" in df.columns and len(delta_cols) > 0:
            delta_rows = []
            manual_series = pd.to_numeric(df["discordance_rate_manual"], errors="coerce")
            for col in delta_cols:
                cur = pd.to_numeric(df[col], errors="coerce")
                delta = cur - manual_series
                for physician, val in zip(df["physician"], delta):
                    if pd.isna(val):
                        continue
                    delta_rows.append({"physician": physician, "Method": method_names.get(col, col), "Delta": float(val)})
            df_delta = pd.DataFrame(delta_rows)
        else:
            df_delta = pd.DataFrame(columns=["physician", "Method", "Delta"])

        if not df_delta.empty:
            order_delta = [method_names[c] for c in delta_cols if method_names.get(c) in df_delta["Method"].values]
            if df_delta["Delta"].nunique() > 1:
                sns.violinplot(data=df_delta, x="Method", y="Delta", order=order_delta, inner=None, color="lightgray", alpha=0.5, ax=ax_delta)
            _scatter_hue_physician_categorical(
                ax_delta, df_delta, "Method", "Delta", "physician",
                order=order_delta, hue_order=list(physicians_list), size=8.0, linewidth=1.0
            )
            x_dict_delta = {method: i for i, method in enumerate(order_delta)}
            texts_delta = [
                ax_delta.text(x=x_dict_delta.get(row["Method"], 0), y=row["Delta"], s=str(row["physician"]), fontsize=7, color='black', alpha=0.8)
                for _, row in df_delta.iterrows() if pd.notna(row["Delta"])
            ]
            with open(os.devnull, 'w') as f, redirect_stdout(f), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adjust_text(texts_delta, ax=ax_delta, expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))

            # Add mean delta directly in x-axis labels for easier visual comparison.
            mean_delta = (
                df_delta.groupby("Method", observed=False)["Delta"]
                .mean()
                .reindex(order_delta)
                .dropna()
            )
            if not mean_delta.empty:
                xticklabels_with_mean = []
                for method in order_delta:
                    if method in mean_delta.index:
                        xticklabels_with_mean.append(f"{method}\n({mean_delta[method]:+.3f})")
                    else:
                        xticklabels_with_mean.append(method)
                ax_delta.set_xticks(np.arange(len(order_delta), dtype=float))
                ax_delta.set_xticklabels(xticklabels_with_mean)
        ax_delta.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
        ax_delta.set_title("Delta vs Manual Pairing", fontsize=14, fontweight='bold')
        ax_delta.set_xlabel("")
        ax_delta.set_ylabel("Method - Manual (Discordance)", fontsize=12)
        ax_delta.tick_params(axis='x', labelsize=9)
        plt.setp(ax_delta.get_xticklabels(), rotation=42, ha="right", rotation_mode="anchor")

        sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", fmt=".2f", vmin=-1, vmax=1, ax=ax_corr, cbar_kws={"shrink": 0.5}, square=True, annot_kws={"size": 10})
        ax_corr.set_title("Spearman Correlations\n(method vs method, all 20 physicians)", fontsize=12, fontweight='bold')
        ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=10)

        global_df = self.results.get("global_correlations")
        if ax_within is not None:
            self._render_within_group_panel(
                ax_within,
                within_group_df,
                method_names,
                value_col="spearman_rho",
                title="Within-group Spearman ρ\n(method vs ground truth, n=4 per group + All)",
                global_df=global_df,
            )
        if ax_kendall is not None:
            self._render_within_group_panel(
                ax_kendall,
                within_group_df,
                method_names,
                value_col="kendall_tau_b",
                title="Within-group Kendall τ_b\n(less inflated; per-group + All)",
                global_df=global_df,
            )

        plt.suptitle("Distribution of Physician Inconsistency", fontsize=18, fontweight='bold', y=1.02)

        strategy = self.config.get("generation_strategy", "score2_five_groups_heter_patients")
        dataset_targets = self.config.get("dataset_targets", {})
        metrics_for_strategy = dataset_targets.get(strategy) if isinstance(dataset_targets, dict) else None
        if isinstance(metrics_for_strategy, dict):
            expected_summary = metrics_for_strategy.get("expected_summary", "No expected results defined.")
        elif isinstance(metrics_for_strategy, str):
            expected_summary = metrics_for_strategy
        else:
            expected_summary = "No expected results defined."
        if not isinstance(expected_summary, str):
            expected_summary = str(expected_summary)
        # Theoretical expected-value annotation from dataset_targets configuration.
        annotation_line = f"Expected Targets [Strategy: {strategy}] -> {expected_summary}"
        wrapped_annotation = textwrap.fill(annotation_line, width=100)
        fig.supxlabel(wrapped_annotation, fontweight="bold", fontsize=13, color="#333333")

        plt.tight_layout(rect=(0, 0.06, 1, 0.99))
        plt.savefig(plot_dir / "comparison_3panels.png", dpi=300, bbox_inches="tight")
        plt.close()
        self.LOGGER.info("Multi-panel comparison plot generated successfully.")

    def _render_within_group_panel(
        self,
        ax,
        within_group_df: pd.DataFrame,
        method_display_names: dict,
        *,
        value_col: str = "spearman_rho",
        title: str = "Within-group Spearman ρ\n(method vs ground truth, n=4 per group)",
        global_df: pd.DataFrame | None = None,
    ) -> None:
        """Render the (method x group) within-group rank correlation heatmap.

        Args:
            ax: matplotlib axis to draw on.
            within_group_df: long-format DataFrame produced by
                :func:`src.analysis.within_group_spearman.compute_within_group_spearman`.
            method_display_names: mapping ``raw_method -> "Display Name"``.
            value_col: which numeric column to display. Currently
                ``"spearman_rho"`` or ``"kendall_tau_b"``; both are produced
                by ``compute_within_group_spearman``. Falls back gracefully if
                the column is absent.
            title: panel title.
            global_df: per-method global rank correlations (one row per method
                with ``method, n_physicians, spearman_rho, kendall_tau_b``).
                When provided, an extra column ``"All (n=N)"`` is appended on
                the right of the heatmap with the global value for the same
                metric (``value_col``). Useful to compare within-group rank
                preservation against the overall correlation.

        Implementation notes:
        - ``dropna=False`` is critical when pivoting: groups whose ground
          truth is constant by construction (e.g., Group 1 in the SCORE2
          fixture: deterministic physicians P1..P4 with p_high=1, p_low=0)
          yield NaN for every method. The default ``dropna=True`` would
          silently remove the entire column.
        - Cells flagged ``degenerate=True`` are explicitly annotated "n/a"
          to make the structural NaN visible to the reader.
        """
        long_df = within_group_df.copy()
        if value_col not in long_df.columns:
            ax.text(0.5, 0.5, f"{value_col} not available", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_axis_off()
            return
        long_df["method_display"] = long_df["method"].map(
            lambda m: method_display_names.get(m, m)
        )
        rho_wide = long_df.pivot_table(
            index="method_display",
            columns="group",
            values=value_col,
            aggfunc="first",
            dropna=False,
        )
        ordered_methods = [
            method_display_names[col]
            for col, _ in METHOD_PLOT_SPECS
            if method_display_names.get(col) in rho_wide.index
            and method_display_names.get(col) != "Manual Pairing"
        ]
        if not ordered_methods:
            ordered_methods = list(rho_wide.index)
        all_groups = sorted(int(g) for g in long_df["group"].dropna().unique())
        rho_wide = rho_wide.reindex(index=ordered_methods, columns=all_groups)
        rho_wide.columns = [f"Grp {int(c)}" for c in rho_wide.columns]

        # Append the "All (n=N)" column from the global rank correlations,
        # when available. Maps raw method names to their display names so the
        # join works on the (already display-name-indexed) heatmap.
        all_col_name: str | None = None
        if global_df is not None and not global_df.empty and value_col in global_df.columns:
            n_global = int(global_df["n_physicians"].max())
            all_col_name = f"All (n={n_global})"
            display_to_value: dict[str, float] = {}
            for _, row_g in global_df.iterrows():
                display_name = method_display_names.get(str(row_g["method"]), str(row_g["method"]))
                value = pd.to_numeric(row_g[value_col], errors="coerce")
                display_to_value[display_name] = float(value) if pd.notna(value) else float("nan")
            rho_wide[all_col_name] = pd.Series(
                {m: display_to_value.get(m, float("nan")) for m in ordered_methods}
            )

        sns.heatmap(
            rho_wide,
            annot=True,
            cmap="RdYlGn",
            fmt=".2f",
            vmin=-1,
            vmax=1,
            ax=ax,
            cbar_kws={"shrink": 0.5},
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 9},
            mask=rho_wide.isna(),
        )

        degenerate_wide = long_df.pivot_table(
            index="method_display",
            columns="group",
            values="degenerate",
            aggfunc="first",
            dropna=False,
        ).reindex(index=ordered_methods, columns=all_groups)
        degenerate_wide.columns = [f"Grp {int(c)}" for c in degenerate_wide.columns]
        for r_idx, method in enumerate(ordered_methods):
            if method not in degenerate_wide.index:
                continue
            for c_idx, group_label in enumerate(degenerate_wide.columns):
                cell = degenerate_wide.loc[method, group_label]
                if pd.notna(cell) and bool(cell):
                    ax.text(
                        c_idx + 0.5,
                        r_idx + 0.5,
                        "n/a",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#555555",
                        fontstyle="italic",
                        fontweight="bold",
                    )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    def run_full_analysis(self) -> dict:
        # Orchestrator entry point:
        # cohort (GLMM or matching-only) -> matching -> figure -> final CSV export.
        self.LOGGER.info("=== Starting Analysis Pipeline ===")
        if self._glmm_enabled():
            self.run_glmm_analysis()
        else:
            self.LOGGER.info("GLMM disabled by config (glmm_enabled=false); running matching methods only.")
            self._prepare_matching_cohort_only()
        if self._is_matching_cohort_ready():
            # self.run_mutual_info_analysis()
            self.run_manual_matching_analysis()
            self.run_euclidean_matching_analysis()
            self.run_discordance_matching_analysis()
            self.run_learning_matching_analysis()
            # self.run_genetic_matching_analysis()  # ~7 min; not justified vs Mahalanobis
            # self.run_genetic_mahalanobis_matching_analysis() 
            # self.run_random_forest_matching_analysis()
            
            self.run_lpa_guided_pairing_analysis()
            # self.run_ensemble_matching_analysis()  # not convinced given the number of other methods
            if "intra_physician_variability" in self.results:
                self._save_global_rank_correlations()
                self._save_within_group_spearman()
                self._save_continuous_heterogeneity_metrics()
            self.plot_glmm_vs_matching_comparison()
            if "intra_physician_variability" in self.results:
                out_path = self.results_dir / "intra_physician_variability.csv"
                self.results["intra_physician_variability"].to_csv(out_path, index=False)
                self.LOGGER.info("Saved intra_physician_variability to %s", out_path)
        self.LOGGER.info("=== Analysis Pipeline Complete ===")
        return self.results

    def _save_continuous_heterogeneity_metrics(self) -> None:
        """Compute and persist discrimination metrics for the continuous-
        heterogeneity SCORE2 fixture.

        Triggered only when the upstream synthetic generator stored its
        per-physician parameters in ``self.config["physician_params"]``. For
        the legacy 5-group fixture, this key is absent and the method is a
        silent no-op.

        Outputs (next to ``intra_physician_variability.csv``):
          - ``continuous_heterogeneity_per_physician.csv``: one row per
            physician, joining true ``p_high`` / ``D*`` with method estimates.
          - ``continuous_heterogeneity_metrics.csv``: per-method Spearman rho
            against true ``D*`` and against empirical Manual rate.
          - ``plots/continuous_heterogeneity/scatter_methods_vs_true.png``:
            faceted scatter of method estimates vs true ``D*`` with the y=x
            reference line and rho printed in each panel.
        """
        from src.analysis.continuous_heterogeneity_metrics import (
            build_per_physician_table,
            compute_continuous_heterogeneity_metrics,
            save_scatter_grid,
        )

        physician_params = self.config.get("physician_params") or {}
        if not physician_params:
            return
        intra_df = self.results.get("intra_physician_variability")
        if intra_df is None or intra_df.empty:
            return

        per_physician = build_per_physician_table(intra_df, physician_params)
        per_physician_path = self.results_dir / "continuous_heterogeneity_per_physician.csv"
        per_physician.to_csv(per_physician_path, index=False)

        metrics_df = compute_continuous_heterogeneity_metrics(intra_df, physician_params)
        metrics_path = self.results_dir / "continuous_heterogeneity_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        scatter_path = (
            self.results_dir / "plots" / "continuous_heterogeneity" / "scatter_methods_vs_true.png"
        )
        # The GLMM overdispersion lives on a different (unbounded) scale than
        # the discordance rate, so the y=x reference line and a shared y-axis
        # would be misleading. Skip it from the scatter grid; its rho remains
        # available in continuous_heterogeneity_metrics.csv (rank-based, hence
        # scale-invariant).
        scatter_methods = [
            m
            for m, _ in METHOD_PLOT_SPECS
            if m not in {"discordance_rate_manual", "overdispersion_local"}
        ]
        save_scatter_grid(
            per_physician,
            method_cols=scatter_methods,
            method_display_names=METHOD_DISPLAY_NAMES,
            output_path=scatter_path,
        )
        self.results["continuous_heterogeneity_metrics"] = metrics_df
        self.results["continuous_heterogeneity_per_physician"] = per_physician
        self.LOGGER.info(
            "Saved continuous-heterogeneity discrimination metrics: %d methods, %d physicians.",
            len(metrics_df), len(per_physician),
        )

    def _save_global_rank_correlations(self) -> None:
        """Compute and persist per-method **global** rank correlations against
        the ground truth (Spearman ρ AND Kendall τ_b on all J physicians).

        This complements the within-group analysis: the "global" correlation
        corresponds to the existing Table 4 reading (n=20 in the SCORE2 5-group
        fixture, n=J in the continuous-heterogeneity fixture). Reporting both
        statistics side by side gives the reader the "less-inflated" Kendall
        reading next to the historical Spearman value.

        Output: ``<results_dir>/global_correlations.csv``. Stored back into
        ``self.results["global_correlations"]`` so the figure rendering layer
        can attach the values as an "All (n=N)" column to the within-group
        heatmaps.

        The function is silent (no error) when the
        ``intra_physician_variability`` table is missing the manual ground-
        truth column.
        """
        from src.analysis.within_group_spearman import (
            compute_global_rank_correlations,
        )

        intra_df = self.results.get("intra_physician_variability")
        if intra_df is None or intra_df.empty:
            return
        global_df = compute_global_rank_correlations(intra_df)
        if global_df.empty:
            self.LOGGER.info(
                "Global rank correlations skipped: ground-truth column missing or no methods."
            )
            return
        out_path = self.results_dir / "global_correlations.csv"
        global_df.to_csv(out_path, index=False)
        self.results["global_correlations"] = global_df
        self.LOGGER.info(
            "Saved global rank correlations (%d methods, n=%d physicians) to %s",
            len(global_df),
            int(global_df["n_physicians"].max()),
            out_path,
        )

    def _save_within_group_spearman(self) -> None:
        """Compute and persist per-(group, method) Spearman rho against the
        ground truth, when the synthetic SCORE2 5-group fixture is in effect.

        Output: ``<results_dir>/within_group_spearman.csv``. Stored back into
        ``self.results["within_group_spearman"]`` so that downstream code
        (``main._save_cross_experiment_outputs``) can aggregate it across
        experiments without re-reading the CSV.

        The function is silent (no error) when:
          - the strategy is not the 5-group SCORE2 fixture (no group structure
            to evaluate, e.g., real data or the continuous-heterogeneity
            fixture where each physician has a distinct ``p_high``),
          - or the ``intra_physician_variability`` table is missing the
            manual ground-truth column.
        """
        from src.analysis.within_group_spearman import (
            compute_within_group_spearman,
            score2_group_assignment,
        )

        # The 5-group analysis only makes sense when physicians come from the
        # discrete 5-group fixture. When `physician_params` is set, we are in
        # the continuous-heterogeneity fixture: the panel labels P1..P20 still
        # match the SCORE2 P\d+ pattern but the physicians are no longer
        # clustered into 5 homogeneous groups, so within-group rho would be
        # meaningless. Skip with an explanatory log.
        if self.config.get("physician_params"):
            self.LOGGER.info(
                "Within-group Spearman skipped: continuous-heterogeneity fixture "
                "(physicians are not clustered into 5 homogeneous groups)."
            )
            return

        intra_df = self.results.get("intra_physician_variability")
        if intra_df is None or intra_df.empty:
            return
        physicians = intra_df["physician"] if "physician" in intra_df.columns else []
        mapping = score2_group_assignment(physicians)
        if not mapping:
            self.LOGGER.info(
                "Within-group Spearman skipped: no physician id matched the SCORE2 P1..P20 convention."
            )
            return
        rho_df = compute_within_group_spearman(intra_df, group_assignment=mapping)
        if rho_df.empty:
            self.LOGGER.info("Within-group Spearman skipped: ground-truth column missing.")
            return
        out_path = self.results_dir / "within_group_spearman.csv"
        rho_df.to_csv(out_path, index=False)
        self.results["within_group_spearman"] = rho_df
        n_degenerate = int(rho_df["degenerate"].sum())
        self.LOGGER.info(
            "Saved within-group Spearman (%d rows, %d degenerate) to %s",
            len(rho_df), n_degenerate, out_path,
        )


__all__ = [
    "Analysis",
    "generate_baseline_table_one",
    "genetic_objective_function",
    "_greedy_nearest_neighbor_pairs",
    "_glmm_fixed_effect_weights",
]
