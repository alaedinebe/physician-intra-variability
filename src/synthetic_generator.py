# -*- coding: utf-8 -*-
"""Synthetic dataset generation for SCORE2 and progressive multi-rule experiments."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

SCORE2_ONLY_STRATEGY = "score2_five_groups_heter_patients"
CONTINUOUS_HETEROGENEITY_STRATEGY = "score2_continuous_heterogeneity"
LATENT_COVARIATE_STRATEGY = "score2_latent_covariate"
MULTI_RULE_STRATEGY_PREFIX = "synthetic_rule"
SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS = 10_000  # from config file, otherwise default value
SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS = 20
SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS = ((1.0, 0.0), (0.9, 0.05), (0.8, 0.1), (0.7, 0.2), (0.5, 0.5))
CONTINUOUS_DEFAULT_N_PHYSICIANS = 50
CONTINUOUS_DEFAULT_P_HIGH_LOW = 0.5
CONTINUOUS_DEFAULT_P_HIGH_HIGH = 1.0
CONTINUOUS_DEFAULT_P_LOW = 0.05
LATENT_COVARIATE_DEFAULT_PREVALENCE = 0.15
LATENT_COVARIATE_DEFAULT_BORDERLINE_WINDOW_PP = 2.0
LATENT_COVARIATE_HIDDEN_COL = "latent_covariate_z"
MULTI_RULE_DEFAULT_N_EXPERIMENTS = 100
PROGRESSIVE_RULE_DEFAULT_N_PASSES = 2

# Strategies for sampling per-variable thresholds in window-threshold rules.
# - "uniform": legacy behaviour, threshold ~ Uniform(min(col), max(col)).
#              Causes E[P(eligible)] ~ 0.5^window_size, which collapses for
#              large windows and produces experiments with no comparable pairs.
# - "target_quantile": threshold = quantile(col, p_star ** (1/window_size)),
#              with p_star drawn per-experiment from
#              Uniform(target_eligibility_low, target_eligibility_high).
#              This guarantees E[P(eligible)] = p_star regardless of window
#              size, eliminating the structural drop of long-window experiments.
THRESHOLD_STRATEGY_UNIFORM = "uniform"
THRESHOLD_STRATEGY_TARGET_QUANTILE = "target_quantile"
THRESHOLD_STRATEGY_DEFAULT = THRESHOLD_STRATEGY_UNIFORM
TARGET_ELIGIBILITY_DEFAULT_LOW = 0.2
TARGET_ELIGIBILITY_DEFAULT_HIGH = 0.8

# Minimum support required per physician at generation time.
MIN_PATIENTS_PER_PHYSICIAN_BASE = 90
DEFAULT_N_MATCHING_COVARIATES = 500
DEFAULT_N_CONTINUOUS_MATCHING_COVARIATES = 200
DEFAULT_N_CATEGORICAL_MATCHING_COVARIATES = 300
DEFAULT_CATEGORICAL_MIN_LEVELS = 2
DEFAULT_CATEGORICAL_MAX_LEVELS = 6
DEFAULT_PRESCRIPTION_MIN_CONDITIONS = 10
BASE_CLINICAL_COVARIATES: tuple[str, ...] = (
    "age",
    "biomarker_hba1c_ngsp_blood",
    "non_hdl_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "estimated_glomerular_filtration_rate",
    "score_questionnaire_HAD",
    "is_smoker",
    "is_male",
    "has_car",
)
DEFAULT_PRESCRIPTION_RULE_VARIABLES: tuple[str, ...] = BASE_CLINICAL_COVARIATES + (
    "synth_cont_001",
    "synth_cont_002",
    "synth_cont_003",
)

AGE_MEAN, AGE_STD, AGE_MIN, AGE_MAX = 60.0, 12.0, 40, 90
HBA1C_MEAN, HBA1C_STD, HBA1C_MIN, HBA1C_MAX = 6.5, 1.5, 4.0, 12.0
# Lognormal calibrated on the same target moments as the normal:
#   E[X] = exp(mu + sigma**2/2) ≈ 6.5
#   SD[X] = sqrt((exp(sigma**2) - 1) * E[X]**2) ≈ 1.5
# Right-skewed and clinically more realistic (long tail toward decompensated
# diabetes), used when `hba1c_law="lognormal"` is requested.
HBA1C_LOGNORMAL_MU = 1.846
HBA1C_LOGNORMAL_SIGMA = 0.228
HBA1C_LAW_NORMAL = "normal"
HBA1C_LAW_LOGNORMAL = "lognormal"
HBA1C_LAW_DEFAULT = HBA1C_LAW_NORMAL
NON_HDL_MMOL_MEAN, NON_HDL_MMOL_STD, NON_HDL_MMOL_MIN, NON_HDL_MMOL_MAX = 3.6, 0.95, 1.0, 8.0
HDL_MMOL_MEAN, HDL_MMOL_STD, HDL_MMOL_MIN, HDL_MMOL_MAX = 1.35, 0.38, 0.4, 2.8
SBP_MEAN, SBP_STD, SBP_MIN, SBP_MAX = 130.0, 20.0, 90, 200
LDL_MMOL_MEAN, LDL_MMOL_STD, LDL_MMOL_MIN, LDL_MMOL_MAX = 1.3, 0.35, 0.4, 2.2
DBP_MEAN, DBP_STD, DBP_MIN, DBP_MAX = 82.0, 11.0, 60, 115
HAD_MEAN_A, HAD_MEAN_B, HAD_COMPONENT_STD, HAD_MIN, HAD_MAX = 6.0, 15.0, 4.0, 2, 38
HAS_CAR_PROB, SMOKER_PROB, MALE_PROB = 0.25, 0.2, 0.6
EGFR_MEAN, EGFR_STD, EGFR_MIN, EGFR_MAX = 90.0, 25.0, 15.0, 140.0

SCORE2_DEFAULT_RISK_REGION = "Low"
_SCORE2_SCALES: dict[tuple[str, str, str], tuple[float, float]] = {
    ("Low", "young", "male"): (-0.5699, 0.7476),
    ("Low", "young", "female"): (-0.7380, 0.7019),
    ("Moderate", "young", "male"): (-0.1565, 0.8009),
    ("Moderate", "young", "female"): (-0.3143, 0.7701),
    ("High", "young", "male"): (0.3207, 0.9360),
    ("High", "young", "female"): (0.5710, 0.9369),
    ("Very high", "young", "male"): (0.5836, 0.8294),
    ("Very high", "young", "female"): (0.9412, 0.8329),
    ("Low", "old", "male"): (-0.34, 1.19),
    ("Low", "old", "female"): (-0.52, 1.01),
    ("Moderate", "old", "male"): (0.01, 1.25),
    ("Moderate", "old", "female"): (-0.1, 1.1),
    ("High", "old", "male"): (0.08, 1.15),
    ("High", "old", "female"): (0.38, 1.09),
    ("Very high", "old", "male"): (0.05, 0.7),
    ("Very high", "old", "female"): (0.38, 0.69),
}


def _validate_counts(n_patients: int, expected_n_pros: int) -> None:
    """Validate the minimum volumes required for generation."""
    if expected_n_pros < 1:
        raise ValueError("expected_n_pros must be >= 1.")
    if n_patients < 1:
        raise ValueError("n_patients must be >= 1.")


def _ensure_min_support(
    n_patients: int,
    expected_n_pros: int,
    logger: Optional[Any],
    *,
    enforce_min_support: bool = True,
) -> int:
    """Adjust `n_patients` to guarantee a minimum support per physician."""
    if not enforce_min_support:
        return n_patients
    min_total = expected_n_pros * MIN_PATIENTS_PER_PHYSICIAN_BASE
    if n_patients >= min_total:
        return n_patients
    if logger:
        logger.warning(
            "n_patients (%d) < n_physicians * %d (= %d); adjusting to %d.",
            n_patients,
            MIN_PATIENTS_PER_PHYSICIAN_BASE,
            min_total,
            min_total,
        )
    return min_total


def _validate_score2_rule_columns(df: pd.DataFrame) -> None:
    """Check the columns required by SCORE2 risk rules."""
    required = {"age", "is_smoker", "systolic_blood_pressure", "non_hdl_cholesterol", "hdl_cholesterol", "is_male"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing columns for SCORE2-style rule: %s." % ", ".join(missing))


def _professional_ids(expected_n_pros: int) -> list[str]:
    return [f"P{i}" for i in range(1, expected_n_pros + 1)]


def _sample_hba1c(
    n: int,
    rng: np.random.Generator,
    law: str = HBA1C_LAW_DEFAULT,
) -> np.ndarray:
    """Sample HbA1c values according to the requested law, then clip and round.

    Two laws are supported:
      - ``"normal"`` (default, historical): truncated normal with mean
        ``HBA1C_MEAN`` and SD ``HBA1C_STD``.
      - ``"lognormal"``: right-skewed lognormal calibrated to match the same
        marginal mean / SD, but with a long tail toward decompensated
        diabetes. Reviewer-driven (May 2026): exposes how matching methods
        behave under non-normal covariate distributions.

    Both laws are clipped to ``[HBA1C_MIN, HBA1C_MAX]``. Invalid law strings
    raise ``ValueError``.
    """
    law_str = str(law or HBA1C_LAW_DEFAULT).strip().lower()
    if law_str == HBA1C_LAW_LOGNORMAL:
        raw = rng.lognormal(
            mean=HBA1C_LOGNORMAL_MU, sigma=HBA1C_LOGNORMAL_SIGMA, size=n,
        )
    elif law_str == HBA1C_LAW_NORMAL:
        raw = rng.normal(HBA1C_MEAN, HBA1C_STD, size=n)
    else:
        raise ValueError(
            f"Unknown HbA1c law: {law!r}; expected '{HBA1C_LAW_NORMAL}' or "
            f"'{HBA1C_LAW_LOGNORMAL}'."
        )
    return np.round(np.clip(raw, HBA1C_MIN, HBA1C_MAX), 2).astype(np.float64)


def _apply_gaussian_copula_pairwise(
    df: pd.DataFrame,
    var_a: str,
    var_b: str,
    rho: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Reorder two columns of `df` to impose a target rank correlation
    `rho`, **without changing the marginals**.

    Method (Gaussian copula via rank reordering):
      1. Draw a bivariate sample N(0, [[1, rho],[rho, 1]]).
      2. Compute the ranks of the two sampled columns.
      3. Sort the original values of `var_a` and `var_b`, then reassign them
         according to the ranks above.

    This construction is exact from a marginal perspective (each variable keeps
    all of its original values, simply permuted). The observed Spearman
    correlation is close to `rho` up to `O(1/sqrt(n))`.

    Args:
        df: input DataFrame, left unchanged (the function returns a copy).
        var_a, var_b: names of the two columns to correlate. Must exist and be
            distinct.
        rho: target correlation in `[-1, 1]`. `rho == 0` returns an unchanged
            copy.
        rng: NumPy random generator for reproducibility.

    Returns:
        A copy of `df` where only `var_a` and `var_b` have been permuted.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1]; got {rho!r}.")
    if var_a not in df.columns:
        raise ValueError(f"var_a={var_a!r} not in dataframe columns.")
    if var_b not in df.columns:
        raise ValueError(f"var_b={var_b!r} not in dataframe columns.")
    if var_a == var_b:
        raise ValueError("var_a and var_b must be different columns.")

    n = len(df)
    if n < 2 or rho == 0.0:
        return df.copy()

    cov = np.array([[1.0, float(rho)], [float(rho), 1.0]], dtype=float)
    z = rng.multivariate_normal(mean=np.zeros(2, dtype=float), cov=cov, size=n)
    rank_a = np.argsort(np.argsort(z[:, 0]))
    rank_b = np.argsort(np.argsort(z[:, 1]))

    sorted_a = np.sort(df[var_a].to_numpy())
    sorted_b = np.sort(df[var_b].to_numpy())

    out = df.copy()
    out[var_a] = sorted_a[rank_a]
    out[var_b] = sorted_b[rank_b]
    return out


def _resolve_matching_covariate_split(
    n_total: int,
    n_continuous: int | None,
    n_categorical: int | None,
) -> tuple[int, int]:
    """Resolve the continuous / categorical split for `n_matching_covariates`."""
    if n_continuous is None and n_categorical is None:
        if n_total == DEFAULT_N_MATCHING_COVARIATES:
            return (
                DEFAULT_N_CONTINUOUS_MATCHING_COVARIATES,
                DEFAULT_N_CATEGORICAL_MATCHING_COVARIATES,
            )
        raise ValueError(
            f"n_matching_covariates={n_total} requires explicit "
            "n_continuous_covariates and n_categorical_covariates "
            f"(defaults apply only when n_total={DEFAULT_N_MATCHING_COVARIATES})."
        )
    if n_continuous is None or n_categorical is None:
        raise ValueError(
            "Both n_continuous_covariates and n_categorical_covariates must be "
            "provided together when overriding the default split."
        )
    n_continuous = int(n_continuous)
    n_categorical = int(n_categorical)
    if n_continuous < 0 or n_categorical < 0:
        raise ValueError(
            f"Covariate counts must be non-negative; got continuous={n_continuous}, "
            f"categorical={n_categorical}."
        )
    if n_continuous + n_categorical != int(n_total):
        raise ValueError(
            f"n_continuous_covariates ({n_continuous}) + n_categorical_covariates "
            f"({n_categorical}) must equal n_matching_covariates ({n_total})."
        )
    return n_continuous, n_categorical


def _append_mixed_matching_covariates(
    df: pd.DataFrame,
    n_continuous: int,
    n_categorical: int,
    rng: np.random.Generator,
    *,
    continuous_prefix: str = "synth_cont_",
    categorical_prefix: str = "synth_cat_",
    categorical_min_levels: int = DEFAULT_CATEGORICAL_MIN_LEVELS,
    categorical_max_levels: int = DEFAULT_CATEGORICAL_MAX_LEVELS,
) -> tuple[pd.DataFrame, list[str]]:
    """Add synthetic continuous and categorical covariates for matching."""
    if n_continuous < 0 or n_categorical < 0:
        raise ValueError(
            f"Covariate counts must be non-negative; got continuous={n_continuous}, "
            f"categorical={n_categorical}."
        )
    if categorical_min_levels < 2 or categorical_max_levels < categorical_min_levels:
        raise ValueError(
            "Require categorical_min_levels >= 2 and categorical_max_levels "
            f">= categorical_min_levels; got [{categorical_min_levels}, {categorical_max_levels}]."
        )

    out = df.copy()
    n_rows = len(out)
    pieces: list[pd.DataFrame] = []
    matching_covariates: list[str] = []

    if n_continuous > 0:
        cont_names = [f"{continuous_prefix}{i:03d}" for i in range(1, n_continuous + 1)]
        cont_values = rng.normal(0.0, 1.0, size=(n_rows, n_continuous)).astype(np.float64)
        pieces.append(pd.DataFrame(cont_values, columns=cont_names, index=out.index))
        matching_covariates.extend(cont_names)

    if n_categorical > 0:
        cat_names = [f"{categorical_prefix}{i:03d}" for i in range(1, n_categorical + 1)]
        cat_values = np.empty((n_rows, n_categorical), dtype=np.int32)
        for j in range(n_categorical):
            n_levels = int(rng.integers(categorical_min_levels, categorical_max_levels + 1))
            cat_values[:, j] = rng.integers(0, n_levels, size=n_rows, dtype=np.int32)
        pieces.append(pd.DataFrame(cat_values, columns=cat_names, index=out.index))
        matching_covariates.extend(cat_names)

    if pieces:
        out = pd.concat([out, *pieces], axis=1)
    return out, matching_covariates


def _build_prescription_eligibility_mask(
    df: pd.DataFrame,
    prescription_vars: list[str],
    min_conditions: int,
) -> np.ndarray:
    """Eligibility mask = SCORE2 moderate/high AND at least `min_conditions` rules on 15 vars."""
    missing = [c for c in prescription_vars if c not in df.columns]
    if missing:
        raise ValueError(f"Missing prescription rule variables: {missing}.")
    if not 0 <= min_conditions <= len(prescription_vars):
        raise ValueError(
            f"min_conditions must be in [0, {len(prescription_vars)}]; got {min_conditions}."
        )
    score2_ok = _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)
    n_satisfied = np.zeros(len(df), dtype=np.int32)
    for col in prescription_vars:
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or series.dtype == object:
            satisfied = series.astype(bool).to_numpy(dtype=bool)
        else:
            threshold = float(pd.to_numeric(series, errors="coerce").median(skipna=True))
            satisfied = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float) >= threshold
        n_satisfied += satisfied.astype(np.int32)
    rule_ok = n_satisfied >= int(min_conditions)
    return score2_ok & rule_ok


def _build_universal_base_cohort(
    n_patients: int,
    expected_n_pros: int,
    random_state: int,
    pairwise_correlation: tuple[str, str, float] | None = None,
    hba1c_law: str = HBA1C_LAW_DEFAULT,
    *,
    enforce_min_support: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a base synthetic cohort before applying prescription rules.

    Args:
        n_patients: number of patients to generate.
        expected_n_pros: number of physicians.
        random_state: reproducible seed.
        pairwise_correlation: if provided, tuple `(var_a, var_b, rho)` that
            imposes a rank correlation `rho` between `var_a` and `var_b` without
            changing the marginals (Gaussian copula). Useful for addressing
            criticism of independent synthetic covariates.
            Disabled by default (`None`) for backward compatibility.
        hba1c_law: sampling law for `biomarker_hba1c_ngsp_blood`.
            ``"normal"`` (default, historical) or ``"lognormal"`` (right-skewed,
            more clinically realistic). See :func:`_sample_hba1c`.

    Returns:
        A tuple `(df, professionals)` with simulated patient covariates and the
        ordered list of physician identifiers.
    """
    _validate_counts(n_patients, expected_n_pros)
    rng = np.random.default_rng(random_state)
    professionals = _professional_ids(expected_n_pros)
    if enforce_min_support:
        base_assignment = np.repeat(professionals, MIN_PATIENTS_PER_PHYSICIAN_BASE)
        if n_patients <= base_assignment.size:
            professional_col = rng.permutation(base_assignment)[:n_patients]
        else:
            remaining = n_patients - int(base_assignment.size)
            professional_col = np.concatenate(
                [base_assignment, rng.choice(professionals, size=remaining, replace=True)]
            )
            rng.shuffle(professional_col)
    else:
        professional_col = rng.choice(professionals, size=n_patients, replace=True)
        rng.shuffle(professional_col)

    had_group_b = rng.binomial(1, 0.5, n_patients).astype(bool)
    had_raw = np.where(
        had_group_b,
        rng.normal(HAD_MEAN_B, HAD_COMPONENT_STD, n_patients),
        rng.normal(HAD_MEAN_A, HAD_COMPONENT_STD, n_patients),
    )

    df = pd.DataFrame(
        {
            "member_pseudo_id": np.arange(1, n_patients + 1, dtype=np.int32),
            "professional_id": professional_col,
            "age": np.clip(rng.normal(AGE_MEAN, AGE_STD, n_patients), AGE_MIN, AGE_MAX).astype(np.int32),
            "biomarker_hba1c_ngsp_blood": _sample_hba1c(n_patients, rng, law=hba1c_law),
            "non_hdl_cholesterol": np.round(np.clip(rng.normal(NON_HDL_MMOL_MEAN, NON_HDL_MMOL_STD, n_patients), NON_HDL_MMOL_MIN, NON_HDL_MMOL_MAX), 2).astype(np.float64),
            "hdl_cholesterol": np.round(np.clip(rng.normal(HDL_MMOL_MEAN, HDL_MMOL_STD, n_patients), HDL_MMOL_MIN, HDL_MMOL_MAX), 2).astype(np.float64),
            "ldl_cholesterol": np.round(np.clip(rng.normal(LDL_MMOL_MEAN, LDL_MMOL_STD, n_patients), LDL_MMOL_MIN, LDL_MMOL_MAX), 2).astype(np.float64),
            "systolic_blood_pressure": np.clip(rng.normal(SBP_MEAN, SBP_STD, n_patients), SBP_MIN, SBP_MAX).astype(np.int32),
            "diastolic_blood_pressure": np.round(np.clip(rng.normal(DBP_MEAN, DBP_STD, n_patients), DBP_MIN, DBP_MAX)).astype(np.int32),
            "estimated_glomerular_filtration_rate": np.round(np.clip(rng.normal(EGFR_MEAN, EGFR_STD, n_patients), EGFR_MIN, EGFR_MAX), 2).astype(np.float64),
            "score_questionnaire_HAD": np.clip(np.round(had_raw).astype(np.int32), HAD_MIN, HAD_MAX),
            "is_smoker": rng.binomial(1, SMOKER_PROB, n_patients).astype(bool),
            "is_male": rng.binomial(1, MALE_PROB, n_patients).astype(bool),
            "has_car": rng.binomial(1, HAS_CAR_PROB, n_patients).astype(bool),
        }
    )
    if pairwise_correlation is not None:
        var_a, var_b, rho = pairwise_correlation
        # Dedicated RNG for the copula so the main sampling stream is unchanged.
        corr_rng = np.random.default_rng(int(random_state) + 1_337)
        df = _apply_gaussian_copula_pairwise(df, str(var_a), str(var_b), float(rho), corr_rng)
    return df, professionals


def _score2_scale_arrays(risk_region: str, age: np.ndarray, is_male: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.zeros(len(age), dtype=float)
    s2 = np.zeros(len(age), dtype=float)
    young = age < 70.0
    for sex_key, sex_bool in (("male", True), ("female", False)):
        for age_key, age_mask in (("young", young), ("old", ~young)):
            mask = (is_male == sex_bool) & age_mask
            key = (risk_region, age_key, sex_key)
            if key not in _SCORE2_SCALES:
                raise ValueError("Unknown SCORE2 risk region: %r" % risk_region)
            s1[mask], s2[mask] = _SCORE2_SCALES[key]
    return s1, s2


def score2_ten_year_risk_percent_vectorized(
    df: pd.DataFrame, risk_region: str = SCORE2_DEFAULT_RISK_REGION, diabetes: Optional[np.ndarray] = None
    ) -> np.ndarray:
    """Compute 10-year SCORE2 cardiovascular risk in a vectorized way.

    Args:
        df: patient data containing the required clinical variables.
        risk_region: SCORE2 risk region (`Low`, `Moderate`, `High`, `Very high`).
        diabetes: optional binary vector aligned with `df`.

    Returns:
        NumPy array of risk percentages, one per patient.
    """
    _validate_score2_rule_columns(df)
    n = len(df)
    age = df["age"].to_numpy(dtype=float)
    is_male = df["is_male"].to_numpy(dtype=bool)
    smk = df["is_smoker"].to_numpy(dtype=float)
    sbp = df["systolic_blood_pressure"].to_numpy(dtype=float)
    non_hdl = df["non_hdl_cholesterol"].to_numpy(dtype=float)
    hdl = df["hdl_cholesterol"].to_numpy(dtype=float)
    tc = non_hdl + hdl
    dm = np.zeros(n, dtype=float) if diabetes is None else np.asarray(diabetes, dtype=float).ravel()
    if dm.shape[0] != n:
        raise ValueError("diabetes array must match DataFrame length.")
    s1, s2 = _score2_scale_arrays(risk_region, age, is_male)
    risk_pct = np.full(n, np.nan, dtype=float)
    young = age < 70.0
    my, fy, mo, fo = is_male & young, (~is_male) & young, is_male & (~young), (~is_male) & (~young)

    if np.any(my):
        i = my
        xx = 0.3742*(age[i]-60.0)/5.0 + 0.6012*smk[i] + 0.2777*(sbp[i]-120.0)/20.0 + 0.6457*dm[i] + 0.1458*(tc[i]-6.0) + (-0.2698)*(hdl[i]-1.3)/0.5 + (-0.0755)*(age[i]-60.0)/5.0*smk[i] + (-0.0255)*(age[i]-60.0)/5.0*(sbp[i]-120.0)/20.0 + (-0.0281)*(age[i]-60.0)/5.0*(tc[i]-6.0) + 0.0426*(age[i]-60.0)/5.0*(hdl[i]-1.3)/0.5 + (-0.0983)*(age[i]-60.0)/5.0*dm[i]
        x2 = np.clip(1.0 - np.power(0.9605, np.exp(xx)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    if np.any(fy):
        i = fy
        xx = 0.4648*(age[i]-60.0)/5.0 + 0.7744*smk[i] + 0.3131*(sbp[i]-120.0)/20.0 + 0.8096*dm[i] + 0.1002*(tc[i]-6.0) + (-0.2606)*(hdl[i]-1.3)/0.5 + (-0.1088)*(age[i]-60.0)/5.0*smk[i] + (-0.0277)*(age[i]-60.0)/5.0*(sbp[i]-120.0)/20.0 + (-0.0226)*(age[i]-60.0)/5.0*(tc[i]-6.0) + 0.0613*(age[i]-60.0)/5.0*(hdl[i]-1.3)/0.5 + (-0.1272)*(age[i]-60.0)/5.0*dm[i]
        x2 = np.clip(1.0 - np.power(0.9776, np.exp(xx)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    if np.any(mo):
        i = mo
        xx = 0.0634*(age[i]-73.0) + 0.4245*dm[i] + 0.3524*smk[i] + 0.0094*(sbp[i]-150.0) + 0.0850*(tc[i]-6.0) + (-0.3564)*(hdl[i]-1.4) + (-0.0174)*(age[i]-73.0)*dm[i] + (-0.0247)*(age[i]-73.0)*smk[i] + (-0.0005)*(age[i]-73.0)*(sbp[i]-150.0) + 0.0073*(age[i]-73.0)*(tc[i]-6.0) + 0.0091*(age[i]-73.0)*(hdl[i]-1.4)
        x2 = np.clip(1.0 - np.power(0.7576, np.exp(xx - 0.0929)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    if np.any(fo):
        i = fo
        xx = 0.0789*(age[i]-73.0) + 0.6010*dm[i] + 0.4921*smk[i] + 0.0102*(sbp[i]-150.0) + 0.0605*(tc[i]-6.0) + (-0.3040)*(hdl[i]-1.4) + (-0.0107)*(age[i]-73.0)*dm[i] + (-0.0255)*(age[i]-73.0)*smk[i] + (-0.0004)*(age[i]-73.0)*(sbp[i]-150.0) + (-0.0009)*(age[i]-73.0)*(tc[i]-6.0) + 0.0154*(age[i]-73.0)*(hdl[i]-1.4)
        x2 = np.clip(1.0 - np.power(0.8082, np.exp(xx - 0.229)), 1e-12, 1.0 - 1e-12)
        risk_pct[i] = (1.0 - np.exp(-np.exp(s1[i] + s2[i] * np.log(-np.log(1.0 - x2))))) * 100.0
    return risk_pct


def _score2_at_least_moderate_risk_mask(age: np.ndarray, risk_pct: np.ndarray) -> np.ndarray:
    a = age.astype(float)
    r = np.asarray(risk_pct, dtype=float)
    return ((a < 50.0) & (r >= 2.5)) | ((a >= 50.0) & (a <= 69.0) & (r >= 5.0)) | ((a > 69.0) & (r >= 7.5))


def _score2_borderline_mask(
    age: np.ndarray, risk_pct: np.ndarray, window_pp: float = LATENT_COVARIATE_DEFAULT_BORDERLINE_WINDOW_PP,
) -> np.ndarray:
    """Identify patients whose 10-year SCORE2 risk falls in a ±window_pp band
    around their age-stratum threshold.

    These are the patients on the fence between "non-eligible" and "eligible"
    under the standard SCORE2 rule. The latent-covariate experiment uses Z to
    decide eligibility within this band; outside the band, eligibility is
    determined unambiguously by SCORE2 risk alone.

    Thresholds (from ESC 2021 SCORE2/SCORE2-OP guidelines):
      - age < 50           : 2.5%
      - 50 <= age <= 69    : 5.0%
      - age > 69           : 7.5%

    Args:
        age: array of patient ages.
        risk_pct: array of SCORE2 10-year risk percentages.
        window_pp: half-width of the borderline window in percentage points.

    Returns:
        Boolean mask: ``True`` where the patient is in the borderline band.
    """
    a = age.astype(float)
    r = np.asarray(risk_pct, dtype=float)
    w = float(window_pp)
    young = a < 50.0
    mid = (a >= 50.0) & (a <= 69.0)
    old = a > 69.0
    return (
        (young & (r >= 2.5 - w) & (r <= 2.5 + w))
        | (mid & (r >= 5.0 - w) & (r <= 5.0 + w))
        | (old & (r >= 7.5 - w) & (r <= 7.5 + w))
    )


def _calculate_score2_moderate_or_high_vectorized(df: pd.DataFrame) -> pd.Series:
    risk_pct = score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION)
    return pd.Series(_score2_at_least_moderate_risk_mask(df["age"].to_numpy(dtype=float), risk_pct), index=df.index)


def build_progressive_rule_plan(
    fixed_effects: list[str],
    n_passes: int = PROGRESSIVE_RULE_DEFAULT_N_PASSES,
) -> list[dict[str, Any]]:
    """
    Build the exact progressive rule schedule:
      exp 1 = SCORE2
      pass 1 windows size 1..N, sliding
      pass 2 idem, with different thresholds at generation time.
    """
    plan: list[dict[str, Any]] = [{"rule_type": "score2", "window_size": 0, "window_start": 0, "pass_index": 0, "variables": []}]
    n_features = len(fixed_effects)
    for pass_index in range(1, int(n_passes) + 1):
        for window_size in range(1, n_features + 1):
            for window_start in range(0, n_features - window_size + 1):
                variables = fixed_effects[window_start : window_start + window_size]
                plan.append(
                    {
                        "rule_type": "window_threshold",
                        "window_size": window_size,
                        "window_start": window_start,
                        "pass_index": pass_index,
                        "variables": variables,
                    }
                )
    return plan


def expected_progressive_rule_count(n_fixed_effects: int, n_passes: int = PROGRESSIVE_RULE_DEFAULT_N_PASSES) -> int:
    """Return the theoretical number of experiments in the progressive plan."""
    return 1 + int(n_passes) * (n_fixed_effects * (n_fixed_effects + 1) // 2)


def _sample_threshold_for_column(values: np.ndarray, rng: np.random.Generator) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    col_min = float(np.min(arr))
    col_max = float(np.max(arr))
    if col_max <= col_min:
        return col_min
    unique_vals = np.unique(arr)
    if unique_vals.size <= 2:
        return float(rng.choice(unique_vals))
    return float(rng.uniform(col_min, col_max))


def _sample_quantile_threshold_for_column(values: np.ndarray, q: float) -> float:
    """Return the q-th quantile of ``values`` to be used as a "<= τ" threshold.

    For continuous variables this guarantees ``Pr(x <= τ) ≈ q`` (up to ties).
    For binary/few-unique-values columns, ``np.quantile`` gracefully returns
    one of the boundary values; the actual achieved fraction may differ from
    ``q`` because the variable can only realise a finite set of probabilities.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    col_min = float(np.min(arr))
    col_max = float(np.max(arr))
    if col_max <= col_min:
        return col_min
    q_clipped = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(arr, q_clipped))


def _build_group_probability_maps(professionals: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    group_boundaries = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20))
    p_high: dict[str, float] = {}
    p_low: dict[str, float] = {}
    for group_index, (lo, hi) in enumerate(group_boundaries):
        hi_p, lo_p = SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS[group_index]
        for pid in professionals[lo:hi]:
            p_high[pid] = hi_p
            p_low[pid] = lo_p
    return p_high, p_low


def _sanitize_strategy_token(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name)).strip("_").lower()


def get_generation_rule_mask(df: pd.DataFrame, strategy: str) -> np.ndarray:
    """Return the eligibility mask of the rule used to generate the dataset.

    Args:
        df: synthetic dataset.
        strategy: generation strategy name.

    Returns:
        Boolean array indicating patients in the "treatable" zone according to
        the generation rule.
    """
    if strategy == LATENT_COVARIATE_STRATEGY:
        # The latent-covariate fixture stores the *modified* eligibility (incl.
        # the hidden Z) in the dataframe under "generation_rule_mask". Manual
        # Pairing must use this mask to remain a valid ground truth.
        if "generation_rule_mask" not in df.columns:
            raise ValueError(
                f"Strategy {LATENT_COVARIATE_STRATEGY!r} requires a "
                "'generation_rule_mask' column in the dataframe."
            )
        return df["generation_rule_mask"].to_numpy(dtype=bool)
    if strategy == CONTINUOUS_HETEROGENEITY_STRATEGY:
        if "generation_rule_mask" in df.columns:
            return df["generation_rule_mask"].to_numpy(dtype=bool)
        _validate_score2_rule_columns(df)
        return _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)
    if strategy == SCORE2_ONLY_STRATEGY:
        _validate_score2_rule_columns(df)
        return _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)
    if "generation_rule_mask" in df.columns:
        return df["generation_rule_mask"].to_numpy(dtype=bool)
    raise ValueError(
        "Unknown synthetic generation strategy: %r. Supported: %s, %s, %s, or datasets containing 'generation_rule_mask'."
        % (strategy, SCORE2_ONLY_STRATEGY, CONTINUOUS_HETEROGENEITY_STRATEGY, LATENT_COVARIATE_STRATEGY)
    )


def generate_progressive_rule_experiment(
    rule_definition: dict[str, Any],
    experiment_index: int,
    fixed_effects: list[str],
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    logger: Optional[Any] = None,
    threshold_strategy: str = THRESHOLD_STRATEGY_DEFAULT,
    target_eligibility_low: float = TARGET_ELIGIBILITY_DEFAULT_LOW,
    target_eligibility_high: float = TARGET_ELIGIBILITY_DEFAULT_HIGH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one synthetic experiment for the given progressive rule definition.

    Args:
        threshold_strategy: How to sample per-variable thresholds. ``"uniform"``
            keeps the legacy ``Uniform(min, max)`` behaviour; ``"target_quantile"``
            picks ``τ_k = quantile(x_k, p_star ** (1/window_size))`` so that the
            joint eligibility ``Pr(eligible) ≈ p_star`` regardless of the
            window size, where ``p_star`` is drawn per-experiment from
            ``Uniform(target_eligibility_low, target_eligibility_high)``.
        target_eligibility_low / target_eligibility_high: Range from which
            ``p_star`` is drawn. Only used when ``threshold_strategy ==
            "target_quantile"``. Both must lie in ``(0, 1]`` with low <= high.
    """
    _validate_counts(n_patients, expected_n_pros)
    if expected_n_pros != 20:
        raise ValueError(
            "generate_progressive_rule_experiment requires expected_n_pros=20 (got %d)."
            % expected_n_pros
        )
    if threshold_strategy not in (THRESHOLD_STRATEGY_UNIFORM, THRESHOLD_STRATEGY_TARGET_QUANTILE):
        raise ValueError(
            f"Unknown threshold_strategy={threshold_strategy!r}; "
            f"expected one of ({THRESHOLD_STRATEGY_UNIFORM!r}, {THRESHOLD_STRATEGY_TARGET_QUANTILE!r})."
        )
    if threshold_strategy == THRESHOLD_STRATEGY_TARGET_QUANTILE:
        if not (0.0 < target_eligibility_low <= target_eligibility_high <= 1.0):
            raise ValueError(
                f"Invalid target eligibility range "
                f"({target_eligibility_low}, {target_eligibility_high}); "
                "expected 0 < low <= high <= 1."
            )
    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    base_seed = int(random_state) + int(experiment_index) * 1_009
    df, professionals = _build_universal_base_cohort(n_patients, expected_n_pros, base_seed)
    df["score2_ten_year_cvd_risk_percent"] = np.round(
        score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION), 2
    )
    rule_type = str(rule_definition.get("rule_type", "window_threshold"))
    pass_index = int(rule_definition.get("pass_index", 0))
    window_size = int(rule_definition.get("window_size", 0))
    window_start = int(rule_definition.get("window_start", 0))
    variables = [str(c) for c in rule_definition.get("variables", [])]

    threshold_rng = np.random.default_rng(base_seed + 20_003 + pass_index * 7_001 + window_size * 199 + window_start * 17)
    thresholds: dict[str, float] = {}
    target_eligibility: Optional[float] = None
    per_variable_quantile: Optional[float] = None
    if rule_type == "score2":
        eligibility_mask = _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)
        rule_name = "score2"
        rule_description = "SCORE2 moderate/high risk"
        generation_strategy_name = SCORE2_ONLY_STRATEGY
    else:
        if not variables:
            raise ValueError("Window-threshold rule must include at least one variable.")
        if threshold_strategy == THRESHOLD_STRATEGY_TARGET_QUANTILE:
            # Draw the target joint eligibility for this experiment, then
            # invert through the window size so that, under independence,
            # Pr(eligible) ≈ p_star regardless of window_size. The same
            # threshold_rng is consumed first for p_star and then for any
            # downstream randomness, keeping the per-experiment seed stable.
            if target_eligibility_low == target_eligibility_high:
                target_eligibility = float(target_eligibility_low)
            else:
                target_eligibility = float(
                    threshold_rng.uniform(target_eligibility_low, target_eligibility_high)
                )
            per_variable_quantile = float(target_eligibility ** (1.0 / max(int(window_size), 1)))
        masks: list[np.ndarray] = []
        for col in variables:
            if col not in df.columns:
                raise ValueError(f"Missing fixed effect column in generated dataset: '{col}'.")
            if threshold_strategy == THRESHOLD_STRATEGY_TARGET_QUANTILE:
                threshold = _sample_quantile_threshold_for_column(
                    df[col].to_numpy(), per_variable_quantile  # type: ignore[arg-type]
                )
            else:
                threshold = _sample_threshold_for_column(df[col].to_numpy(), threshold_rng)
            thresholds[col] = threshold
            masks.append(df[col].to_numpy(dtype=float) <= threshold)
        eligibility_mask = np.logical_and.reduce(masks) if masks else np.zeros(len(df), dtype=bool)
        vars_token = "_".join(_sanitize_strategy_token(col) for col in variables)
        rule_name = f"w{window_size}_{vars_token}_pass{pass_index}"
        threshold_str = ", ".join(f"{k} <= {v:.4g}" for k, v in thresholds.items())
        if threshold_strategy == THRESHOLD_STRATEGY_TARGET_QUANTILE:
            rule_description = (
                f"All true among: {threshold_str} "
                f"(target p*={target_eligibility:.3f}, per-variable q={per_variable_quantile:.3f})"
            )
        else:
            rule_description = f"All true among: {threshold_str}"
        generation_strategy_name = f"{MULTI_RULE_STRATEGY_PREFIX}_{rule_name}"

    p_high, p_low = _build_group_probability_maps(professionals)

    p_vec = np.where(
        eligibility_mask,
        df["professional_id"].map(p_high).to_numpy(dtype=float),
        df["professional_id"].map(p_low).to_numpy(dtype=float),
    )
    outcome_rng = np.random.default_rng(base_seed + 20_000)
    strategy_name = f"exp_{experiment_index:03d}_{rule_name}"
    outcome_col = f"outcome_{strategy_name}"
    df[outcome_col] = outcome_rng.binomial(1, p_vec).astype(np.int32)
    df["recommendation"] = df[outcome_col]
    df["generation_rule_mask"] = np.asarray(eligibility_mask, dtype=bool)
    df["generation_rule_name"] = rule_name
    df["generation_rule_description"] = rule_description
    df["generation_rule_pass_index"] = pass_index
    df["generation_rule_window_size"] = window_size

    expected_summary = (
        f"Rule '{rule_name}' ({rule_description}) with fixed five-group prescription probabilities "
        "(100/0, 90/5, 80/10, 70/20, 50/50)."
    )
    if logger:
        logger.info(
            "Synthetic experiment %03d generated with strategy '%s' (%s).",
            experiment_index,
            strategy_name,
            rule_description,
        )
    return df, {
        "strategy_name": strategy_name,
        "generation_strategy": generation_strategy_name,
        "expected_summary": expected_summary,
        "outcome_col": outcome_col,
        "rule_name": rule_name,
        "rule_description": rule_description,
        "rule_thresholds": thresholds,
        "pass_index": pass_index,
        "window_size": window_size,
        "window_start": window_start,
        "variables": variables,
        "threshold_strategy": threshold_strategy,
        "target_eligibility": target_eligibility,
        "per_variable_quantile": per_variable_quantile,
    }


def generate_multi_rule_prescription_experiment(
    experiment_index: int,
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    logger: Optional[Any] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backward-compatible wrapper using progressive rule generation with defaults."""
    default_fixed_effects = [
        "age",
        "is_male",
        "is_smoker",
        "ldl_cholesterol",
        "non_hdl_cholesterol",
        "hdl_cholesterol",
        "systolic_blood_pressure",
        "biomarker_hba1c_ngsp_blood",
        "estimated_glomerular_filtration_rate",
    ]
    plan = build_progressive_rule_plan(default_fixed_effects, n_passes=PROGRESSIVE_RULE_DEFAULT_N_PASSES)
    if experiment_index < 0 or experiment_index >= len(plan):
        raise ValueError(f"experiment_index={experiment_index} out of bounds for default progressive plan ({len(plan)}).")
    return generate_progressive_rule_experiment(
        rule_definition=plan[experiment_index],
        experiment_index=experiment_index + 1,
        fixed_effects=default_fixed_effects,
        n_patients=n_patients,
        expected_n_pros=expected_n_pros,
        random_state=random_state,
        logger=logger,
    )


def generate_score2_five_groups_heter_patients(
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    logger: Optional[Any] = None,
    pairwise_correlation: tuple[str, str, float] | None = None,
    hba1c_law: str = HBA1C_LAW_DEFAULT,
    ) -> pd.DataFrame:
    """Generate a synthetic dataset with 20 physicians and 5 SCORE2-based categories.

    Args:
        pairwise_correlation: optional `(var_a, var_b, rho)` to inject a
            Spearman-style correlation between two covariates while preserving
            marginal distributions (Gaussian copula). Defaults to ``None``
            (i.e. all covariates are sampled independently, as historically).
        hba1c_law: distribution used for ``biomarker_hba1c_ngsp_blood``.
            ``"normal"`` (default) or ``"lognormal"`` for a clinically
            realistic right-skewed marginal.
    """
    _validate_counts(n_patients, expected_n_pros)
    if expected_n_pros != 20:
        raise ValueError(
            "score2_five_groups_heter_patients requires expected_n_pros=20 to keep fixed groups P1..P20 (got %d)."
            % expected_n_pros
        )
    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    df, professionals = _build_universal_base_cohort(
        n_patients, expected_n_pros, random_state,
        pairwise_correlation=pairwise_correlation,
        hba1c_law=hba1c_law,
    )
    df["score2_ten_year_cvd_risk_percent"] = np.round(
        score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION), 2
    )
    score2_indication = _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)

    group_boundaries = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20))
    p_high: dict[str, float] = {}
    p_low: dict[str, float] = {}
    for group_index, (lo, hi) in enumerate(group_boundaries):
        hi_p, lo_p = SCORE2_FIVE_CATEGORIES_CLUSTER_PROBS[group_index]
        for pid in professionals[lo:hi]:
            p_high[pid] = hi_p
            p_low[pid] = lo_p

    rng = np.random.default_rng(random_state + 40)
    p_vec = np.where(
        score2_indication,
        df["professional_id"].map(p_high).to_numpy(dtype=float),
        df["professional_id"].map(p_low).to_numpy(dtype=float),
    )
    outcome_col = "outcome_score2_five_groups_heter_patients"
    df[outcome_col] = rng.binomial(1, p_vec).astype(np.int32)
    df["recommendation"] = df[outcome_col]

    if logger:
        logger.info(
            "Synthetic dataset '%s' generated: %d patients, %d physicians "
            "(5 physician categories under SCORE2: group1 100%%/0%%, group2 90%%/5%%, "
            "group3 80%%/10%%, group4 70%%/20%%, group5 50%%/50%%).",
            SCORE2_ONLY_STRATEGY,
            len(df),
            expected_n_pros,
        )
    return df


def generate_score2_continuous_heterogeneity(
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = CONTINUOUS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    p_high_low: float = CONTINUOUS_DEFAULT_P_HIGH_LOW,
    p_high_high: float = CONTINUOUS_DEFAULT_P_HIGH_HIGH,
    p_low_fixed: float = CONTINUOUS_DEFAULT_P_LOW,
    logger: Optional[Any] = None,
    pairwise_correlation: tuple[str, str, float] | None = None,
    hba1c_law: str = HBA1C_LAW_DEFAULT,
    *,
    enforce_min_support: bool = True,
    n_matching_covariates: int | None = None,
    n_continuous_covariates: int | None = None,
    n_categorical_covariates: int | None = None,
    prescription_rule_variables: list[str] | None = None,
    prescription_min_conditions: int = DEFAULT_PRESCRIPTION_MIN_CONDITIONS,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], list[str]]:
    """Generate a SCORE2 cohort with **continuous** physician heterogeneity.

    Unlike :func:`generate_score2_five_groups_heter_patients`, which assigns
    each physician to one of 5 fixed (p_high, p_low) groups, this fixture
    draws each physician's ``p_high`` independently and uniformly from
    ``[p_high_low, p_high_high]`` while keeping ``p_low`` fixed for all
    physicians. The motivation is to test rank discrimination "at the margin"
    on a continuous spectrum of heterogeneity rather than on 5 well-separated
    discrete groups.

    Mathematical detail: the within-eligible discordance rate is
    ``D* = 2 * p_high * (1 - p_high)``, which is non-monotone on ``[0, 1]``
    (peak at 0.5). Constraining ``p_high`` to ``[0.5, 1.0]`` makes ``D*``
    monotonically *decreasing* with ``p_high``, so ranking by ``p_high`` is
    equivalent to ranking by physician consistency. This is consistent with
    the existing 5-group fixture where p_high also ranges over ``[0.5, 1.0]``.

    Args:
        n_patients: Number of patients in the synthetic cohort.
        expected_n_pros: Number of physicians ``J``. Each physician will be
            assigned a unique ``p_high`` drawn uniformly from
            ``[p_high_low, p_high_high]``. Default 50.
        random_state: Master seed used to initialize three independent RNGs
            (cohort generation, ``p_high`` draws, outcome Bernoulli draws) so
            that varying any of the three sources of randomness independently
            is possible without re-implementing the function.
        p_high_low, p_high_high: Closed-interval bounds for the uniform draw
            of each physician's ``p_high``. Defaults ``[0.5, 1.0]``. Must
            satisfy ``0 <= low <= high <= 1``.
        p_low_fixed: Single ``p_low`` value applied to every physician. Default
            ``0.05`` (low baseline prescription rate when not indicated).
        logger: Optional logger; receives an INFO line summarising the run.

    Returns:
        Tuple ``(df, physician_params, matching_covariates)``:
            - ``df``: synthetic cohort, with the standard SCORE2 outcome
              column ``outcome_score2_continuous_heterogeneity`` and a
              mirror ``recommendation`` column.
            - ``physician_params``: dict mapping each physician id to a small
              dict ``{"p_high": float, "p_low": float, "expected_discordance":
              float}``. The expected_discordance equals
              ``2 * p_high * (1 - p_high)``. This metadata is used downstream
              by the analysis pipeline to compute discrimination metrics
              against the *true* ``p_high`` (not just empirical rankings).
            - ``matching_covariates``: ordered list of covariate column names
              for matching/GLMM when ``n_matching_covariates`` is set (200
              continuous ``synth_cont_*`` + 300 categorical ``synth_cat_*`` by
              default when ``n_matching_covariates=500``); empty list otherwise
              (legacy behaviour).
    """
    _validate_counts(n_patients, expected_n_pros)
    if not 0.0 <= p_high_low <= p_high_high <= 1.0:
        raise ValueError(
            f"Require 0 <= p_high_low ({p_high_low}) <= p_high_high "
            f"({p_high_high}) <= 1."
        )
    if not 0.0 <= p_low_fixed <= 1.0:
        raise ValueError(f"p_low_fixed must be in [0, 1]; got {p_low_fixed!r}.")

    n_patients = _ensure_min_support(
        n_patients, expected_n_pros, logger, enforce_min_support=enforce_min_support,
    )
    df, professionals = _build_universal_base_cohort(
        n_patients, expected_n_pros, random_state,
        pairwise_correlation=pairwise_correlation,
        hba1c_law=hba1c_law,
        enforce_min_support=enforce_min_support,
    )

    matching_covariates: list[str] = []
    use_extended_covariates = n_matching_covariates is not None and int(n_matching_covariates) > 0
    if use_extended_covariates:
        n_continuous, n_categorical = _resolve_matching_covariate_split(
            int(n_matching_covariates),
            n_continuous_covariates,
            n_categorical_covariates,
        )
        rng_cov = np.random.default_rng(random_state + 55)
        df, matching_covariates = _append_mixed_matching_covariates(
            df,
            n_continuous,
            n_categorical,
            rng_cov,
        )

    df["score2_ten_year_cvd_risk_percent"] = np.round(
        score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION), 2
    )

    if use_extended_covariates:
        rule_vars = (
            list(prescription_rule_variables)
            if prescription_rule_variables is not None
            else list(DEFAULT_PRESCRIPTION_RULE_VARIABLES)
        )
        eligibility_mask = _build_prescription_eligibility_mask(
            df, rule_vars, prescription_min_conditions,
        )
        df["generation_rule_mask"] = eligibility_mask
    else:
        eligibility_mask = _calculate_score2_moderate_or_high_vectorized(df).to_numpy(dtype=bool)

    rng_phys = np.random.default_rng(random_state + 70)
    p_high_draws = rng_phys.uniform(low=p_high_low, high=p_high_high, size=expected_n_pros)
    physician_params: dict[str, dict[str, float]] = {}
    p_high_map: dict[str, float] = {}
    p_low_map: dict[str, float] = {}
    for pid, p_h in zip(professionals, p_high_draws):
        p_high_map[pid] = float(p_h)
        p_low_map[pid] = float(p_low_fixed)
        physician_params[pid] = {
            "p_high": float(p_h),
            "p_low": float(p_low_fixed),
            "expected_discordance": float(2.0 * p_h * (1.0 - p_h)),
        }

    rng_outcome = np.random.default_rng(random_state + 80)
    p_vec = np.where(
        eligibility_mask,
        df["professional_id"].map(p_high_map).to_numpy(dtype=float),
        df["professional_id"].map(p_low_map).to_numpy(dtype=float),
    )
    outcome_col = "outcome_score2_continuous_heterogeneity"
    df[outcome_col] = rng_outcome.binomial(1, p_vec).astype(np.int32)
    df["recommendation"] = df[outcome_col]

    if logger:
        logger.info(
            "Synthetic dataset '%s' generated: %d patients, %d physicians, "
            "p_high ~ Uniform(%.2f, %.2f), p_low fixed at %.2f%s.",
            CONTINUOUS_HETEROGENEITY_STRATEGY,
            len(df),
            expected_n_pros,
            p_high_low,
            p_high_high,
            p_low_fixed,
            f", {len(matching_covariates)} matching covariates "
            f"({sum(1 for c in matching_covariates if c.startswith('synth_cont_'))} continuous, "
            f"{sum(1 for c in matching_covariates if c.startswith('synth_cat_'))} categorical)"
            if matching_covariates
            else "",
        )
    return df, physician_params, matching_covariates


def generate_score2_latent_covariate(
    n_patients: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PATIENTS,
    expected_n_pros: int = SCORE2_FIVE_GROUPS_DEFAULT_N_PHYSICIANS,
    random_state: int = 42,
    z_prevalence: float = LATENT_COVARIATE_DEFAULT_PREVALENCE,
    borderline_window_pp: float = LATENT_COVARIATE_DEFAULT_BORDERLINE_WINDOW_PP,
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    """Generate a SCORE2 cohort with a **latent (hidden)** binary covariate
    that drives eligibility for borderline-risk patients.

    Sensitivity-analysis fixture for the second reviewer comment: the blind
    matching methods only see the observable covariates and therefore *cannot*
    distinguish two borderline patients who differ only on the hidden Z.
    They will judge them as comparable, treat their differing prescriptions as
    intra-physician variability, and over-estimate discordance accordingly.
    The Manual Pairing ground truth uses the full eligibility rule (including
    Z), so the resulting Δ (method − manual) directly quantifies how much of
    the apparent variability is in fact rational decision-making on hidden
    information.

    Mechanism:
      - ``Z ~ Bernoulli(z_prevalence)`` is generated independently of every
        other covariate (so blind methods truly cannot infer it).
      - Patients in the SCORE2 *borderline* band (risk within
        ``±borderline_window_pp`` of their age-stratum threshold) get their
        eligibility set to ``Z`` (Z=1 → eligible, Z=0 → non-eligible).
      - Patients clearly above or below the threshold keep the standard
        SCORE2 eligibility (Z plays no role for them).
      - Outcomes are then drawn under the same five-group physician
        behaviour model as :func:`generate_score2_five_groups_heter_patients`,
        so the analysis pipeline output is directly comparable to the
        headline experiment.

    The generator returns a DataFrame with the standard SCORE2 columns plus:

      - ``latent_covariate_z`` (int 0/1): the hidden covariate. **Must be
        excluded from the analysis ``fixed_effects`` list**; the wiring in
        ``main.py`` enforces this.
      - ``is_borderline`` (bool): identifies patients in the borderline band.
      - ``score2_indication_baseline`` (bool): standard SCORE2 eligibility,
        ignoring Z. Saved for diagnostic purposes.
      - ``generation_rule_mask`` (bool): the **modified** eligibility rule
        actually used to draw outcomes. Returned by
        :func:`get_generation_rule_mask` for this strategy so the Manual
        Pairing ground truth uses the correct mask.
      - ``outcome_score2_latent_covariate`` (int 0/1): prescription outcome.
      - ``recommendation`` (int 0/1): mirror of the outcome column.

    Args:
        n_patients: cohort size. Adjusted upward if below the per-physician
            minimum support to keep the analysis pipeline well-defined.
        expected_n_pros: number of physicians (must remain 20 to reuse the
            5-group structure).
        random_state: master seed; three independent RNG streams are derived
            (cohort generation, Z draws, outcome Bernoulli draws).
        z_prevalence: probability of ``Z = 1``. Default 0.15 per the reviewer.
        borderline_window_pp: half-width of the SCORE2 borderline band in
            percentage points. Default 2 pp.
        logger: optional logger.

    Returns:
        Synthetic cohort DataFrame as described above.
    """
    _validate_counts(n_patients, expected_n_pros)
    if expected_n_pros != 20:
        raise ValueError(
            "score2_latent_covariate requires expected_n_pros=20 to reuse the "
            "five-group physician structure (got %d)." % expected_n_pros
        )
    if not 0.0 <= z_prevalence <= 1.0:
        raise ValueError(f"z_prevalence must be in [0, 1]; got {z_prevalence!r}.")
    if borderline_window_pp <= 0.0:
        raise ValueError(
            f"borderline_window_pp must be strictly positive; got {borderline_window_pp!r}."
        )

    n_patients = _ensure_min_support(n_patients, expected_n_pros, logger)
    df, professionals = _build_universal_base_cohort(n_patients, expected_n_pros, random_state)

    # Step 1 — compute SCORE2 risk and the unmodified eligibility.
    df["score2_ten_year_cvd_risk_percent"] = np.round(
        score2_ten_year_risk_percent_vectorized(df, risk_region=SCORE2_DEFAULT_RISK_REGION), 2
    )
    age_arr = df["age"].to_numpy(dtype=float)
    risk_arr = df["score2_ten_year_cvd_risk_percent"].to_numpy(dtype=float)
    score2_indication = _score2_at_least_moderate_risk_mask(age_arr, risk_arr)
    df["score2_indication_baseline"] = score2_indication.astype(bool)

    # Step 2 — identify the borderline stratum (risk within ±window_pp of
    # the age-specific SCORE2 threshold).
    borderline = _score2_borderline_mask(age_arr, risk_arr, window_pp=borderline_window_pp)
    df["is_borderline"] = borderline.astype(bool)

    # Step 3 — draw the latent covariate Z and apply it to borderline patients.
    rng_z = np.random.default_rng(random_state + 90)
    z = rng_z.binomial(1, p=z_prevalence, size=len(df)).astype(np.int32)
    df[LATENT_COVARIATE_HIDDEN_COL] = z

    eligible = score2_indication.copy()
    eligible[borderline] = z[borderline].astype(bool)
    df["generation_rule_mask"] = eligible

    # Step 4 — assign 5-group physician parameters (same structure as the
    # headline experiment, so the resulting deltas are directly comparable).
    p_high_map, p_low_map = _build_group_probability_maps(professionals)

    rng_outcome = np.random.default_rng(random_state + 100)
    p_vec = np.where(
        eligible,
        df["professional_id"].map(p_high_map).to_numpy(dtype=float),
        df["professional_id"].map(p_low_map).to_numpy(dtype=float),
    )
    outcome_col = "outcome_score2_latent_covariate"
    df[outcome_col] = rng_outcome.binomial(1, p_vec).astype(np.int32)
    df["recommendation"] = df[outcome_col]

    if logger:
        n_borderline = int(borderline.sum())
        n_z_in_borderline = int(z[borderline].sum())
        logger.info(
            "Synthetic dataset '%s' generated: %d patients, %d physicians, "
            "Z ~ Bernoulli(%.2f), borderline ±%.1f pp around SCORE2 thresholds. "
            "Borderline stratum: %d patients (%.1f%%); Z=1 within borderline: "
            "%d (%.1f%%).",
            LATENT_COVARIATE_STRATEGY,
            len(df),
            expected_n_pros,
            z_prevalence,
            borderline_window_pp,
            n_borderline,
            100.0 * n_borderline / len(df),
            n_z_in_borderline,
            100.0 * n_z_in_borderline / max(1, n_borderline),
        )
    return df
