# -*- coding: utf-8 -*-
"""Orchestrator of the benchmark: experimental plan and long output table.

Usage (from the repository root):

    python -m src.benchmark.runner --config configs/benchmark.yaml
    python -m src.benchmark.runner --config configs/benchmark.yaml --smoke   # small, fast check

Each experiment is written to its own shard
(``<output_dir>/shards/<experiment_id>.csv.gz``) as soon as it finishes, so
an interrupted run resumes where it stopped: experiments whose shard exists
are skipped. ``--merge`` concatenates the shards into
``<output_dir>/benchmark_long.csv.gz``.

Provenance. Each launch has a UTC timestamp identifier (``launch_id``) and a
code fingerprint (SHA-256 of every ``src/**/*.py`` file and of the resolved
configuration). Both are written in every row of the output table, in
``fitted_parameters/<experiment_id>.json`` with the start and end times of
the experiment, and in ``manifests/<launch_id>.json`` (``manifest.json`` is a
copy of the latest one). A shard is reused only if its fingerprint equals the
current one: if the code or the configuration changed since it was written,
the launch stops and lists the stale experiments (``--allow-stale`` keeps
them, ``--rerun-stale`` recomputes them). A lock file prevents two launches
from writing to the same run directory at the same time.

One row of the output table is one (experiment, method, design, physician).
The columns are documented in ``OUTPUT_COLUMNS``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.benchmark.designs import (
    DEFAULT_K,
    DESIGNS,
    FULL_FIT_DESIGNS,
    DesignResult,
    equivalent_design,
    run_design,
    stratified_folds,
    supports,
)
from src.benchmark.estimators import Cohort, Estimator, PhysicianEstimate, build_estimators
from src.synthetic_generator import (
    CONTINUOUS_HETEROGENEITY_STRATEGY,
    SCORE2_ONLY_STRATEGY,
    THRESHOLD_STRATEGY_TARGET_JOINT,
    apply_recorded_rule,
    build_base_cohort,
    build_progressive_rule_plan,
    draw_prescriptions,
    five_group_physician_parameters,
    generate_progressive_rule_experiment,
    generate_score2_continuous_heterogeneity,
    generate_score2_five_groups_heter_patients,
    generate_score2_latent_influence,
    get_generation_rule_mask,
)

LOGGER = logging.getLogger("benchmark")

FEATURES = [
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
OUTCOME_COL, PHYSICIAN_COL = "recommendation", "professional_id"
DEVELOPMENT_SEED_OFFSET = 500_000
SCORE2_FAMILIES = frozenset({"score2", "sensitivity_independent", "sensitivity_diabetes_hba1c",
                             "grid_cohort_panel"})
CONTINUOUS_FAMILIES = frozenset({"continuous", "grid_panel"})
FOLD_SEED_OFFSET = 7_919

OUTPUT_COLUMNS = {
    # experiment
    "run_id": "Identifier of the run (configuration name).",
    "experiment_id": "Unique experiment identifier.",
    "family": "score2 | progressive | sensitivity_independent | sensitivity_diabetes_hba1c | latent | continuous.",
    "replicate": "Replicate index (Monte Carlo replications of a family).",
    "seed": "Master seed of the experiment.",
    "latent_gamma": "Scenario E1: influence of the unobserved determinant U (NaN elsewhere).",
    "pass_index": "Progressive rules: pass (1 or 2); 0 otherwise.",
    "window_size": "Progressive rules: number of covariates in the rule; 0 for SCORE2.",
    "window_start": "Progressive rules: position of the window in the covariate list.",
    "rule_variables": "Covariates of the eligibility rule (';'-separated).",
    "n_binary_rule_variables": "Number of binary covariates in the rule.",
    "threshold_strategy": "Threshold calibration strategy of progressive rules.",
    "target_eligibility": "Target joint eligibility p* (progressive rules).",
    "realised_eligibility": "Realised eligible fraction in the cohort.",
    "max_attainable_eligibility": "Largest eligibility reachable by the window (target_joint).",
    "calibration_attainable": "Whether the target eligibility was reached within 0.01.",
    "covariate_correlation": "Gaussian-copula correlation 'a~b:rho', empty if none.",
    "hba1c_law": "HbA1c law: normal | lognormal | lognormal_diabetes_mixture.",
    "covariate_model": "Covariate model: literature (revision) | submission (independent marginals).",
    "n_patients_cohort": "Number of patients in the cohort.",
    "n_physicians": "Number of physicians in the cohort.",
    # physician truth
    "physician": "Physician identifier.",
    "physician_group": "Five-group design: group 1 to 5; empty for continuous heterogeneity.",
    "p_high": "Prescribing probability for eligible patients.",
    "p_low": "Prescribing probability for ineligible patients.",
    "n_patients": "Patients of the physician.",
    "n_eligible": "Truly eligible patients of the physician.",
    "D_star": "Theoretical within-eligible discordance 2 p_high (1 - p_high).",
    "D_star_intrinsic": "Intrinsic inconsistency (scenario E1: patients sharing the unobserved U); equals D_star elsewhere.",
    "D_tilde": "Empirical discordance over all pairs of eligible patients (manual pairing).",
    "D_star_ineligible": "Theoretical discordance among ineligible patients 2 p_low (1 - p_low).",
    "D_tilde_ineligible": "Empirical discordance over all pairs of ineligible patients.",
    # method and design
    "method": "Estimator name.",
    "supervised": "Whether the estimator uses the outcome when learning.",
    "design": "Evaluation design.",
    "computed_as": "Design actually computed (identical by construction for unsupervised methods).",
    "caliper_percentile": "Caliper percentile used for pairing (100 = no caliper; NaN for the GLMM).",
    "score_type": "discordance | pearson_overdispersion.",
    "score": "Estimated discordance rate, or GLMM mean squared Pearson residual.",
    "n_candidates": "Patients in the physician's candidate pool for this design.",
    "n_pairs": "Retained pairs.",
    "n_patients_paired": "Patients appearing in at least one retained pair.",
    "retention": "n_patients_paired / n_candidates.",
    "caliper_threshold": "Caliper distance (mean over cells when pairing is within folds).",
    "pair_distance_median": "Median distance of retained pairs.",
    "pair_distance_q25": "25th percentile of retained-pair distances.",
    "pair_distance_q75": "75th percentile of retained-pair distances.",
    "pair_distance_p90": "90th percentile of retained-pair distances.",
    "frac_pairs_both_eligible": "Share of pairs with both patients truly eligible.",
    "frac_pairs_mixed": "Share of pairs with exactly one truly eligible patient.",
    "frac_pairs_both_ineligible": "Share of pairs with no truly eligible patient.",
    "disc_pairs_both_eligible": "Discordance among pairs with both patients eligible.",
    "disc_pairs_mixed": "Discordance among mixed pairs.",
    "disc_pairs_both_ineligible": "Discordance among pairs with both patients ineligible.",
    "smd_within_max": "Max over covariates of the mean absolute within-pair difference, in cohort SD units.",
    "smd_within_mean": "Mean over covariates of the same quantity.",
    "fit_seconds": "Wall-clock time of fitting for this design (0 when the full-data fit is reused).",
    "estimate_seconds": "Wall-clock time of distance computation, pairing and scoring for this design.",
    "n_fits": "Number of fits performed for this design.",
    "fit_reused": "Whether the full-data fit of in_sample was reused.",
    # provenance
    "launch_id": "UTC timestamp of the launch that computed the experiment (YYYYmmddTHHMMSSZ).",
    "code_fingerprint": "SHA-256 (12 hex) of src/**/*.py and of the resolved configuration.",
}


# -----------------------------------------------------------------------------
# Experiment specification and generation
# -----------------------------------------------------------------------------


@dataclass
class ExperimentSpec:
    experiment_id: str
    family: str
    seed: int
    n_patients: int
    n_physicians: int
    replicate: int = 0
    rule: Optional[dict[str, Any]] = None
    plan_index: Optional[int] = None
    covariate_correlation: Optional[tuple[str, str, float]] = None
    hba1c_law: str = "normal"
    covariate_model: str = "literature"
    designs: list[str] = field(default_factory=list)
    methods: Optional[list[str]] = None
    caliper_sensitivity: list[Optional[float]] = field(default_factory=list)
    caliper_sensitivity_designs: list[str] = field(default_factory=list)
    latent_gamma: Optional[float] = None


@dataclass
class GeneratedExperiment:
    spec: ExperimentSpec
    df: pd.DataFrame
    eligible: np.ndarray
    physician_parameters: dict[str, dict[str, float]]
    metadata: dict[str, Any]


def generate_experiment(spec: ExperimentSpec, threshold_strategy: str) -> GeneratedExperiment:
    """Simulate the benchmark cohort of one experiment."""
    meta: dict[str, Any] = {}
    if spec.family in SCORE2_FAMILIES:
        df = generate_score2_five_groups_heter_patients(
            n_patients=spec.n_patients, expected_n_pros=spec.n_physicians, random_state=spec.seed,
            pairwise_correlation=spec.covariate_correlation, hba1c_law=spec.hba1c_law,
            covariate_model=spec.covariate_model,
        )
        eligible = get_generation_rule_mask(df, SCORE2_ONLY_STRATEGY)
        params = five_group_physician_parameters(spec.n_physicians)
        meta = {"rule_type": "score2", "variables": []}
    elif spec.family == "progressive":
        df, meta = generate_progressive_rule_experiment(
            rule_definition=spec.rule, experiment_index=int(spec.plan_index) + 1, fixed_effects=FEATURES,
            n_patients=spec.n_patients, expected_n_pros=spec.n_physicians, random_state=spec.seed,
            threshold_strategy=threshold_strategy, covariate_model=spec.covariate_model,
            pairwise_correlation=spec.covariate_correlation, hba1c_law=spec.hba1c_law,
        )
        eligible = df["generation_rule_mask"].to_numpy(dtype=bool)
        params = five_group_physician_parameters(spec.n_physicians)
    elif spec.family == "latent":
        df, params = generate_score2_latent_influence(
            n_patients=spec.n_patients, expected_n_pros=spec.n_physicians, random_state=spec.seed,
            gamma=float(spec.latent_gamma), covariate_model=spec.covariate_model, hba1c_law=spec.hba1c_law,
        )
        eligible = df["generation_rule_mask"].to_numpy(dtype=bool)
        meta = {"rule_type": "score2", "variables": []}
    elif spec.family in CONTINUOUS_FAMILIES:
        df, raw_params, _ = generate_score2_continuous_heterogeneity(
            n_patients=spec.n_patients, expected_n_pros=spec.n_physicians, random_state=spec.seed,
            pairwise_correlation=spec.covariate_correlation, hba1c_law=spec.hba1c_law,
            covariate_model=spec.covariate_model,
        )
        eligible = get_generation_rule_mask(df, CONTINUOUS_HETEROGENEITY_STRATEGY)
        params = {k: {"p_high": v["p_high"], "p_low": v["p_low"], "group": None} for k, v in raw_params.items()}
        meta = {"rule_type": "score2", "variables": []}
    else:
        raise ValueError(f"Unknown family {spec.family!r}.")
    return GeneratedExperiment(spec=spec, df=df, eligible=np.asarray(eligible, dtype=bool),
                               physician_parameters=params, metadata=meta)


def generate_development_cohort(exp: GeneratedExperiment) -> Cohort:
    """Independent cohort from the same generating process and the same rule.

    Covariates, physician allocation and prescriptions are redrawn with an
    independent seed; the eligibility rule (and, for progressive rules, its
    recorded thresholds) and the physician prescribing probabilities are kept.
    """
    spec = exp.spec
    dev_seed = spec.seed + DEVELOPMENT_SEED_OFFSET + 1_009 * int(spec.plan_index or 0)
    df = build_base_cohort(spec.n_patients, spec.n_physicians, dev_seed,
                           pairwise_correlation=spec.covariate_correlation, hba1c_law=spec.hba1c_law,
                           covariate_model=spec.covariate_model)
    mask = apply_recorded_rule(df, exp.metadata)
    df[OUTCOME_COL] = draw_prescriptions(df, mask, exp.physician_parameters, dev_seed + 1)
    return Cohort.from_frame(df, FEATURES, OUTCOME_COL, PHYSICIAN_COL)


# -----------------------------------------------------------------------------
# Plan
# -----------------------------------------------------------------------------


def build_plan(cfg: dict[str, Any]) -> list[ExperimentSpec]:
    """Expand the configuration into the list of experiments."""
    fam = cfg["families"]
    base_seed = int(cfg.get("seed", 42))
    default_model = str(cfg.get("covariate_model", "literature"))
    default_hba1c = str(cfg.get("hba1c_law", "lognormal"))
    specs: list[ExperimentSpec] = []

    def _designs(block: dict[str, Any]) -> list[str]:
        designs = list(block.get("designs", ["in_sample"]))
        unknown = set(designs) - set(DESIGNS)
        if unknown:
            raise ValueError(f"Unknown designs in configuration: {sorted(unknown)}")
        return designs

    def _caliper_sensitivity(block: dict[str, Any]) -> dict[str, Any]:
        """Alternative calipers (None = no caliper) evaluated on full-data designs."""
        # 100 = no caliper (the 100th percentile threshold is +inf, see matching_core).
        alts = [100.0 if v is None else float(v) for v in block.get("caliper_sensitivity", [])]
        sens_designs = list(block.get("caliper_sensitivity_designs", ["in_sample"]))
        bad = set(sens_designs) - set(FULL_FIT_DESIGNS)
        if bad:
            raise ValueError(f"Caliper sensitivity is limited to full-data designs; got {sorted(bad)}.")
        return {"caliper_sensitivity": alts, "caliper_sensitivity_designs": sens_designs}

    if fam.get("score2", {}).get("enabled", False):
        b = fam["score2"]
        specs.append(ExperimentSpec("score2", "score2", int(b.get("seed", base_seed + 100)),
                                    int(b["n_patients"]), int(b["n_physicians"]),
                                    hba1c_law=default_hba1c, covariate_model=default_model,
                                    designs=_designs(b), methods=b.get("methods"),
                                    **_caliper_sensitivity(b)))

    if fam.get("progressive", {}).get("enabled", False):
        b = fam["progressive"]
        plan = build_progressive_rule_plan(FEATURES, n_passes=int(b.get("n_passes", 2)))
        cross_fit_every = int(b.get("cross_fitting_every", 0))
        rules = [(i, r) for i, r in enumerate(plan) if r["rule_type"] != "score2"]
        for k, (i, rule) in enumerate(rules):
            designs = _designs(b)
            in_subset = bool(cross_fit_every and k % cross_fit_every == 0)
            if in_subset:
                designs = designs + [d for d in b.get("cross_fitting_designs", []) if d not in designs]
            exp_id = f"prog_p{rule['pass_index']}_w{rule['window_size']}_s{rule['window_start']}"
            specs.append(ExperimentSpec(exp_id, "progressive", int(b.get("seed", base_seed)),
                                        int(b["n_patients"]), int(b["n_physicians"]), rule=rule,
                                        plan_index=i, hba1c_law=default_hba1c, covariate_model=default_model,
                                        designs=designs, methods=b.get("methods"),
                                        **(_caliper_sensitivity(b) if in_subset else {})))

    for name in ("sensitivity_independent", "sensitivity_diabetes_hba1c"):
        b = fam.get(name, {})
        if not b.get("enabled", False):
            continue
        corr = None
        if b.get("correlation"):
            c = b["correlation"]
            corr = (str(c["var_a"]), str(c["var_b"]), float(c["rho"]))
        specs.append(ExperimentSpec(name, name, int(b.get("seed", base_seed + 200)), int(b["n_patients"]),
                                    int(b["n_physicians"]), covariate_correlation=corr,
                                    hba1c_law=str(b.get("hba1c_law", default_hba1c)),
                                    covariate_model=str(b.get("covariate_model", default_model)),
                                    designs=_designs(b), methods=b.get("methods")))

    if fam.get("latent", {}).get("enabled", False):
        b = fam["latent"]
        for r in range(int(b.get("n_replicates", 1))):
            for g in b["gammas"]:
                specs.append(ExperimentSpec(f"latent_g{float(g):g}_r{r:02d}", "latent",
                                            int(b.get("seed", base_seed + 400)) + 1_000 * r,
                                            int(b["n_patients"]), int(b["n_physicians"]), replicate=r,
                                            hba1c_law=default_hba1c, covariate_model=default_model,
                                            latent_gamma=float(g), designs=_designs(b), methods=b.get("methods")))

    if fam.get("continuous", {}).get("enabled", False):
        b = fam["continuous"]
        n_cf = int(b.get("cross_fitting_replicates", 0))
        for r in range(int(b.get("n_replicates", 1))):
            designs = _designs(b)
            if r < n_cf:
                designs = designs + [d for d in b.get("cross_fitting_designs", []) if d not in designs]
            specs.append(ExperimentSpec(f"continuous_r{r:03d}", "continuous",
                                        int(b.get("seed", base_seed + 300)) + 1_000 * r,
                                        int(b["n_patients"]), int(b["n_physicians"]), replicate=r,
                                        hba1c_law=default_hba1c, covariate_model=default_model,
                                        designs=designs, methods=b.get("methods")))
    # Monte Carlo grids (R2 comments 4 and 10): independent cohorts per cell.
    # grid_panel: continuous heterogeneity, patients per physician x physicians.
    # grid_cohort_panel: five-group SCORE2, cohort size x physicians (replaces
    # the B = 10 resamples of one cohort per cell of the first submission).
    for name, size_key, tag in (("grid_panel", "patients_per_physician", "m"),
                                ("grid_cohort_panel", "n_patients", "n")):
        b = fam.get(name, {})
        if not b.get("enabled", False):
            continue
        seed0 = int(b.get("seed", base_seed))
        cell = 0
        for size in b[size_key]:
            for J in b["n_physicians"]:
                n = int(size) * int(J) if size_key == "patients_per_physician" else int(size)
                for r in range(int(b["n_replicates"])):
                    specs.append(ExperimentSpec(
                        f"{name}_{tag}{int(size)}_J{int(J)}_r{r:03d}", name, seed0 + 10_000 * cell + r,
                        n, int(J), replicate=r, hba1c_law=default_hba1c, covariate_model=default_model,
                        designs=_designs(b), methods=b.get("methods")))
                cell += 1

    ids = [s.experiment_id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate experiment identifiers in the plan.")
    return specs


# -----------------------------------------------------------------------------
# Truth and pair diagnostics
# -----------------------------------------------------------------------------


def _all_pairs_discordance(y: np.ndarray) -> float:
    """Discordance over all unordered pairs: 2 k (n - k) / (n (n - 1))."""
    n = len(y)
    if n < 2:
        return float("nan")
    k = float(np.sum(y))
    return float(2.0 * k * (n - k) / (n * (n - 1)))


def physician_truth(exp: GeneratedExperiment) -> pd.DataFrame:
    df, m = exp.df, exp.eligible
    phys = df[PHYSICIAN_COL].astype(str).to_numpy()
    y = df[OUTCOME_COL].to_numpy(dtype=int)
    rows = []
    for p in np.unique(phys):
        sel = phys == p
        par = exp.physician_parameters[p]
        rows.append({
            "physician": p,
            "physician_group": par.get("group"),
            "p_high": par["p_high"],
            "p_low": par["p_low"],
            "n_patients": int(sel.sum()),
            "n_eligible": int((sel & m).sum()),
            "D_star": 2 * par["p_high"] * (1 - par["p_high"]),
            "D_star_intrinsic": par.get("D_intrinsic", 2 * par["p_high"] * (1 - par["p_high"])),
            "D_tilde": _all_pairs_discordance(y[sel & m]),
            "D_star_ineligible": 2 * par["p_low"] * (1 - par["p_low"]),
            "D_tilde_ineligible": _all_pairs_discordance(y[sel & ~m]),
        })
    return pd.DataFrame(rows)


def pair_diagnostics(e: PhysicianEstimate, cohort: Cohort, eligible: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
    a, b = e.pair_index_a, e.pair_index_b
    out: dict[str, Any] = {}
    if len(a) == 0:
        return {k: float("nan") for k in (
            "pair_distance_median", "pair_distance_q25", "pair_distance_q75", "pair_distance_p90",
            "frac_pairs_both_eligible", "frac_pairs_mixed", "frac_pairs_both_ineligible",
            "disc_pairs_both_eligible", "disc_pairs_mixed", "disc_pairs_both_ineligible",
            "smd_within_max", "smd_within_mean")}
    q = np.quantile(e.pair_distance, [0.5, 0.25, 0.75, 0.9])
    out.update(pair_distance_median=q[0], pair_distance_q25=q[1], pair_distance_q75=q[2], pair_distance_p90=q[3])
    n_elig = eligible[a].astype(int) + eligible[b].astype(int)
    disc = cohort.y[a] != cohort.y[b]
    for label, value in (("both_eligible", 2), ("mixed", 1), ("both_ineligible", 0)):
        sel = n_elig == value
        out[f"frac_pairs_{label}"] = float(sel.mean())
        out[f"disc_pairs_{label}"] = float(disc[sel].mean()) if sel.any() else float("nan")
    smd = np.mean(np.abs(cohort.X[a] - cohort.X[b]), axis=0) / np.where(sd > 0, sd, 1.0)
    out.update(smd_within_max=float(np.max(smd)), smd_within_mean=float(np.mean(smd)))
    return out


# -----------------------------------------------------------------------------
# Running one experiment
# -----------------------------------------------------------------------------


def _experiment_columns(exp: GeneratedExperiment, run_id: str, threshold_strategy: str) -> dict[str, Any]:
    s, md = exp.spec, exp.metadata
    corr = "" if s.covariate_correlation is None else "{}~{}:{}".format(*s.covariate_correlation)
    variables = md.get("variables") or []
    return {
        "run_id": run_id, "experiment_id": s.experiment_id, "family": s.family, "replicate": s.replicate,
        "seed": s.seed, "latent_gamma": float("nan") if s.latent_gamma is None else s.latent_gamma,
        "pass_index": int((s.rule or {}).get("pass_index", 0)),
        "window_size": int((s.rule or {}).get("window_size", 0)),
        "window_start": int((s.rule or {}).get("window_start", 0)),
        "rule_variables": ";".join(variables),
        "n_binary_rule_variables": len(md.get("binary_variables") or []),
        "threshold_strategy": threshold_strategy if s.family == "progressive" else "",
        "target_eligibility": md.get("target_eligibility"),
        "realised_eligibility": float(exp.eligible.mean()),
        "max_attainable_eligibility": md.get("max_attainable_eligibility"),
        "calibration_attainable": md.get("calibration_attainable"),
        "covariate_correlation": corr, "hba1c_law": s.hba1c_law, "covariate_model": s.covariate_model,
        "n_patients_cohort": int(len(exp.df)), "n_physicians": int(exp.df[PHYSICIAN_COL].nunique()),
    }


def run_experiment(exp: GeneratedExperiment, estimators: list[Estimator], cfg: dict[str, Any],
                   run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every (method, design) of one experiment; return its rows and fitted parameters."""
    spec = exp.spec
    cohort = Cohort.from_frame(exp.df, FEATURES, OUTCOME_COL, PHYSICIAN_COL)
    cohort.eligible = exp.eligible  # read only by explicitly oracle estimators (glmm_oracle)
    folds = stratified_folds(cohort.physician, int(cfg.get("k_folds", DEFAULT_K)),
                             seed=spec.seed + FOLD_SEED_OFFSET + 13 * int(spec.plan_index or 0))
    sd = np.std(cohort.X, axis=0)
    truth = physician_truth(exp).set_index("physician")
    exp_cols = _experiment_columns(exp, run_id, cfg.get("threshold_strategy", THRESHOLD_STRATEGY_TARGET_JOINT))
    caliper = cfg.get("caliper_percentile", 25.0)
    dev_cohort: Optional[Cohort] = None
    rows: list[dict[str, Any]] = []
    fitted_params: dict[str, Any] = {}

    for est in estimators:
        if spec.methods is not None and est.name not in spec.methods:
            continue
        rules = (cfg.get("method_rules") or {}).get(est.name, {})
        if spec.family in (rules.get("exclude_families") or []):
            continue
        only_in = rules.get("designs_only_in_families") or {}
        designs = [d for d in spec.designs if supports(est, d)
                   and d not in (rules.get("exclude_designs") or [])
                   and (d not in only_in or spec.family in only_in[d])]
        # Full-data designs first, so that their single fit can be reused.
        designs.sort(key=lambda d: (d not in FULL_FIT_DESIGNS, d != "in_sample"))
        computed: dict[str, DesignResult] = {}
        prefitted: Optional[Estimator] = None

        def _emit(res: DesignResult, design: str, target: str, caliper_used: Optional[float]) -> None:
            for e in res.estimates:
                row = dict(exp_cols)
                row.update(truth.loc[e.physician].to_dict())
                row.update({
                    "physician": e.physician, "method": est.name, "supervised": est.supervised,
                    "design": design, "computed_as": target,
                    "caliper_percentile": float("nan") if caliper_used is None else float(caliper_used),
                    "score_type": "pearson_overdispersion" if est.kind == "model" else "discordance",
                    "score": e.score, "n_candidates": e.n_candidates, "n_pairs": e.n_pairs,
                    "n_patients_paired": e.n_patients_paired,
                    "retention": e.n_patients_paired / e.n_candidates if e.n_candidates else float("nan"),
                    "caliper_threshold": e.caliper_threshold,
                    "fit_seconds": res.fit_seconds, "estimate_seconds": res.estimate_seconds,
                    "n_fits": res.n_fits,
                    "fit_reused": bool(prefitted is not None and target in FULL_FIT_DESIGNS and target != "in_sample"),
                })
                row.update(pair_diagnostics(e, cohort, exp.eligible, sd))
                rows.append(row)

        for design in designs:
            target = equivalent_design(est, design)
            if target not in computed:
                if target == "external" and dev_cohort is None:
                    dev_cohort = generate_development_cohort(exp)
                t0 = time.perf_counter()
                res = run_design(est, cohort, target, folds=folds, eligible=exp.eligible,
                                 development_cohort=dev_cohort, caliper_percentile=caliper, prefitted=prefitted)
                LOGGER.info("  %-28s %-18s %6.1fs", est.name, target, time.perf_counter() - t0)
                computed[target] = res
                if target == "in_sample" and res.fitted is not None:
                    prefitted = res.fitted
                    fitted_params[est.name] = res.fitted.fitted_parameters()
                    diag_cfg = cfg.get("glmm_diagnostics") or {}
                    if (est.name == "glmm" and diag_cfg.get("enabled", False)
                            and spec.family in (diag_cfg.get("families") or [])):
                        from src.benchmark.glmm_diagnostics import glmm_diagnostics
                        t1 = time.perf_counter()
                        fitted_params["glmm_diagnostics"] = glmm_diagnostics(
                            res.fitted, cohort, truth["D_star"].to_dict(), seed=spec.seed,
                            n_boot=int(diag_cfg.get("n_boot", 2000)))
                        LOGGER.info("  %-28s %-18s %6.1fs", "glmm_diagnostics", target, time.perf_counter() - t1)
            _emit(computed[target], design, target, None if est.kind == "model" else caliper)

        # Caliper sensitivity: same fit, alternative calipers (matching methods only).
        if est.kind == "matching" and prefitted is not None:
            for alt in spec.caliper_sensitivity:
                if float(alt) == float(caliper):
                    continue
                for design in spec.caliper_sensitivity_designs:
                    if design not in designs:
                        continue
                    target = equivalent_design(est, design)
                    t0 = time.perf_counter()
                    res = run_design(est, cohort, target, eligible=exp.eligible, folds=folds,
                                     caliper_percentile=alt, prefitted=prefitted)
                    LOGGER.info("  %-28s %-18s %6.1fs (caliper %s)", est.name, target,
                                time.perf_counter() - t0, "none" if alt >= 100 else f"{alt:g}")
                    _emit(res, design, target, alt)
    table = pd.DataFrame(rows)
    prov = cfg.get("_provenance") or {}
    table["launch_id"] = prov.get("launch_id")
    table["code_fingerprint"] = prov.get("code_fingerprint")
    missing = [c for c in OUTPUT_COLUMNS if c not in table.columns]
    if missing and len(table):
        raise RuntimeError(f"Output columns missing: {missing}")
    return table.reindex(columns=list(OUTPUT_COLUMNS)), fitted_params


# -----------------------------------------------------------------------------
# Run management
# -----------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


REPO_ROOT = Path(__file__).resolve().parents[2]
# Post-processing modules that do not change any shard.
FINGERPRINT_EXCLUDED = frozenset({"metrics.py"})


def code_fingerprint(cfg: dict[str, Any]) -> str:
    """SHA-256 (12 hex) of the source files and of the resolved configuration."""
    import hashlib

    h = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if "__pycache__" in path.parts or path.name in FINGERPRINT_EXCLUDED:
            continue
        h.update(path.relative_to(REPO_ROOT).as_posix().encode())
        h.update(path.read_bytes())
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    h.update(json.dumps(clean, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def shard_fingerprint(out_dir: Path, experiment_id: str) -> Optional[str]:
    path = out_dir / "fitted_parameters" / f"{experiment_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("provenance", {}).get("code_fingerprint")
    except Exception:
        return None


class RunLock:
    """Exclusive lock on a run directory (stale locks of dead processes are taken over)."""

    def __init__(self, out_dir: Path, suffix: str = "") -> None:
        self.path = out_dir / f"run.lock{suffix}"

    def __enter__(self) -> "RunLock":
        if self.path.exists():
            try:
                pid = int(json.loads(self.path.read_text())["pid"])
                os.kill(pid, 0)
                alive = True
            except (ProcessLookupError, ValueError, KeyError, json.JSONDecodeError):
                alive = False
            except PermissionError:
                alive = True
            if alive:
                raise RuntimeError(f"{self.path} is held by a running process (pid {pid}); "
                                   "another launch is writing to this run directory.")
            LOGGER.warning("Stale lock %s removed.", self.path)
            self.path.unlink()
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as fh:
            json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, fh)
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def write_manifest(out_dir: Path, cfg: dict[str, Any], plan: list[ExperimentSpec],
                   extra: Optional[dict[str, Any]] = None) -> None:
    import numpy, pandas, scipy, sklearn, statsmodels  # noqa: E401
    prov = cfg.get("_provenance") or {}
    manifest = {
        "launch_id": prov.get("launch_id"),
        "code_fingerprint": prov.get("code_fingerprint"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "git_uncommitted_changes": _git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "libraries": {m.__name__: m.__version__ for m in (numpy, pandas, scipy, sklearn, statsmodels)},
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "n_experiments": len(plan),
        "experiments": [{"experiment_id": s.experiment_id, "family": s.family, "designs": s.designs} for s in plan],
        "columns": OUTPUT_COLUMNS,
        **(extra or {}),
    }
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, default=str)
    (out_dir / "manifests" / f"{prov.get('launch_id', 'unknown')}.json").write_text(text, encoding="utf-8")
    (out_dir / "manifest.json").write_text(text, encoding="utf-8")


def merge_shards(out_dir: Path) -> Path:
    shards = sorted((out_dir / "shards").glob("*.csv.gz"))
    if not shards:
        raise FileNotFoundError(f"No shard in {out_dir / 'shards'}.")
    table = pd.concat((pd.read_csv(p) for p in shards), ignore_index=True)
    if "code_fingerprint" in table.columns:
        fps = table.groupby("code_fingerprint", dropna=False)["experiment_id"].nunique()
        if len(fps) > 1:
            LOGGER.warning("Shards come from %d code fingerprints: %s", len(fps), fps.to_dict())
    path = out_dir / "benchmark_long.csv.gz"
    table.to_csv(path, index=False)
    LOGGER.info("Merged %d shards, %d rows -> %s", len(shards), len(table), path)
    return path


def apply_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Shrink a configuration for a quick end-to-end check."""
    cfg = json.loads(json.dumps(cfg))
    cfg["run_id"] = f"{cfg.get('run_id', 'run')}_smoke"
    cfg["genetic"] = dict(cfg.get("genetic", {}), enabled=False)
    for name, block in cfg["families"].items():
        if name == "grid_panel":
            block.update(patients_per_physician=[50, 100], n_physicians=[20], n_replicates=1)
            continue
        if name == "grid_cohort_panel":
            block.update(n_patients=[1_000, 2_000], n_physicians=[5, 20], n_replicates=1)
            continue
        block["n_patients"] = min(int(block["n_patients"]), 2_000 if name != "continuous" else 2_500)
        if name == "continuous":
            block["n_replicates"] = min(int(block.get("n_replicates", 1)), 1)
    if "progressive" in cfg["families"]:
        cfg["families"]["progressive"]["max_experiments"] = int(cfg["families"]["progressive"].get("max_experiments", 4))
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment identifiers.")
    parser.add_argument("--smoke", action="store_true", help="Small cohorts, no genetic matching, few experiments.")
    parser.add_argument("--merge", action="store_true", help="Merge existing shards and exit.")
    parser.add_argument("--list", action="store_true", help="Print the plan and exit.")
    stale = parser.add_mutually_exclusive_group()
    stale.add_argument("--allow-stale", action="store_true",
                       help="Keep shards computed with another code fingerprint (not recommended).")
    stale.add_argument("--rerun-stale", action="store_true",
                       help="Recompute shards computed with another code fingerprint.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Run the plan in this many parallel processes (experiments split between them).")
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--launch-id", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    tag = "" if args.worker_index is None else f"[w{args.worker_index}] "
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s {tag}%(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        cfg = apply_smoke(cfg)
    run_id = str(cfg.get("run_id", "run"))
    out_dir = Path(cfg.get("output_dir", "results")) / run_id
    if args.merge:
        merge_shards(out_dir)
        return 0

    plan = build_plan(cfg)
    max_prog = cfg["families"].get("progressive", {}).get("max_experiments")
    if max_prog is not None:
        kept, n_prog = [], 0
        for s in plan:
            if s.family == "progressive":
                n_prog += 1
                if n_prog > int(max_prog):
                    continue
            kept.append(s)
        plan = kept
    if args.only:
        plan = [s for s in plan if s.experiment_id in set(args.only)]
    if args.list:
        for s in plan:
            print(f"{s.experiment_id:28s} {s.family:28s} n={s.n_patients:<6d} J={s.n_physicians:<4d} {','.join(s.designs)}")
        print(f"{len(plan)} experiments")
        return 0

    launch_id = args.launch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = code_fingerprint(cfg)
    cfg["_provenance"] = {"launch_id": launch_id, "code_fingerprint": fingerprint}
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    (out_dir / "fitted_parameters").mkdir(parents=True, exist_ok=True)
    LOGGER.info("Launch %s, code fingerprint %s, output %s", launch_id, fingerprint, out_dir)

    stale_ids = [s.experiment_id for s in plan
                 if (out_dir / "shards" / f"{s.experiment_id}.csv.gz").exists()
                 and shard_fingerprint(out_dir, s.experiment_id) != fingerprint]
    if stale_ids and not (args.allow_stale or args.rerun_stale):
        LOGGER.error("%d existing shard(s) were computed with another code or configuration: %s. "
                     "Delete them, or relaunch with --rerun-stale (recompute) or --allow-stale (keep).",
                     len(stale_ids), ", ".join(stale_ids[:10]) + (" ..." if len(stale_ids) > 10 else ""))
        return 2
    if stale_ids:
        LOGGER.warning("%d stale shard(s) %s.", len(stale_ids), "will be recomputed" if args.rerun_stale else "kept")

    rerun = stale_ids if args.rerun_stale else []
    if args.worker_index is not None:
        with RunLock(out_dir, suffix=f".w{args.worker_index}"):
            return _run_plan(plan, cfg, out_dir, run_id, rerun, merge=False)
    with RunLock(out_dir):
        write_manifest(out_dir, cfg, plan, extra={"stale_shards": stale_ids,
                                                  "stale_policy": "rerun" if args.rerun_stale else
                                                  ("keep" if args.allow_stale else "none"),
                                                  "workers": args.workers})
        if args.workers <= 1:
            return _run_plan(plan, cfg, out_dir, run_id, rerun)
        return _run_parallel(args, plan, out_dir, launch_id)


def _run_parallel(args: argparse.Namespace, plan: list[ExperimentSpec], out_dir: Path, launch_id: str) -> int:
    """Split the plan between worker processes (largest cohorts first, round robin) and merge at the end.

    Each worker gets an equal share of the cores for its BLAS/OpenMP threads and
    for the random forest (``BENCHMARK_N_JOBS``). Runtimes measured in parallel
    mode are therefore not comparable with those of the main run.
    """
    n_workers = int(args.workers)
    order = sorted(range(len(plan)), key=lambda i: -plan[i].n_patients)
    parts: list[list[str]] = [[] for _ in range(n_workers)]
    for rank, i in enumerate(order):
        parts[rank % n_workers].append(plan[i].experiment_id)
    threads = max(1, (os.cpu_count() or n_workers) // n_workers)
    env = dict(os.environ, BENCHMARK_N_JOBS=str(threads), OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               VECLIB_MAXIMUM_THREADS=str(threads))
    (out_dir / "logs").mkdir(exist_ok=True)
    procs = []
    for k, ids in enumerate(parts):
        if not ids:
            continue
        cmd = [sys.executable, "-m", "src.benchmark.runner", "--config", str(args.config),
               "--worker-index", str(k), "--launch-id", launch_id, "--only", *ids]
        if args.smoke:
            cmd.append("--smoke")
        if args.rerun_stale:
            cmd.append("--rerun-stale")
        if args.allow_stale:
            cmd.append("--allow-stale")
        log = (out_dir / "logs" / f"{launch_id}_w{k}.log").open("w", encoding="utf-8")
        procs.append((k, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        LOGGER.info("Worker %d: %d experiments, %d threads, log %s", k, len(ids), threads, log.name)
    failed = []
    for k, proc, log in procs:
        if proc.wait() != 0:
            failed.append(k)
        log.close()
    if failed:
        LOGGER.error("Workers %s failed; see %s. Relaunch the same command to resume.", failed, out_dir / "logs")
        return 1
    merge_shards(out_dir)
    return 0


def _run_plan(plan: list[ExperimentSpec], cfg: dict[str, Any], out_dir: Path, run_id: str,
              rerun: list[str], merge: bool = True) -> int:
    g = cfg.get("genetic", {})
    estimators = build_estimators(genetic_variant=str(g.get("variant", "genmatch")),
                                  include_genetic=bool(g.get("enabled", True)),
                                  genetic_workers=int(g.get("workers", -1)),
                                  include_glmm_variants=bool((cfg.get("glmm_variants") or {}).get("enabled", False)))
    if "legacy" in (g.get("also_run_variants") or []) and g.get("enabled", True):
        from src.benchmark.estimators import GeneticMahalanobisEstimator
        estimators.insert(-1, GeneticMahalanobisEstimator(variant="legacy", workers=int(g.get("workers", -1))))

    threshold_strategy = str(cfg.get("threshold_strategy", THRESHOLD_STRATEGY_TARGET_JOINT))
    t_run = time.perf_counter()
    for i, spec in enumerate(plan, start=1):
        shard = out_dir / "shards" / f"{spec.experiment_id}.csv.gz"
        if shard.exists() and spec.experiment_id not in rerun:
            LOGGER.info("[%d/%d] %s already done, skipped.", i, len(plan), spec.experiment_id)
            continue
        LOGGER.info("[%d/%d] %s (%s)", i, len(plan), spec.experiment_id, spec.family)
        t0 = time.perf_counter()
        started = datetime.now(timezone.utc).isoformat()
        exp = generate_experiment(spec, threshold_strategy)
        table, fitted = run_experiment(exp, estimators, cfg, run_id)
        tmp = shard.with_suffix(".tmp")
        table.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(shard)
        with (out_dir / "fitted_parameters" / f"{spec.experiment_id}.json").open("w", encoding="utf-8") as fh:
            json.dump({"provenance": dict(cfg["_provenance"], experiment_id=spec.experiment_id,
                                          started_utc=started,
                                          finished_utc=datetime.now(timezone.utc).isoformat(),
                                          git_commit=_git_commit()),
                       "metadata": exp.metadata, "fitted": fitted}, fh, indent=1, default=str)
        LOGGER.info("[%d/%d] %s done in %.0fs (%d rows).", i, len(plan), spec.experiment_id,
                    time.perf_counter() - t0, len(table))
    if merge:
        merge_shards(out_dir)
    LOGGER.info("Run finished in %.1f min.", (time.perf_counter() - t_run) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
