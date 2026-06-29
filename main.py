#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.analysis_pipeline import Analysis
from src.synthetic_generator import (
    CONTINUOUS_HETEROGENEITY_STRATEGY,
    LATENT_COVARIATE_STRATEGY,
    LATENT_COVARIATE_HIDDEN_COL,
    SCORE2_ONLY_STRATEGY,
    generate_score2_continuous_heterogeneity,
    generate_score2_five_groups_heter_patients,
    generate_score2_latent_covariate,
)

DEFAULT_CONFIG = Path("configs/article_stats.yaml")


def setup_logger(level: str = "info") -> logging.Logger:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger("main")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _effective_seed(cfg: dict) -> int:
    project = cfg.get("project") or {}
    dataset = cfg.get("dataset") or {}
    base = int(project.get("random_state", 42))
    offset = int((dataset.get("score2_five_groups") or {}).get("random_state_offset", 0))
    return base + offset


def _pairwise_correlation(cfg: dict) -> tuple[str, str, float] | None:
    block = (cfg.get("dataset") or {}).get("covariate_correlation") or {}
    if not block.get("enabled"):
        return None
    return (str(block["var_a"]), str(block["var_b"]), float(block["rho"]))


def _hba1c_law(cfg: dict) -> str:
    block = (cfg.get("dataset") or {}).get("hba1c_distribution") or {}
    law = str(block.get("law", "normal")).strip().lower()
    return law if law in {"normal", "lognormal"} else "normal"


def generate_dataset(cfg: dict, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """Generate the synthetic cohort described in the config."""
    dataset = cfg.get("dataset") or {}
    strategy = str(dataset.get("generation_strategy", SCORE2_ONLY_STRATEGY))
    seed = _effective_seed(cfg)
    corr = _pairwise_correlation(cfg)
    hba1c_law = _hba1c_law(cfg)

    five_groups = dataset.get("score2_five_groups") or {}
    n_patients = int(five_groups.get("n_patients", 10_000))

    extra_analysis: dict = {"generation_strategy": strategy}

    if strategy == SCORE2_ONLY_STRATEGY:
        n_physicians = int(five_groups.get("n_physicians", 20))
        df = generate_score2_five_groups_heter_patients(
            n_patients=n_patients,
            expected_n_pros=n_physicians,
            random_state=seed,
            logger=logger,
            pairwise_correlation=corr,
            hba1c_law=hba1c_law,
        )
        return df, extra_analysis

    if strategy == CONTINUOUS_HETEROGENEITY_STRATEGY:
        cont = dataset.get("continuous_heterogeneity") or {}
        df, physician_params, _matching_covariates = generate_score2_continuous_heterogeneity(
            n_patients=n_patients,
            expected_n_pros=int(cont.get("n_physicians", 50)),
            random_state=seed,
            p_high_low=float(cont.get("p_high_low", 0.5)),
            p_high_high=float(cont.get("p_high_high", 1.0)),
            p_low_fixed=float(cont.get("p_low_fixed", 0.05)),
            logger=logger,
            pairwise_correlation=corr,
            hba1c_law=hba1c_law,
        )
        extra_analysis["physician_params"] = physician_params
        return df, extra_analysis

    if strategy == LATENT_COVARIATE_STRATEGY:
        n_physicians = int(five_groups.get("n_physicians", 20))
        latent = dataset.get("latent_covariate") or {}
        df = generate_score2_latent_covariate(
            n_patients=n_patients,
            expected_n_pros=n_physicians,
            random_state=seed,
            z_prevalence=float(latent.get("z_prevalence", 0.15)),
            borderline_window_pp=float(latent.get("borderline_window_pp", 2.0)),
            logger=logger,
        )
        return df, extra_analysis

    raise ValueError(
        f"Generation strategy not supported by main.py: {strategy!r}. "
        f"Expected one of: {SCORE2_ONLY_STRATEGY!r}, {CONTINUOUS_HETEROGENEITY_STRATEGY!r}, "
        f"or {LATENT_COVARIATE_STRATEGY!r}."
    )


def build_analysis_config(cfg: dict, extra: dict | None = None) -> dict:
    analysis = cfg.get("analysis") or {}
    dataset = cfg.get("dataset") or {}
    caliper = analysis.get("caliper") or {}

    fixed_effects = list(analysis.get("fixed_effects") or [])
    strategy = str(dataset.get("generation_strategy", SCORE2_ONLY_STRATEGY))
    if strategy == LATENT_COVARIATE_STRATEGY:
        fixed_effects = [c for c in fixed_effects if c != LATENT_COVARIATE_HIDDEN_COL]

    out: dict = {
        "outcome_col": analysis.get("outcome_col", "recommendation"),
        "physician_col": analysis.get("physician_col", "professional_id"),
        "fixed_effects": fixed_effects,
        "min_patients_per_physician": int(analysis.get("min_patients_per_physician", 30)),
        "generation_strategy": strategy,
        "rf_n_estimators": int(analysis.get("rf_n_estimators", 300)),
        "pairing_strategy": analysis.get("pairing_strategy", "optimal"),
        "lpa_enabled": bool(analysis.get("lpa_enabled", True)),
        "lpa_n_components_min": int(analysis.get("lpa_n_components_min", 2)),
        "lpa_n_components_max": int(analysis.get("lpa_n_components_max", 10)),
        "lpa_covariance_type": analysis.get("lpa_covariance_type", "full"),
        "lpa_random_state": int(analysis.get("lpa_random_state", 42)),
        "lpa_n_init": int(analysis.get("lpa_n_init", 10)),
        "lpa_robust_scaling": bool(analysis.get("lpa_robust_scaling", True)),
        "lpa_distance_metric": analysis.get("lpa_distance_metric", "hybrid"),
        "lpa_hybrid_alpha": float(analysis.get("lpa_hybrid_alpha", 0.5)),
        "lpa_pairing_strategy": analysis.get("lpa_pairing_strategy", "optimal"),
        "caliper_enabled": bool(caliper.get("enabled", False)),
        "caliper_percentile": float(caliper.get("percentile", 25.0)),
        "glmm_enabled": bool(analysis.get("glmm_enabled", True)),
        "enable_manual_pairing": bool(analysis.get("enable_manual_pairing", True)),
        "dataset_targets": analysis.get("dataset_targets") or {},
    }
    if extra:
        out.update(extra)
    return out


def resolve_results_dir(cfg: dict) -> Path:
    paths = cfg.get("paths") or {}
    project = cfg.get("project") or {}
    root = Path(paths.get("experiments_root", "experiments"))
    date_fmt = paths.get("experiments_date_format", "%y_%m_%d")
    prefix = paths.get("scratch_dirname_prefix", "exp")
    name = project.get("default_experiment_name", "run")
    return root / datetime.now().strftime(date_fmt) / f"{prefix}_{name}"


def save_config_copy(cfg: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "config_used.yaml"
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SCORE2 cohort and run the intra-physician variability analysis.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration file (default: {DEFAULT_CONFIG}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    logging_cfg = cfg.get("logging") or {}
    logger = setup_logger(logging_cfg.get("level", "info"))

    dataset_cfg = cfg.get("dataset") or {}
    source = str(dataset_cfg.get("source", "synthetic")).strip().lower()
    if source != "synthetic":
        logger.warning(
            "dataset.source=%r ignored: this entry point only supports synthetic generation.",
            source,
        )

    analysis_cfg_block = cfg.get("analysis") or {}
    if not analysis_cfg_block.get("enabled", True):
        logger.error("analysis.enabled=false: nothing to run.")
        return 1

    results_dir = resolve_results_dir(cfg)
    save_config_copy(cfg, results_dir)
    logger.info("Results directory: %s", results_dir)

    logger.info("Generating synthetic cohort...")
    df, extra = generate_dataset(cfg, logger)
    logger.info("Cohort generated: %d patients, %d physicians.", len(df), df["professional_id"].nunique())

    paths = cfg.get("paths") or {}
    if dataset_cfg.get("save_raw_csv"):
        data_dir = results_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = data_dir / Path(paths.get("dataset_csv", "dataset.csv")).name
        df.to_csv(dataset_path, index=False)
        logger.info("Dataset exported: %s", dataset_path)

    analysis_config = build_analysis_config(cfg, extra)
    pipeline = Analysis(
        df=df,
        results_dir=str(results_dir),
        LOGGER=logger,
        config=analysis_config,
    )
    pipeline.run_full_analysis()
    logger.info("Done. Artifacts written to %s", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
