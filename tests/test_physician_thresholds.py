"""Scenario E2: physician-specific SCORE2 thresholds, truth within each physician's own stratum."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmark import metrics as M
from src.benchmark.estimators import EuclideanEstimator, LearnedWeightsEstimator
from src.benchmark.runner import (
    ExperimentSpec,
    _all_pairs_discordance,
    build_plan,
    generate_development_cohort,
    generate_experiment,
    physician_truth,
    run_experiment,
)
from src.synthetic_generator import (
    SCORE2_ONLY_STRATEGY,
    apply_recorded_rule,
    generate_score2_five_groups_heter_patients,
    generate_score2_physician_thresholds,
    get_generation_rule_mask,
    physician_specific_score2_mask,
    physician_threshold_multipliers,
    score2_ten_year_risk_percent_vectorized,
)

N, J, SEED = 2000, 10, 31


@pytest.fixture(scope="module")
def cohorts():
    return {sd: generate_score2_physician_thresholds(N, J, SEED, threshold_log_sd=sd) for sd in (0.0, 0.2)}


def test_zero_spread_reproduces_the_reference_experiment(cohorts):
    df0, params, mult = cohorts[0.0]
    ref = generate_score2_five_groups_heter_patients(n_patients=N, expected_n_pros=J, random_state=SEED)
    assert set(mult.values()) == {1.0}
    pd.testing.assert_frame_equal(df0.drop(columns="generation_rule_mask"), ref)
    assert np.array_equal(df0["generation_rule_mask"].to_numpy(), get_generation_rule_mask(ref, SCORE2_ONLY_STRATEGY))


def test_multipliers_share_draws_across_spreads():
    pids = [f"P{i}" for i in range(1, 11)]
    a = physician_threshold_multipliers(pids, 0.1, SEED)
    b = physician_threshold_multipliers(pids, 0.2, SEED)
    for p in pids:  # log k scales linearly with the spread (common random numbers)
        assert np.log(b[p]) == pytest.approx(2 * np.log(a[p]))
    with pytest.raises(ValueError):
        physician_threshold_multipliers(pids, -0.1, SEED)


def test_each_physician_applies_its_own_thresholds(cohorts):
    df, params, mult = cohorts[0.2]
    assert len(set(np.round(list(mult.values()), 6))) == J
    risk = score2_ten_year_risk_percent_vectorized(df)
    age = df["age"].to_numpy()
    k = df["professional_id"].map(mult).to_numpy()
    t = np.where(age < 50, 2.5, np.where(age <= 69, 5.0, 7.5)) * k
    assert np.array_equal(df["generation_rule_mask"].to_numpy(), risk >= t)
    # Same patients and same outcome stream as the common-threshold cohort: only eligibility changes.
    df0 = cohorts[0.0][0]
    assert np.array_equal(df["age"].to_numpy(), df0["age"].to_numpy())
    assert (df["generation_rule_mask"] != df0["generation_rule_mask"]).any()


def test_outcomes_follow_physician_specific_eligibility(cohorts):
    df, params, _ = cohorts[0.2]
    m = df["generation_rule_mask"].to_numpy()
    y = df["recommendation"].to_numpy()
    group1 = df["professional_id"].map({p: v["group"] for p, v in params.items()}).to_numpy() == 1
    assert np.all(y[group1 & m] == 1) and np.all(y[group1 & ~m] == 0)


@pytest.fixture(scope="module")
def experiment():
    spec = ExperimentSpec("physthr_test", "physician_thresholds", seed=SEED, n_patients=N, n_physicians=J,
                          threshold_log_sd=0.2,
                          designs=["in_sample", "external", "oracle_eligible", "oracle_ineligible"])
    return generate_experiment(spec, "target_joint")


def test_truth_is_defined_within_each_physicians_stratum(experiment):
    truth = physician_truth(experiment).set_index("physician")
    df, m = experiment.df, experiment.eligible
    y = df["recommendation"].to_numpy()
    phys = df["professional_id"].astype(str).to_numpy()
    for p, row in truth.iterrows():
        sel = (phys == p) & m
        assert row["n_eligible"] == sel.sum()
        assert row["D_tilde"] == pytest.approx(_all_pairs_discordance(y[sel]))
        assert row["D_star"] == pytest.approx(2 * row["p_high"] * (1 - row["p_high"]))
        assert row["threshold_multiplier"] == pytest.approx(experiment.metadata["threshold_multipliers"][p])


def test_development_cohort_reuses_the_physician_thresholds(experiment):
    dev = generate_development_cohort(experiment)
    assert dev.n == N
    # apply_recorded_rule reproduces the benchmark eligibility on the benchmark cohort.
    assert np.array_equal(apply_recorded_rule(experiment.df, experiment.metadata), experiment.eligible)
    assert np.array_equal(physician_specific_score2_mask(experiment.df, experiment.metadata["threshold_multipliers"]),
                          experiment.eligible)


def test_run_and_tables_end_to_end(experiment):
    table, _ = run_experiment(experiment, [EuclideanEstimator(), LearnedWeightsEstimator()],
                              {"caliper_percentile": 25.0}, run_id="test")
    assert len(table) == 2 * 4 * J
    assert table["threshold_log_sd"].eq(0.2).all()
    ora = table[table["design"] == "oracle_eligible"]
    assert np.allclose(ora["n_candidates"], ora["n_eligible"])  # oracle pool = physician's own stratum
    tables = M.physician_thresholds_tables(table)
    assert {"E2_experiment_metrics", "E2_summary", "E2_decomposition", "E2_eligibility"} <= set(tables)
    dec = tables["E2_decomposition"]
    tot = dec[dec["metric"] == "total_bias"].set_index("method")["mean"]
    parts = dec[dec["metric"].isin(["comparability_bias", "estimation_bias"])].groupby("method")["mean"].sum()
    assert np.allclose(tot.sort_index(), parts.sort_index())


def test_plan_and_existing_families_unchanged():
    cfg = {"seed": 42, "families": {
        "score2": {"enabled": True, "n_patients": 1000, "n_physicians": 10, "designs": ["in_sample"]},
        "physician_thresholds": {"enabled": True, "n_patients": 1000, "n_physicians": 10, "seed": 642,
                                 "threshold_log_sds": [0, 0.1], "n_replicates": 2, "designs": ["in_sample"]}}}
    plan = build_plan(cfg)
    ids = [s.experiment_id for s in plan]
    assert ids == ["score2", "physthr_sd0_r00", "physthr_sd0.1_r00", "physthr_sd0_r01", "physthr_sd0.1_r01"]
    assert plan[0].threshold_log_sd is None
    assert plan[1].seed == plan[2].seed != plan[3].seed
