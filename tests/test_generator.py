"""Synthetic data generator: physician model, eligibility rules, thresholds and E1 truths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.benchmark.runner import FEATURES
from src.synthetic_generator import (
    MIN_PATIENTS_PER_PHYSICIAN_BASE,
    SCORE2_ONLY_STRATEGY,
    _five_group_size,
    build_progressive_rule_plan,
    calibrate_joint_thresholds,
    expected_progressive_rule_count,
    five_group_physician_parameters,
    generate_score2_five_groups_heter_patients,
    get_generation_rule_mask,
    latent_influence_physician_truth,
    max_attainable_joint_eligibility,
)


def test_five_group_parameters_match_table_4():
    params = five_group_physician_parameters(20)
    assert len(params) == 20
    expected = {1: (1.0, 0.0), 2: (0.9, 0.05), 3: (0.8, 0.1), 4: (0.7, 0.2), 5: (0.5, 0.5)}
    groups = pd.Series({p: v["group"] for p, v in params.items()})
    assert groups.value_counts().to_dict() == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
    for par in params.values():
        assert (par["p_high"], par["p_low"]) == pytest.approx(expected[par["group"]])


def test_five_group_design_requires_a_multiple_of_five():
    assert _five_group_size(100) == 20
    with pytest.raises(ValueError):
        _five_group_size(12)


def test_cohort_shape_floor_and_reproducibility(score2_frame):
    df = score2_frame
    assert set(FEATURES) <= set(df.columns)
    assert df["professional_id"].nunique() == 10
    assert df["professional_id"].value_counts().min() >= MIN_PATIENTS_PER_PHYSICIAN_BASE
    again = generate_score2_five_groups_heter_patients(n_patients=2000, expected_n_pros=10, random_state=11)
    pd.testing.assert_frame_equal(df, again)
    other = generate_score2_five_groups_heter_patients(n_patients=2000, expected_n_pros=10, random_state=12)
    assert not np.array_equal(df["age"].to_numpy(), other["age"].to_numpy())


def test_prescriptions_follow_the_physician_model(score2_frame):
    df = score2_frame
    eligible = get_generation_rule_mask(df, SCORE2_ONLY_STRATEGY)
    params = five_group_physician_parameters(10)
    phys = df["professional_id"].astype(str).to_numpy()
    y = df["recommendation"].to_numpy()
    group1 = np.array([params[p]["group"] == 1 for p in phys])
    # Group 1 is deterministic: every eligible patient is prescribed, no ineligible one is.
    assert np.all(y[group1 & eligible] == 1)
    assert np.all(y[group1 & ~eligible] == 0)
    # Group 5 prescribes with probability 0.5 whatever the eligibility.
    group5 = np.array([params[p]["group"] == 5 for p in phys])
    assert abs(y[group5].mean() - 0.5) < 0.08


def test_literature_covariate_moments():
    df = generate_score2_five_groups_heter_patients(n_patients=20000, expected_n_pros=20, random_state=5)
    assert df["age"].mean() == pytest.approx(57.6, abs=0.5)
    assert df["ldl_cholesterol"].mean() == pytest.approx(3.56, abs=0.05)
    assert df["hdl_cholesterol"].mean() == pytest.approx(1.45, abs=0.03)
    assert df["is_male"].mean() == pytest.approx(0.44, abs=0.02)
    assert df["is_smoker"].mean() == pytest.approx(0.15, abs=0.02)
    # Non-HDL is LDL plus triglycerides / 2.2, hence never below LDL.
    assert np.all(df["non_hdl_cholesterol"] >= df["ldl_cholesterol"] - 1e-9)


def test_progressive_plan_has_45_windows_per_pass():
    plan = build_progressive_rule_plan(FEATURES, n_passes=2)
    windows = [r for r in plan if r["rule_type"] != "score2"]
    assert len(plan) == expected_progressive_rule_count(9, 2) == 91
    assert len(windows) == 90
    for r in windows:
        assert len(r["variables"]) == r["window_size"]
        assert r["variables"] == FEATURES[r["window_start"]:r["window_start"] + r["window_size"]]


def _calibration_frame(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.uniform(size=n), "c": rng.normal(size=n) * 2 + rng.normal(size=n),
        "flag": rng.binomial(1, 0.7, size=n),
    })


@pytest.mark.parametrize("target", [0.2, 0.45, 0.7])
def test_joint_threshold_calibration_reaches_the_target(target):
    df = _calibration_frame()
    out = calibrate_joint_thresholds(df, ["a", "b", "c"], target)
    assert out["attainable"]
    assert out["realised_eligibility"] == pytest.approx(target, abs=0.01)
    assert out["mask"].mean() == pytest.approx(out["realised_eligibility"])
    assert out["per_variable_quantile"] <= 0.95
    # The mask is exactly the conjunction of the recorded conditions.
    mask = np.ones(len(df), dtype=bool)
    for col, (op, value) in out["conditions"].items():
        assert op == "<="
        mask &= df[col].to_numpy() <= value
    assert np.array_equal(mask, out["mask"])


def test_binary_covariates_are_fixed_at_their_majority_level():
    df = _calibration_frame()
    out = calibrate_joint_thresholds(df, ["flag"], 0.4)
    assert out["per_variable_quantile"] is None
    assert out["conditions"]["flag"] == ("==", 1.0)
    assert out["realised_eligibility"] == pytest.approx(df["flag"].mean())
    assert not out["attainable"]  # 0.4 cannot be reached with a single binary covariate


def test_target_above_the_attainable_maximum_is_flagged():
    df = _calibration_frame()
    cap = max_attainable_joint_eligibility(df, ["a", "b", "c"])
    out = calibrate_joint_thresholds(df, ["a", "b", "c"], min(0.99, cap + 0.1))
    assert not out["attainable"]
    assert out["realised_eligibility"] == pytest.approx(cap, abs=1e-12)


@pytest.mark.parametrize("p_high", [0.5, 0.7, 0.9])
def test_latent_truths_without_unobserved_influence(p_high):
    t = latent_influence_physician_truth(p_high, 0.0)
    assert t["D_observable"] == pytest.approx(2 * p_high * (1 - p_high))
    assert t["D_intrinsic"] == pytest.approx(t["D_observable"])


@pytest.mark.parametrize("p_high,gamma", [(0.5, 1.0), (0.7, 2.0), (0.9, 4.0)])
def test_latent_truths_with_unobserved_influence(p_high, gamma):
    t = latent_influence_physician_truth(p_high, gamma)
    # Monte Carlo check of the Gauss-Hermite quadrature: E[pi(U)] = p_high and gap = 2 Var(pi(U)).
    u = np.random.default_rng(1).standard_normal(400_000)
    pi = 1.0 / (1.0 + np.exp(-(t["alpha"] + gamma * u)))
    assert pi.mean() == pytest.approx(p_high, abs=2e-3)
    assert t["D_observable"] == pytest.approx(2 * p_high * (1 - p_high))
    assert t["D_intrinsic"] < t["D_observable"]
    assert t["D_observable"] - t["D_intrinsic"] == pytest.approx(2 * pi.var(), abs=2e-3)


def test_deterministic_physician_has_zero_discordance():
    t = latent_influence_physician_truth(1.0, 4.0)
    assert t["D_observable"] == 0.0 and t["D_intrinsic"] == 0.0
