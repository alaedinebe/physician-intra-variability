"""Ground truth per physician, long output table, experiment metrics, decomposition, provenance."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.benchmark import metrics as M
from src.benchmark.estimators import EuclideanEstimator, LearnedWeightsEstimator
from src.benchmark.runner import (
    ExperimentSpec,
    _all_pairs_discordance,
    code_fingerprint,
    generate_experiment,
    physician_truth,
    run_experiment,
)
from src.synthetic_generator import THRESHOLD_STRATEGY_TARGET_JOINT


@pytest.mark.parametrize("y", [[0, 1], [1, 1, 1], [0, 1, 1, 0, 1, 0, 0], list(np.random.default_rng(0).integers(0, 2, 25))])
def test_all_pairs_discordance_matches_brute_force(y):
    y = np.asarray(y)
    brute = np.mean([y[i] != y[j] for i, j in itertools.combinations(range(len(y)), 2)])
    assert _all_pairs_discordance(y) == pytest.approx(brute)
    assert np.isnan(_all_pairs_discordance(np.array([1])))


@pytest.fixture(scope="module")
def experiment():
    spec = ExperimentSpec("score2_test", "score2", seed=21, n_patients=2000, n_physicians=10,
                          designs=["in_sample", "oracle_eligible", "oracle_ineligible"])
    return generate_experiment(spec, THRESHOLD_STRATEGY_TARGET_JOINT)


@pytest.fixture(scope="module")
def long_table(experiment):
    table, fitted = run_experiment(experiment, [EuclideanEstimator(), LearnedWeightsEstimator()],
                                   {"caliper_percentile": 25.0}, run_id="test")
    return table


def test_physician_truth(experiment):
    truth = physician_truth(experiment)
    assert len(truth) == 10
    assert np.allclose(truth["D_star"], 2 * truth["p_high"] * (1 - truth["p_high"]))
    y = experiment.df["recommendation"].to_numpy()
    phys = experiment.df["professional_id"].astype(str).to_numpy()
    for _, row in truth.iterrows():
        sel = (phys == row["physician"]) & experiment.eligible
        assert row["D_tilde"] == pytest.approx(_all_pairs_discordance(y[sel]))
        assert row["n_eligible"] == sel.sum()
    # Deterministic physicians (group 1) have zero reference discordance.
    assert np.allclose(truth.loc[truth["physician_group"] == 1, "D_tilde"], 0.0)


def test_long_table_has_one_row_per_method_design_physician(long_table):
    assert len(long_table) == 2 * 3 * 10
    assert not long_table.duplicated(["method", "design", "physician"]).any()
    assert set(long_table["score_type"]) == {"discordance"}
    elig = long_table[long_table["design"] == "oracle_eligible"]
    assert np.allclose(elig["frac_pairs_both_eligible"], 1.0)


def test_experiment_metrics_against_manual_computation(long_table):
    exp = M.experiment_metrics(long_table, reference="D_tilde")
    row = exp[(exp["method"] == "euclidean") & (exp["design"] == "in_sample")].iloc[0]
    g = long_table[(long_table["method"] == "euclidean") & (long_table["design"] == "in_sample")]
    e = g["score"] - g["D_tilde"]
    assert row["bias"] == pytest.approx(e.mean())
    assert row["mae"] == pytest.approx(e.abs().mean())
    assert row["rmse"] == pytest.approx(np.sqrt((e ** 2).mean()))
    slope, intercept = np.polyfit(g["D_tilde"], g["score"], 1)
    assert row["calibration_slope"] == pytest.approx(slope)
    assert row["calibration_intercept"] == pytest.approx(intercept)
    from scipy.stats import spearmanr

    assert row["spearman"] == pytest.approx(spearmanr(g["score"], g["D_star"]).statistic)


def test_error_decomposition_is_exact(long_table):
    dec = M.decomposition(long_table, reference="D_star")
    assert set(dec["method"]) == {"euclidean", "learned_weights"}
    assert np.allclose(dec["total_bias"], dec["comparability_bias"] + dec["estimation_bias"])


def test_noise_floor_is_method_independent(long_table):
    nf = M.noise_floor(long_table)
    assert len(nf) == 1
    phys = long_table.drop_duplicates("physician")
    assert nf["mae"].iloc[0] == pytest.approx((phys["D_tilde"] - phys["D_star"]).abs().mean())


def test_calibration_and_rank_helpers_handle_degenerate_inputs():
    assert all(np.isnan(M._calibration(np.array([1.0, 2.0]), np.array([1.0, 2.0]))))
    assert all(np.isnan(M._calibration(np.ones(5), np.ones(5))))
    assert all(np.isnan(M._rank(np.arange(5.0), np.ones(5))))
    assert M._rank(np.arange(5.0), np.arange(5.0))[0] == pytest.approx(1.0)


def test_paired_bootstrap_is_reproducible():
    rng = np.random.default_rng(0)
    exp = pd.DataFrame({"experiment_id": np.repeat(np.arange(30), 2), "family": "progressive",
                        "method": np.tile(["a", "b"], 30), "bias": rng.normal(size=60)})
    s1 = M.paired_bootstrap_summary(exp, ["bias"], by=["method"])
    s2 = M.paired_bootstrap_summary(exp, ["bias"], by=["method"])
    pd.testing.assert_frame_equal(s1, s2)


def test_code_fingerprint_depends_on_the_configuration():
    a = code_fingerprint({"seed": 1})
    assert a == code_fingerprint({"seed": 1, "_private": "ignored"})
    assert a != code_fingerprint({"seed": 2})
    assert len(a) == 12
