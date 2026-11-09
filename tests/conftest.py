"""Shared fixtures of the unit tests.

Run from the repository root:

    python -m pytest tests

The fixtures simulate small cohorts with the benchmark generator, so that the
tests exercise the same code paths as the experiments while running in a few
seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.estimators import Cohort  # noqa: E402
from src.benchmark.runner import FEATURES, OUTCOME_COL, PHYSICIAN_COL  # noqa: E402
from src.synthetic_generator import (  # noqa: E402
    SCORE2_ONLY_STRATEGY,
    generate_score2_five_groups_heter_patients,
    get_generation_rule_mask,
)


@pytest.fixture(scope="session")
def score2_frame():
    """Five-group SCORE2 cohort: 2,000 patients, 10 physicians (100 per physician)."""
    return generate_score2_five_groups_heter_patients(n_patients=2000, expected_n_pros=10, random_state=11)


@pytest.fixture(scope="session")
def score2_cohort(score2_frame):
    df = score2_frame
    eligible = get_generation_rule_mask(df, SCORE2_ONLY_STRATEGY)
    cohort = Cohort.from_frame(df, FEATURES, outcome_col=OUTCOME_COL, physician_col=PHYSICIAN_COL)
    cohort.eligible = np.asarray(eligible, dtype=bool)
    return cohort


@pytest.fixture()
def toy_cohort():
    """Tiny cohort with three continuous covariates and two physicians (60 patients each)."""
    rng = np.random.default_rng(3)
    n = 120
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    physician = np.repeat(["A", "B"], n // 2)
    return Cohort(X=X, y=y, physician=physician, feature_names=["x1", "x2", "x3"])
