"""The legacy Genetic Mahalanobis estimator reduces to Mahalanobis NN matching with replacement.

For arbitrary (random) weights, the legacy distance equals the Mahalanobis
distance to machine precision, and the legacy scores equal those of the
Mahalanobis estimator with nearest-neighbour pairing with replacement. The
difference between the legacy estimator and the Mahalanobis method of the
benchmark (optimal one-to-one pairing) is therefore due to the pairing rule only.

    python dev/check_genetic_legacy_reduces_to_mahalanobis.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.benchmark.estimators import (Cohort, GeneticMahalanobisEstimator,  # noqa: E402
                                      MahalanobisEstimator, estimate_matching)
from src.benchmark.runner import (FEATURES, OUTCOME_COL, PHYSICIAN_COL,  # noqa: E402
                                  ExperimentSpec, generate_experiment)

exp = generate_experiment(ExperimentSpec("score2", "score2", 142, 10000, 20, designs=["in_sample"]), "target_joint")
cohort = Cohort.from_frame(exp.df, FEATURES, OUTCOME_COL, PHYSICIAN_COL)
idx = np.arange(cohort.n)
maha = MahalanobisEstimator()
maha.prepare(cohort)
maha_nn = MahalanobisEstimator()
maha_nn.pairing = "nn_replacement"
maha_nn.prepare(cohort)
worst_d, worst_s = 0.0, 0.0
for seed in range(5):
    legacy = GeneticMahalanobisEstimator(variant="legacy")
    legacy.weights_ = np.sqrt(np.random.default_rng(seed).uniform(0, 10, cohort.X.shape[1]) + 1e-6)
    legacy.raw_weights_ = legacy.weights_ ** 2
    legacy.prepare(cohort)
    for p in np.unique(cohort.physician)[:3]:
        sub = np.where(cohort.physician == p)[0]
        worst_d = max(worst_d, float(np.abs(legacy.distance_matrix(sub) - maha.distance_matrix(sub)).max()))
    a = {e.physician: e.score for e in estimate_matching(legacy, cohort, idx, None, 25.0, 30)}
    b = {e.physician: e.score for e in estimate_matching(maha_nn, cohort, idx, None, 25.0, 30)}
    worst_s = max(worst_s, max(abs(a[k] - b[k]) for k in a))
print(f"max |distance legacy - Mahalanobis|           = {worst_d:.2e}")
print(f"max |score legacy - Mahalanobis NN replacement| = {worst_s:.2e}")
assert worst_d < 1e-9 and worst_s == 0.0
print("OK: the legacy weights have no effect.")
