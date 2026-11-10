# Physician intra-variability

Code for the simulation study "Evaluating estimators of within-physician prescribing discordance: a synthetic-ground-truth simulation study" (revised manuscript, release `v2.0-revision`).

The study generates synthetic patient cohorts in which the prescribing rule of every physician is known, so that the true within-eligible prescribing discordance of each physician is known by construction. It then measures how well eight estimators recover it: seven matching estimators (Euclidean, Mahalanobis, LPA-guided, Genetic Mahalanobis, Learned Weights, Mutual Information, Random Forest Proximity) and a generalized linear mixed model (GLMM).

No patient data are used. Every dataset is regenerated from the code and the configurations in `configs/`.

## Project structure

```text
.
├── configs/                    # One YAML file per family of experiments
├── src/
│   ├── synthetic_generator.py  # Synthetic cohorts, eligibility rules, physician models
│   ├── benchmark/              # Benchmark of the revised manuscript
│   │   ├── estimators.py       #   one class per estimator
│   │   ├── designs.py          #   in-sample, cross-fitted and oracle designs
│   │   ├── runner.py           #   experimental plan and long output table
│   │   ├── metrics.py          #   performance measures and summary tables
│   │   └── glmm_diagnostics.py #   GLMM specifications and diagnostics
│   └── analysis/               # Pipeline of the first submission (see below)
├── scripts/                    # Tables and figures of the manuscript
├── tests/                      # Unit tests
├── dev/                        # Development checks
└── main.py                     # Entry point of the first-submission pipeline
```

## Installation

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the manuscript

All commands are run from the repository root. Outputs are written to `results/<run_id>/`.

Each configuration file can be inspected or checked quickly before a full run:

```bash
python -m src.benchmark.runner --config configs/benchmark.yaml --list    # print the plan
python -m src.benchmark.runner --config configs/benchmark.yaml --smoke   # small end-to-end check
```

### 1. Run the experiments

| Experiments | Configuration | Output directory |
|---|---|---|
| Main benchmark: SCORE2 reference experiment, continuous heterogeneity, progressive multi-rule experiments, unobserved determinant, sensitivity analyses | `configs/benchmark.yaml` | `results/revision_main` |
| Monte Carlo grids over cohort size and number of physicians | `configs/grids.yaml` | `results/revision_grids` |
| Grid cells with small physician panels | `configs/grids_small.yaml` | `results/revision_grids_small` |
| GLMM specifications under continuous heterogeneity | `configs/glmm_continuous.yaml` | `results/revision_glmm_continuous` |
| Physician-specific eligibility thresholds | `configs/physician_thresholds.yaml` | `results/revision_physician_thresholds` |

```bash
python -m src.benchmark.runner --config configs/benchmark.yaml
python -m src.benchmark.runner --config configs/grids.yaml --workers 4
python -m src.benchmark.runner --config configs/grids_small.yaml --workers 4
python -m src.benchmark.runner --config configs/glmm_continuous.yaml --workers 4
python -m src.benchmark.runner --config configs/physician_thresholds.yaml --workers 4
```

Runs are resumable: each experiment is written to its own file as soon as it finishes, and an interrupted run skips the experiments already done. Runtimes measured with `--workers` above 1 are not comparable with those of a single-worker run.

### 2. Compute the performance measures

```bash
python -m src.benchmark.metrics --run results/revision_main
python -m src.benchmark.metrics --run results/revision_grids --add-run results/revision_grids_small
python -m src.benchmark.metrics --run results/revision_glmm_continuous
python -m src.benchmark.metrics --run results/revision_physician_thresholds
```

Each command writes summary tables to `<run>/tables/`.

### 3. Build the tables and figures

```bash
python scripts/manuscript_tables.py             # results/manuscript/tables/
python scripts/manuscript_figures.py            # results/manuscript/figures/
python scripts/physician_thresholds_report.py   # results/revision_physician_thresholds/report/
```

## Result tables

The summary tables behind the numbers of the manuscript are included, so that they can be checked without rerunning the simulations:

| Directory | Content |
|---|---|
| `results/revision_main/tables/` | Main benchmark |
| `results/revision_grids/tables/` | Monte Carlo grids, small-panel cells included |
| `results/revision_glmm_continuous/tables/` | GLMM specifications under continuous heterogeneity |
| `results/revision_physician_thresholds/tables/` and `report/` | Physician-specific thresholds: summary tables, Table S12, Figure S1 |

Each run directory also contains `manifest.json`, which records the commit, the code fingerprint, the library versions and the resolved configuration of the run. The long tables with one row per experiment, method, design and physician (`benchmark_long.csv.gz`) are not included because of their size; step 1 regenerates them.

## Provenance

Every row of the output table records the identifier of the launch and a fingerprint of the code and of the resolved configuration (SHA-256 of every `src/**/*.py` file and of the configuration). Results computed with a different version of the code cannot be mixed by mistake: if the code or the configuration has changed, the runner stops and lists the experiments concerned.

## First-submission pipeline

`main.py`, `src/analysis/` and `configs/article_stats.yaml` are the pipeline of the first submission. They are kept for reference and are not used by the revised manuscript.

```bash
python main.py --config configs/article_stats.yaml
```

## Tests

Unit tests cover the data generator (physician model, covariate model, eligibility rules and threshold calibration, closed-form truths of the unobserved-determinant scenario), the distances and the pairing algorithm (one-to-one pairing, caliper, matching with replacement, cancellation of the weights in the original Genetic Mahalanobis variant), the training designs (folds, cross-fitting without leakage, leave-one-physician-out, oracle strata), the GLMM and its variants (Pearson-residual score, calibration diagnostics) and the metrics (reference discordance, bias, MAE, calibration slope, Spearman correlation, error decomposition, code fingerprint).

```bash
pip install -r requirements.txt
python -m pytest            # about 15 seconds
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Alaedine Benani
- alaedine.benani@zoi.com
- https://alaedinebe.github.io/benani/
