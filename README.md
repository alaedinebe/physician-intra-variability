# Physician Intra-Variability

Pipeline for analyzing intra-physician prescription variability from synthetic SCORE2 cohorts and several comparison methods (GLMM + matching).

This repository is a focused version of the analysis code and synthetic generator used in the *intra-physician variability* study.

## Overview

This repository lets you:

- generate synthetic patient cohorts under controlled clinical rules;
- run a multi-method statistical pipeline;
- compare approaches for estimating intra-physician discordance;
- produce reproducible artifacts (CSV, JSON reports, figures, effective configuration).

The entry point is `main.py`. The analytical core lives in `src/analysis/analysis_pipeline.py` and the synthetic generator in `src/synthetic_generator.py`.

## Project structure

```text
.
├── configs/                    # YAML configurations
├── experiments/                # Timestamped run outputs
├── src/
│   ├── analysis/               # Analytical pipeline, matching, visualizations
│   └── synthetic_generator.py  # Synthetic SCORE2 cohort generation
├── main.py                     # CLI orchestrator
└── LICENSE
```

## Prerequisites

- Python 3.11 or newer
- Python dependencies listed below

## Installation

Create a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas pyyaml numpy scipy scikit-learn matplotlib seaborn adjustText statsmodels
```

## Running

### Standard run

```bash
python main.py --config configs/article_stats.yaml
```

Results are written to `experiments/<date>/exp_<experiment_name>/`, with a copy of the effective configuration (`config_used.yaml`).

### Synthetic generation strategies

In `configs/article_stats.yaml`, the field `dataset.generation_strategy` accepts:

| Strategy | Description |
|----------|-------------|
| `score2_five_groups_heter_patients` | 20 physicians split into 5 groups with fixed prescription probabilities (default) |
| `score2_continuous_heterogeneity` | Continuous heterogeneity: each physician draws its own `p_high` |
| `score2_latent_covariate` | Latent covariate influencing prescription in the SCORE2 borderline zone |

Additional generator options:

- `dataset.covariate_correlation`: controlled correlation between two covariates (Gaussian copula);
- `dataset.hba1c_distribution`: HbA1c distribution (`normal` or `lognormal`);
- `dataset.save_raw_csv: true`: export the raw dataset to `data/dataset.csv`.

> **Note:** `main.py` only supports synthetic generation (`dataset.source: synthetic`). Other modes described in the YAML are ignored with a warning.

## Analysis methods

The pipeline compares, among others:

1. **Manual Pairing** — positive control based on the generation clinical rule;
2. **Euclidean / Mahalanobis / Learned Weights** — distance-based matching in covariate space;
3. **Genetic Matching** — optimization of covariate weights;
4. **Random Forest Proximity / Mutual Information** — learned similarity;
5. **LPA-guided Pairing** — pairing guided by a latent Gaussian mixture;
6. **Bayesian GLMM** — physician random effects and local overdispersion;
7. **Ensemble** — aggregation of the scores above.

Parameters (fixed effects, caliper, pairing strategy, LPA, etc.) are configured in the `analysis` section of the YAML.

## Main outputs

After a successful run, the results directory contains in particular:

| File / folder | Content |
|---------------|---------|
| `config_used.yaml` | Effective run configuration |
| `intra_physician_variability.csv` | Discordance scores per physician and per method |
| `global_correlations.csv` | Global correlations between methods |
| `within_group_spearman.csv` | Within-group Spearman correlations |
| `glmm/` | JSON report, calibration curves, Pearson residuals |
| `clinical_reviews/` | Matched pairs exported per method |
| `plots/method_comparison/comparison_3panels.png` | Visual comparison of methods |

In `score2_continuous_heterogeneity` mode, additional metrics are produced (`continuous_heterogeneity_*.csv` and associated figures).

## Configuration

The file `configs/article_stats.yaml` groups:

- project metadata and random seed;
- paths and experiment naming;
- synthetic generation parameters;
- analysis pipeline configuration (GLMM, caliper, LPA, fixed effects).

## License

MIT — see [LICENSE](LICENSE).

## Contact

- Alaedine Benani
- alaedine.benani@zoi.com
- https://alaedinebe.github.io/benani/
