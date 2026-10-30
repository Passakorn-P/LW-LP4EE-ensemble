# Replication Package: LW-LP4EE Ensemble: A Practical Approach to Locally Weighted Linear Programming for Software Effort Estimation

This repository contains the replication package for the manuscript *LW-LP4EE Ensemble: A Practical Approach to Locally Weighted Linear Programming for Software Effort Estimation*. It provides the source code, datasets and execution configuration required to reproduce all the experiments for the four research questions.

---

## Repository Structure

```
LW-LP4EE-ensemble/
├── codes/
│   ├── run.py                        # Entry point: run the full pipeline from here
│   ├── configs.py                    # Datasets, kernels, bandwidths, named subsets, baseline grids, paths
│   ├── data_handler.py               # Dataset loading and the nine preprocessing variants
│   ├── error_measures.py             # MAE, MdAE, LSD and per-instance basis errors
│   ├── models/
│   │   ├── lw_lp4ee.py               # LW-LP4EE model (lpSolve)
│   │   ├── cached_lw_lp4ee.py        # Cache-backed fixed and nested-LOO bandwidth lookups
│   │   ├── lp4ee.py                  # LP4EE baseline through R (rpy2)
│   │   ├── lp.R                      # Original R implementation of LP4EE
│   │   └── sropt.py                  # SROpt baseline (lpSolve)
│   ├── experiments/
│   │   ├── cache.py                  # Stage 1: LW-LP4EE prediction cache builder
│   │   ├── lw_experiments.py         # RQ1-RQ3 experiments computed from the cache
│   │   └── baselines.py              # The eight baselines (outer LOO + inner-LOO grid search)
│   └── evaluations/
│       ├── plots.py                  # RQ1-RQ3 tables and figures
│       ├── kernel_dissimilarity.py   # Kernel dissimilarity analysis and kernel-shape figure
│       └── stats.py                  # RQ4: WRS tests, Holm adjustment, win/tie/loss table, figure
├── datasets/                    # 8 SEACRAFT effort datasets (xml + conf)
├── results/                     # Generated at runtime
└── requirements.txt
```

---

## Prerequisites

### 1) Python environment

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

`lpsolve55` (Python bindings of lp_solve 5.5) is **not** available from PyPI and must be installed separately (for example from conda or by building the lp_solve Python driver). Verify with:

```bash
python -c "from lpsolve55 import lpsolve"
```

### 2) R, `linprog` and the WRS package

R is used through `rpy2` for the LP4EE baseline and for the statistical tests.

- `linprog` and `lpSolve` (CRAN): `install.packages(c("linprog", "lpSolve"))` — used by `codes/models/lp.R` (`linprog::solveLP(lpSolve = TRUE)`).
- `WRS` (**not** `WRS2`): from GitHub <https://github.com/nicebread/wrs>, following the instructions in its README.
  - `WRS::bprm` — Brunner test (p-value)
  - `WRS::cid` — Cliff's delta effect size

Verify in the R console:

```R
library(linprog)
library(WRS)
```

---

## Running the Pipeline

All stages are managed via a single entry point and must be run in this order from `codes/`:

```bash
cd codes

# Stage 1: Build the LW-LP4EE prediction cache (10 kernels x 10 bandwidths, outer + nested LOO)
python run.py --stage cache

# Stage 2: Run the cache-backed experiments and report RQ1-RQ3
python run.py --stage rq1-rq3

# Stage 3: Run the 8 baselines
python run.py --stage baselines

# Stage 4: Compare the ensemble against the baselines and report RQ4
python run.py --stage rq4
```

Optional flags: 
- `--datasets <names...>` (cache and baselines stages), 
- `--models <names...>` (baselines stage), 
- `--reset` (recompute existing experiment files or baseline predictions).
> **Note (dependency of RQ1-RQ3 on LP4EE):** RQ1-RQ3 count improvements over the original LP4EE. If `results/baseline_results/LP4EE/` is missing, `--stage rq1-rq3` computes only the `LP4EE` baseline first.

### Output paths (relative to project root)

| Stage | Output |
|---|---|
| `cache` | `results/lw_cache/<dataset>.parquet`, `results/lw_cache/<dataset>_nested.parquet` |
| `rq1-rq3` | `results/exp_from_cache/exp1_fixed_all_kernels.parquet`, `exp2_cv_all_subsets_kernels_full.parquet`, `exp3_ensemble_full.parquet` |
| `rq1-rq3` | `results/tables/rq1_improvement_counts.csv`, `kernel_dissimilarity.txt` |
| `rq1-rq3` | `results/plot/rq2_loocv_kernel_performance.eps`, `rq3_ensemble_joint_region.eps`, `rq3_2_joint_region.eps`, `related_work_kernels_grid.eps` |
| `baselines` | `results/baseline_results/<model>/<dataset>.parquet` |
| `rq4` | `results/stat_tests/wtl_summary_sropt_style.tex`, `wtl_overall_lsd_aligned.csv`, `wtl_by_dataset_lsd_aligned.csv`, `raw_pairwise_lsd_aligned.parquet` |
| `rq4` | `results/plot/rq4_raw_error_range_by_dataset.eps`, `results/tables/rq4_metrics_by_dataset.csv`, `rq4_ranks_*.csv` |

`LP4EE` and `LP4EE_SROpt` use fixed MinMax scaling with all features; the other baselines select their own preprocessing by grid search.

---

## Datasets

The 8 SEACRAFT datasets in `datasets/` (Cocomo81-e, Cocomo81-non-e, Desharnais, Kitchenham, Maxwell, Miyazaki94, Nasa93-c2, Nasa93-c5) are stored as `<name>.xml` + `<name>.conf`. The `.conf` file gives the target column (renamed to `Effort`), the size column (renamed to `Size`) and the ignored columns.

**ISBSG datasets are intentionally not included.** The paper additionally uses subsets of the ISBSG repository, which is commercial and not freely redistributable. This package therefore reproduces the SEACRAFT part of the study.

---

## Experimental Design Summary

| Parameter | Configuration |
|:--|:--|
| **Datasets** | 8 SEACRAFT effort datasets (in `datasets/`) |
| **Evaluation** | Leave-one-out cross-validation (outer); nested leave-one-out for bandwidth selection |
| **Model** | LW-LP4EE, solved with lpSolve; MinMax scaling fitted on the training part of every fold |
| **Kernels** | rectangular, epanechnikov, tricube, gaussian, triangle, triweight, biweight, cosine, logistic, sigmoid |
| **Bandwidths** | 0.1, 0.2, ..., 1.0 (k = round(n · b) nearest neighbours define the radius) |
| **RQ1** | Fixed bandwidth, every (kernel, bandwidth); improvement count over LP4EE on MAE / MdAE / LSD |
| **RQ2** | Nested-LOO bandwidth selection over every bandwidth subset of size >= 2 |
| **RQ3** | Mean-ensembles over all non-empty kernel subsets x bandwidth subsets; named subsets (All, Top, Diverse, Similar) |
| **Adopted ensemble (RQ4)** | Diverse3 kernels (rectangular, cosine, sigmoid) x bandwidths {0.1, 0.9, 1.0} |
| **Baselines (RQ4)** | LP4EE, SROpt, ABE, ANN, CART, GBM, RF, SVR (grid search with inner LOO MSE) |
| **Metrics reported** | MAE, MdAE, LSD |
| **Statistical test** | Brunner matched-pairs test (`WRS::bprm`) on absolute errors (MAE/MdAE) or absolute log errors (LSD), Holm adjustment per dataset, alpha = 0.05 |
| **Effect size** | Cliff's delta (`WRS::cid`), threshold = 0.147 (negligible) |

---

## Execution Notes and System Constraints

- Run every stage from `codes/` (the paths `../datasets/`, `../results/` and `models/lp.R` are relative to it).
- The cache stage is by far the most expensive: it solves one LP per (outer fold, inner fold, kernel, bandwidth). `N_WORKERS` in `configs.py` (CPU count minus 1) sets the number of processes in every parallel stage. Finished datasets are skipped on restart.
- BLAS thread counts are capped at 1 per worker with `threadpoolctl` during the baseline grid searches.
