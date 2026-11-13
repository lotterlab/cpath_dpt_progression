# Analysis Code

This directory computes diffusion pseudotime (DPT) from previously generated
embeddings and runs the downstream analyses and plotting for the paper.

All scripts are run from the `analyze_embeddings/` directory.

For the command behind each figure and table, see
[../REPRODUCING.md](../REPRODUCING.md).

## Configuration (`config.py`)

Before running any scripts, review `config.py`. The key sections are:

**`RESULTS_RUN`** — Name of the results folder that plotting scripts read
(`results/<RESULTS_RUN>/`) and of the plots folder they write
(`plots/<RESULTS_RUN>/`). The default, `paper`, holds the cached results behind
the paper. Override it for one command with `RESULTS_RUN=<folder> python ...`.

**Data locations** — `SPIDER_SKIN_BUCKET`, `SPIDER_COLORECTAL_BUCKET` and
`SPIDER_BREAST_BUCKET` name the Google Cloud Storage buckets holding the
embeddings, and `SPIDER_IMAGE_ROOT` the local folder holding the unpacked
SPIDER images. Each can be set as an environment variable. See "Data access"
in the [top-level README](../README.md#data-access).

**`PROGRESSIONS`** — List of all disease progressions. Each entry specifies a GCS bucket, embedding prefix, ordered class list, root class (earliest stage), and image subdirectory. All 4 progressions are always defined; use the `--progressions` CLI argument on each script to select which ones to include in a run (defaults to all).

**`EXPECTED_MODELS`** — The 6 foundation models to evaluate: `virchow2`, `uni2`, `conch`, `gigapath`, `musk`, `dinov2`.

**`EVALUATION`** — Sampling and evaluation parameters:
- `n_per_class` (default: 1000) — number of patches sampled per disease stage
- `max_per_slide` (default: 50) — cap per slide to reduce slide-level bias
- `seed` (default: 42) — random seed for reproducibility
- `embedding_type` (default: `"final_embedding"`) — which embedding layer to use
- `n_bootstrap_ci` (default: 100) — bootstrap resampling iterations for metric confidence intervals (`evaluate_dpt.py`)
- `n_null_permutations` (default: 10) — label-shuffle permutations for null tau distribution (`generate_null_distribution.py`)
- `n_fewshot_trials` (default: 10) — repeated few-shot sampling trials per shot count (`evaluate_generalizability.py`)
- `fewshot_n_shots` (default: `[5, 10, 20]`) — shots per class for few-shot probing (`evaluate_generalizability.py`)

**`TEST_EVALUATION`** — Same keys as `EVALUATION` with minimal values for quick smoke tests (`n_per_class=50`, `n_bootstrap_ci=2`, etc.). Used automatically when any script is run with `--test`.

**`DPT`** — Diffusion pseudotime parameters:
- `n_neighbors` (default: 100) — k for the k-NN graph
- `n_diffusion_components` (default: 10) — number of diffusion components

**Embedding type lists** — `PATCH_EMBEDDINGS`, `CLS_EMBEDDINGS`, `REGISTER_EMBEDDINGS`, and `FINAL_EMBEDDING` define which intermediate-layer embeddings to evaluate.

## Output structure

```
results/
  paper/                  cached results used in the paper (read by plotting scripts)
    full_manifold_evaluation.csv
    null_manifold_evaluation.csv
    stage_permutation_specificity.csv
    generalizability_results.csv
    hyperparam_sweep_k_results.csv
    baseline_evaluation.csv
    ordering_agreement.csv
    cell_counts/          HistoPLUS cell counts along pseudotime, one CSV per model
    split_half/           slide-level halves and empirical sampling, one folder per progression
  2026-11-06_120000/      a new run: each data generation script writes to results/<timestamp>/
  test/                   --test mode output
plots/
  paper/                  the paper's figures and tables
```

Data generation scripts never write into `results/paper/`. To plot from a new
run, gather its CSVs into one folder under `results/` and set `RESULTS_RUN` to
that folder's name; plots then go to `plots/<that name>/`.

## Test mode

All scripts support a `--test` flag that runs with minimal samples for quick verification that data paths, progressions, and dependencies are working. Test mode:

- Uses `n_per_class=50`, `max_per_slide=10` (instead of 1000/50)
- Reduces bootstrap/permutation iterations to 2
- Writes output to `results/test/` (fixed directory, no timestamp)
- Does **not** affect plotting scripts or `results/paper/`

```bash
# Verify a single progression works end-to-end
python generate_data/evaluate_dpt.py --test --progressions BDC

# Verify all progressions
python generate_data/evaluate_dpt.py --test
```

## Running everything

`run_all.py` runs the six core data generation scripts in sequence (`evaluate_dpt.py` first, since `evaluate_generalizability.py` depends on its output). Scripts 1-4 run on all progressions with full config parameters. The hyperparameter sweep runs on its default progression (CRC-Serrated) and trajectory cell counts run on CRC-Conventional with UNI-2 at 2,000 patches per class. The baseline, ordering-agreement and split-half scripts (7-9 below) are run separately.

```bash
# Full pipeline
python generate_data/run_all.py

# Smoke test all scripts
python generate_data/run_all.py --test
```

If a script fails, the runner logs the error and continues to the next one. A summary is printed at the end showing pass/fail status and elapsed time for each step.

---

## Data generation scripts (`generate_data/`)

### 1. Full manifold evaluation (`evaluate_dpt.py`)

Computes DPT and a suite of manifold quality metrics (Kendall's tau, spectral gap, neighborhood purity, silhouette, trustworthiness, intrinsic dimension, pairwise AUROC) for every combination of progression, model, and embedding type. This is the primary evaluation script. Runs `n_bootstrap_ci` iterations (default: 100) on `final_embedding` to produce confidence intervals; other embedding types get a single point estimate.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `EVALUATION.n_bootstrap_ci` — bootstrap iterations for confidence intervals
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters
- `PATCH_EMBEDDINGS`, `CLS_EMBEDDINGS`, `REGISTER_EMBEDDINGS` — layer-wise embedding types to scan

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, final_embedding only, output to `results/test/` |

**Example:**
```bash
# Full run (all progressions, all models) — takes ~12 hours
python generate_data/evaluate_dpt.py

# Run only BDC and SCC progressions
python generate_data/evaluate_dpt.py --progressions BDC SCC

# Quick smoke test
python generate_data/evaluate_dpt.py --test --progressions BDC
```

**Output:** `results/<timestamp>/full_manifold_evaluation.csv`

---

### 2. Null distribution (`generate_null_distribution.py`)

Generates a null baseline by computing DPT on real data, then shuffling class labels and recalculating Kendall's tau. This measures what tau you'd expect by chance, providing a statistical baseline for the real results.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `EVALUATION.n_null_permutations` — number of label-shuffle permutations
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, output to `results/test/` |

**Example:**
```bash
# Standard run
python generate_data/generate_null_distribution.py

# Quick smoke test on one progression
python generate_data/generate_null_distribution.py --test --progressions BDC
```

**Output:** `results/<timestamp>/null_manifold_evaluation.csv`

---

### 3. Stage permutation test (`evaluate_stage_permutations.py`)

Exact permutation test that verifies high tau is specific to the biological ordering. Computes tau for ALL possible orderings of disease stages (3 classes = 6 permutations, 4 classes = 24). If the canonical ordering consistently yields the highest tau, the signal is biologically specific.

**Config fields used:**
- `PROGRESSIONS` — which disease progressions to evaluate (uses `classes` to generate permutations)
- `EXPECTED_MODELS` — which models to evaluate
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, output to `results/test/` |

**Example:**
```bash
# Full run
python generate_data/evaluate_stage_permutations.py

# Run on specific progressions
python generate_data/evaluate_stage_permutations.py --progressions CRC-Conventional CRC-Serrated

# Quick smoke test
python generate_data/evaluate_stage_permutations.py --test --progressions BDC
```

**Output:** `results/<timestamp>/stage_permutation_specificity.csv`

---

### 4. Generalizability analysis (`evaluate_generalizability.py`)

Leave-one-out cross-disease evaluation. For each target progression, calculates the average tau on the other progressions ("reference score"), then trains few-shot logistic regression probes (5, 10, 20 shots per class) on the target. Tests the hypothesis that high-tau models are more data-efficient on unseen diseases.

**Prerequisite:** Requires a completed `full_manifold_evaluation.csv` from script 1. It is read from `results/<RESULTS_RUN>/` (`config.FULL_RESULTS_OUTPUT_PATH`), so by default the cached paper results are used.

**Config fields used:**
- `PROGRESSIONS` — progressions for leave-one-out (default: all 4)
- `FULL_RESULTS_OUTPUT_PATH` — reads tau values from the main evaluation
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide` — sampling parameters
- `EVALUATION.n_fewshot_trials` — repeated sampling trials per shot count
- `EVALUATION.fewshot_n_shots` — shots per class for few-shot probing

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progressions` | all | Progression names to include (e.g. `--progressions BDC SCC CRC-Conventional`) |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, output to `results/test/` |

**Example:**
```bash
# Full run (requires all 4 progressions in full_manifold_evaluation.csv)
python generate_data/evaluate_generalizability.py

# Quick smoke test
python generate_data/evaluate_generalizability.py --test
```

**Output:** `results/<timestamp>/generalizability_results.csv`

---

### 5. Cell count trajectory analysis (`analyze_trajectory_cellcounts.py`)

Computes DPT for a single progression/model, then runs HistoPLUS cell segmentation on each patch to extract per-cell-type counts along the pseudotime trajectory. Requires local image files and the HistoPLUS package.

**Config fields used:**
- `PROGRESSIONS` — looks up the named progression
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed`, `EVALUATION.embedding_type` — sampling and embedding parameters
- `DPT.n_neighbors`, `DPT.n_diffusion_components` — DPT graph parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progression` | `CRC-Conventional` | Disease progression name (must match a `config.PROGRESSIONS` entry) |
| `--model` | `uni2` | Foundation model to use for pseudotime |
| `--n_per_class` | 1000 (from config) | Patches per class |
| `--test` | off | Test mode: 3 patches per class, output to `results/test/` |
| `--seed` | 42 (from config) | Random seed |
| `--data_dir` | (auto from bucket) | Override local image directory |

**Example:**
```bash
# Quick test
python generate_data/analyze_trajectory_cellcounts.py --test

# Full run for a specific progression and model
python generate_data/analyze_trajectory_cellcounts.py --progression BDC --model uni2 --n_per_class 500

# Custom image directory
python generate_data/analyze_trajectory_cellcounts.py --data_dir ~/data/spider-breast
```

**Output:** `results/<timestamp>/<progression>_<model>_<n>_cellcounts.csv`

---

### 6. Hyperparameter sweep (`hyper_param_sweep.py`)

Sweeps over the number of nearest neighbors (k) used in DPT graph construction, measuring tau, trustworthiness, and spectral gap at each value. Also generates a visualization plot. `--plot-only` skips the sweep and redraws the plot from the stored CSV.

**Config fields used:**
- `PROGRESSIONS` — looks up the target progression
- `EXPECTED_MODELS` — which models to sweep
- `EVALUATION.n_per_class`, `EVALUATION.max_per_slide`, `EVALUATION.seed` — sampling parameters

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--progression` | `CRC-Serrated` | Target progression |
| `--k_values` | `10 30 50 75 100 200` | Space-separated list of k values to test |
| `--test` | off | Test mode: uses `TEST_EVALUATION` config, k_values=[50, 100], output to `results/test/` |

**Example:**
```bash
# Default sweep
python generate_data/hyper_param_sweep.py

# Sweep a different progression with custom k values
python generate_data/hyper_param_sweep.py --progression BDC --k_values 25 50 100 150

# Quick smoke test
python generate_data/hyper_param_sweep.py --test
```

**Output:** `results/<timestamp>/hyperparam_sweep_k_results.csv` and `k_sweep_analysis.png`

---

### 7. Ordering baselines (`evaluate_baselines.py`)

Computes Kendall's tau for four alternatives to DPT on the same cohorts: cosine distance to the earliest-class medoid, projection onto the first principal component, the first diffusion component alone (DC1), and shortest-path distance over DPT's k-NN graph. Used for Tables 2 and 3.

```bash
python generate_data/evaluate_baselines.py
python generate_data/evaluate_baselines.py --test --progressions BDC
```

**Output:** `results/<timestamp>/baseline_evaluation.csv`

---

### 8. Cross-method ordering agreement (`evaluate_ordering_agreement.py`)

Measures whether the ordering methods recover the same per-patch ordering, as the Spearman correlation between each pair of methods for every model and progression (Appendix C.1).

```bash
python generate_data/evaluate_ordering_agreement.py --save-scores
```

**Output:** `results/<timestamp>/ordering_agreement.csv`, `ordering_agreement_tau_check.csv`, and with `--save-scores` the per-patch scores in `ordering_scores.npz`

---

### 9. Slide-level halves and empirical sampling (`evaluate_split_half.py`)

Recomputes tau when patches are sampled at the cohort's empirical class frequencies instead of equally per class, and within two disjoint slide-level halves of the cohort. Each condition is repeated over 10 seeds. Used for Table 1.

```bash
python generate_data/evaluate_split_half.py --progression SCC --n-total 4000
python generate_data/evaluate_split_half.py --progression BDC --n-total 1500
```

**Output:** `results/<timestamp>/split_half.csv` (or the folder given with `--out`)

---

## Plotting scripts (`plotting/`)

Each script reads from `results/<RESULTS_RUN>/` and writes to `plots/<RESULTS_RUN>/`.

| Script | Produces |
|---|---|
| `plot_method_overview_figure.py` | Fig 1 |
| `plot_trajectory_fidelity.py` | Fig 2, Table 4 |
| `plot_avg_permutation_results.py` | Fig 3 |
| `plot_permutation_results.py` | Fig 11 |
| `plot_layer_analysis.py` | Figs 4, 12-14 |
| `plot_generalizability_test.py` | Figs 5, 15, 16 |
| `plot_cell_pattern_v2.py` | Figs 6, 17, 18, 19 |
| `plot_separability_vs_fidelity_simulation.py` | Fig 8 |
| `plot_model_comparison.py` | Fig 9 |
| `plot_appendix.py` | Fig 10, Table 5 |
| `inspect_trajectory.py` | Fig 20 |
| `plot_trajectory_patch_grid.py` | Fig 21 |
| `make_tables.py` | Tables 1-3 |
| `plot_config.py` | Shared labels, colors and style |

Fig 7 is drawn by `generate_data/hyper_param_sweep.py --plot-only`.

---

## Directory structure

**`generate_data/`**
Data generation scripts (described above). Each writes output to `results/<timestamp>/`.

**`analysis/`**
Core library code. `dpt.py` contains the DPT implementation, metric functions, and data structures (`DPTConfig`, `DPTResult`, `DPTMetric`) used by the generation scripts. `baselines.py` contains the alternative ordering methods.

**`data/`**
Dataset loader (`progression_embedding_dataset.py`) that handles GCS-backed embedding retrieval, slide-aware sampling, and in-memory caching.

**`results/`**
CSV outputs from data generation scripts. `results/paper/` holds the cached results used in the paper; new runs go to `results/<timestamp>/` and `--test` output to `results/test/`.

**`plotting/`**
Scripts that generate the paper's figures and tables from `results/<RESULTS_RUN>/`.

**`plots/`**
Rendered figures and tables, organized by analysis type (e.g. `emergence/`, `trajectory_fidelity/`, `permutation/`). `plots/paper/` holds the versions in the paper.

**`validate_stored_embeddings.py`**
Checks that every model has embeddings for the same patches in each bucket.
