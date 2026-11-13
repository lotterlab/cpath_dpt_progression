# Reproducing the paper

Every figure and table in the paper is listed below with the command that
produces it, the cached file it reads, and where the output is stored.

Run all commands from `analyze_embeddings/` unless a row says otherwise.
Stored results are in `analyze_embeddings/results/paper/` and stored figures in
`analyze_embeddings/plots/paper/`; the paths in the tables are relative to those
two folders.

## What you need for each item

| Level | Needs | Covers |
|---|---|---|
| **A** | Only this repository (`pip install -r requirements.txt`) | Redrawing figures and tables from the cached results |
| **B** | Level A plus the patch embeddings in a Google Cloud Storage bucket | Recomputing the cached results; Fig 9 |
| **C** | Level B plus the SPIDER images; a GPU for embeddings; HistoPLUS for cell segmentation | Generating embeddings; cell counts; Figs 1, 20 and 21 |

See [Data access](README.md#data-access) for how to set up levels B and C.

Scripts in `generate_data/` write to a new `results/<timestamp>/` folder and
never overwrite `results/paper/`. To plot from a new run, pass its folder name
as `RESULTS_RUN`, for example
`RESULTS_RUN=2026-11-06_120000 python plotting/plot_trajectory_fidelity.py`.

## Main text

| Item | Command | Reads | Output (`plots/paper/`) | Level |
|---|---|---|---|---|
| Fig 1, method overview | `python plotting/plot_method_overview_figure.py --progression SCC --model virchow2` | embeddings, images in the bucket | `method_overview/method_SCC_virchow2.png` | C |
| Fig 2, trajectory fidelity | `python plotting/plot_trajectory_fidelity.py` | `full_manifold_evaluation.csv`, `null_manifold_evaluation.csv` | `trajectory_fidelity/model_comparison_with_null.png` | A |
| Fig 3, stage-order permutations (averaged) | `python plotting/plot_avg_permutation_results.py` | `stage_permutation_specificity.csv`, `full_manifold_evaluation.csv` | `permutation/permutation_plot_averaged.png` | A |
| Fig 4, fidelity across depth | `python plotting/plot_layer_analysis.py` | `full_manifold_evaluation.csv` | `emergence/emergence_depth_tau_cls.png` | A |
| Fig 5, held-out rank transfer | `python plotting/plot_generalizability_test.py` | `generalizability_results.csv`, `full_manifold_evaluation.csv` | `generalization/main_panel_b_rank_preservation.png` | A |
| Fig 6, cell types along pseudotime (UNI-2) | `python plotting/plot_cell_pattern_v2.py` | `cell_counts/crc_conventional_uni2_2000_cellcounts.csv` | `biological_validation_analysis/standardized_abundance.png` | A |

## Appendix

| Item | Command | Reads | Output (`plots/paper/`) | Level |
|---|---|---|---|---|
| Fig 7, sensitivity to k | `python generate_data/hyper_param_sweep.py --plot-only` | `hyperparam_sweep_k_results.csv` | `hyperparams/k_sweep_analysis.png` | A |
| Table 1, slide halves and empirical sampling | `python plotting/make_tables.py` | `split_half/*/split_half.csv`, `full_manifold_evaluation.csv` | `tables/table1_resampling.csv` | A |
| Fig 8, separability vs fidelity simulation | `python plotting/plot_separability_vs_fidelity_simulation.py` | nothing (simulated data) | `simulation/separability_vs_fidelity.png` | A |
| Fig 9, eosinophils for MuSK vs DINOv2, and the silhouette scores in Appendix C | `python plotting/plot_model_comparison.py --models musk dinov2` | embeddings, `cell_counts/*_{musk,dinov2}_2000_cellcounts.csv` | `model_comparison/per_celltype/eosinophils.png`, `model_comparison/cluster_metrics_and_dpt.csv` | B |
| Table 2, DPT vs other ordering methods | `python plotting/make_tables.py` | `baseline_evaluation.csv`, `full_manifold_evaluation.csv` | `tables/table2_ordering_baselines.csv` | A |
| Table 3, model rankings per method | `python plotting/make_tables.py` | same as Table 2 | `tables/table3_model_rankings.csv` | A |
| Per-patch agreement between methods (C.1 text) | `python plotting/make_tables.py` | `ordering_agreement.csv` | `tables/ordering_agreement_summary.csv` | A |
| Table 4, all fidelities with 95% CI | `python plotting/plot_trajectory_fidelity.py` | `full_manifold_evaluation.csv` | `trajectory_fidelity/model_progression_trajectory_fidelity.csv` | A |
| Table 5, manifold quality metrics | `python plotting/plot_appendix.py` | `full_manifold_evaluation.csv` | `appendix_metrics_table.csv` | A |
| Fig 10, manifold stability | `python plotting/plot_appendix.py` | `full_manifold_evaluation.csv` | `appendix_manifold_robustness.png` | A |
| Fig 11, stage-order permutations per progression | `python plotting/plot_permutation_results.py --null-as-dots --no-ci` | `stage_permutation_specificity.csv`, `full_manifold_evaluation.csv` | `permutation/permutation_plot_dots_no_ci.png` | A |
| Figs 12–14, depth analysis by token type | `python plotting/plot_layer_analysis.py` | `full_manifold_evaluation.csv` | `emergence/emergence_three_panel_{patch_mean,cls,register_mean}.png` | A |
| Fig 15, fidelity vs few-shot F1 within disease | `python plotting/plot_generalizability_test.py` | `generalizability_results.csv`, `full_manifold_evaluation.csv` | `generalization/main_panel_a_within_disease.png` | A |
| Fig 16, reference τ vs reference F1 | `python plotting/plot_generalizability_test.py` | same as Fig 15 | `generalization/appendix_reference_metric_comparison.png` | A |
| Fig 17 (panels a–c), sampling diagnostics | `python plotting/plot_cell_pattern_v2.py` | `cell_counts/crc_conventional_uni2_2000_cellcounts.csv` | `biological_validation_analysis/appendix/pseudotime_density_by_stage.png` (a), `pseudotime_density_by_slide.png` (b), `smoothing_robustness_fibroblasts.png` (c) | A |
| Fig 18, all 13 cell types | `python plotting/plot_cell_pattern_v2.py` | same as Fig 17 | `biological_validation_analysis/appendix/cell_types_zscore_and_raw.png` | A |
| Fig 19a, Virchow-2 | `python plotting/plot_cell_pattern_v2.py --model virchow2 --input results/paper/cell_counts/crc_conventional_virchow2_2000_cellcounts.csv` | the `--input` file | `biological_validation_analysis/virchow2/standardized_abundance.png` | A |
| Fig 19b, Prov-GigaPath | `python plotting/plot_cell_pattern_v2.py --model gigapath --input results/paper/cell_counts/crc_conventional_gigapath_2000_cellcounts.csv` | the `--input` file | `biological_validation_analysis/gigapath/standardized_abundance.png` | A |
| Fig 20, sampled patches with cell segmentation | `python plotting/inspect_trajectory.py --progression CRC-Conventional --model uni2 --seed 2539709969` | embeddings, local images, HistoPLUS | `trajectories/trajectory_uni2_CRC-Conventional_prop_seed2539709969_all.png` | C |
| Fig 21, sampled patches and pseudotime neighbours | `python plotting/plot_trajectory_patch_grid.py --progression CRC-Conventional --model uni2 --seed 1426409426` | embeddings, local images | `trajectories/trajectory_combined_uni2_CRC-Conventional_8x4_seed1426409426.png` (and `.csv` listing every patch shown) | C |
| Table 6, TCGA-KIRC endpoints | `python make_table6.py` from `tcga_kirc/analysis/` | `tcga_kirc/analysis/results/paper/label_shuffle_null.csv` | `tcga_kirc/analysis/results/paper/table6_kirc_concordance.csv` | A |

Figure and table numbers follow the camera-ready paper. Several scripts also
write extra diagnostic files next to the ones listed here; these are not in the
paper. In Figs 3, 10, 11 and 12–14 the horizontal jitter of points and the
bootstrapped bands are drawn without a fixed seed, so a redrawn figure differs
from the stored one in those details only; the plotted values are identical.

## Numbers quoted in the text

Values stated in the prose rather than in a figure or table. Paths are relative
to `results/paper/` or `plots/paper/` as above.

| Statement (section) | Value | Where it comes from |
|---|---|---|
| Mean τ per model: UNI-2, Virchow-2, MuSK (4.1.1) | 0.73, 0.72, 0.59 | `tables/table3_model_rankings.csv`, DPT rows (mean of `tau` over progressions, `final_embedding` rows of `full_manifold_evaluation.csv`) |
| DINOv2 on BDC; MuSK range across progressions (4.1.1) | 0.22; 0.34–0.70 | `trajectory_fidelity/model_progression_trajectory_fidelity.csv` (Table 4) |
| DPT improvement over centroid distance and PCA (4.1.2, C.1) | 14%, 36% | `tables/table2_ordering_baselines.csv`: (DPT − baseline) / baseline for each model–progression pair, averaged over the 24 pairs |
| Mean τ for DPT, DC1, k-NN geodesic (4.1.2, C.1) | 0.637, 0.604, 0.582 | `tables/table2_ordering_baselines.csv`, Mean row |
| Per-patch Spearman agreement with DPT (4.1.2, C.1) | 0.86, 0.83, 0.76, 0.68 | `tables/ordering_agreement_summary.csv` |
| τ at each depth (4.2, Appendix G) | e.g. MuSK 0.30 → 0.16 → 0.64 | `full_manifold_evaluation.csv`, rows `cls_0.125` … `cls_1.0`, mean of `tau` over progressions per model |
| Within-disease τ vs 5-shot F1 (4.3, H.1) | ρ = 0.94, 0.83, 0.77, 0.77; mean 0.83 | printed by `plotting/plot_generalizability_test.py`; also annotated on Fig 15 |
| Held-out rank transfer (abstract, 4.3, H.2) | ρ = 1.00, 0.89, 0.89, 0.89; mean 0.91 | same script; annotated on Figs 5 and 16 |
| Fibroblast z-scores along pseudotime (4.4) | ≈2, −0.7 at t = 0.18, 0.4 at t = 0.21 | read from the Fig 6 curve (`plotting/plot_cell_pattern_v2.py`) |
| Slides per half (B.1) | 729/730, 334/334, 343/343, 93/94 | `split_half/*/split_half.csv`, columns `slides_halfA`, `slides_halfB` |
| Simulation τ (Appendix C) | 0.819 vs 0.273 | printed by and drawn on Fig 8 (`plotting/plot_separability_vs_fidelity_simulation.py`, seed 42) |
| Silhouette of MuSK and DINOv2 on CRC-Conventional (Appendix C) | 0.062 for both | `model_comparison/cluster_metrics_and_dpt.csv`, column `silhouette` |
| Intrinsic dimension, Virchow-2 on BDC (E.1) | 12.7 → 4.0 | `full_manifold_evaluation.csv`, columns `id_raw`, `id_diff` (`final_embedding` row) |
| Spectral gap, ID reduction and trustworthiness per progression (E.2) | e.g. BDC 0.050, 69.6%, 0.977 | `appendix_metrics_table.csv` (Table 5) |
| TCGA-KIRC cohort sizes (Appendix J) | 379 patients; n = 377, 345, 371 (81 events) | `tcga_kirc/analysis/results/paper/table6_kirc_concordance.csv` (`*_n` columns); printed by `tcga_kirc/stage_kirc.py` |

## Recomputing the cached results

These need the embeddings (level B). Each row lists the stored file a script
produced. Run from `analyze_embeddings/`.

| Stored file (`results/paper/`) | Command | Used by |
|---|---|---|
| `full_manifold_evaluation.csv` | `python generate_data/evaluate_dpt.py` | Figs 2–5, 10–16; Tables 1–5 |
| `null_manifold_evaluation.csv` | `python generate_data/generate_null_distribution.py` | Fig 2 |
| `stage_permutation_specificity.csv` | `python generate_data/evaluate_stage_permutations.py` | Figs 3, 11 |
| `generalizability_results.csv` | `python generate_data/evaluate_generalizability.py` (after `evaluate_dpt.py`) | Figs 5, 15, 16 |
| `hyperparam_sweep_k_results.csv` | `python generate_data/hyper_param_sweep.py` | Fig 7 |
| `baseline_evaluation.csv` | `python generate_data/evaluate_baselines.py` | Tables 2, 3 |
| `ordering_agreement.csv`, `ordering_agreement_tau_check.csv`, `ordering_scores.npz` | `python generate_data/evaluate_ordering_agreement.py --save-scores` | Appendix C.1 |
| `split_half/scc/` | `python generate_data/evaluate_split_half.py --progression SCC --n-total 4000` | Table 1 |
| `split_half/crcconv/` | `python generate_data/evaluate_split_half.py --progression CRC-Conventional --n-total 4000` | Table 1 |
| `split_half/crcserr/` | `python generate_data/evaluate_split_half.py --progression CRC-Serrated --n-total 3000` | Table 1 |
| `split_half/bdc_n1500/` | `python generate_data/evaluate_split_half.py --progression BDC --n-total 1500` | Table 1 |
| `cell_counts/crc_conventional_<model>_2000_cellcounts.csv` | `python generate_data/analyze_trajectory_cellcounts.py --progression CRC-Conventional --model <model> --n_per_class 2000` (level C) | Figs 6, 9, 17, 18, 19 |

`python generate_data/run_all.py` runs the first five rows and the UNI-2 cell
counts in order. Add `--test` to any script for a quick run on a few patches;
test output goes to `results/test/`.

The split-half script runs every condition over 10 seeds. Table 1 reports seed
42, the seed of the main evaluation.

For the TCGA-KIRC analysis (Appendix J, Table 6) see
[tcga_kirc/README.md](tcga_kirc/README.md).
