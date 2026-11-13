# Cached results used in the paper

Each file is the output of one script in `generate_data/`. Plotting scripts read
from here by default. See [REPRODUCING.md](../../../REPRODUCING.md) for the
command that produced each file and the figures and tables that use it.

| File | Contents |
|---|---|
| `full_manifold_evaluation.csv` | Trajectory fidelity (tau, with bootstrap CI for the final embedding) and manifold metrics for every progression, model and embedding type |
| `null_manifold_evaluation.csv` | Tau after shuffling class labels |
| `stage_permutation_specificity.csv` | Tau for every possible ordering of the stages |
| `generalizability_results.csv` | Few-shot F1 on each held-out progression and mean tau on the other three |
| `hyperparam_sweep_k_results.csv` | Tau, trustworthiness and spectral gap across k (CRC-Serrated) |
| `baseline_evaluation.csv` | Tau for the four alternative ordering methods |
| `ordering_agreement.csv` | Per-patch Spearman agreement between each pair of ordering methods |
| `ordering_agreement_tau_check.csv` | Tau of each method recomputed during the agreement run, as a consistency check |
| `ordering_scores.npz` | Per-patch scores of every ordering method, from which the agreement values are computed |
| `cell_counts/` | HistoPLUS cell counts and pseudotime per patch on CRC-Conventional (2,000 patches per class), one CSV per model |
| `split_half/` | Tau under empirical vs balanced class sampling, on the full cohort and two disjoint slide-level halves, over 10 seeds. `split_half.csv` has one row per model, condition and seed; `table.md` summarises the mean and range over seeds |
