#!/usr/bin/env python
"""
Cross-Method Ordering Agreement.

Measures whether independent ordering methods recover the SAME per-patch
trajectory, not merely the same Kendall's tau against the stage labels.

Motivation: two methods can both score tau = 0.70 while ordering patches
completely differently, because tau measures each method's agreement with the
stage labels rather than with the other method. Agreement at the level of tau
VALUES (across conditions) therefore cannot establish that the recovered
trajectory belongs to the embeddings rather than to one method's inductive
bias. Per-patch rank correlation can.

Methods compared (all produce a per-patch scalar increasing along progression):

    centroid_distance      cosine distance from the root-class centroid
    pca_projection         projection onto PC1
    knn_geodesic           shortest path from the root medoid
    diffusion_component_1  leading non-trivial diffusion coordinate
    dpt                    diffusion pseudotime

INDEPENDENCE IS NOT UNIFORM, and this determines how each pair should be read:

    centroid_distance, pca_projection   share NOTHING with DPT (no graph)
    knn_geodesic                        shares DPT's graph and root
    diffusion_component_1               shares DPT's graph, root and diffusion

High dpt-vs-knn_geodesic agreement is close to guaranteed by construction and
is weak evidence. The load-bearing comparisons are dpt vs centroid_distance and
dpt vs pca_projection.

Two agreement statistics are reported per pair:

    spearman_overall       rank correlation over all patches
    spearman_within_stage  mean of the per-stage rank correlations

The second is the stricter test. Any two methods that both track the stage
labels will correlate overall simply because both increase with stage.
Computing agreement WITHIN each stage removes that shared component and asks
whether the methods agree on the ordering where the labels give no guidance.

Usage:
    python generate_data/evaluate_ordering_agreement.py
    python generate_data/evaluate_ordering_agreement.py --progressions BDC
    python generate_data/evaluate_ordering_agreement.py --test --progressions BDC
    python generate_data/evaluate_ordering_agreement.py --save-scores
"""

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# --- Import Path Setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

import config
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import cohort_to_anndata
from analysis.baselines import (
    prepare_neighbor_graph,
    score_centroid_distance,
    score_pca_projection,
    score_diffusion_component,
    score_knn_geodesic,
    score_dpt,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

N_NEIGHBORS = config.DPT["n_neighbors"]
N_DCS = config.DPT["n_diffusion_components"]

# Methods that share no machinery with DPT, so their agreement with DPT is
# genuine cross-method evidence rather than a consequence of shared inputs.
DPT_INDEPENDENT = {"centroid_distance", "pca_projection"}


def compute_scores(adata, root_class: str) -> Dict[str, np.ndarray]:
    """
    Compute every method's per-patch score on one cohort.

    The graph and diffusion map are built once and shared: prepare_neighbor_graph
    is idempotent and the diffmap calls are guarded, so DC1, the geodesic and
    DPT reuse a single construction. DC1 is computed before DPT because
    sc.tl.dpt writes into the same AnnData.
    """
    prepare_neighbor_graph(adata, N_NEIGHBORS)

    scores = {
        "centroid_distance": score_centroid_distance(adata, root_class),
        "pca_projection": score_pca_projection(adata),
        "knn_geodesic": score_knn_geodesic(adata, root_class, N_NEIGHBORS),
        "diffusion_component_1": score_diffusion_component(
            adata, root_class, N_NEIGHBORS
        ),
    }
    scores["dpt"] = score_dpt(adata, root_class, N_NEIGHBORS, N_DCS)
    return {k: v for k, v in scores.items() if v is not None}


def within_stage_spearman(a: np.ndarray, b: np.ndarray, stages: np.ndarray) -> float:
    """
    Mean per-stage Spearman between two score vectors.

    Removes the component of agreement that follows trivially from both methods
    tracking the stage labels. Stages with fewer than 10 usable patches, or with
    no variance in either vector, are skipped.
    """
    rhos = []
    for s in np.unique(stages):
        m = stages == s
        if m.sum() < 10:
            continue
        x, y = a[m], b[m]
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        rho = spearmanr(x, y).correlation
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


def evaluate_single_condition(
    dataset: ProgressionEmbeddingDataset,
    patch_ids: Set[str],
    model: str,
    ordered_classes: List[str],
    root_class: str,
    save_scores: bool,
) -> Dict[str, Any]:
    """Return pairwise agreement rows, tau-check rows, and optionally raw scores."""
    try:
        cohort_df = dataset.get_cohort(
            patch_ids, model=model, embedding_type="final_embedding"
        )
    except Exception:
        return {"pairs": [], "tau_check": [], "scores": None}

    if cohort_df.empty:
        return {"pairs": [], "tau_check": [], "scores": None}

    adata = cohort_to_anndata(cohort_df, ordered_classes)
    stages = adata.obs["stage_int"].values
    scores = compute_scores(adata, root_class)

    # Recompute tau from each score vector. These must match the values in
    # baseline_evaluation.csv / full_manifold_evaluation.csv; the comparison is
    # done after the run and is what verifies the shared scoring path.
    tau_check = []
    for name, vec in scores.items():
        finite = np.isfinite(vec)
        tau = (
            kendalltau(vec[finite], stages[finite]).correlation
            if finite.sum() >= 10 else np.nan
        )
        tau_check.append({
            "method": name,
            "tau_recomputed": tau,
            "frac_valid": float(finite.mean()),
        })

    pairs = []
    for a, b in itertools.combinations(sorted(scores), 2):
        va, vb = scores[a], scores[b]
        both = np.isfinite(va) & np.isfinite(vb)
        if both.sum() < 10:
            continue
        pairs.append({
            "method_a": a,
            "method_b": b,
            "spearman_overall": spearmanr(va[both], vb[both]).correlation,
            "spearman_within_stage": within_stage_spearman(
                va[both], vb[both], stages[both]
            ),
            "n_patches": int(both.sum()),
            # True only when neither method shares DPT's graph with the other.
            "dpt_independent_pair": (a in DPT_INDEPENDENT) or (b in DPT_INDEPENDENT),
        })

    return {
        "pairs": pairs,
        "tau_check": tau_check,
        "scores": scores if save_scores else None,
    }


def run_full_evaluation(
    eval_config: dict = None,
    progressions: list = None,
    save_scores: bool = False,
):
    if eval_config is None:
        eval_config = config.EVALUATION

    all_pairs, all_tau, score_store = [], [], {}

    for prog_config in config.get_progressions(progressions):
        prog_name = prog_config["name"]
        ordered_classes = prog_config["classes"]

        logger.info("=" * 60)
        logger.info(f"Processing Progression: {prog_name}")
        logger.info("=" * 60)

        registry_config = RegistryConfig(
            bucket=prog_config["bucket"],
            prefix=prog_config["prefix"],
            ordered_classes=ordered_classes,
            models=config.EXPECTED_MODELS,
            progression_name=prog_name,
            scan_all_models=False,
        )
        dataset = ProgressionEmbeddingDataset(registry_config)

        patch_ids = dataset.sample_patch_ids(
            n_per_class=eval_config["n_per_class"],
            max_per_slide=eval_config["max_per_slide"],
            seed=eval_config["seed"],
        )

        for model in tqdm(config.EXPECTED_MODELS, desc="  Models"):
            try:
                dataset.load_model_into_memory(model, patch_ids)
                out = evaluate_single_condition(
                    dataset=dataset,
                    patch_ids=patch_ids,
                    model=model,
                    ordered_classes=ordered_classes,
                    root_class=prog_config["root_class"],
                    save_scores=save_scores,
                )
                for row in out["pairs"]:
                    row["progression"], row["model"] = prog_name, model
                    all_pairs.append(row)
                for row in out["tau_check"]:
                    row["progression"], row["model"] = prog_name, model
                    all_tau.append(row)
                if out["scores"] is not None:
                    for name, vec in out["scores"].items():
                        score_store[f"{prog_name}|{model}|{name}"] = vec
            except Exception as e:
                logger.error(f"Failed to process model {model}: {e}")
            finally:
                dataset.clear_cache()

    return pd.DataFrame(all_pairs), pd.DataFrame(all_tau), score_store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-patch cross-method ordering agreement."
    )
    parser.add_argument("--progressions", type=str, nargs="+", default=None)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: minimal samples, output to results/test/")
    parser.add_argument("--save-scores", action="store_true",
                        help="Also save raw per-patch score vectors as .npz")
    return parser.parse_args()


def main():
    args = parse_args()

    eval_config = config.TEST_EVALUATION if args.test else config.EVALUATION
    df_pairs, df_tau, scores = run_full_evaluation(
        eval_config=eval_config,
        progressions=args.progressions,
        save_scores=args.save_scores,
    )

    if df_pairs.empty:
        logger.warning("No results generated.")
        return

    if args.test:
        out_dir = config.RESULTS_DIR / "test"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = config.get_output_dir()

    pairs_path = out_dir / "ordering_agreement.csv"
    tau_path = out_dir / "ordering_agreement_tau_check.csv"
    df_pairs.to_csv(pairs_path, index=False)
    df_tau.to_csv(tau_path, index=False)
    logger.info(f"\nAgreement saved to {pairs_path}")
    logger.info(f"Tau cross-check saved to {tau_path}")

    if scores:
        npz_path = out_dir / "ordering_scores.npz"
        np.savez_compressed(npz_path, **scores)
        logger.info(f"Per-patch scores saved to {npz_path}")

    summary = (
        df_pairs.groupby(["method_a", "method_b"])
        [["spearman_overall", "spearman_within_stage"]]
        .mean()
        .round(3)
        .sort_values("spearman_overall", ascending=False)
    )
    print("\nMean agreement across conditions:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
