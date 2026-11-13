#!/usr/bin/env python
"""
Baseline Ordering Evaluation Script.

Evaluates ordering baselines across all (progression, model) combinations for
comparison against DPT. Two families:

  Naive (no neighbor graph):  centroid_distance, pca_projection
  DPT ablations (share DPT's graph):  diffusion_component_1, knn_geodesic

Uses the same cohort sampling strategy and data pipeline as evaluate_dpt.py.

Bootstrap CIs are computed for the naive baselines only. The ablations build a
kNN graph per call, which at k=100 on 4000x1536 embeddings would cost hours
across 100 iterations x 24 conditions; they report point estimates instead.

Usage:
    python generate_data/evaluate_baselines.py
    python generate_data/evaluate_baselines.py --progressions BDC SCC
    python generate_data/evaluate_baselines.py --test --progressions BDC
    python generate_data/evaluate_baselines.py --methods diffusion_component_1 knn_geodesic
    python generate_data/evaluate_baselines.py --selftest
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Set, Dict, Tuple, Any, Callable

# --- Import Path Setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import cohort_to_anndata
from analysis.baselines import (
    BaselineResult,
    compute_centroid_distance,
    compute_pca_projection,
    compute_diffusion_component,
    compute_knn_geodesic,
    prepare_neighbor_graph,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def calc_ci(values: List[float]) -> Tuple[float, float]:
    """Calculate 95% CI from bootstrap samples."""
    if not values or len(values) < 2:
        return np.nan, np.nan
    return np.percentile(values, [2.5, 97.5])


@dataclass
class BaselineSpec:
    """
    One ordering method and how it should be evaluated.

    Attributes:
        name: Method identifier written to the CSV.
        fn: Callable (adata, root_class) -> BaselineResult. Graph methods have
            n_neighbors bound at construction so the driver never varies it.
        needs_graph: Whether the method requires a kNN graph. Used to build the
            graph once per cohort and share it across methods.
        bootstrap: Whether to compute bootstrap CIs. False for graph methods.
    """
    name: str
    fn: Callable
    needs_graph: bool
    bootstrap: bool


# n_neighbors is bound here, from the same config block compute_dpt reads, so
# the ablations cannot drift away from DPT's graph construction.
_N_NEIGHBORS = config.DPT["n_neighbors"]

BASELINE_SPECS: List[BaselineSpec] = [
    BaselineSpec(
        name="centroid_distance",
        fn=lambda adata, root_class: compute_centroid_distance(adata, root_class),
        needs_graph=False,
        bootstrap=True,
    ),
    BaselineSpec(
        name="pca_projection",
        fn=lambda adata, root_class: compute_pca_projection(adata),
        needs_graph=False,
        bootstrap=True,
    ),
    BaselineSpec(
        name="diffusion_component_1",
        fn=partial(compute_diffusion_component, n_neighbors=_N_NEIGHBORS),
        needs_graph=True,
        bootstrap=False,
    ),
    BaselineSpec(
        name="knn_geodesic",
        fn=partial(compute_knn_geodesic, n_neighbors=_N_NEIGHBORS),
        needs_graph=True,
        bootstrap=False,
    ),
]

ALL_METHOD_NAMES = [s.name for s in BASELINE_SPECS]


def select_specs(method_names: List[str] = None) -> List[BaselineSpec]:
    """Filter the registry by method name, preserving registry order."""
    if not method_names:
        return list(BASELINE_SPECS)
    unknown = set(method_names) - set(ALL_METHOD_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown method(s): {sorted(unknown)}. Available: {ALL_METHOD_NAMES}"
        )
    return [s for s in BASELINE_SPECS if s.name in method_names]


# -----------------------------------------------------------------------------
# Core Evaluation
# -----------------------------------------------------------------------------

def evaluate_single_condition(
    dataset: ProgressionEmbeddingDataset,
    patch_ids: Set[str],
    model: str,
    ordered_classes: List[str],
    root_class: str,
    n_bootstrap: int,
    specs: List[BaselineSpec],
) -> List[Dict[str, Any]]:
    """
    Evaluate the selected baselines for one (model) condition.

    Bootstrap iterations form the OUTER loop, so each resampled cohort is
    fetched once and shared across every method that needs it. Since no
    bootstrapped method needs a graph, the bootstrap loop builds zero graphs.

    Returns a list of row dicts (one per baseline method).
    """
    try:
        cohort_df = dataset.get_cohort(
            patch_ids, model=model, embedding_type="final_embedding"
        )
    except Exception:
        return []

    if cohort_df.empty:
        return []

    adata = cohort_to_anndata(cohort_df, ordered_classes)

    # Build the shared graph once if any selected method needs it.
    if any(s.needs_graph for s in specs):
        try:
            prepare_neighbor_graph(adata, _N_NEIGHBORS)
        except Exception as e:
            logger.error(f"Neighbor graph construction failed: {e}")

    # --- Point estimates on the full cohort ---
    main_results = {s.name: s.fn(adata, root_class) for s in specs}

    # --- Bootstrap CIs (naive baselines only) ---
    boot_specs = [
        s for s in specs if s.bootstrap and main_results[s.name].is_valid
    ]
    boot_taus: Dict[str, List[float]] = {s.name: [] for s in boot_specs}

    if n_bootstrap > 0 and boot_specs:
        for i in range(n_bootstrap):
            boot_ids = dataset.bootstrap_patch_ids(patch_ids, seed=i)
            boot_df = dataset.get_cohort(boot_ids, model, "final_embedding")
            if boot_df.empty:
                continue
            boot_adata = cohort_to_anndata(boot_df, ordered_classes)
            for spec in boot_specs:
                boot_result = spec.fn(boot_adata, root_class)
                if boot_result.is_valid:
                    boot_taus[spec.name].append(boot_result.tau)

    # --- Assemble rows ---
    rows = []
    for spec in specs:
        result = main_results[spec.name]
        row = {
            "method": spec.name,
            "embedding_type": "final_embedding",
            "n_samples": len(patch_ids),
            "bootstrap_iters": n_bootstrap if spec.bootstrap else 0,
            "tau": result.tau,
            "p_value": result.p_value,
            "frac_valid": result.frac_valid,
        }

        if boot_taus.get(spec.name):
            lower, upper = calc_ci(boot_taus[spec.name])
            row["tau_ci_lower"] = lower
            row["tau_ci_upper"] = upper

        rows.append(row)

    return rows


def run_full_evaluation(
    eval_config: dict = None,
    progressions: list = None,
    methods: list = None,
) -> pd.DataFrame:

    if eval_config is None:
        eval_config = config.EVALUATION

    specs = select_specs(methods)
    logger.info(f"Methods: {[s.name for s in specs]}")

    n_bootstraps = eval_config["n_bootstrap_ci"]
    n_per_class = eval_config["n_per_class"]
    max_per_slide = eval_config["max_per_slide"]

    all_results = []

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
            n_per_class=n_per_class,
            max_per_slide=max_per_slide,
            seed=eval_config["seed"],
        )

        for model in tqdm(config.EXPECTED_MODELS, desc="  Models"):
            try:
                dataset.load_model_into_memory(model, patch_ids)

                baseline_rows = evaluate_single_condition(
                    dataset=dataset,
                    patch_ids=patch_ids,
                    model=model,
                    ordered_classes=ordered_classes,
                    root_class=prog_config["root_class"],
                    n_bootstrap=n_bootstraps,
                    specs=specs,
                )

                for row in baseline_rows:
                    row["progression"] = prog_name
                    row["model"] = model
                    all_results.append(row)

            except Exception as e:
                logger.error(f"Failed to process model {model}: {e}")
            finally:
                dataset.clear_cache()

    return pd.DataFrame(all_results)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _synthetic_adata(n_per_class: int = 50, reverse: bool = False) -> Any:
    """
    Build an AnnData whose embeddings lie on a known 1-D arc.

    Points are placed along a quarter circle in the first two dimensions, so
    cosine distance increases monotonically with progression. A correct
    ordering method must recover tau close to 1.

    The arc is continuous, not four separated clumps: with discrete clumps the
    kNN graph fragments into one component per class and the diffusion map is
    degenerate. Classes are quartiles of a continuous position.

    Args:
        n_per_class: Samples per stage.
        reverse: If True, reverse only the stage LABELS while the root stays at
            the geometric start of the arc. A root-anchored method must then
            return NEGATIVE tau; a method that fits its sign to the labels
            cannot. (Reversing the root as well would cancel out and test
            nothing.)
    """
    import scanpy as sc

    rng = np.random.default_rng(0)
    classes = ["s0", "s1", "s2", "s3"]
    n = n_per_class * len(classes)

    # Continuous position along the arc, so the kNN graph stays connected
    t = np.linspace(0.0, 1.0, n)
    theta = t * (np.pi / 2)

    X = rng.normal(0, 0.01, (n, 16))
    X[:, 0] = np.cos(theta)
    X[:, 1] = np.sin(theta)

    # Classes are quartiles of the continuous position
    labels = np.array(classes)[np.minimum((t * len(classes)).astype(int),
                                          len(classes) - 1)]
    obs = pd.DataFrame({
        "patch_id": [f"p{i}" for i in range(n)],
        "class": labels,
        "slide_id": ["sl0"] * n,
    })

    # Root is always the geometric start of the arc; only labels may reverse.
    ordered = list(reversed(classes)) if reverse else classes
    class_map = {c: i for i, c in enumerate(ordered)}
    obs["stage_int"] = obs["class"].map(class_map)

    adata = sc.AnnData(X=X, obs=obs)
    return adata, classes[0]


def run_selftest() -> bool:
    """
    Verify the ablations on synthetic data with a known ordering.

    Checks:
      1. Both methods recover tau close to 1 on a monotone arc.
      2. DC1 returns NEGATIVE tau when the labels are reversed, confirming the
         root-anchored sign is not silently floored at 0 the way the
         label-fitted PCA baseline is.
    """
    ok = True
    k = 15

    adata, root = _synthetic_adata()
    dc1 = compute_diffusion_component(adata, root, n_neighbors=k)
    geo = compute_knn_geodesic(adata, root, n_neighbors=k)

    print(f"  DC1 forward       tau = {dc1.tau:.4f}  (expect > 0.8)")
    print(f"  Geodesic forward  tau = {geo.tau:.4f}  (expect > 0.8), "
          f"frac_valid = {geo.frac_valid:.3f}")

    if not (dc1.is_valid and dc1.tau > 0.8):
        print("  FAIL: DC1 did not recover the known ordering.")
        ok = False
    if not (geo.is_valid and geo.tau > 0.8):
        print("  FAIL: geodesic did not recover the known ordering.")
        ok = False
    if geo.frac_valid < 1.0:
        print(f"  WARN: synthetic graph disconnected ({geo.frac_valid:.1%}).")

    adata_rev, root_rev = _synthetic_adata(reverse=True)
    dc1_rev = compute_diffusion_component(adata_rev, root_rev, n_neighbors=k)
    print(f"  DC1 reversed      tau = {dc1_rev.tau:.4f}  (expect < -0.8)")

    if not (dc1_rev.is_valid and dc1_rev.tau < -0.8):
        print("  FAIL: DC1 sign is not root-anchored (tau should go negative).")
        ok = False

    print("  SELFTEST PASSED" if ok else "  SELFTEST FAILED")
    return ok


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ordering baselines (naive baselines and DPT ablations)."
    )
    parser.add_argument(
        "--progressions", type=str, nargs="+", default=None,
        help="Progression names to include (default: all). Choices: SCC, CRC-Conventional, CRC-Serrated, BDC",
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help=f"Methods to evaluate (default: all). Choices: {ALL_METHOD_NAMES}",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: minimal samples, output to results/test/",
    )
    parser.add_argument(
        "--selftest", action="store_true",
        help="Run synthetic correctness checks on the ablations and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.selftest:
        print("Running synthetic self-test...")
        sys.exit(0 if run_selftest() else 1)

    if args.test:
        df = run_full_evaluation(
            eval_config=config.TEST_EVALUATION,
            progressions=args.progressions,
            methods=args.methods,
        )
    else:
        df = run_full_evaluation(
            progressions=args.progressions,
            methods=args.methods,
        )

    if not df.empty:
        if args.test:
            output_dir = config.RESULTS_DIR / "test"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "baseline_evaluation.csv"
        else:
            output_dir = config.get_output_dir()
            output_path = output_dir / "baseline_evaluation.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"\nBaseline Evaluation Complete. Saved to {output_path}")
        if args.methods:
            logger.warning(
                f"Only {args.methods} were computed. This CSV does not contain "
                f"the other methods -- merge with a prior run before plotting."
            )

        summary_cols = ["progression", "model", "method", "tau"]
        summary = df[summary_cols].sort_values(
            ["progression", "model", "method"], ascending=True
        )
        print("\nSummary:")
        print(summary.to_markdown(index=False, floatfmt=".3f"))
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    main()
