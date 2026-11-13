#!/usr/bin/env python
"""
Model Comparison: Cluster Quality vs Trajectory Fidelity.

For two models on CRC-Conventional, computes:
  1. Cluster quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, KNN accuracy)
  2. UMAP embeddings colored by disease stage
  3. DPT pseudotime + Kendall's tau
  4. HistoPLUS-style standardized abundance curves (Z-score of cell-type proportions)

Produces a side-by-side comparison figure and a metrics CSV.

Usage (from analyze_embeddings/):
    # Paper Fig 9 (MuSK vs DINOv2 eosinophil curves) and the silhouette scores
    # quoted in Appendix C. Needs the embeddings.
    python plotting/plot_model_comparison.py --models musk dinov2

    # Curves only, from the stored cell-count CSVs (no embeddings needed)
    python plotting/plot_model_comparison.py --models musk dinov2 --plots_only \
        --cellcount_csvs results/paper/cell_counts/crc_conventional_musk_2000_cellcounts.csv \
                         results/paper/cell_counts/crc_conventional_dinov2_2000_cellcounts.csv

    python plotting/plot_model_comparison.py --test

Output: plots/<RESULTS_RUN>/model_comparison/ (per_celltype/eosinophils.png is Fig 9).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from scipy.stats import kendalltau
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# --- Path setup ---
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import (
    DPTConfig,
    DPTMetric,
    cohort_to_anndata,
    compute_dpt,
)
from analysis.baselines import compute_centroid_distance
from plotting.plot_config import set_icml_style, FONT_SIZES

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

PROG_NAME = "CRC-Conventional"

STAGE_COLORS = {
    "Adenoma low grade": "#2ECC71",
    "Adenoma high grade": "#F1C40F",
    "Adenocarcinoma low grade": "#E67E22",
    "Adenocarcinoma high grade": "#C0392B",
}

STAGE_ORDER = [
    "Adenoma low grade",
    "Adenoma high grade",
    "Adenocarcinoma low grade",
    "Adenocarcinoma high grade",
]

# Cell-type features for abundance plot (same as plot_cell_pattern_v2)
ABUNDANCE_FEATURES = {
    "cancer_cells_prop": ("Cancer Cells", "#D62828", "--"),
    "fibroblasts_prop": ("Fibroblasts", "#8D6E63", "-"),
    "apoptotic_bodies_prop": ("Apoptosis", "#9C27B0", "-"),
    "neutrophils_prop": ("Neutrophils", "#F4A261", "-"),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_embeddings(prog_config: dict, model_name: str, n_per_class: int,
                    max_per_slide: int, seed: int) -> pd.DataFrame:
    """Load embeddings from GCS for a single model."""
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[model_name],
        progression_name=prog_config["name"],
        scan_all_models=False,
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    patch_ids = dataset.sample_patch_ids(
        n_per_class=n_per_class, max_per_slide=max_per_slide, seed=seed,
    )
    dataset.load_model_into_memory(model_name, patch_ids)
    cohort_df = dataset.get_cohort(patch_ids, model_name, config.EVALUATION["embedding_type"])
    logger.info(f"  {model_name}: loaded {len(cohort_df)} patches")
    return cohort_df


# ── Cluster quality ───────────────────────────────────────────────────────────

def compute_cluster_metrics(X: np.ndarray, labels: np.ndarray,
                            class_names: List[str] = None) -> Dict[str, float]:
    """Compute clustering quality metrics, including per-class silhouette."""
    from sklearn.metrics import silhouette_samples

    metrics = {}
    metrics["silhouette"] = silhouette_score(X, labels, metric="cosine")

    # Per-class silhouette scores
    sample_scores = silhouette_samples(X, labels, metric="cosine")
    for label_int in sorted(np.unique(labels)):
        mask = labels == label_int
        name = class_names[label_int] if class_names else str(label_int)
        metrics[f"silhouette_{name}"] = sample_scores[mask].mean()

    metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    metrics["davies_bouldin"] = davies_bouldin_score(X, labels)

    # KNN accuracy (5-fold CV)
    knn = KNeighborsClassifier(n_neighbors=15, metric="cosine")
    scores = cross_val_score(knn, X, labels, cv=5, scoring="accuracy")
    metrics["knn_accuracy"] = scores.mean()

    return metrics


# ── DPT ───────────────────────────────────────────────────────────────────────

def compute_pseudotime(cohort_df: pd.DataFrame, prog_config: dict) -> Tuple[sc.AnnData, float]:
    """Run DPT and return annotated AnnData + tau."""
    adata = cohort_to_anndata(cohort_df, prog_config["classes"])
    dpt_cfg = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics={DPTMetric.TAU},
    )
    result = compute_dpt(adata, prog_config["root_class"], dpt_cfg)

    # Normalize pseudotime to [0, 1]
    valid = np.isfinite(adata.obs["dpt_pseudotime"])
    pt = adata.obs.loc[valid, "dpt_pseudotime"]
    adata.obs["pseudotime_norm"] = np.nan
    adata.obs.loc[valid, "pseudotime_norm"] = (pt - pt.min()) / (pt.max() - pt.min())

    return adata, result.tau


# ── Cell count CSV lookup ─────────────────────────────────────────────────────

def find_cellcount_csv(model_name: str, results_run: str = None) -> Path:
    """Look for an existing cell-count CSV in the given (or active) results run."""
    run = results_run or config.RESULTS_RUN
    results_dir = _project_root / "results" / run
    # Cell-count CSVs live in results/<run>/cell_counts/ (older runs: results/<run>/)
    candidates = sorted(results_dir.glob(f"cell_counts/*{model_name}*cellcounts*.csv")) \
        or sorted(results_dir.glob(f"*{model_name}*cellcounts*.csv"))
    if candidates:
        return candidates[-1]
    return None


# ── Plotting helpers ──────────────────────────────────────────────────────────

def get_smoothed_data(df: pd.DataFrame, columns: List[str],
                      window_fraction: float = 0.15) -> pd.DataFrame:
    """Rolling-window smoother (mirrors plot_cell_pattern_v2)."""
    window = int(len(df) * window_fraction)
    smoothed = df[columns].rolling(window=window, center=True, min_periods=window // 4).mean()
    smoothed["pseudotime"] = df["pseudotime_norm"].values
    return smoothed.dropna()


def plot_umap(ax, X: np.ndarray, classes: np.ndarray, title: str):
    """UMAP scatter colored by disease stage."""
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=42)
    emb = reducer.fit_transform(X)
    for stage in STAGE_ORDER:
        mask = classes == stage
        if not mask.any():
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=STAGE_COLORS[stage], label=stage,
            s=8, alpha=0.5, edgecolors="none",
        )
    ax.set_title(title, fontsize=FONT_SIZES["plot_title"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("UMAP-2", fontsize=FONT_SIZES["axis_label"])


def plot_abundance(ax, df: pd.DataFrame, tau: float, model_label: str,
                   xlim: Tuple[float, float] = (0, 1), method_label: str = "DPT"):
    """Standardised abundance Z-score curves (mirrors plot_cell_pattern_v2)."""
    cols = list(ABUNDANCE_FEATURES.keys())
    smooth = get_smoothed_data(df, cols)

    # Z-score
    z_df = (smooth[cols] - smooth[cols].mean()) / smooth[cols].std()
    z_df["pseudotime"] = smooth["pseudotime"]

    for col, (label, color, style) in ABUNDANCE_FEATURES.items():
        alpha = 0.5 if "cancer" in col else 1.0
        ax.plot(z_df["pseudotime"], z_df[col], label=label,
                color=color, linestyle=style, linewidth=2.0, alpha=alpha)

    # Stage boxes
    y_min, y_max = -2.5, 4.0
    box_height = 0.25
    box_y_pos = y_min + 0.1
    for stage in STAGE_ORDER:
        mask = df["class_label"] == stage
        if not mask.any():
            continue
        pts = df.loc[mask, "pseudotime_norm"]
        q25, q75 = pts.quantile(0.25), pts.quantile(0.75)
        median = pts.median()
        color = STAGE_COLORS.get(stage, "gray")
        rect = mpatches.Rectangle(
            (q25, box_y_pos), q75 - q25, box_height,
            facecolor=color, alpha=0.6, edgecolor="black", linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.plot([median, median], [box_y_pos, box_y_pos + box_height],
                color="black", linewidth=1.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Pseudotime", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(f"{model_label}  ({method_label} τ = {tau:.3f})", fontsize=FONT_SIZES["plot_title"])
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_comparison(model_data: dict, out_dir: Path):
    """
    Produce two separate PNGs:
      - umap_comparison.png: UMAP per model with silhouette score
      - abundance_comparison.png: Abundance curves per model with DPT tau
    """
    set_icml_style()
    models = list(model_data.keys())
    from plotting.plot_config import MODEL_LABELS
    labels = {m: MODEL_LABELS.get(m, m) for m in models}

    # ── UMAP figure ───────────────────────────────────────────────────────
    fig_umap, axes_umap = plt.subplots(1, 2, figsize=(14, 5))

    for col_idx, model_name in enumerate(models):
        md = model_data[model_name]
        label = labels[model_name]
        sil = md["cluster_metrics"].get("silhouette", np.nan)

        plot_umap(
            axes_umap[col_idx],
            md["X"], md["classes"],
            title=f"{label}  (Silhouette = {sil:.3f})",
        )

    # Shared UMAP legend
    stage_patches = [
        mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor="black", alpha=0.6, label=s)
        for s in STAGE_ORDER
    ]
    axes_umap[0].legend(
        handles=stage_patches, fontsize=FONT_SIZES["small"],
        loc="lower left", framealpha=0.9,
    )
    axes_umap[0].set_ylabel("UMAP-2", fontsize=FONT_SIZES["axis_label"])

    fig_umap.suptitle(
        "CRC-Conventional: Embedding Space by Disease Stage",
        fontsize=FONT_SIZES["title"], fontweight="bold", y=1.02,
    )
    fig_umap.tight_layout(w_pad=3.0)
    umap_path = out_dir / "umap_comparison.png"
    fig_umap.savefig(umap_path, dpi=300, bbox_inches="tight")
    plt.close(fig_umap)
    logger.info(f"Saved figure → {umap_path}")

    # ── Abundance figure ──────────────────────────────────────────────────
    plot_abundance_comparison(model_data, out_dir, method_label="DPT")


def plot_abundance_comparison(model_data: dict, out_dir: Path,
                              xlims: Dict[str, Tuple[float, float]] = None,
                              method_label: str = "DPT"):
    """Side-by-side abundance comparison (1 column per model)."""
    set_icml_style()
    from plotting.plot_config import MODEL_LABELS

    models = list(model_data.keys())
    labels = {m: MODEL_LABELS.get(m, m) for m in models}

    if xlims is None:
        xlims = {m: (0, 1) for m in models}

    fig_ab, axes_ab = plt.subplots(1, 2, figsize=(14, 5))

    for col_idx, model_name in enumerate(models):
        md = model_data[model_name]
        label = labels[model_name]
        xl = xlims.get(model_name, (0, 1))

        if md["cellcount_df"] is not None:
            plot_abundance(axes_ab[col_idx], md["cellcount_df"], md["tau"], label,
                           xlim=xl, method_label=method_label)
            axes_ab[col_idx].legend(fontsize=FONT_SIZES["small"], loc="upper left",
                                    ncol=2, frameon=True, columnspacing=1.0)
        else:
            axes_ab[col_idx].text(
                0.5, 0.5, "Cell-count CSV not found",
                transform=axes_ab[col_idx].transAxes,
                ha="center", va="center", fontsize=FONT_SIZES["subtitle"],
            )
            axes_ab[col_idx].set_title(
                f"{label}  ({method_label} τ = {md['tau']:.3f})",
                fontsize=FONT_SIZES["plot_title"],
            )

    axes_ab[0].set_ylabel("Standardized Abundance (Z-Score)", fontsize=FONT_SIZES["axis_label"])

    fig_ab.suptitle(
        f"CRC-Conventional: Cell-Type Abundance Along {method_label} Pseudotime",
        fontsize=FONT_SIZES["title"], fontweight="bold", y=1.02,
    )
    fig_ab.tight_layout(w_pad=3.0)
    ab_path = out_dir / "abundance_comparison.png"
    fig_ab.savefig(ab_path, dpi=300, bbox_inches="tight")
    plt.close(fig_ab)
    logger.info(f"Saved figure → {ab_path}")


ALL_CELL_FEATURES = {
    "cancer_cells_prop": ("Cancer Cells", "#D62828"),
    "lymphocytes_prop": ("Lymphocytes", "#3498DB"),
    "fibroblasts_prop": ("Fibroblasts", "#8D6E63"),
    "plasmocytes_prop": ("Plasmocytes", "#1ABC9C"),
    "eosinophils_prop": ("Eosinophils", "#E91E63"),
    "neutrophils_prop": ("Neutrophils", "#F4A261"),
    "macrophages_prop": ("Macrophages", "#9B59B6"),
    "smooth_muscle_prop": ("Smooth Muscle", "#607D8B"),
    "endothelial_prop": ("Endothelial", "#00BCD4"),
    "red_blood_cells_prop": ("Red Blood Cells", "#F44336"),
    "epithelial_normal_prop": ("Epithelial (Normal)", "#4CAF50"),
    "mitotic_figures_prop": ("Mitotic Figures", "#FF9800"),
    "apoptotic_bodies_prop": ("Apoptotic Bodies", "#9C27B0"),
}


def _compute_per_model_xlims(model_data: dict) -> Dict[str, Tuple[float, float]]:
    """
    Compute per-model x-axis limits for centroid baseline plots.
    Picks the xlim so the 95th percentile of pseudotime lands at 75% of the axis.
    """
    xlims = {}
    for model_name, md in model_data.items():
        if md["cellcount_df"] is None:
            xlims[model_name] = (0, 1)
            continue
        pt = md["cellcount_df"]["pseudotime_norm"].dropna()
        p95 = pt.quantile(0.95)
        # Scale so p95 sits at 0.75 of the axis
        x_max = p95 / 0.75 if p95 > 0 else 1.0
        xlims[model_name] = (0, x_max)
    return xlims


def plot_per_celltype(model_data: dict, out_dir: Path,
                      xlims: Dict[str, Tuple[float, float]] = None,
                      method_label: str = "DPT"):
    """
    One PNG per cell type with one row per model (stacked vertically).
    Each row has its own stage boxes from that model's pseudotime.
    Uses 0.05 smoothing.
    """
    set_icml_style()
    from plotting.plot_config import MODEL_LABELS

    models = [m for m in model_data if model_data[m]["cellcount_df"] is not None]
    if not models:
        logger.info("No cell-count data — skipping per-celltype plots")
        return

    if xlims is None:
        xlims = {m: (0, 1) for m in models}

    cols = [c for c in ALL_CELL_FEATURES
            if all(c in model_data[m]["cellcount_df"].columns for m in models)]

    celltype_dir = out_dir / "per_celltype"
    celltype_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        cell_name, cell_color = ALL_CELL_FEATURES[col]
        n_models = len(models)
        fig, axes = plt.subplots(n_models, 1, figsize=(8, 4.0 * n_models),
                                 squeeze=False)

        for row_idx, model_name in enumerate(models):
            ax = axes[row_idx, 0]
            md = model_data[model_name]
            df = md["cellcount_df"]
            label = MODEL_LABELS.get(model_name, model_name)
            tau = md["tau"]
            xl = xlims.get(model_name, (0, 1))

            smooth = get_smoothed_data(df, [col], window_fraction=0.04)
            z = (smooth[col] - smooth[col].mean()) / smooth[col].std()

            ax.plot(smooth["pseudotime"], z,
                    color=cell_color, linewidth=2.0, alpha=0.85)

            # Stage boxes from this model's pseudotime
            y_min, y_max = -3.0, 4.0
            box_height = 0.25
            box_y_pos = y_min + 0.1
            for stage in STAGE_ORDER:
                mask = df["class_label"] == stage
                if not mask.any():
                    continue
                pts = df.loc[mask, "pseudotime_norm"]
                q25, q75 = pts.quantile(0.25), pts.quantile(0.75)
                median = pts.median()
                scolor = STAGE_COLORS.get(stage, "gray")
                rect = mpatches.Rectangle(
                    (q25, box_y_pos), q75 - q25, box_height,
                    facecolor=scolor, alpha=0.6, edgecolor="black", linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.plot([median, median], [box_y_pos, box_y_pos + box_height],
                        color="black", linewidth=1.5)

            ax.set_xlim(*xl)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
            ax.set_ylabel("Z-Score", fontsize=FONT_SIZES["axis_label"])
            ax.set_title(f"{label}  ({method_label} τ = {tau:.3f})",
                         fontsize=FONT_SIZES["subtitle"])
            ax.grid(True, alpha=0.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1, 0].set_xlabel("Diffusion Pseudotime", fontsize=FONT_SIZES["axis_label"])

        # Stage legend at bottom
        stage_patches = [
            mpatches.Patch(facecolor=STAGE_COLORS[s], edgecolor="black", alpha=0.6, label=s)
            for s in STAGE_ORDER
        ]
        fig.legend(handles=stage_patches, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=len(STAGE_ORDER),
                   fontsize=FONT_SIZES["small"], frameon=False)

        fig.suptitle(cell_name, fontsize=FONT_SIZES["plot_title"],
                     fontweight="bold", y=1.01)
        fig.tight_layout(h_pad=2.0)

        safe_name = col.replace("_prop", "")
        fig.savefig(celltype_dir / f"{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"Saved {len(cols)} per-celltype plots → {celltype_dir}")


def compute_centroid_pseudotime_for_csv(
    cellcount_df: pd.DataFrame,
    model_name: str,
    prog_config: dict,
) -> Tuple[pd.DataFrame, float]:
    """
    Load embeddings for the patches in a cell-count CSV, compute centroid-distance
    pseudotime, and return the re-sorted DataFrame + tau.
    """
    # Zero-pad patch IDs to 7 digits to match GCS/registry format
    patch_ids = set(cellcount_df["patch_id"].astype(str).str.zfill(7).values)

    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[model_name],
        progression_name=prog_config["name"],
        scan_all_models=False,
    )
    dataset = ProgressionEmbeddingDataset(registry_config)
    # Skip load_model_into_memory (strict validation fails when patch IDs
    # come from a different model's sampling). Fall back to GCS directly.
    cohort_df = dataset.get_cohort(patch_ids, model_name, config.EVALUATION["embedding_type"])
    logger.info(f"  {model_name}: fetched {len(cohort_df)}/{len(patch_ids)} patches from GCS")

    # Build AnnData for the baseline computation
    adata = cohort_to_anndata(cohort_df, prog_config["classes"])

    # Compute centroid distance baseline
    baseline_result = compute_centroid_distance(adata, prog_config["root_class"])
    tau = baseline_result.tau
    logger.info(f"  {model_name} centroid-distance τ = {tau:.4f}")

    # Extract per-patch centroid distances as pseudotime
    from scipy.spatial.distance import cdist
    root_mask = adata.obs["class"] == prog_config["root_class"]
    centroid = adata.X[root_mask].mean(axis=0).reshape(1, -1)
    distances = cdist(adata.X, centroid, metric="cosine").flatten()

    # Build a mapping: patch_id -> centroid distance
    pt_df = pd.DataFrame({
        "patch_id": adata.obs["patch_id"].values,
        "centroid_pseudotime": distances,
    })

    # Standard min-max normalization
    pt_min = pt_df["centroid_pseudotime"].min()
    pt_max = pt_df["centroid_pseudotime"].max()
    pt_df["centroid_pseudotime_norm"] = (pt_df["centroid_pseudotime"] - pt_min) / (pt_max - pt_min)

    # Merge into cell-count df
    cellcount_df = cellcount_df.copy()
    cellcount_df["patch_id"] = cellcount_df["patch_id"].astype(str).str.zfill(7)
    merged = cellcount_df.merge(pt_df, on="patch_id", how="inner")

    # Replace pseudotime columns with centroid-distance pseudotime
    merged["pseudotime"] = merged["centroid_pseudotime"]
    merged["pseudotime_norm"] = merged["centroid_pseudotime_norm"]
    merged = merged.drop(columns=["centroid_pseudotime", "centroid_pseudotime_norm"])
    merged = merged.sort_values("pseudotime_norm").reset_index(drop=True)

    n_missing = len(cellcount_df) - len(merged)
    logger.info(f"  Matched {len(merged)}/{len(cellcount_df)} patches for centroid baseline")
    if n_missing > 0:
        logger.error(f"  MISSING {n_missing} patches — these patch IDs had no embeddings in GCS for {model_name}")
    return merged, tau


def save_metrics_csv(model_data: dict, out_dir: Path):
    """Save cluster metrics + tau to CSV."""
    rows = []
    for model_name, md in model_data.items():
        row = {"model": model_name, "tau": md["tau"]}
        row.update(md["cluster_metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "cluster_metrics_and_dpt.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics → {csv_path}")
    print(df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs=2, default=["musk", "virchow2"],
                        help="Two models to compare (default: musk virchow2)")
    parser.add_argument("--n_per_class", type=int, default=1000,
                        help="Patches per class (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 50 patches per class")
    parser.add_argument("--cellcount_csvs", nargs=2, default=None,
                        help="Explicit cell-count CSV paths, one per model "
                             "(in same order as --models)")
    parser.add_argument("--results_run", type=str, default=None,
                        help="Results run directory name for cell-count CSV auto-discovery "
                             "(default: config.RESULTS_RUN). Ignored if --cellcount_csvs given.")
    parser.add_argument("--cluster_metrics_only", action="store_true",
                        help="Compute cluster metrics for ALL models and ALL progressions, "
                             "save CSV, and skip DPT/plots/HistoPLUS")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip embedding loading, cluster metrics, and DPT. "
                             "Only generate abundance figures from cell-count CSVs.")
    parser.add_argument("--centroid_baseline", action="store_true",
                        help="Recompute pseudotime using centroid-distance baseline "
                             "instead of DPT. Loads embeddings for patches in the "
                             "cell-count CSVs. Output goes to a centroid/ subdir.")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    if args.test:
        args.n_per_class = 50

    max_per_slide = config.EVALUATION["max_per_slide"]

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = config.PLOTS_OUTPUT_DIR / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cluster-metrics-only mode ─────────────────────────────────────────
    if args.cluster_metrics_only:
        all_progressions = config.get_progressions()
        all_models = config.EXPECTED_MODELS
        rows = []

        for prog_config in all_progressions:
            prog_name = prog_config["name"]
            for model_name in all_models:
                logger.info(f"[{prog_name}] {model_name}...")
                try:
                    cohort_df = load_embeddings(
                        prog_config, model_name, args.n_per_class,
                        max_per_slide, args.seed,
                    )
                    X = np.vstack(cohort_df["embedding"].values)
                    class_map = {c: i for i, c in enumerate(prog_config["classes"])}
                    labels_int = np.array([class_map[c] for c in cohort_df["class"].values])
                    metrics = compute_cluster_metrics(X, labels_int, prog_config["classes"])
                    row = {"progression": prog_name, "model": model_name}
                    row.update(metrics)
                    rows.append(row)
                    logger.info(f"  {metrics}")
                except Exception as e:
                    logger.warning(f"  Failed: {e}")

        df = pd.DataFrame(rows)
        csv_path = out_dir / "all_cluster_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved → {csv_path}")
        print(df.to_string(index=False))
        return

    # ── Plots-only mode (no embedding loading / DPT) ────────────────────
    if args.plots_only:
        if not args.cellcount_csvs:
            logger.error("--plots_only requires --cellcount_csvs")
            sys.exit(1)

        model_data = {}
        for idx, model_name in enumerate(args.models):
            cc_path = Path(args.cellcount_csvs[idx])
            if not cc_path.exists():
                logger.error(f"Cell-count CSV not found: {cc_path}")
                sys.exit(1)
            cellcount_df = pd.read_csv(cc_path).sort_values("pseudotime_norm").reset_index(drop=True)

            # Read tau from the CSV if pseudotime was already computed there
            # (the CSV has pseudotime_norm from the original DPT run)
            # We recompute tau from the stored pseudotime + class labels
            class_map = {c: i for i, c in enumerate(
                config.get_progressions([PROG_NAME])[0]["classes"]
            )}
            valid = cellcount_df["pseudotime_norm"].notna()
            stages = cellcount_df.loc[valid, "class_label"].map(class_map)
            tau, _ = kendalltau(cellcount_df.loc[valid, "pseudotime_norm"], stages)
            logger.info(f"  {model_name}: DPT τ = {tau:.4f} (from cell-count CSV)")

            model_data[model_name] = {
                "X": None,
                "classes": None,
                "cluster_metrics": {},
                "tau": tau,
                "adata": None,
                "cellcount_df": cellcount_df,
            }

        plot_abundance_comparison(model_data, out_dir, method_label="DPT")
        plot_per_celltype(model_data, out_dir)

        # Also produce centroid baseline figures if requested
        if args.centroid_baseline:
            logger.info("Computing centroid-distance baseline...")
            prog_config = config.get_progressions([PROG_NAME])[0]
            centroid_dir = out_dir / "centroid"
            centroid_dir.mkdir(parents=True, exist_ok=True)

            centroid_data = {}
            for idx, model_name in enumerate(args.models):
                cc_df = model_data[model_name]["cellcount_df"]
                merged, tau = compute_centroid_pseudotime_for_csv(
                    cc_df, model_name, prog_config,
                )
                centroid_data[model_name] = {
                    "X": None,
                    "classes": None,
                    "cluster_metrics": {},
                    "tau": tau,
                    "adata": None,
                    "cellcount_df": merged,
                }

            centroid_xlims = _compute_per_model_xlims(centroid_data)
            logger.info(f"  Centroid baseline x-axis zooms: {centroid_xlims}")
            plot_abundance_comparison(centroid_data, centroid_dir,
                                      xlims=centroid_xlims, method_label="Centroid")
            plot_per_celltype(centroid_data, centroid_dir,
                              xlims=centroid_xlims, method_label="Centroid")

        return

    # ── Full comparison mode ──────────────────────────────────────────────
    prog_config = config.get_progressions([PROG_NAME])[0]

    model_data = {}

    for model_name in args.models:
        logger.info(f"Processing {model_name}...")

        # 1. Load embeddings
        cohort_df = load_embeddings(
            prog_config, model_name, args.n_per_class, max_per_slide, args.seed,
        )
        X = np.vstack(cohort_df["embedding"].values)
        classes = cohort_df["class"].values

        # 2. Cluster metrics
        class_map = {c: i for i, c in enumerate(prog_config["classes"])}
        labels_int = np.array([class_map[c] for c in classes])
        cluster_metrics = compute_cluster_metrics(X, labels_int, prog_config["classes"])
        logger.info(f"  Cluster metrics: {cluster_metrics}")

        # 3. DPT
        adata, tau = compute_pseudotime(cohort_df, prog_config)
        logger.info(f"  DPT τ = {tau:.4f}")

        # 4. Cell-count CSV (if available)
        col_idx = args.models.index(model_name)
        if args.cellcount_csvs and args.cellcount_csvs[col_idx]:
            cc_path = Path(args.cellcount_csvs[col_idx])
        else:
            cc_path = find_cellcount_csv(model_name, args.results_run)
        cellcount_df = None
        if cc_path is not None and cc_path.exists():
            logger.info(f"  Found cell counts: {cc_path}")
            cellcount_df = pd.read_csv(cc_path).sort_values("pseudotime_norm").reset_index(drop=True)
        else:
            logger.info(f"  No cell-count CSV found for {model_name}")

        model_data[model_name] = {
            "X": X,
            "classes": classes,
            "cluster_metrics": cluster_metrics,
            "tau": tau,
            "adata": adata,
            "cellcount_df": cellcount_df,
        }

    # Plot & save
    plot_comparison(model_data, out_dir)
    plot_per_celltype(model_data, out_dir)
    save_metrics_csv(model_data, out_dir)


if __name__ == "__main__":
    main()
