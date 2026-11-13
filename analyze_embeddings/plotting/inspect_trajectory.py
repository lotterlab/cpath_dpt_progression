#!/usr/bin/env python
"""
Trajectory Inspector with HistoPLUS Cell Counting.

Loads a model/progression, computes Diffusion Pseudotime (DPT),
samples patches along the manifold, runs HistoPLUS to get cell COUNTS
(not pixel fractions), and generates visualization figures.

Usage:
    python plotting/inspect_trajectory.py --progression SCC --model uni2
    python plotting/inspect_trajectory.py --progression CRC-Conventional --model conch --seed 42
    python plotting/inspect_trajectory.py --progression BDC --model gigapath --num_samples 10
"""

import sys
import os
import argparse
from pathlib import Path
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import textwrap

import numpy as np
import pandas as pd
import torch

# Plotting & Image Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from PIL import Image
from skimage.segmentation import find_boundaries

# Ensure we can import from project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import config
except ImportError:
    print("❌ Critical Error: Could not import 'config.py'.")
    sys.exit(1)

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

# HistoPLUS imports
try:
    from histoplus.helpers.segmentor import CellViTSegmentor
    HISTOPLUS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: HistoPLUS not found. Install with: pip install histoplus")
    HISTOPLUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_NUM_SAMPLES = 8
DEFAULT_JITTER = 0.05

# Output directory for trajectory plots
TRAJECTORY_PLOTS_DIR = config.PLOTS_OUTPUT_DIR / "trajectories"

# HistoPLUS 20x model parameters
MPP_20X = 0.5
INFERENCE_IMAGE_SIZE = 224

# Local image directories (mapping bucket names to local paths)
LOCAL_IMAGE_DIRS = config.LOCAL_IMAGE_DIRS  # set SPIDER_IMAGE_ROOT or edit config.py

# HistoPLUS class mapping (from model output names to readable names)
HISTOPLUS_TO_COLUMN = {
    "Cancer cell": "cancer_cells",
    "Lymphocytes": "lymphocytes",
    "Fibroblasts": "fibroblasts",
    "Plasmocytes": "plasmocytes",
    "Eosinophils": "eosinophils",
    "Neutrophils": "neutrophils",
    "Macrophages": "macrophages",
    "Muscle Cell": "smooth_muscle",
    "Endothelial Cell": "endothelial",
    "Red blood cell": "red_blood_cells",
    "Epithelial": "epithelial_normal",
    "Apoptotic Body": "apoptotic_bodies",
    "Mitotic Figures": "mitotic_figures",
    "Minor Stromal Cell": "minor_stromal",
}

# Reverse mapping for display
COLUMN_TO_DISPLAY = {v: k for k, v in HISTOPLUS_TO_COLUMN.items()}

# Cell types for plotting
CELL_TYPE_COLUMNS = list(HISTOPLUS_TO_COLUMN.values())

# Key cell types for the summary figure
KEY_CELL_TYPES = [
    "cancer_cells",
    "epithelial_normal",
    "fibroblasts",
    "lymphocytes",
    "neutrophils",
    "apoptotic_bodies",
]

# Rare / small classes drawn with higher opacity in the overlay so they are visible
EMPHASIZED_CELL_TYPES = {"neutrophils", "apoptotic_bodies", "mitotic_figures"}
OVERLAY_ALPHA = 0.65
OVERLAY_ALPHA_EMPHASIZED = 0.9
# The tissue under the masks is toned down so the mask colors stand out: partly desaturated,
# then faded toward white (0 = unchanged, 1 = fully grey / fully white).
OVERLAY_BACKGROUND_DESATURATE = 0.6
OVERLAY_BACKGROUND_FADE = 0.5
# Each mask gets a thin outline in a darker shade of its own color
OVERLAY_OUTLINE_DARKEN = 0.55

# Whether to plot proportions (fraction of total cells) or raw counts
PLOT_PROPORTIONS = True

# Color palette for visualization (RGB tuples, 0-1 range)
CELL_COLORS = {
    "cancer_cells": (0.9, 0.1, 0.1),         # Red
    "epithelial_normal": (0.0, 0.8, 0.8),    # Cyan
    "fibroblasts": (0.2, 0.8, 0.2),          # Green
    "lymphocytes": (0.1, 0.1, 0.9),          # Blue
    "macrophages": (0.6, 0.1, 0.6),          # Purple
    "neutrophils": (1.0, 0.5, 0.0),          # Orange
    "eosinophils": (1.0, 0.2, 0.8),          # Magenta
    "plasmocytes": (0.4, 0.4, 0.8),          # Light purple
    "smooth_muscle": (0.6, 0.4, 0.2),        # Brown
    "endothelial": (0.2, 0.6, 0.4),          # Teal
    "red_blood_cells": (0.8, 0.2, 0.2),      # Dark red
    "mitotic_figures": (1.0, 1.0, 0.0),      # Yellow
    "apoptotic_bodies": (0.0, 0.0, 0.0),     # Black
    "minor_stromal": (0.5, 0.5, 0.5),        # Grey
}

# HistoPLUS class index to column name (for overlay visualization).
# Fallback only: HistoPLUSCellCounter.__init__ rebuilds this from the loaded
# model's own class_mapping so the overlay colors always match the model.
HISTOPLUS_CLASS_IDX = {
    1: "cancer_cells",
    2: "lymphocytes",
    3: "fibroblasts",
    4: "plasmocytes",
    5: "eosinophils",
    6: "neutrophils",
    7: "macrophages",
    8: "smooth_muscle",
    9: "endothelial",
    10: "red_blood_cells",
    11: "epithelial_normal",
    12: "mitotic_figures",
    13: "apoptotic_bodies",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CellCounts:
    """Container for cell counts from a single patch."""
    total_cells: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    mean_confidence: float = 0.0
    type_map: Optional[np.ndarray] = None  # For visualization

    def __post_init__(self):
        if not self.counts:
            self.counts = {col: 0 for col in CELL_TYPE_COLUMNS}

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "total_cells": self.total_cells,
            "mean_confidence": self.mean_confidence,
        }
        for col in CELL_TYPE_COLUMNS:
            result[col] = self.counts.get(col, 0)
        return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_hf_token():
    """Load HuggingFace token from environment or file."""
    token = os.getenv("HF_TOKEN")
    if not token:
        token_path = Path("~/.hf_token").expanduser()
        if token_path.exists():
            token = token_path.read_text().strip()

    if token:
        os.environ["HF_TOKEN"] = token
        logger.info("✓ HuggingFace token loaded")
    else:
        logger.warning("⚠️ No HF_TOKEN found. HistoPLUS download may fail.")


def get_progression_config(progression_name: str) -> dict:
    """Get progression config by name."""
    for prog in config.PROGRESSIONS:
        if prog["name"] == progression_name:
            return prog

    available = [p["name"] for p in config.PROGRESSIONS]
    raise ValueError(f"Progression '{progression_name}' not found. Available: {available}")


def parse_args():
    available_progressions = [p["name"] for p in config.PROGRESSIONS]

    parser = argparse.ArgumentParser(
        description="Inspect DPT trajectory with HistoPLUS cellular analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available progressions: {', '.join(available_progressions)}
Available models: {', '.join(config.EXPECTED_MODELS)}

Examples:
    python plotting/inspect_trajectory.py --progression SCC --model uni2
    python plotting/inspect_trajectory.py --progression CRC-Conventional --model conch --seed 42
        """
    )
    parser.add_argument(
        "--progression",
        type=str,
        required=True,
        choices=available_progressions,
        help="Disease progression to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=config.EXPECTED_MODELS,
        help="Foundation model to use for pseudotime"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Patches to sample along trajectory (default: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)"
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_JITTER,
        help=f"Randomness range around each quantile point (default: {DEFAULT_JITTER})"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override local image directory"
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Plot raw cell counts instead of proportions (default: proportions)"
    )
    return parser.parse_args()


# =============================================================================
# SAMPLING FUNCTIONS
# =============================================================================

def sample_patches_quantile(
    trajectory_df: pd.DataFrame,
    num_samples: int,
    rng: np.random.Generator,
    jitter: float = 0.05
) -> pd.DataFrame:
    """
    Sample patches at evenly-spaced RANKS (percentiles) with optional jitter.

    This adapts to the actual distribution of data - if most patches cluster
    in early pseudotime, we'll still sample across all stages proportionally.
    """
    n_total = len(trajectory_df)
    target_ranks = np.linspace(0, n_total - 1, num_samples).astype(int)

    selected_ilocs = []
    for rank in target_ranks:
        if jitter > 0:
            max_offset = int(n_total * jitter)
            if max_offset > 0:
                offset = rng.integers(-max_offset, max_offset + 1)
                rank_jittered = np.clip(rank + offset, 0, n_total - 1)
            else:
                rank_jittered = rank
        else:
            rank_jittered = rank

        selected_ilocs.append(rank_jittered)

    # Remove duplicates while preserving order
    seen = set()
    unique_ilocs = [x for x in selected_ilocs if not (x in seen or seen.add(x))]

    return trajectory_df.iloc[unique_ilocs].copy()


# =============================================================================
# HISTOPLUS INFERENCE
# =============================================================================

class HistoPLUSCellCounter:
    """
    HistoPLUS wrapper that returns cell COUNTS (not pixel fractions).

    Uses the 20x model with proper postprocessing for instance-level
    cell detections with types and confidence scores.
    """

    def __init__(self, device: Optional[str] = None):
        if not HISTOPLUS_AVAILABLE:
            raise RuntimeError("HistoPLUS not installed. Install with: pip install histoplus")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Initializing HistoPLUS (20x model) on {self.device}...")

        self.segmentor = CellViTSegmentor.from_histoplus(
            mpp=MPP_20X,
            mixed_precision=(self.device == "cuda"),
            inference_image_size=INFERENCE_IMAGE_SIZE,
        )

        if hasattr(self.segmentor, 'model'):
            self.segmentor.model = self.segmentor.model.to(self.device)
            self.segmentor.model.eval()

        self.postprocess_fn = self.segmentor.get_postprocess_fn()
        self.class_mapping = self.segmentor.class_mapping
        logger.info(f"   Cell types: {list(self.class_mapping.values())}")
        self._sync_class_idx()

    def _sync_class_idx(self):
        """Rebuild HISTOPLUS_CLASS_IDX from the model's class_mapping (idx -> name
        or name -> idx), so overlay colors use the model's actual label indices."""
        global HISTOPLUS_CLASS_IDX
        items = list(self.class_mapping.items())
        if not items:
            return
        if isinstance(items[0][0], str):           # name -> idx
            items = [(idx, name) for name, idx in items]
        rebuilt = {}
        for idx, name in items:
            col = HISTOPLUS_TO_COLUMN.get(name)
            if col is not None and int(idx) > 0:
                rebuilt[int(idx)] = col
        if rebuilt:
            HISTOPLUS_CLASS_IDX = rebuilt
            logger.info(f"   Overlay class indices synced from model ({len(rebuilt)} classes)")

        # Normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        logger.info("✓ HistoPLUS ready")

    def preprocess(self, img_pil: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input."""
        if img_pil.size != (INFERENCE_IMAGE_SIZE, INFERENCE_IMAGE_SIZE):
            img_pil = img_pil.resize(
                (INFERENCE_IMAGE_SIZE, INFERENCE_IMAGE_SIZE),
                Image.Resampling.LANCZOS
            )

        img_arr = np.array(img_pil)
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    @torch.no_grad()
    def count_cells(self, img_pil: Image.Image) -> CellCounts:
        """
        Run inference and return cell COUNTS.

        Also returns the type_map for overlay visualization.
        """
        # Preprocess
        img_tensor = self.preprocess(img_pil)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Run model forward pass
        outputs = self.segmentor.model(img_tensor)

        # Get type map for visualization
        type_logits = outputs["tp"]
        type_map = torch.argmax(type_logits, dim=1).cpu().numpy()[0]

        # Process for cell counting
        np_pred = torch.softmax(outputs["np"], dim=1)[:, 1]
        tp_pred = torch.argmax(outputs["tp"], dim=1, keepdim=False)
        hv_pred = outputs["hv"]

        outputs_np = {
            "np": np_pred.cpu().numpy(),
            "tp": tp_pred.cpu().numpy(),
            "hv": hv_pred.cpu().numpy(),
        }

        # Run postprocessing to get individual cell instances
        tile_predictions = self.postprocess_fn(outputs_np)

        if not tile_predictions:
            return CellCounts(type_map=type_map)

        tile_pred = tile_predictions[0]

        # Count cells by type
        counts = CellCounts(type_map=type_map)
        counts.total_cells = len(tile_pred.cell_types)

        if counts.total_cells == 0:
            return counts

        counts.counts = {col: 0 for col in CELL_TYPE_COLUMNS}

        confidences = []
        for cell_type, confidence in zip(tile_pred.cell_types, tile_pred.cell_type_probabilities):
            col_name = HISTOPLUS_TO_COLUMN.get(cell_type)
            if col_name:
                counts.counts[col_name] += 1
            confidences.append(confidence)

        counts.mean_confidence = np.mean(confidences) if confidences else 0.0

        return counts


# =============================================================================
# TRAJECTORY COMPUTATION
# =============================================================================

def compute_trajectory(
    prog_config: dict,
    model_name: str,
    num_samples: int,
    rng: np.random.Generator,
    jitter: float
) -> pd.DataFrame:
    """Compute DPT trajectory and select patches using quantile sampling."""
    registry_config = RegistryConfig(
        bucket=prog_config["bucket"],
        prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"],
        models=[model_name],
        progression_name=prog_config["name"],
        scan_all_models=False
    )
    dataset = ProgressionEmbeddingDataset(registry_config)

    try:
        patch_ids = dataset.sample_patch_ids(
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
            seed=config.EVALUATION["seed"]
        )

        dataset.load_model_into_memory(model_name, patch_ids)
        cohort_df = dataset.get_cohort(patch_ids, model_name, config.FINAL_EMBEDDING)

        if cohort_df.empty:
            raise ValueError(f"No data found for {model_name}")

        adata = cohort_to_anndata(cohort_df, prog_config["classes"])

        dpt_conf = DPTConfig(
            n_neighbors=config.DPT["n_neighbors"],
            n_diffusion_components=config.DPT["n_diffusion_components"],
            metrics={DPTMetric.TAU},
            subsample_size=2000
        )

        logger.info("⚙️  Computing Diffusion Pseudotime...")
        result = compute_dpt(adata, prog_config["root_class"], dpt_conf)
        logger.info(f"   Trajectory Fidelity: τ = {result.tau:.3f}")

        if "dpt_pseudotime" not in adata.obs:
            raise ValueError("DPT computation failed")

        trajectory_df = adata.obs.copy()
        trajectory_df = trajectory_df[np.isfinite(trajectory_df["dpt_pseudotime"])]

        pt_min = trajectory_df["dpt_pseudotime"].min()
        pt_max = trajectory_df["dpt_pseudotime"].max()
        trajectory_df["dpt_pseudotime_norm"] = (
            (trajectory_df["dpt_pseudotime"] - pt_min) / (pt_max - pt_min)
        )

        trajectory_df = trajectory_df.sort_values("dpt_pseudotime_norm")

        # Use quantile sampling with jitter
        selected_df = sample_patches_quantile(trajectory_df, num_samples, rng, jitter)
        selected_df['tau'] = result.tau

        return selected_df

    finally:
        dataset.clear_cache()


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_overlay(img_pil: Image.Image, type_map: np.ndarray, alpha: float = OVERLAY_ALPHA) -> Image.Image:
    """Create a visualization overlay showing cell type predictions.

    The tissue is toned down (partly desaturated and faded toward white) so the masks are
    what the eye sees first. Every predicted class is filled with its CELL_COLORS color and
    outlined in a darker shade of it. Rare classes in EMPHASIZED_CELL_TYPES are blended at
    OVERLAY_ALPHA_EMPHASIZED so small objects (neutrophils, apoptotic bodies) stay visible."""
    img_arr = np.array(img_pil).astype(np.float32)

    # Resize type_map if needed
    if type_map.shape != (img_pil.size[1], img_pil.size[0]):
        from skimage.transform import resize
        type_map = resize(type_map, (img_pil.size[1], img_pil.size[0]),
                          order=0, preserve_range=True).astype(int)

    # Toned-down tissue
    grey = (img_arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    base = (1 - OVERLAY_BACKGROUND_DESATURATE) * img_arr + OVERLAY_BACKGROUND_DESATURATE * grey
    base = (1 - OVERLAY_BACKGROUND_FADE) * base + OVERLAY_BACKGROUND_FADE * 255.0

    label_rgb = np.zeros_like(img_arr)
    alpha_map = np.full(type_map.shape, alpha, dtype=np.float32)

    for class_idx, col_name in HISTOPLUS_CLASS_IDX.items():
        mask = (type_map == class_idx)
        if not np.any(mask):
            continue

        color = CELL_COLORS.get(col_name, (0.5, 0.5, 0.5))
        for c in range(3):
            label_rgb[mask, c] = color[c] * 255
        if col_name in EMPHASIZED_CELL_TYPES:
            alpha_map[mask] = OVERLAY_ALPHA_EMPHASIZED

    foreground_mask = (type_map > 0)
    overlay = base.copy()

    if np.any(foreground_mask):
        a = alpha_map[foreground_mask][:, None]
        overlay[foreground_mask] = (
            (1 - a) * base[foreground_mask] +
            a * label_rgb[foreground_mask]
        )

    # Outline each mask, just inside its edge, in a darker shade of its own color
    outline = find_boundaries(type_map, mode='inner') & foreground_mask
    overlay[outline] = label_rgb[outline] * OVERLAY_OUTLINE_DARKEN

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def classes_in_type_map(type_map: Optional[np.ndarray]) -> set:
    """Column names of all cell classes that have at least one pixel in the type map."""
    if type_map is None:
        return set()
    present = set(np.unique(type_map).tolist())
    return {col for idx, col in HISTOPLUS_CLASS_IDX.items() if idx in present}


def add_overlay_color_key(ax, patches_data: list, title: str = "Segmentation Overlay\nCell Type Key"):
    """Draw the key for the segmentation overlays: one swatch per cell class that appears
    in any overlay, in the same CELL_COLORS the masks are filled with."""
    present = set()
    for pdata in patches_data:
        present |= pdata.get('present_classes', set())
    # Stable order: KEY types first, then the rest in HistoPLUS order
    ordered = [ct for ct in KEY_CELL_TYPES if ct in present] + \
              [ct for ct in CELL_TYPE_COLUMNS if ct in present and ct not in KEY_CELL_TYPES]
    handles = [mpatches.Patch(facecolor=CELL_COLORS.get(ct, (0.5, 0.5, 0.5)), edgecolor='#444444',
                              linewidth=0.5, label=COLUMN_TO_DISPLAY.get(ct, ct)) for ct in ordered]
    ax.axis('off')
    if not handles:
        return
    legend = ax.legend(handles=handles, loc='center left', bbox_to_anchor=(0.0, 0.5), ncol=1,
                       frameon=True, edgecolor='#bbbbbb', fancybox=False, borderpad=0.9,
                       fontsize=12, title=title, title_fontsize=12,
                       handlelength=1.5, handleheight=1.1, labelspacing=0.55)
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_multialignment('center')


def _image_grid(fig, n: int, trend_height: float):
    """GridSpec shared by both figures. Rows: original patches | segmentation overlays |
    trend plot. Columns: one per patch, plus a last column for the overlay color key."""
    return GridSpec(3, n + 1, figure=fig, height_ratios=[1, 1, trend_height],
                    width_ratios=[1] * n + [1.15], hspace=0.15, wspace=0.08)


def _row_label(ax, text: str):
    """Rotated label to the left of the first image in a row."""
    ax.text(-0.07, 0.5, text, transform=ax.transAxes, rotation=90, ha='center', va='center',
            fontsize=10, fontweight='bold')


def _stage_label(stage_name: str, width: int = 18) -> str:
    return textwrap.fill(stage_name, width=width, break_on_hyphens=False)


def generate_figure_key_classes(
    patches_data: list,
    model_name: str,
    progression_name: str,
    tau: float,
    seed: int,
    output_path: Path,
    use_proportions: bool = PLOT_PROPORTIONS
):
    """Generate figure with key cell types (counts or proportions)."""
    n = len(patches_data)

    fig = plt.figure(figsize=(n * 2.5 + 2.9, 9.4))
    gs = _image_grid(fig, n, trend_height=0.9)

    pseudotimes = []
    trends = {ct: [] for ct in KEY_CELL_TYPES}

    for i, pdata in enumerate(patches_data):
        pt = pdata['pt']
        pseudotimes.append(pt)

        # Row 1: Original Image
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(pdata['orig_img'])
        ax_orig.axis('off')
        ax_orig.set_title(f"PT: {pt:.2f}\n{_stage_label(pdata['stage'])}", fontsize=9, fontweight='bold')

        # Row 2: Segmentation Overlay
        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(pdata['overlay_img'])
        ax_seg.axis('off')
        if i == 0:
            _row_label(ax_orig, "H&E patch")
            _row_label(ax_seg, "Segmentation overlay")

        # Get values (proportion or count)
        total = pdata['total_cells']
        for ct in KEY_CELL_TYPES:
            count = pdata['counts'].get(ct, 0)
            if use_proportions and total > 0:
                value = (count / total) * 100  # percentage
            else:
                value = count
            trends[ct].append(value)

    # Right of the image rows: color key for the segmentation overlays
    add_overlay_color_key(fig.add_subplot(gs[0:2, n]), patches_data)

    # Row 3: Trend Plot (always shows every KEY cell type; flat zero = not detected)
    ax_trend = fig.add_subplot(gs[2, :n])

    for ct in KEY_CELL_TYPES:
        color = CELL_COLORS.get(ct, (0.5, 0.5, 0.5))
        y_vals = trends[ct]

        display_name = COLUMN_TO_DISPLAY.get(ct, ct.replace("_", " ").title())
        if max(y_vals) <= 0:
            display_name += " (not detected)"

        ax_trend.plot(
            pseudotimes, y_vals,
            marker='o', markersize=6,
            linewidth=2,
            label=display_name,
            color=color,
            alpha=0.85
        )

    ax_trend.set_xlabel("Diffusion Pseudotime (Normalized)", fontsize=11)
    
    if use_proportions:
        ax_trend.set_ylabel("% of Total Cells", fontsize=11)
    else:
        ax_trend.set_ylabel("Cell Count", fontsize=11)
    
    ax_trend.set_xlim(-0.05, 1.05)
    ax_trend.set_ylim(bottom=0)
    
    if use_proportions:
        ax_trend.set_ylim(0, 105)  # Cap at 105% for clean display
    
    ax_trend.grid(True, linestyle='--', alpha=0.3)
    ax_trend.spines['top'].set_visible(False)
    ax_trend.spines['right'].set_visible(False)

    ax_trend.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=min(len(KEY_CELL_TYPES), 6),
        frameon=False,
        fontsize=12
    )

    seed_str = f"seed={seed}" if seed is not None else "random"
    metric_type = "Cell Proportions" if use_proportions else "Cell Counts"
    fig.suptitle(
        f"{metric_type} Along Trajectory: {progression_name}\n"
        f"Model: {model_name} (τ = {tau:.2f}) | {seed_str}",
        fontsize=12,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"✅ Key classes figure saved to {output_path}")


def generate_figure_all_classes(
    patches_data: list,
    model_name: str,
    progression_name: str,
    tau: float,
    seed: int,
    output_path: Path,
    use_proportions: bool = PLOT_PROPORTIONS
):
    """Generate figure with ALL cell types plotted (counts or proportions)."""
    n = len(patches_data)

    fig = plt.figure(figsize=(n * 2.5 + 2.9, 11.4))
    gs = _image_grid(fig, n, trend_height=1.2)

    pseudotimes = []
    trends = {ct: [] for ct in CELL_TYPE_COLUMNS}

    for i, pdata in enumerate(patches_data):
        pt = pdata['pt']
        pseudotimes.append(pt)

        # Row 1: Original Image
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(pdata['orig_img'])
        ax_orig.axis('off')
        ax_orig.set_title(f"PT: {pt:.2f}\n{_stage_label(pdata['stage'])}", fontsize=9, fontweight='bold')

        # Row 2: Segmentation Overlay
        ax_seg = fig.add_subplot(gs[1, i])
        ax_seg.imshow(pdata['overlay_img'])
        ax_seg.axis('off')
        if i == 0:
            _row_label(ax_orig, "H&E patch")
            _row_label(ax_seg, "Segmentation overlay")

        # Get values (proportion or count)
        total = pdata['total_cells']
        for ct in CELL_TYPE_COLUMNS:
            count = pdata['counts'].get(ct, 0)
            if use_proportions and total > 0:
                value = (count / total) * 100  # percentage
            else:
                value = count
            trends[ct].append(value)

    # Right of the image rows: color key for the segmentation overlays
    add_overlay_color_key(fig.add_subplot(gs[0:2, n]), patches_data)

    # Row 3: Trend Plot with ALL classes
    ax_trend = fig.add_subplot(gs[2, :n])

    # Sort classes by max value for cleaner legend ordering
    class_max_vals = [(ct, max(trends[ct])) for ct in CELL_TYPE_COLUMNS]
    sorted_classes = [ct for ct, _ in sorted(class_max_vals, key=lambda x: -x[1])]

    # Threshold for plotting (1% for proportions, 1 cell for counts)
    threshold = 1.0 if use_proportions else 1

    for ct in sorted_classes:
        color = CELL_COLORS.get(ct, (0.5, 0.5, 0.5))
        y_vals = trends[ct]

        display_name = COLUMN_TO_DISPLAY.get(ct, ct.replace("_", " ").title())

        if max(y_vals) >= threshold:
            if use_proportions:
                label = f"{display_name} (max: {max(y_vals):.1f}%)"
            else:
                label = f"{display_name} (max: {int(max(y_vals))})"
            
            ax_trend.plot(
                pseudotimes, y_vals,
                marker='o', markersize=5,
                linewidth=1.5,
                label=label,
                color=color,
                alpha=0.8
            )

    ax_trend.set_xlabel("Diffusion Pseudotime (Normalized)", fontsize=11)
    
    if use_proportions:
        ax_trend.set_ylabel("% of Total Cells", fontsize=11)
    else:
        ax_trend.set_ylabel("Cell Count", fontsize=11)
    
    ax_trend.set_xlim(-0.05, 1.05)
    ax_trend.set_ylim(bottom=0)
    
    if use_proportions:
        ax_trend.set_ylim(0, 105)
    
    ax_trend.grid(True, linestyle='--', alpha=0.3)
    ax_trend.spines['top'].set_visible(False)
    ax_trend.spines['right'].set_visible(False)

    ax_trend.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True,
        fontsize=12,
        fancybox=True
    )

    seed_str = f"seed={seed}" if seed is not None else "random"
    metric_type = "Cell Proportions" if use_proportions else "Cell Counts"
    fig.suptitle(
        f"{metric_type} Along Trajectory: {progression_name}\n"
        f"Model: {model_name} (τ = {tau:.2f}) | {seed_str}",
        fontsize=12,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"✅ All classes figure saved to {output_path}")


def print_summary_table(patches_data: list, use_proportions: bool = PLOT_PROPORTIONS):
    """Print a summary table of cell counts and proportions."""
    print("\n" + "=" * 120)
    print("CELL COUNT SUMMARY")
    print("=" * 120)

    # Header with key classes
    header = f"{'PT':<8} | {'Stage':<25} | {'Total':<8} | "
    header += " | ".join([f"{ct[:12]:<8}" for ct in KEY_CELL_TYPES])
    print(header)
    print("-" * 120)

    for pdata in patches_data:
        total = pdata['total_cells']
        row = f"{pdata['pt']:<8.2f} | {pdata['stage'][:25]:<25} | {total:<8} | "
        
        if use_proportions and total > 0:
            values = [f"{(pdata['counts'].get(ct, 0) / total) * 100:<7.1f}%" for ct in KEY_CELL_TYPES]
        else:
            values = [f"{pdata['counts'].get(ct, 0):<8}" for ct in KEY_CELL_TYPES]
        row += " | ".join(values)
        print(row)

    print("=" * 120)

    # Also print all cell types with any signal
    print("\nALL DETECTED CELL TYPES (max >= 1):")
    print("-" * 80)

    all_counts = {ct: [] for ct in CELL_TYPE_COLUMNS}
    all_totals = [pdata['total_cells'] for pdata in patches_data]
    
    for pdata in patches_data:
        for ct in CELL_TYPE_COLUMNS:
            all_counts[ct].append(pdata['counts'].get(ct, 0))

    for ct in CELL_TYPE_COLUMNS:
        max_val = max(all_counts[ct])
        if max_val >= 1:
            display_name = COLUMN_TO_DISPLAY.get(ct, ct)
            
            # Calculate proportions
            props = []
            for count, total in zip(all_counts[ct], all_totals):
                if total > 0:
                    props.append((count / total) * 100)
                else:
                    props.append(0)
            
            print(f"  {display_name:<25}: max={max_val:>4} ({max(props):>5.1f}%), "
                  f"mean={np.mean(all_counts[ct]):>5.1f} ({np.mean(props):>5.1f}%)")

    print("=" * 120)
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    load_hf_token()
    args = parse_args()

    progression_name = args.progression
    model_name = args.model
    num_samples = args.num_samples
    jitter = args.jitter
    use_proportions = not args.counts  # Default is proportions, --counts switches to raw

    # Set up random seed
    if args.seed is not None:
        seed = args.seed
        logger.info(f"🎲 Using fixed seed: {seed}")
    else:
        seed = int(time.time() * 1000) % (2**32)
        logger.info(f"🎲 Using random seed: {seed} (pass --seed {seed} to reproduce)")

    rng = np.random.default_rng(seed)

    # Load progression config
    prog_config = get_progression_config(progression_name)

    # Determine local image directory
    if args.data_dir:
        local_base = Path(args.data_dir)
    else:
        local_base = LOCAL_IMAGE_DIRS.get(prog_config["bucket"])
        if local_base is None:
            logger.error(f"❌ No local directory configured for bucket: {prog_config['bucket']}")
            logger.error(f"   Use --data_dir to specify the image directory")
            sys.exit(1)

    if not local_base.exists():
        logger.error(f"❌ Local image directory not found: {local_base}")
        sys.exit(1)

    logger.info(f"🔬 Analyzing: {model_name} on {progression_name}")
    logger.info(f"   Sampling {num_samples} patches (quantile, jitter={jitter})")
    logger.info(f"   Using images from: {local_base}")

    if not HISTOPLUS_AVAILABLE:
        logger.error("❌ HistoPLUS required. Install with: pip install histoplus")
        sys.exit(1)

    try:
        cell_counter = HistoPLUSCellCounter()
    except Exception as e:
        logger.error(f"❌ Failed to initialize HistoPLUS: {e}")
        sys.exit(1)

    # Compute trajectory with quantile sampling
    logger.info("\n📊 Computing trajectory...")
    selected_df = compute_trajectory(prog_config, model_name, num_samples, rng, jitter)
    tau = selected_df['tau'].iloc[0]

    # Process patches
    logger.info(f"\n🔬 Processing {len(selected_df)} patches with HistoPLUS...")

    patches_data = []

    for idx, (_, row) in enumerate(selected_df.iterrows()):
        cls = row['class']
        slide_id = str(row['slide_id'])
        patch_id = str(row['patch_id'])

        if not slide_id.startswith("slide_"):
            slide_id = f"slide_{slide_id}"
        if not patch_id.startswith("patch_"):
            patch_id = f"patch_{patch_id}"

        # Construct local file path
        local_path = local_base / "train" / cls / slide_id / f"{patch_id}.png"

        try:
            if not local_path.exists():
                raise FileNotFoundError(f"Image not found: {local_path}")

            img = Image.open(local_path).convert("RGB")

            # Get cell counts
            cell_counts = cell_counter.count_cells(img)

            # Create overlay for visualization
            if cell_counts.type_map is not None:
                overlay = create_overlay(img, cell_counts.type_map)
            else:
                overlay = img

            patches_data.append({
                'pt': row['dpt_pseudotime_norm'],
                'orig_img': img,
                'overlay_img': overlay,
                'counts': cell_counts.counts,
                'total_cells': cell_counts.total_cells,
                'stage': cls,
                'present_classes': classes_in_type_map(cell_counts.type_map),
            })

            cancer_count = cell_counts.counts.get("cancer_cells", 0)
            epithelial_count = cell_counts.counts.get("epithelial_normal", 0)
            logger.info(
                f"   [{idx+1}/{len(selected_df)}] PT={row['dpt_pseudotime_norm']:.2f} | "
                f"Total: {cell_counts.total_cells} | Cancer: {cancer_count} | Epithelial: {epithelial_count}"
            )

        except Exception as e:
            logger.warning(f"   ⚠️ Failed to process {patch_id}: {e}")
            continue

    if not patches_data:
        logger.error("❌ No patches processed successfully.")
        sys.exit(1)

    # Print summary table
    print_summary_table(patches_data, use_proportions=use_proportions)

    # Generate figures
    TRAJECTORY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    prog_slug = progression_name.replace(" ", "_")
    metric_slug = "prop" if use_proportions else "counts"
    seed_suffix = f"_seed{seed}"
    base_filename = f"trajectory_{model_name}_{prog_slug}_{metric_slug}{seed_suffix}"

    # Figure 1: Key classes only
    logger.info(f"🎨 Generating key classes figure...")
    generate_figure_key_classes(
        patches_data,
        model_name,
        progression_name,
        tau,
        seed,
        TRAJECTORY_PLOTS_DIR / f"{base_filename}_key.png",
        use_proportions=use_proportions
    )

    # Figure 2: All classes
    logger.info(f"🎨 Generating all classes figure...")
    generate_figure_all_classes(
        patches_data,
        model_name,
        progression_name,
        tau,
        seed,
        TRAJECTORY_PLOTS_DIR / f"{base_filename}_all.png",
        use_proportions=use_proportions
    )

    # Summary
    metric_type = "proportions" if use_proportions else "counts"
    print(f"\n✅ Analysis complete!")
    print(f"   Model: {model_name} (τ = {tau:.2f})")
    print(f"   Progression: {progression_name}")
    print(f"   Metric: {metric_type}")
    print(f"   Seed: {seed}")
    print(f"\n   Outputs:")
    print(f"     Key classes: {TRAJECTORY_PLOTS_DIR / f'{base_filename}_key.png'}")
    print(f"     All classes: {TRAJECTORY_PLOTS_DIR / f'{base_filename}_all.png'}")
    
    counts_flag = " --counts" if not use_proportions else ""
    print(f"\n💡 To reproduce: python plotting/inspect_trajectory.py --progression {progression_name} --model {model_name} --seed {seed}{counts_flag}")


if __name__ == "__main__":
    main()