#!/usr/bin/env python
"""
Plotting Script: Patches sampled along the Diffusion Pseudotime trajectory (paper Fig 21).

PURPOSE
    Show, in one figure, how DPT turns patch embeddings into a disease trajectory, and what
    the tissue looks like along it. The reader should be able to follow a handful of sampled
    patches from the space the trajectory is computed in, to their position in pseudotime,
    to the histology itself, and see that patches which are close in pseudotime look alike.

THE FIGURE (trajectory_combined_*.png)
    Top left    Trajectory in diffusion space. DC1 vs DC2 (the first two non-trivial diffusion
                components, the space DPT distances are measured in) for every patch in the
                cohort, colored by ground-truth stage. Stars are the sampled patches, hollow
                circles their adjacent patches. The sign of DC1 is chosen so the root is on
                the left (the sign of an eigenvector is arbitrary, so this is a mirror image).
    Top right   Pseudotime distribution of the cohort. Histogram of normalized pseudotime,
                stacked by ground-truth stage, so it also shows how DPT orders the stages.
    Bottom      Samples from the trajectory. One column per sampled patch: the sampled patch
                on top (heavy frame, header = its number and pseudotime), then patches that
                are adjacent to it in pseudotime (thin frames). Frame color = ground-truth stage.
    Each sampled patch carries the same numbered badge (1..N in pseudotime order) in all
    three views, which is what ties them together.

HOW PATCHES ARE CHOSEN
    Cohort and DPT are identical to inspect_trajectory.py (Fig 20): config.EVALUATION cohort
    and seed, default DPT config. The trajectory is therefore deterministic.
    Sampled patches sit at evenly spaced RANKS of the pseudotime ordering, first and last
    included, so column 1 is the start of the trajectory (t = 0) and the last column its end
    (t = 1). They follow the data density: where most patches are, most samples are.
    Adjacent patches are drawn at random (--seed) from the --pool patches nearest in
    pseudotime, preferring distinct slides and a slide other than the sampled patch's.

Usage (from analyze_embeddings/):
    python plotting/plot_trajectory_patch_grid.py --progression CRC-Conventional --model uni2 --seed 1426409426
    python plotting/plot_trajectory_patch_grid.py --progression SCC --model uni2 --n-anchors 10 --neighbors 4

Output (plots/<RESULTS_RUN>/trajectories/, suffix = {model}_{prog}_{anchors}x{1+neighbors}_seed{SEED}):
    trajectory_combined_<suffix>.png   the figure described above
    trajectory_combined_<suffix>.csv   every patch shown: column, role (sampled|adjacent), pseudotime,
                                       class, slide, patch, dc1, dc2
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from plotting import plot_config as pcfg
from data.progression_embedding_dataset import (
    ProgressionEmbeddingDataset,
    RegistryConfig,
)
from analysis.dpt import DPTConfig, DPTMetric, cohort_to_anndata, compute_dpt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = config.PLOTS_OUTPUT_DIR / "trajectories"

DEFAULT_N_ANCHORS = 8
DEFAULT_NEIGHBORS = 3       # adjacent patches shown under each anchor
DEFAULT_POOL = 20           # adjacent patches are drawn from the POOL nearest in pseudotime

# Same local image layout as inspect_trajectory.py
LOCAL_IMAGE_DIRS = config.LOCAL_IMAGE_DIRS  # set SPIDER_IMAGE_ROOT or edit config.py

# Fixed per-stage colors (ColorBrewer Set1), assigned in progression order.
# Same palette as plot_method_overview_figure.py.
STAGE_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628"]

INK = "#222222"
INK_MUTED = "#666666"


# ── Args ──────────────────────────────────────────────────────────────────────
def get_progression_config(progression_name: str) -> dict:
    for prog in config.PROGRESSIONS:
        if prog["name"] == progression_name:
            return prog
    available = [p["name"] for p in config.PROGRESSIONS]
    raise ValueError(f"Progression '{progression_name}' not found. Available: {available}")


def parse_args():
    available = [p["name"] for p in config.PROGRESSIONS]
    p = argparse.ArgumentParser(
        description="Anchor patches along a DPT trajectory with their pseudotime-adjacent patches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--progression", required=True, choices=available)
    p.add_argument("--model", required=True, choices=config.EXPECTED_MODELS)
    p.add_argument("--n-anchors", type=int, default=DEFAULT_N_ANCHORS,
                   help=f"Number of anchor pseudotimes / columns (default {DEFAULT_N_ANCHORS})")
    p.add_argument("--neighbors", type=int, default=DEFAULT_NEIGHBORS,
                   help=f"Adjacent patches shown under each anchor (default {DEFAULT_NEIGHBORS})")
    p.add_argument("--pool", type=int, default=DEFAULT_POOL,
                   help=f"Adjacent patches are drawn from the POOL nearest in pseudotime (default {DEFAULT_POOL})")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for which adjacent patches are drawn from the pool (default: random)")
    p.add_argument("--data_dir", type=str, default=None, help="Override local image directory")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# ── Trajectory ────────────────────────────────────────────────────────────────
def compute_trajectory(prog_config: dict, model_name: str):
    """Same cohort + DPT as inspect_trajectory.compute_trajectory. Returns
    (obs DataFrame sorted by dpt_pseudotime_norm, tau). The DataFrame also carries
    dc1 / dc2: the first two non-trivial diffusion components, i.e. the space DPT
    itself is computed in (same indexing as plot_method_overview_figure.py)."""
    reg = RegistryConfig(
        bucket=prog_config["bucket"], prefix=prog_config["prefix"],
        ordered_classes=prog_config["classes"], models=[model_name],
        progression_name=prog_config["name"], scan_all_models=False,
    )
    dataset = ProgressionEmbeddingDataset(reg)
    try:
        patch_ids = dataset.sample_patch_ids(
            n_per_class=config.EVALUATION["n_per_class"],
            max_per_slide=config.EVALUATION["max_per_slide"],
            seed=config.EVALUATION["seed"],
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
            subsample_size=2000,
        )
        logger.info("⚙️  Computing Diffusion Pseudotime...")
        result = compute_dpt(adata, prog_config["root_class"], dpt_conf)
        logger.info(f"   Trajectory Fidelity: τ = {result.tau:.3f}")
        if "dpt_pseudotime" not in adata.obs:
            raise ValueError("DPT computation failed")

        # Diffusion components 1 and 2 (column 0 of X_diffmap is the trivial steady state)
        adata.obs["dc1"] = adata.obsm["X_diffmap"][:, 1]
        adata.obs["dc2"] = adata.obsm["X_diffmap"][:, 2]

        df = adata.obs.copy()
        df = df[np.isfinite(df["dpt_pseudotime"])]
        lo, hi = df["dpt_pseudotime"].min(), df["dpt_pseudotime"].max()
        df["dpt_pseudotime_norm"] = (df["dpt_pseudotime"] - lo) / (hi - lo)
        df = df.sort_values("dpt_pseudotime_norm").reset_index(drop=True)
        # The sign of a diffusion component (an eigenvector) is arbitrary. Pick the sign of DC1
        # that puts the root (t = 0) on the left of the plot; a mirror image, distances unchanged.
        if df["dc1"].iloc[0] > df["dc1"].median():
            df["dc1"] = -df["dc1"]
        return df, float(result.tau)
    finally:
        dataset.clear_cache()


def select_anchor_ranks(df: pd.DataFrame, n_anchors: int) -> List[int]:
    """Row positions (df is sorted by pseudotime) of the anchor patches: evenly
    spaced ranks including the first (t=0) and last (t=1) patch of the ordering."""
    return [int(r) for r in np.linspace(0, len(df) - 1, n_anchors)]


def nearest_in_pseudotime(df: pd.DataFrame, rank: int, k: int) -> pd.DataFrame:
    """The k rows nearest to row `rank` in pseudotime (excluding the anchor itself)."""
    t = df["dpt_pseudotime_norm"].values
    d = np.abs(t - t[rank])
    d[rank] = np.inf
    idx = np.argsort(d, kind="stable")[:k]
    return df.iloc[idx]


def sample_adjacent(df: pd.DataFrame, rank: int, k: int, pool: int,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Draw k patches from the `pool` nearest in pseudotime, preferring distinct slides
    (and a slide different from the anchor's), then sort by pseudotime."""
    anchor_slide = df.iloc[rank]["slide_id"]
    cand = nearest_in_pseudotime(df, rank, pool)
    shuffled = cand.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    preferred = shuffled[shuffled["slide_id"] != anchor_slide].drop_duplicates("slide_id").head(k)
    if len(preferred) < k:
        rest = shuffled.drop(preferred.index).head(k - len(preferred))
        preferred = pd.concat([preferred, rest])
    return preferred.sort_values("dpt_pseudotime_norm")


# ── Images ────────────────────────────────────────────────────────────────────
def _with_prefix(value: str, prefix: str) -> str:
    value = str(value)
    return value if value.startswith(prefix) else f"{prefix}{value}"


def load_local_image(local_base: Path, cls: str, slide_id: str, patch_id: str) -> Optional[Image.Image]:
    path = local_base / "train" / cls / _with_prefix(slide_id, "slide_") / f"{_with_prefix(patch_id, 'patch_')}.png"
    if not path.exists():
        logger.warning(f"Image not found: {path}")
        return None
    return Image.open(path).convert("RGB")


# ── Figure ────────────────────────────────────────────────────────────────────
def _draw_patch(ax, img, class_idx: int, linewidth: float):
    ax.set_xticks([]); ax.set_yticks([])
    if img is not None:
        ax.imshow(img)
    else:
        ax.set_facecolor("#f0f0f0")
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                fontsize=pcfg.FONT_SIZES["small"], color=INK_MUTED)
    color = STAGE_COLORS[class_idx % len(STAGE_COLORS)]
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor(color); s.set_linewidth(linewidth)


def _title_lines(model_name: str, tau: float, n: int, seed: int) -> str:
    return f"{pcfg.MODEL_LABELS.get(model_name, model_name)} (τ = {tau:.2f}) · n = {n:,} patches · seed = {seed}"


def _label_offsets(ax, stars: np.ndarray, data: np.ndarray, label_size=(23.0, 23.0),
                   radii=(23.0, 32.0, 42.0, 54.0), n_angles: int = 24):
    """Offsets (in points) for the star labels, chosen greedily in display space so that each
    label covers as few data points as possible, covers no star, stays inside the axes and
    does not overlap a label placed before it. Call after the axes limits and layout are final."""
    pt = ax.figure.dpi / 72.0                                  # pixels per point
    S, D = ax.transData.transform(stars) / pt, ax.transData.transform(data) / pt
    box = ax.get_window_extent()
    lo, hi = np.array([box.x0, box.y0]) / pt, np.array([box.x1, box.y1]) / pt
    half = np.array(label_size) / 2.0
    angles = 2 * np.pi * np.arange(n_angles) / n_angles
    placed, offsets = [], [None] * len(S)
    # most crowded stars first: they have the fewest free positions
    crowding = (np.linalg.norm(S[:, None] - S[None], axis=-1) < 60).sum(axis=1)
    for i in np.argsort(-crowding, kind="stable"):
        best = (np.inf, (10.0, 10.0))
        for r in radii:
            for a in angles:
                off = np.array([r * np.cos(a), r * np.sin(a)])
                c = S[i] + off
                cost = (np.abs(D - c) < half).all(axis=1).sum()                 # data hidden
                cost += 200 * (np.abs(S - c) < half + 8).all(axis=1).sum()      # stars hidden
                cost += 1000 * sum((np.abs(c - q) < 2 * half + 3).all() for q in placed)
                cost += 1000 * ((c - half < lo).any() or (c + half > hi).any())
                cost += 0.05 * r                                                # prefer short leaders
                if cost < best[0]:
                    best = (cost, (off[0], off[1]))
        offsets[i] = best[1]
        placed.append(S[i] + np.array(best[1]))
    return offsets


def _marker_handles(sampled_label: str = "sampled patch"):
    return [plt.Line2D([], [], marker="*", color=INK, markersize=13, linestyle="none",
                       markeredgecolor="white", label=sampled_label),
            plt.Line2D([], [], marker="o", color=INK, markersize=6, linestyle="none",
                       markerfacecolor="none", label="adjacent patch")]


def _draw_diffmap(ax, df: pd.DataFrame, anchor_ranks: List[int], adjacent: List[pd.DataFrame],
                  classes: List[str]):
    """Scatter of DC1 vs DC2 for the whole cohort, colored by ground-truth stage, with the
    sampled patches as black stars and their adjacent patches as hollow circles."""
    anchors = df.iloc[anchor_ranks]
    adj_all = pd.concat(adjacent)
    # Random draw order so no stage hides another
    order = np.random.default_rng(0).permutation(len(df))
    stage_rgb = np.array([matplotlib.colors.to_rgb(STAGE_COLORS[classes.index(c) % len(STAGE_COLORS)])
                          for c in df["class"]])
    ax.scatter(df["dc1"].values[order], df["dc2"].values[order], c=stage_rgb[order],
               s=12, alpha=0.75, edgecolors="none", rasterized=True)
    ax.scatter(adj_all["dc1"], adj_all["dc2"], s=30, facecolors="none",
               edgecolors=INK, linewidths=0.9, zorder=6)
    ax.scatter(anchors["dc1"], anchors["dc2"], marker="*", s=280, c=INK,
               edgecolors="white", linewidths=0.8, zorder=7)
    ax.margins(0.08)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK_MUTED); sp.set_linewidth(0.6)
    ax.set_xlabel("Diffusion component 1", fontsize=pcfg.FONT_SIZES["axis_label"], color=INK)
    ax.set_ylabel("Diffusion component 2", fontsize=pcfg.FONT_SIZES["axis_label"], color=INK)


BADGE_FONTSIZE = 12


def _badge(ax, number: int, xy, xytext=(0, 0), xycoords="data", leader: bool = False):
    """The numbered marker that identifies one sampled patch. Drawn identically in the
    diffusion map, on the pseudotime histogram and above the patch columns. `xytext` is the
    badge centre as an offset in points from `xy`; with `leader`, a thin line joins the two."""
    arrow = dict(arrowstyle="-", color=INK, linewidth=0.7, shrinkA=0, shrinkB=3) if leader else None
    return ax.annotate(str(number), xy, xycoords=xycoords, xytext=xytext, textcoords="offset points",
                       fontsize=BADGE_FONTSIZE, fontweight="bold", color="white",
                       ha="center", va="center", zorder=9, annotation_clip=False, arrowprops=arrow,
                       bbox=dict(boxstyle="circle,pad=0.3", facecolor=INK, edgecolor="white", linewidth=0.8))


def plot_combined(df: pd.DataFrame, anchor_ranks: List[int], adjacent: List[pd.DataFrame],
                  classes: List[str], model_name: str, prog_name: str, tau: float, seed: int,
                  local_base: Path, output_path: Path, dpi: int):
    """One figure, three views of the same sampled patches: the diffusion map (top left), the
    pseudotime distribution (top right) and the patches themselves (below). Each sampled patch
    carries the same numbered badge, 1..N in pseudotime order, in all three views. The sampled
    row is set apart from the adjacent patches by heavier frames."""
    pcfg.set_icml_style()
    n_cols = len(anchor_ranks)
    n_rows = 1 + max(len(a) for a in adjacent)
    fs = pcfg.FONT_SIZES
    stage_color = [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(len(classes))]

    # Vertical layout, in inches from the top of the figure
    fig_w = n_cols * 1.55 + 0.9
    plots_top, plots_h = 1.15, 3.5                 # title + plot titles above
    grid_top = plots_top + plots_h + 1.75          # x labels, the grid title, then the badge headers
    grid_h = n_rows * 1.52 + 0.15
    fig_h = grid_top + grid_h + 0.80               # stage legend below
    left, right = 0.075, 0.99
    fig = plt.figure(figsize=(fig_w, fig_h))

    t_all = df["dpt_pseudotime_norm"].values
    anchors = df.iloc[anchor_ranks]
    anchor_t = anchors["dpt_pseudotime_norm"].values

    # ── Top row: diffusion map | pseudotime distribution ──
    gs_top = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.25], wspace=0.16, left=left, right=right,
                      top=1 - plots_top / fig_h, bottom=1 - (plots_top + plots_h) / fig_h)
    ax_map, ax_pt = fig.add_subplot(gs_top[0, 0]), fig.add_subplot(gs_top[0, 1])

    _draw_diffmap(ax_map, df, anchor_ranks, adjacent, classes)
    ax_map.set_title("Trajectory in diffusion space (DC1 vs DC2)", fontsize=fs["axis_label"], color=INK, pad=10)
    ax_map.legend(handles=_marker_handles(), loc="best", frameon=False, fontsize=fs["legend"],
                  borderaxespad=0.6, handletextpad=0.4)

    bins = np.linspace(0, 1, 101)
    per_class = [df.loc[df["class"] == c, "dpt_pseudotime_norm"].values for c in classes]
    ax_pt.hist(per_class, bins=bins, stacked=True, linewidth=0, color=stage_color, alpha=0.9)
    hist_total, _ = np.histogram(t_all, bins=bins)
    # Each sampled patch's badge sits on the distribution at its pseudotime (its local density)
    for i, t in enumerate(anchor_t):
        _badge(ax_pt, i + 1, (t, hist_total[min(int(t * 100), 99)]))
    ax_pt.set_xlim(-0.03, 1.03); ax_pt.set_ylim(0, hist_total.max() * 1.12)
    ax_pt.set_xticks(np.linspace(0, 1, 11))
    ax_pt.tick_params(axis="x", labelsize=fs["small"], colors=INK, length=3)
    ax_pt.tick_params(axis="y", labelsize=fs["small"], colors=INK_MUTED, length=2)
    ax_pt.set_xlabel("Diffusion pseudotime (normalized)  →", fontsize=fs["axis_label"], color=INK)
    ax_pt.set_ylabel("Number of patches", fontsize=fs["axis_label"], color=INK)
    ax_pt.set_title("Pseudotime distribution of the cohort", fontsize=fs["axis_label"], color=INK, pad=10)
    for sp in ax_pt.spines.values():
        sp.set_color(INK_MUTED); sp.set_linewidth(0.6)

    # ── Patch grid: sampled row | spacer | adjacent rows ──
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[1.0, 0.16] + [1.0] * (n_rows - 1),
                  hspace=0.08, wspace=0.08, left=left, right=right,
                  top=1 - grid_top / fig_h, bottom=1 - (grid_top + grid_h) / fig_h)
    adjacent_axes = []
    for c, (rank, adj) in enumerate(zip(anchor_ranks, adjacent)):
        rows = [df.iloc[rank]] + [r for _, r in adj.iterrows()]
        for r_i in range(n_rows):
            ax = fig.add_subplot(gs[0 if r_i == 0 else r_i + 1, c])
            if r_i < len(rows):
                row = rows[r_i]
                img = load_local_image(local_base, row["class"], row["slide_id"], row["patch_id"])
                _draw_patch(ax, img, classes.index(row["class"]), linewidth=5.5 if r_i == 0 else 2.0)
            else:
                ax.axis("off")
            if r_i > 0:
                adjacent_axes.append(ax)
            if r_i == 0:
                # Column header: the patch's badge, then its pseudotime
                _badge(ax, c + 1, (0.5, 1.0), (-34, 18), xycoords="axes fraction")
                ax.annotate(f"t = {anchor_t[c]:.2f}", (0.5, 1.0), xycoords="axes fraction", xytext=(-18, 18),
                            textcoords="offset points", ha="left", va="center", fontsize=fs["axis_label"],
                            fontweight="bold", color=INK, annotation_clip=False)

    fig.text((left + right) / 2, 1 - (grid_top - 0.80) / fig_h, "Samples from the trajectory",
             ha="center", va="center", fontsize=fs["axis_label"], color=INK)

    # Left-margin caption + bracket for the adjacent block
    x_brace, x_text = 0.052, 0.040
    y0 = min(a.get_position().y0 for a in adjacent_axes)
    y1 = max(a.get_position().y1 for a in adjacent_axes)
    fig.add_artist(plt.Line2D([x_brace, x_brace], [y0, y1], transform=fig.transFigure,
                              color=INK_MUTED, linewidth=1.0))
    for y in (y0, y1):
        fig.add_artist(plt.Line2D([x_brace, x_brace + 0.004], [y, y], transform=fig.transFigure,
                                  color=INK_MUTED, linewidth=1.0))
    fig.text(x_text, (y0 + y1) / 2, "adjacent patches\nin pseudotime",
             rotation=90, ha="center", va="center", fontsize=fs["small"], color=INK)

    handles = [mpatches.Patch(facecolor=col, edgecolor="white", label=c) for col, c in zip(stage_color, classes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(classes), frameon=False,
               bbox_to_anchor=(0.5, 0.08 / fig_h), fontsize=fs["legend"], handlelength=1.4, columnspacing=1.8,
               title="Ground-truth stage (point, frame and histogram color)", title_fontsize=fs["legend"])

    fig.suptitle(
        f"Patches sampled along the pseudotime trajectory: {pcfg.get_progression_title(prog_name)}\n"
        + _title_lines(model_name, tau, len(df), seed),
        fontsize=fs["subtitle"], color=INK, y=1 - 0.08 / fig_h)

    # Diffusion map: a badge next to each star, in the emptiest nearby spot. Placed once the
    # layout is final so the placement can see where the data is.
    fig.canvas.draw()
    pts = anchors[["dc1", "dc2"]].values
    offsets = _label_offsets(ax_map, pts, df[["dc1", "dc2"]].values)
    for i, (xy, off) in enumerate(zip(pts, offsets)):
        _badge(ax_map, i + 1, tuple(xy), off, leader=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"✅ Figure saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    prog_config = get_progression_config(args.progression)
    classes = prog_config["classes"]

    if args.seed is not None:
        seed = args.seed
        logger.info(f"🎲 Using fixed seed: {seed}")
    else:
        seed = int(time.time() * 1000) % (2**32)
        logger.info(f"🎲 Using random seed: {seed} (pass --seed {seed} to reproduce)")
    rng = np.random.default_rng(seed)

    local_base = Path(args.data_dir) if args.data_dir else LOCAL_IMAGE_DIRS.get(prog_config["bucket"])
    if local_base is None or not local_base.exists():
        logger.error(f"❌ Local image directory not found: {local_base} (use --data_dir)")
        sys.exit(1)

    logger.info(f"🔬 {args.model} on {args.progression}: {args.n_anchors} anchors "
                f"× {args.neighbors} adjacent patches")
    df, tau = compute_trajectory(prog_config, args.model)
    anchor_ranks = select_anchor_ranks(df, args.n_anchors)
    adjacent = [sample_adjacent(df, r, args.neighbors, args.pool, rng) for r in anchor_ranks]
    for i, (r, adj) in enumerate(zip(anchor_ranks, adjacent)):
        a = df.iloc[r]
        logger.info(f"   anchor {i + 1}: t={a['dpt_pseudotime_norm']:.3f} ({a['class']}), "
                    f"adjacent t ∈ [{adj['dpt_pseudotime_norm'].min():.3f}, {adj['dpt_pseudotime_norm'].max():.3f}]")

    prog_slug = args.progression.replace(" ", "_")
    base = f"trajectory_combined_{args.model}_{prog_slug}_{args.n_anchors}x{1 + args.neighbors}_seed{seed}"
    out_png = OUTPUT_DIR / f"{base}.png"
    out_csv = OUTPUT_DIR / f"{base}.csv"

    plot_combined(df, anchor_ranks, adjacent, classes, args.model, args.progression, tau, seed,
                  local_base, out_png, args.dpi)

    rows = []
    for c, (r, adj) in enumerate(zip(anchor_ranks, adjacent)):
        a = df.iloc[r]
        rows.append({"column": c + 1, "role": "sampled", "pseudotime_norm": a["dpt_pseudotime_norm"],
                     "class": a["class"], "slide_id": a["slide_id"], "patch_id": a["patch_id"],
                     "dc1": a["dc1"], "dc2": a["dc2"]})
        for _, row in adj.iterrows():
            rows.append({"column": c + 1, "role": "adjacent", "pseudotime_norm": row["dpt_pseudotime_norm"],
                         "class": row["class"], "slide_id": row["slide_id"], "patch_id": row["patch_id"],
                         "dc1": row["dc1"], "dc2": row["dc2"]})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info(f"✅ Sampled-patch table saved to {out_csv}")
    print(f"\n💡 To reproduce: python plotting/plot_trajectory_patch_grid.py --progression {args.progression} "
          f"--model {args.model} --n-anchors {args.n_anchors} --neighbors {args.neighbors} --seed {seed}")


if __name__ == "__main__":
    main()
