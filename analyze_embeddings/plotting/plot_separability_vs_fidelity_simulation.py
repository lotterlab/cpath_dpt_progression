"""
Simulation: High class separability does NOT guarantee high trajectory fidelity.

Generates two scenarios with identical cluster geometry (same separability) but
different stage-to-cluster assignments, producing different DPT Kendall's tau.

Self-contained: needs no embeddings or cached results.

Usage (from analyze_embeddings/):
    python plotting/plot_separability_vs_fidelity_simulation.py [--n_per_class 50] [--noise 0.15] [--seed 42]

Output:
    plots/<RESULTS_RUN>/simulation/separability_vs_fidelity.png
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import kendalltau

# Allow imports from parent package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from plotting.plot_config import set_icml_style, FONT_SIZES


# ── Centroid definitions ──────────────────────────────────────────────────────
CENTROIDS = {
    "Class A": np.array([1.0, 0.0]),
    "Class B": np.array([0.0, 1.0]),
    "Class C": np.array([0.5, 0.5]),
}

# Scenario definitions: stage ordering over the *same* clusters.
# Scenario A traverses A→B→C (skips the geometrically intermediate cluster).
# Scenario B traverses A→C→B (follows the natural geometric path).
SCENARIOS = {
    # Scenario A: (1,0)→Stage 1, (0.5,0.5)→Stage 2, (0,1)→Stage 3
    "Scenario A": ["Class A", "Class C", "Class B"],
    # Scenario B: Stage 2 and 3 flipped — (0,1)→Stage 2, (0.5,0.5)→Stage 3
    "Scenario B": ["Class A", "Class B", "Class C"],
}

CLASS_COLORS = {
    "Class A": "#1f77b4",
    "Class B": "#ff7f0e",
    "Class C": "#2ca02c",
}

CLASS_MARKERS = {
    "Class A": "o",
    "Class B": "s",
    "Class C": "D",
}


# ── Data generation ───────────────────────────────────────────────────────────

def generate_embeddings(n_per_class: int, noise: float, seed: int) -> pd.DataFrame:
    """Generate 2-D embeddings around each centroid.  Shared across scenarios."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls_name, centroid in CENTROIDS.items():
        for _ in range(n_per_class):
            vec = centroid + noise * rng.standard_normal(2)
            rows.append({"class": cls_name, "x": vec[0], "y": vec[1]})
    return pd.DataFrame(rows)


# ── DPT tau via scanpy ────────────────────────────────────────────────────────

def compute_tau(df: pd.DataFrame, stage_order: list[str], n_neighbors: int = 15) -> float:
    """Build a diffusion map on the 2-D points and return Kendall's tau."""
    X = df[["x", "y"]].values
    obs = df[["class"]].copy()
    class_map = {c: i for i, c in enumerate(stage_order)}
    obs["stage_int"] = obs["class"].map(class_map)

    adata = sc.AnnData(X=X, obs=obs)

    n_nbrs = min(n_neighbors, adata.n_obs - 1)
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_nbrs, metric="euclidean", method="gauss")
    sc.tl.diffmap(adata)

    # Root cell = medoid of the first stage
    root_class = stage_order[0]
    mask = adata.obs["class"] == root_class
    X_cls = adata.X[np.where(mask)[0]]
    centroid = X_cls.mean(axis=0).reshape(1, -1)
    from scipy.spatial.distance import cdist
    dists = cdist(X_cls, centroid, metric="euclidean")
    adata.uns["iroot"] = np.where(mask)[0][np.argmin(dists)]

    sc.tl.dpt(adata, n_dcs=10)

    valid = np.isfinite(adata.obs["dpt_pseudotime"])
    tau, _ = kendalltau(
        adata.obs["dpt_pseudotime"][valid],
        adata.obs["stage_int"][valid],
    )
    return tau


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_simulation(df: pd.DataFrame, taus: dict[str, float], out_path: str):
    set_icml_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, (scenario_name, stage_order) in zip(axes, SCENARIOS.items()):
        tau = taus[scenario_name]
        class_to_stage = {cls: idx + 1 for idx, cls in enumerate(stage_order)}

        # Plot each class
        for cls_name in CENTROIDS:
            mask = df["class"] == cls_name
            stage = class_to_stage[cls_name]
            ax.scatter(
                df.loc[mask, "x"],
                df.loc[mask, "y"],
                c=CLASS_COLORS[cls_name],
                marker=CLASS_MARKERS[cls_name],
                s=30,
                alpha=0.6,
                edgecolors="none",
                label=f"Stage {stage}",
            )

        # Mark centroids
        for cls_name, centroid in CENTROIDS.items():
            ax.scatter(
                *centroid,
                c=CLASS_COLORS[cls_name],
                marker="X",
                s=120,
                edgecolors="k",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                f"Stage {class_to_stage[cls_name]}",
                xy=centroid,
                xytext=(24, 4),
                textcoords="offset points",
                fontsize=FONT_SIZES["small"],
                color="black",
                fontweight="bold",
            )

        # Draw arrows showing progression order
        for i in range(len(stage_order) - 1):
            src = CENTROIDS[stage_order[i]]
            dst = CENTROIDS[stage_order[i + 1]]
            ax.annotate(
                "",
                xy=dst,
                xytext=src,
                arrowprops=dict(arrowstyle="->", color="#7a7a7a", lw=1.5, connectionstyle="arc3,rad=0.15"),
            )

        ax.set_title(
            f"{scenario_name}   (DPT τ = {tau:.3f})",
            fontsize=FONT_SIZES["subtitle"],
            pad=24,
        )

        # Colored subtitle: "Disease progression order: Stage 1 → Stage 2 → Stage 3"
        # where each stage label is colored by the class it maps to in this scenario.
        from matplotlib.offsetbox import TextArea, DrawingArea, HPacker, AnnotationBbox
        parts = [TextArea("Disease progression order: ",
                          textprops=dict(fontsize=FONT_SIZES["subtitle"], color="black"))]
        for i, cls_name in enumerate(stage_order):
            da = DrawingArea(16, 16, 0, 0)
            marker_line = Line2D(
                [8], [8],
                marker=CLASS_MARKERS[cls_name],
                markersize=10,
                markerfacecolor=CLASS_COLORS[cls_name],
                markeredgecolor=CLASS_COLORS[cls_name],
                linestyle="None",
            )
            da.add_artist(marker_line)
            parts.append(da)
            if i < len(stage_order) - 1:
                parts.append(TextArea(" → ",
                                      textprops=dict(fontsize=FONT_SIZES["subtitle"], color="black")))
        hbox = HPacker(children=parts, align="center", pad=0, sep=2)
        ab = AnnotationBbox(hbox, (0.5, 1.02), xycoords="axes fraction",
                            frameon=False, box_alignment=(0.5, 0.0))
        ax.add_artist(ab)
        ax.set_xlabel("Dimension 1", fontsize=FONT_SIZES["axis_label"])

    axes[0].set_ylabel("Dimension 2", fontsize=FONT_SIZES["axis_label"])

    fig.suptitle(
        "Same Class Separability, Different Trajectory Fidelity",
        fontsize=FONT_SIZES["title"],
        y=1.02,
        fontweight="bold",
    )
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_per_class", type=int, default=50,
                        help="Number of embeddings per class (default: 50)")
    parser.add_argument("--noise", type=float, default=0.15,
                        help="Gaussian noise scale λ (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: plots/<RESULTS_RUN>/simulation/separability_vs_fidelity.png)")
    args = parser.parse_args()

    # Generate shared embeddings
    df = generate_embeddings(args.n_per_class, args.noise, args.seed)

    # Compute tau for each scenario
    taus = {}
    for name, stage_order in SCENARIOS.items():
        tau = compute_tau(df, stage_order)
        taus[name] = tau
        print(f"{name}: DPT τ = {tau:.4f}")

    # Output path
    if args.out is None:
        out_dir = config.PLOTS_OUTPUT_DIR / "simulation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "separability_vs_fidelity.png")
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plot_simulation(df, taus, out_path)


if __name__ == "__main__":
    main()
