#!/usr/bin/env python
"""
Plotting Script: Stage Permutation Specificity (Averaged Across Progressions).
Single-column figure showing pooled null distributions vs. true ordering.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
pcfg.set_icml_style()

PERM_FILE = config.PERMUTATION_RESULTS_OUTPUT_PATH
REAL_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_EMBEDDING = 'final_embedding'

# Color for the real/biological ordering stars
STAR_COLOR = '#c0392b'  # Deep red - stands out against gray

def _filter_to_matching_first_stage(df_perm: pd.DataFrame) -> pd.DataFrame:
    """Drop permuted rows whose first stage doesn't match the canonical first stage.

    The canonical first stage is taken from the `type == 'canonical'` row for each
    progression (constant across models). Returns a filtered copy of df_perm
    containing only canonical rows plus permuted rows whose first stage matches.
    """
    canonical_first = (
        df_perm[df_perm['type'] == 'canonical']
        .groupby('progression')['order_str']
        .first()
        .apply(lambda s: s.split(' -> ')[0])
        .to_dict()
    )
    first_stage = df_perm['order_str'].str.split(' -> ').str[0]
    canonical_match = df_perm['progression'].map(canonical_first) == first_stage
    keep = (df_perm['type'] == 'canonical') | canonical_match
    return df_perm[keep].copy()


def plot_specificity_averaged(match_first_stage: bool = False, null_as_dots: bool = False):
    # 1. Load Data
    df_real = pd.read_csv(REAL_FILE)
    df_perm = pd.read_csv(PERM_FILE)

    if match_first_stage:
        df_perm = _filter_to_matching_first_stage(df_perm)

    # 2. Filter Data
    df_noise = df_perm[df_perm['type'] == 'permuted'].copy()
    df_signal = df_real[df_real['embedding_type'] == TARGET_EMBEDDING].copy()

    # --- Enforce Shared Model Order ---
    df_noise['model'] = pd.Categorical(df_noise['model'], categories=pcfg.MODEL_ORDER, ordered=True)
    df_signal['model'] = pd.Categorical(df_signal['model'], categories=pcfg.MODEL_ORDER, ordered=True)

    # 3. Pool null distributions across progressions for each model
    df_noise_pooled = df_noise.groupby(['model']).apply(
        lambda x: x, include_groups=False
    ).reset_index(drop=True)

    # 4. Compute mean across progressions for real values
    df_signal_avg = df_signal.groupby('model').agg(
        tau_mean=('tau', 'mean'),
    ).reset_index()

    # 5. Setup Figure (wider to accommodate labels)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # 6. Plot Null Distribution - pooled across progressions
    if null_as_dots:
        sns.stripplot(
            data=df_noise,
            x='model',
            y='tau',
            ax=ax,
            order=pcfg.MODEL_ORDER,
            color='#555555',
            alpha=0.55,
            jitter=0.2,
            size=3,
            zorder=2,
        )
    else:
        sns.boxplot(
            data=df_noise,
            x='model',
            y='tau',
            ax=ax,
            color='#e0e0e0',
            width=0.4,
            showfliers=False,
            order=pcfg.MODEL_ORDER,
            boxprops={'linewidth': 1.5, 'edgecolor': '#555555'},
            medianprops={'color': '#333333', 'linewidth': 1.5},
            whiskerprops={'linewidth': 1.2},
            capprops={'linewidth': 1.2}
        )
        # Overlay the underlying permutation points at low alpha
        sns.stripplot(
            data=df_noise,
            x='model',
            y='tau',
            ax=ax,
            order=pcfg.MODEL_ORDER,
            color='#555555',
            alpha=0.15,
            jitter=0.15,
            size=2.5,
            zorder=3,
        )

    # 7. Plot True Signal (Colored Star) - averaged across progressions
    for j, model in enumerate(pcfg.MODEL_ORDER):
        model_data = df_signal_avg[df_signal_avg['model'] == model]

        if not model_data.empty:
            val = model_data['tau_mean'].values[0]
            ax.scatter(j, val, color=STAR_COLOR, s=250, marker='*',
                       edgecolor='white', linewidth=0.5, zorder=10)

    # 8. Formatting
    ax.set_ylabel(r"Trajectory Fidelity ($\tau$)", fontsize=pcfg.FONT_SIZES['axis_label'])
    ax.set_xlabel("")
    
    # X-axis labels with slight rotation to prevent overlap
    labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
    ax.set_xticklabels(labels, fontweight='bold', fontsize=pcfg.FONT_SIZES['tick_label'],
                       rotation=20, ha='right')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='grey')
    ax.xaxis.grid(False)
    
    sns.despine(ax=ax, trim=True)

    # 9. Legend (above plot, horizontal)
    if null_as_dots:
        null_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
                             markeredgecolor='#555555', alpha=0.55, markersize=8,
                             label='Random Orderings')
    else:
        null_legend = Line2D([0], [0], marker='s', color='w', markerfacecolor='#e0e0e0',
                             markeredgecolor='#555555', markersize=10,
                             label='Random Orderings')
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=STAR_COLOR,
               markeredgecolor='white', markersize=14, label='Biological Ordering'),
        null_legend,
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False,
              fontsize=pcfg.FONT_SIZES['legend'])

    # 10. Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # 11. Save
    output_subdir = OUTPUT_DIR / "permutation"
    output_subdir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if match_first_stage:
        suffix += "_match_first_stage"
    if null_as_dots:
        suffix += "_dots"
    output_path = output_subdir / f"permutation_plot_averaged{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot averaged permutation specificity.")
    parser.add_argument(
        "--match-first-stage", action="store_true",
        help="Drop permutations whose first stage differs from the canonical first stage."
    )
    parser.add_argument(
        "--null-as-dots", action="store_true",
        help="Render the null distribution as a stripplot (dots) instead of a boxplot."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_specificity_averaged(
        match_first_stage=args.match_first_stage,
        null_as_dots=args.null_as_dots,
    )