#!/usr/bin/env python
"""
Plotting Script: Stage Permutation Specificity (Final Paper Version).
Grid layout, one panel per progression, with colored stars for the true ordering.
"""

import argparse
import sys
from pathlib import Path
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import plotting.plot_config as pcfg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Use the shared style from plot_config
pcfg.set_icml_style()

PERM_FILE = config.PERMUTATION_RESULTS_OUTPUT_PATH
REAL_FILE = config.FULL_RESULTS_OUTPUT_PATH
OUTPUT_DIR = config.PLOTS_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The specific embedding type that contains the canonical signal + CIs
TARGET_EMBEDDING = 'final_embedding'

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


def plot_specificity_grid(match_first_stage: bool = False, null_as_dots: bool = False, show_ci: bool = True):
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
    
    # Sort data
    df_noise = df_noise.sort_values('model')
    df_signal = df_signal.sort_values('model')

    unique_progressions = df_noise['progression'].unique()
    num_progs = len(unique_progressions)
    
    # 3. Setup Grid (2 Columns)
    cols = 2
    rows = math.ceil(num_progs / cols)
    
    # Slightly taller figure to accommodate the new Global Title
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.0 * rows), 
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, prog in enumerate(unique_progressions):
        ax = axes_flat[i]
        
        # --- LOOKUP COLOR FROM CONFIG ---
        star_color = pcfg.PROGRESSION_COLORS.get(prog, '#D62728')
        
        # Subset data
        noise_subset = df_noise[df_noise['progression'] == prog]
        signal_subset = df_signal[df_signal['progression'] == prog]
        
        # --- A. Plot Null Distribution ---
        if null_as_dots:
            sns.stripplot(data=noise_subset, x='model', y='tau', ax=ax,
                          order=pcfg.MODEL_ORDER,
                          color='#555555', alpha=0.55, jitter=0.2, size=4, zorder=2)
        else:
            sns.boxplot(data=noise_subset, x='model', y='tau', ax=ax,
                        color='#e0e0e0', width=0.5, showfliers=False,
                        boxprops={'linewidth': 1.5, 'edgecolor': '#555555'},
                        medianprops={'color': '#333333', 'linewidth': 1.5})
        
        # --- B. Plot True Signal (Star + CI) ---
        for j, model in enumerate(pcfg.MODEL_ORDER):
            model_signal = signal_subset[signal_subset['model'] == model]
            
            if not model_signal.empty:
                val = model_signal['tau'].values[0]
                lower = model_signal['tau_ci_lower'].values[0]
                upper = model_signal['tau_ci_upper'].values[0]
                
                # 1. Plot the Star (Colored by Disease)
                ax.scatter(j, val, color=star_color, s=250, marker='*', 
                           edgecolor='white', linewidth=0.5, zorder=10)
                
                # 2. Plot the Error Bars
                if show_ci and not np.isnan(lower) and not np.isnan(upper):
                    yerr = [[val - lower], [upper - val]]
                    ax.errorbar(j, val, yerr=yerr, fmt='none', ecolor='black',
                                elinewidth=1.8, capsize=5, zorder=11)

        # --- Formatting ---
        ax.set_title(f"{prog}", fontweight='bold', fontsize=14, pad=10)
        
        if i % cols == 0:
            ax.set_ylabel(r"Trajectory Fidelity ($\tau$)", labelpad=10)
        else:
            ax.set_ylabel("")
            
        if i >= (rows - 1) * cols:
            labels = [pcfg.MODEL_LABELS.get(m, m) for m in pcfg.MODEL_ORDER]
            labels = [l.replace('Prov-GigaPath', 'Prov-\nGigaPath') if 'Prov-GigaPath' in l else l for l in labels]
            ax.set_xticklabels(labels, rotation=0, fontweight='bold', fontsize=10)
            ax.set_xlabel("")
        else:
            ax.set_xlabel("")

        ax.set_ylim(top=0.8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='grey')
        ax.xaxis.grid(False)
        sns.despine(ax=ax, trim=True)

    # Hide unused axes
    for k in range(i + 1, len(axes_flat)):
        axes_flat[k].axis('off')

    # --- GLOBAL TITLE & LEGEND ---
    
    # # 1. Add Global Title
    # fig.suptitle("Specificity of trajectory fidelity: Permutation testing against null stage orders", 
    #              fontsize=16, fontweight='bold', y=0.98)

    # 2. Add Legend
    if null_as_dots:
        null_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
                             markeredgecolor='#555555', alpha=0.55, markersize=8,
                             label='Random Orderings')
    else:
        null_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='#e0e0e0',
                             markeredgecolor='grey', markersize=8,
                             label='Random Orderings')
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markersize=14, label='Biological Ordering'),
        null_legend,
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False, fontsize=12)

    # 3. Adjust Layout
    # rect=[left, bottom, right, top] -> top=0.90 reserves top 10% for title/legend
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    output_subdir = OUTPUT_DIR / "permutation"
    output_subdir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if match_first_stage:
        suffix += "_match_first_stage"
    if null_as_dots:
        suffix += "_dots"
    if not show_ci:
        suffix += "_no_ci"
    output_path = output_subdir / f"permutation_plot{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-progression permutation specificity.")
    parser.add_argument(
        "--match-first-stage", action="store_true",
        help="Drop permutations whose first stage differs from the canonical first stage."
    )
    parser.add_argument(
        "--null-as-dots", action="store_true",
        help="Render the null distribution as a stripplot (dots) instead of a boxplot."
    )
    parser.add_argument(
        "--no-ci", action="store_true",
        help="Hide the confidence interval error bars on the canonical (star) points."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_specificity_grid(
        match_first_stage=args.match_first_stage,
        null_as_dots=args.null_as_dots,
        show_ci=not args.no_ci,
    )