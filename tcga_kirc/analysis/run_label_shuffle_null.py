#!/usr/bin/env python3
"""Label-shuffled null for every model x endpoint on KIRC.

For each (model, endpoint) the endpoint is permuted across patients B times and
the metric recomputed, giving an empirical null for "no association". This is
the right null here because it works identically for Kendall tau and for the
C-index, and because it respects the data structure: DPT is transductive, so
patients are not independent draws and the analytic Kendall p-value is
optimistic.

Reported per cell:
  observed, null mean, null SD, null 95% band, permutation p, and z = the
  standardized distance above the null. z is the useful quantity -- with
  B = 2000 the p-value floors at 1/(B+1) = 0.0005 and stops discriminating,
  while z keeps ranking models and is comparable across endpoints whose
  metrics live on different scales (tau vs C-index).

Writes the summary and the full null draws, then plots observed vs null band.

Usage (from tcga_kirc/analysis/):
    python run_label_shuffle_null.py                           # cached paper results
    python run_label_shuffle_null.py --results results/<run>   # a new run_dpt.py run
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from lifelines.utils import concordance_index
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests

import config

# Figure colours
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e7e6e2"

warnings.filterwarnings("ignore")

SERIES_1 = "#2a78d6"      # observed value
NULL_FILL = "#d8d7d2"     # null band -- reference, deliberately not a series hue

# (label, kind, column). Stage is included but flagged: it is the most
# image-adjacent endpoint and the least independent of the grade ordering.
TASKS = [("stage", "tau", "stage_ordinal"),
         ("mRNAsi", "tau", "mRNAsi"),
         ("aneuploidy", "tau", "ANEUPLOIDY_SCORE"),
         ("OS", "cox", "OS"),
         ("DSS", "cox", "DSS"),
         ("PFI", "cox", "PFI")]


def metric(kind, pt, y=None, dur=None, ev=None):
    if kind == "tau":
        return kendalltau(pt, y, variant="b").statistic
    return concordance_index(dur, -pt, ev)


def cell(sub, kind, col, B, rng):
    """Observed metric plus its label-shuffled null distribution."""
    if kind == "tau":
        m = sub["mean_pseudotime"].notna() & sub[col].notna()
        pt = sub.loc[m, "mean_pseudotime"].to_numpy()
        y = sub.loc[m, col].to_numpy()
        dur = ev = None
    else:
        tm = f"{col}.time"
        m = sub["mean_pseudotime"].notna() & sub[col].notna() & sub[tm].notna()
        pt = sub.loc[m, "mean_pseudotime"].to_numpy()
        y = None
        dur = sub.loc[m, tm].to_numpy()
        ev = sub.loc[m, col].to_numpy()

    obs = metric(kind, pt, y, dur, ev)
    null = np.empty(B)
    for b in range(B):
        i = rng.permutation(len(pt))
        null[b] = metric(kind, pt,
                         None if y is None else y[i],
                         None if dur is None else dur[i],
                         None if ev is None else ev[i])
    return obs, null, int(m.sum())


def plot(summary, out_path):
    tasks = [t[0] for t in TASKS]
    models = list(summary.groupby("model")["z"].mean().sort_values().index)  # worst at bottom
    n = len(models)
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 0.42 * n + 2.2),
                             facecolor=SURFACE, sharey=True)

    for ax, task in zip(np.atleast_1d(axes), tasks):
        s = summary[summary["endpoint"] == task].set_index("model").loc[models]
        ypos = np.arange(n)[::-1]
        ax.set_facecolor(SURFACE)
        # null 95% band
        ax.barh(ypos, s["null_hi"] - s["null_lo"], left=s["null_lo"], height=0.55,
                color=NULL_FILL, linewidth=0, zorder=2)
        ax.plot(s["null_mean"], ypos, ls="none", marker="|", ms=9,
                color=INK_MUTED, zorder=3)
        ax.plot(s["observed"], ypos, ls="none", marker="o", ms=8,
                mfc=SERIES_1, mec=SURFACE, mew=1.6, zorder=4)

        # Reserve a right-hand margin for the z column so it never sits on a dot.
        lo = min(s["null_lo"].min(), s["observed"].min())
        hi = max(s["null_hi"].max(), s["observed"].max())
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo - pad, hi + pad + (hi - lo) * 0.34)
        for y, (_, r) in zip(ypos, s.iterrows()):
            ax.annotate(f"z={r['z']:.1f}", xy=(0.995, y), xycoords=("axes fraction", "data"),
                        ha="right", va="center", fontsize=7.4, color=INK_SECONDARY)
        ax.set_yticks(ypos); ax.set_yticklabels(models, fontsize=8.8, color=INK_PRIMARY)
        ax.set_title(task, fontsize=9.6, color=INK_PRIMARY, fontweight="semibold", pad=6)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(labelsize=8, colors=INK_SECONDARY, length=0)
        ax.set_ylim(-0.7, n - 0.3)

    fig_h = fig.get_size_inches()[1]
    fig.text(0.006, 1 - 0.28 / fig_h, "KIRC — observed association vs label-shuffled null",
             ha="left", va="top", fontsize=12.5, color=INK_PRIMARY, fontweight="semibold")
    fig.text(0.006, 1 - 0.54 / fig_h,
             "Grey bar = 95% of the null (endpoint permuted across patients, B=2000); "
             "tick = null mean; dot = observed. z = SDs above the null.",
             ha="left", va="top", fontsize=8.4, color=INK_SECONDARY)
    fig.text(0.006, 0.10 / fig_h,
             "Left panels are Kendall τ_b, right three are Harrell's C. "
             "'stage' is the most image-adjacent endpoint and the least independent of the grade ordering.",
             ha="left", va="bottom", fontsize=8.0, color=INK_MUTED)
    fig.tight_layout(rect=[0, 0.26 / fig_h, 1, 1 - 0.72 / fig_h])
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(config.PAPER_RESULTS_DIR))
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--plots", default=None,
                   help="Folder for the null-band figure (default: <results>/plots)")
    args = p.parse_args()

    res = Path(args.results)
    d = pd.read_csv(res / "dpt_patient_level.csv")
    rng = np.random.default_rng(config.SEED)

    rows, draws = [], []
    for model in config.MODELS:
        sub = d[d["model"] == model]
        for label, kind, col in TASKS:
            obs, null, n = cell(sub, kind, col, args.n_perm, rng)
            sd = null.std(ddof=1)
            rows.append({
                "model": model, "endpoint": label, "metric": "tau_b" if kind == "tau" else "C-index",
                "n": n, "observed": obs, "null_mean": null.mean(), "null_sd": sd,
                "null_lo": np.percentile(null, 2.5), "null_hi": np.percentile(null, 97.5),
                "z": (obs - null.mean()) / sd,
                "p_perm": (1 + np.sum(null >= obs)) / (args.n_perm + 1),
            })
            draws.append(pd.DataFrame({"model": model, "endpoint": label, "draw": null}))
            print(f"  {model:9} {label:11} obs={obs:+.3f}  null={null.mean():+.3f}"
                  f"±{sd:.3f}  z={rows[-1]['z']:5.1f}  p={rows[-1]['p_perm']:.4f}")

    summary = pd.DataFrame(rows)
    summary["q_BH"] = multipletests(summary["p_perm"], method="fdr_bh")[1]
    summary["sig_FDR05"] = summary["q_BH"] < 0.05
    summary.to_csv(res / "label_shuffle_null.csv", index=False)
    pd.concat(draws, ignore_index=True).to_csv(res / "label_shuffle_null_draws.csv", index=False)

    out = Path(args.plots) if args.plots else res / "plots"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "kirc_label_shuffle_null.png"
    plot(summary, path)

    print("\nz (SDs above the label-shuffled null), models x endpoints:")
    piv = summary.pivot(index="model", columns="endpoint", values="z")[[t[0] for t in TASKS]]
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean", ascending=False).round(1).to_string())
    print(f"\nWrote {res}/label_shuffle_null.csv and {path}")


if __name__ == "__main__":
    main()
