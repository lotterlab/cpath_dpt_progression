#!/usr/bin/env python
"""Fidelity under the cohort's own class distribution, and across random halves.

Two questions the fixed-weight imbalance sweep does not answer:

  1. What is tau when classes are sampled at their EMPIRICAL frequency -- the
     mix a cohort actually has -- rather than forced to be equal?
  2. Is tau reproducible? Split the cohort into two disjoint halves and score
     both, under each sampling scheme.

Conditions per replicate:
    full_empirical     whole cohort, classes in their natural proportions
    full_balanced      whole cohort, equal quota per class (the published scheme)
    halfA_empirical    half A, natural proportions
    halfA_balanced     half A, equal quota
    halfB_empirical    half B, natural proportions
    halfB_balanced     half B, equal quota

The split is BY SLIDE, not by patch: patches from one slide are highly
correlated, so a patch-level split would leak between halves. Each half draws
n_total/2 patches, so a half is half the data in both slides and patches.
Comparing a half to the full cohort therefore confounds cohort size with source
diversity; the clean comparisons are empirical-vs-balanced within a condition,
and halfA-vs-halfB within a scheme.

Every model sees the identical draws. Each seed is an independent slide split
AND an independent patch draw.

Usage:
    python generate_data/evaluate_split_half.py                          # SCC, all models
    python generate_data/evaluate_split_half.py --progression BDC --n-total 2700
    python generate_data/evaluate_split_half.py --test
"""

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from analysis.dpt import DPTConfig, DPTMetric, cohort_to_anndata, compute_dpt
from data.progression_embedding_dataset import ProgressionEmbeddingDataset, RegistryConfig

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["full_empirical", "full_balanced",
              "halfA_empirical", "halfA_balanced",
              "halfB_empirical", "halfB_balanced"]


def quota_from_counts(counts: dict, classes: list, n_total: int) -> dict:
    """Scale available-patch counts to a quota summing exactly to n_total."""
    avail = np.array([counts.get(c, 0) for c in classes], dtype=float)
    raw = avail / avail.sum() * n_total
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - q))[: n_total - q.sum()]:
        q[i] += 1
    return dict(zip(classes, q.tolist()))


def balanced_quota(classes: list, n_total: int) -> dict:
    base, rem = divmod(n_total, len(classes))
    q = {c: base for c in classes}
    for c in classes[:rem]:
        q[c] += 1
    return q


def draw_conditions(dataset, classes, n_total, seed, max_per_slide):
    """One replicate: split slides in half, then draw all six cohorts."""
    rng = np.random.default_rng(seed)
    slides = np.array(dataset.slide_ids())
    rng.shuffle(slides)
    cut = len(slides) // 2
    halfA, halfB = set(slides[:cut]), set(slides[cut:])

    emp_full = quota_from_counts(dataset.class_patch_counts(), classes, n_total)
    bal_full = balanced_quota(classes, n_total)
    half_n = n_total // 2

    out = {}
    for name, quota, sl in (
        ("full_empirical", emp_full, None),
        ("full_balanced", bal_full, None),
        ("halfA_empirical", quota_from_counts(dataset.class_patch_counts(), classes, half_n), halfA),
        ("halfA_balanced", balanced_quota(classes, half_n), halfA),
        ("halfB_empirical", quota_from_counts(dataset.class_patch_counts(), classes, half_n), halfB),
        ("halfB_balanced", balanced_quota(classes, half_n), halfB),
    ):
        ids = dataset.sample_patch_ids(n_per_class=quota, max_per_slide=max_per_slide,
                                       seed=seed, slides=sl)
        out[name] = (quota, ids, n_total if sl is None else half_n)
    return out, len(halfA), len(halfB)


def main():
    p = argparse.ArgumentParser(description="Empirical vs balanced sampling, and split-half reproducibility")
    p.add_argument("--progression", default="SCC")
    p.add_argument("--models", nargs="+", default=config.EXPECTED_MODELS)
    p.add_argument("--n-total", type=int, default=4000)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--embedding-type", default=config.FINAL_EMBEDDING)
    p.add_argument("--out", default=None)
    p.add_argument("--test", action="store_true")
    args = p.parse_args()

    eval_config = config.TEST_EVALUATION if args.test else config.EVALUATION
    n_total = 400 if args.test else args.n_total
    n_seeds = 2 if args.test else args.n_seeds
    max_per_slide = eval_config["max_per_slide"]
    base_seed = eval_config["seed"]

    prog = config.get_progressions([args.progression])[0]
    classes, root_class = prog["classes"], prog["root_class"]
    dataset = ProgressionEmbeddingDataset(RegistryConfig(
        bucket=prog["bucket"], prefix=prog["prefix"], ordered_classes=classes,
        models=config.EXPECTED_MODELS, progression_name=args.progression,
        scan_all_models=False))

    counts = dataset.class_patch_counts()
    total_avail = sum(counts.get(c, 0) for c in classes)
    print(f"progression : {args.progression}  ({len(classes)} classes)")
    print(f"models      : {args.models}")
    print(f"n_total     : {n_total} (halves draw {n_total // 2}), {n_seeds} seeds")
    print(f"slides      : {len(dataset.slide_ids())} distinct, split ~50/50 per seed")
    print("empirical class distribution (available patches):")
    for c in classes:
        print(f"    {c:34} {counts.get(c,0):>7,}  ({counts.get(c,0)/total_avail*100:5.1f}%)")
    print(f"    -> empirical quota at n={n_total}: "
          f"{list(quota_from_counts(counts, classes, n_total).values())}")
    print(f"    -> balanced  quota at n={n_total}: "
          f"{list(balanced_quota(classes, n_total).values())}\n")

    dpt_config = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics={DPTMetric.TAU, DPTMetric.SPECTRAL_GAP},
        ordered_classes=classes)

    draws = {}
    for s in tqdm(range(n_seeds), desc="sampling replicates", unit="seed"):
        draws[base_seed + s] = draw_conditions(dataset, classes, n_total,
                                               base_seed + s, max_per_slide)
    union = set().union(*(ids for d, _, _ in draws.values() for _, ids, _ in d.values()))
    print(f"union of all draws: {len(union):,} distinct patches to fetch per model\n")

    rows = []
    bar = tqdm(total=len(args.models) * len(CONDITIONS) * n_seeds,
               desc="scoring", unit="dpt", smoothing=0.05)
    for model in args.models:
        bar.set_postfix_str(f"{model} · fetching embeddings")
        dataset.load_model_into_memory(model, union)
        for cond in CONDITIONS:
            taus = []
            for s in range(n_seeds):
                seed = base_seed + s
                bar.set_postfix_str(f"{model} · {cond} · seed {seed}")
                cond_map, nA, nB = draws[seed]
                quota, ids, target = cond_map[cond]
                cohort = dataset.get_cohort(ids, model=model,
                                            embedding_type=args.embedding_type)
                bar.update(1)
                if cohort.empty:
                    continue
                adata = cohort_to_anndata(cohort, classes)
                res = compute_dpt(adata, root_class, dpt_config)
                actual = cohort["class"].value_counts().to_dict()
                row = {"progression": args.progression, "model": model,
                       "condition": cond, "seed": seed, "tau": res.tau,
                       "p_value": res.p_value, "spectral_gap": res.spectral_gap,
                       "n_requested": target, "n_actual": int(len(cohort)),
                       "slides_halfA": nA, "slides_halfB": nB}
                for c in classes:
                    row[f"req__{c}"] = quota[c]
                    row[f"got__{c}"] = int(actual.get(c, 0))
                rows.append(row)
                taus.append(res.tau)
            if taus:
                na = rows[-1]["n_actual"]
                short = " SHORTFALL" if na < rows[-1]["n_requested"] else ""
                tqdm.write(f"  {model:9} {cond:16} n={na:>5}  "
                           f"tau={np.mean(taus):+.4f} "
                           f"[{np.min(taus):+.4f},{np.max(taus):+.4f}]{short}")
        dataset.clear_cache()
    bar.close()

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results.")
        return 1
    out_dir = Path(args.out) if args.out else config.get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "split_half.csv", index=False)

    print("\n" + "=" * 84)
    print(f"{args.progression}: EMPIRICAL vs BALANCED sampling, full cohort and random halves")
    print("=" * 84)
    piv = (df.groupby(["model", "condition"])
             .tau.agg(["mean", "min", "max"]).reset_index())
    order = (df[df.condition == "full_balanced"].groupby("model").tau.mean()
             .sort_values(ascending=False).index.tolist())
    print(f"\n{'model':10}" + "".join(f"{c:>24}" for c in CONDITIONS))
    for m in order:
        cells = ""
        for c in CONDITIONS:
            r = piv[(piv.model == m) & (piv.condition == c)]
            cells += (f"{r['mean'].iloc[0]:.3f} [{r['min'].iloc[0]:.3f},{r['max'].iloc[0]:.3f}]".rjust(24)
                      if len(r) else "n/a".rjust(24))
        print(f"{m:10}{cells}")

    print("\nhalf-to-half agreement (|tau_A - tau_B| within a seed, mean over seeds):")
    for scheme in ("empirical", "balanced"):
        w = df.pivot_table(index=["model", "seed"], columns="condition", values="tau")
        d = (w[f"halfA_{scheme}"] - w[f"halfB_{scheme}"]).abs().groupby("model").mean()
        print(f"  {scheme:10} " + "  ".join(f"{m}={d[m]:.3f}" for m in order))

    print(f"\nWrote {out_dir}/split_half.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
