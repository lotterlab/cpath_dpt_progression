#!/usr/bin/env python3
"""KIRC grade-progression DPT, rooted at the G1 medoid.

Runs the shared DPT procedure (analyze_embeddings/analysis/dpt.py) on a
KIRC primary-tumour cohort ordered G1 < G2 < G3 < G4, then correlates
patient-level mean pseudotime against stage, stemness, survival, and
aneuploidy.

The paper reports stage, aneuploidy score and disease-specific survival
(Table 6); the other endpoints are computed alongside them.

Usage (from tcga_kirc/analysis/):
    python run_dpt.py                                 # -> results/<timestamp>/
    python run_dpt.py --models uni2 --out results/tmp
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config  # noqa: F401  -- wires up sys.path
import cohort as cohort_mod
import metrics as metrics_mod

from analysis.dpt import DPTConfig, DPTMetric, cohort_to_anndata, compute_dpt


def run_model(model: str, cohort_df: pd.DataFrame) -> pd.DataFrame:
    emb = cohort_mod.load_embeddings(model, set(cohort_df["key"]))
    merged = (
        cohort_df[["key", "patient", "grade", "split", "tss"]]
        .merge(emb[["key", "slide_id", "patch_id", "embedding"]], on="key", how="inner")
        .reset_index(drop=True)
    )
    if len(merged) != len(cohort_df):
        raise RuntimeError(f"{model}: matched {len(merged)} of {len(cohort_df)} patches")

    merged["class"] = merged["grade"]
    adata = cohort_to_anndata(merged, config.ORDERED_GRADES)

    dpt_config = DPTConfig(
        n_neighbors=config.DPT["n_neighbors"],
        n_diffusion_components=config.DPT["n_diffusion_components"],
        metrics={DPTMetric.TAU, DPTMetric.SPECTRAL_GAP},
        ordered_classes=config.ORDERED_GRADES,
    )
    result = compute_dpt(adata, config.ROOT_GRADE, dpt_config)

    if "dpt_pseudotime" not in adata.obs:
        raise RuntimeError(f"{model}: DPT produced no pseudotime")

    merged["dpt_pseudotime"] = np.asarray(adata.obs["dpt_pseudotime"].values, dtype=float)
    merged["model"] = model

    n_bad = int((~np.isfinite(merged["dpt_pseudotime"])).sum())
    root_grade = merged.loc[int(adata.uns["iroot"]), "grade"]
    print(f"    patch-level tau vs grade = {result.tau:.4f} (p={result.p_value:.2e}), "
          f"root idx = {adata.uns['iroot']} (grade {root_grade}), non-finite = {n_bad}")

    return merged[["model", "key", "patient", "slide_id", "patch_id", "grade",
                   "split", "tss", "dpt_pseudotime"]]


def aggregate_patients(patch_df: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    finite = patch_df[np.isfinite(patch_df["dpt_pseudotime"])]
    agg = (finite.groupby(["model", "patient"])
           .agg(mean_pseudotime=("dpt_pseudotime", "mean"),
                median_pseudotime=("dpt_pseudotime", "median"),
                max_pseudotime=("dpt_pseudotime", "max"),
                n_patches=("dpt_pseudotime", "size"),
                grade=("grade", "first"))
           .reset_index())
    return agg.merge(clinical.drop(columns=["grade"]), on="patient", how="left")


def summarize(patient_df: pd.DataFrame, models: list, agg: str = "mean") -> pd.DataFrame:
    col = f"{agg}_pseudotime"
    rows = []
    for model in models:
        d = patient_df[patient_df["model"] == model]
        pt = d[col]

        stage = metrics_mod.tau_b(pt, d["stage_ordinal"])
        sten = metrics_mod.tau_b(pt, d["mRNAsi"])
        os_ = metrics_mod.c_index(pt, d["OS.time"], d["OS"])
        dss = metrics_mod.c_index(pt, d["DSS.time"], d["DSS"])
        pfi = metrics_mod.c_index(pt, d["PFI.time"], d["PFI"])
        aneu = metrics_mod.tau_b(pt, d["ANEUPLOIDY_SCORE"])

        rows.append({
            "model": model, "agg": agg, "n_patients": len(d),
            "stage_tau_b": stage["tau"], "stage_p": stage["p"], "stage_n": stage["n"],
            "mrnasi_tau_b": sten["tau"], "mrnasi_p": sten["p"], "mrnasi_n": sten["n"],
            "os_c_index": os_["c_index"], "os_n": os_["n"], "os_events": os_["n_events"],
            "dss_c_index": dss["c_index"], "dss_n": dss["n"], "dss_events": dss["n_events"],
            "pfi_c_index": pfi["c_index"], "pfi_n": pfi["n"], "pfi_events": pfi["n_events"],
            "aneuploidy_tau_b": aneu["tau"], "aneuploidy_p": aneu["p"],
            "aneuploidy_n": aneu["n"],
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="KIRC grade-progression DPT (G1-rooted)")
    p.add_argument("--models", nargs="+", default=config.MODELS, choices=config.MODELS)
    p.add_argument("--agg", nargs="+", choices=["mean", "median", "max"],
                   default=["mean"])
    p.add_argument("--patches-per-patient", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (
        config.RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    cohort_df = cohort_mod.build_cohort(n_per_patient=args.patches_per_patient)
    clinical = cohort_mod.load_clinical()
    print("=" * 72)

    frames = []
    for model in args.models:
        print(f"\n  Model: {model}")
        try:
            frames.append(run_model(model, cohort_df))
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")

    if not frames:
        print("No models succeeded.")
        return 1

    patch_df = pd.concat(frames, ignore_index=True)
    patient_df = aggregate_patients(patch_df, clinical)
    models_done = [m for m in args.models if m in set(patch_df["model"])]
    summary = pd.concat([summarize(patient_df, models_done, a) for a in args.agg],
                        ignore_index=True)

    patch_df.to_parquet(out_dir / "dpt_patch_level.parquet", index=False)
    patient_df.to_csv(out_dir / "dpt_patient_level.csv", index=False)
    summary.to_csv(out_dir / "summary_table.csv", index=False)

    for agg in args.agg:
        print("\n" + "=" * 72)
        print(f"KIRC -- patient-level {agg.upper()} pseudotime (G1-rooted) "
              f"vs clinical endpoints")
        print("=" * 72)
        d = summary[summary["agg"] == agg]
        print(d[["model", "stage_tau_b", "mrnasi_tau_b", "aneuploidy_tau_b",
                 "os_c_index", "dss_c_index", "pfi_c_index"]]
              .to_markdown(index=False, floatfmt=".3f"))
        print("\n  p-values:")
        print(d[["model", "stage_p", "mrnasi_p", "aneuploidy_p"]]
              .to_markdown(index=False, floatfmt=".2e"))

    r = summary.iloc[0]
    print(f"\n  n: stage {r['stage_n']}, mRNAsi {r['mrnasi_n']}, "
          f"OS {r['os_n']} ({r['os_events']} events), "
          f"DSS {r['dss_n']} ({r['dss_events']} events), "
          f"PFI {r['pfi_n']} ({r['pfi_events']} events), "
          f"aneuploidy {r['aneuploidy_n']}")
    print(f"\nWrote:\n  {out_dir}/dpt_patch_level.parquet"
          f"\n  {out_dir}/dpt_patient_level.csv"
          f"\n  {out_dir}/summary_table.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
