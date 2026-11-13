"""Stage the TCGA-KIRC cohort and report its statistics.

Builds data/manifest_kirc.parquet plus the KIRC clinical tables, and prints the
distributions needed to design the pseudotime experiment (sample types, grade,
stage, patches per patient, root feasibility).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLINICAL = ROOT / "data" / "clinical"

LABEL = "Kidney_renal_clear_cell_carcinoma"
CDR_TYPE = "KIRC"

# Literal placeholder strings in the CDR sheet -- not NaN.
MISSING_TOKENS = {"[Not Available]", "[Unknown]", "[Not Applicable]",
                  "[Discrepancy]", "GX"}

CDR_COLS = ["bcr_patient_barcode", "histological_grade", "ajcc_pathologic_tumor_stage",
            "OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time",
            "age_at_initial_pathologic_diagnosis"]

GRADES = ["G1", "G2", "G3", "G4"]
STAGES = ["Stage I", "Stage II", "Stage III", "Stage IV"]


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def normalize(series: pd.Series) -> pd.Series:
    """Placeholder strings -> NaN. Applied BEFORE any filtering."""
    return series.where(~series.isin(MISSING_TOKENS))


def collapse_stage(series: pd.Series) -> pd.Series:
    """'Stage IIIA' -> 'Stage III'. Substage letters are stripped."""
    return series.str.replace(r"^(Stage (?:IV|III|II|I))[A-C]?$", r"\1", regex=True)


def stage_mrnasi() -> pd.DataFrame:
    """Per-patient stemness index (mRNAsi) from the PanCanStem per-sample scores.

    The raw file has one column per sample; samples are collapsed to patients
    (first 12 barcode characters) by mean. Reuses clinical/mrnasi.csv if present.
    """
    path = CLINICAL / "mrnasi.csv"
    if path.exists():
        return pd.read_csv(path)
    scores = pd.read_csv(RAW / "StemnessScores_RNAexp.tsv", sep="\t", index_col=0).T
    scores["bcr_patient_barcode"] = scores.index.str[:12]
    mrnasi = (scores.groupby("bcr_patient_barcode")
              .agg(mRNAsi=("RNAss", "mean"), EREG_mRNAsi=("EREG.EXPss", "mean"),
                   n_samples=("RNAss", "size"))
              .reset_index())
    CLINICAL.mkdir(parents=True, exist_ok=True)
    mrnasi.to_csv(path, index=False)
    print(f"  -> {path}")
    return mrnasi


def build_manifest() -> pd.DataFrame:
    records = pd.read_json(ROOT / "data" / "kirc_patch_records.jsonl", lines=True,
                           dtype={c: str for c in
                                  ["key", "path", "label", "split", "patient",
                                   "tss", "sample_type"]})
    out = ROOT / "data" / "manifest_kirc.parquet"
    records.to_parquet(out, index=False)
    print(f"Wrote {out}  ({len(records):,} patches)")
    return records


def stage_clinical() -> pd.DataFrame:
    cdr = pd.read_excel(RAW / "TCGA-CDR-SupplementalTableS1.xlsx", sheet_name="TCGA-CDR")
    k = cdr.loc[cdr["type"] == CDR_TYPE, CDR_COLS].reset_index(drop=True)

    rule("2. RAW CDR VALUES BEFORE CLEANING (KIRC, n=%d)" % len(k))
    print("histological_grade:")
    print(k["histological_grade"].value_counts(dropna=False).to_string())
    print("\najcc_pathologic_tumor_stage:")
    print(k["ajcc_pathologic_tumor_stage"].value_counts(dropna=False).to_string())

    k["grade"] = normalize(k["histological_grade"])
    k["stage"] = collapse_stage(normalize(k["ajcc_pathologic_tumor_stage"]))

    print("\nAFTER normalizing placeholders to NaN and collapsing substages:")
    print(f"  grade non-null: {k['grade'].notna().sum()}/{len(k)}   "
          f"{k['grade'].value_counts().reindex(GRADES).to_dict()}")
    print(f"  stage non-null: {k['stage'].notna().sum()}/{len(k)}   "
          f"{k['stage'].value_counts().reindex(STAGES).to_dict()}")

    path = CLINICAL / "tcga_cdr_kirc.csv"
    k.to_csv(path, index=False)
    print(f"\n  -> {path}")
    return k


def stage_extras() -> pd.DataFrame:
    """ANEUPLOIDY_SCORE from the cBioPortal sample file, collapsed to patient."""
    sample = pd.read_csv(RAW / "cbioportal" / "kirc_data_clinical_sample.txt",
                         sep="\t", comment="#")
    have = [c for c in ("ANEUPLOIDY_SCORE", "FRACTION_GENOME_ALTERED",
                        "MSI_SENSOR_SCORE", "TMB_NONSYNONYMOUS")
            if c in sample.columns]
    missing = [c for c in ("ANEUPLOIDY_SCORE", "FRACTION_GENOME_ALTERED")
               if c not in sample.columns]

    rule("2b. OPTIONAL cBioPortal COLUMNS")
    print(f"  present : {have}")
    print(f"  ABSENT  : {missing or 'none'}")

    extras = (sample[["PATIENT_ID"] + have]
              .groupby("PATIENT_ID", as_index=False).mean(numeric_only=True)
              .rename(columns={"PATIENT_ID": "bcr_patient_barcode"}))
    for c in have:
        print(f"  {c}: {extras[c].notna().sum()}/{len(extras)} patients")

    path = CLINICAL / "kirc_extra.csv"
    extras.to_csv(path, index=False)
    print(f"  -> {path}")
    return extras


def main():
    rule("1. COHORT — patch staging")
    mf = build_manifest()
    print(f"  label     : {LABEL}")
    print(f"  patients  : {mf['patient'].nunique()}")
    print(f"  patches   : {len(mf):,}")
    print(f"  splits    : {mf.groupby('split')['patient'].nunique().to_dict()}")

    print("\n  sample_type distribution (patches / patients):")
    for st, g in mf.groupby("sample_type"):
        print(f"    {st}: {len(g):>6,} patches  {g['patient'].nunique():>4} patients")

    primary = mf[mf["sample_type"] == "01"]
    print(f"\n  after excluding sample_type != '01': "
          f"{len(primary):,} patches, {primary['patient'].nunique()} patients")

    cdr = stage_clinical()
    extras = stage_extras()

    mrnasi = stage_mrnasi()

    rule("1b. JOIN RATES to the manifest cohort (primary tumour only)")
    pats = set(primary["patient"])
    n = len(pats)
    for name, ids in [("TCGA-CDR", set(cdr["bcr_patient_barcode"])),
                      ("  ...with a usable grade",
                       set(cdr.loc[cdr["grade"].notna(), "bcr_patient_barcode"])),
                      ("  ...with a usable stage",
                       set(cdr.loc[cdr["stage"].notna(), "bcr_patient_barcode"])),
                      ("mRNAsi", set(mrnasi["bcr_patient_barcode"])),
                      ("ANEUPLOIDY_SCORE",
                       set(extras.loc[extras["ANEUPLOIDY_SCORE"].notna(),
                                      "bcr_patient_barcode"])
                       if "ANEUPLOIDY_SCORE" in extras else set())]:
        j = pats & ids
        print(f"  {name:28} {len(j):>4}/{n} ({len(j)/n*100:.1f}%)")

    rule("3. PER-PATIENT PATCH COUNTS (primary tumour only)")
    counts = primary.groupby("patient").size()
    qs = np.percentile(counts, [25, 50, 75])
    print(f"  min {counts.min()}  Q1 {qs[0]:.0f}  median {qs[1]:.0f}  "
          f"Q3 {qs[2]:.0f}  max {counts.max()}   mean {counts.mean():.1f}")
    print(f"  distribution: {counts.value_counts().sort_index().head(12).to_dict()}")

    print("\n  cap -> cohort size and patients falling short:")
    for cap in [10, 15, 20, 25, 30, 40, 50]:
        short = int((counts < cap).sum())
        print(f"    {cap:>3}: N={int(np.minimum(counts, cap).sum()):>6,}  "
              f"short: {short:>3} ({short/len(counts)*100:.0f}%)")

    rule("4. ROOT FEASIBILITY — grade counts among cohort patients")
    grade = cdr.set_index("bcr_patient_barcode")["grade"]
    pat_grade = pd.Series(sorted(pats)).map(grade)
    print("  grade distribution among manifest patients:")
    counts_g = pat_grade.value_counts().reindex(GRADES, fill_value=0)
    for g in GRADES:
        npatch = primary[primary["patient"].map(grade) == g].shape[0]
        print(f"    {g}: {counts_g[g]:>4} patients  {npatch:>6,} patches")
    print(f"    missing/NaN: {int(pat_grade.isna().sum())} patients")


if __name__ == "__main__":
    main()
