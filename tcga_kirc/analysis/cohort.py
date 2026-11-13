"""Cohort construction for the KIRC grade-progression experiment."""

import numpy as np
import pandas as pd

import config


def _normalize(series: pd.Series) -> pd.Series:
    """Placeholder strings -> NaN. Applied before any filtering."""
    return series.where(~series.isin(config.MISSING_TOKENS))


def _collapse_stage(series: pd.Series) -> pd.Series:
    """'Stage IIIA' -> 'Stage III'. No substages occur in KIRC, but be safe."""
    return series.str.replace(r"^(Stage (?:IV|III|II|I))[A-C]?$", r"\1", regex=True)


def load_clinical() -> pd.DataFrame:
    """One row per KIRC patient: grade, stage ordinal, mRNAsi, aneuploidy, OS, PFI."""
    cdr = pd.read_csv(config.CLINICAL_DIR / "tcga_cdr_kirc.csv")
    cdr["grade"] = _normalize(cdr["histological_grade"])
    cdr["stage"] = _collapse_stage(_normalize(cdr["ajcc_pathologic_tumor_stage"]))
    cdr["stage_ordinal"] = cdr["stage"].map(config.STAGE_ORDINAL)

    mrnasi = (pd.read_csv(config.CLINICAL_DIR / "mrnasi.csv")
              [["bcr_patient_barcode", "mRNAsi"]])
    extra = pd.read_csv(config.CLINICAL_DIR / "kirc_extra.csv")
    extra_cols = [c for c in ("ANEUPLOIDY_SCORE",) if c in extra.columns]

    clinical = (
        cdr[["bcr_patient_barcode", "grade", "stage", "stage_ordinal",
             "OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time",
             "age_at_initial_pathologic_diagnosis"]]
        .merge(mrnasi, on="bcr_patient_barcode", how="left")
        .merge(extra[["bcr_patient_barcode"] + extra_cols],
               on="bcr_patient_barcode", how="left")
        .rename(columns={"bcr_patient_barcode": "patient"})
    )
    return clinical


def build_cohort(verbose: bool = True, n_per_patient: int = None) -> pd.DataFrame:
    """KIRC, primary tumour only, patients with a usable histological grade."""
    manifest = pd.read_parquet(config.MANIFEST_PATH)
    clinical = load_clinical()

    n0 = manifest["patient"].nunique()
    kirc = manifest[manifest["label"] == config.KIRC_LABEL]
    n1 = kirc["patient"].nunique()

    primary = kirc[kirc["sample_type"].isin(config.KEPT_SAMPLE_TYPES)]
    n2 = primary["patient"].nunique()

    grade = clinical.set_index("patient")["grade"]
    primary = primary.assign(grade=primary["patient"].map(grade))
    cohort = primary[primary["grade"].notna()].copy()
    n3 = cohort["patient"].nunique()

    if verbose:
        print("Cohort construction")
        print(f"  manifest patients                    : {n0}")
        print(f"  KIRC label                           : {n1}")
        print(f"  primary tumour (sample_type 01) only : {n2}")
        print(f"  with a usable histological grade     : {n3}")

    assert n3 == config.EXPECTED_PATIENTS, (
        f"Expected {config.EXPECTED_PATIENTS} patients, got {n3}")

    sampled = sample_patches(cohort, n_per_patient)

    if verbose:
        per = sampled.groupby("patient").size()
        print(f"  sampled patches                      : {len(sampled):,} "
              f"({per.min()}-{per.max()} per patient)")
        print(f"  grade distribution (patients)        : "
              f"{sampled.groupby('grade')['patient'].nunique().reindex(config.ORDERED_GRADES).to_dict()}")
        root_n = (sampled['grade'] == config.ROOT_GRADE).sum()
        print(f"  root pool ({config.ROOT_GRADE})                       : "
              f"{sampled.loc[sampled.grade == config.ROOT_GRADE, 'patient'].nunique()} patients, "
              f"{root_n} patches")

    return sampled


def sample_patches(cohort: pd.DataFrame, n_per_patient: int = None) -> pd.DataFrame:
    """Cap patches per patient, uniform at random, seeded and reproducible."""
    cap = config.N_PATCHES_PER_PATIENT if n_per_patient is None else n_per_patient
    rng = np.random.default_rng(config.SEED)

    def take(group: pd.DataFrame) -> pd.DataFrame:
        n = min(cap, len(group))
        idx = rng.choice(len(group), size=n, replace=False)
        return group.iloc[np.sort(idx)]

    ordered = cohort.sort_values(["patient", "key"]).reset_index(drop=True)
    return (ordered.groupby("patient", group_keys=False)[list(ordered.columns)]
            .apply(take).reset_index(drop=True))


def load_embeddings(model: str, keys: set) -> pd.DataFrame:
    shards = sorted((config.EMBEDDINGS_DIR / model).glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards for '{model}' in {config.EMBEDDINGS_DIR}")
    frames = [pd.read_parquet(s) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    return df[df["key"].isin(keys)]
