"""Configuration for the TCGA-KIRC grade-progression DPT experiment (Appendix J).

Same DPT procedure as the SPIDER progressions -- the code is imported from
../../analyze_embeddings/analysis/dpt.py, not reimplemented. Only the cohort
definition and the endpoint set differ.
"""

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
TCGA_ROOT = ANALYSIS_DIR.parent
REPO_ROOT = TCGA_ROOT.parent
ANALYZE_EMBEDDINGS_DIR = REPO_ROOT / "analyze_embeddings"

# Reuse the shared DPT module. Appended so this folder's own `config` keeps
# priority over analyze_embeddings/config.py.
if str(ANALYZE_EMBEDDINGS_DIR) not in sys.path:
    sys.path.append(str(ANALYZE_EMBEDDINGS_DIR))

MANIFEST_PATH = TCGA_ROOT / "data" / "manifest_kirc.parquet"
EMBEDDINGS_DIR = TCGA_ROOT / "data" / "embeddings_kirc"
CLINICAL_DIR = TCGA_ROOT / "data" / "clinical"
RESULTS_DIR = ANALYSIS_DIR / "results"
# Cached results behind Table 6. New runs write to results/<timestamp>/.
PAPER_RESULTS_DIR = RESULTS_DIR / "paper"

MODELS = ["uni2", "virchow2", "dinov2", "conch", "gigapath", "musk"]


# ===============================
# Cohort definition
# ===============================
KIRC_LABEL = "Kidney_renal_clear_cell_carcinoma"

# Keep primary solid tumour only. One patient also has sample_type 05
# (additional new primary, 30 patches); they keep their 01 patches.
KEPT_SAMPLE_TYPES = {"01"}

# Literal placeholder strings in the CDR sheet, normalized to NaN BEFORE any
# filtering. GX is an explicit "grade cannot be assessed".
MISSING_TOKENS = {
    "[Not Available]",
    "[Unknown]",
    "[Not Applicable]",
    "[Discrepancy]",
    "GX",
}

# Progression order and DPT stage ordering.
ORDERED_GRADES = ["G1", "G2", "G3", "G4"]
ROOT_GRADE = "G1"

# 375 of 379 cohort patients have exactly 30 patches, so at this cap the
# sampling step is nearly a no-op -- only the two patients with 60 and 90
# patches are actually subsampled.
N_PATCHES_PER_PATIENT = 30
SEED = 42

# AJCC stage, anatomic extent of disease, increasing.
STAGE_ORDINAL = {
    "Stage I": 0,
    "Stage II": 1,
    "Stage III": 2,
    "Stage IV": 3,
}

EXPECTED_PATIENTS = 379


# ===============================
# DPT parameters
# ===============================
# Identical to analyze_embeddings/config.py DPT; cosine distance and the
# 'gauss' kernel are set inside analysis/dpt.compute_dpt. scanpy exposes no
# alpha parameter (see ../README.md section 5).
DPT = {
    "n_neighbors": 100,
    "n_diffusion_components": 10,
}
