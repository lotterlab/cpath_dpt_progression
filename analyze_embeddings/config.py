"""Configuration for the analysis pipeline.

Run every script from the analyze_embeddings/ directory; paths below are
relative to it.
"""
import os
from datetime import datetime
from typing import List, TypedDict
from pathlib import Path

# ── Results and figures ───────────────────────────────────────────────────────
# RESULTS_RUN names the results folder that plotting scripts read from and the
# plots folder they write to. "paper" holds the cached results behind every
# figure and table in the paper. Data generation scripts write to a new
# timestamped folder (see get_output_dir); to plot from one, set RESULTS_RUN to
# its name or export RESULTS_RUN=<folder>.
RESULTS_DIR = Path("results")
RESULTS_RUN = os.environ.get("RESULTS_RUN", "paper")

RESULTS_RUN_DIR = RESULTS_DIR / RESULTS_RUN
PLOTS_OUTPUT_DIR = Path("plots") / RESULTS_RUN

FULL_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "full_manifold_evaluation.csv"
NULL_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "null_manifold_evaluation.csv"
PERMUTATION_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "stage_permutation_specificity.csv"
GENERALIZABILITY_RESULTS_OUTPUT_PATH = RESULTS_RUN_DIR / "generalizability_results.csv"
K_SWEEP_RESULTS_PATH = RESULTS_RUN_DIR / "hyperparam_sweep_k_results.csv"
BASELINE_RESULTS_PATH = RESULTS_RUN_DIR / "baseline_evaluation.csv"
ORDERING_AGREEMENT_PATH = RESULTS_RUN_DIR / "ordering_agreement.csv"
CELL_COUNTS_DIR = RESULTS_RUN_DIR / "cell_counts"
SPLIT_HALF_DIR = RESULTS_RUN_DIR / "split_half"

# ── Data locations ────────────────────────────────────────────────────────────
# Embeddings are read from Google Cloud Storage at
#   gs://<bucket>/<prefix>/<model>/*.parquet
# as written by generate_embeddings/. The defaults are the authors' buckets,
# which are not public: point these at your own buckets by editing the values
# or exporting the environment variables. See the README ("Data access").
SPIDER_SKIN_BUCKET = os.environ.get("SPIDER_SKIN_BUCKET", "spider-skin")
SPIDER_COLORECTAL_BUCKET = os.environ.get("SPIDER_COLORECTAL_BUCKET", "spider-colorectal")
SPIDER_BREAST_BUCKET = os.environ.get("SPIDER_BREAST_BUCKET", "spider-breast")
EMBEDDING_PREFIX = "embeddings/computed"

# Scripts that show or segment H&E patches read the SPIDER images from local
# disk (the unpacked datasets, same layout generate_embeddings/ reads). Most of
# them also accept --data_dir to override the folder for a single run.
SPIDER_IMAGE_ROOT = Path(os.environ.get("SPIDER_IMAGE_ROOT", Path.home() / "data"))
LOCAL_IMAGE_DIRS = {
    SPIDER_COLORECTAL_BUCKET: SPIDER_IMAGE_ROOT / "spider-col",
    SPIDER_SKIN_BUCKET: SPIDER_IMAGE_ROOT / "spider-skin",
    SPIDER_BREAST_BUCKET: SPIDER_IMAGE_ROOT / "spider-breast",
}


def get_output_dir() -> Path:
    """Create and return a new timestamped output directory, results/YYYY-MM-DD_HHMMSS/.

    Data generation scripts write here so a new run never overwrites the cached
    paper results. Point RESULTS_RUN at the new folder to plot from it.
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = RESULTS_DIR / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class ProgressionConfig(TypedDict):
    name: str
    bucket: str
    prefix: str
    classes: List[str]
    root_class: str
    image_subdir: str

# The six foundation models evaluated in the paper.
EXPECTED_MODELS = [
    "virchow2",
    "uni2",
    "conch",
    "gigapath",
    "musk",
    "dinov2"
]

# The four disease progressions. "classes" is the ordinal stage order and
# "root_class" the earliest stage (the DPT root is its medoid). "image_subdir"
# is the SPIDER image layout used for that dataset.
PROGRESSIONS: List[ProgressionConfig] = [
    {
        "name": "SCC",
        "bucket": SPIDER_SKIN_BUCKET,
        "prefix": EMBEDDING_PREFIX,
        "classes": [
            "Epidermis",
            "Actinic keratosis",
            "Carcinoma in situ",
            "Squamous cell carcinoma"
        ],
        "root_class": "Epidermis",
        "image_subdir": "imagenet_context1"
    },
    {
        "name": "CRC-Conventional",
        "bucket": SPIDER_COLORECTAL_BUCKET,
        "prefix": EMBEDDING_PREFIX,
        "classes": [
            "Adenoma low grade",
            "Adenoma high grade",
            "Adenocarcinoma low grade",
            "Adenocarcinoma high grade"
        ],
        "root_class": "Adenoma low grade",
        "image_subdir": "imagenet"
    },
    {
        "name": "CRC-Serrated",
        "bucket": SPIDER_COLORECTAL_BUCKET,
        "prefix": EMBEDDING_PREFIX,
        "classes": [
            "Hyperplastic polyp",
            "Sessile serrated lesion",
            "Adenocarcinoma high grade"
        ],
        "root_class": "Hyperplastic polyp",
        "image_subdir": "imagenet"
    },
    {
        "name": "BDC",
        "bucket": SPIDER_BREAST_BUCKET,
        "prefix": EMBEDDING_PREFIX,
        "classes": [
            "Ductal carcinoma in situ (low-grade)",
            "Ductal carcinoma in situ (high-grade)",
            "Invasive non-special type carcinoma"
        ],
        "root_class": "Ductal carcinoma in situ (low-grade)",
        "image_subdir": "imagenet-central"
    }
]


def get_progressions(names: List[str] = None) -> List[ProgressionConfig]:
    """Return progression configs filtered by name. If names is None, return all."""
    if names is None:
        return PROGRESSIONS
    available = {p["name"]: p for p in PROGRESSIONS}
    result = []
    for name in names:
        if name not in available:
            raise ValueError(
                f"Unknown progression '{name}'. Available: {list(available.keys())}"
            )
        result.append(available[name])
    return result


EVALUATION = {
    "n_per_class": 1000,
    "max_per_slide": 50,
    "seed": 42,
    "embedding_type": "final_embedding",
    # evaluate_dpt: bootstrap resampling iterations for metric confidence intervals
    "n_bootstrap_ci": 100,
    # generate_null_distribution: label-shuffle permutations for null tau distribution
    "n_null_permutations": 10,
    # evaluate_generalizability: repeated few-shot sampling trials per shot count
    "n_fewshot_trials": 10,
    # evaluate_generalizability: shots per class for few-shot probing
    "fewshot_n_shots": [5, 10, 20],
}

TEST_EVALUATION = {
    "n_per_class": 50,
    "max_per_slide": 10,
    "seed": 42,
    "embedding_type": "final_embedding",
    "n_bootstrap_ci": 2,
    "n_null_permutations": 2,
    "n_fewshot_trials": 2,
    "fewshot_n_shots": [5],
}

DPT = {
    "n_neighbors": 100,
    "n_diffusion_components": 10,
}

PATCH_EMBEDDINGS = [
    "patch_mean_0.125",
    "patch_mean_0.375",
    "patch_mean_0.625",
    "patch_mean_0.875",
    "patch_mean_1.0",
]

CLS_EMBEDDINGS = [
    "cls_0.125",
    "cls_0.375",
    "cls_0.625",
    "cls_0.875",
    "cls_1.0",
]

REGISTER_EMBEDDINGS = [
    "register_mean_0.125",
    "register_mean_0.375",
    "register_mean_0.625",
    "register_mean_0.875",
    "register_mean_1.0",
]

FINAL_EMBEDDING = "final_embedding"
