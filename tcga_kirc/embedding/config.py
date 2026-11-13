"""Configuration for the TCGA-KIRC CLS-token embedding run.

Mirrors generate_embeddings/config.py, with two differences: patches are read
from the staged manifest rather than an ImageNet-style tree, and shards are
written to the local filesystem rather than GCS.
"""

import sys
from pathlib import Path

# ===============================
# Paths
# ===============================
EMBEDDING_DIR = Path(__file__).resolve().parent
TCGA_ROOT = EMBEDDING_DIR.parent
REPO_ROOT = TCGA_ROOT.parent
GENERATE_EMBEDDINGS_DIR = REPO_ROOT / "generate_embeddings"

# Model adapters are shared with the SPIDER experiment rather than copied, so a
# fix to an adapter benefits both. Appended (never inserted) so that this
# package's own `config` module keeps priority over generate_embeddings/config.py.
if str(GENERATE_EMBEDDINGS_DIR) not in sys.path:
    sys.path.append(str(GENERATE_EMBEDDINGS_DIR))

# ===============================
# Cohorts
# ===============================
# Each cohort names its manifest, its expected labels, and where embeddings go.
# Adding a cohort here is all that is needed to embed a new tumour type.
class CohortConfig:
    def __init__(self, name, dataset_name, manifest, labels, output_dir):
        self.name = name
        self.dataset_name = dataset_name
        self.manifest_path = TCGA_ROOT / "data" / manifest
        self.labels = labels
        self.output_root = TCGA_ROOT / "data" / output_dir

    def get_path(self, test_run: bool) -> Path:
        return self.output_root / "test_run" if test_run else self.output_root


COHORTS = {
    "kirc": CohortConfig(
        name="kirc",
        dataset_name="tcga-ut-kirc",
        manifest="manifest_kirc.parquet",
        labels=["Kidney_renal_clear_cell_carcinoma"],
        output_dir="embeddings_kirc",
    ),
}

DEFAULT_COHORT = "kirc"


def get_cohort(name: str) -> CohortConfig:
    if name not in COHORTS:
        raise ValueError(f"Unknown cohort '{name}'. Available: {list(COHORTS)}")
    return COHORTS[name]


# ===============================
# Constants
# ===============================
SHARD_SIZE = 10000

# JPEG decode + transform runs in worker processes so it overlaps GPU compute.
NUM_WORKERS = 8

# Mixed precision. The SPIDER pipeline runs fp32; on a T4 that caps ViT-H
# throughput at ~10 img/s, which is ~8 hours for all six models. fp16 autocast
# cuts that several-fold. Embeddings are therefore NOT bit-identical to the
# SPIDER run -- see README section 4. Disable with main.py --fp32.
USE_AMP = True
AMP_DTYPE = "float16"

# Patches per label under --test-run.
TEST_RUN_PER_LABEL = 10

EXPECTED_MODELS = [
    "virchow2",
    "uni2",
    "conch",
    "gigapath",
    "musk",
    "dinov2",
]
