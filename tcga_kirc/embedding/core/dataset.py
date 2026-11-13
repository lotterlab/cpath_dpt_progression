import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import NUM_WORKERS, TCGA_ROOT, TEST_RUN_PER_LABEL

# Columns carried from the manifest into every embedding row.
META_COLUMNS = ["key", "label", "split", "patient", "tss", "sample_type"]


def load_manifest(cohort, test_run: bool = False) -> pd.DataFrame:
    """Load a cohort's staged patch manifest, optionally trimmed for a test run."""
    if not cohort.manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {cohort.manifest_path}. Stage the cohort first."
        )

    df = pd.read_parquet(cohort.manifest_path)

    unknown = set(df["label"]) - set(cohort.labels)
    if unknown:
        raise ValueError(f"Manifest contains unexpected labels: {sorted(unknown)}")

    if test_run:
        df = df.groupby("label", group_keys=False).head(TEST_RUN_PER_LABEL)

    return df.reset_index(drop=True)


class ManifestPatchDataset(Dataset):
    """Serves transformed patches listed in the manifest.

    __getitem__ returns (tensor, row_index) so the caller can map results back
    to manifest metadata. Unreadable files yield (None, row_index) and are
    dropped by the collate function.
    """

    def __init__(self, manifest: pd.DataFrame, transform):
        self.paths = manifest["path"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = TCGA_ROOT / self.paths[idx]
        try:
            with Image.open(path) as img:
                return self.transform(img.convert("RGB")), idx
        except Exception as e:  # noqa: BLE001 - one bad patch must not kill the run
            print(f"      Error loading {path}: {e}")
            return None, idx


def _collate(batch):
    kept = [(t, i) for t, i in batch if t is not None]
    if not kept:
        return None, []
    tensors, indices = zip(*kept)
    return torch.stack(tensors), list(indices)


def build_loader(manifest: pd.DataFrame, transform, batch_size: int) -> DataLoader:
    return DataLoader(
        ManifestPatchDataset(manifest, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=_collate,
        pin_memory=True,
    )
