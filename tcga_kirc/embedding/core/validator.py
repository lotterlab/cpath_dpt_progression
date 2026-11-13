import torch

from config import TCGA_ROOT
from models import get_model_adapter

from .dataset import load_manifest


class DryRunValidator:
    """Checks the manifest and every requested model before a long run."""

    def __init__(self, device: str):
        self.device = device

    def validate_manifest(self, cohort, sample: int = 200) -> bool:
        print(f"📁 Checking manifest for cohort '{cohort.name}'")
        try:
            manifest = load_manifest(cohort)
        except Exception as e:  # noqa: BLE001
            print(f"   ❌ {e}")
            return False

        missing = [p for p in manifest["path"].head(sample) if not (TCGA_ROOT / p).exists()]
        print(f"   ✓ {len(manifest):,} patches, {manifest['patient'].nunique()} patients, "
              f"labels: {sorted(manifest['label'].unique())}")
        if missing:
            print(f"   ❌ {len(missing)}/{sample} sampled patch files missing, e.g. {missing[0]}")
            return False
        print(f"   ✓ first {sample} patch files present on disk")
        return True

    def validate_model(self, model_name: str) -> bool:
        print(f"🤖 Checking model: {model_name}")
        adapter = None
        try:
            # load() already runs a dummy forward pass at the model's own input
            # resolution, so a successful load validates the CLS output shape.
            adapter = get_model_adapter(model_name, self.device)
            adapter.load()
            print(f"   ✓ loaded, embed_dim={adapter.embed_dim}, "
                  f"batch_size={adapter.default_batch_size}, "
                  f"registers={adapter.num_registers}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"   ❌ {type(e).__name__}: {e}")
            return False
        finally:
            if adapter is not None:
                adapter.cleanup()
                torch.cuda.empty_cache()
