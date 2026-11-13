import numpy as np
import torch

from config import AMP_DTYPE, USE_AMP
from models.base import PathologyModelAdapter


class ClsExtractor:
    """Extracts the final-layer CLS token only.

    The SPIDER experiment's LayerExtractor hooks intermediate blocks and derives
    cls / register_mean / patch_mean at five depths. Here only the final CLS
    token is kept, which every adapter already returns from forward_features():
    timm ViTs take index 0 of the post-norm sequence, CONCH returns its pooled
    visual embedding, and MUSK returns vision_cls. No hooks are needed.

    The forward pass runs under fp16 autocast by default (see config.USE_AMP);
    the CLS vector is cast back to fp32 before normalization, so stored
    embeddings are always fp32. Vectors are L2-normalized to match the SPIDER
    run.
    """

    def __init__(self, adapter: PathologyModelAdapter, use_amp: bool = USE_AMP):
        self.adapter = adapter
        self.use_amp = use_amp and adapter.device == "cuda"
        self.amp_dtype = getattr(torch, AMP_DTYPE)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> np.ndarray:
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            output = self.adapter.forward_features(images)

        if isinstance(output, tuple):
            output = output[0]

        if output.ndim != 2:
            raise RuntimeError(
                f"Expected [Batch, Dim] from forward_features, got {tuple(output.shape)}"
            )

        # Cast back to fp32 before normalizing so autocast never reaches storage.
        embeddings = output.detach().float().cpu().numpy()

        if not np.isfinite(embeddings).all():
            n_bad = int((~np.isfinite(embeddings)).any(axis=1).sum())
            raise RuntimeError(
                f"{n_bad}/{len(embeddings)} embeddings contain NaN or Inf. "
                f"If this model overflows under {AMP_DTYPE}, rerun with --fp32."
            )

        return self._l2_normalize(embeddings)

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 0)

    def get_provenance_meta(self) -> dict:
        return {
            "embedding": "final_layer_cls_token",
            "l2_normalized": True,
            "cls_token_index": self.adapter.cls_token_index,
            "embedding_dim": self.adapter.embed_dim,
            "autocast": self.amp_dtype.__str__().replace("torch.", "") if self.use_amp else "off",
            "stored_dtype": "float32",
        }
