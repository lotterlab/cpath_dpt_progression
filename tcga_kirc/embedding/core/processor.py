import gc

import torch
from tqdm import tqdm

from config import USE_AMP
from models import get_model_adapter

from .dataset import META_COLUMNS, build_loader
from .extractor import ClsExtractor
from .writer import ParquetShardWriter


class ClsEmbeddingProcessor:
    """Embeds every manifest patch with one model and writes parquet shards.

    Mirrors LayerEmbeddingProcessor from the SPIDER experiment, but emits a
    single row per patch (the final CLS token) instead of one row per
    token-type x depth combination.
    """

    def __init__(self, device: str, use_amp: bool = USE_AMP):
        self.device = device
        self.use_amp = use_amp

    def process_model(self, model_name: str, manifest, cohort,
                      test_run: bool = False) -> dict:
        print(f"\n   🤖 Processing Model: {model_name}")

        adapter = get_model_adapter(model_name, self.device)
        adapter.load()

        extractor = ClsExtractor(adapter, use_amp=self.use_amp)
        writer = ParquetShardWriter(model_name, cohort, test_run)
        batch_size = adapter.default_batch_size

        try:
            loader = build_loader(manifest, adapter.get_transform(), batch_size)
            meta_records = manifest[META_COLUMNS].to_dict("records")

            n_batches = (len(manifest) + batch_size - 1) // batch_size
            for tensors, indices in tqdm(loader, total=n_batches, desc=f"      {model_name}"):
                if tensors is None:
                    continue

                embeddings = extractor.extract(tensors.to(self.device, non_blocking=True))

                writer.add([
                    self._build_row(meta_records[row_idx], model_name,
                                    embeddings[i], cohort.dataset_name)
                    for i, row_idx in enumerate(indices)
                ])

            writer.flush()
            meta = writer.write_provenance({
                "dataset": cohort.dataset_name,
                "batch_size": batch_size,
                "manifest_rows": len(manifest),
                "test_run": test_run,
                **extractor.get_provenance_meta(),
            })
            print(f"      ✓ {writer.rows_written:,} rows in {writer.shard_index} shard(s)"
                  f" -> {writer.out_dir}")
            return meta

        finally:
            adapter.cleanup()
            del extractor, adapter
            gc.collect()
            torch.cuda.empty_cache()

    @staticmethod
    def _build_row(meta: dict, model_name: str, embedding, dataset_name: str) -> dict:
        key = meta["key"]
        slide_id, patch_id = key.split("/", 1)
        return {
            "dataset": dataset_name,
            "class": meta["label"],
            "slide_id": slide_id,
            "patch_id": patch_id,
            "model": model_name,
            "embedding": embedding.tolist(),
            "dim": int(embedding.shape[0]),
            "key": key,
            "patient": meta["patient"],
            "split": meta["split"],
            "tss": meta["tss"],
            "sample_type": meta["sample_type"],
        }
