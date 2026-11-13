"""Stream the tcga-ut shards and stage patches for a given label set.

Shards are class-mixed, so every shard is opened and filtered. JPEG bytes are copied verbatim -- no decode, resize, crop, or normalization.

Usage:
    python extract_patches.py --labels Kidney_renal_clear_cell_carcinoma \
                              --records data/kirc_patch_records.jsonl
"""

import argparse
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHARD_DIR = ROOT / "data" / "tcga-ut" / "data"
PATCH_DIR = ROOT / "data" / "patches"


def split_of(shard_path: Path) -> str:
    """dataset_internal_<split>_part###.tar -> <split>."""
    return shard_path.stem.split("_")[2]


def process_shard(shard_path_str: str, keep_labels: frozenset):
    shard_path = Path(shard_path_str)
    split = split_of(shard_path)
    records = []
    pending = {}

    with tarfile.open(shard_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if name.endswith(".jpg"):
                key, kind = name[:-4], "jpg"
            elif name.endswith(".json"):
                key, kind = name[:-5], "json"
            else:
                continue

            slot = pending.setdefault(key, {})
            if kind == "json":
                slot["label"] = json.load(tar.extractfile(member))["label"]
            else:
                slot["jpg"] = tar.extractfile(member).read()

            if "label" not in slot or "jpg" not in slot:
                continue

            label = slot["label"]
            data = slot.pop("jpg")
            del pending[key]

            if label not in keep_labels:
                continue

            slide_key, patch_id = key.split("/", 1)
            out_path = PATCH_DIR / label / slide_key / f"{patch_id}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)

            records.append({
                "key": key,
                "path": str(out_path.relative_to(ROOT)),
                "label": label,
                "split": split,
                "patient": key[:12],
                "tss": key[5:7],
                "sample_type": key[13:15],
            })

    incomplete = [k for k, v in pending.items() if v.get("label") in keep_labels]
    return shard_path.name, records, incomplete


def main():
    p = argparse.ArgumentParser(description="Stage tcga-ut patches for given labels")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--records", required=True, help="Output .jsonl of patch records")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    keep = frozenset(args.labels)
    shards = sorted(SHARD_DIR.glob("dataset_internal_*_part*.tar"))
    print(f"Streaming {len(shards)} shards, keeping {sorted(keep)}")
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    all_records, orphans = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fn = partial(process_shard, keep_labels=keep)
        for i, (name, records, incomplete) in enumerate(
                pool.map(fn, [str(s) for s in shards]), 1):
            all_records.extend(records)
            orphans.extend(f"{name}:{k}" for k in incomplete)
            if i % 10 == 0 or i == len(shards):
                print(f"  [{i}/{len(shards)}] running total: {len(all_records):,}", flush=True)

    if orphans:
        print(f"WARNING: {len(orphans)} keys missing a jpg/json pair: {orphans[:5]}")

    out = ROOT / args.records
    with out.open("w") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(all_records):,} records to {out}")


if __name__ == "__main__":
    main()
