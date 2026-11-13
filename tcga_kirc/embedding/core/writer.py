import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import SHARD_SIZE

PROVENANCE_FILENAME = "provenance.json"


class ParquetShardWriter:
    """Buffers rows and flushes fixed-size parquet shards to the local disk.

    A model's directory is considered complete once provenance.json exists;
    main.py uses that marker to resume an interrupted run per-model.
    """

    def __init__(self, model_name: str, cohort, test_run: bool):
        self.model_name = model_name
        self.out_dir = Path(cohort.get_path(test_run)) / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.shard_index = 0
        self.rows_written = 0

    def add(self, rows: list):
        self.rows.extend(rows)
        if len(self.rows) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        table = pa.Table.from_pandas(df, preserve_index=False)
        path = self.out_dir / f"shard_{self.shard_index:05d}.parquet"
        pq.write_table(table, path, compression="snappy")

        self.rows_written += len(self.rows)
        self.shard_index += 1
        self.rows = []
        del df, table

    def write_provenance(self, meta: dict):
        """Written last -- its presence marks the model as complete."""
        payload = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "rows": self.rows_written,
            "shards": self.shard_index,
            **meta,
        }
        (self.out_dir / PROVENANCE_FILENAME).write_text(json.dumps(payload, indent=2))
        return payload


def is_complete(model_name: str, cohort, test_run: bool) -> bool:
    out_dir = Path(cohort.get_path(test_run)) / model_name
    return (out_dir / PROVENANCE_FILENAME).exists()


def clear_model_output(model_name: str, cohort, test_run: bool):
    out_dir = Path(cohort.get_path(test_run)) / model_name
    if not out_dir.exists():
        return
    for path in list(out_dir.glob("shard_*.parquet")) + [out_dir / PROVENANCE_FILENAME]:
        if path.exists():
            path.unlink()
