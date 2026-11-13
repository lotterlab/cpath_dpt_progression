#!/usr/bin/env python3
"""Write Table 6: concordance between pseudotime and non-morphological endpoints.

Reads label_shuffle_null.csv (written by run_label_shuffle_null.py) and keeps the
three endpoints reported in the paper. Each cell is the observed value, its
z-score against the label-shuffled null, and the one-sided permutation p-value.

Usage (from tcga_kirc/analysis/):
    python make_table6.py                           # cached paper results
    python make_table6.py --results results/<run>
"""

import argparse
from pathlib import Path

import pandas as pd

import config

MODELS = {"uni2": "UNI-2", "virchow2": "Virchow-2", "conch": "CONCH",
          "musk": "MuSK", "gigapath": "Prov-GigaPath", "dinov2": "DINOv2"}
ENDPOINTS = {"stage": "stage_tau", "aneuploidy": "aneuploidy_tau", "DSS": "dss_c_index"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(config.PAPER_RESULTS_DIR))
    args = p.parse_args()

    res = Path(args.results)
    null = pd.read_csv(res / "label_shuffle_null.csv")
    null = null[null["endpoint"].isin(ENDPOINTS)]

    rows = []
    for model, label in MODELS.items():
        row = {"model": label}
        for endpoint, name in ENDPOINTS.items():
            cell = null[(null["model"] == model) & (null["endpoint"] == endpoint)].iloc[0]
            row[name] = cell["observed"]
            row[f"{name}_z"] = cell["z"]
            row[f"{name}_p"] = cell["p_perm"]
            row[f"{name}_n"] = cell["n"]
        rows.append(row)

    table = pd.DataFrame(rows)
    path = res / "table6_kirc_concordance.csv"
    table.to_csv(path, index=False, float_format="%.4f")
    print(table.round(3).to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
