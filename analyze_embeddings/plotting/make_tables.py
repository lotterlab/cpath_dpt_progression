"""Build the appendix tables that compare sampling schemes and ordering methods.

Reads the cached results in results/<RESULTS_RUN>/ and writes CSVs to
plots/<RESULTS_RUN>/tables/:

  table1_resampling.csv          Table 1  tau under slide-level halves and
                                          empirical class sampling
  table2_ordering_baselines.csv  Table 2  tau for DPT vs. the four alternative
                                          ordering methods
  table3_model_rankings.csv      Table 3  models ranked by mean tau under each
                                          ordering method
  ordering_agreement_summary.csv          mean per-patch Spearman agreement
                                          between DPT and each other method
                                          (Appendix C.1 text)

Tables 4 and 5 are written by plot_trajectory_fidelity.py and plot_appendix.py.

Usage (from analyze_embeddings/):
    python plotting/make_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from plotting.plot_config import MODEL_LABELS

OUTPUT_DIR = config.PLOTS_OUTPUT_DIR / "tables"

MODELS = ["dinov2", "conch", "musk", "uni2", "virchow2", "gigapath"]
PROGRESSIONS = ["SCC", "CRC-Conventional", "CRC-Serrated", "BDC"]

# Split-half results, one folder per progression. BDC has too few slides to
# fill both halves at the main-text cohort size, so its run uses 1,500 patches.
SPLIT_HALF_RUNS = {
    "SCC": "scc",
    "CRC-Conventional": "crcconv",
    "CRC-Serrated": "crcserr",
    "BDC": "bdc_n1500",
}
# The split-half script repeats every condition over 10 seeds; the table
# reports the seed that matches the main evaluation (config.EVALUATION["seed"]).
TABLE1_SEED = config.EVALUATION["seed"]

METHODS = {
    "centroid_distance": "Centroid Dist.",
    "pca_projection": "PCA Proj.",
    "knn_geodesic": "k-NN Geod.",
    "diffusion_component_1": "DC1",
    "dpt": "DPT",
}


def load_dpt_tau() -> pd.Series:
    """Final-layer DPT tau per (progression, model) from the main evaluation."""
    df = pd.read_csv(config.FULL_RESULTS_OUTPUT_PATH)
    df = df[df["embedding_type"] == config.FINAL_EMBEDDING]
    return df.set_index(["progression", "model"])["tau"]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by progression then model, as in the paper."""
    index = pd.MultiIndex.from_product([PROGRESSIONS, MODELS], names=["progression", "model"])
    return df.reindex(index)


def with_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index()
    out["model"] = out["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return out


def build_table1(dpt_tau: pd.Series) -> pd.DataFrame:
    frames = []
    for progression, folder in SPLIT_HALF_RUNS.items():
        df = pd.read_csv(config.SPLIT_HALF_DIR / folder / "split_half.csv")
        frames.append(df[df["seed"] == TABLE1_SEED])
    runs = pd.concat(frames).pivot_table(
        index=["progression", "model"], columns="condition", values="tau"
    )
    table = pd.DataFrame(
        {
            "Partition A": runs["halfA_balanced"],
            "Partition B": runs["halfB_balanced"],
            "Full (balanced)": dpt_tau,
            "Empirical": runs["full_empirical"],
        }
    )
    table = ordered(table)
    mean = table.groupby(level="model").mean().reindex(MODELS)
    mean.index = pd.MultiIndex.from_product([["Mean"], mean.index], names=table.index.names)
    return pd.concat([table, mean])


def build_table2(dpt_tau: pd.Series) -> pd.DataFrame:
    base = pd.read_csv(config.BASELINE_RESULTS_PATH).pivot_table(
        index=["progression", "model"], columns="method", values="tau"
    )
    base["dpt"] = dpt_tau
    return ordered(base[list(METHODS)])


def build_table3(table2: pd.DataFrame) -> pd.DataFrame:
    mean_by_model = table2.groupby(level="model").mean()
    rows = []
    for method in ["dpt", "knn_geodesic", "centroid_distance", "diffusion_component_1", "pca_projection"]:
        ranked = mean_by_model[method].sort_values(ascending=False)
        for rank, (model, tau) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "ordering_method": METHODS[method],
                    "rank": rank,
                    "model": MODEL_LABELS.get(model, model),
                    "mean_tau": tau,
                }
            )
    return pd.DataFrame(rows)


def build_agreement_summary() -> pd.DataFrame:
    df = pd.read_csv(config.ORDERING_AGREEMENT_PATH)
    df = df[(df["method_a"] == "dpt") | (df["method_b"] == "dpt")].copy()
    df["method"] = np.where(df["method_a"] == "dpt", df["method_b"], df["method_a"])
    summary = (
        df.groupby("method")["spearman_overall"]
        .mean()
        .sort_values(ascending=False)
        .rename("mean_spearman_with_dpt")
        .reset_index()
    )
    summary["method"] = summary["method"].map(METHODS)
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dpt_tau = load_dpt_tau()

    table1 = build_table1(dpt_tau)
    table2 = build_table2(dpt_tau)
    table3 = build_table3(table2)
    agreement = build_agreement_summary()

    table2_out = with_labels(table2.rename(columns=METHODS))
    mean_row = {"progression": "Mean", "model": "", **table2.rename(columns=METHODS).mean().to_dict()}
    table2_out = pd.concat([table2_out, pd.DataFrame([mean_row])], ignore_index=True)

    outputs = {
        "table1_resampling.csv": with_labels(table1),
        "table2_ordering_baselines.csv": table2_out,
        "table3_model_rankings.csv": table3,
        "ordering_agreement_summary.csv": agreement,
    }
    for name, df in outputs.items():
        path = OUTPUT_DIR / name
        df.to_csv(path, index=False, float_format="%.3f")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
