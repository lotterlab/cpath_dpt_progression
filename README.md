# Do Pathology Foundation Models Encode Disease Progression?

Code and cached results for *Do Pathology Foundation Models Encode Disease
Progression? A Pseudotime Analysis of Visual Representations* (Pritika Vig,
Ren-Chin Wu, William Lotter). Paper: [arXiv:2601.21334](https://arxiv.org/abs/2601.21334).

The paper asks whether pathology foundation models arrange disease stages along
a continuous progression in their embedding space. It applies diffusion
pseudotime (DPT) to patch embeddings from six models on four cancer
progressions from the SPIDER dataset and measures *trajectory fidelity*:
Kendall's τ between inferred pseudotime and the known stage order.

**[REPRODUCING.md](REPRODUCING.md) lists every figure, table and in-text number
in the paper with the command that produces it, the cached file it reads and
where the stored output lives.**

## Repository layout

```
generate_embeddings/   Step 1. Embed SPIDER patches with each foundation model
                       and write Parquet shards to a storage bucket.
analyze_embeddings/    Step 2. DPT, trajectory fidelity and all downstream
                       analyses on the SPIDER embeddings.
  analysis/              DPT, metrics and alternative ordering methods (library code)
  data/                  Embedding loader with slide-aware sampling
  generate_data/         Scripts that compute results from embeddings
  plotting/              Scripts that draw figures and tables from results
  results/paper/         Cached results behind every figure and table
  plots/paper/           The figures and tables as they appear in the paper
tcga_kirc/             Appendix J. The same analysis on TCGA-KIRC against stage,
                       aneuploidy and survival.
docs/                  Project page (served with GitHub Pages)
REPRODUCING.md         Figure-by-figure and table-by-table commands
requirements.txt       Pinned dependencies
```

## Quick start: redraw the figures from cached results

No data download or GPU is needed for this.

```bash
git clone https://github.com/lotterlab/cpath_dpt_progression.git
cd cpath_dpt_progression
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # the first block of the file is enough

cd analyze_embeddings
python plotting/plot_trajectory_fidelity.py    # Fig 2 and Table 4
python plotting/make_tables.py                 # Tables 1-3
```

Outputs are written to `analyze_embeddings/plots/paper/`.

## Running the full pipeline

1. **Generate embeddings** with [generate_embeddings/](generate_embeddings/README.md).
   Six models (DINOv2, CONCH, MuSK, UNI-2, Virchow-2, Prov-GigaPath) are run on
   the SPIDER skin, colorectal and breast datasets.
2. **Analyse them** with [analyze_embeddings/](analyze_embeddings/README.md).
   `generate_data/` scripts read the embeddings and write result CSVs;
   `plotting/` scripts turn those into figures and tables.
3. **TCGA-KIRC** (Appendix J) has its own staging, embedding and analysis steps
   in [tcga_kirc/](tcga_kirc/README.md).

Embedding the four SPIDER cohorts with six models took about 24 GPU-hours on a
single NVIDIA T4, and the analyses about another 24 hours.

## Data access

The cached results in this repository are enough to redraw the paper's figures.
To recompute them you need the data below, which is not redistributed here.

**SPIDER patches.** Download and unpack the three datasets:
[SPIDER-skin](https://huggingface.co/datasets/histai/SPIDER-skin),
[SPIDER-colorectal](https://huggingface.co/datasets/histai/SPIDER-colorectal),
[SPIDER-breast](https://huggingface.co/datasets/histai/SPIDER-breast).

**Model weights.** Virchow-2, UNI-2, CONCH, Prov-GigaPath and MuSK are gated on
Hugging Face; request access and put your token in `$HF_TOKEN` or
`~/.hf_token`. Links are in the
[embedding README](generate_embeddings/README.md).

**Embeddings bucket.** The analysis reads embeddings from Google Cloud Storage.
The buckets used for the paper are private, so create your own (one per dataset)
and let `generate_embeddings/` fill them. The layout the analysis expects is:

```
gs://<bucket>/embeddings/computed/<model>/shard_00000.parquet, ...
```

with one row per patch and embedding type, and columns `class`, `slide_id`,
`patch_id`, `embedding_type` and `embedding`. `embedding_type` is
`final_embedding` or a layer-wise type such as `cls_0.625`.

Point the analysis at your buckets by setting these before running, or by
editing the defaults at the top of
[analyze_embeddings/config.py](analyze_embeddings/config.py):

```bash
export SPIDER_SKIN_BUCKET=my-spider-skin
export SPIDER_COLORECTAL_BUCKET=my-spider-colorectal
export SPIDER_BREAST_BUCKET=my-spider-breast
gcloud auth application-default login
```

**Local images.** The scripts that display or segment patches (cell counts,
Figs 20 and 21) read the unpacked SPIDER images from
`$SPIDER_IMAGE_ROOT/{spider-skin,spider-col,spider-breast}/train/<class>/<slide>/<patch>.png`.
`SPIDER_IMAGE_ROOT` defaults to `~/data`. Fig 1 reads its example patches from
the bucket instead, at `gs://<bucket>/<image_subdir>/train/...`, where
`image_subdir` is set per progression in `config.py`.

## Citation

```bibtex
@inproceedings{vig2026pseudotime,
  title     = {Do Pathology Foundation Models Encode Disease Progression?
               A Pseudotime Analysis of Visual Representations},
  author    = {Vig, Pritika and Wu, Ren-Chin and Lotter, William},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  eprint    = {2601.21334},
  archivePrefix = {arXiv}
}
```
