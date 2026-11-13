# TCGA-KIRC: pseudotime vs non-morphological endpoints (Appendix J)

This folder applies the same diffusion pseudotime (DPT) procedure as the main
analysis to TCGA kidney renal clear cell carcinoma (KIRC). Patches are ordered
along a grade progression (G1 → G4, rooted at the medoid of G1 patches), and
patient-level mean pseudotime is compared with three endpoints that are not
defined by local morphology: pathological stage, aneuploidy score and
disease-specific survival (Table 6).

The DPT code is imported from `../analyze_embeddings/analysis/dpt.py` and the
model adapters from `../generate_embeddings/models/`; nothing is reimplemented
here.

## Reproducing Table 6 from the cached results

```bash
cd tcga_kirc/analysis
python make_table6.py
```

This reads `results/paper/label_shuffle_null.csv` and writes
`results/paper/table6_kirc_concordance.csv`. No data download is needed.

## Cached results (`analysis/results/paper/`)

| File | Contents | Written by |
|---|---|---|
| `dpt_patch_level.parquet` | Pseudotime of every patch for every model | `run_dpt.py` |
| `dpt_patient_level.csv` | Patient-level mean pseudotime joined to the clinical endpoints | `run_dpt.py` |
| `summary_table.csv` | Kendall's τ_b and C-index per model and endpoint | `run_dpt.py` |
| `label_shuffle_null.csv` | Observed value, label-shuffled null, z-score and permutation p-value per model and endpoint | `run_label_shuffle_null.py` |
| `label_shuffle_null_draws.csv` | The 2,000 null draws behind each cell | `run_label_shuffle_null.py` |
| `table6_kirc_concordance.csv` | Table 6 | `make_table6.py` |

The scripts also compute overall survival, progression-free interval and the
mRNAsi stemness index. The paper reports stage, aneuploidy and disease-specific
survival; the other endpoints remain in the cached files because the null
draws for all endpoints come from one random sequence.

## Running the pipeline from scratch

All data is staged under `tcga_kirc/data/`, which is not tracked by git.

**1. Download the inputs**

| Input | Source | Save as |
|---|---|---|
| H&E patches | [`dakomura/tcga-ut`](https://huggingface.co/datasets/dakomura/tcga-ut) on Hugging Face (gated), internal split: 256×256 patches at 0.5 µm/px from pathologist-annotated tumour regions | `data/tcga-ut/data/*.tar` |
| Survival, grade, stage | TCGA-CDR, `TCGA-CDR-SupplementalTableS1.xlsx`, from the [GDC PanCanAtlas page](https://gdc.cancer.gov/about-data/publications/pancanatlas) | `data/raw/TCGA-CDR-SupplementalTableS1.xlsx` |
| Aneuploidy score | `data_clinical_sample.txt` of the cBioPortal study `kirc_tcga_pan_can_atlas_2018` | `data/raw/cbioportal/kirc_data_clinical_sample.txt` |
| Stemness index | `StemnessScores_RNAexp_20170127.2.tsv.gz` from the UCSC Xena PanCanAtlas hub, unzipped | `data/raw/StemnessScores_RNAexp.tsv` |

**2. Stage the patches and clinical tables**

```bash
cd tcga_kirc
python extract_patches.py --labels Kidney_renal_clear_cell_carcinoma \
                          --records data/kirc_patch_records.jsonl
python stage_kirc.py
```

`stage_kirc.py` writes `data/manifest_kirc.parquet` and the tables in
`data/clinical/`, and prints the cohort statistics.

**3. Embed the patches** (GPU)

```bash
cd embedding
python main.py --all-models
```

One final-layer [CLS] vector per patch is written to
`data/embeddings_kirc/<model>/`. The run uses fp16 autocast by default; pass
`--fp32` for full precision.

**4. Run DPT and the significance test**

```bash
cd ../analysis
python run_dpt.py --out results/my_run
python run_label_shuffle_null.py --results results/my_run
python make_table6.py --results results/my_run
```

## Cohort

KIRC primary tumours (sample type `01`) with a usable histological grade: 379
patients, with up to 30 patches per patient. Endpoint coverage is 377 patients
for stage, 345 for aneuploidy score, and 371 (81 events) for disease-specific
survival. Significance is assessed against a patient-level label-shuffled null
with 2,000 draws, one-sided. DPT settings match the main analysis (k = 100,
10 diffusion components, cosine distance). Settings are in `analysis/config.py`.
