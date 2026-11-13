#!/usr/bin/env python3
"""TCGA-KIRC CLS-token embedding extractor.

Embeds every patch in ../data/manifest_kirc.parquet with the shared pathology
foundation-model adapters and writes one final-layer CLS vector per patch to
local parquet shards.
"""

import argparse
import os
from pathlib import Path

import torch

from config import COHORTS, DEFAULT_COHORT, EXPECTED_MODELS, USE_AMP, get_cohort
from core.dataset import load_manifest
from core.processor import ClsEmbeddingProcessor
from core.validator import DryRunValidator
from core.writer import clear_model_output, is_complete


def load_hf_token():
    """Load HuggingFace token from environment or file."""
    token = os.getenv("HF_TOKEN")
    if not token and Path("~/.hf_token").expanduser().exists():
        token = Path("~/.hf_token").expanduser().read_text().strip()
    if token:
        os.environ["HF_TOKEN"] = token
        print("✓ HF Token loaded")
    else:
        print("⚠️ No HF Token found. Some models may fail.")


def parse_args():
    parser = argparse.ArgumentParser(description="TCGA CLS Embedding Extractor")

    parser.add_argument('--cohort', choices=list(COHORTS), default=DEFAULT_COHORT,
                        help=f"Cohort to embed (default: {DEFAULT_COHORT})")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--models', nargs='+', choices=EXPECTED_MODELS,
                       help="Process a specific set of models")
    group.add_argument('--all-models', action='store_true',
                       help="Process all configured models sequentially")

    parser.add_argument('--dry-run', action='store_true',
                        help='Validate the manifest and model loading without embedding')
    parser.add_argument('--test-run', action='store_true',
                        help='Process only 10 patches per label into embeddings/test_run/')
    parser.add_argument('--overwrite', action='store_true',
                        help='Re-embed models that already have output (default: skip)')
    parser.add_argument('--fp32', action='store_true',
                        help='Disable fp16 autocast for bit-parity with the SPIDER run (slow)')

    return parser.parse_args()


def main():
    args = parse_args()
    load_hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Device: {device}")

    models = EXPECTED_MODELS if args.all_models else args.models
    cohort = get_cohort(args.cohort)
    print(f"✓ Cohort: {cohort.name} ({cohort.dataset_name})")

    if args.dry_run:
        print("\n🔎 STARTING DRY RUN VALIDATION")
        validator = DryRunValidator(device)
        all_passed = validator.validate_manifest(cohort)
        for model_name in models:
            if not validator.validate_model(model_name):
                all_passed = False
        print("\n✅ All validations passed." if all_passed
              else "\n❌ Validation failed. See errors above.")
        return

    use_amp = USE_AMP and not args.fp32
    manifest = load_manifest(cohort, test_run=args.test_run)
    print(f"✓ Manifest: {len(manifest):,} patches "
          f"({manifest['patient'].nunique()} patients)")
    print(f"✓ Output:   {cohort.get_path(args.test_run)}")
    print(f"✓ Precision: {'fp16 autocast' if use_amp else 'fp32'}")

    processor = ClsEmbeddingProcessor(device, use_amp=use_amp)
    for idx, model_name in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"Model {idx}/{len(models)}: {model_name}")
        print(f"{'='*60}")

        if is_complete(model_name, cohort, args.test_run):
            if not args.overwrite:
                print("   ⏭️  Already complete (provenance.json present), skipping. "
                      "Use --overwrite to redo.")
                continue
            print("   ♻️  Overwriting existing output")
            clear_model_output(model_name, cohort, args.test_run)

        try:
            processor.process_model(model_name, manifest, cohort,
                                    test_run=args.test_run)
        except Exception as e:  # noqa: BLE001 - one bad model must not kill the run
            print(f"   ❌ Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
