#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

GENERATION_ROOT="${GENERATION_ROOT:-results/generation}"
OUT_DIR="${OUT_DIR:-results/cheap_metrics_75k}"
FIGURES_DIR="${FIGURES_DIR:-results/figures}"

mkdir -p "$OUT_DIR" "$FIGURES_DIR"

echo "Running SQ1 cheap metrics..."
python scripts/metrics/compute_sq1_cheap_metrics_75k.py \
  --generation-root "$GENERATION_ROOT" \
  --out-dir "$OUT_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --runs 5

echo "Running SQ2 BERTScore metrics..."
python scripts/metrics/compute_sq2_bertscore_75k.py \
  --generation-root "$GENERATION_ROOT" \
  --out-dir "$OUT_DIR" \
  --figures-dir "$FIGURES_DIR" \
  --runs 5 \
  --model-type roberta-large \
  --device auto \
  --batch-size 16 \
  --chunk-size 512

echo "Done."
