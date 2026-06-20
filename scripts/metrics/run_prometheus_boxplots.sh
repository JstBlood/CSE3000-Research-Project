#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p results/figures
mkdir -p results/prometheus_600sample_gen_run_01_v3/summary

python scripts/metrics/plot_prometheus_sq1_boxplots.py
python scripts/metrics/plot_prometheus_sq2_boxplots.py

echo "Prometheus boxplots complete."
