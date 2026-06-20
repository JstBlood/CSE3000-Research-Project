#!/bin/bash
#SBATCH --job-name=judge_sq1_600
#SBATCH --partition=gpu-a100-small
#SBATCH --array=1-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mberzins/logs/prometheus_600sample_gen_run_01_v3/sq1_%A_%a.out
#SBATCH --error=/scratch/mberzins/logs/prometheus_600sample_gen_run_01_v3/sq1_%A_%a.err

set -euo pipefail

module load miniforge3
conda activate /scratch/mberzins/envs/qwen36-transformers

PROMETHEUS_MODEL=${PROMETHEUS_MODEL:-}
if [ -z "$PROMETHEUS_MODEL" ]; then
  PROMETHEUS_MODEL=$(find /scratch/mberzins/models -maxdepth 3 -type f -name config.json 2>/dev/null | grep -i prometheus | head -n 1 | xargs -r dirname)
fi

if [ -z "$PROMETHEUS_MODEL" ] || [ ! -d "$PROMETHEUS_MODEL" ]; then
  echo "ERROR: Could not find Prometheus model."
  echo "Set PROMETHEUS_MODEL explicitly."
  exit 1
fi

CHUNK_INDEX=$(printf "%04d" $((SLURM_ARRAY_TASK_ID - 1)))
INPUT_JSONL=/scratch/mberzins/results/bodmas_5000_qwen35_2b_promptv2_5runs/prometheus_600sample_gen_run_01_v3/chunks/sq1/chunk_${CHUNK_INDEX}.jsonl
OUTPUT_JSONL=/scratch/mberzins/results/bodmas_5000_qwen35_2b_promptv2_5runs/prometheus_600sample_gen_run_01_v3/outputs/sq1/chunk_${CHUNK_INDEX}_judged.jsonl

echo "SQ1 chunk $CHUNK_INDEX"
echo "Input: $INPUT_JSONL"
echo "Output: $OUTPUT_JSONL"
echo "Model: $PROMETHEUS_MODEL"
date

python /scratch/mberzins/scripts/run_prometheus_feedback_decimal_v3.py \
  --task sq1 \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --model "$PROMETHEUS_MODEL" \
  --judge-run-id 1 \
  --seed $((7100 + SLURM_ARRAY_TASK_ID)) \
  --max-feedback-tokens 128 \
  --feedback-temperature 0.20 \
  --feedback-top-p 0.90 \
  --candidate-batch-size 8

date
