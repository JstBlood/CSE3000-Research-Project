#!/bin/bash
#SBATCH --job-name=gen_qwen_v3
#SBATCH --partition=gpu-a100-small
#SBATCH --array=1-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/mberzins/logs/qwen35_2b_promptv3_5runs/%x_%A_%a.out
#SBATCH --error=/scratch/mberzins/logs/qwen35_2b_promptv3_5runs/%x_%A_%a.err

set -euo pipefail

module load miniforge3
conda activate /scratch/mberzins/envs/qwen36-transformers

RUN_ID=${SLURM_ARRAY_TASK_ID}
RUN_PADDED=$(printf "%02d" "${RUN_ID}")
SEED=$((1000 + RUN_ID))

INPUT_JSONL=/scratch/mberzins/datasets/sban_bodmas/sban_bodmas_eval_test_5000.jsonl
MODEL=/scratch/mberzins/models/Qwen3.5-2B

RESULTS_ROOT=/scratch/mberzins/results/bodmas_5000_qwen35_2b_promptv2_5runs
OUT_DIR=${RESULTS_ROOT}/gen_run_${RUN_PADDED}
OUT_FILE=${OUT_DIR}/descriptions.jsonl

mkdir -p "${OUT_DIR}"

echo "Starting generation run ${RUN_ID}"
echo "Output file: ${OUT_FILE}"
echo "Seed: ${SEED}"
date

python /scratch/mberzins/scripts/generate_descriptions_v3.py \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUT_FILE}" \
  --model "${MODEL}" \
  --run-id "${RUN_ID}" \
  --seed "${SEED}" \
  --max-new-tokens 160 \
  --temperature 0.30 \
  --top-p 0.90 \
  --repetition-penalty 1.05

echo "Finished generation run ${RUN_ID}"
date
