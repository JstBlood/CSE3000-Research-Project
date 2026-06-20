#!/bin/bash
set -euo pipefail

RESULTS_ROOT=/scratch/mberzins/results/bodmas_5000_qwen35_2b_promptv2_5runs
DESC_FILE=${RESULTS_ROOT}/gen_run_01/descriptions.jsonl

JUDGE_ROOT=${RESULTS_ROOT}/prometheus_30k_gen_run_01_v3
INPUT_DIR=${JUDGE_ROOT}/inputs
CHUNK_DIR=${JUDGE_ROOT}/chunks
OUTPUT_DIR=${JUDGE_ROOT}/outputs
SLURM_DIR=${JUDGE_ROOT}/slurm
LOG_DIR=/scratch/mberzins/logs/prometheus_30k_gen_run_01_v3

CHUNK_SIZE=250

mkdir -p "$INPUT_DIR" "$CHUNK_DIR" "$OUTPUT_DIR" "$SLURM_DIR" "$LOG_DIR"

if [ ! -f "$DESC_FILE" ]; then
  echo "ERROR: missing gen_run_01 descriptions file:"
  echo "$DESC_FILE"
  exit 1
fi

DESC_COUNT=$(wc -l < "$DESC_FILE")
echo "gen_run_01 line count: $DESC_COUNT / 15000"

if [ "$DESC_COUNT" -ne 15000 ]; then
  echo "ERROR: gen_run_01 is not complete yet. Wait until it has 15000 rows."
  exit 1
fi

python /scratch/mberzins/scripts/prepare_prometheus_inputs_gen01_v3.py \
  --descriptions "$DESC_FILE" \
  --judge-root "$JUDGE_ROOT" \
  --generation-run-id 1 \
  --expected-descriptions 15000 \
  --expected-samples 5000

rm -rf "$CHUNK_DIR/sq1" "$CHUNK_DIR/sq2"
mkdir -p "$CHUNK_DIR/sq1" "$CHUNK_DIR/sq2" "$OUTPUT_DIR/sq1" "$OUTPUT_DIR/sq2"

split -d -a 4 -l "$CHUNK_SIZE" --additional-suffix=.jsonl \
  "$INPUT_DIR/sq1_inputs.jsonl" \
  "$CHUNK_DIR/sq1/chunk_"

split -d -a 4 -l "$CHUNK_SIZE" --additional-suffix=.jsonl \
  "$INPUT_DIR/sq2_inputs.jsonl" \
  "$CHUNK_DIR/sq2/chunk_"

N_SQ1=$(find "$CHUNK_DIR/sq1" -name 'chunk_*.jsonl' | wc -l)
N_SQ2=$(find "$CHUNK_DIR/sq2" -name 'chunk_*.jsonl' | wc -l)

echo "SQ1 chunks: $N_SQ1"
echo "SQ2 chunks: $N_SQ2"

if [ "$N_SQ1" -ne 60 ]; then
  echo "ERROR: expected 60 SQ1 chunks, got $N_SQ1"
  exit 1
fi

if [ "$N_SQ2" -ne 60 ]; then
  echo "ERROR: expected 60 SQ2 chunks, got $N_SQ2"
  exit 1
fi

cat > "$SLURM_DIR/judge_sq1_gen01_v3.slurm" <<SLURM
#!/bin/bash
#SBATCH --job-name=judge_sq1_v3
#SBATCH --partition=gpu-a100-small
#SBATCH --array=1-${N_SQ1}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/sq1_%A_%a.out
#SBATCH --error=${LOG_DIR}/sq1_%A_%a.err

set -euo pipefail

module load miniforge3
conda activate /scratch/mberzins/envs/qwen36-transformers

PROMETHEUS_MODEL=\${PROMETHEUS_MODEL:-}
if [ -z "\$PROMETHEUS_MODEL" ]; then
  PROMETHEUS_MODEL=\$(find /scratch/mberzins/models -maxdepth 3 -type f -name config.json 2>/dev/null | grep -i prometheus | head -n 1 | xargs -r dirname)
fi

if [ -z "\$PROMETHEUS_MODEL" ] || [ ! -d "\$PROMETHEUS_MODEL" ]; then
  echo "ERROR: Could not find Prometheus model."
  echo "Set PROMETHEUS_MODEL explicitly before sbatch, for example:"
  echo "PROMETHEUS_MODEL=/scratch/mberzins/models/YOUR_PROMETHEUS_FOLDER sbatch ${SLURM_DIR}/judge_sq1_gen01_v3.slurm"
  exit 1
fi

CHUNK_INDEX=\$(printf "%04d" \$((SLURM_ARRAY_TASK_ID - 1)))
INPUT_JSONL=${CHUNK_DIR}/sq1/chunk_\${CHUNK_INDEX}.jsonl
OUTPUT_JSONL=${OUTPUT_DIR}/sq1/chunk_\${CHUNK_INDEX}_judged.jsonl

echo "Task: SQ1"
echo "Chunk: \$CHUNK_INDEX"
echo "Input: \$INPUT_JSONL"
echo "Output: \$OUTPUT_JSONL"
echo "Prometheus model: \$PROMETHEUS_MODEL"
date

python /scratch/mberzins/scripts/run_prometheus_feedback_decimal_v3.py \
  --task sq1 \
  --input-jsonl "\$INPUT_JSONL" \
  --output-jsonl "\$OUTPUT_JSONL" \
  --model "\$PROMETHEUS_MODEL" \
  --judge-run-id 1 \
  --seed \$((7000 + SLURM_ARRAY_TASK_ID)) \
  --max-feedback-tokens 128 \
  --feedback-temperature 0.20 \
  --feedback-top-p 0.90 \
  --candidate-batch-size 8

date
SLURM

cat > "$SLURM_DIR/judge_sq2_gen01_v3.slurm" <<SLURM
#!/bin/bash
#SBATCH --job-name=judge_sq2_v3
#SBATCH --partition=gpu-a100-small
#SBATCH --array=1-${N_SQ2}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/sq2_%A_%a.out
#SBATCH --error=${LOG_DIR}/sq2_%A_%a.err

set -euo pipefail

module load miniforge3
conda activate /scratch/mberzins/envs/qwen36-transformers

PROMETHEUS_MODEL=\${PROMETHEUS_MODEL:-}
if [ -z "\$PROMETHEUS_MODEL" ]; then
  PROMETHEUS_MODEL=\$(find /scratch/mberzins/models -maxdepth 3 -type f -name config.json 2>/dev/null | grep -i prometheus | head -n 1 | xargs -r dirname)
fi

if [ -z "\$PROMETHEUS_MODEL" ] || [ ! -d "\$PROMETHEUS_MODEL" ]; then
  echo "ERROR: Could not find Prometheus model."
  echo "Set PROMETHEUS_MODEL explicitly before sbatch, for example:"
  echo "PROMETHEUS_MODEL=/scratch/mberzins/models/YOUR_PROMETHEUS_FOLDER sbatch ${SLURM_DIR}/judge_sq2_gen01_v3.slurm"
  exit 1
fi

CHUNK_INDEX=\$(printf "%04d" \$((SLURM_ARRAY_TASK_ID - 1)))
INPUT_JSONL=${CHUNK_DIR}/sq2/chunk_\${CHUNK_INDEX}.jsonl
OUTPUT_JSONL=${OUTPUT_DIR}/sq2/chunk_\${CHUNK_INDEX}_judged.jsonl

echo "Task: SQ2"
echo "Chunk: \$CHUNK_INDEX"
echo "Input: \$INPUT_JSONL"
echo "Output: \$OUTPUT_JSONL"
echo "Prometheus model: \$PROMETHEUS_MODEL"
date

python /scratch/mberzins/scripts/run_prometheus_feedback_decimal_v3.py \
  --task sq2 \
  --input-jsonl "\$INPUT_JSONL" \
  --output-jsonl "\$OUTPUT_JSONL" \
  --model "\$PROMETHEUS_MODEL" \
  --judge-run-id 1 \
  --seed \$((8000 + SLURM_ARRAY_TASK_ID)) \
  --max-feedback-tokens 128 \
  --feedback-temperature 0.20 \
  --feedback-top-p 0.90 \
  --candidate-batch-size 8

date
SLURM

chmod +x "$SLURM_DIR/judge_sq1_gen01_v3.slurm"
chmod +x "$SLURM_DIR/judge_sq2_gen01_v3.slurm"

echo
echo "Prepared Prometheus gen_run_01 v3 judging."
echo "Judge root: $JUDGE_ROOT"
echo
echo "Submit SQ1:"
echo "sbatch $SLURM_DIR/judge_sq1_gen01_v3.slurm"
echo
echo "Submit SQ2:"
echo "sbatch $SLURM_DIR/judge_sq2_gen01_v3.slurm"
