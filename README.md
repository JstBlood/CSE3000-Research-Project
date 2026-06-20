# Representation-Induced Variation in LLM Program Descriptions

This repository contains the final scripts, prompts, logs, and result files for a controlled measurement study evaluating whether an LLM generates consistent high-level program descriptions across binary, assembly, and source-code representations.

The main research question is:

> Can an LLM provide consistent high-level descriptions of the same program when represented as source code, assembly code, and binary code?

The repository is intended to make the experiment reproducible on DelftBlue, with one important limitation: the raw SBAN/BODMAS dataset is not included. A reproducer must provide the same prepared dataset files or adapt the dataset-loading paths in the scripts.

---

## Study Overview

The experiment evaluates representation-induced variation in generated natural-language program descriptions. Each program is represented in three aligned forms:

- binary/disassembly text
- assembly code
- source code

For each representation, Qwen3.5-2B generates a concise high-level behaviour description. The study then evaluates:

- **SQ1:** cross-representation consistency between generated descriptions
- **SQ2:** reference-based description quality against the SBAN natural-language description
- **generation stability:** variation across repeated runs

The intended interpretation is practical: if an LLM describes the same program differently depending on whether it sees source, assembly, or binary, then LLM-assisted reverse engineering and low-level code analysis may be unreliable unless representation effects are understood.

---

## Repository Structure

```text
.
├── README.md
├── .gitignore
├── prompts/
│   ├── generation/
│   │   └── qwen_generation_prompt_v3.txt
│   └── prometheus/
│       ├── sq1_prometheus_feedback_first_decimal_v3.txt
│       └── sq2_prometheus_feedback_first_decimal_v3.txt
├── scripts/
│   ├── generation/
│   │   ├── generate_descriptions_v3.py
│   │   └── generate_qwen35_2b_promptv3_5runs.slurm
│   ├── judge/
│   │   ├── prepare_prometheus_inputs_gen01_v3.py
│   │   ├── run_prometheus_feedback_decimal_v3.py
│   │   ├── setup_prometheus_gen01_v3.sh
│   │   ├── judge_sq1_600sample_v3.slurm
│   │   └── judge_sq2_600sample_v3.slurm
│   ├── metrics/
│   │   └── compute_and_plot_cheap_metrics_75k.py
│   └── validation/
│       ├── validate_final_repo.py
│       └── check_final_scripts_and_prompts.py
├── results/
│   ├── generation/
│   │   ├── gen_run_01/descriptions.jsonl
│   │   ├── gen_run_02/descriptions.jsonl
│   │   ├── gen_run_03/descriptions.jsonl
│   │   ├── gen_run_04/descriptions.jsonl
│   │   └── gen_run_05/descriptions.jsonl
│   ├── cheap_metrics_75k/
│   ├── prometheus_600sample_gen_run_01_v3/
│   └── figures/
└── logs/
    ├── generation/
    └── prometheus/
```

---

## Dataset

The experiment uses a 5000-sample SBAN evaluation subset. Each sample must contain:

- one SHA-256 or equivalent stable sample identifier
- binary/disassembly text
- assembly code
- source code
- one natural-language reference description

The raw dataset is **not included** in this repository. To reproduce the experiment, place a compatible prepared dataset in the paths expected by the generation scripts, or edit the path constants in the scripts before running.

The final generation stage expects:

```text
5000 samples × 3 representations × 5 runs = 75000 generated descriptions
```

Labels such as benign or malware are used only for dataset sampling/bookkeeping. They are not included in the generation or judge prompts.

---

## Models

The experiment uses:

```text
Generator: Qwen3.5-2B
Judge:     Prometheus-7B-v2.0
```

Model weights are not stored in this repository. Download them separately on DelftBlue or point the scripts to an existing model cache.

---

## Reproduction Summary

A full reproduction has this sequence:

```text
1. Clone this repository.
2. Prepare the DelftBlue scratch folder layout.
3. Create the Python/Conda environment.
4. Download or point to the required model weights.
5. Place the prepared SBAN dataset at the configured dataset path.
6. Run Qwen description generation for five repeated runs.
7. Prepare Prometheus judge inputs from generation run 1.
8. Run Prometheus SQ1 and SQ2 judging on the fixed 600-sample subset.
9. Merge Prometheus chunk outputs.
10. Transfer final outputs to a local machine.
11. Run cheap metrics and plotting.
```

---

## 1. Clone the Repository

On DelftBlue set up ssh key and run:

```bash
git clone git@github.com:JstBlood/CSE3000-Research-Project.git
```

---

## 2. Prepare DelftBlue Folder Layout

The original experiment used `/scratch/mberzins`, replace `mberzins` with your own net-id.

```bash
export NETID="$USER"
export SCRATCH_ROOT="/scratch/$NETID"

mkdir -p "$SCRATCH_ROOT"/{scripts,logs,models,envs,results,datasets}
mkdir -p "$SCRATCH_ROOT/logs/qwen35_2b_promptv3_5runs"
mkdir -p "$SCRATCH_ROOT/logs/prometheus_600sample_gen_run_01_v3"
```

Copy the scripts from the repository to scratch:

```bash
cp scripts/generation/*.py "$SCRATCH_ROOT/scripts/"
cp scripts/generation/*.slurm "$SCRATCH_ROOT/scripts/"
cp scripts/judge/*.py "$SCRATCH_ROOT/scripts/"
cp scripts/judge/*.sh "$SCRATCH_ROOT/scripts/"
cp scripts/judge/*.slurm "$SCRATCH_ROOT/scripts/" 2>/dev/null || true
```

## 3. Create the Python Environment on DelftBlue

The scripts were run from a Conda environment stored on `/scratch` to avoid filling the home directory.

If DelftBlue exposes Conda/Miniforge through modules, load the relevant module first. The exact module name may depend on DelftBlue configuration.

Create the environment:

```bash
export ENV_PATH="/scratch/$USER/envs/qwen36-transformers"

module load miniforge3
conda create -p "$ENV_PATH" python=3.11 -y
conda activate "$ENV_PATH"

python -m pip install --upgrade pip
python -m pip install \
  torch \
  transformers \
  accelerate \
  bitsandbytes \
  sentencepiece \
  protobuf \
  pandas \
  numpy \
  scipy \
  tqdm \
  matplotlib \
  sentence-transformers \
  bert-score \
  rouge-score \
  huggingface_hub[cli]
```

Check the environment:

```bash
python - <<'PY'
import torch
import transformers
import pandas
import numpy
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY
```

On a login node, `torch.cuda.is_available()` may be `False`. This is normal. CUDA availability should be checked inside a GPU Slurm job.

---

## 4. Download or Point to the Models

Use your own Hugging Face authentication.

```bash
export MODEL_ROOT="/scratch/$USER/models"
mkdir -p "$MODEL_ROOT"

hf download Qwen/Qwen3.5-2B \
  --local-dir "$MODEL_ROOT/Qwen3.5-2B"

hf download prometheus-eval/prometheus-7b-v2.0 \
  --local-dir "$MODEL_ROOT/prometheus-7b-v2.0"
```

If the scripts use model IDs directly instead of local paths, this step can be replaced by using the Hugging Face cache. For reproducibility, local model directories on `/scratch` are preferred.

---

## 5. Place the Prepared Dataset

The raw SBAN/BODMAS dataset is not included. A compatible prepared dataset must be available before generation.

Place the prepared dataset under a scratch location such as:

```text
/scratch/<netid>/datasets/sban_bodmas/
```

Then edit or verify the dataset paths in:

```text
scripts/generation/generate_descriptions_v3.py
scripts/generation/generate_qwen35_2b_promptv3_5runs.slurm
```

The generation script should be able to load exactly 5000 complete samples with all three representations and the reference description.

---

## 6. Run Description Generation

Submit the five-run generation array:

```bash
sbatch "$SCRATCH_ROOT/scripts/generate_qwen35_2b_promptv3_5runs.slurm"
```

Monitor queue state:

```bash
squeue -u "$USER"
```

Monitor generation logs:

```bash
tail -f "$SCRATCH_ROOT/logs/qwen35_2b_promptv3_5runs/"*.out
```

Check errors:

```bash
tail -n 80 "$SCRATCH_ROOT/logs/qwen35_2b_promptv3_5runs/"*.err
```

Check generation progress:

```bash
RESULTS_ROOT="/scratch/$USER/results/bodmas_5000_qwen35_2b_promptv2_5runs"

for f in "$RESULTS_ROOT"/gen_run_*/descriptions.jsonl; do
  [ -f "$f" ] || continue
  run=$(basename "$(dirname "$f")")
  count=$(wc -l < "$f")
  percent=$(awk -v c="$count" 'BEGIN { printf "%.1f", 100*c/15000 }')
  echo "$run: $count / 15000 descriptions ($percent%)"
done
```

The Slurm array can be resubmitted if it times out. The generation script is designed to skip valid existing rows and continue missing work.

---

## 7. Prepare Prometheus Judge Inputs

Prometheus is run only on a fixed 600-sample subset from generation run 1 because the full LLM-as-judge evaluation is computationally expensive.

The subset configuration used in this experiment is:

```text
generation run: gen_run_01
subset size:    600 samples
seed:           20260620
chunk size:     400 rows
SQ1 total:      1800 judgments
SQ2 total:      1800 judgments
```

Run the setup script:

```bash
bash "$SCRATCH_ROOT/scripts/setup_prometheus_gen01_v3.sh"
```

Verify that the input/chunk folders were created:

```bash
SUBSET_ROOT="/scratch/$USER/results/bodmas_5000_qwen35_2b_promptv2_5runs/prometheus_600sample_gen_run_01_v3"

find "$SUBSET_ROOT" -maxdepth 3 -type f | sort | head -50
```

Expected chunk layout:

```text
chunks/sq1/chunk_0000.jsonl
chunks/sq1/chunk_0001.jsonl
chunks/sq1/chunk_0002.jsonl
chunks/sq1/chunk_0003.jsonl
chunks/sq1/chunk_0004.jsonl
chunks/sq2/chunk_0000.jsonl
chunks/sq2/chunk_0001.jsonl
chunks/sq2/chunk_0002.jsonl
chunks/sq2/chunk_0003.jsonl
chunks/sq2/chunk_0004.jsonl
```

---

## 8. Run Prometheus SQ1 and SQ2 Judging

Submit SQ1:

```bash
sbatch "$SUBSET_ROOT/slurm/judge_sq1_600sample_v3.slurm"
```

Submit SQ2:

```bash
sbatch "$SUBSET_ROOT/slurm/judge_sq2_600sample_v3.slurm"
```

Monitor:

```bash
squeue -u "$USER"
```

Check progress:

```bash
SUBSET_ROOT="/scratch/$USER/results/bodmas_5000_qwen35_2b_promptv2_5runs/prometheus_600sample_gen_run_01_v3"

SQ1=$(cat "$SUBSET_ROOT"/outputs/sq1/chunk_*_judged.jsonl 2>/dev/null | wc -l)
SQ2=$(cat "$SUBSET_ROOT"/outputs/sq2/chunk_*_judged.jsonl 2>/dev/null | wc -l)
TOTAL=$((SQ1 + SQ2))

echo "SQ1: $SQ1 / 1800"
echo "SQ2: $SQ2 / 1800"
echo "Total: $TOTAL / 3600"
awk -v t="$TOTAL" 'BEGIN { printf "Overall progress: %.2f%%\n", 100*t/3600 }'
```

If a chunk times out, no completed rows are lost. The judge script writes row by row. Resubmit only incomplete array tasks.

Mapping:

```text
array task 1 -> chunk_0000
array task 2 -> chunk_0001
array task 3 -> chunk_0002
array task 4 -> chunk_0003
array task 5 -> chunk_0004
```

Chunk count check:

```bash
for task in sq1 sq2; do
  echo "$task chunks:"
  for i in 0000 0001 0002 0003 0004; do
    in="$SUBSET_ROOT/chunks/$task/chunk_${i}.jsonl"
    out="$SUBSET_ROOT/outputs/$task/chunk_${i}_judged.jsonl"
    expected=$(wc -l < "$in")
    done=$(cat "$out" 2>/dev/null | wc -l)
    echo "$task chunk_$i: $done / $expected"
  done
  echo
done
```

Example resubmission for incomplete SQ1 chunks 0002 and 0003:

```bash
sbatch --array=3,4 "$SUBSET_ROOT/slurm/judge_sq1_600sample_v3.slurm"
```

---

## 9. Merge Prometheus Outputs

After SQ1 and SQ2 each reach 1800 rows:

```bash
SUBSET_ROOT="/scratch/$USER/results/bodmas_5000_qwen35_2b_promptv2_5runs/prometheus_600sample_gen_run_01_v3"

cat "$SUBSET_ROOT"/outputs/sq1/chunk_*_judged.jsonl \
  > "$SUBSET_ROOT/sq1_prometheus_judge_scores_decimal_600sample_gen_run_01_v3.jsonl"

cat "$SUBSET_ROOT"/outputs/sq2/chunk_*_judged.jsonl \
  > "$SUBSET_ROOT/sq2_prometheus_reference_judge_scores_decimal_600sample_gen_run_01_v3.jsonl"

wc -l "$SUBSET_ROOT/sq1_prometheus_judge_scores_decimal_600sample_gen_run_01_v3.jsonl"
wc -l "$SUBSET_ROOT/sq2_prometheus_reference_judge_scores_decimal_600sample_gen_run_01_v3.jsonl"
```

Expected:

```text
1800
1800
```

---

## 10. Transfer Final DelftBlue Outputs to Local Machine

Run these commands on your local machine, not on DelftBlue.

```bash
LOCAL_REPO="<your-path>"
REMOTE="<net-id>@login.delftblue.tudelft.nl"
REMOTE_ROOT="/scratch/<net-id>/results/bodmas_5000_qwen35_2b_promptv2_5runs"
```

Transfer generation outputs:

```bash
rsync -avPz \
  --include='gen_run_*/' \
  --include='gen_run_*/descriptions.jsonl' \
  --exclude='*' \
  "$REMOTE:$REMOTE_ROOT/" \
  "$LOCAL_REPO/results/generation/"
```

Transfer Prometheus results:

```bash
rsync -avPz \
  "$REMOTE:$REMOTE_ROOT/prometheus_600sample_gen_run_01_v3/" \
  "$LOCAL_REPO/results/prometheus_600sample_gen_run_01_v3/"
```

Transfer logs:

```bash
rsync -avPz \
  "$REMOTE:/scratch/<net-id>/logs/qwen35_2b_promptv3_5runs/" \
  "$LOCAL_REPO/logs/generation/"

rsync -avPz \
  "$REMOTE:/scratch/<net-id>/logs/prometheus_600sample_gen_run_01_v3/" \
  "$LOCAL_REPO/logs/prometheus/"
```

Replace <net-id> or <your-path> variables with the relevant DelftBlue account name and local path.

---

## 11. Run Cheap Metrics and Plots Locally

Create a local environment:

```bash
cd "<your-path>"

python3 -m venv rp-results-env
source rp-results-env/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  pandas \
  numpy \
  scipy \
  matplotlib \
  sentence-transformers \
  bert-score \
  rouge-score \
  torch \
  transformers
```

Run the metrics script:

```bash
export RUNS_ROOT="<your-path>/results/generation"
export OUT_DIR="<your-path>/results/cheap_metrics_75k"

python scripts/metrics/compute_and_plot_cheap_metrics_75k.py
```

The metric script computes:

```text
SQ1:
- sentence-transformer cosine similarity
- ROUGE-L

SQ2:
- BERTScore precision
- BERTScore recall
- BERTScore F1
```

Figures and summary files should be written to:

```text
results/cheap_metrics_75k/
results/figures/
```

## Important Exclusions

This repository intentionally excludes:

- raw SBAN dataset files
- model weights
- Conda environments
- Hugging Face caches
- temporary scratch folders

The repository keeps final scripts, prompts, Slurm files, logs, generated descriptions, metric outputs, and figures.
