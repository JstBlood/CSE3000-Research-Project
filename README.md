# Representation-Induced Variation in LLM Program Descriptions

This repository contains the final scripts, prompts, logs, and result files for a controlled measurement study evaluating whether an LLM generates consistent high-level program descriptions across binary, assembly, and source-code representations.

## Study Overview

The experiment evaluates representation-induced variation in generated natural-language program descriptions. Each program is represented in three aligned forms:

- binary/disassembly text
- assembly code
- source code

For each representation, Qwen3.5-2B generates a concise high-level behaviour description. The study then evaluates:

- SQ1: cross-representation consistency between generated descriptions
- SQ2: reference-based description quality against the SBAN natural-language description
- intra-representation generation stability across repeated runs

## Dataset

The experiment uses a 5000-sample SBAN evaluation subset. Each sample has three aligned input representations and one natural-language reference description.

The raw dataset is not included in this repository. Final generated descriptions and metric outputs are stored under `results/`.

## Description Generation

Descriptions were generated with Qwen3.5-2B. The prompt receives only the current representation text. It does not receive the reference description, malware/benign label, sample identifier, dataset metadata, or alternative representations of the same program.

Each sample is evaluated under three representations and five repeated generation runs.

Final generation outputs:

    results/generation/gen_run_01/descriptions.jsonl
    results/generation/gen_run_02/descriptions.jsonl
    results/generation/gen_run_03/descriptions.jsonl
    results/generation/gen_run_04/descriptions.jsonl
    results/generation/gen_run_05/descriptions.jsonl

Each row stores:

- cleaned generated description
- raw model output
- representation label
- sample identifier
- reference description
- generation metadata
- generation status

The final valid generation output contains 75,000 descriptions:

    5000 samples × 3 representations × 5 runs = 75000 descriptions

## Cheap Metrics

Cheap metrics are computed over all five generation runs.

For SQ1 cross-representation consistency:

- sentence-transformer cosine similarity
- ROUGE-L

For SQ2 reference-based quality:

- BERTScore precision
- BERTScore recall
- BERTScore F1

The plotted cheap-metric distributions average repeated generations per sample before plotting, so each sample contributes once per representation pair or representation.

Cheap metric outputs are stored in:

    results/cheap_metrics_75k/

Final figures are stored in:

    results/figures/

## Prometheus LLM-as-Judge

Prometheus-7B-v2.0 is used as an independent rubric-based LLM judge. Because LLM-as-judge evaluation with chain-of-thought prompting is very computationally expensive (and due to deadline time constraints), Prometheus is applied to a fixed 600-sample subset from generation run 1.

Prometheus subset size:

    SQ1: 600 samples × 3 representation pairs = 1800 judgments
    SQ2: 600 samples × 3 representations = 1800 judgments
    Total: 3600 judgments

Prometheus uses a feedback-first scoring procedure:

1. The judge writes brief rubric-based feedback without assigning a score.
2. The script computes constrained likelihoods over decimal scores from 1.0 to 5.0.
3. A prior score preference is subtracted for calibration.
4. The final reported score is the expected score over the calibrated score distribution.

Prometheus outputs include:

- original judge input
- sample and generation-run metadata
- judge prompt version
- judge seed
- brief rubric-based feedback
- calibrated expected score
- hard maximum-probability score
- coarse score-bucket probabilities
- full decimal score probability distribution

Prometheus outputs are stored in:

    results/prometheus_600sample_gen_run_01_v3/

## Scripts

Generation scripts:

    scripts/generation/generate_descriptions_v3.py
    scripts/generation/generate_qwen35_2b_promptv3_5runs.slurm

Prometheus judge scripts:

    scripts/judge/prepare_prometheus_inputs_gen01_v3.py
    scripts/judge/run_prometheus_feedback_decimal_v3.py
    scripts/judge/setup_prometheus_gen01_v3.sh

Metric scripts:

    scripts/metrics/compute_and_plot_cheap_metrics_75k.py

Validation script:

    scripts/validate_final_repo.py

## Logs

Slurm logs are stored in:

    logs/generation/
    logs/prometheus/

The logs document runtime behaviour, job progress, errors, and completion status. The actual judge feedback and scores are stored in the Prometheus JSONL result files, not only in the logs.

## Important Exclusions

This repository intentionally excludes:

- Conda environments
- LLM files
