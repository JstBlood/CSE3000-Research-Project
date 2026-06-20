#!/usr/bin/env python3
"""
Compute SQ1 cheap metrics over all generation runs.

SQ1 = cross-representation consistency between generated descriptions
for the same sample and same generation run.

Input:
  results/generation/gen_run_01/descriptions.jsonl
  ...
  results/generation/gen_run_05/descriptions.jsonl

Output:
  results/cheap_metrics_75k/sq1_pairwise_metrics_all_runs.csv
  results/cheap_metrics_75k/sq1_pairwise_metrics_mean_over_runs.csv
  results/cheap_metrics_75k/sq1_summary_by_pair.csv
  results/figures/sq1_cheap_metric_boxplots_mean_over_runs.pdf
  results/figures/sq1_cheap_metric_boxplots_mean_over_runs.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PAIR_ORDER = [
    ("binary", "assembly"),
    ("b    ("b    ("b    ("b  ("as    ("b    ("b    ("b    ("b  ("as    ("b    ("b  y",    ("b    ("b    ("b    ("b  ("as    ("b    ("b    ("b    ("b  ("asâ€    ("b    ("b    ("b    ("b  ("as    ("b  mbl    (ou    ("b    ("b    ("b    ("b  ("as    ("b    ("b       in    ("b    ("b    ("b    ("b  ("as    ("b    ("b        ("ry_disassembly": "binary",
    "disassembly": "binary",
    "disasm": "binary",
    "asm": "assembly",
    "assembly": "assembly",
    "assembly code": "assembly",
    "assembly_code": "assembly",
    "src": "source",
    "source": "source",
    "source code": "source",
    "source_code": "source",
}


def pick_first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def normalize_representation(value: Any) -> str:
    if value is None:
        raise ValueError("Missing representation field.")

    raw = str(value).strip()
    key = raw.lower().replace("-", "_").replace(" ", "_")

    direct = raw.lower().strip()
    if direct in REP_ALIASES:
        return REP_ALIASES[direct]

    if key in REP_ALIASES:
        return REP_ALIASES[key]

    if "binary" in key or "disassembl" in key or "disasm" in key:
        return "binary"
    if "assembly" in key or key == "asm":
        return "assembly"
    if "source" in key or key == "src":
        return "source"

    raise ValueError(f"Unknown representation value: {value!r}")


def load_generation_rows(generation_root: Path, runs: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for run_idx in range(1, runs + 1):
        run_id = f"gen_run_{run_idx:02d}"
        path = generation_root / run_id / "descriptions.jsonl"

        if not path.exists():
            raise FileNotFoundError(f"Missing generation file: {path}")

        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue

                row = json.loads(line)

                sample_id = pick_first(
                    row,
                    [
                        "sample_id",
                        "program_id",
                        "id",
                        "sha256",
                        "file_id",
                        "binary_id",
                        "index",
                    ],
                )
                representation = pick_first(
                    row,
                    [
                        "representation",
                        "representation_type",
                        "input_representation",
                        "rep",
                        "view",
                    ],
                )
                description = pick_first(
                    row,
                    [
                        "generated_description",
                        "description",
                        "generated_nld",
                        "model_description",
                        "output",
                    ],
                )

                if sample_id is None:
                    raise KeyError(f"{path}:{line_no} has no sample identifier field.")
                if representation is None:
                    raise KeyError(f"{path}:{line_no} has no representation field.")
                if description is None:
                    raise KeyError(f"{path}:{line_no} has no generated description field.")

                rows.append(
                    {
                        "run_id": run_id,
                        "sample_id": str(sample_id),
                        "representation": normalize_representation(representation),
                        "generated_description": str(description).strip(),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No generation rows loaded.")

    return df


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    if len(b) > len(a):
        a, b = b, a

    previous = [0] * (len(b) + 1)

    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current

    return previous[-1]


def rouge_l_f1(text_a: str, text_b: str) -> float:
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()

    if not tokens_a or not tokens_b:
        return 0.0

    lcs = lcs_length(tokens_a, tokens_b)

    precision = lcs / len(tokens_a)
    recall = lcs / len(tokens_b)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def build_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    grouped = df.groupby(["run_id", "sample_id"], sort=False)

    for (run_id, sample_id), group in tqdm(grouped, desc="Building SQ1 pairs"):
        by_rep = {
            row["representation"]: row["generated_description"]
            for _, row in group.iterrows()
        }

        missing = [rep for rep in ["binary", "assembly", "source"] if rep not in by_rep]
        if missing:
            continue

        for rep_a, rep_b in PAIR_ORDER:
            records.append(
                {
                    "eval_id": f"{run_id}::{sample_id}::{rep_a}::{rep_b}",
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "pair": PAIR_LABELS[(rep_a, rep_b)],
                    "description_a": by_rep[rep_a],
                    "description_b": by_rep[rep_b],
                }
            )

    pair_df = pd.DataFrame(records)
    if pair_df.empty:
        raise ValueError("No complete SQ1 representation pairs could be built.")

    return pair_df


def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(texts))

    print(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return {text: emb for text, emb in zip(unique_texts, embeddings)}


def compute_sq1_metrics(pair_df: pd.DataFrame, model_name: str, batch_size: int) -> pd.DataFrame:
    texts = pair_df["description_a"].tolist() + pair_df["description_b"].tolist()
    embedding_by_text = compute_embeddings(texts, model_name, batch_size)

    cosine_values: List[float] = []
    rouge_values: List[float] = []

    for _, row in tqdm(pair_df.iterrows(), total=len(pair_df), desc="Computing SQ1 metrics"):
        desc_a = row["description_a"]
        desc_b = row["description_b"]

        emb_a = embedding_by_text[desc_a]
        emb_b = embedding_by_text[desc_b]

        cosine_values.append(float(np.dot(emb_a, emb_b)))
        rouge_values.append(float(rouge_l_f1(desc_a, desc_b)))

    pair_df = pair_df.copy()
    pair_df["sentence_transformer_cosine"] = cosine_values
    pair_df["rouge_l_f1"] = rouge_values

    return pair_df


def summarize_by_pair(mean_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        mean_df.groupby("pair", sort=False)[["sentence_transformer_cosine", "rouge_l_f1"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def save_boxplot(mean_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("sentence_transformer_cosine", "Sentence-transformer cosine"),
        ("rouge_l_f1", "ROUGE-L F1"),
    ]

    for metric, ylabel in metrics:
        labels = [PAIR_LABELS[pair] for pair in PAIR_ORDER]
        data = [
            mean_df.loc[mean_df["pair"] == label, metric].dropna().to_numpy()
            for label in labels
        ]

        plt.figure(figsize=(7.5, 4.5))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.ylabel(ylabel)
        plt.xlabel("Representation pair")
        plt.title(f"SQ1 {ylabel} by representation pair")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        stem = f"sq1_{metric}_boxplot_mean_over_runs"
        plt.savefig(figures_dir / f"{stem}.pdf")
        plt.savefig(figures_dir / f"{stem}.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=Path("results/generation"),
        help="Folder containing gen_run_XX/descriptions.jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/cheap_metrics_75k"),
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for figure files.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of generation runs.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic cosine similarity.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for sentence-transformer encoding.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading generation outputs...")
    generation_df = load_generation_rows(args.generation_root, args.runs)
    print(f"Loaded {len(generation_df):,} generation rows.")

    print("Building SQ1 pairwise comparisons...")
    pair_df = build_pair_rows(generation_df)
    print(f"Built {len(pair_df):,} SQ1 pair rows.")

    print("Computing SQ1 cheap metrics...")
    scored_df = compute_sq1_metrics(pair_df, args.embedding_model, args.batch_size)

    all_runs_path = args.out_dir / "sq1_pairwise_metrics_all_runs.csv"
    scored_df.to_csv(all_runs_path, index=False)
    print(f"Wrote: {all_runs_path}")

    mean_df = (
        scored_df.groupby(["sample_id", "rep_a", "rep_b", "pair"], sort=False)[
            ["sentence_transformer_cosine", "rouge_l_f1"]
        ]
        .mean()
        .reset_index()
    )

    mean_path = args.out_dir / "sq1_pairwise_metrics_mean_over_runs.csv"
    mean_df.to_csv(mean_path, index=False)
    print(f"Wrote: {mean_path}")

    summary_df = summarize_by_pair(mean_df)
    summary_path = args.out_dir / "sq1_summary_by_pair.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")

    save_boxplot(mean_df, args.figures_dir)
    print(f"Wrote SQ1 figures to: {args.figures_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
