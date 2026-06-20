#!/usr/bin/env python3
"""
Compute SQ2 BERTScore metrics over all generation runs.

SQ2 = reference-based quality/alignment between generated descriptions
and the SBAN natural-language reference description.

Input:
  results/generation/gen_run_01/descriptions.jsonl
  ...
  results/generation/gen_run_05/descriptions.jsonl

Output:
  results/cheap_metrics_75k/sq2_bertscore_all_runs.csv
  results/cheap_metrics_75k/sq2_bertscore_mean_over_runs.csv
  results/cheap_metrics_75k/sq2_summary_by_representation.csv
  results/figures/sq2_bertscore_*_boxplot_mean_over_runs.pdf/png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from bert_score import BERTScorer
from tqdm import tqdm


REP_ORDER = ["binary", "assembly", "source"]

REP_LABELS = {
    "binary": "Binary",
    "assembly": "Assembly",
    "source": "Source",
}

REP_ALIASES = {
    "binary": "binary",
    "bin": "binary",
    "binary/disassembly": "binary",
    "binary_disassembly": "binary",
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


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


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
                generated_description = pick_first(
                    row,
                    [
                        "generated_description",
                        "description",
                        "generated_nld",
                        "model_description",
                        "output",
                    ],
                )
                reference_description = pick_first(
                    row,
                    [
                        "reference_nld",
                        "reference_description",
                        "reference",
                        "nld",
                        "target_description",
                        "gold_description",
                    ],
                )

                if sample_id is None:
                    raise KeyError(f"{path}:{line_no} has no sample identifier field.")
                if representation is None:
                    raise KeyError(f"{path}:{line_no} has no representation field.")
                if generated_description is None:
                    raise KeyError(f"{path}:{line_no} has no generated description field.")
                if reference_description is None:
                    raise KeyError(f"{path}:{line_no} has no reference description field.")

                rep = normalize_representation(representation)

                rows.append(
                    {
                        "eval_id": f"{run_id}::{sample_id}::{rep}",
                        "run_id": run_id,
                        "sample_id": str(sample_id),
                        "representation": rep,
                        "representation_label": REP_LABELS[rep],
                        "generated_description": str(generated_description).strip(),
                        "reference_description": str(reference_description).strip(),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No generation rows loaded.")

    return df


def read_completed_eval_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    try:
        df = pd.read_csv(path, usecols=["eval_id"])
        return set(df["eval_id"].astype(str))
    except Exception:
        return set()


def append_csv(df: pd.DataFrame, path: Path) -> None:
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def run_bertscore(
    df: pd.DataFrame,
    out_path: Path,
    model_type: str,
    device: str,
    batch_size: int,
    chunk_size: int,
    rescale_with_baseline: bool,
    overwrite: bool,
) -> pd.DataFrame:
    if overwrite and out_path.exists():
        out_path.unlink()

    completed = read_completed_eval_ids(out_path)
    todo = df.loc[~df["eval_id"].isin(completed)].copy()

    print(f"Total rows: {len(df):,}")
    print(f"Already completed: {len(completed):,}")
    print(f"Rows to score: {len(todo):,}")

    if len(todo) > 0:
        print(f"Loading BERTScorer model: {model_type}")
        print(f"Device: {device}")
        scorer = BERTScorer(
            model_type=model_type,
            lang="en",
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )

        for start in tqdm(range(0, len(todo), chunk_size), desc="BERTScore chunks"):
            chunk = todo.iloc[start:start + chunk_size].copy()

            candidates = chunk["generated_description"].tolist()
            references = chunk["reference_description"].tolist()

            precision, recall, f1 = scorer.score(
                candidates,
                references,
                batch_size=batch_size,
                verbose=False,
            )

            chunk["bertscore_precision"] = precision.detach().cpu().numpy()
            chunk["bertscore_recall"] = recall.detach().cpu().numpy()
            chunk["bertscore_f1"] = f1.detach().cpu().numpy()

            append_csv(chunk, out_path)

    result_df = pd.read_csv(out_path)
    return result_df


def summarize_by_representation(mean_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        mean_df.groupby("representation_label", sort=False)[
            ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
        ]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def save_boxplots(mean_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("bertscore_precision", "BERTScore precision"),
        ("bertscore_recall", "BERTScore recall"),
        ("bertscore_f1", "BERTScore F1"),
    ]

    labels = [REP_LABELS[rep] for rep in REP_ORDER]

    for metric, ylabel in metrics:
        data = [
            mean_df.loc[mean_df["representation_label"] == label, metric].dropna().to_numpy()
            for label in labels
        ]

        plt.figure(figsize=(7.0, 4.5))
        plt.boxplot(data, tick_labels=labels, showfliers=False)
        plt.ylabel(ylabel)
        plt.xlabel("Representation")
        plt.title(f"SQ2 {ylabel} by representation")
        plt.tight_layout()

        stem = f"sq2_{metric}_boxplot_mean_over_runs"
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
        "--model-type",
        type=str,
        default="roberta-large",
        help="BERTScore backbone model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="BERTScore model batch size.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Number of rows scored before appending to CSV.",
    )
    parser.add_argument(
        "--rescale-with-baseline",
        action="store_true",
        help="Use BERTScore baseline rescaling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing SQ2 all-runs CSV instead of resuming.",
    )
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    print("Loading generation outputs...")
    generation_df = load_generation_rows(args.generation_root, args.runs)
    print(f"Loaded {len(generation_df):,} generation rows.")

    all_runs_path = args.out_dir / "sq2_bertscore_all_runs.csv"

    scored_df = run_bertscore(
        df=generation_df,
        out_path=all_runs_path,
        model_type=args.model_type,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        rescale_with_baseline=args.rescale_with_baseline,
        overwrite=args.overwrite,
    )

    print(f"Wrote/resumed: {all_runs_path}")

    mean_df = (
        scored_df.groupby(
            ["sample_id", "representation", "representation_label"],
            sort=False,
        )[["bertscore_precision", "bertscore_recall", "bertscore_f1"]]
        .mean()
        .reset_index()
    )

    mean_path = args.out_dir / "sq2_bertscore_mean_over_runs.csv"
    mean_df.to_csv(mean_path, index=False)
    print(f"Wrote: {mean_path}")

    summary_df = summarize_by_representation(mean_df)
    summary_path = args.out_dir / "sq2_summary_by_representation.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")

    save_boxplots(mean_df, args.figures_dir)
    print(f"Wrote SQ2 figures to: {args.figures_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
