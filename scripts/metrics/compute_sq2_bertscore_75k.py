#!/usr/bin/env python3
"""
Compute and plot SQ2 BERTScore metrics over all generation runs.

SQ2 = reference-based alignment between each generated description and
the SBAN natural-language reference description.

Expected input:
  results/generation/gen_run_01/descriptions.jsonl
  ...
  results/generation/gen_run_05/descriptions.jsonl

Outputs:
  results/cheap_metrics_75k/sq2_bertscore_all_runs.csv
  results/cheap_metrics_75k/sq2_bertscore_mean_over_runs.csv
  results/cheap_metrics_75k/sq2_bertscore_summary_by_representation.csv
  results/figures/sq2_bertscore_boxplots_mean_over_runs.pdf
  results/figures/sq2_bertscore_boxplots_mean_over_runs.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from bert_score import BERTScorer
from tqdm import tqdm


REP_ORDER = ["Binary", "Assembly", "Source"]

REP_COLORS = {
    "Binary": "#cfe8f3",
    "Assembly": "#d7f0d0",
    "Source": "#f9dfc7",
}

MEDIAN_COLOR = "#e67e22"

REP_ALIASES = {
    "binary": "Binary",
    "bin": "Binary",
    "binary/disassembly": "Binary",
    "binary_disassembly": "Binary",
    "disassembly": "Binary",
    "disasm": "Binary",
    "assembly": "Assembly",
    "asm": "Assembly",
    "assembly code": "Assembly",
    "assembly_code": "Assembly",
    "source": "Source",
    "src": "Source",
    "source code": "Source",
    "source_code": "Source",
}


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def normalize_representation(value: Any) -> str:
    raw = str(value).strip()
    lower = raw.lower()
    key = lower.replace("-", "_").replace(" ", "_")

    if lower in REP_ALIASES:
        return REP_ALIASES[lower]
    if key in REP_ALIASES:
        return REP_ALIASES[key]

    if "binary" in key or "disasm" in key or "disassembl" in key:
        return "Binary"
    if "assembly" in key or key == "asm":
        return "Assembly"
    if "source" in key or key == "src":
        return "Source"

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
    rows: list[dict[str, Any]] = []

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

                sample_id = first_present(
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
                representation = first_present(
                    row,
                    [
                        "representation",
                        "representation_type",
                        "input_representation",
                        "rep",
                        "view",
                    ],
                )
                generated_description = first_present(
                    row,
                    [
                        "generated_description",
                        "description",
                        "generated_nld",
                        "model_description",
                        "output",
                    ],
                )
                reference_description = first_present(
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

                rep_label = normalize_representation(representation)

                rows.append(
                    {
                        "eval_id": f"{run_id}::{sample_id}::{rep_label.lower()}",
                        "run_id": run_id,
                        "sample_id": str(sample_id),
                        "representation": rep_label,
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


def compute_bertscore(
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

    print(f"Total SQ2 rows: {len(df):,}")
    print(f"Already completed rows: {len(completed):,}")
    print(f"Rows to score: {len(todo):,}")

    if len(todo) > 0:
        print(f"Loading BERTScore model: {model_type}")
        print(f"Device: {device}")

        scorer = BERTScorer(
            model_type=model_type,
            lang="en",
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )

        for start in tqdm(range(0, len(todo), chunk_size), desc="BERTScore chunks"):
            chunk = todo.iloc[start:start + chunk_size].copy()

            precision, recall, f1 = scorer.score(
                chunk["generated_description"].tolist(),
                chunk["reference_description"].tolist(),
                batch_size=batch_size,
                verbose=False,
            )

            chunk["bertscore_precision"] = precision.detach().cpu().numpy()
            chunk["bertscore_recall"] = recall.detach().cpu().numpy()
            chunk["bertscore_f1"] = f1.detach().cpu().numpy()

            append_csv(chunk, out_path)

    if not out_path.exists():
        raise FileNotFoundError(f"BERTScore output was not created: {out_path}")

    result = pd.read_csv(out_path)

    for col in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def dynamic_ylim(values: list[np.ndarray], lower_bound: float = 0.0, upper_bound: float = 1.0) -> tuple[float, float]:
    combined = np.concatenate([v for v in values if len(v) > 0])
    if len(combined) == 0:
        return lower_bound, upper_bound

    vmin = float(np.nanmin(combined))
    vmax = float(np.nanmax(combined))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return lower_bound, upper_bound

    span = vmax - vmin
    min_span = 0.08
    if span < min_span:
        mid = (vmin + vmax) / 2
        vmin = mid - min_span / 2
        vmax = mid + min_span / 2
    else:
        pad = span * 0.12
        vmin -= pad
        vmax += pad

    return max(lower_bound, vmin), min(upper_bound, vmax)


def style_boxplot(boxplot: dict, labels: list[str]) -> None:
    for patch, label in zip(boxplot["boxes"], labels):
        patch.set_facecolor(REP_COLORS[label])
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.1)

    for median in boxplot["medians"]:
        median.set_color(MEDIAN_COLOR)
        median.set_linewidth(1.5)

    for whisker in boxplot["whiskers"]:
        whisker.set_color("#222222")
        whisker.set_linewidth(1.0)

    for cap in boxplot["caps"]:
        cap.set_color("#222222")
        cap.set_linewidth(1.0)


def save_boxplots(mean_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("bertscore_precision", "BERTScore Precision"),
        ("bertscore_recall", "BERTScore Recall"),
        ("bertscore_f1", "BERTScore F1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=False)

    for ax, (metric, title) in zip(axes, metrics):
        data = [
            mean_df.loc[mean_df["representation"] == rep, metric].dropna().to_numpy()
            for rep in REP_ORDER
        ]

        bp = ax.boxplot(
            data,
            labels=["", "", ""],
            showfliers=False,
            patch_artist=True,
            widths=0.55,
        )
        style_boxplot(bp, REP_ORDER)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(*dynamic_ylim(data, 0.0, 1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=REP_COLORS[label], edgecolor="#222222", label=label)
        for label in REP_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    pdf_path = figures_dir / "sq2_bertscore_boxplots_mean_over_runs.pdf"
    png_path = figures_dir / "sq2_bertscore_boxplots_mean_over_runs.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


def save_summary(mean_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        mean_df.groupby("representation")[["bertscore_precision", "bertscore_recall", "bertscore_f1"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary["representation"] = pd.Categorical(summary["representation"], categories=REP_ORDER, ordered=True)
    summary = summary.sort_values("representation")
    summary.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, default=Path("results/generation"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cheap_metrics_75k"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model-type", type=str, default="roberta-large")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--rescale-with-baseline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)

    print("Loading generation outputs...")
    generation_df = load_generation_rows(args.generation_root, args.runs)
    print(f"Loaded generation rows: {len(generation_df):,}")

    all_runs_path = args.out_dir / "sq2_bertscore_all_runs.csv"

    scored_df = compute_bertscore(
        df=generation_df,
        out_path=all_runs_path,
        model_type=args.model_type,
        device=device,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        rescale_with_baseline=args.rescale_with_baseline,
        overwrite=args.overwrite,
    )

    mean_df = (
        scored_df.groupby(["sample_id", "representation"], sort=False)[
            ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
        ]
        .mean()
        .reset_index()
    )

    mean_path = args.out_dir / "sq2_bertscore_mean_over_runs.csv"
    mean_df.to_csv(mean_path, index=False)
    print(f"Wrote: {mean_path}")

    save_summary(mean_df, args.out_dir / "sq2_bertscore_summary_by_representation.csv")
    save_boxplots(mean_df, args.figures_dir)

    print("Done.")


if __name__ == "__main__":
    main()
