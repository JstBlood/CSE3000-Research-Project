#!/usr/bin/env python3
"""
Compute and plot SQ1 cheap metrics over all generation runs.

SQ1 = cross-representation consistency between generated descriptions
for the same sample and generation run.

Expected input:
  results/generation/gen_run_01/descriptions.jsonl
  ...
  results/generation/gen_run_05/descriptions.jsonl

Outputs:
  results/cheap_metrics_75k/sq1_cheap_metrics_all_runs.csv
  results/cheap_metrics_75k/sq1_cheap_metrics_mean_over_runs.csv
  results/cheap_metrics_75k/sq1_cheap_metrics_summary_by_pair.csv
  results/figures/sq1_cheap_metric_boxplots_mean_over_runs.pdf
  results/figures/sq1_cheap_metric_boxplots_mean_over_runs.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PAIR_ORDER = [
    "Binary–Assembly",
    "Binary–Source",
    "Assembly–Source",
]

PAIR_REPS = [
    ("binary", "assembly", "Binary–Assembly"),
    ("binary", "source", "Binary–Source"),
    ("assembly", "source", "Assembly–Source"),
]

PAIR_COLORS = {
    "Binary–Assembly": "#cfe8f3",
    "Binary–Source": "#d7f0d0",
    "Assembly–Source": "#f9dfc7",
}

MEDIAN_COLOR = "#e67e22"

REP_ALIASES = {
    "binary": "binary",
    "bin": "binary",
    "binary/disassembly": "binary",
    "binary_disassembly": "binary",
    "disassembly": "binary",
    "disasm": "binary",
    "assembly": "assembly",
    "asm": "assembly",
    "assembly code": "assembly",
    "assembly_code": "assembly",
    "source": "source",
    "src": "source",
    "source code": "source",
    "source_code": "source",
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
        return "binary"
    if "assembly" in key or key == "asm":
        return "assembly"
    if "source" in key or key == "src":
        return "source"

    raise ValueError(f"Unknown representation value: {value!r}")


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
                description = first_present(
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


def build_sq1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    incomplete = 0

    for (run_id, sample_id), group in tqdm(
        df.groupby(["run_id", "sample_id"], sort=False),
        desc="Building SQ1 pairs",
    ):
        by_rep = {
            row["representation"]: row["generated_description"]
            for _, row in group.iterrows()
        }

        if not all(rep in by_rep for rep in ["binary", "assembly", "source"]):
            incomplete += 1
            continue

        for rep_a, rep_b, pair_label in PAIR_REPS:
            records.append(
                {
                    "eval_id": f"{run_id}::{sample_id}::{rep_a}::{rep_b}",
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "rep_a": rep_a,
                    "rep_b": rep_b,
                    "pair": pair_label,
                    "description_a": by_rep[rep_a],
                    "description_b": by_rep[rep_b],
                }
            )

    if incomplete:
        print(f"Skipped incomplete sample/run groups: {incomplete}")

    pair_df = pd.DataFrame(records)
    if pair_df.empty:
        raise ValueError("No SQ1 pairs could be built.")

    return pair_df


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    # Use the shorter sequence for the DP width.
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


def compute_sentence_transformer_cosines(
    pair_df: pd.DataFrame,
    model_name: str,
    batch_size: int,
) -> list[float]:
    all_texts = pair_df["description_a"].tolist() + pair_df["description_b"].tolist()
    unique_texts = list(dict.fromkeys(all_texts))

    print(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_by_text = {text: emb for text, emb in zip(unique_texts, embeddings)}

    scores: list[float] = []
    for _, row in tqdm(pair_df.iterrows(), total=len(pair_df), desc="Cosine similarity"):
        emb_a = embedding_by_text[row["description_a"]]
        emb_b = embedding_by_text[row["description_b"]]
        scores.append(float(np.dot(emb_a, emb_b)))

    return scores


def compute_metrics(pair_df: pd.DataFrame, model_name: str, batch_size: int) -> pd.DataFrame:
    scored = pair_df.copy()
    scored["sentence_transformer_cosine"] = compute_sentence_transformer_cosines(
        scored,
        model_name=model_name,
        batch_size=batch_size,
    )

    scored["rouge_l_f1"] = [
        rouge_l_f1(a, b)
        for a, b in tqdm(
            zip(scored["description_a"], scored["description_b"]),
            total=len(scored),
            desc="ROUGE-L F1",
        )
    ]

    return scored


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
        patch.set_facecolor(PAIR_COLORS[label])
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
        ("sentence_transformer_cosine", "Sentence-transformer cosine"),
        ("rouge_l_f1", "ROUGE-L F1"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=False)

    for ax, (metric, title) in zip(axes, metrics):
        data = [
            mean_df.loc[mean_df["pair"] == pair, metric].dropna().to_numpy()
            for pair in PAIR_ORDER
        ]

        bp = ax.boxplot(
            data,
            labels=["", "", ""],
            showfliers=False,
            patch_artist=True,
            widths=0.55,
        )
        style_boxplot(bp, PAIR_ORDER)

        ax.set_title(title, fontsize=12, pad=8)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(*dynamic_ylim(data, 0.0, 1.0))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=PAIR_COLORS[label], edgecolor="#222222", label=label)
        for label in PAIR_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    pdf_path = figures_dir / "sq1_cheap_metric_boxplots_mean_over_runs.pdf"
    png_path = figures_dir / "sq1_cheap_metric_boxplots_mean_over_runs.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


def save_summary(mean_df: pd.DataFrame, out_path: Path) -> None:
    summary = (
        mean_df.groupby("pair")[["sentence_transformer_cosine", "rouge_l_f1"]]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary["pair"] = pd.Categorical(summary["pair"], categories=PAIR_ORDER, ordered=True)
    summary = summary.sort_values("pair")
    summary.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, default=Path("results/generation"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/cheap_metrics_75k"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading generation outputs...")
    generation_df = load_generation_rows(args.generation_root, args.runs)
    print(f"Loaded generation rows: {len(generation_df):,}")

    print("Building SQ1 pair rows...")
    pair_df = build_sq1_pairs(generation_df)
    print(f"Built SQ1 pair rows: {len(pair_df):,}")

    print("Computing SQ1 metrics...")
    scored_df = compute_metrics(pair_df, args.embedding_model, args.batch_size)

    all_runs_path = args.out_dir / "sq1_cheap_metrics_all_runs.csv"
    scored_df.to_csv(all_runs_path, index=False)
    print(f"Wrote: {all_runs_path}")

    mean_df = (
        scored_df.groupby(["sample_id", "rep_a", "rep_b", "pair"], sort=False)[
            ["sentence_transformer_cosine", "rouge_l_f1"]
        ]
        .mean()
        .reset_index()
    )

    mean_path = args.out_dir / "sq1_cheap_metrics_mean_over_runs.csv"
    mean_df.to_csv(mean_path, index=False)
    print(f"Wrote: {mean_path}")

    save_summary(mean_df, args.out_dir / "sq1_cheap_metrics_summary_by_pair.csv")
    save_boxplots(mean_df, args.figures_dir)

    print("Done.")


if __name__ == "__main__":
    main()
