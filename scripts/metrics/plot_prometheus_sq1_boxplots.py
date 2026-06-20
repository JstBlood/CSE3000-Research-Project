#!/usr/bin/env python3
"""
Plot SQ1 Prometheus LLM-as-judge expected scores in the thesis boxplot style.

SQ1 = cross-representation consistency.

Input:
  results/prometheus_600sample_gen_run_01_v3/
    sq1_prometheus_judge_scores_decimal_600sample_gen_run_01_v3.jsonl

Outputs:
  results/prometheus_600sample_gen_run_01_v3/summary/sq1_prometheus_summary_by_pair.csv
  results/figures/sq1_prometheus_score_expected_by_pair_boxplot.pdf
  results/figures/sq1_prometheus_score_expected_by_pair_boxplot.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


PAIR_ORDER = ["Binary–Assembly", "Binary–Source", "Assembly–Source"]

PAIR_COLORS = {
    "Binary–Assembly": "#cfe8f3",
    "Binary–Source": "#d7f0d0",
    "Assembly–Source": "#f9dfc7",
}

MEDIAN_COLOR = "#e67e22"


def normalize_text(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-").replace("–", "-")


def first_present(row: dict, keys: list[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def infer_pair(row: dict) -> str:
    explicit = first_present(row, ["pair", "representation_pair", "pair_label"])
    if explicit is not None:
        text = normalize_text(explicit)
        if "binary" in text and "assembly" in text:
            return "Binary–Assembly"
        if "binary" in text and "source" in text:
            return "Binary–Source"
        if "assembly" in text and "source" in text:
            return "Assembly–Source"

    rep_a = first_present(
        row,
        ["rep_a", "representation_a", "left_representation", "description_a_representation"],
    )
    rep_b = first_present(
        row,
        ["rep_b", "representation_b", "right_representation", "description_b_representation"],
    )

    combined = f"{normalize_text(rep_a) if rep_a is not None else ''} {normalize_text(rep_b) if rep_b is not None else ''}"

    if not combined.strip():
        combined = normalize_text(row.get("eval_id", ""))

    has_binary = "binary" in combined or "disasm" in combined or "disassembly" in combined
    has_assembly = "assembly" in combined or "asm" in combined
    has_source = "source" in combined or "src" in combined

    if has_binary and has_assembly:
        return "Binary–Assembly"
    if has_binary and has_source:
        return "Binary–Source"
    if has_assembly and has_source:
        return "Assembly–Source"

    raise ValueError(f"Could not infer SQ1 pair for eval_id={row.get('eval_id')!r}")


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue

            row = json.loads(line)

            if "score_expected" not in row:
                raise KeyError(f"{path}:{line_no} missing score_expected")

            rows.append(
                {
                    "eval_id": row.get("eval_id"),
                    "sample_id": row.get("sample_id"),
                    "run_id": row.get("run_id"),
                    "pair": infer_pair(row),
                    "score_expected": float(row["score_expected"]),
                    "score_hard": float(row["score_hard"]) if row.get("score_hard") is not None else None,
                    "p1": float(row["p1"]) if row.get("p1") is not None else None,
                    "p2": float(row["p2"]) if row.get("p2") is not None else None,
                    "p3": float(row["p3"]) if row.get("p3") is not None else None,
                    "p4": float(row["p4"]) if row.get("p4") is not None else None,
                    "p5": float(row["p5"]) if row.get("p5") is not None else None,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No rows loaded from {path}")

    return df


def dynamic_ylim(values: pd.Series, score_min: float = 1.0, score_max: float = 5.0) -> tuple[float, float]:
    """Use a data-dependent y-axis so the boxes are not visually compressed."""
    clean = values.dropna().astype(float)
    if clean.empty:
        return score_min, score_max

    data_min = float(clean.min())
    data_max = float(clean.max())
    data_range = data_max - data_min

    # For concentrated distributions, use a minimum visible span.
    min_span = 0.75
    pad = max(0.08, data_range * 0.18)

    lower = data_min - pad
    upper = data_max + pad

    if upper - lower < min_span:
        centre = (data_min + data_max) / 2.0
        lower = centre - min_span / 2.0
        upper = centre + min_span / 2.0

    lower = max(score_min, math.floor(lower * 10) / 10)
    upper = min(score_max, math.ceil(upper * 10) / 10)

    if upper - lower < min_span:
        if lower <= score_min:
            upper = min(score_max, lower + min_span)
        elif upper >= score_max:
            lower = max(score_min, upper - min_span)

    return lower, upper


def save_summary(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby("pair")["score_expected"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )

    summary["pair"] = pd.Categorical(summary["pair"], categories=PAIR_ORDER, ordered=True)
    summary = summary.sort_values("pair")
    summary.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")


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


def save_boxplot(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = [
        df.loc[df["pair"] == pair, "score_expected"].dropna().to_numpy()
        for pair in PAIR_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    bp = ax.boxplot(
        data,
        labels=["", "", ""],
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    style_boxplot(bp, PAIR_ORDER)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(*dynamic_ylim(df["score_expected"]))
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
        bbox_to_anchor=(0.5, 0.94),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))

    pdf_path = figures_dir / "sq1_prometheus_score_expected_by_pair_boxplot.pdf"
    png_path = figures_dir / "sq1_prometheus_score_expected_by_pair_boxplot.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/prometheus_600sample_gen_run_01_v3/"
            "sq1_prometheus_judge_scores_decimal_600sample_gen_run_01_v3.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path(
            "results/prometheus_600sample_gen_run_01_v3/summary/"
            "sq1_prometheus_summary_by_pair.csv"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
    )
    args = parser.parse_args()

    df = load_jsonl(args.input)

    print(f"Loaded SQ1 rows: {len(df)}")
    print(df["pair"].value_counts().reindex(PAIR_ORDER))

    save_summary(df, args.summary_out)
    save_boxplot(df, args.figures_dir)


if __name__ == "__main__":
    main()
