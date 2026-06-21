#!/usr/bin/env python3
"""
Compute the Friedman test for SQ2 BERTScore F1 across representations.

Input:
  results/cheap_metrics_75k/sq2_bertscore_mean_over_runs.csv

Expected data shape:
  one row per sample and representation after averaging over generation runs.

Default outputs:
  results/cheap_metrics_75k/statistics/
    sq2_friedman_bertscore_f1_summary.csv
    sq2_friedman_bertscore_f1.json
    sq2_friedman_bertscore_f1_latex.txt

No post-hoc test is computed here. This script reports only:
  - per-representation descriptive statistics
  - Friedman chi-square statistic
  - p-value
  - Kendall's W effect size

This script intentionally has no scipy dependency. For three representations,
the Friedman p-value uses the chi-square survival function with df=2:
  p = exp(-chi2 / 2)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


REPRESENTATION_ORDER = ["binary", "assembly", "source"]

REP_ALIASES = {
    "b": "binary",
    "bin": "binary",
    "binary": "binary",
    "binary/disassembly": "binary",
    "binary_disassembly": "binary",
    "disassembly": "binary",
    "disasm": "binary",
    "a": "assembly",
    "asm": "assembly",
    "assembly": "assembly",
    "s": "source",
    "src": "source",
    "source": "source",
    "source_code": "source",
    "source code": "source",
}


def first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    columns_list = list(columns)
    lowered = {c.lower(): c for c in columns_list}
    for candidate in candidates:
        if candidate in columns_list:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_rep(value: object) -> str:
    key = str(value).strip().lower().replace("-", "_")
    return REP_ALIASES.get(key, key)


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "<0.001"
    if p_value < 0.01:
        return f"{p_value:.4f}"
    return f"{p_value:.3f}"


def compute_friedman(pivot: pd.DataFrame) -> dict:
    """
    Compute Friedman chi-square statistic for three paired groups.

    Ranks are assigned within each sample. Larger metric values receive larger ranks.
    """
    values = pivot[REPRESENTATION_ORDER].astype(float)

    n = len(values)
    k = len(REPRESENTATION_ORDER)
    if k != 3:
        raise ValueError("This script expects exactly three representations.")
    if n < 2:
        raise ValueError("Need at least two complete paired samples.")

    ranks = values.rank(axis=1, method="average", ascending=True)
    rank_sums = ranks.sum(axis=0)
    mean_ranks = ranks.mean(axis=0)

    chi2 = (12.0 / (n * k * (k + 1))) * float((rank_sums ** 2).sum()) - 3.0 * n * (k + 1)
    chi2 = max(0.0, chi2)

    # For k=3, df = k - 1 = 2.
    # The chi-square(df=2) survival function is exp(-x/2).
    df = k - 1
    p_value = math.exp(-chi2 / 2.0)

    # Kendall's W effect size for Friedman test.
    kendalls_w = chi2 / (n * (k - 1))

    return {
        "n_complete_samples": int(n),
        "k_representations": int(k),
        "df": int(df),
        "friedman_chi2": float(chi2),
        "p_value": float(p_value),
        "kendalls_w": float(kendalls_w),
        "mean_ranks": {rep: float(mean_ranks[rep]) for rep in REPRESENTATION_ORDER},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Friedman test for SQ2 BERTScore F1 across binary, assembly, and source."
    )
    parser.add_argument(
        "--input",
        default="results/cheap_metrics_75k/sq2_bertscore_mean_over_runs.csv",
        help="Input CSV with one row per sample and representation after run averaging.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/cheap_metrics_75k/statistics",
        help="Directory where statistics files are written.",
    )
    parser.add_argument("--sample-col", default=None, help="Sample/program identifier column.")
    parser.add_argument("--representation-col", default=None, help="Representation column.")
    parser.add_argument("--metric-col", default=None, help="Metric column, normally BERTScore F1.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip() for c in df.columns]

    sample_col = args.sample_col or first_existing_column(
        df.columns,
        [
            "sample_id",
            "sample_sha256",
            "sha256",
            "program_id",
            "program",
            "id",
            "sample_index",
            "sample_idx",
            "idx",
        ],
    )
    representation_col = args.representation_col or first_existing_column(
        df.columns,
        [
            "representation",
            "rep",
            "input_representation",
            "representation_type",
            "source_representation",
        ],
    )
    metric_col = args.metric_col or first_existing_column(
        df.columns,
        [
            "bertscore_f1",
            "bert_score_f1",
            "bert_f1",
            "f1",
            "F1",
            "bertscore_f1_mean",
            "bert_f1_mean",
            "score_f1",
        ],
    )

    missing = []
    if sample_col is None:
        missing.append("sample column")
    if representation_col is None:
        missing.append("representation column")
    if metric_col is None:
        missing.append("metric column")
    if missing:
        raise ValueError(
            "Could not auto-detect "
            + ", ".join(missing)
            + f". Available columns: {list(df.columns)}. "
            + "Pass --sample-col, --representation-col, and/or --metric-col explicitly."
        )

    work = df[[sample_col, representation_col, metric_col]].copy()
    work.columns = ["sample_id", "representation", "metric"]
    work["representation"] = work["representation"].map(normalize_rep)
    work["metric"] = pd.to_numeric(work["metric"], errors="coerce")

    before_rows = len(work)
    work = work.dropna(subset=["sample_id", "representation", "metric"])
    dropped_rows = before_rows - len(work)

    unexpected_reps = sorted(set(work["representation"]) - set(REPRESENTATION_ORDER))
    if unexpected_reps:
        print(f"Warning: ignoring unexpected representations: {unexpected_reps}")
        work = work[work["representation"].isin(REPRESENTATION_ORDER)].copy()

    pivot = work.pivot_table(
        index="sample_id",
        columns="representation",
        values="metric",
        aggfunc="mean",
    )

    for rep in REPRESENTATION_ORDER:
        if rep not in pivot.columns:
            raise ValueError(f"Missing representation after pivot: {rep}")

    pivot = pivot[REPRESENTATION_ORDER]
    incomplete_samples = int(pivot.isna().any(axis=1).sum())
    pivot = pivot.dropna(axis=0, how="any")

    result = compute_friedman(pivot)

    summary = (
        pivot.describe()
        .T[["count", "mean", "std", "min", "50%", "max"]]
        .rename(columns={"50%": "median"})
        .reset_index()
        .rename(columns={"representation": "representation"})
    )
    summary["mean_rank"] = summary["representation"].map(result["mean_ranks"])

    result.update(
        {
            "input_csv": str(input_path),
            "sample_col": sample_col,
            "representation_col": representation_col,
            "metric_col": metric_col,
            "dropped_rows_with_missing_values": int(dropped_rows),
            "incomplete_samples_dropped": int(incomplete_samples),
        }
    )

    summary_path = output_dir / "sq2_friedman_bertscore_f1_summary.csv"
    json_path = output_dir / "sq2_friedman_bertscore_f1.json"
    latex_path = output_dir / "sq2_friedman_bertscore_f1_latex.txt"

    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    latex = (
        "% Copy-paste values for the paper:\n"
        f"\\(\\chi^2({result['df']}) = {result['friedman_chi2']:.2f}\\), "
        f"\\(p = {format_p_value(result['p_value'])}\\), "
        f"\\(N = {result['n_complete_samples']}\\), "
        f"\\(W = {result['kendalls_w']:.4f}\\).\n"
    )
    latex_path.write_text(latex, encoding="utf-8")

    print("\nSQ2 Friedman test over BERTScore F1")
    print("-----------------------------------")
    print(f"Input: {input_path}")
    print(f"Sample column: {sample_col}")
    print(f"Representation column: {representation_col}")
    print(f"Metric column: {metric_col}")
    print(f"Complete paired samples: {result['n_complete_samples']}")
    print(f"Dropped rows with missing values: {dropped_rows}")
    print(f"Dropped incomplete paired samples: {incomplete_samples}")
    print(f"Friedman chi-square({result['df']}): {result['friedman_chi2']:.6f}")
    print(f"p-value: {result['p_value']:.12g}")
    print(f"Kendall's W: {result['kendalls_w']:.6f}")

    print("\nDescriptive statistics and mean ranks:")
    print(summary.to_string(index=False))

    print("\nWrote:")
    print(f"  {summary_path}")
    print(f"  {json_path}")
    print(f"  {latex_path}")


if __name__ == "__main__":
    main()
