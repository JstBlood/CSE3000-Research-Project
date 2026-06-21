#!/usr/bin/env python3
"""
Create one combined SQ1 boxplot figure with:
  1. sentence-transformer cosine similarity
  2. ROUGE-L F1
  3. Prometheus expected score

Default inputs match the current repo filenames:
  results/cheap_metrics_75k/sq1_cheap_metrics_mean_over_runs.csv
  results/prometheus_600sample_gen_run_01_v3/sq1_prometheus_judge_scores_decimal_600sample_gen_run_01_v3.jsonl

Default outputs:
  results/figures/sq1_combined_metric_boxplots.pdf
  results/figures/sq1_combined_metric_boxplots.png

Design choices:
  - no overall plot title
  - metric names shown as subplot titles on top
  - no x-axis labels or x-axis tick text
  - visible color legend above the plots, with layout space reserved so it does not overlap
  - dynamic y-axis limits so tightly clustered distributions do not look squished
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

PAIR_ORDER = [
    "Binary–Assembly",
    "Binary–Source",
    "Assembly–Source",
]

PAIR_COLORS = {
    "Binary–Assembly": "#cfe8f3",
    "Binary–Source": "#d7f0d0",
    "Assembly–Source": "#f9dfc7",
}

MEDIAN_COLOR = "#e67e22"

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
    "assembly code": "assembly",
    "assembly_code": "assembly",
    "s": "source",
    "src": "source",
    "source": "source",
    "source code": "source",
    "source_code": "source",
}

PAIR_COL_CANDIDATES = [
    "pair",
    "representation_pair",
    "rep_pair",
    "comparison",
    "pair_label",
    "metadata.pair",
    "metadata.representation_pair",
    "metadata.rep_pair",
    "input.pair",
    "input.representation_pair",
    "input.rep_pair",
    "original_input.pair",
    "original_input.representation_pair",
    "original_input.rep_pair",
    "judge_input.pair",
    "judge_input.representation_pair",
    "judge_input.rep_pair",
]

LEFT_REP_COL_CANDIDATES = [
    "rep_a",
    "representation_a",
    "left_representation",
    "representation_left",
    "description_a_representation",
    "input.rep_a",
    "input.representation_a",
    "metadata.rep_a",
    "metadata.representation_a",
    "original_input.rep_a",
    "original_input.representation_a",
]

RIGHT_REP_COL_CANDIDATES = [
    "rep_b",
    "representation_b",
    "right_representation",
    "representation_right",
    "description_b_representation",
    "input.rep_b",
    "input.representation_b",
    "metadata.rep_b",
    "metadata.representation_b",
    "original_input.rep_b",
    "original_input.representation_b",
]

COSINE_COL_CANDIDATES = [
    "sentence_transformer_cosine",
    "cosine",
    "cosine_similarity",
    "sentence_transformer_cosine_similarity",
    "semantic_cosine",
    "st_cosine",
    "mean_cosine",
    "cosine_mean",
]

ROUGE_COL_CANDIDATES = [
    "rouge_l_f1",
    "rougeL_f1",
    "rouge_l",
    "rouge_l_score",
    "rouge_l_fmeasure",
    "rougel_f1",
    "rouge_f1",
    "mean_rouge_l_f1",
    "rouge_l_f1_mean",
]

PROM_SCORE_COL_CANDIDATES = [
    "score_expected",
    "expected_score",
    "calibrated_expected_score",
    "calibrated_score_expected",
    "score_expected_calibrated",
    "prometheus_score_expected",
    "prometheus_expected_score",
    "judge_score_expected",
    "score",
    "mean_score",
    "result.score_expected",
    "result.expected_score",
    "scores.score_expected",
    "scores.expected_score",
    "metadata.score_expected",
]


def read_jsonl_flat(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_no} of {path}: {exc}') from exc
    if not rows:
        raise ValueError(f'No rows loaded from JSONL: {path}')
    return pd.json_normalize(rows, sep='.')


def first_existing_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    columns_list = [str(c) for c in columns]
    lowered = {c.lower(): c for c in columns_list}
    for candidate in candidates:
        if candidate in columns_list:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_rep(value: object) -> str:
    raw = str(value).strip().lower()
    key = raw.replace('-', '_').replace(' ', '_')
    if raw in REP_ALIASES:
        return REP_ALIASES[raw]
    if key in REP_ALIASES:
        return REP_ALIASES[key]
    if 'binary' in key or 'disasm' in key:
        return 'binary'
    if 'assembly' in key or key == 'asm':
        return 'assembly'
    if 'source' in key or key == 'src':
        return 'source'
    return key


def canonical_pair_from_reps(left: object, right: object) -> str | None:
    l = normalize_rep(left)
    r = normalize_rep(right)
    reps = {l, r}
    if reps == {'binary', 'assembly'}:
        return 'Binary–Assembly'
    if reps == {'binary', 'source'}:
        return 'Binary–Source'
    if reps == {'assembly', 'source'}:
        return 'Assembly–Source'
    return None


def normalize_pair(value: object) -> str | None:
    raw = str(value).strip()
    if raw in PAIR_ORDER:
        return raw

    key = raw.lower()
    key = key.replace('—', '–')
    key = key.replace('--', '–')
    key = key.replace('-', '–')
    key = key.replace('_', ' ')
    key = ' '.join(key.split())

    direct_aliases = {
        'binary–assembly': 'Binary–Assembly',
        'binary assembly': 'Binary–Assembly',
        'binary vs assembly': 'Binary–Assembly',
        'b a': 'Binary–Assembly',
        'b–a': 'Binary–Assembly',
        'binary–source': 'Binary–Source',
        'binary source': 'Binary–Source',
        'binary vs source': 'Binary–Source',
        'b s': 'Binary–Source',
        'b–s': 'Binary–Source',
        'assembly–source': 'Assembly–Source',
        'assembly source': 'Assembly–Source',
        'assembly vs source': 'Assembly–Source',
        'a s': 'Assembly–Source',
        'a–s': 'Assembly–Source',
    }
    if key in direct_aliases:
        return direct_aliases[key]

    for sep in ['–', '/', '|', ',', ' vs ']:
        if sep in key:
            parts = [p.strip() for p in key.split(sep) if p.strip()]
            if len(parts) == 2:
                return canonical_pair_from_reps(parts[0], parts[1])
    return None


def infer_pair_column(df: pd.DataFrame, explicit: str | None, label: str) -> str | None:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f'{label}: requested pair column not found: {explicit}\nAvailable columns: {list(df.columns)}')
        return explicit

    col = first_existing_column(df.columns, PAIR_COL_CANDIDATES)
    if col is not None:
        return col

    pair_like_cols = [c for c in df.columns if 'pair' in str(c).lower() or 'comparison' in str(c).lower()]
    for candidate in pair_like_cols:
        normalized = df[candidate].dropna().head(50).map(normalize_pair)
        if len(normalized) > 0 and normalized.notna().mean() >= 0.8:
            return candidate
    return None


def add_normalized_pair(df: pd.DataFrame, explicit_pair_col: str | None, label: str) -> pd.DataFrame:
    df = df.copy()
    pair_col = infer_pair_column(df, explicit_pair_col, label)

    if pair_col is not None:
        df['pair_normalized'] = df[pair_col].map(normalize_pair)
    else:
        left_col = first_existing_column(df.columns, LEFT_REP_COL_CANDIDATES)
        right_col = first_existing_column(df.columns, RIGHT_REP_COL_CANDIDATES)
        if left_col is None or right_col is None:
            raise ValueError(
                f'{label}: could not infer pair information.\n'
                f'Available columns: {list(df.columns)}\n'
                'Pass --cheap-pair-col or --prom-pair-col.'
            )
        df['pair_normalized'] = [
            canonical_pair_from_reps(left, right)
            for left, right in zip(df[left_col], df[right_col])
        ]

    missing = int(df['pair_normalized'].isna().sum())
    if missing:
        examples = []
        if pair_col is not None:
            examples = sorted(df.loc[df['pair_normalized'].isna(), pair_col].astype(str).unique())[:10]
        raise ValueError(
            f'{label}: could not normalize {missing} pair values.\n'
            f'Problem examples: {examples}\n'
            f'Available columns: {list(df.columns)}'
        )

    df = df[df['pair_normalized'].isin(PAIR_ORDER)].copy()
    return df


def get_numeric_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], label: str) -> str:
    col = explicit or first_existing_column(df.columns, candidates)
    if col is None:
        raise ValueError(f'Could not infer {label} column.\nAvailable columns: {list(df.columns)}')
    if col not in df.columns:
        raise ValueError(f'Requested {label} column not found: {col}\nAvailable columns: {list(df.columns)}')

    df[col] = pd.to_numeric(df[col], errors='coerce')
    n_numeric = int(df[col].notna().sum())
    if n_numeric == 0:
        raise ValueError(f'{label} column contains no numeric values: {col}')
    return col


def values_by_pair(df: pd.DataFrame, value_col: str) -> list[list[float]]:
    out: list[list[float]] = []
    for pair in PAIR_ORDER:
        values = df.loc[df['pair_normalized'] == pair, value_col].dropna().astype(float).tolist()
        if not values:
            raise ValueError(f'No values found for pair {pair} in column {value_col}')
        out.append(values)
    return out


def style_boxplot(boxplot: dict, pair_labels: list[str]) -> None:
    for patch, pair in zip(boxplot['boxes'], pair_labels):
        patch.set_facecolor(PAIR_COLORS[pair])
        patch.set_edgecolor('#222222')
        patch.set_linewidth(1.0)

    for median in boxplot['medians']:
        median.set_color('#e67e22')
        median.set_linewidth(1.5)

    for whisker in boxplot['whiskers']:
        whisker.set_color('#222222')
        whisker.set_linewidth(0.9)

    for cap in boxplot['caps']:
        cap.set_color('#222222')
        cap.set_linewidth(0.9)


def compute_dynamic_ylim(values: list[list[float]], hard_min: float, hard_max: float, min_span_fraction: float = 0.16) -> tuple[float, float]:
    flat = [float(v) for group in values for v in group]
    if not flat:
        return hard_min, hard_max

    observed_min = min(flat)
    observed_max = max(flat)
    observed_span = observed_max - observed_min
    full_span = hard_max - hard_min

    if observed_span <= 0:
        center = observed_min
        span = full_span * max(min_span_fraction, 0.10)
    else:
        span = max(observed_span * 1.35, full_span * min_span_fraction)
        center = 0.5 * (observed_min + observed_max)

    lower = center - span / 2.0
    upper = center + span / 2.0

    if lower < hard_min:
        shift = hard_min - lower
        lower += shift
        upper += shift
    if upper > hard_max:
        shift = upper - hard_max
        lower -= shift
        upper -= shift

    lower = max(hard_min, lower)
    upper = min(hard_max, upper)

    min_span = full_span * min_span_fraction
    if (upper - lower) < min_span:
        needed = min_span - (upper - lower)
        expand_down = min(needed / 2.0, lower - hard_min)
        expand_up = min(needed / 2.0, hard_max - upper)
        lower -= expand_down
        upper += expand_up
        remaining = min_span - (upper - lower)
        if remaining > 0:
            extra_down = min(remaining, lower - hard_min)
            lower -= extra_down
            remaining -= extra_down
        if remaining > 0:
            extra_up = min(remaining, hard_max - upper)
            upper += extra_up

    return lower, upper


def draw_panel(ax, values: list[list[float]], panel_title: str, hard_min: float, hard_max: float, min_span_fraction: float) -> None:
    bp = ax.boxplot(
        values,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        labels=['', '', ''],
    )
    style_boxplot(bp, PAIR_ORDER)

    ax.set_title(panel_title, fontsize=10, pad=8)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='x', which='both', length=0, labelbottom=False)
    ax.set_ylim(*compute_dynamic_ylim(values, hard_min, hard_max, min_span_fraction=min_span_fraction))
    ax.grid(axis='y', linestyle='--', alpha=0.30)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot combined SQ1 cheap metrics and Prometheus boxplots.')
    parser.add_argument('--cheap-input', default='results/cheap_metrics_75k/sq1_cheap_metrics_mean_over_runs.csv', help='CSV containing SQ1 cheap metric distributions.')
    parser.add_argument('--prometheus-input', default='results/prometheus_600sample_gen_run_01_v3/sq1_prometheus_judge_scores_decimal_600sample_gen_run_01_v3.jsonl', help='JSONL containing SQ1 Prometheus judgments.')
    parser.add_argument('--output-pdf', default='results/figures/sq1_combined_metric_boxplots.pdf', help='Output PDF path.')
    parser.add_argument('--output-png', default='results/figures/sq1_combined_metric_boxplots.png', help='Output PNG path.')
    parser.add_argument('--cheap-pair-col', default=None)
    parser.add_argument('--prom-pair-col', default=None)
    parser.add_argument('--cosine-col', default=None)
    parser.add_argument('--rouge-col', default=None)
    parser.add_argument('--prom-score-col', default=None)
    parser.add_argument('--min-span-fraction', type=float, default=0.16, help='Minimum visible y-axis span as a fraction of the full metric range.')
    args = parser.parse_args()

    cheap_input = Path(args.cheap_input)
    prom_input = Path(args.prometheus_input)
    output_pdf = Path(args.output_pdf)
    output_png = Path(args.output_png)

    if not cheap_input.exists():
        raise FileNotFoundError(f'Cheap metrics input not found: {cheap_input}')
    if not prom_input.exists():
        raise FileNotFoundError(f'Prometheus input not found: {prom_input}')

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    cheap = pd.read_csv(cheap_input)
    cheap.columns = [str(c).strip() for c in cheap.columns]
    cheap = add_normalized_pair(cheap, args.cheap_pair_col, 'cheap metrics')
    cosine_col = get_numeric_column(cheap, args.cosine_col, COSINE_COL_CANDIDATES, 'sentence-transformer cosine')
    rouge_col = get_numeric_column(cheap, args.rouge_col, ROUGE_COL_CANDIDATES, 'ROUGE-L F1')

    prom = read_jsonl_flat(prom_input)
    prom.columns = [str(c).strip() for c in prom.columns]
    prom = add_normalized_pair(prom, args.prom_pair_col, 'Prometheus')
    prom_score_col = get_numeric_column(prom, args.prom_score_col, PROM_SCORE_COL_CANDIDATES, 'Prometheus expected score')

    cosine_values = values_by_pair(cheap, cosine_col)
    rouge_values = values_by_pair(cheap, rouge_col)
    prom_values = values_by_pair(prom, prom_score_col)

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'axes.titlesize': 10,
    })

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.55))
    draw_panel(axes[0], cosine_values, 'Cosine similarity', 0.0, 1.0, args.min_span_fraction)
    draw_panel(axes[1], rouge_values, 'ROUGE-L F1', 0.0, 1.0, args.min_span_fraction)
    draw_panel(axes[2], prom_values, 'Prometheus score', 1.0, 5.0, args.min_span_fraction)

    legend_handles = [
        mpatches.Patch(facecolor=PAIR_COLORS[pair], edgecolor='#222222', label=pair)
        for pair in PAIR_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 0.975),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84), w_pad=2.1)

    fig.savefig(output_pdf, bbox_inches='tight')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Wrote:')
    print(f'  {output_pdf}')
    print(f'  {output_png}')
    print('\nDetected columns:')
    print(f'  Cheap input: {cheap_input}')
    print(f'  Prometheus input: {prom_input}')
    print(f'  Cosine column: {cosine_col}')
    print(f'  ROUGE-L column: {rouge_col}')
    print(f'  Prometheus score column: {prom_score_col}')


if __name__ == '__main__':
    main()
