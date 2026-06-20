
import argparse
import json
from pathlib import Path
from typing import Any, Dict


REP_ORDER = ["binary", "assembly", "source"]

PAIR_SPECS = [
    ("binary", "assembly", "Binary–Assembly", "binary_assembly"),
    ("binary", "source", "Binary–Source", "binary_source"),
    ("assembly", "source", "Assembly–Source", "assembly_source"),
]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} on line {line_number}: {e}") from e


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_rep(rep: str) -> str:
    rep = str(rep).strip().lower()
    aliases = {
        "bin": "binary",
        "binary": "binary",
        "asm": "assembly",
        "assembly": "assembly",
        "source": "source",
        "src": "source",
        "source_code": "source",
    }
    return aliases.get(rep, rep)


def is_bad_description(desc: str) -> bool:
    desc = str(desc).strip()
    low = desc.lower()
    if not desc:
        return True
    if low.startswith("<think"):
        return True
    if "thinking process" in low[:80]:
        return True
    if len(desc.split()) < 4:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions", required=True)
    parser.add_argument("--judge-root", required=True)
    parser.add_argument("--generation-run-id", type=int, default=1)
    parser.add_argument("--expected-descriptions", type=int, default=15000)
    parser.add_argument("--expected-samples", type=int, default=5000)
    args = parser.parse_args()

    desc_path = Path(args.descriptions)
    judge_root = Path(args.judge_root)

    if not desc_path.exists():
        raise FileNotFoundError(desc_path)

    rows = list(read_jsonl(desc_path))
    print(f"Loaded descriptions: {len(rows)}")

    if len(rows) != args.expected_descriptions:
        raise SystemExit(
            f"ERROR: expected {args.expected_descriptions} descriptions in gen_run_01, found {len(rows)}. "
            f"Wait until generation run 01 is complete before preparing Prometheus inputs."
        )

    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = {}
    bad_rows = 0
    wrong_run = 0

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        rep = normalize_rep(row.get("representation", ""))
        desc = str(row.get("generated_description", "")).strip()
        run_id = int(row.get("generation_run_id", -1))
        status = str(row.get("generation_status", "ok"))

        if run_id != args.generation_run_id:
            wrong_run += 1

        if not sample_id or rep not in REP_ORDER or status != "ok" or is_bad_description(desc):
            bad_rows += 1
            continue

        by_sample.setdefault(sample_id, {})
        if rep in by_sample[sample_id]:
            raise SystemExit(f"ERROR: duplicate row for sample_id={sample_id}, representation={rep}")

        by_sample[sample_id][rep] = row

    if wrong_run:
        raise SystemExit(f"ERROR: found {wrong_run} rows with generation_run_id different from {args.generation_run_id}")

    if bad_rows:
        raise SystemExit(f"ERROR: found {bad_rows} invalid rows. Do not run Prometheus until all generated descriptions are valid.")

    complete_sample_ids = [
        sid for sid, reps in by_sample.items()
        if all(rep in reps for rep in REP_ORDER)
    ]

    complete_sample_ids.sort(key=lambda sid: int(by_sample[sid]["binary"].get("input_index", 10**18)))

    if len(complete_sample_ids) != args.expected_samples:
        raise SystemExit(
            f"ERROR: expected {args.expected_samples} complete samples, found {len(complete_sample_ids)}."
        )

    sq1_rows = []
    sq2_rows = []

    for sample_counter, sample_id in enumerate(complete_sample_ids, start=1):
        reps = by_sample[sample_id]
        input_index = int(reps["binary"].get("input_index", sample_counter))

        for rep_a, rep_b, pair_label, pair_key in PAIR_SPECS:
            row_a = reps[rep_a]
            row_b = reps[rep_b]

            sq1_rows.append({
                "eval_id": f"sq1_gen{args.generation_run_id:02d}_{sample_counter:05d}_{pair_key}_{sample_id[:12]}",
                "generation_run_id": args.generation_run_id,
                "sample_id": sample_id,
                "input_index": input_index,
                "pair": pair_label,
                "description_a_representation": rep_a,
                "description_b_representation": rep_b,
                "description_a": str(row_a["generated_description"]).strip(),
                "description_b": str(row_b["generated_description"]).strip(),
            })

        for rep in REP_ORDER:
            row = reps[rep]
            sq2_rows.append({
                "eval_id": f"sq2_gen{args.generation_run_id:02d}_{sample_counter:05d}_{rep}_{sample_id[:12]}",
                "generation_run_id": args.generation_run_id,
                "sample_id": sample_id,
                "input_index": input_index,
                "representation": rep,
                "generated_description": str(row["generated_description"]).strip(),
                "reference_nld": str(row.get("reference_nld", "")).strip(),
            })

    for row in sq2_rows:
        if not row["reference_nld"]:
            raise SystemExit(f"ERROR: empty reference_nld for eval_id={row['eval_id']}")

    input_dir = judge_root / "inputs"
    write_jsonl(input_dir / "sq1_inputs.jsonl", sq1_rows)
    write_jsonl(input_dir / "sq2_inputs.jsonl", sq2_rows)

    print(f"Complete samples: {len(complete_sample_ids)}")
    print(f"SQ1 inputs: {len(sq1_rows)}")
    print(f"SQ2 inputs: {len(sq2_rows)}")
    print(f"Wrote: {input_dir / 'sq1_inputs.jsonl'}")
    print(f"Wrote: {input_dir / 'sq2_inputs.jsonl'}")

    if len(sq1_rows) != 15000:
        raise SystemExit(f"ERROR: expected 15000 SQ1 rows, got {len(sq1_rows)}")
    if len(sq2_rows) != 15000:
        raise SystemExit(f"ERROR: expected 15000 SQ2 rows, got {len(sq2_rows)}")


if __name__ == "__main__":
    main()
