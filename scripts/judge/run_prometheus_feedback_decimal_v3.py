from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


SCORES = [round(x / 10, 1) for x in range(10, 51)]
SCORE_STRINGS = [f"{s:.1f}" for s in SCORES]


SQ1_BASE_TEMPLATE = """###Task:
Rate the semantic similarity of two generated high-level program behaviour descriptions for the same program.

###Description A:
{description_a}

###Description B:
{description_b}

###Evaluation focus:
Compare the described program behaviour, not exact wording.
Consider whether both descriptions describe the same main action, affected object, condition, input/output, and effect.
Relevant behaviours may include file operations, registry access, process creation, service handling, networking, memory or string manipulation, configuration handling, cleanup, or control-flow behaviour.
Do not penalize harmless paraphrasing.
Penalize changed behaviour, missing main behaviour, unsupported extra behaviour, contradictions, and overly generic descriptions.
Use the full 1.0--5.0 scale. Choose the score that best reflects the rubric, using one decimal place when the quality falls between two rubric levels.

###Rubric:
1.0 = Different behaviour: the descriptions are unrelated, contradictory, or describe different main actions.
2.0 = Weak overlap: the descriptions share a small topic or object, but the main behaviour or effect is different.
3.0 = Partial similarity: the descriptions describe related behaviour, but one misses or changes an important action, condition, object, or effect.
4.0 = Mostly equivalent: the descriptions describe the same main behaviour, with only minor missing details or harmless abstraction differences.
5.0 = Equivalent: the descriptions express the same behaviour, including the main action and important effect, with no meaningful contradiction or unsupported extra behaviour.
"""


SQ2_BASE_TEMPLATE = """###Task:
Rate how well a generated program behaviour description matches a reference program behaviour description.

###Generated description:
{generated_description}

###Reference description:
{reference_nld}

###Evaluation focus:
Use the reference description as the grading target.
Compare program behaviour, not exact wording.
Check whether the generated description captures the same main action, affected object, condition, input/output, and effect as the reference.
Relevant behaviours may include file operations, registry access, process creation, service handling, networking, memory or string manipulation, configuration handling, cleanup, or control-flow behaviour.
Do not penalize harmless paraphrasing.
Penalize missing important behaviour, changed behaviour, unsupported added behaviour, contradictions, and overly generic descriptions.
Do not reward a description for being plausible if the behaviour is not supported by the reference.
Use the full 1.0--5.0 scale. Choose the score that best reflects the rubric, using one decimal place when the quality falls between two rubric levels.

###Rubric:
1.0 = Incorrect: unrelated, contradictory, meaningless, or describes a different main behaviour than the reference.
2.0 = Weak match: mentions a similar topic or object, but misses or changes the main action or effect.
3.0 = Partial match: captures part of the reference behaviour, but omits, adds, or changes an important action, condition, object, or effect.
4.0 = Mostly correct: captures the main reference behaviour, with only minor omissions, harmless abstraction, or small wording differences.
5.0 = Equivalent: semantically matches the reference behaviour, including the main action and important effect, with no unsupported additions or contradictions.
"""


FEEDBACK_PROMPT = """###Instruction:
Write brief feedback explaining the comparison according to the rubric.
Mention only the most important reason for the score.
Do not give a numeric score yet.
Do not add information not present in the descriptions.

Feedback:
"""


SCORE_PROMPT_TEMPLATE = """###Feedback:
{judge_feedback}

###Instruction:
Based on the rubric and feedback, choose one numeric score from 1.0 to 5.0.
The score may use at most one decimal place.
Return only the score.

[RESULT]
"""


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} on line {line_number}: {e}") from e


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_eval_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for row in read_jsonl(path):
        eval_id = str(row.get("eval_id", "")).strip()
        if eval_id and row.get("score_expected") is not None:
            done.add(eval_id)
    return done


def build_base_prompt(task: str, row: Dict[str, Any], prior: bool = False) -> str:
    if task == "sq1":
        if prior:
            return SQ1_BASE_TEMPLATE.format(
                description_a="<omitted>",
                description_b="<omitted>",
            )
        return SQ1_BASE_TEMPLATE.format(
            description_a=str(row["description_a"]).strip(),
            description_b=str(row["description_b"]).strip(),
        )

    if task == "sq2":
        if prior:
            return SQ2_BASE_TEMPLATE.format(
                generated_description="<omitted>",
                reference_nld="<omitted>",
            )
        return SQ2_BASE_TEMPLATE.format(
            generated_description=str(row["generated_description"]).strip(),
            reference_nld=str(row["reference_nld"]).strip(),
        )

    raise ValueError(f"Unknown task: {task}")


def clean_feedback(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = text.split("[RESULT]")[0].strip()
    text = re.sub(r"\s+", " ", text).strip()

    # Remove accidental score statements if the model disobeys the feedback prompt.
    text = re.sub(r"(score|rating)\s*[:=]\s*[1-5](?:\.\d)?\s*\.?$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"numeric score\s*[:=]\s*[1-5](?:\.\d)?\s*\.?$", "", text, flags=re.IGNORECASE).strip()

    if not text:
        text = "The comparison could not be explained clearly, so the score is based on the rubric text only."

    return text


def generate_feedback(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return clean_feedback(raw)


def candidate_avg_logprobs_batched(
    model,
    tokenizer,
    prompt: str,
    candidate_strings: List[str],
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    candidate_id_lists = [
        tokenizer(candidate, add_special_tokens=False).input_ids
        for candidate in candidate_strings
    ]

    scores = []

    for start in range(0, len(candidate_strings), batch_size):
        batch_candidate_ids = candidate_id_lists[start:start + batch_size]

        sequences = [prompt_ids + cand_ids for cand_ids in batch_candidate_ids]
        max_len = max(len(seq) for seq in sequences)

        input_rows = []
        attention_rows = []

        for seq in sequences:
            pad_len = max_len - len(seq)
            input_rows.append(seq + [pad_id] * pad_len)
            attention_rows.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits

        for b, cand_ids in enumerate(batch_candidate_ids):
            token_logprobs = []
            for j, tok_id in enumerate(cand_ids):
                pos = len(prompt_ids) + j - 1
                log_probs = torch.log_softmax(logits[b, pos, :], dim=-1)
                token_logprobs.append(float(log_probs[tok_id].detach().cpu()))

            if not token_logprobs:
                scores.append(float("-inf"))
            else:
                scores.append(float(np.mean(token_logprobs)))

        del input_ids, attention_mask, outputs, logits
        torch.cuda.empty_cache()

    return np.array(scores, dtype=np.float64)


def calibrated_distribution(row_logprobs: np.ndarray, prior_logprobs: np.ndarray) -> Dict[str, Any]:
    calibrated = row_logprobs - prior_logprobs

    max_val = np.max(calibrated)
    exp_vals = np.exp(calibrated - max_val)
    probs = exp_vals / np.sum(exp_vals)

    expected = float(sum(score * prob for score, prob in zip(SCORES, probs)))
    hard_index = int(np.argmax(probs))
    hard_score = float(SCORES[hard_index])

    prob_dict = {
        score_string: float(prob)
        for score_string, prob in zip(SCORE_STRINGS, probs)
    }

    p1 = float(sum(prob for score, prob in zip(SCORES, probs) if 1.0 <= score < 1.5))
    p2 = float(sum(prob for score, prob in zip(SCORES, probs) if 1.5 <= score < 2.5))
    p3 = float(sum(prob for score, prob in zip(SCORES, probs) if 2.5 <= score < 3.5))
    p4 = float(sum(prob for score, prob in zip(SCORES, probs) if 3.5 <= score < 4.5))
    p5 = float(sum(prob for score, prob in zip(SCORES, probs) if 4.5 <= score <= 5.0))

    return {
        "score_expected": expected,
        "score_hard": hard_score,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "score_probabilities_json": json.dumps(prob_dict, ensure_ascii=False),
    }


def load_model_and_tokenizer(model_path: str):
    print(f"Loading tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Prometheus model in 4-bit: {model_path}", flush=True)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["sq1", "sq2"])
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-run-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--max-feedback-tokens", type=int, default=128)
    parser.add_argument("--feedback-temperature", type=float, default=0.20)
    parser.add_argument("--feedback-top-p", type=float, default=0.90)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)

    rows = list(read_jsonl(input_path))
    done = load_done_eval_ids(output_path)

    print(f"Task: {args.task}", flush=True)
    print(f"Input rows: {len(rows)}", flush=True)
    print(f"Already done in output: {len(done)}", flush=True)
    print(f"Output: {output_path}", flush=True)

    model, tokenizer = load_model_and_tokenizer(args.model)

    prior_base = build_base_prompt(args.task, {}, prior=True)
    prior_score_prompt = prior_base + "\n" + SCORE_PROMPT_TEMPLATE.format(judge_feedback="<omitted>")
    prior_logprobs = candidate_avg_logprobs_batched(
        model=model,
        tokenizer=tokenizer,
        prompt=prior_score_prompt,
        candidate_strings=SCORE_STRINGS,
        batch_size=args.candidate_batch_size,
    )

    print("Computed prior score preferences.", flush=True)

    prompt_version = f"{args.task}_feedback_first_calibrated_decimal_v3"

    processed = 0
    skipped = 0

    for row_index, row in enumerate(rows, start=1):
        eval_id = str(row.get("eval_id", "")).strip()
        if not eval_id:
            raise ValueError(f"Missing eval_id in row {row_index}")

        if eval_id in done:
            skipped += 1
            continue

        base_prompt = build_base_prompt(args.task, row, prior=False)
        feedback_prompt = base_prompt + "\n" + FEEDBACK_PROMPT

        judge_feedback = generate_feedback(
            model=model,
            tokenizer=tokenizer,
            prompt=feedback_prompt,
            max_new_tokens=args.max_feedback_tokens,
            temperature=args.feedback_temperature,
            top_p=args.feedback_top_p,
        )

        score_prompt = base_prompt + "\n" + SCORE_PROMPT_TEMPLATE.format(judge_feedback=judge_feedback)

        row_logprobs = candidate_avg_logprobs_batched(
            model=model,
            tokenizer=tokenizer,
            prompt=score_prompt,
            candidate_strings=SCORE_STRINGS,
            batch_size=args.candidate_batch_size,
        )

        score_fields = calibrated_distribution(row_logprobs, prior_logprobs)

        out_row = dict(row)
        out_row.update({
            "judge_run_id": args.judge_run_id,
            "judge_seed": args.seed,
            "judge_model": args.model,
            "judge_prompt_version": prompt_version,
            "judge_feedback": judge_feedback,
        })
        out_row.update(score_fields)

        append_jsonl(output_path, out_row)
        done.add(eval_id)
        processed += 1

        if processed % 5 == 0:
            print(
                f"processed_new={processed} skipped={skipped} row_index={row_index}/{len(rows)} "
                f"last_expected={out_row['score_expected']:.3f} last_hard={out_row['score_hard']:.1f}",
                flush=True,
            )

    print("DONE", flush=True)
    print(f"processed_new={processed}", flush=True)
    print(f"skipped={skipped}", flush=True)
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
