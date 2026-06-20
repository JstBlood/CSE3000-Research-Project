from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


GENERATION_PROMPT_TEMPLATE = """You will be given one program representation. The representation may be source code, assembly code, or binary/disassembly text.

Task:
Write one concise natural-language description of what the program or function does.

Strict output rules:
- Return only the final description.
- Do not include reasoning, analysis, explanations, bullet points, labels, or extra text.
- Do not output <think>, Thinking Process, or any hidden reasoning.
- One sentence only.
- Start with "The code" or "The function" when appropriate.
- Describe the main high-level behaviour, not the implementation line by line.
- Focus on actions such as file operations, registry access, process creation, service handling, networking, memory/string manipulation, configuration handling, or control-flow behaviour.
- Include important conditions when they affect the behaviour, for example: if a file exists, if a registry key is missing, if a process is found, or if an input is invalid.
- Mention important effects, such as creating files, writing registry values, downloading files, starting processes, modifying strings, allocating memory, or cleaning up resources.
- Do not over-generalize if a more specific behaviour is supported by the input.
- Do not describe irrelevant low-level details such as register names, stack offsets, temporary variables, addresses, labels, or compiler artifacts.
- Do not classify the program as malware, benign, suspicious, safe, harmful, or malicious.
- Do not mention the representation type.
- Do not invent behaviour that is not supported by the input.
- If the exact purpose is unclear, give the most specific cautious description supported by the code, using wording such as "appears to" only when necessary.

Program input:
{representation_text}
"""


REPRESENTATION_KEYS = {
    "binary": [
        "binary",
        "binary_code",
        "bin",
        "bin_code",
        "bytes",
        "hex",
        "binary_text",
    ],
    "assembly": [
        "assembly",
        "assembly_code",
        "asm",
        "asm_code",
        "disassembly",
        "disassembly_code",
    ],
    "source": [
        "source",
        "source_code",
        "src",
        "src_code",
        "code",
    ],
}

ID_KEYS = ["ID", "id", "sample_id", "sha256", "hash", "index"]
NLD_KEYS = ["NLD", "nld", "reference_nld", "reference", "description"]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def get_first(row: Dict[str, Any], keys: list[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def get_sample_id(row: Dict[str, Any], fallback_index: int) -> str:
    value = get_first(row, ID_KEYS)
    if value is None:
        return f"row_{fallback_index}"
    return str(value)


def get_representation_text(row: Dict[str, Any], rep: str) -> Optional[str]:
    value = get_first(row, REPRESENTATION_KEYS[rep])
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


def get_reference_nld(row: Dict[str, Any]) -> Optional[str]:
    value = get_first(row, NLD_KEYS)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def build_generation_prompt(representation_text: str) -> str:
    return GENERATION_PROMPT_TEMPLATE.format(representation_text=representation_text)


def remove_thinking_blocks(text: str) -> str:
    text = text.strip()

    # If a complete thinking block exists, keep only text after it.
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Remove broken leading thinking markers if present.
    text = re.sub(r"^\s*<think>\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*Thinking Process\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*Analysis\s*:\s*", "", text, flags=re.IGNORECASE).strip()

    return text.strip()


def clean_model_output(text: str) -> str:
    text = text.strip()

    # Remove common chat/special tokens if they survive decoding.
    special_patterns = [
        r"<\|im_start\|>\s*assistant",
        r"<\|im_start\|>\s*user",
        r"<\|im_end\|>",
        r"<\|endoftext\|>",
        r"<\|end\|>",
    ]
    for pat in special_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    text = remove_thinking_blocks(text)

    # Remove accidental prefixes.
    text = re.sub(r"^(description|answer|output|final answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^[-*]\s*", "", text).strip()

    # Keep only the first non-empty paragraph if the model ignores one-sentence instruction.
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if parts:
        text = parts[0]

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Strip surrounding quotes.
    text = text.strip(" \"'`")

    return text


def is_bad_description(description: str) -> bool:
    desc = str(description).strip()
    low = desc.lower()

    if not desc:
        return True
    if low.startswith("<think"):
        return True
    if low in {"thinking process:", "thinking process", "<think>", "<think> thinking process:"}:
        return True
    if len(desc.split()) < 4:
        return True

    return False


def load_done_keys(output_jsonl: str) -> set[tuple[str, str, int]]:
    done = set()
    if not os.path.exists(output_jsonl):
        return done

    for row in read_jsonl(output_jsonl):
        sample_id = str(row.get("sample_id", ""))
        rep = str(row.get("representation", ""))
        run_id = int(row.get("generation_run_id", -1))
        desc = str(row.get("generated_description", "")).strip()
        status = str(row.get("generation_status", "ok"))

        # Only skip rows that are already valid. Broken rows do not block regeneration.
        if sample_id and rep and run_id >= 0 and status == "ok" and not is_bad_description(desc):
            done.add((sample_id, rep, run_id))

    return done


def encode_prompt(tokenizer, prompt: str, device):
    messages = [{"role": "user", "content": prompt}]

    # Qwen thinking models should use the chat template with enable_thinking=False.
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask, True
    except TypeError:
        # Fallback for older tokenizer/transformers versions.
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            return input_ids, attention_mask, True
        except Exception:
            pass

    # Final fallback: plain prompt tokenization.
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask, False


def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[str, str, int, int, bool]:
    device = next(model.parameters()).device

    input_ids, attention_mask, used_chat_template = encode_prompt(tokenizer, prompt, device)
    input_tokens = int(input_ids.shape[1])

    model_max = getattr(model.config, "max_position_embeddings", None)
    if model_max is not None and input_tokens + max_new_tokens > model_max:
        raise ValueError(
            f"Input too long for model context: input_tokens={input_tokens}, "
            f"max_new_tokens={max_new_tokens}, model_max={model_max}. "
            f"No truncation was applied."
        )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1]:]

    raw_model_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
    decoded_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    description = clean_model_output(decoded_output)
    if is_bad_description(description):
        # Try cleaning the raw output too, in case the non-special decode hid useful text.
        description_from_raw = clean_model_output(raw_model_output)
        if not is_bad_description(description_from_raw):
            description = description_from_raw

    return description, raw_model_output, input_tokens, int(generated_ids.shape[0]), used_chat_template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.30)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)

    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--representations", default="binary,assembly,source")
    args = parser.parse_args()

    reps = [r.strip().lower() for r in args.representations.split(",") if r.strip()]
    for rep in reps:
        if rep not in REPRESENTATION_KEYS:
            raise ValueError(f"Unknown representation: {rep}")

    set_seed(args.seed)

    print(f"Loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    done = load_done_keys(args.output_jsonl)

    processed_samples = 0
    generated = 0
    skipped_missing = 0
    skipped_done = 0
    bad_outputs = 0
    chat_template_used_count = 0

    for row_index, row in enumerate(read_jsonl(args.input_jsonl), start=1):
        if args.limit_samples is not None and processed_samples >= args.limit_samples:
            break

        sample_id = get_sample_id(row, row_index)
        reference_nld = get_reference_nld(row)

        processed_samples += 1

        for rep in reps:
            key = (sample_id, rep, args.run_id)
            if key in done:
                skipped_done += 1
                continue

            representation_text = get_representation_text(row, rep)
            if representation_text is None:
                skipped_missing += 1
                print(f"Missing {rep} for sample {sample_id}", flush=True)
                continue

            prompt = build_generation_prompt(representation_text)

            try:
                description, raw_model_output, input_tokens, output_tokens, used_chat_template = generate_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
            except Exception as e:
                print(f"ERROR sample={sample_id} rep={rep}: {e}", flush=True)
                raise

            if used_chat_template:
                chat_template_used_count += 1

            generation_status = "ok"
            if is_bad_description(description):
                generation_status = "bad_description"
                bad_outputs += 1
                print(
                    f"BAD_OUTPUT sample={sample_id} rep={rep} raw={raw_model_output[:120]!r} cleaned={description[:120]!r}",
                    flush=True,
                )

            out_row = {
                "sample_id": sample_id,
                "input_index": row_index,
                "representation": rep,
                "generated_description": description,
                "raw_model_output": raw_model_output,
                "reference_nld": reference_nld,
                "generation_run_id": args.run_id,
                "generation_seed": args.seed,
                "generation_prompt_version": "specific_nld_style_v3_no_think_raw_saved",
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens,
                "input_char_len": len(representation_text),
                "prompt_char_len": len(prompt),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "used_chat_template": used_chat_template,
                "generation_status": generation_status,
            }

            append_jsonl(args.output_jsonl, out_row)
            generated += 1
            done.add(key)

            if generated % 25 == 0:
                print(
                    f"generated={generated} processed_samples={processed_samples} "
                    f"skipped_done={skipped_done} skipped_missing={skipped_missing} bad_outputs={bad_outputs}",
                    flush=True,
                )

    print("DONE", flush=True)
    print(f"processed_samples={processed_samples}", flush=True)
    print(f"generated={generated}", flush=True)
    print(f"skipped_done={skipped_done}", flush=True)
    print(f"skipped_missing={skipped_missing}", flush=True)
    print(f"bad_outputs={bad_outputs}", flush=True)
    print(f"chat_template_used_count={chat_template_used_count}", flush=True)


if __name__ == "__main__":
    main()
