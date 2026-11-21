#!/usr/bin/env python3
"""Generate the Clef-Flash conformance goldens on a CPU box.

The 9B text backbone is instantiated on the meta device and its weights are
assigned from memory-mapped safetensors shards, so only the page cache and the
activations occupy RAM. The untied lm_head is never materialized: the joint head
only reads the option-token rows, which are gathered from its safetensors slice.

The release is served in bf16 (the model card's default), so the Rust fp32 trunk
is compared against it with a 5e-3 probability tolerance.

    uv run --python 3.13 --no-project --with "torch==2.15.*" \
        --with "transformers==5.23.*" --with "safetensors" \
        python conformance/scripts/clef_reference.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5TextModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "conformance" / "cases" / "clef.json"
FIXTURE = ROOT / "conformance" / "fixtures" / "clef" / "clef-flash.jsonl"
MAX_LENGTH = 16_384
MODEL = "clef-flash"

def model_dir() -> Path:
    return Path(os.environ.get("OLAYA_HOME", Path.home() / ".olaya")) / "models" / MODEL


# The author's reference module ships with the release (joint_schema_model.py).
sys.path.insert(0, str(Path(os.environ.get("CLEF_CODE", model_dir()))))
from joint_schema_model import JointSchemaHead, collate_records, encode_record, render  # noqa: E402


class LmRows:
    """`output_embedding_weight[token_ids]` without loading the whole lm_head."""

    def __init__(self, handle, name):
        self.slice = handle.get_slice(name)

    def __getitem__(self, ids):
        values = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return torch.stack([self.slice[int(index)] for index in values])


def load_text(directory: Path):
    config = Qwen3_5Config.from_pretrained(str(directory)).text_config
    with torch.device("meta"):
        text = Qwen3_5TextModel(config)
    handles = []
    for shard in sorted(directory.glob("model-*.safetensors")):
        handle = safe_open(str(shard), framework="pt", device="cpu")
        handles.append(handle)
        state = {
            key.removeprefix("model.language_model."): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith("model.language_model.")
        }
        text.load_state_dict(state, strict=False, assign=True)
        del state
    # Non-persistent rotary buffers stay on the meta device after the load.
    text.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
    lm_handle = next(handle for handle in handles if "lm_head.weight" in handle.keys())
    return text.eval(), handles, LmRows(lm_handle, "lm_head.weight")


def load_cases():
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    for case in cases:
        repeat = case.pop("state_repeat", None)
        if repeat is not None:
            case["state"] = repeat["text"] * repeat["times"]
    return cases


def verdict(tokenizer, case):
    questions = case["questions"]
    if case["state"] is None or isinstance(case["state"], (bool, int, float)):
        return "invalid_request"
    if any(question.get("type") == "box" for question in questions.values()):
        return "invalid_request"
    try:
        encode_record(tokenizer, case, max_length=MAX_LENGTH)
    except ValueError:
        return "input_too_long"
    empty = dict(case, state="")
    fixed = len(encode_record(tokenizer, empty, max_length=10**9).input_ids)
    state_tokens = tokenizer(render(case["state"]), add_special_tokens=False).input_ids
    if len(state_tokens) > MAX_LENGTH - fixed:
        return "state_truncated"
    return None


def answers(spec, probabilities):
    kind = spec["type"]
    if kind == "noul":
        return {"type": "noul", "noul": probabilities["true"]}
    if kind == "choice":
        labels = list(spec["criteria"])
        best = max(labels, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": best,
            "probabilities": {label: probabilities[label] for label in labels},
        }
    levels = spec["criteria"]
    return {
        "type": "score",
        "score": sum(index * probabilities[str(index)] for index in range(len(levels))),
        "legend": {str(index): level for index, level in enumerate(levels)},
        "probabilities": {str(index): probabilities[str(index)] for index in range(len(levels))},
    }


def main():
    directory = model_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    cases = load_cases()
    verdicts = {case["id"]: verdict(tokenizer, case) for case in cases}
    text, handles, output_weight = load_text(directory)
    head_config = json.loads((directory / "joint_head_config.json").read_text())
    head = JointSchemaHead(**head_config)
    head.load_state_dict(load_file(str(directory / "joint_head.safetensors")), strict=True)
    head = head.to(dtype=torch.bfloat16).eval()

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w", encoding="utf-8") as output:
        for case in cases:
            error = verdicts[case["id"]]
            answers_by_id = None
            if error is None:
                encoded = encode_record(tokenizer, case, max_length=MAX_LENGTH)
                batch = collate_records([encoded], 0, torch.device("cpu"))
                with torch.inference_mode():
                    hidden = text(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                        return_dict=True,
                    ).last_hidden_state
                    logits = head(
                        hidden,
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["records"],
                        output_weight,
                    )[0]
                answers_by_id = {}
                for question, question_logits in zip(encoded.questions, logits):
                    probabilities = dict(
                        zip(
                            question.option_ids,
                            question_logits.float().softmax(-1).tolist(),
                        )
                    )
                    answers_by_id[question.question_id] = answers(
                        case["questions"][question.question_id], probabilities
                    )
            record = {
                "id": case["id"],
                "model": MODEL,
                "state": case["state"],
                "questions": case["questions"],
                "error": {"code": error} if error else None,
                "answers": answers_by_id,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{case['id']:<28} {error or 'ok'}", flush=True)


if __name__ == "__main__":
    main()
