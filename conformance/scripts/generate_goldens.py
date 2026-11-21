#!/usr/bin/env python3
"""Generate conformance goldens from the upstream references.

The fixtures live in `conformance/fixtures/<family>/<model>.jsonl`; each line is
one request from `conformance/cases/<family>.json` with the upstream
probabilities (full precision) or the upstream 422 verdict.

    uv run --python 3.13 --with "torch==2.14.*" --with "transformers==5.17.*" \
        --with "decima-ai==2.3.0" --with "safetensors" \
        python conformance/scripts/generate_goldens.py decima decima-small
    uv run --python 3.13 --with "torch==2.14.*" --with "transformers==5.17.*" \
        python conformance/scripts/generate_goldens.py nli nli-modernbert nli-deberta-v3
    uv run --python 3.13 --with "torch==2.15.*" --with "transformers==5.23.*" \
        --with "safetensors" \
        python conformance/scripts/generate_goldens.py kev kev-0.8b

Decima answers come from `decima.model.Decima` (decima 2.3.0, PyTorch fp32);
NLI answers from `transformers.AutoModelForSequenceClassification` (fp32) on the
same rows the Rust family builds. Kev answers merge the PEFT LoRA adapter into
`Qwen3.5-0.8B-Base` and apply `head.pt`, with the author's render/encode layout
(`kev.api.render`, `kev.model.build_row`). All three add the family rules that
the Rust code implements: decima's 2-10 ordinal limit, the state-row truncation
check, and the classification heads' class columns. Kev reads the base model
from `~/.olaya/models/qwen35-0.8b-base` unless `KEV_BASE_DIR` says otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "conformance" / "fixtures"
CASES = ROOT / "conformance" / "cases"
MAX_QUESTIONS = 256
MAX_LEVELS = 10


def models_dir() -> Path:
    return Path(os.environ.get("OLAYA_HOME", Path.home() / ".olaya")) / "models"


def manifest(family: str, model: str) -> dict:
    path = ROOT / "registry" / "v1" / "library" / family / "manifests" / f"{model}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(family: str) -> list[dict]:
    path = CASES / f"{family}.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for case in cases:
        repeat = case.pop("state_repeat", None)
        if repeat is not None:
            case["state"] = repeat["text"] * repeat["times"]
        out.append(case)
    return out


def softmax(values):
    values = [float(v) for v in values]
    top = max(values)
    exp = [math.exp(v - top) for v in values]
    total = sum(exp)
    return [v / total for v in exp]


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def state_string(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def render(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def fill(template: str, fields: dict) -> str:
    out, rest = [], template
    while True:
        open_at = rest.find("{")
        if open_at < 0:
            out.append(rest)
            return "".join(out)
        out.append(rest[:open_at])
        after = rest[open_at + 1 :]
        end = 0
        while end < len(after) and (after[end].isalnum() or after[end] == "_"):
            end += 1
        if end > 0 and after[end : end + 1] == "}":
            out.append(fields[after[:end]])
            rest = after[end + 1 :]
        else:
            out.append("{")
            rest = after


# --------------------------------------------------------------------- decima


def decima_question(qid: str, spec: dict):
    from decima.systemone import to_question

    t, q, keys = to_question(qid, spec)
    if t == "choice" and q.kind == "score" and len(keys) > MAX_LEVELS:
        raise ValueError(f"questions.{qid}: an ordinal choice takes 2-{MAX_LEVELS} labels")
    return t, q, keys


def decima_verdict(model, state, questions):
    from decima.systemone import SystemOneError

    if isinstance(state, (bool, int, float)) or state is None:
        return "invalid_request"
    if not isinstance(questions, dict) or not (1 <= len(questions) <= MAX_QUESTIONS):
        return "invalid_request"
    s = state_string(state)
    for qid, spec in questions.items():
        try:
            _, q, _ = decima_question(qid, spec)
        except (SystemOneError, ValueError, KeyError, TypeError):
            return "invalid_request"
        rendered = model.cfg.state_of(s, q.text, q.lang)
        if len(model.tok(rendered)["input_ids"]) > model.cfg.max_state_tokens:
            return "state_truncated"
    return None


def decima_answers(model, state, questions):
    s = state_string(state)
    out = {}
    for qid, spec in questions.items():
        _, q, keys = decima_question(qid, spec)
        probs = [float(p) for p in model.decide(s, q).probs]
        if spec["type"] == "noul":
            out[qid] = {"type": "noul", "noul": probs[0]}
        elif spec["type"] == "choice":
            out[qid] = {
                "type": "choice",
                "choice": keys[argmax(probs)],
                "probabilities": dict(zip(keys, probs)),
            }
        else:
            levels = spec["criteria"]
            out[qid] = {
                "type": "score",
                "score": sum(i * p for i, p in enumerate(probs)),
                "legend": {str(i): level for i, level in enumerate(levels)},
                "probabilities": {str(i): p for i, p in enumerate(probs)},
            }
    return out


def generate_decima(model: str, cases: list[dict]) -> list[dict]:
    from decima.model import Decima

    path = models_dir() / model / manifest("decima", model).get("subfolder", "pytorch")
    d = Decima(str(path), "cpu")
    records = []
    for case in cases:
        error = decima_verdict(d, case["state"], case["questions"])
        record = {
            "id": case["id"],
            "model": model,
            "state": case["state"],
            "questions": case["questions"],
            "error": {"code": error} if error else None,
            "answers": None,
        }
        if error is None:
            record["answers"] = decima_answers(d, case["state"], case["questions"])
        records.append(record)
        print(f"{model:<20} {case['id']:<40} {error or 'ok'}", flush=True)
    return records


# ------------------------------------------------------------------------ kev

KEV_SPECIAL_RE = None


def kev_render(value, indent=0):
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(f"{pad}- {kev_render(x, indent + 1).lstrip()}" for x in value)
    return "\n".join(
        f"{pad}{k}:\n{kev_render(x, indent + 1)}"
        if isinstance(x, (dict, list))
        else f"{pad}{k}: {kev_render(x)}"
        for k, x in value.items()
    )


def kev_option_text(name, description):
    return name if description is None or description == "" else f"{name}: {kev_render(description)}"


def kev_user_tokens(tokenizer, text):
    import re

    global KEV_SPECIAL_RE
    if KEV_SPECIAL_RE is None:
        KEV_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
    return tokenizer.encode(KEV_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).ids


def kev_row(tokenizer, special, state, instructions, options):
    ids = [special["state"]] + kev_user_tokens(tokenizer, kev_render(state))
    ids += [special["question"]] + kev_user_tokens(tokenizer, kev_render(instructions))
    positions = []
    for option in options:
        ids.append(special["option_open"])
        ids += kev_user_tokens(tokenizer, option)
        ids.append(special["option_close"])
        positions.append(len(ids) - 1)
    ids.append(special["decide"])
    return ids, positions, len(ids) - 1


def kev_options(spec):
    kind = spec["type"]
    if kind == "noul":
        criteria = spec.get("criteria") or {}
        return [kev_option_text("no", criteria.get("false")), kev_option_text("yes", criteria.get("true"))]
    if kind == "choice":
        return [kev_option_text(k, v) for k, v in spec["criteria"].items()]
    return [kev_render(level) for level in spec["criteria"]]


def generate_kev(model: str, cases: list[dict]) -> list[dict]:
    import tokenizers
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration

    directory = models_dir() / model
    decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
    special = decision["special_tokens"]
    max_state = decision["max_state_tokens"]
    max_row = decision["max_row_tokens"]
    base_directory = Path(
        os.environ.get("KEV_BASE_DIR", models_dir() / "qwen35-0.8b-base")
    )
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(base_directory), dtype=torch.float32, attn_implementation="eager"
    ).eval()
    text = lm.model.language_model

    adapter = load_file(str(directory / "adapter_model.safetensors"))
    scale = float(
        (decision.get("upstream", {}).get("lora") or {}).get("scaling", 2.0)
    )
    lora_a = {}
    for name, tensor in adapter.items():
        name = name.removeprefix("base_model.model.")
        if ".lora_A.weight" in name:
            lora_a[name.split(".lora_A.weight")[0]] = tensor.float()
    with torch.no_grad():
        for name, tensor in adapter.items():
            name = name.removeprefix("base_model.model.")
            if ".lora_B.weight" not in name:
                continue
            module = name.split(".lora_B.weight")[0]
            parameter = text
            for part in f"{module}.weight".split("."):
                parameter = getattr(parameter, part)
            parameter.add_(tensor.float() @ lora_a[module] * scale)

    head = torch.load(str(directory / "head.pt"), map_location="cpu", weights_only=True)
    qw, qb = head["head"]["q.weight"], head["head"]["q.bias"]
    kw, kb = head["head"]["k.weight"], head["head"]["k.bias"]
    temperature = float(head["temperature"])

    records = []
    for case in cases:
        state = case["state"]
        questions = case["questions"]
        error = None
        if state is None or isinstance(state, (bool, int, float)):
            error = "invalid_request"
        elif any(spec.get("type") == "box" for spec in questions.values()):
            error = "invalid_request"
        else:
            state_ids = kev_user_tokens(tokenizer, kev_render(state))
            if len(state_ids) + 1 > max_state:
                error = "state_truncated"
        if error is None:
            parsed = {}
            for qid, spec in questions.items():
                ids, positions, decide = kev_row(
                    tokenizer, special, state, spec.get("instructions"), kev_options(spec)
                )
                if len(ids) > max_row:
                    error = "input_too_long"
                    break
                parsed[qid] = (ids, positions, decide)
        record = {
            "id": case["id"],
            "model": model,
            "state": state,
            "questions": questions,
            "error": {"code": error} if error else None,
            "answers": None,
        }
        if error is None:
            answers = {}
            for qid, spec in questions.items():
                ids, positions, decide = parsed[qid]
                with torch.no_grad():
                    hidden = text(input_ids=torch.tensor([ids])).last_hidden_state[0]
                    decide_h = hidden[decide]
                    option_h = hidden[torch.tensor(positions)]
                    logits = ((option_h @ kw.T + kb) * (decide_h @ qw.T + qb)).sum(-1) / 16.0
                    probs = torch.softmax(logits / temperature, -1).tolist()
                if spec["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": probs[1]}
                elif spec["type"] == "choice":
                    labels = list(spec["criteria"])
                    answers[qid] = {
                        "type": "choice",
                        "choice": labels[argmax(probs)],
                        "probabilities": dict(zip(labels, probs)),
                    }
                else:
                    levels = spec["criteria"]
                    answers[qid] = {
                        "type": "score",
                        "score": sum(i * p for i, p in enumerate(probs)),
                        "legend": {str(i): kev_render(level) for i, level in enumerate(levels)},
                        "probabilities": {str(i): p for i, p in enumerate(probs)},
                    }
            record["answers"] = answers
        records.append(record)
        print(f"{model:<20} {case['id']:<40} {error or 'ok'}", flush=True)
    return records


# ------------------------------------------------------------------------ nli

NLI_TEMPLATES = {
    "modernbert": {
        "choice_with_description": "{description}",
        "choice_without_description": "This text is about {label}.",
        "score": "{description}",
        "noul_pair": "{description}",
        "noul_single": "{instructions}",
    },
    "deberta-v3": {
        "choice_with_description": 'The answer to "{instructions}" is: {option}',
        "choice_without_description": 'The answer to "{instructions}" is: {option}',
        "score": 'The answer to "{instructions}" is: {description}',
        "noul_pair": "{description}",
        "noul_single": "{instructions}",
    },
}


def nli_specials(config: dict, tokenizer):
    def first(*names):
        for name in names:
            token = tokenizer.token_to_id(name)
            if token is not None:
                return token
        return None

    cls = config.get("cls_token_id", None)
    if cls is None:
        cls = first("[CLS]", "<bos>", "<s>")
    sep = config.get("sep_token_id", None)
    if sep is None:
        sep = first("[SEP]", "<eos>", "</s>")
    pad = config.get("pad_token_id", None)
    if pad is None:
        pad = first("[PAD]")
    if cls is None or sep is None:
        raise SystemExit("tokenizer has no CLS/SEP token")
    return int(cls), int(sep), int(pad or 0)


def nli_hypotheses(qid: str, spec: dict, templates: dict):
    """(hypotheses, single) for one question, mirroring olaya-candle's nli.rs."""
    ins = spec.get("instructions")
    instructions = qid if ins is None else render(ins)
    kind = spec["type"]
    crit = spec.get("criteria")
    if kind == "choice":
        out = []
        for label, value in crit.items():
            description = "" if value is None else render(value)
            option = label if description == "" else f"{label}: {description}"
            template = (
                templates["choice_without_description"]
                if description == ""
                else templates["choice_with_description"]
            )
            out.append(
                fill(
                    template,
                    {
                        "instructions": instructions,
                        "label": label,
                        "description": description,
                        "option": option,
                    },
                )
            )
        return out, False
    if kind == "score":
        out = []
        for index, level in enumerate(crit):
            out.append(
                fill(
                    templates["score"],
                    {
                        "instructions": instructions,
                        "level": str(index),
                        "description": render(level),
                    },
                )
            )
        return out, False
    criteria = crit if isinstance(crit, dict) else {}
    by_key = {str(k).lower(): v for k, v in criteria.items()}
    is_false = by_key.get("false")
    is_true = by_key.get("true")

    def described(value):
        if value is None or value == "":
            return None
        return render(value)

    false_text, true_text = described(is_false), described(is_true)
    if false_text is not None and true_text is not None:
        return (
            [
                fill(templates["noul_pair"], {"instructions": instructions, "description": false_text}),
                fill(templates["noul_pair"], {"instructions": instructions, "description": true_text}),
            ],
            False,
        )
    return ([fill(templates["noul_single"], {"instructions": instructions})], True)


def generate_nli(model: str, cases: list[dict]) -> list[dict]:
    import tokenizers
    import torch
    from transformers import AutoModelForSequenceClassification

    entry = manifest("nli", model)
    arch = entry["arch"]
    templates = NLI_TEMPLATES["deberta-v3" if arch.startswith("deberta") else "modernbert"]
    directory = models_dir() / model
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    cls, sep, pad = nli_specials(config, tokenizer)
    max_len = 512
    model_obj = (
        AutoModelForSequenceClassification.from_pretrained(str(directory), dtype=torch.float32)
        .eval()
    )
    entailment, not_entailment = 0, 1

    records = []
    for case in cases:
        if isinstance(case["state"], (bool, int, float)) or case["state"] is None:
            records.append({**case, "model": model, "error": {"code": "invalid_request"}, "answers": None})
            continue
        premise = tokenizer.encode(state_string(case["state"]), add_special_tokens=False).ids
        questions = case["questions"]
        parsed, rows, error = [], [], None
        for qid, spec in questions.items():
            hypotheses, single = nli_hypotheses(qid, spec, templates)
            question_rows = []
            for hypothesis in hypotheses:
                ids = tokenizer.encode(hypothesis, add_special_tokens=False).ids
                room = max_len - len(ids) - 3
                if room <= 0:
                    error = "invalid_request"
                    break
                if len(premise) > room:
                    error = "state_truncated"
                    break
                question_rows.append(
                    [cls] + premise[:room] + [sep] + ids + [sep]
                )
            if error:
                break
            parsed.append((qid, spec, single, len(hypotheses)))
            rows.extend(question_rows)
        if error:
            records.append(
                {"id": case["id"], "model": model, "state": case["state"], "questions": questions,
                 "error": {"code": error}, "answers": None}
            )
            print(f"{model:<20} {case['id']:<40} {error}", flush=True)
            continue

        logits = []
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                batch = rows[start : start + 16]
                width = max(len(row) for row in batch)
                ids = torch.full((len(batch), width), pad, dtype=torch.long)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for index, row in enumerate(batch):
                    ids[index, : len(row)] = torch.tensor(row)
                    mask[index, : len(row)] = 1
                logits.extend(
                    model_obj(input_ids=ids, attention_mask=mask).logits.double().tolist()
                )

        answers, cursor = {}, 0
        for qid, spec, single, count in parsed:
            block = logits[cursor : cursor + count]
            cursor += count
            if single:
                option_logits = [block[0][not_entailment], block[0][entailment]]
            else:
                option_logits = [row[entailment] for row in block]
            probs = softmax(option_logits)
            if spec["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": probs[1]}
            elif spec["type"] == "choice":
                labels = list(spec["criteria"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": labels[argmax(probs)],
                    "probabilities": dict(zip(labels, probs)),
                }
            else:
                levels = spec["criteria"]
                answers[qid] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(probs)),
                    "legend": {str(i): level for i, level in enumerate(levels)},
                    "probabilities": {str(i): p for i, p in enumerate(probs)},
                }
        records.append(
            {"id": case["id"], "model": model, "state": case["state"], "questions": questions,
             "error": None, "answers": answers}
        )
        print(f"{model:<20} {case['id']:<40} ok", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["decima", "nli", "kev"])
    parser.add_argument("models", nargs="+")
    parser.add_argument("--cases", type=Path, default=None)
    args = parser.parse_args()

    cases = load_cases(args.family) if args.cases is None else json.loads(args.cases.read_text(encoding="utf-8"))

    out_dir = FIXTURES / args.family
    out_dir.mkdir(parents=True, exist_ok=True)
    generate = {
        "decima": generate_decima,
        "nli": generate_nli,
        "kev": generate_kev,
    }[args.family]
    for model in args.models:
        records = generate(model, cases)
        path = out_dir / f"{model}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} records to {path}")


if __name__ == "__main__":
    main()
