# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build Laya prompts and decode `token_classify` outputs into typed answers.

Ported from `laya.common.build_sequence` and `laya.agent.Agent` so answers
match the reference implementation (https://github.com/NandhaKishorM/laya).
"""

import json
import math
from typing import Any

DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


def serialize_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def check_question(qid: str, qdef: Any) -> None:
    if not isinstance(qdef, dict):
        raise ValueError(f"question {qid!r}: definition must be a dict")
    t = qdef.get("type")
    if t not in ("choice", "score", "noul"):
        raise ValueError(f"question {qid!r}: unknown type {t!r}")
    if "instructions" not in qdef:
        raise ValueError(f"question {qid!r}: no 'instructions'")
    crit = qdef.get("criteria")
    if t == "choice" and (not isinstance(crit, (dict, list)) or not crit):
        raise ValueError(f"question {qid!r}: choice needs a non-empty 'criteria'")
    if t == "score" and (not isinstance(crit, list) or not crit):
        raise ValueError(f"question {qid!r}: score needs a non-empty 'criteria' list")
    if t == "noul" and crit is not None and not isinstance(crit, dict):
        raise ValueError(f"question {qid!r}: noul 'criteria' must be a dict")
    if "labels" in qdef and t != "noul":
        raise ValueError(f"question {qid!r}: 'labels' is only supported for noul")


def normalize_question(qdef: dict) -> dict:
    t, crit = qdef["type"], qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    elif t == "noul" and isinstance(crit, dict):
        crit = {str(k).lower(): v for k, v in crit.items()}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    return {"t": t, "ins": ins, "crit": crit, "labels": qdef.get("labels")}


def render_options(q: dict) -> list[str]:
    t, crit = q["t"], q["crit"]
    if t == "choice":
        return [
            k if v is None or v == "" else f"{k}: {render_criterion(v)}"
            for k, v in crit.items()
        ]
    if t == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
    crit = crit or {}
    labels = q["labels"] or DEFAULT_NOUL_LABELS
    defaults = {
        "false": "no, the statement does not hold",
        "true": "yes, the statement holds",
    }
    return [
        f"{labels[key].strip()}: "
        + (
            render_criterion(crit[key])
            if crit.get(key) not in (None, "")
            else defaults[key]
        )
        for key in ("false", "true")
    ]


def build_prompt(
    tokenizer,
    state: str | dict | list,
    q: dict,
    max_len: int,
    head_max_len: int,
) -> list[int]:
    """`[CLS] <type> question: ins [SEP] [MASK] opt0 ... [SEP] state [SEP]`"""
    mask_tok = tokenizer.mask_token

    def encode(text: str) -> list[int]:
        return tokenizer(text.replace(mask_tok, " "), add_special_tokens=False)[
            "input_ids"
        ]

    head_ids = encode(f"{q['t']} question: {q['ins']}")
    opt_ids = [
        [tokenizer.mask_token_id] + encode(" " + opt)[:48] for opt in render_options(q)
    ]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // len(opt_ids))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    ids = (
        [tokenizer.cls_token_id]
        + head_ids[: max(8, opt_budget)]
        + [tokenizer.sep_token_id]
    )
    for o in opt_ids:
        ids.extend(o)
    ids.append(tokenizer.sep_token_id)

    room = max(0, max_len - len(ids) - 1)
    state_ids = encode(serialize_state(state))
    # Conversations are newest-last, so keep their tail.
    if isinstance(state, list):
        state_ids = state_ids[max(0, len(state_ids) - room) :]
    else:
        state_ids = state_ids[:room]
    ids = (ids + state_ids + [tokenizer.sep_token_id])[:max_len]
    if ids.count(tokenizer.mask_token_id) != len(opt_ids):
        raise ValueError(f"question options exceed max_len={max_len}")
    return ids


def confidence(p: list[float]) -> float:
    k = len(p)
    if k < 2:
        return 1.0
    ent = -sum(x * math.log(min(max(x, 1e-12), 1.0)) for x in p)
    return min(max(1.0 - ent / math.log(k), 0.0), 1.0)


def decode_answer(q: dict, rows: list[list[float]]) -> dict:
    """`rows` is one question's `token_classify` output: `[p_option, *p_act]`."""
    p = [row[0] for row in rows]
    action = {"act_probability": round(rows[0][1], 4)}
    if q["t"] == "choice":
        keys = list(q["crit"])
        return {
            "type": "choice",
            "choice": keys[max(range(len(p)), key=p.__getitem__)],
            "probabilities": {k: round(v, 4) for k, v in zip(keys, p)},
            "confidence": round(confidence(p), 4),
            "action": action,
        }
    if q["t"] == "score":
        return {
            "type": "score",
            "score": round(sum(i * v for i, v in enumerate(p)), 4),
            "legend": {str(i): c for i, c in enumerate(q["crit"])},
            "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)},
            "confidence": round(confidence(p), 4),
            "action": action,
        }
    return {
        "type": "noul",
        "noul": round(p[1], 4),
        "confidence": round(max(p[1], 1.0 - p[1]), 4),
        "action": action,
    }
