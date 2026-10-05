# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GSM8K data, prompt format and scoring shared by run_live.py (offline LLM) and
mono_client.py (server).

5-shot = the first 5 train exemplars, "Question: q\\nAnswer: a\\n\\n" x5 + "Question:
q\\nAnswer:", greedy, max 256 tokens, stop STOP; strict = "#### N", flexible = last
number; stderr = sqrt(p(1-p)/(n-1)). Pure Python (no vLLM / torch import)."""

import json
import math
import os

import regex as re

DATA = os.environ.get("MONO_GSM8K_DATA", os.path.expanduser("~/.cache/vllm_mono/gsm8k"))
STOP = ["Question:", "</s>", "<|im_end|>"]
NSHOT = 5
MAX_TOKENS = 256


def gsm8k_data(data_dir: str = DATA) -> dict:
    """{"train": [...], "test": [...]} from cached jsonl; downloads openai/gsm8k once if
    missing."""
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        fn = os.path.join(data_dir, f"gsm8k_{split}.jsonl")
        if not os.path.exists(fn):
            from datasets import load_dataset

            ds = load_dataset("openai/gsm8k", "main", split=split)
            with open(fn, "w") as f:
                for r in ds:
                    f.write(
                        json.dumps(dict(question=r["question"], answer=r["answer"]))
                        + "\n"
                    )
        out[split] = [json.loads(line) for line in open(fn)]
    return out


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    return s


def strict(text):
    m = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return norm_num(m.group(1)) if m else None


def flexible(text):
    ms = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not ms:
        return None
    g = ms[-1]
    return norm_num(g[0] or g[1])


def same(p, g):
    if p is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def stderr(k, n):
    p = k / n
    return math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else 0.0


def gsm_prefix(d, nshot=NSHOT):
    return "".join(
        f"Question: {s['question']}\nAnswer: {s['answer']}\n\n"
        for s in d["train"][:nshot]
    )


def prompt(prefix, q):
    return prefix + f"Question: {q['question']}\nAnswer:"


def gold(q):
    return norm_num(q["answer"].split("####")[-1])
