# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone GPQA-Diamond accuracy harness for the OSCAR KV-cache backend.

Compares Qwen3-32B accuracy under BF16 vs OSCAR INT2 KV cache. Not a pytest;
run directly:

    python tests/quantization/oscar_gpqa_eval.py --kv bf16   --n 32
    python tests/quantization/oscar_gpqa_eval.py --kv oscar  --n 32 --rot-dir <dir>
"""

import argparse
import os
import random
import re

LETTERS = ["A", "B", "C", "D"]


def build_prompts(n, seed=0):
    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    rng = random.Random(seed)
    items = []
    for ex in ds:
        correct = ex["Correct Answer"].strip()
        opts = [
            correct,
            ex["Incorrect Answer 1"].strip(),
            ex["Incorrect Answer 2"].strip(),
            ex["Incorrect Answer 3"].strip(),
        ]
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold = LETTERS[order.index(0)]
        body = "\n".join(f"{LETTERS[i]}) {shuffled[i]}" for i in range(4))
        q = (
            f"Answer the following multiple choice question. The last line of "
            f"your response must be exactly 'Answer: X' where X is A, B, C or D.\n\n"
            f"{ex['Question'].strip()}\n\n{body}"
        )
        items.append((q, gold))
    rng.shuffle(items)
    return items[:n]


def extract_letter(text):
    m = re.findall(r"[Aa]nswer:\s*\(?([ABCD])\)?", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{\s*([ABCD])\s*\}", text)
    if m:
        return m[-1].upper()
    # last standalone letter fallback
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", choices=["bf16", "oscar"], required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--rot-dir", default="")
    args = ap.parse_args()

    if args.kv == "oscar":
        assert args.rot_dir, "--rot-dir required for oscar"
        os.environ["VLLM_OSCAR_K_ROTATION_PATH"] = os.path.join(
            args.rot_dir, "k_rotation_qqt_r_h_pbr.pt"
        )
        os.environ["VLLM_OSCAR_V_ROTATION_PATH"] = os.path.join(
            args.rot_dir, "v_rotation_sst_r_h_pbr.pt"
        )
        os.environ["VLLM_OSCAR_K_CLIP_RATIO"] = "0.96"
        os.environ["VLLM_OSCAR_V_CLIP_RATIO"] = "0.92"
        kv_cache_dtype = "oscar_int2"
    else:
        kv_cache_dtype = "auto"

    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    items = build_prompts(args.n)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for q, _ in items
    ]

    llm = LLM(
        model=args.model,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=True,
        gpu_memory_utilization=0.92,
        max_model_len=args.max_tokens + 1024,
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    outs = llm.generate(prompts, sp)

    correct = 0
    for (_, gold), o in zip(items, outs):
        pred = extract_letter(o.outputs[0].text)
        correct += int(pred == gold)
    acc = 100.0 * correct / len(items)
    print(f"OSCAR_EVAL kv={args.kv} n={len(items)} correct={correct} acc={acc:.2f}%")


if __name__ == "__main__":
    main()
