#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Needle-in-a-Haystack benchmark for TurboQuant KV cache quantization.

Usage:
    python tests/quantization/bench_turboquant_needle.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --contexts 4096 8192 \
        --depths 0.25 0.5 0.75
"""

import argparse
import gc

import torch

NEEDLE = "The secret project code name is AURORA-7749."
RETRIEVAL_Q = "What is the secret project code name mentioned in the text?"
ANSWER_KEY = "AURORA-7749"

FILLER_PARAGRAPH = (
    "The quarterly budget review meeting covered several operational topics "
    "including vendor contract renewals, facility maintenance schedules, "
    "employee onboarding procedures, and updates to the internal compliance "
    "documentation. The committee discussed standardizing the procurement "
    "workflow across regional offices to improve efficiency and reduce "
    "administrative overhead. No major policy changes were proposed. "
)


def build_haystack(tokenizer, target_tokens: int, needle_depth: float) -> str:
    """Build a haystack document with a needle inserted at the given depth."""
    # Build filler text
    filler_tokens = tokenizer.encode(FILLER_PARAGRAPH, add_special_tokens=False)
    tokens_per_para = len(filler_tokens)

    # Reserve space for needle + question
    needle_tokens = len(tokenizer.encode(NEEDLE, add_special_tokens=False))
    question_tokens = len(tokenizer.encode(RETRIEVAL_Q, add_special_tokens=False))
    available = target_tokens - needle_tokens - question_tokens - 50  # margin

    n_paras = max(1, available // tokens_per_para)
    haystack_parts = [FILLER_PARAGRAPH] * n_paras

    # Insert needle at specified depth
    insert_pos = max(0, int(len(haystack_parts) * needle_depth))
    haystack_parts.insert(insert_pos, NEEDLE)

    document = "\n".join(haystack_parts)
    prompt = (
        f"Read the following document carefully:\n\n{document}\n\n"
        f"Question: {RETRIEVAL_Q}\n"
        f"Answer:"
    )
    return prompt


def run_one_test(
    model_name: str,
    kv_cache_dtype: str,
    target_tokens: int,
    needle_depth: float,
    gpu_memory_utilization: float = 0.5,
) -> tuple[bool, str]:
    """Run a single needle test. Returns (found, answer_text)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=target_tokens + 512,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    prompt = build_haystack(tokenizer, target_tokens, needle_depth)

    actual_tokens = len(tokenizer.encode(prompt))
    print(f"    Prompt tokens: {actual_tokens}")

    sampling = SamplingParams(max_tokens=50, temperature=0)
    outputs = llm.generate([prompt], sampling)
    answer = outputs[0].outputs[0].text.strip()

    found = ANSWER_KEY in answer
    # Cleanup
    del llm
    gc.collect()
    torch.accelerator.empty_cache()
    return found, answer


def main():
    parser = argparse.ArgumentParser(
        description="Needle-in-a-Haystack benchmark for TurboQuant"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    configs = [
        ("baseline", "auto"),
        ("turboquant", "turboquant"),
    ]

    print(f"Model: {args.model}")
    print(f"Contexts: {args.contexts}")
    print(f"Depths: {args.depths}")
    print()

    results = {}
    for config_name, kv_dtype in configs:
        print(f"=== Config: {config_name} (kv_cache_dtype={kv_dtype}) ===")
        for ctx in args.contexts:
            for depth in args.depths:
                key = (config_name, ctx, depth)
                print(f"  Context={ctx}, Depth={depth:.0%}...")
                try:
                    found, answer = run_one_test(
                        args.model, kv_dtype, ctx, depth, args.gpu_memory_utilization
                    )
                    results[key] = found
                    status = "PASS" if found else "FAIL"
                    print(f"    {status}: {answer[:80]}")
                except Exception as e:
                    results[key] = False
                    print(f"    ERROR: {e}")
        print()

    # Summary table
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    header = f"{'Config':<15} {'Context':<10}"
    for d in args.depths:
        header += f" {'D=' + f'{d:.0%}':<10}"
    print(header)
    print("-" * 70)

    for config_name, _ in configs:
        for ctx in args.contexts:
            row = f"{config_name:<15} {ctx:<10}"
            for depth in args.depths:
                key = (config_name, ctx, depth)
                status = "PASS" if results.get(key, False) else "FAIL"
                row += f" {status:<10}"
            print(row)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{total} passed")


if __name__ == "__main__":
    main()
