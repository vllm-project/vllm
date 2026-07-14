#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Argmax-agreement diagnostic for kernel-selection non-determinism.

Distinguishes harmless floating-point rounding noise from genuine kernel
correctness bugs. Generate prompt logits twice in two separate processes
(kernel selection happens once per process at compile/warmup time), then
compare per position:

- Top-1 agreement rate: fraction of positions where both runs pick the
  same argmax token.
- For every flip, the top-1/top-2 logit gap in each run.

Interpretation: flips only at tiny gaps (< ~1e-4, model-declared ties)
with agreement in the high 99.9%s means the wobble is pure rounding
noise. Flips at large gaps indicate a real correctness bug in the
timing-selected kernels.

Usage:
    # Step 1: generate logits twice (separate processes, wobbling config)
    python examples/offline_inference/score_mode_argmax_diag.py generate \\
        --model /path/to/model --output-dir ./diag_run_a \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1
    python examples/offline_inference/score_mode_argmax_diag.py generate \\
        --model /path/to/model --output-dir ./diag_run_b \\
        --dataset wikitext --dataset-config wikitext-2-raw-v1

    # Step 2: compare
    python examples/offline_inference/score_mode_argmax_diag.py compare \\
        --dir-a ./diag_run_a --dir-b ./diag_run_b
"""

import argparse
import os
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

from score_mode_kld import (
    apply_deterministic_llm_kwargs,
    load_dataset_texts,
)


def generate_logits(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    texts = load_dataset_texts(args.dataset, args.dataset_config)
    concatenated_text = "\n\n".join(texts)

    max_tokens_for_eval = args.context_length + (args.num_windows - 1) * args.stride
    max_chars = max_tokens_for_eval * 5
    if len(concatenated_text) > max_chars:
        concatenated_text = concatenated_text[:max_chars]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    encoded = tokenizer(
        concatenated_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens_for_eval,
    )
    tokens = encoded["input_ids"]
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    num_tokens = len(tokens)

    llm_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enable_prefix_caching": False,
        "max_model_len": args.context_length * 2,
    }
    if args.deterministic:
        apply_deterministic_llm_kwargs(llm_kwargs)
        print("Deterministic config: all timing-based autotuners disabled")
    else:
        print("Wobbling config: combo kernels active (diagnostic target)")

    os.makedirs(args.output_dir, exist_ok=True)
    llm = LLM(model=args.model, **llm_kwargs)
    sampling_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        return_prompt_logits=True,
    )

    window_idx = 0
    window_starts = range(
        0, num_tokens - args.context_length + args.stride, args.stride
    )
    for start_idx in window_starts:
        end_idx = start_idx + args.context_length
        if end_idx > num_tokens or window_idx >= args.num_windows:
            break
        window_tokens = tokens[start_idx:end_idx]
        if len(window_tokens) < 2:
            continue
        prompt: TokensPrompt = {
            "prompt_token_ids": window_tokens,
            "target_token_ids": window_tokens[1:],
        }
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        prompt_logits = outputs[0].prompt_logits
        if prompt_logits is not None:
            save_file(
                {"logits": prompt_logits.cpu()},
                os.path.join(args.output_dir, f"logits_{window_idx}.safetensors"),
            )
            window_idx += 1

    print(f"Saved {window_idx} windows of logits to {args.output_dir}/")


def _load_window(directory: str, idx: int) -> torch.Tensor | None:
    path = os.path.join(directory, f"logits_{idx}.safetensors")
    if not os.path.exists(path):
        return None
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor("logits")


def compare_logits(args: argparse.Namespace) -> None:
    total_positions = 0
    total_flips = 0
    max_abs_diff = 0.0
    flip_gaps: list[tuple[int, int, float, float]] = []

    window_idx = 0
    while True:
        logits_a = _load_window(args.dir_a, window_idx)
        logits_b = _load_window(args.dir_b, window_idx)
        if logits_a is None or logits_b is None:
            break

        a = logits_a.float()
        b = logits_b.float()
        vs = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :vs], b[..., :vs]

        max_abs_diff = max(max_abs_diff, (a - b).abs().max().item())

        top2_a = a.topk(2, dim=-1)
        top2_b = b.topk(2, dim=-1)
        argmax_a = top2_a.indices[:, 0]
        argmax_b = top2_b.indices[:, 0]
        gaps_a = top2_a.values[:, 0] - top2_a.values[:, 1]
        gaps_b = top2_b.values[:, 0] - top2_b.values[:, 1]

        flips = (argmax_a != argmax_b).nonzero(as_tuple=True)[0]
        total_positions += a.shape[0]
        total_flips += flips.numel()
        for pos in flips.tolist():
            flip_gaps.append(
                (window_idx, pos, gaps_a[pos].item(), gaps_b[pos].item())
            )
        window_idx += 1

    if total_positions == 0:
        raise ValueError(
            f"No comparable windows found in {args.dir_a} and {args.dir_b}"
        )

    agreement = 100.0 * (total_positions - total_flips) / total_positions
    print(f"\nWindows compared:    {window_idx}")
    print(f"Positions compared:  {total_positions}")
    print(f"Max |logit| diff:    {max_abs_diff:.3e}")
    print(f"Top-1 agreement:     {agreement:.4f}% ({total_flips} flips)")

    if total_flips:
        worst_gap = max(max(ga, gb) for _, _, ga, gb in flip_gaps)
        print("\nFlips (window, position, top1-top2 gap run A, gap run B):")
        for w, p, ga, gb in flip_gaps[:50]:
            print(f"  w{w} pos{p}: gap_a={ga:.3e} gap_b={gb:.3e}")
        if total_flips > 50:
            print(f"  ... and {total_flips - 50} more")
        print(f"\nLargest gap at any flip: {worst_gap:.3e}")
        if worst_gap < 1e-3:
            print(
                "VERDICT: all flips occur at model-declared ties (gap < 1e-3). "
                "Pure rounding noise; no accuracy impact."
            )
        else:
            print(
                "VERDICT: flip(s) found at large logit gaps. This suggests a "
                "genuine correctness bug in the timing-selected kernels — "
                "report upstream (pytorch combo kernels / vllm PR #26682)."
            )
    else:
        print("\nVERDICT: argmax identical at every position across both runs.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    gen = sub.add_parser("generate", help="Generate prompt logits for all windows")
    gen.add_argument("--model", type=str, required=True)
    gen.add_argument("--output-dir", type=str, required=True)
    gen.add_argument("--dataset", type=str, required=True)
    gen.add_argument("--dataset-config", type=str, default=None)
    gen.add_argument("--context-length", type=int, default=2048)
    gen.add_argument("--stride", type=int, default=512)
    gen.add_argument(
        "--num-windows",
        type=int,
        default=4,
        help="Windows to save (default: 4; each window is a large file)",
    )
    gen.add_argument("--tensor-parallel-size", type=int, default=1)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.add_argument(
        "--deterministic",
        action="store_true",
        help="Use the frozen (combo-kernels-off) config instead of the "
        "wobbling one; two such runs should be bit-identical",
    )

    cmp_p = sub.add_parser("compare", help="Compare two generated logit dirs")
    cmp_p.add_argument("--dir-a", type=str, required=True)
    cmp_p.add_argument("--dir-b", type=str, required=True)

    args = parser.parse_args()
    if args.mode == "generate":
        generate_logits(args)
    else:
        compare_logits(args)


if __name__ == "__main__":
    main()
