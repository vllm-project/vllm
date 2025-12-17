#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Quick A/B harness to gauge quality impact of rank-based K compression.

Example:
  python tools/eval_kv_key_compression.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --max-tokens 64 \\
    --energy 0.995 \\
    --compare

Notes:
- Uses enforce_eager to keep Python hooks active.
- Compression is enabled via envs; runs are executed in separate processes
  to avoid env caching side effects.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import RequestOutputKind


DEFAULT_PROMPTS = [
    "def merge_sort(arr):",
    "class BinarySearchTree:",
    "def solve_sudoku(board):",
]


def _logprob_for_token(
    logprobs: Any, position: int, token_id: int
) -> float | None:
    """Extract logprob of the sampled token at a given position."""
    if logprobs is None:
        return None

    # FlatLogprobs
    if hasattr(logprobs, "start_indices"):
        start = logprobs.start_indices[position]
        end = logprobs.end_indices[position]
        for idx in range(start, end):
            if logprobs.token_ids[idx] == token_id:
                return float(logprobs.logprobs[idx])
        return None

    # List[dict]
    entry = logprobs[position] if position < len(logprobs) else None
    if not entry:
        return None
    token_info = entry.get(token_id)
    if token_info is None:
        return None
    return float(getattr(token_info, "logprob", None))


def run_single(
    model: str,
    prompts: list[str],
    max_tokens: int,
    logprobs: int,
) -> dict[str, Any]:
    """Run one configuration (baseline or compressed) and return metrics."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=logprobs,
        prompt_logprobs=logprobs,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    llm = LLM(
        model=model,
        enforce_eager=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    runs: list[dict[str, Any]] = []

    for out in outputs:
        seq = out.outputs[0]
        token_ids = seq.token_ids
        chosen_logprobs = [
            _logprob_for_token(seq.logprobs, idx, tok_id)
            for idx, tok_id in enumerate(token_ids)
        ]
        valid = [lp for lp in chosen_logprobs if lp is not None]
        runs.append(
            {
                "prompt": out.prompt,
                "text": seq.text,
                "token_ids": token_ids,
                "logprobs": chosen_logprobs,
                "avg_logprob": float(sum(valid) / len(valid)) if valid else None,
            }
        )

    return {"runs": runs}


def _launch_subprocess(args: argparse.Namespace, enable_compression: bool) -> dict:
    """Run this script in a fresh process to avoid env caching issues."""
    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = env.get("VLLM_LOGGING_LEVEL", "ERROR")
    env["VLLM_USE_V1_MODEL_RUNNER"] = env.get("VLLM_USE_V1_MODEL_RUNNER", "1")
    env["VLLM_KV_KEY_COMPRESS_ENABLED"] = "1" if enable_compression else "0"

    if enable_compression:
        env["VLLM_KV_KEY_COMPRESS_ENERGY"] = str(args.energy)
        env["VLLM_KV_KEY_COMPRESS_MAX_RANK"] = str(args.max_rank or 0)
        env["VLLM_KV_KEY_COMPRESS_LAYERS"] = args.layers or ""
        env["VLLM_KV_KEY_COMPRESS_MIN_TOKENS"] = str(args.min_tokens)
        env["VLLM_KV_KEY_COMPRESS_RECOMPUTE_EVERY"] = str(args.recompute_every)

    cmd = [
        sys.executable,
        __file__,
        "--single-run",
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--logprobs",
        str(args.logprobs),
    ]

    for prompt in args.prompts:
        cmd.extend(["--prompt", prompt])

    completed = subprocess.run(
        cmd, env=env, check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def compare_runs(
    baseline: dict[str, Any], compressed: dict[str, Any]
) -> dict[str, Any]:
    """Token-level diff between baseline and compressed runs."""
    pairs: list[dict[str, Any]] = []
    for b, c in zip(baseline["runs"], compressed["runs"]):
        common_len = min(len(b["token_ids"]), len(c["token_ids"]))
        matches = sum(
            1
            for i in range(common_len)
            if b["token_ids"][i] == c["token_ids"][i]
        )

        def sum_logprobs(logprobs):
            vals = [
                lp for lp in (logprobs or [])[:common_len] if lp is not None
            ]
            return sum(vals) if vals else None

        b_lp = sum_logprobs(b.get("logprobs"))
        c_lp = sum_logprobs(c.get("logprobs"))
        delta_lp = None
        if b_lp is not None and c_lp is not None:
            delta_lp = c_lp - b_lp

        pairs.append(
            {
                "prompt": b["prompt"],
                "token_match_rate": matches / common_len if common_len else 0.0,
                "delta_logprob": delta_lp,
                "baseline_avg_logprob": b.get("avg_logprob"),
                "compressed_avg_logprob": c.get("avg_logprob"),
            }
        )

    return {
        "pairs": pairs,
        "mean_token_match_rate": sum(p["token_match_rate"] for p in pairs)
        / len(pairs),
        "mean_delta_logprob": sum(
            p["delta_logprob"] for p in pairs if p["delta_logprob"] is not None
        )
        / max(
            1,
            len(
                [
                    p
                    for p in pairs
                    if p["delta_logprob"] is not None
                ]
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KV key rank compression quality."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt string (repeatable). Defaults to 3 code prompts.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--energy", type=float, default=0.995)
    parser.add_argument("--max-rank", type=int, default=0)
    parser.add_argument("--layers", type=str, default="")
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--recompute-every", type=int, default=8)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline vs compressed and report deltas.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prompts:
        args.prompts = DEFAULT_PROMPTS

    if args.compare:
        baseline = _launch_subprocess(args, enable_compression=False)
        compressed = _launch_subprocess(args, enable_compression=True)
        summary = compare_runs(baseline, compressed)
        print(json.dumps(summary, indent=2))
        return

    # Single run (used by subprocess or if user only wants one config).
    result = run_single(
        model=args.model,
        prompts=args.prompts,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
