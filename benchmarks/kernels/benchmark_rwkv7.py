# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare an RWKV7 candidate kernel policy with Torch through the vLLM engine.

Each backend runs in a fresh subprocess so environment selection, CUDA graphs,
and recurrent caches cannot leak between runs. The report identifies the first
greedy-token divergence and includes the top log-probabilities at that position.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_PROMPTS = [
    "The Eiffel Tower is located in",
    "A short proof that there are infinitely many primes begins",
    "Write a Python function that computes Fibonacci numbers:",
    "The most important property of recurrent neural networks is",
    "Once upon a time in a quiet village",
    "Explain why the sky appears blue during the day.",
    "In numerical analysis, floating point accumulation order",
    "Translate to Chinese: artificial intelligence inference engine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--require-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--candidate-backend", choices=("auto", "triton"), default="triton"
    )
    parser.add_argument(
        "--worker",
        choices=("auto", "torch", "triton"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.warmup_runs < 0 or args.repeats < 1:
        raise ValueError("--warmup-runs must be >= 0 and --repeats must be >= 1")
    prompts = list(args.prompt)
    if args.prompts_file is not None:
        loaded = json.loads(args.prompts_file.read_text())
        if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
            raise ValueError("--prompts-file must contain a JSON array of strings")
        prompts.extend(loaded)
    return prompts or DEFAULT_PROMPTS


def serialize_logprobs(logprobs: Any) -> list[list[dict[str, Any]]]:
    serialized = []
    for position in logprobs or []:
        candidates = [
            {
                "token_id": int(token_id),
                "logprob": float(value.logprob),
                "rank": value.rank,
            }
            for token_id, value in position.items()
        ]
        candidates.sort(key=lambda item: item["logprob"], reverse=True)
        serialized.append(candidates)
    return serialized


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker is not None
    os.environ["VLLM_RWKV7_KERNEL"] = args.worker

    from vllm import LLM, SamplingParams

    prompts = load_prompts(args)
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
        ignore_eos=args.ignore_eos,
    )
    for _ in range(args.warmup_runs):
        llm.generate(prompts, sampling_params, use_tqdm=False)
    samples = []
    signatures = []
    outputs = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        samples.append(time.perf_counter() - started)
        signatures.append([tuple(item.outputs[0].token_ids) for item in outputs])
    assert outputs is not None
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError(f"{args.worker} output changed across repeated runs")
    elapsed = statistics.median(samples)
    result = {
        "backend": args.worker,
        "elapsed_s": elapsed,
        "samples_s": samples,
        "output_tokens": sum(len(item.outputs[0].token_ids) for item in outputs),
        "requests": [
            {
                "token_ids": list(item.outputs[0].token_ids),
                "logprobs": serialize_logprobs(item.outputs[0].logprobs),
            }
            for item in outputs
        ],
    }
    result["output_tok_s"] = result["output_tokens"] / elapsed
    print("RESULT_JSON " + json.dumps(result), flush=True)


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    command = [sys.executable, __file__]
    for name in (
        "model",
        "max_tokens",
        "logprobs",
        "warmup_runs",
        "repeats",
        "dtype",
        "max_model_len",
        "max_num_batched_tokens",
        "gpu_memory_utilization",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.tokenizer is not None:
        command.extend(["--tokenizer", args.tokenizer])
    for prompt in args.prompt:
        command.extend(["--prompt", prompt])
    if args.prompts_file is not None:
        command.extend(["--prompts-file", str(args.prompts_file)])
    command.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    command.append(
        "--async-scheduling" if args.async_scheduling else "--no-async-scheduling"
    )
    command.append("--ignore-eos" if args.ignore_eos else "--no-ignore-eos")
    command.extend(["--worker", backend])

    result = None
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith("RESULT_JSON "):
            result = json.loads(line.removeprefix("RESULT_JSON "))
        else:
            print(f"[{backend}] {line}", end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{backend} worker exited with status {return_code}")
    if result is None:
        raise RuntimeError(f"{backend} worker produced no result")
    return result


def top_margin(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    return candidates[0]["logprob"] - candidates[1]["logprob"]


def compare_results(
    torch_result: dict[str, Any], candidate_result: dict[str, Any]
) -> dict[str, Any]:
    if len(torch_result["requests"]) != len(candidate_result["requests"]):
        raise RuntimeError("Reference and candidate returned different batch sizes")
    request_reports = []
    exact_requests = 0
    for request_idx, (torch_request, candidate_request) in enumerate(
        zip(torch_result["requests"], candidate_result["requests"])
    ):
        torch_ids = torch_request["token_ids"]
        candidate_ids = candidate_request["token_ids"]
        first_difference = next(
            (
                index
                for index, (torch_id, candidate_id) in enumerate(
                    zip(torch_ids, candidate_ids)
                )
                if torch_id != candidate_id
            ),
            None,
        )
        if first_difference is None and len(torch_ids) != len(candidate_ids):
            first_difference = min(len(torch_ids), len(candidate_ids))
        exact = first_difference is None
        exact_requests += int(exact)
        report: dict[str, Any] = {
            "request": request_idx,
            "exact": exact,
            "torch_length": len(torch_ids),
            "candidate_length": len(candidate_ids),
            "first_difference": first_difference,
        }
        if first_difference is not None:
            position = first_difference
            torch_top = (
                torch_request["logprobs"][position]
                if position < len(torch_request["logprobs"])
                else []
            )
            candidate_top = (
                candidate_request["logprobs"][position]
                if position < len(candidate_request["logprobs"])
                else []
            )
            report.update(
                torch_token=torch_ids[position] if position < len(torch_ids) else None,
                candidate_token=(
                    candidate_ids[position] if position < len(candidate_ids) else None
                ),
                torch_top_logprobs=torch_top,
                candidate_top_logprobs=candidate_top,
                torch_top2_margin=top_margin(torch_top),
                candidate_top2_margin=top_margin(candidate_top),
            )
        request_reports.append(report)
    return {
        "exact": exact_requests == len(request_reports),
        "exact_requests": exact_requests,
        "total_requests": len(request_reports),
        "candidate_backend": candidate_result["backend"],
        "torch_output_tok_s": torch_result["output_tok_s"],
        "candidate_output_tok_s": candidate_result["output_tok_s"],
        "speedup": candidate_result["output_tok_s"] / torch_result["output_tok_s"],
        "requests": request_reports,
    }


def main() -> None:
    args = parse_args()
    if args.worker is not None:
        run_worker(args)
        return

    torch_result = run_backend(args, "torch")
    candidate_result = run_backend(args, args.candidate_backend)
    comparison = compare_results(torch_result, candidate_result)
    print(json.dumps(comparison, indent=2))
    if args.require_exact and not comparison["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
