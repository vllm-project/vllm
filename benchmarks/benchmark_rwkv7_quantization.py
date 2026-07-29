# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark RWKV7 16-bit, online INT8, TorchAO INT8/INT4, and BitsAndBytes.

Every setting runs in a fresh process.  Besides median output throughput, the
report extracts vLLM's model-resident GPU-memory measurement and compares
greedy tokens with the FP16 run.  The default gates encode the production
target: quantized model memory must decrease and throughput must not regress.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import regex as re

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

MODEL_MEMORY_PATTERN = re.compile(r"Model loading took ([0-9.]+) GiB memory")
NOISY_BNB_MESSAGES = ("MatMul8bitLt: inputs will be cast",)
SETTINGS = (
    "fp16",
    "online-int8",
    "torchao-int8",
    "torchao-int4",
    "bnb-int8",
    "bnb-nf4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="FP16 reference checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument("--int8-model", help="Pre-quantized BitsAndBytes INT8 model")
    parser.add_argument(
        "--int4-model",
        help="Pre-quantized BitsAndBytes NF4 model; omit for inflight NF4",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=SETTINGS,
        default=("fp16", "online-int8"),
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--logprobs", type=int, default=1)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="full-engine warmups; three also settles lazy BitsAndBytes state",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="half")
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
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-gates", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--require-repeatable", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--min-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-memory-reduction", type=float, default=0.01)
    parser.add_argument(
        "--worker-setting",
        choices=SETTINGS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-model", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-inflight-nf4", action="store_true", help=argparse.SUPPRESS
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


def run_worker(args: argparse.Namespace) -> None:
    assert args.worker_setting is not None
    assert args.worker_model is not None

    # Quantization benchmarks measure weight formats, not the experimental
    # recurrent kernel. Keep the recurrent path deterministic and fail-closed.
    import os

    os.environ["VLLM_RWKV7_KERNEL"] = "torch"
    if args.require_repeatable:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Keep the benchmark self-contained on CUDA hosts without a full nvcc
    # toolchain; sampling is outside the measured model/kernel scope.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams

    if args.require_repeatable:
        import torch

        torch.use_deterministic_algorithms(True)

    llm_kwargs: dict[str, Any] = {}
    if args.worker_setting == "online-int8":
        # vLLM's online-quant configuration was generalized from the original
        # scheme/override API to QuantSpec. Supporting both keeps this benchmark
        # usable while bisecting releases and downstream integration branches.
        from vllm.config import quantization as online_quant_config

        if hasattr(online_quant_config, "QuantizationConfigArgs"):
            quantization_config = {"linear": "int8_per_channel_static"}
        else:
            quantization_config = {
                "linear_scheme_override": "int8_per_channel_weight_only"
            }
        llm_kwargs.update(
            quantization="online",
            quantization_config=quantization_config,
        )
    elif args.worker_setting == "bnb-nf4" and args.worker_inflight_nf4:
        llm_kwargs["quantization"] = "bitsandbytes"
    elif args.worker_setting.startswith("torchao-"):
        from torchao.core.config import config_to_dict
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
        )

        if args.worker_setting == "torchao-int8":
            torchao_config = Int8WeightOnlyConfig()
        else:
            if args.dtype not in ("bfloat16", "bf16"):
                raise ValueError("torchao-int4 requires --dtype bfloat16")
            torchao_config = Int4WeightOnlyConfig(
                group_size=128,
                int4_packing_format="tile_packed_to_4d",
            )
        llm_kwargs.update(
            quantization="torchao",
            hf_overrides={
                "quantization_config_dict_json": json.dumps(
                    config_to_dict(torchao_config)
                )
            },
        )

    load_started = time.perf_counter()
    llm = LLM(
        model=args.worker_model,
        tokenizer=args.tokenizer or args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **llm_kwargs,
    )
    load_s = time.perf_counter() - load_started
    prompts = load_prompts(args)
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
    repeat_mismatch = None
    for run_index, signature in enumerate(signatures[1:], start=1):
        if signature == signatures[0]:
            continue
        for request_index, (reference_ids, candidate_ids) in enumerate(
            zip(signatures[0], signature)
        ):
            if reference_ids == candidate_ids:
                continue
            common = min(len(reference_ids), len(candidate_ids))
            token_index = next(
                (
                    index
                    for index in range(common)
                    if reference_ids[index] != candidate_ids[index]
                ),
                common,
            )
            repeat_mismatch = {
                "run": run_index,
                "request": request_index,
                "token": token_index,
            }
            break
        if repeat_mismatch is None:
            repeat_mismatch = {"run": run_index, "request": None, "token": None}
        break
    if repeat_mismatch is not None and args.require_repeatable:
        raise RuntimeError(
            f"{args.worker_setting} output changed across repeated runs: "
            f"{repeat_mismatch}"
        )

    elapsed = statistics.median(samples)
    output_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
    result = {
        "setting": args.worker_setting,
        "model": args.worker_model,
        "load_s": load_s,
        "elapsed_s": elapsed,
        "samples_s": samples,
        "output_tokens": output_tokens,
        "output_tok_s": output_tokens / elapsed,
        "repeatable": repeat_mismatch is None,
        "repeat_mismatch": repeat_mismatch,
        "requests": [list(item.outputs[0].token_ids) for item in outputs],
    }
    print("RESULT_JSON " + json.dumps(result), flush=True)


def _setting_model(args: argparse.Namespace, setting: str) -> tuple[str, bool]:
    if setting == "fp16":
        return args.model, False
    if setting == "online-int8" or setting.startswith("torchao-"):
        return args.model, False
    if setting == "bnb-int8":
        if args.int8_model is None:
            raise ValueError("--int8-model is required for bnb-int8")
        return args.int8_model, False
    if args.int4_model is None:
        return args.model, True
    return args.int4_model, False


def run_setting(args: argparse.Namespace, setting: str) -> dict[str, Any]:
    model, inflight_nf4 = _setting_model(args, setting)
    command = [
        sys.executable,
        __file__,
        "--model",
        args.model,
        "--worker-model",
        model,
        "--worker-setting",
        setting,
    ]
    for name in (
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
    command.append(
        "--require-repeatable" if args.require_repeatable else "--no-require-repeatable"
    )
    if inflight_nf4:
        command.append("--worker-inflight-nf4")

    result = None
    model_memory_gib = None
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        if match := MODEL_MEMORY_PATTERN.search(line):
            model_memory_gib = float(match.group(1))
        if line.startswith("RESULT_JSON "):
            result = json.loads(line.removeprefix("RESULT_JSON "))
        elif not any(message in line for message in NOISY_BNB_MESSAGES):
            print(f"[{setting}] {line}", end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{setting} worker exited with status {return_code}")
    if result is None:
        raise RuntimeError(f"{setting} worker produced no result")
    result["model_memory_gib"] = model_memory_gib
    return result


def compare_with_fp16(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    if len(reference["requests"]) != len(candidate["requests"]):
        raise RuntimeError("FP16 and quantized settings returned different batch sizes")

    matching_tokens = 0
    total_tokens = 0
    exact_requests = 0
    request_reports = []
    for index, (ref_ids, candidate_ids) in enumerate(
        zip(reference["requests"], candidate["requests"])
    ):
        common = min(len(ref_ids), len(candidate_ids))
        first_difference = next(
            (pos for pos in range(common) if ref_ids[pos] != candidate_ids[pos]),
            None,
        )
        if first_difference is None and len(ref_ids) != len(candidate_ids):
            first_difference = common
        exact = first_difference is None
        exact_requests += int(exact)
        matching_tokens += sum(
            ref_token == candidate_token
            for ref_token, candidate_token in zip(ref_ids, candidate_ids)
        )
        total_tokens += max(len(ref_ids), len(candidate_ids))
        request_reports.append(
            {
                "request": index,
                "exact": exact,
                "first_difference": first_difference,
                "fp16_length": len(ref_ids),
                "candidate_length": len(candidate_ids),
            }
        )

    memory_reduction = None
    if (
        reference["model_memory_gib"] is not None
        and candidate["model_memory_gib"] is not None
    ):
        memory_reduction = 1.0 - (
            candidate["model_memory_gib"] / reference["model_memory_gib"]
        )
    return {
        "setting": candidate["setting"],
        "speed_ratio": candidate["output_tok_s"] / reference["output_tok_s"],
        "memory_reduction": memory_reduction,
        "token_agreement": matching_tokens / total_tokens if total_tokens else 1.0,
        "exact_requests": exact_requests,
        "total_requests": len(request_reports),
        "fp16_output_tok_s": reference["output_tok_s"],
        "candidate_output_tok_s": candidate["output_tok_s"],
        "fp16_model_memory_gib": reference["model_memory_gib"],
        "candidate_model_memory_gib": candidate["model_memory_gib"],
        "requests": request_reports,
    }


def main() -> None:
    args = parse_args()
    if args.worker_setting is not None:
        run_worker(args)
        return
    if "fp16" not in args.settings:
        raise ValueError("--settings must include fp16 as the reference")

    results = {setting: run_setting(args, setting) for setting in args.settings}
    comparisons = [
        compare_with_fp16(results["fp16"], results[setting])
        for setting in args.settings
        if setting != "fp16"
    ]
    report = {"results": results, "comparisons": comparisons}
    print(json.dumps(report, indent=2))

    failed = []
    for comparison in comparisons:
        if comparison["speed_ratio"] < args.min_speed_ratio:
            failed.append(
                f"{comparison['setting']} speed_ratio={comparison['speed_ratio']:.4f}"
            )
        reduction = comparison["memory_reduction"]
        if reduction is None or reduction < args.min_memory_reduction:
            failed.append(f"{comparison['setting']} memory_reduction={reduction}")
    if args.require_gates and failed:
        print("FAILED_GATES " + "; ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
