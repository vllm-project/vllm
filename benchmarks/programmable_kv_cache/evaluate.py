# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare model accuracy with real prefix reuse and forced recomputation."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmarks.programmable_kv_cache.client import answer_number
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Downloaded GSM8K repository containing main/*.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-fewshot", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--kv-cache-memory-bytes", type=int)
    memory.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--min-accuracy", type=float, default=0.8)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--enable-kv-control", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    train = pq.read_table(
        args.dataset / "main/train-00000-of-00001.parquet"
    ).to_pylist()
    test = pq.read_table(args.dataset / "main/test-00000-of-00001.parquet").to_pylist()
    if not 0 < args.num_samples <= len(test) or not 0 <= args.num_fewshot <= len(train):
        parser.error("Invalid sample or few-shot count")
    indices = random.Random(args.seed).sample(range(len(test)), args.num_samples)
    examples = train[: args.num_fewshot]
    llm_args = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 2048,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "mamba_cache_mode": "align",
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "dtype": "bfloat16",
    }
    if args.enable_kv_control:
        llm_args["kv_control_config"] = {"engine_id": "evaluation"}
    llm = LLM(**llm_args)
    report: dict[str, Any] = {
        "configuration": llm_args,
        "seed": args.seed,
        "num_fewshot": args.num_fewshot,
        "prompt_format": "gsm8k_completion_cot",
        "sampling": {"temperature": 0, "max_tokens": args.max_tokens},
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "max_accuracy_drop": args.max_accuracy_drop,
            "max_truncated_fraction": 0.01,
        },
        "indices": indices,
        "runs": {},
    }
    try:
        report["cache_groups"] = llm.llm_engine.engine_core.call_utility(
            "get_kv_cache_group_metadata"
        )
        tokenizer = llm.get_tokenizer()
        shared = "".join(
            f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
            for example in examples
        )
        shared_tokens = len(tokenizer.encode(shared, add_special_tokens=False))
        assert shared_tokens >= max(g["block_size"] for g in report["cache_groups"]), (
            "Few-shot prefix is too short to exercise this model's reusable boundaries"
        )
        prompts = [
            {
                "prompt_token_ids": tokenizer.encode(
                    shared + f"Question: {test[i]['question']}\nAnswer:",
                    add_special_tokens=False,
                )
            }
            for i in indices
        ]
        assert (
            max(len(p["prompt_token_ids"]) for p in prompts) + args.max_tokens
            <= args.max_model_len
        )
        llm.generate(
            prompts[:1], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
        )
        for mode, skip in (("recompute", True), ("prefix_cache", False)):
            assert llm.reset_prefix_cache()
            started = time.perf_counter()
            outputs = llm.generate(
                prompts,
                SamplingParams(
                    temperature=0,
                    max_tokens=args.max_tokens,
                    stop=["Question:", "Assistant:", "<|separator|>"],
                    skip_reading_prefix_cache=skip,
                ),
                use_tqdm=False,
            )
            elapsed = time.perf_counter() - started
            rows = []
            for index, output in zip(indices, outputs, strict=True):
                assert output.prompt_token_ids is not None
                completion = output.outputs[0]
                predicted = answer_number(completion.text)
                expected = answer_number(test[index]["answer"])
                rows.append(
                    {
                        "index": index,
                        "expected": expected,
                        "predicted": predicted,
                        "correct": expected is not None and predicted == expected,
                        "text": completion.text,
                        "token_ids": list(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "prompt_tokens": len(output.prompt_token_ids),
                        "cached_tokens": output.num_cached_tokens,
                    }
                )
            accuracy = sum(row["correct"] for row in rows) / len(rows)
            report["runs"][mode] = {
                "seconds": elapsed,
                "accuracy": accuracy,
                "length_limited": sum(row["finish_reason"] == "length" for row in rows),
                "rows": rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "samples": len(rows),
                        "accuracy": accuracy,
                        "length_limited": report["runs"][mode]["length_limited"],
                        "cached_tokens": sum(row["cached_tokens"] or 0 for row in rows),
                    }
                ),
                flush=True,
            )
        baseline, cached = report["runs"]["recompute"], report["runs"]["prefix_cache"]
        report["answer_mismatches"] = sum(
            a["predicted"] != b["predicted"]
            for a, b in zip(baseline["rows"], cached["rows"], strict=True)
        )
        report["token_mismatches"] = sum(
            a["token_ids"] != b["token_ids"]
            for a, b in zip(baseline["rows"], cached["rows"], strict=True)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert all(not row["cached_tokens"] for row in baseline["rows"])
        assert any(row["cached_tokens"] for row in cached["rows"])
        assert baseline["length_limited"] <= 0.01 * args.num_samples
        assert cached["length_limited"] <= 0.01 * args.num_samples
        assert min(baseline["accuracy"], cached["accuracy"]) >= args.min_accuracy
        assert baseline["accuracy"] - cached["accuracy"] <= args.max_accuracy_drop
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
