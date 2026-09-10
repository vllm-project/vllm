# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded full-model engine benchmark for PCP final-row restoration."""

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--batch-submission", choices=("queued", "immediate"), default="queued"
    )
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        load_format="fastsafetensors",
        tensor_parallel_size=1,
        prefill_context_parallel_size=4,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        kv_cache_dtype="fp8",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        num_gpu_blocks_override=1024,
        enforce_eager=True,
        disable_log_stats=False,
        disable_custom_all_reduce=True,
        moe_backend="flashinfer_cutlass",
        attention_config={"mla_prefill_backend": "FLASH_ATTN"},
        kernel_config={"enable_jit_warmup": False, "enable_flashinfer_autotune": False},
        compilation_config={"mode": "NONE", "cudagraph_mode": "NONE"},
        seed=0,
    )
    tokenizer = llm.get_tokenizer()
    prompts = []
    for index in range(4):
        text = (
            f"Section {index}. Explain context parallel inference "
            "and its memory requirements. " * 200
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)[:1024]
        assert len(tokens) == 1024
        prompts.append({"prompt_token_ids": tokens})
    sampling = SamplingParams(
        temperature=0, max_tokens=32, min_tokens=32, ignore_eos=True
    )
    trials = []
    for iteration in range(5):
        # Queue the entire batch before scheduling so submission races do not
        # change the prefill/decode step mix between benchmark arms.
        if args.batch_submission == "queued":
            llm.sleep(level=0, mode="keep")
        start = time.perf_counter()
        if args.batch_submission == "queued":
            llm.enqueue(prompts, sampling, use_tqdm=False)
            llm.wake_up(tags=["scheduling"])
            outputs = llm.wait_for_completion(use_tqdm=False)
        else:
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        ttfts = []
        tpots = []
        for output in outputs:
            metrics = output.metrics
            assert metrics is not None and metrics.first_token_latency > 0
            assert not metrics.is_corrupted
            ttfts.append(metrics.first_token_latency)
            tpots.append((metrics.last_token_ts - metrics.first_token_ts) / 31)
        if iteration >= 2:
            trials.append(
                {
                    "elapsed_s": elapsed,
                    "ttft_ms": [value * 1000 for value in ttfts],
                    "tpot_ms": [value * 1000 for value in tpots],
                    "output_tokens_per_s": sum(
                        len(o.outputs[0].token_ids) for o in outputs
                    )
                    / elapsed,
                    "tokens": [list(o.outputs[0].token_ids) for o in outputs],
                }
            )
    result = {
        "settings": {
            "tp": 1,
            "pcp": 4,
            "input_tokens": 1024,
            "output_tokens": 32,
            "concurrency": 4,
            "warmups": 2,
            "trials": 3,
            "eager": True,
            "batch_submission": args.batch_submission,
        },
        "trials": trials,
        "median_ttft_ms": statistics.median(
            statistics.mean(t["ttft_ms"]) for t in trials
        ),
        "median_output_tokens_per_s": statistics.median(
            t["output_tokens_per_s"] for t in trials
        ),
        "median_tpot_ms": statistics.median(
            statistics.mean(t["tpot_ms"]) for t in trials
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}), flush=True)


if __name__ == "__main__":
    main()
