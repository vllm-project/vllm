# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end smoke test for N-gram speculative decoding.

Uses the vLLM LLM engine with ngram speculative config to generate text
and report acceptance metrics — the same pattern as test_eagle.py.

Run with:
    python examples/features/speculative_decoding/test_ngram.py
"""

import argparse
import os
import time

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

if __name__ == "__main__":
    prompts = [
        "Guess who am i, i am a famous movie star"
    ]

    sampling_params = SamplingParams(
        temperature=0.0, top_p=0.95, top_k=40, max_tokens=50
    )
    num_speculative_tokens = 3

    llm = LLM(
        model="/root/.cache/models/modelscope/models/unsloth/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        enforce_eager=True,
        # distributed_executor_backend="mp",
        gpu_memory_utilization=0.3,
        disable_log_stats=False,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": num_speculative_tokens,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        },
        max_model_len=4096,
        enable_prefix_caching=False,
    )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(output.outputs[0].token_ids)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    metrics = llm.get_metrics()

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_speculative_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    print("-" * 50)
    print(f"total_num_output_tokens: {total_num_output_tokens}")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    )
    print(f"mean acceptance length: {acceptance_length:.2f}")
    print("-" * 50)

    # Print acceptance at each token position
    for i in range(len(acceptance_counts)):
        acceptance_rate = (
            acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        )
        print(f"acceptance at token {i}: {acceptance_rate:.2f}")
