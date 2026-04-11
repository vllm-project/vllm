# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for vllm-project/vllm#39589.

Concurrent variable-length prefills at temperature=0 must be deterministic.
Before the fix, stale block IDs leaking in the tail of reused block_table
rows caused FlashInfer's page-indices kernel to read KV data from other
requests, producing non-deterministic output.
"""

import pytest
import torch

from vllm import LLM, SamplingParams

from ....utils import create_new_process_for_each_test


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="concurrent-prefill repro requires a CUDA device",
)
@create_new_process_for_each_test()
def test_concurrent_variable_length_prefill_is_deterministic():
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        enforce_eager=True,  # rule out cudagraph-shape interactions
        max_model_len=4096,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    # Mix lengths so the bug's trigger (variable-length concurrent prefill)
    # is actually exercised.
    prompts = [
        "word " * 128,
        "token " * 256,
        "phrase " * 64,
        "chunk " * 512,
    ]
    baseline = llm.generate(prompts, sp)
    for trial in range(20):
        out = llm.generate(prompts, sp)
        for b, o in zip(baseline, out):
            assert b.outputs[0].text == o.outputs[0].text, (
                f"trial={trial} diverged for prompt len="
                f"{len(b.prompt.split())}: {b.outputs[0].text!r} vs "
                f"{o.outputs[0].text!r}"
            )
