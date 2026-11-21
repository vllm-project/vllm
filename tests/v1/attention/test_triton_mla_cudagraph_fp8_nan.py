# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for vLLM #57017.

DeepSeek-V2-Lite with online FP8 on an sm_89 GPU (TRITON_MLA, the only MLA
backend on Ada) returned NaN logits once >=3 sequences decoded in one
CUDA-graph batch; eager and ``cudagraph_mode=NONE`` were fine.

Root cause: the MLA split-KV decode logits workspace (``Att_Out`` in
``triton_mla.forward_mqa``), served from the shared ``WorkspaceManager``
scratch pool via ``get_simultaneous`` (which returns offset-0 slices that
alias across MLA layers / other scratch users), corrupted under CUDA-graph
replay at the size-4+ capture bucket -> NaN attention output -> NaN logits.

This test loads the model with dummy weights (online FP8, default
compile+CUDA graphs) and asserts that a 3-sequence decode batch produces no
NaN log-probs. Default ``cudagraph_capture_sizes`` is ``[1, 2, 4, 8, ...]``,
so 3 sequences land in the size-4 graph -- the first previously-broken bucket.
"""

import math

import pytest
import torch

from vllm import LLM, SamplingParams

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
# 3 prompts: with default cudagraph_capture_sizes [1, 2, 4, 8, ...], 3
# sequences pad to the size-4 captured graph (the first broken bucket).
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "Quantum computing and large language models are reshaping AI. " * 8,
    "Explain general relativity in simple terms for a beginner. " * 8,
]


def _nan_logprob_count(outs):
    n_nan = n_total = 0
    for o in outs:
        for step in o.outputs[0].logprobs or []:
            items = step.items() if isinstance(step, dict) else []
            for _tok, lp in items:
                v = getattr(lp, "logprob", None)
                if isinstance(v, float):
                    n_total += 1
                    if math.isnan(v):
                        n_nan += 1
    return n_nan, n_total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsv2lite_fp8_cudagraph_no_nan():
    """3-seq MLA decode under CUDA graphs must not yield NaN log-probs."""
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        quantization="fp8",
        dtype="bfloat16",
        max_model_len=4096,
        load_format="dummy",
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=1)
    outs = llm.generate(PROMPTS, sp)
    n_nan, n_total = _nan_logprob_count(outs)
    assert n_nan == 0, (
        f"NaN log-probs in MLA cudagraph decode: {n_nan}/{n_total}; "
        f"texts={[o.outputs[0].text[:24] for o in outs]}"
    )
