# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-only (non-causal) attention under FULL CUDA graphs: a single prompt
replayed through a graph captured on a many-request dummy batch must give the
same result as eager execution."""

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

MODEL = "hmellor/tiny-random-LlamaForCausalLM"


def _last_logprobs(cudagraph_mode: str, prompts: list[list[int]]) -> list[list[float]]:
    llm = LLM(
        MODEL,
        hf_overrides={"is_causal": False},
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.3,
        dtype="float32",
        compilation_config={
            "cudagraph_mode": cudagraph_mode,
            "max_cudagraph_capture_size": 1024,
        },
    )
    params = SamplingParams(
        max_tokens=1, temperature=0.0, logprob_token_ids=list(range(100, 132))
    )
    out = []
    for prompt in prompts:  # one request per step: each replays a FULL graph
        (result,) = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        out.append([lp.logprob for lp in result.outputs[0].logprobs[0].values()])
    del llm
    torch.cuda.empty_cache()
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA graphs need CUDA")
def test_encoder_only_full_cudagraph_matches_eager():
    generator = torch.Generator().manual_seed(0)
    prompts = [
        torch.randint(100, 30000, (n,), generator=generator).tolist()
        for n in (23, 150, 600, 1000)
    ]
    expected = _last_logprobs("NONE", prompts)
    actual = _last_logprobs("FULL", prompts)
    for exp, act in zip(expected, actual):
        torch.testing.assert_close(
            torch.tensor(act), torch.tensor(exp), atol=1e-4, rtol=1e-4
        )
