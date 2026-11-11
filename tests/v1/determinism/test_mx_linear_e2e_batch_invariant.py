# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 and MXFP4 linear layers must give a prompt the same tokens and
logprobs whatever else is batched with it under VLLM_BATCH_INVARIANT=1, on
every --linear-backend: selection routes them to a batch-invariant kernel and
overrides a backend (Marlin) that picks its tiling from the token count."""

import random

import pytest
import torch
from utils import _extract_step_logprobs, _random_prompt

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA MX linear kernel selection"
)

MODELS = {
    "mxfp8": "vllm-project/Qwen3-0.6B-MXFP8",
    "mxfp4": "nm-testing/TinyLlama-1.1B-Chat-v1.0-MXFP4",
}
BATCH_SIZES = (4, 16, 32)
NEEDLE = "Write one factual sentence about the moon."


@pytest.mark.parametrize("linear_backend", ["auto", "marlin"])
@pytest.mark.parametrize("model", MODELS.values(), ids=MODELS.keys())
def test_mx_linear_generation_is_batch_invariant(monkeypatch, model, linear_backend):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    random.seed(12345)
    llm = LLM(
        model=model,
        kernel_config={"linear_backend": linear_backend},
        attention_config={"backend": "FLASH_ATTN"},
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=max(BATCH_SIZES),
        max_model_len=2048,
        gpu_memory_utilization=0.3,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5)
    try:
        baseline = llm.generate([NEEDLE], sampling, use_tqdm=False)[0]
        base_logprobs, base_tokens = _extract_step_logprobs(baseline)
        assert base_logprobs is not None

        for batch_size in BATCH_SIZES:
            prompts = [_random_prompt(32, 96) for _ in range(batch_size)]
            position = random.randrange(batch_size)
            prompts[position] = NEEDLE
            output = llm.generate(prompts, sampling, use_tqdm=False)[position]
            assert output.prompt == NEEDLE
            logprobs, tokens = _extract_step_logprobs(output)

            assert tokens == base_tokens, f"tokens changed at batch size {batch_size}"
            assert torch.equal(logprobs, base_logprobs), (
                f"logprobs changed at batch size {batch_size}: "
                f"{(logprobs - base_logprobs).abs().max().item():.3e}"
            )
    finally:
        del llm
        cleanup_dist_env_and_memory()
