# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-level correctness tests for the MORI-backed UMBP runtime."""

import os
import time

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.platforms import current_platform

pytest.importorskip("mori.cpp")

if not current_platform.is_cuda_alike():
    pytest.skip("requires CUDA or ROCm", allow_module_level=True)


def _flush_gpu_cache(llm: LLM, sampling_params: SamplingParams) -> None:
    cache_config = llm.llm_engine.vllm_config.cache_config
    total_tokens = int(cache_config.num_gpu_blocks * cache_config.block_size * 1.5)
    tokens_per_request = 512
    num_requests = (total_tokens + tokens_per_request - 1) // tokens_per_request
    for start in range(0, num_requests, 16):
        prompts = [
            TokensPrompt(prompt_token_ids=[(index % 100) + 3] * tokens_per_request)
            for index in range(start, min(start + 16, num_requests))
        ]
        llm.generate(prompts, sampling_params, use_tqdm=False)


@pytest.mark.skipif(
    "VLLM_UMBP_TEST_MODEL" not in os.environ,
    reason="set VLLM_UMBP_TEST_MODEL to a local model path",
)
def test_mori_external_kv_preserves_generated_tokens(tmp_path):
    model = os.environ["VLLM_UMBP_TEST_MODEL"]
    llm = LLM(
        model=model,
        enforce_eager=True,
        compilation_config={"custom_ops": ["none"]},
        max_model_len=1024,
        kv_cache_memory_bytes=16 << 20,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                "capacity_bytes": 256 << 20,
                "lookup_dir": str(tmp_path),
                "num_workers": 2,
                "timeout_ms": 60000,
            },
        ),
    )
    sampling_params = SamplingParams(max_tokens=4, temperature=0)
    prompt = "The first ten natural numbers are " * 32

    cold = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
    _flush_gpu_cache(llm, SamplingParams(max_tokens=1, temperature=0))
    time.sleep(1)
    restored = llm.generate(prompt, sampling_params, use_tqdm=False)[0]

    assert restored.outputs[0].token_ids == cold.outputs[0].token_ids
    assert restored.num_cached_tokens > 0
