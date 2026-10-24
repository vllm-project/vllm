# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-level correctness tests for the MORI-backed UMBP runtime."""

import os
import time

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import CUDAGraphMode, KVTransferConfig
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter

if not hasattr(pytest.importorskip("mori.cpp"), "UMBPClient"):
    pytest.skip("MORI was built without UMBP", allow_module_level=True)

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


def _restore_counters(llm: LLM) -> dict[str, int]:
    names = {
        "vllm:prefix_cache_hits",
        "vllm:external_prefix_cache_hits",
        "vllm:umbp_transfer_completed",
        "vllm:umbp_transfer_failed",
    }
    values = dict.fromkeys(names, 0)
    for metric in llm.get_metrics():
        if (
            isinstance(metric, Counter)
            and metric.name in names
            and metric.labels.get("operation", "load") == "load"
        ):
            values[metric.name] += metric.value
    return values


@pytest.mark.parametrize("lazy_offload", [False, True])
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "full_graph"])
def test_mori_external_kv_preserves_generated_tokens(
    tmp_path, lazy_offload, enforce_eager
):
    model = os.environ.get("VLLM_UMBP_TEST_MODEL", "Qwen/Qwen3-0.6B")
    llm = LLM(
        model=model,
        enforce_eager=enforce_eager,
        compilation_config={
            "custom_ops": ["none"] if enforce_eager else ["all"],
            "cudagraph_mode": "NONE" if enforce_eager else "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1],
        },
        attention_config={"backend": "TRITON_ATTN"},
        disable_log_stats=False,
        max_model_len=1024,
        gpu_memory_utilization=0.05,
        # Keep GPU capacity small enough for the flush to drive lazy offload.
        block_size=16,
        num_gpu_blocks_override=128,
        enable_prefix_caching=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UMBPStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "mode": "embedded",
                # Holds the prompt and every flush request: only GPU memory evicts.
                "capacity_bytes": 2 << 30,
                "lookup_dir": str(tmp_path),
                "num_workers": 2,
                "timeout_ms": 60000,
                "load_async": False,
                "lazy_offload": lazy_offload,
            },
        ),
    )
    if not enforce_eager:
        assert (
            llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
            == CUDAGraphMode.FULL_DECODE_ONLY
        )
    sampling_params = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
    # Restore 16 full blocks, leaving a one-token batch eligible for FULL replay.
    token_ids = llm.get_tokenizer().encode("The first ten natural numbers are " * 64)
    prompt = TokensPrompt(prompt_token_ids=token_ids[:257])
    assert len(prompt["prompt_token_ids"]) == 257

    cold = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
    # Drive lazy stores, then clear only GPU prefixes once store pins are released.
    _flush_gpu_cache(llm, SamplingParams(max_tokens=1, temperature=0))
    deadline = time.monotonic() + 30
    reset = llm.reset_prefix_cache(reset_connector=False)
    while not reset and time.monotonic() < deadline:
        time.sleep(0.01)
        reset = llm.reset_prefix_cache(reset_connector=False)
    assert reset, "UMBP stores did not release their GPU block pins"
    before = _restore_counters(llm)
    restored = llm.generate(prompt, sampling_params, use_tqdm=False)[0]
    after = _restore_counters(llm)

    assert restored.outputs[0].token_ids == cold.outputs[0].token_ids
    assert restored.num_cached_tokens == 256
    assert after["vllm:prefix_cache_hits"] == before["vllm:prefix_cache_hits"]
    assert (
        after["vllm:external_prefix_cache_hits"]
        - before["vllm:external_prefix_cache_hits"]
        == 256
    )
    assert (
        after["vllm:umbp_transfer_completed"] > before["vllm:umbp_transfer_completed"]
    )
    assert after["vllm:umbp_transfer_failed"] == before["vllm:umbp_transfer_failed"]
