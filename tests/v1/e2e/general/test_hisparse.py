# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import prometheus_client
import pytest

from tests.conftest import VllmRunner
from vllm.config import AttentionConfig, HiSparseConfig
from vllm.platforms import current_platform

MODEL = "deepseek-ai/DeepSeek-V3.2"

HF_OVERRIDES = {
    "num_hidden_layers": 8,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "index_topk": 128,
}


def _num_hisparse_spills(runner: VllmRunner) -> int:
    client = runner.llm.llm_engine.engine_core
    coordinator = client.engine_core.scheduler.kv_cache_manager.hisparse_coordinator
    return coordinator.next_spill_id


def _offload_load_bytes() -> float:
    return sum(
        sample.value
        for metric in prometheus_client.REGISTRY.collect()
        if metric.name == "vllm:kv_offload_load_bytes"
        for sample in metric.samples
        if sample.name == "vllm:kv_offload_load_bytes_total"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
@pytest.mark.parametrize("kv_offloading_size", [None, 1], ids=["standalone", "offload"])
def test_hisparse_spill_and_prefix_restore(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
    kv_offloading_size: int | None,
):
    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 9:
        pytest.skip("Sparse MLA requires Hopper or newer")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")

    target = [1000 + i % 64 for i in range(257)]
    pressure = [
        [2000 + request_idx * 128 + i % 64 for i in range(257)]
        for request_idx in range(4)
    ]

    with vllm_runner(
        MODEL,
        load_format="dummy",
        hf_overrides=HF_OVERRIDES,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(
                host_pool_gib=1,
                device_buffer_size=128,
            )
        ),
        block_size=64,
        max_model_len=320,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        num_gpu_blocks_override=56,
        kv_offloading_size=kv_offloading_size,
        disable_log_stats=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        compilation_config={"cudagraph_capture_sizes": [1, 4]},
    ) as runner:
        expected = runner.generate_greedy([target], max_tokens=8)
        runner.generate_greedy(pressure, max_tokens=8)

        assert _num_hisparse_spills(runner) > 0
        load_bytes = _offload_load_bytes()
        actual = runner.generate_greedy([target], max_tokens=8)
        runner.generate_greedy([[42]], max_tokens=1)

        assert actual == expected
        if kv_offloading_size is not None:
            assert _offload_load_bytes() > load_bytes
