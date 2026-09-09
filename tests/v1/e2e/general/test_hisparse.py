# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import prometheus_client
import pytest
import torch

from tests.conftest import VllmRunner
from vllm import SamplingParams
from vllm.config import AttentionConfig, HiSparseConfig, KVTransferConfig
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


def _shrink_config(config):
    overrides = HF_OVERRIDES.copy()
    if any("MTP" in architecture for architecture in config.architectures):
        overrides.pop("num_hidden_layers")
    config.update(overrides)
    return config


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


def _check_last_token_host_mirrors(model) -> int:
    torch.accelerator.synchronize()
    checked = set()
    for layer in model.modules():
        cache = getattr(layer, "hisparse_cache", None)
        if cache is None or id(cache) in checked:
            continue
        assert cache.view is not None
        source_slot = int(cache.slot_mapping[0].item())
        host_slot = int(cache.mirror_slot_mapping[0].item())
        assert source_slot >= 0 and host_slot >= 0
        block, row = divmod(source_slot, cache.view.block_size)
        expected = cache.view.cache[block, row].cpu().view(torch.uint8)
        actual = cache.runtime.host_cache[host_slot].view(torch.uint8)
        assert torch.equal(actual, expected), "Decode KV was not mirrored to host"
        checked.add(id(cache))
    return len(checked)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
def test_hisparse_fullgraph_decode_mirrors_written_rows(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
):
    """Graph-replayed decode must mirror KV before any page is spilled."""
    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 9:
        pytest.skip("Sparse MLA requires Hopper or newer")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")

    with vllm_runner(
        MODEL,
        load_format="dummy",
        hf_overrides=_shrink_config,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(
                device_buffer_size=512, eager_host_mirror=True
            )
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="HiSparseConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"host_pool_gib": 1},
        ),
        block_size=64,
        max_model_len=320,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        num_gpu_blocks_override=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        async_scheduling=True,
        compilation_config={
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 4],
        },
    ) as runner:
        runner.llm.generate(
            [{"prompt_token_ids": [1000 + i % 64 for i in range(32)]}],
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        )
        assert all(runner.llm.apply_model(_check_last_token_host_mirrors))


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="HiSparse requires NVIDIA CUDA"
)
@pytest.mark.parametrize(
    "with_offloading", [False, True], ids=["standalone", "offload"]
)
def test_hisparse_spill_and_prefix_restore(
    monkeypatch: pytest.MonkeyPatch,
    vllm_runner: type[VllmRunner],
    with_offloading: bool,
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
    hisparse_connector = {
        "kv_connector": "HiSparseConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"host_pool_gib": 1},
    }
    kv_transfer_config = KVTransferConfig(**hisparse_connector)
    if with_offloading:
        kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "connectors": [
                    hisparse_connector,
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {"cpu_bytes_to_use": 1 << 30},
                    },
                ]
            },
        )

    with vllm_runner(
        MODEL,
        load_format="dummy",
        hf_overrides=_shrink_config,
        attention_config=AttentionConfig(
            hisparse_config=HiSparseConfig(
                device_buffer_size=512,
            )
        ),
        kv_transfer_config=kv_transfer_config,
        block_size=64,
        max_model_len=320,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        num_gpu_blocks_override=128,
        disable_log_stats=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        compilation_config={"cudagraph_capture_sizes": [1, 4]},
    ) as runner:
        expected = runner.generate_greedy([target], max_tokens=8)
        runner.generate_greedy(pressure, max_tokens=8)

        assert _num_hisparse_spills(runner) > 0
        load_bytes = _offload_load_bytes()
        actual = runner.generate_greedy([target], max_tokens=8)
        runner.generate_greedy([[42]], max_tokens=1)

        assert actual == expected
        if with_offloading:
            assert _offload_load_bytes() > load_bytes
