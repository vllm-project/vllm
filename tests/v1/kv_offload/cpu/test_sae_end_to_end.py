# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test: spec -> manager -> policy -> counters."""

import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def _key(i: int) -> OffloadKey:
    return make_offload_key(str(i).encode(), 0)


def _make_vllm_config() -> VllmConfig:
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=10000,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 1024 * 1024,
            "eviction_policy": "sae",
            "sae_decay_interval": 100,
        },
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_kv_cache_config() -> KVCacheConfig:
    num_blocks = 16
    num_kv_heads = 1
    head_size = 1
    dtype = torch.float32
    page_size = 2 * num_kv_heads * head_size * torch.finfo(dtype).bits // 8
    kv_tensor = KVCacheTensor(
        size=num_blocks * page_size, shared_by=["layer"], block_stride=0
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[kv_tensor],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    dtype=dtype,
                ),
            )
        ],
    )


def test_sae_end_to_end_smoke():
    spec = CPUOffloadingSpec(_make_vllm_config(), _make_kv_cache_config())
    assert spec.eviction_policy == "sae"

    mgr = spec.get_manager()
    assert isinstance(mgr._policy, SAECachePolicy)
    assert mgr._policy._decay_interval == 100

    req = ReqContext(req_id="smoke")
    assert mgr.lookup(_key(1), req) == LookupResult.MISS

    stats = mgr.get_stats()
    assert stats is not None
    data = stats.data["data"]
    assert data[CPUOffloadingMetrics.CPU_BLOCK_LOOKUP][("sae",)] == 1
    assert data[CPUOffloadingMetrics.CPU_BLOCK_MISS][("sae",)] == 1
    assert data[CPUOffloadingMetrics.CPU_BLOCK_HIT][("sae",)] == 0
    assert data[CPUOffloadingMetrics.BLOCK_EVICTION][("sae",)] == 0
