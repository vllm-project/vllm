# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
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
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def _make_vllm_config(extra_config: dict) -> VllmConfig:
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
            **extra_config,
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


def test_unknown_eviction_policy_raises():
    with pytest.raises(ValueError, match="eviction_policy"):
        CPUOffloadingSpec(
            _make_vllm_config({"eviction_policy": "bogus"}),
            _make_kv_cache_config(),
        )


def test_sae_key_under_non_sae_policy_raises():
    with pytest.raises(ValueError, match="sae_decay_interval"):
        CPUOffloadingSpec(
            _make_vllm_config(
                {
                    "eviction_policy": "lru",
                    "sae_decay_interval": 500,
                }
            ),
            _make_kv_cache_config(),
        )


def test_out_of_range_decay_factor_raises():
    with pytest.raises(ValueError, match="sae_decay_factor"):
        CPUOffloadingSpec(
            _make_vllm_config(
                {
                    "eviction_policy": "sae",
                    "sae_decay_factor": 1.5,
                }
            ),
            _make_kv_cache_config(),
        )


def test_out_of_range_decay_interval_raises():
    with pytest.raises(ValueError, match="sae_decay_interval"):
        CPUOffloadingSpec(
            _make_vllm_config(
                {
                    "eviction_policy": "sae",
                    "sae_decay_interval": 0,
                }
            ),
            _make_kv_cache_config(),
        )


def test_out_of_range_ghost_norm_raises():
    with pytest.raises(ValueError, match="sae_ghost_norm"):
        CPUOffloadingSpec(
            _make_vllm_config(
                {
                    "eviction_policy": "sae",
                    "sae_ghost_norm": 0.0,
                }
            ),
            _make_kv_cache_config(),
        )


def test_valid_sae_config_stores_kwargs():
    spec = CPUOffloadingSpec(
        _make_vllm_config(
            {
                "eviction_policy": "sae",
                "sae_decay_interval": 250,
            }
        ),
        _make_kv_cache_config(),
    )
    assert spec.eviction_policy == "sae"
    assert spec._sae_policy_kwargs["decay_interval"] == 250


def test_get_manager_returns_sae_policy_when_selected():
    spec = CPUOffloadingSpec(
        _make_vllm_config({"eviction_policy": "sae"}),
        _make_kv_cache_config(),
    )
    mgr = spec.get_manager()
    assert isinstance(mgr._policy, SAECachePolicy)


def test_default_policy_still_lru_when_not_specified():
    spec = CPUOffloadingSpec(
        _make_vllm_config({}),
        _make_kv_cache_config(),
    )
    assert spec.eviction_policy == "lru"
    assert spec._sae_policy_kwargs == {}


def test_build_metric_definitions_includes_four_labelled_counters():
    definitions = CPUOffloadingSpec.build_metric_definitions({})
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    for name in (
        CPUOffloadingMetrics.CPU_BLOCK_LOOKUP,
        CPUOffloadingMetrics.CPU_BLOCK_HIT,
        CPUOffloadingMetrics.CPU_BLOCK_MISS,
        CPUOffloadingMetrics.BLOCK_EVICTION,
    ):
        assert name in definitions
        assert definitions[name].labelnames == ("policy",)
