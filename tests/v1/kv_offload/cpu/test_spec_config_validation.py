# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest

from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.policies.sae import SAECachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def _make_offloading_config(extra_config: dict[str, Any]) -> OffloadingConfig:
    normalized_extra_config: dict[str, Any] = {
        "cpu_bytes_to_use": 1024 * 1024,
        **extra_config,
    }
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(16, ("layer",)),),
        worker_kv_bytes_per_block=8,
        enable_kv_cache_events=False,
        extra_config=normalized_extra_config,
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )


def test_unknown_eviction_policy_raises():
    with pytest.raises(ValueError, match="eviction_policy"):
        CPUOffloadingSpec(_make_offloading_config({"eviction_policy": "bogus"}))


def test_sae_key_under_non_sae_policy_raises():
    with pytest.raises(ValueError, match="sae_decay_interval"):
        CPUOffloadingSpec(
            _make_offloading_config(
                {
                    "eviction_policy": "lru",
                    "sae_decay_interval": 500,
                }
            )
        )


def test_out_of_range_decay_factor_raises():
    with pytest.raises(ValueError, match="sae_decay_factor"):
        CPUOffloadingSpec(
            _make_offloading_config(
                {
                    "eviction_policy": "sae",
                    "sae_decay_factor": 1.5,
                }
            )
        )


def test_out_of_range_decay_interval_raises():
    with pytest.raises(ValueError, match="sae_decay_interval"):
        CPUOffloadingSpec(
            _make_offloading_config(
                {
                    "eviction_policy": "sae",
                    "sae_decay_interval": 0,
                }
            )
        )


def test_out_of_range_ghost_norm_raises():
    with pytest.raises(ValueError, match="sae_ghost_norm"):
        CPUOffloadingSpec(
            _make_offloading_config(
                {
                    "eviction_policy": "sae",
                    "sae_ghost_norm": 0.0,
                }
            )
        )


def test_valid_sae_config_stores_kwargs():
    spec = CPUOffloadingSpec(
        _make_offloading_config(
            {
                "eviction_policy": "sae",
                "sae_decay_interval": 250,
            }
        )
    )
    assert spec.eviction_policy == "sae"
    assert spec._sae_policy_kwargs["decay_interval"] == 250


def test_get_manager_returns_sae_policy_when_selected():
    spec = CPUOffloadingSpec(_make_offloading_config({"eviction_policy": "sae"}))
    mgr = spec.get_manager()
    assert isinstance(mgr._policy, SAECachePolicy)


def test_default_policy_still_lru_when_not_specified():
    spec = CPUOffloadingSpec(_make_offloading_config({}))
    assert spec.eviction_policy == "lru"
    assert spec._sae_policy_kwargs == {}


def test_build_metric_definitions_includes_four_counters():
    definitions = CPUOffloadingSpec.build_metric_definitions({})
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics

    for name in (
        CPUOffloadingMetrics.CPU_BLOCK_LOOKUP,
        CPUOffloadingMetrics.CPU_BLOCK_HIT,
        CPUOffloadingMetrics.CPU_BLOCK_MISS,
        CPUOffloadingMetrics.BLOCK_EVICTION,
    ):
        assert name in definitions
        assert definitions[name].labelnames == ()
