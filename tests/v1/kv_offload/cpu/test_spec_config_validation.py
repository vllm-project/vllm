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
    spec = CPUOffloadingSpec(_make_offloading_config({"eviction_policy": "bogus"}))
    with pytest.raises(ValueError, match="bogus"):
        spec.get_manager()


def test_get_manager_returns_sae_policy_when_selected():
    spec = CPUOffloadingSpec(_make_offloading_config({"eviction_policy": "sae"}))
    mgr = spec.get_manager()
    assert isinstance(mgr._policy, SAECachePolicy)


def test_default_policy_still_lru_when_not_specified():
    spec = CPUOffloadingSpec(_make_offloading_config({}))
    assert spec.eviction_policy == "lru"
