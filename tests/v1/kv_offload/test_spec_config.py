# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for tunables parsed by the shared OffloadingSpec base class."""

from collections.abc import Mapping
from typing import Any

import pytest

from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

# Both concrete specs must pick these up: TieringOffloadingSpec derives from
# CPUOffloadingSpec, which derives from OffloadingSpec.
SPEC_CLASSES = [CPUOffloadingSpec, TieringOffloadingSpec]


def make_config(extra_config: Mapping[str, Any]) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(tokens_per_block=16, layer_names=("layer0",)),),
        worker_kv_bytes_per_block=1024,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": 1 << 20, **extra_config},
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
            is_parallelism_agnostic=True,
        ),
    )


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
def test_hit_pending_deadline_default(spec_cls):
    """The default clears the P2P load+abort-ack ceiling (30s + 10s)."""
    assert spec_cls(make_config({})).hit_pending_deadline_s == 60.0


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
def test_hit_pending_deadline_override(spec_cls):
    spec = spec_cls(make_config({"hit_pending_deadline_s": 12.5}))
    assert spec.hit_pending_deadline_s == 12.5


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
def test_hit_pending_deadline_zero_allowed(spec_cls):
    """0 is the documented opt-out, not an error."""
    assert (
        spec_cls(make_config({"hit_pending_deadline_s": 0})).hit_pending_deadline_s == 0
    )


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
def test_hit_pending_deadline_rejects_negative(spec_cls):
    with pytest.raises(ValueError, match="hit_pending_deadline_s"):
        spec_cls(make_config({"hit_pending_deadline_s": -1}))


@pytest.mark.parametrize("spec_cls", SPEC_CLASSES)
def test_hit_pending_deadline_rejects_nan(spec_cls):
    """NaN must not slip past validation.

    `nan < 0` is False, so a bare `< 0` guard admits it; every later
    `now - start < nan` is then also False, expiring the request on its
    second deferred pass instead of honouring any deadline.
    """
    with pytest.raises(ValueError, match="hit_pending_deadline_s"):
        spec_cls(make_config({"hit_pending_deadline_s": float("nan")}))
