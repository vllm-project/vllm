# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector-level behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.utils import create_vllm_config
from tests.v1.simple_kv_offload.test_scheduler import (
    _BYTES_PER_BLOCK,
    BLOCK_SIZE,
    DTYPE,
    HEAD_SIZE,
    NUM_KV_HEADS,
    _make_lookahead_config,
)
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.engine import core as engine_core_module
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)


def _make_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    fa_layers = ["layer0"]
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    groups = [KVCacheGroupSpec(fa_layers, spec)]
    size = _BYTES_PER_BLOCK * num_blocks
    tensors = [
        KVCacheTensor(
            size=size,
            layers=fa_layers,
            layer_stride=size,
            block_stride=_BYTES_PER_BLOCK,
        )
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def _make_connector(
    extra_config: dict | None = None,
    num_gpu_blocks: int = 16,
    cpu_bytes: int = _BYTES_PER_BLOCK * 8,
) -> SimpleCPUOffloadConnector:
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
    config_dict = {
        "cpu_bytes_to_use": cpu_bytes,
    }
    if extra_config:
        config_dict.update(extra_config)
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config=config_dict,
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=kv_cache_config,
    )
    return connector


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_lookahead", [1, 5])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_multi_cpu_coordinator_uses_parent_hash_protocol(
    num_lookahead, nested, supported
):
    config = _make_lookahead_config(num_lookahead)
    children = [
        {
            "kv_connector": "SimpleCPUOffloadConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"num_cpu_blocks": 8},
        },
        {
            "kv_connector": "SimpleCPUOffloadConnector"
            if supported
            else "ExampleConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"num_cpu_blocks": 8},
        },
    ]
    if nested:
        children[0] = {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"connectors": [children[0]]},
        }
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"connectors": children},
    )
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    connector = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER, _make_kv_cache_config()
    )
    child = connector.sub_connectors[0]
    if nested:
        child = child.sub_connectors[0]
    expected = num_lookahead if supported else 0
    assert config.num_hash_lookahead_tokens == expected
    assert not config.speculative_config.disable_lookahead_block_hashes
    coord = child.scheduler_manager.cpu_coordinator
    assert coord.block_pool.num_hash_lookahead_tokens == expected
    assert coord.drop_eagle_group_ids == (set() if supported else {0})


def test_disk_mode_rejects_non_positive_capacity() -> None:
    """Disk mode with disk_capacity_bytes <= 0 raises ValueError.

    Without this guard, the worker operates in disk mode but the scheduler
    classifies events as MEDIUM_CPU instead of MEDIUM_STORAGE, silently
    mislabeling events.
    """
    with pytest.raises(ValueError, match="disk_capacity_bytes > 0"):
        _make_connector(
            extra_config={
                "kv_offload_backend": "disk",
                "disk_path": "/tmp/fake",
                "disk_capacity_bytes": 0,
            }
        )


def _make_ring_kv_cache_config(num_blocks: int = 16) -> KVCacheConfig:
    """A paged full-attention group beside a per-request ring group."""
    full = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )
    ring = CircularBufferSpec(
        block_size=6, num_kv_heads=1, head_size=32, head_size_v=0, dtype=DTYPE
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=_BYTES_PER_BLOCK * num_blocks,
                layers=["attn"],
                layer_stride=_BYTES_PER_BLOCK * num_blocks,
                block_stride=_BYTES_PER_BLOCK,
            ),
            KVCacheTensor(
                size=ring.page_size_bytes * num_blocks,
                layers=["ring"],
                layer_stride=ring.page_size_bytes * num_blocks,
                block_stride=ring.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], full),
            KVCacheGroupSpec(["ring"], ring),
        ],
    )


def test_ring_scratch_group_is_accepted_and_never_offloaded() -> None:
    """The ring group's capacity is not a token granularity, it holds no
    hashed blocks, and it never participates in CPU hits."""
    vllm_config = create_vllm_config(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": _BYTES_PER_BLOCK * 8},
    )
    connector = SimpleCPUOffloadConnector(
        vllm_config=vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=_make_ring_kv_cache_config(),
    )
    scheduler = connector.scheduler_manager
    assert scheduler.fa_gidx == 0
    cpu_groups = scheduler.cpu_kv_cache_config.kv_cache_groups
    assert [g.kv_cache_spec.prefix_cacheable for g in cpu_groups] == [True, False]
    # Only the paged group takes part in CPU-side hit lookup.
    attention_groups = scheduler.cpu_coordinator.attention_groups
    assert [g.group_ids for g in attention_groups] == [[0]]


@pytest.mark.parametrize(
    "connector, prefix_caching, worker_blocks, expected_blocks",
    [
        (None, True, 37, None),
        ("NixlConnector", True, 37, None),
        ("SimpleCPUOffloadConnector", False, 37, None),
        ("SimpleCPUOffloadConnector", True, 37, 37),
        ("MultiConnector", True, 37, 37),
        ("SimpleCPUOffloadConnector", True, None, None),
    ],
)
def test_engine_queries_offload_blocks_only_when_enabled(
    monkeypatch, connector, prefix_caching, worker_blocks, expected_blocks
) -> None:
    """Only active offload queries workers, which must return a block count."""
    config = create_vllm_config(
        kv_connector=connector or "SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "connectors": [
                {"kv_connector": "SimpleCPUOffloadConnector", "kv_role": "kv_both"}
            ]
        },
    )
    if connector is None:
        config.kv_transfer_config = None
    config.cache_config.enable_prefix_caching = prefix_caching
    kv_config = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[])
    executor = MagicMock()
    executor.get_kv_cache_specs.return_value = [{}]
    executor.get_simple_cpu_offload_num_blocks.return_value = worker_blocks
    engine = object.__new__(engine_core_module.EngineCore)
    engine.model_executor = executor
    monkeypatch.setattr(
        engine_core_module, "get_kv_cache_configs", lambda *args: [kv_config]
    )

    if worker_blocks is None:
        with pytest.raises(AssertionError, match="workers must report"):
            engine._initialize_kv_caches(config)
        return
    engine._initialize_kv_caches(config)

    assert executor.get_simple_cpu_offload_num_blocks.call_count == (
        expected_blocks is not None
    )
    assert config.cache_config.simple_cpu_offload_num_blocks == expected_blocks
