# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.lmcache_mp_utils import (
    validate_lmcache_mp_hma_block_sizes,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


def _make_vllm_config(**kwargs) -> VllmConfig:
    """Create config tests on CPU without relying on wheel platform metadata."""
    return VllmConfig(device_config=DeviceConfig("cpu"), **kwargs)


@pytest.mark.parametrize(
    "kv_offloading_backend,kv_offloading_size,tp,pp,expected_backend,expected_bytes",
    [
        ("native", 4.0, 1, 1, "OffloadingConnector", 4.0 * (1 << 30)),
        # bytes per rank: 8.0 GiB / (2 * 2) = 2.0 GiB
        ("native", 8.0, 2, 2, "OffloadingConnector", 8.0 * (1 << 30)),
        ("lmcache", 4.0, 1, 1, "LMCacheConnectorV1", 4.0),
        ("lmcache", 8.0, 2, 2, "LMCacheConnectorV1", 2.0),
        ("lmcache_mp", 4.0, 1, 1, "LMCacheMPConnector", 4.0),
        # Local MP supports arbitrary TP; the total size is split per worker.
        ("lmcache_mp", 8.0, 4, 1, "LMCacheMPConnector", 2.0),
        # When kv_offloading_size is None, offloading is disabled (backend is ignored)
        ("native", None, 1, 1, None, None),
    ],
)
def test_kv_connector(
    kv_offloading_backend, kv_offloading_size, tp, pp, expected_backend, expected_bytes
):
    kv_transfer_config = (
        KVTransferConfig(kv_connector_extra_config={"existing_key": "existing_value"})
        if expected_backend is not None
        else None
    )

    vllm_config = _make_vllm_config(
        cache_config=CacheConfig(
            kv_offloading_backend=kv_offloading_backend,
            kv_offloading_size=kv_offloading_size,
        ),
        kv_transfer_config=kv_transfer_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
    )

    # No KV transfer config expected
    if expected_backend is None:
        assert vllm_config.kv_transfer_config is expected_backend
        return

    kv_transfer_config = vllm_config.kv_transfer_config
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config

    assert kv_transfer_config.kv_connector == expected_backend
    assert kv_transfer_config.kv_role == "kv_both"

    if kv_offloading_backend == "native":
        assert kv_connector_extra_config["cpu_bytes_to_use"] == expected_bytes
        # Existing config should be preserved
        assert kv_connector_extra_config["existing_key"] == "existing_value"
    elif kv_offloading_backend == "lmcache":
        assert kv_connector_extra_config["lmcache.local_cpu"] is True
        assert kv_connector_extra_config["lmcache.max_local_cpu_size"] == expected_bytes
        # Preserve the existing LMCache V1 shortcut behavior.
        assert "existing_key" not in kv_connector_extra_config
    elif kv_offloading_backend == "lmcache_mp":
        assert kv_connector_extra_config["lmcache.mp.deployment"] == "local"
        assert (
            kv_connector_extra_config["lmcache.mp.local_cpu_size_gb"] == expected_bytes
        )
        # Unrelated connector settings remain available to the MP adapter.
        assert kv_connector_extra_config["existing_key"] == "existing_value"


def test_lmcache_offloading_rejects_external_deployment():
    """The convenience offloading flag has unambiguous local semantics."""
    with pytest.raises(ValueError, match="selects the in-process LMCache"):
        _make_vllm_config(
            cache_config=CacheConfig(
                kv_offloading_backend="lmcache_mp",
                kv_offloading_size=4.0,
            ),
            kv_transfer_config=KVTransferConfig(
                kv_role="kv_both",
                kv_connector_extra_config={"lmcache.mp.deployment": "external"},
            ),
        )


def test_lmcache_local_offloading_enables_hma_by_default(monkeypatch):
    """The new local MP shortcut is the only connector that defaults HMA on."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)
    config = _make_vllm_config(
        cache_config=CacheConfig(
            kv_offloading_backend="lmcache_mp",
            kv_offloading_size=4.0,
        )
    )

    assert config.kv_transfer_config.kv_connector == "LMCacheMPConnector"
    assert config.scheduler_config.disable_hybrid_kv_cache_manager is False


def test_lmcache_local_respects_explicit_hma_disable(monkeypatch):
    """An explicit user choice still overrides the local MP default."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)
    config = _make_vllm_config(
        cache_config=CacheConfig(
            kv_offloading_backend="lmcache_mp",
            kv_offloading_size=4.0,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=True,
        ),
    )

    assert config.scheduler_config.disable_hybrid_kv_cache_manager is True


def test_external_connector_preserves_legacy_hma_default(monkeypatch):
    """External MP remains HMA-off by default like every existing connector."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)
    config = _make_vllm_config(
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.deployment": "external"},
        ),
    )

    assert config.scheduler_config.disable_hybrid_kv_cache_manager is True


def test_external_connector_allows_explicit_hma_enable(monkeypatch):
    """External MP may opt into HMA; the factory validates its capability."""
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "support_hybrid_kv_cache", lambda: True)
    config = _make_vllm_config(
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            disable_hybrid_kv_cache_manager=False,
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.deployment": "external"},
        ),
    )

    assert config.scheduler_config.disable_hybrid_kv_cache_manager is False


def test_empty_kv_transfer_config_preserves_legacy_hma_behavior():
    """A config without a connector must not fail capability resolution."""
    config = _make_vllm_config(kv_transfer_config=KVTransferConfig())

    assert config.scheduler_config.disable_hybrid_kv_cache_manager is True


def test_lmcache_local_offloading_supports_single_node_multiprocess():
    """Each local MP worker owns one embedded cache shard."""
    config = _make_vllm_config(
        cache_config=CacheConfig(
            kv_offloading_backend="lmcache_mp",
            kv_offloading_size=8.0,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=2),
    )

    assert config.parallel_config.distributed_executor_backend == "mp"
    assert (
        config.kv_transfer_config.kv_connector_extra_config[
            "lmcache.mp.local_cpu_size_gb"
        ]
        == 4.0
    )


def test_lmcache_mp_deployment_routes_to_independent_facades():
    """Local and external modes share a user name but resolve separate classes."""
    external = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
    )
    local = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"lmcache.mp.deployment": "local"},
    )

    assert (
        KVConnectorFactory._get_effective_connector_name(external)
        == "LMCacheMPConnector"
    )
    assert (
        KVConnectorFactory._get_effective_connector_name(local)
        == "LMCacheMPLocalConnector"
    )


def test_lmcache_local_rejects_nonlocal_executor(monkeypatch):
    """Local mode must not silently place worker shards on other machines."""

    class LocalLMCacheMPConnector:
        pass

    config = _make_vllm_config(
        cache_config=CacheConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.deployment": "local"},
        ),
    )
    config.parallel_config.distributed_executor_backend = "ray"
    monkeypatch.setattr(
        KVConnectorFactory,
        "_get_connector_class_with_compat",
        lambda _config: (LocalLMCacheMPConnector, False),
    )

    with pytest.raises(ValueError, match="requires a single-machine"):
        KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER)


def test_lmcache_local_rejects_multiple_nodes():
    with pytest.raises(ValueError, match="requires a single-machine"):
        _make_vllm_config(
            cache_config=CacheConfig(
                kv_offloading_backend="lmcache_mp",
                kv_offloading_size=8.0,
            ),
            parallel_config=ParallelConfig(
                tensor_parallel_size=2,
                distributed_executor_backend="mp",
                nnodes=2,
            ),
        )


@pytest.mark.parametrize(
    "parallel_kwargs,error_fragment",
    [
        ({"pipeline_parallel_size": 2}, "PP=2"),
        ({"data_parallel_size": 2}, "DP=2"),
    ],
)
def test_lmcache_local_rejects_pipeline_and_data_parallelism(
    parallel_kwargs, error_fragment
):
    """Local worker-owned shards currently implement TP geometry only."""
    with pytest.raises(ValueError, match=error_fragment):
        _make_vllm_config(
            cache_config=CacheConfig(
                kv_offloading_backend="lmcache_mp",
                kv_offloading_size=8.0,
            ),
            parallel_config=ParallelConfig(**parallel_kwargs),
        )


def test_lmcache_external_deployment_keeps_its_existing_topology_path():
    """The local-only guard must not alter an external MP configuration."""
    config = _make_vllm_config(
        cache_config=CacheConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.deployment": "external"},
        ),
        parallel_config=ParallelConfig(pipeline_parallel_size=2),
    )

    assert config.kv_transfer_config.kv_connector == "LMCacheMPConnector"


def test_lmcache_mp_rejects_auto_deployment():
    """Deployment selection is explicit; there is no topology auto mode."""
    with pytest.raises(ValueError, match="must be 'local' or 'external'"):
        _make_vllm_config(
            cache_config=CacheConfig(),
            kv_transfer_config=KVTransferConfig(
                kv_connector="LMCacheMPConnector",
                kv_role="kv_both",
                kv_connector_extra_config={"lmcache.mp.deployment": "auto"},
            ),
        )


def test_kv_offloading_size_only_uses_native_default():
    """Test that setting only kv_offloading_size enables native offloading."""
    vllm_config = _make_vllm_config(
        cache_config=CacheConfig(
            kv_offloading_size=4.0,
            # kv_offloading_backend not set, should default to "native"
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.kv_connector == "OffloadingConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_connector_extra_config["cpu_bytes_to_use"] == 4.0 * (1 << 30)


def test_lmcache_mp_hma_accepts_uniform_group_block_sizes():
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full.0"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["swa.0"],
                SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                    sliding_window=128,
                ),
            ),
        ],
    )

    validate_lmcache_mp_hma_block_sizes(kv_cache_config)


def test_lmcache_mp_hma_rejects_mixed_group_block_sizes():
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full.0"],
                FullAttentionSpec(
                    block_size=32,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["swa.0"],
                SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                    sliding_window=128,
                ),
            ),
        ],
    )

    with pytest.raises(ValueError, match=r"same block size.*\[16, 32\]"):
        validate_lmcache_mp_hma_block_sizes(kv_cache_config)
