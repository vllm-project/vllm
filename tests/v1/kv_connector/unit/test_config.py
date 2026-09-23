# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

from types import SimpleNamespace
from typing import cast

import pytest

from vllm.config import CacheConfig, KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

pytestmark = pytest.mark.cpu_test


def _dsv41_handoff_config(**overrides) -> VllmConfig:
    values = {
        "connector": "NixlConnector",
        "role": "kv_producer",
        "architecture": "DeepseekV41ForCausalLM",
        "use_v2": True,
        "pp": 1,
        "pcp": 1,
        "ubatching": False,
        "async_scheduling": False,
        "bounded_replay": True,
        "num_layers": 40,
        "kv_sources": (2, 8, 14, 20),
        "index_sources": (2, 8, 14, 20, 24, 28, 32, 36),
        "engram_layers": (1, 14),
        "speculative_config": None,
    }
    values.update(overrides)
    return cast(
        VllmConfig,
        SimpleNamespace(
            uses_dsv41_encoder_only_handoff=True,
            kv_transfer_config=KVTransferConfig(
                kv_connector=values["connector"],
                kv_role=values["role"],
                dsv41_encoder_only_prefill=True,
            ),
            model_config=SimpleNamespace(
                architecture=values["architecture"],
                hf_text_config=SimpleNamespace(
                    sliding_window=128,
                    num_hidden_layers=values["num_layers"],
                    kv_source_layer_ids=values["kv_sources"],
                    index_source_layer_ids=values["index_sources"],
                    engram_layer_ids=values["engram_layers"],
                ),
            ),
            use_v2_model_runner=values["use_v2"],
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=values["pp"],
                prefill_context_parallel_size=values["pcp"],
                use_ubatching=values["ubatching"],
            ),
            cache_config=SimpleNamespace(swa_bounded_replay=values["bounded_replay"]),
            scheduler_config=SimpleNamespace(
                async_scheduling=values["async_scheduling"]
            ),
            speculative_config=values["speculative_config"],
        ),
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.skip_global_cleanup
def test_dsv41_encoder_only_handoff_accepts_initial_boundary(role):
    VllmConfig._verify_dsv41_encoder_only_handoff(_dsv41_handoff_config(role=role))


@pytest.mark.skip_global_cleanup
def test_dsv41_encoder_only_handoff_separates_producer_compile_hash():
    producer = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_producer",
        dsv41_encoder_only_prefill=True,
    )
    consumer = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        dsv41_encoder_only_prefill=True,
    )

    assert producer.compute_hash() != consumer.compute_hash()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"connector": "ExampleConnector"}, "direct NixlConnector P/D"),
        ({"role": "kv_both"}, "dedicated kv_producer or kv_consumer"),
        ({"architecture": "LlamaForCausalLM"}, "DeepseekV41ForCausalLM only"),
        ({"use_v2": False}, "requires model runner V2"),
        ({"pp": 2}, "requires PP=1"),
        ({"pcp": 2}, "does not support prefill context parallelism"),
        ({"ubatching": True}, "does not support DBO or microbatching"),
        ({"async_scheduling": True}, "requires --no-async-scheduling"),
        ({"bounded_replay": False}, "requires SWA bounded replay"),
        ({"num_layers": 39}, "requires the 40-layer"),
        ({"index_sources": (2, 8, 14)}, "complete Main-KV/Indexer-K sources"),
        ({"engram_layers": (20,)}, "boundary cannot be an Engram layer"),
        (
            {"role": "kv_consumer", "speculative_config": SimpleNamespace()},
            "does not yet support speculative decoding",
        ),
    ],
)
@pytest.mark.skip_global_cleanup
def test_dsv41_encoder_only_handoff_rejects_unsupported_config(overrides, error):
    with pytest.raises(ValueError, match=error):
        VllmConfig._verify_dsv41_encoder_only_handoff(
            _dsv41_handoff_config(**overrides)
        )


class _StubLMCacheMPConnector:
    """Stand-in for LMCacheMPConnector used in config-translation tests.

    The real connector module hard-imports the optional ``lmcache`` package
    at module load time, which is not installed in the cpu_test image. This
    test only asserts on the connector *name* and the ``extra_config`` dict
    produced by ``VllmConfig``, never instantiates the connector, so a bare
    placeholder class is sufficient. Not subclassing ``SupportsHMA`` mirrors
    the real connector's HMA support (it does not support HMA either)."""


@pytest.fixture
def stub_lmcache_mp_connector(monkeypatch):
    """Replace the lazy loader so VllmConfig.__post_init__ does not import
    ``lmcache_mp_connector`` (and thus ``lmcache``) during config tests."""
    monkeypatch.setitem(
        KVConnectorFactory._registry,
        "LMCacheMPConnector",
        lambda: _StubLMCacheMPConnector,
    )


@pytest.mark.parametrize(
    "kv_offloading_backend,kv_offloading_size,tp,pp,expected_backend,expected_bytes",
    [
        ("native", 4.0, 1, 1, "OffloadingConnector", 4.0 * (1 << 30)),
        # bytes per rank: 8.0 GiB / (2 * 2) = 2.0 GiB
        ("native", 8.0, 2, 2, "OffloadingConnector", 8.0 * (1 << 30)),
        # ``lmcache`` backend now defaults to LMCacheMPConnector. The KV
        # storage capacity is owned by the standalone LMCache server, so
        # ``kv_offloading_size`` is intentionally not propagated.
        ("lmcache", 4.0, 1, 1, "LMCacheMPConnector", None),
        ("lmcache", 8.0, 2, 2, "LMCacheMPConnector", None),
        # When kv_offloading_size is None, offloading is disabled (backend is ignored)
        ("native", None, 1, 1, None, None),
    ],
)
def test_kv_connector(
    stub_lmcache_mp_connector,
    kv_offloading_backend,
    kv_offloading_size,
    tp,
    pp,
    expected_backend,
    expected_bytes,
):
    kv_transfer_config = (
        KVTransferConfig(kv_connector_extra_config={"existing_key": "existing_value"})
        if expected_backend is not None
        else None
    )

    vllm_config = VllmConfig(
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
        # MP mode does not push lmcache.local_cpu / max_local_cpu_size into
        # extra config (the LMCache server owns capacity). Pre-existing
        # extra config entries are preserved as-is.
        assert "lmcache.local_cpu" not in kv_connector_extra_config
        assert "lmcache.max_local_cpu_size" not in kv_connector_extra_config
        assert kv_connector_extra_config["existing_key"] == "existing_value"


def _build_config(
    *,
    kv_connector: str | None,
    enable_sleep_mode: bool = False,
    enable_cumem_allocator: bool = False,
) -> VllmConfig:
    """Build a VllmConfig that exercises _verify_kv_transfer_compat without
    requiring a real model (avoids HF downloads in CI)."""
    from types import SimpleNamespace

    kv_transfer_config = (
        KVTransferConfig(kv_connector=kv_connector, kv_role="kv_both")
        if kv_connector is not None
        else None
    )
    cfg = VllmConfig.__new__(VllmConfig)
    cfg.kv_transfer_config = kv_transfer_config
    cfg.model_config = SimpleNamespace(
        enable_sleep_mode=enable_sleep_mode,
        enable_cumem_allocator=(enable_cumem_allocator or enable_sleep_mode),
    )
    cfg._verify_kv_transfer_compat()
    return cfg


@pytest.mark.parametrize(
    "kv_connector", ["NixlConnector", "MooncakeConnectorV1", "SomeOOTConnector"]
)
def test_kv_connector_rejects_expandable_segments(monkeypatch, kv_connector):
    """KV connectors that pin KV cache memory (e.g. via ibv_reg_mr) are
    invalidated when expandable_segments lets the CUDA VMM allocator remap
    the underlying physical pages. We can't enumerate every connector that
    does this (especially OOT ones), so reject the combination whenever any
    connector is configured."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with pytest.raises(ValueError, match="expandable_segments"):
        _build_config(kv_connector=kv_connector)


def test_kv_connector_allows_expandable_segments_with_sleep_mode(monkeypatch):
    """Sleep mode routes KV allocations through CuMemAllocator's pool, which
    auto-disables expandable_segments (see #40812)."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _build_config(kv_connector="NixlConnector", enable_sleep_mode=True)


def test_kv_connector_allows_expandable_segments_with_cumem_allocator(
    monkeypatch,
):
    """Manual CuMem allocation must also bypass expandable_segments."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _build_config(kv_connector="NixlConnector", enable_cumem_allocator=True)


def test_kv_connector_allows_other_alloc_conf(monkeypatch):
    """Other PYTORCH_CUDA_ALLOC_CONF values must not be rejected."""
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:False"
    )
    _build_config(kv_connector="NixlConnector")


def test_no_kv_connector_ignores_expandable_segments(monkeypatch):
    """The expandable_segments check only applies when a KV connector is
    configured."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _build_config(kv_connector=None)


def test_kv_offloading_size_only_uses_native_default():
    """Test that setting only kv_offloading_size enables native offloading."""
    vllm_config = VllmConfig(
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
