# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import contextlib

import pytest

from vllm.config import CacheConfig, KVTransferConfig, ParallelConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

pytestmark = pytest.mark.cpu_test


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
    kv_connector_extra_config: dict | None = None,
    enable_sleep_mode: bool = False,
    enable_cumem_allocator: bool = False,
) -> VllmConfig:
    """Build a VllmConfig that exercises _verify_kv_transfer_compat without
    requiring a real model (avoids HF downloads in CI)."""
    from types import SimpleNamespace

    kv_transfer_config = (
        KVTransferConfig(
            kv_connector=kv_connector,
            kv_role="kv_both",
            kv_connector_extra_config=kv_connector_extra_config or {},
        )
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


@pytest.mark.parametrize(
    ("kv_connector", "enable_sleep_mode", "enable_cumem_allocator"),
    [("MooncakeConnector", True, False), ("NixlConnector", False, True)],
    ids=["sleep_mode", "cumem_allocator"],
)
def test_kv_connector_allows_expandable_segments_with_cumem_allocator(
    monkeypatch, kv_connector, enable_sleep_mode, enable_cumem_allocator
):
    """KV allocations in CuMemAllocator's pool, which sleep mode also enables,
    auto-disable expandable_segments (see #40812)."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _build_config(
        kv_connector=kv_connector,
        enable_sleep_mode=enable_sleep_mode,
        enable_cumem_allocator=enable_cumem_allocator,
    )


_MOONCAKE = {"kv_connector": "MooncakeConnector", "kv_role": "kv_both"}
_MOONCAKE_TCP = {**_MOONCAKE, "kv_connector_extra_config": {"mooncake_protocol": "tcp"}}
_OFFLOADING = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both"}
_NIXL = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}


@pytest.mark.parametrize("enable_sleep_mode", [False, True], ids=["awake", "sleep"])
@pytest.mark.parametrize(
    ("kv_connector", "kv_connector_extra_config", "supported"),
    [
        ("NixlConnector", {}, False),
        ("MooncakeConnector", {"mooncake_protocol": "tcp"}, False),
        ("MooncakeConnector", {"mooncake_protocol": "rdma"}, True),
        ("MooncakeConnector", {}, True),
        ("MooncakeStoreConnector", {}, False),
        ("MoRIIOConnector", {}, False),
        ("OffloadingConnector", {}, True),
        ("ExampleConnector", {}, True),
        ("MultiConnector", {"connectors": [_MOONCAKE, _OFFLOADING]}, True),
        ("MultiConnector", {"connectors": [_MOONCAKE, _NIXL]}, False),
        ("MultiConnector", {"connectors": [_OFFLOADING, _MOONCAKE_TCP]}, False),
    ],
    ids=[
        "nixl",
        "mooncake-tcp",
        "mooncake-rdma",
        "mooncake-default",
        "mooncake-store",
        "moriio",
        "offloading",
        "example",
        "multi-mooncake-offloading",
        "multi-mooncake-nixl",
        "multi-offloading-mooncake-tcp",
    ],
)
def test_sleep_mode_requires_kv_connector_support(
    kv_connector, kv_connector_extra_config, supported, enable_sleep_mode
):
    """Sleep mode refuses a connector that cannot follow the KV cache remap."""
    refused = enable_sleep_mode and not supported
    with (
        pytest.raises(ValueError, match=f"^{kv_connector} does not support sleep mode")
        if refused
        else contextlib.nullcontext()
    ):
        _build_config(
            kv_connector=kv_connector,
            kv_connector_extra_config=kv_connector_extra_config,
            enable_sleep_mode=enable_sleep_mode,
        )


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
