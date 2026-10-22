# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV cache offloading configuration."""

import pytest

from vllm.config import CacheConfig, KVTransferConfig, ParallelConfig, VllmConfig
from vllm.config.kv_transfer import KVRole
from vllm.config.speculative import RemoteDraftConfig, SpeculativeConfig
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
    assert kv_transfer_config is not None
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
    kv_role: KVRole = "kv_both",
    remote_draft: bool | None = None,
    kv_offloading_backend: str | None = None,
) -> VllmConfig:
    """Build a VllmConfig that exercises _verify_kv_transfer_compat without
    requiring a real model (avoids HF downloads in CI)."""
    from types import SimpleNamespace

    kv_transfer_config = (
        KVTransferConfig(
            kv_connector=kv_connector,
            kv_role=kv_role,
            kv_connector_extra_config=kv_connector_extra_config or {},
        )
        if kv_connector is not None
        else None
    )
    cfg = VllmConfig.__new__(VllmConfig)
    cfg.kv_transfer_config = kv_transfer_config
    cfg.speculative_config = None
    if remote_draft is not None:
        # Use the real placement predicate without loading a draft checkpoint.
        cfg.speculative_config = SpeculativeConfig.__new__(SpeculativeConfig)
        cfg.speculative_config.remote_draft = (
            RemoteDraftConfig(endpoint="unix:///test-draft.sock")
            if remote_draft
            else None
        )
    cfg.model_config = SimpleNamespace(
        enable_sleep_mode=enable_sleep_mode,
        enable_cumem_allocator=(enable_cumem_allocator or enable_sleep_mode),
    )
    cfg.cache_config = CacheConfig(
        enable_prefix_caching=False,
        **(
            {"kv_offloading_backend": kv_offloading_backend, "kv_offloading_size": 1.0}
            if kv_offloading_backend is not None
            else {}
        ),
    )
    cfg._post_init_kv_transfer_config()
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
    _build_config(kv_connector="MooncakeConnector", enable_sleep_mode=True)


def test_kv_connector_allows_expandable_segments_with_cumem_allocator(
    monkeypatch,
):
    """Manual CuMem allocation must also bypass expandable_segments."""
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _build_config(kv_connector="NixlConnector", enable_cumem_allocator=True)


@pytest.mark.parametrize(
    ("kv_connector", "extra_config", "env"),
    [
        ("NixlConnector", {"backends": ["LIBFABRIC"]}, None),
        ("NixlPushConnector", {}, None),
        ("MooncakeConnector", {"mooncake_protocol": "tcp"}, None),
        ("MooncakeConnector", {}, "MC_USE_TENT"),
    ],
)
def test_sleep_mode_refuses_unsupported_kv_connector(
    monkeypatch, kv_connector, extra_config, env
):
    if env:
        monkeypatch.setenv(env, "1")
    with pytest.raises(ValueError, match=f"^{kv_connector} does not support sleep"):
        _build_config(
            kv_connector=kv_connector,
            kv_connector_extra_config=extra_config,
            enable_sleep_mode=True,
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


@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_both"])
def test_remote_draft_rejects_external_kv_prefixes(kv_role):
    """External hits skip target prefill even when prefix caching is disabled."""
    with pytest.raises(ValueError, match="remote draft.*KV.*consum"):
        _build_config(
            kv_connector="ExampleConnector", kv_role=kv_role, remote_draft=True
        )


@pytest.mark.parametrize("remote_draft", [None, False])
@pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_both", "kv_producer"])
def test_native_draft_keeps_kv_connector_roles(remote_draft, kv_role):
    _build_config(
        kv_connector="ExampleConnector", kv_role=kv_role, remote_draft=remote_draft
    )


@pytest.mark.parametrize("kv_connector", [None, "ExampleConnector"])
def test_remote_draft_keeps_no_connector_and_producer_only(kv_connector):
    _build_config(kv_connector=kv_connector, kv_role="kv_producer", remote_draft=True)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("remote_draft", [False, True])
@pytest.mark.parametrize("name", ["NixlConnector", "NixlPullConnector"])
def test_remote_draft_checks_producer_bidirectional_reads(
    nested, bidirectional, remote_draft, name
):
    child = {
        "kv_connector": name,
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"bidirectional_kv_xfer": bidirectional},
    }
    if nested:
        child = {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"connectors": [child]},
        }
    if remote_draft and bidirectional:
        with pytest.raises(ValueError, match="external KV hits skip"):
            _build_config(**child, remote_draft=remote_draft)
    else:
        _build_config(**child, remote_draft=remote_draft)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("consumer", [False, True])
def test_remote_draft_checks_multi_connector_child_roles(nested, consumer):
    """MultiConnector executes children using their own independent roles."""
    child: dict[str, object] = {
        "kv_connector": "ExampleConnector",
        "kv_role": "kv_consumer" if consumer else "kv_producer",
    }
    if nested:
        child = {
            "kv_connector": "MultiConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"connectors": [child]},
        }
    kwargs = dict(
        kv_connector="MultiConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"connectors": [child]},
        remote_draft=True,
    )
    if consumer:
        with pytest.raises(ValueError, match="remote draft.*KV.*consum"):
            _build_config(**kwargs)
    else:
        _build_config(**kwargs)


@pytest.mark.parametrize("backend", ["native", "lmcache"])
@pytest.mark.parametrize("remote_draft", [None, False, True])
def test_remote_draft_checks_normalized_kv_offloading(backend, remote_draft):
    kwargs = dict(
        kv_connector=None,
        kv_offloading_backend=backend,
        remote_draft=remote_draft,
    )
    if remote_draft:
        with pytest.raises(ValueError, match="remote draft.*KV.*consum"):
            _build_config(**kwargs)
    else:
        cfg = _build_config(**kwargs)
        assert cfg.kv_transfer_config.is_kv_consumer


def test_kv_offloading_size_only_uses_native_default():
    """Test that setting only kv_offloading_size enables native offloading."""
    vllm_config = VllmConfig(
        cache_config=CacheConfig(
            kv_offloading_size=4.0,
            # kv_offloading_backend not set, should default to "native"
        ),
    )

    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    kv_connector_extra_config = kv_transfer_config.kv_connector_extra_config
    assert kv_transfer_config.kv_connector == "OffloadingConnector"
    assert kv_transfer_config.kv_role == "kv_both"
    assert kv_connector_extra_config["cpu_bytes_to_use"] == 4.0 * (1 << 30)
