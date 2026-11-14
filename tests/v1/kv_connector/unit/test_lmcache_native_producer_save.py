# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer saves must survive requests that have no disagg handoff spec."""

import importlib
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

_ADAPTER = (
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter"
)
_PKG = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"


def _ensure_module(name: str, created: list[str]) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
        created.append(name)
        if "." in name:
            parent_name, child = name.rsplit(".", 1)
            setattr(_ensure_module(parent_name, created), child, module)
    return module


def _stub_lmcache(created: list[str]) -> None:
    """Satisfy the adapter's import-time LMCache dependencies."""

    def identity(fn):
        return fn

    utils = _ensure_module("lmcache.utils", created)
    utils._lmcache_nvtx_annotate = identity  # type: ignore[attr-defined]
    utils.init_logger = logging.getLogger  # type: ignore[attr-defined]

    config = _ensure_module("lmcache.config", created)
    config.LMCacheEngineMetadata = MagicMock  # type: ignore[attr-defined]

    logging_mod = _ensure_module("lmcache.logging", created)
    logging_mod.init_logger = logging.getLogger  # type: ignore[attr-defined]

    observability = _ensure_module("lmcache.observability", created)
    observability.LMCStatsMonitor = MagicMock  # type: ignore[attr-defined]

    cache_engine = _ensure_module("lmcache.v1.cache_engine", created)
    cache_engine.LMCacheEngine = MagicMock  # type: ignore[attr-defined]
    cache_engine.LMCacheEngineBuilder = MagicMock  # type: ignore[attr-defined]

    blend = _ensure_module("lmcache.v1.compute.blend", created)
    blend.LMCBlenderBuilder = MagicMock  # type: ignore[attr-defined]

    v1_config = _ensure_module("lmcache.v1.config", created)
    v1_config.LMCacheEngineConfig = MagicMock  # type: ignore[attr-defined]
    v1_config._validate_and_set_config_value = MagicMock()  # type: ignore[attr-defined]

    gpu = _ensure_module("lmcache.v1.gpu_connector", created)
    gpu.VLLMBufferLayerwiseGPUConnector = MagicMock  # type: ignore[attr-defined]
    gpu.VLLMPagedMemGPUConnectorV2 = MagicMock  # type: ignore[attr-defined]
    gpu.VLLMPagedMemLayerwiseGPUConnector = MagicMock  # type: ignore[attr-defined]

    api = _ensure_module("lmcache.v1.internal_api_server.api_server", created)
    api.InternalAPIServer = MagicMock  # type: ignore[attr-defined]

    lookup = _ensure_module("lmcache.v1.lookup_client", created)
    lookup.LookupClientFactory = MagicMock  # type: ignore[attr-defined]

    async_lookup = _ensure_module(
        "lmcache.v1.lookup_client.lmcache_async_lookup_client", created
    )
    async_lookup.LMCacheAsyncLookupServer = MagicMock  # type: ignore[attr-defined]

    zmq_server = _ensure_module("lmcache.v1.offload_server.zmq_server", created)
    zmq_server.ZMQOffloadServer = MagicMock  # type: ignore[attr-defined]

    runtime = _ensure_module("lmcache.v1.plugin.runtime_plugin_launcher", created)
    runtime.RuntimePluginLauncher = MagicMock  # type: ignore[attr-defined]


def _load_adapter() -> tuple[types.ModuleType, list[str]]:
    try:
        return importlib.import_module(_ADAPTER), []
    except ImportError:
        created: list[str] = []
        sys.modules.pop(_ADAPTER, None)
        sys.modules.pop(_PKG, None)
        _stub_lmcache(created)
        package = types.ModuleType(_PKG)
        package.__path__ = [  # type: ignore[attr-defined]
            "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration"
        ]
        package.__package__ = _PKG
        sys.modules[_PKG] = package
        created.append(_PKG)
        return importlib.import_module(_ADAPTER), created


@pytest.fixture(scope="module")
def adapter():
    module, created = _load_adapter()
    yield module
    for name in created:
        sys.modules.pop(name, None)


@pytest.fixture
def cpu_tensors(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)


def _request(adapter, *, can_save: bool, disagg_spec, skip_leading_tokens: int):
    token_ids = list(range(8))
    return adapter.ReqMeta(
        req_id="req-1",
        token_ids=token_ids,
        slot_mapping=torch.arange(len(token_ids)),
        is_last_prefill=True,
        save_spec=adapter.SaveSpec(skip_leading_tokens, can_save),
        disagg_spec=disagg_spec,
    )


def _wait_for_save(adapter, request):
    impl = adapter.LMCacheConnectorV1Impl.__new__(adapter.LMCacheConnectorV1Impl)
    impl.kv_role = "kv_producer"
    impl.use_layerwise = False
    impl._lmcache_chunk_size = 4
    impl.kv_caches = {"layer0": torch.zeros(1)}
    impl.lmcache_engine = MagicMock()
    impl.layerwise_storers = []
    parent = MagicMock()
    metadata = adapter.LMCacheConnectorMetadata()
    metadata.add_request(request)
    parent._get_connector_metadata.return_value = metadata
    impl._parent = parent
    impl.wait_for_save()
    return impl.lmcache_engine


def test_producer_without_disagg_spec_stores_locally(adapter, cpu_tensors):
    """An ordinary producer request is stored locally and does not abort."""
    engine = _wait_for_save(
        adapter,
        _request(
            adapter,
            can_save=True,
            disagg_spec=None,
            skip_leading_tokens=0,
        ),
    )

    engine.store.assert_called_once()
    assert engine.store.call_args.kwargs["transfer_spec"] is None
    assert engine.store.call_args.kwargs["offset"] == 0


def test_producer_without_disagg_spec_skips_when_save_disabled(adapter, cpu_tensors):
    """A producer request with no handoff spec still honors can_save=False."""
    engine = _wait_for_save(
        adapter,
        _request(
            adapter,
            can_save=False,
            disagg_spec=None,
            skip_leading_tokens=0,
        ),
    )

    engine.store.assert_not_called()


def test_producer_with_disagg_spec_caps_already_transferred_tokens(
    adapter, cpu_tensors
):
    """A real handoff still limits the save to tokens not yet transferred."""
    spec = adapter.DisaggSpec(
        req_id="remote-1",
        receiver_id="host1",
        receiver_host="host",
        receiver_init_port=1,
        receiver_alloc_port=2,
        num_transferred_tokens=5,
    )
    engine = _wait_for_save(
        adapter,
        _request(adapter, can_save=True, disagg_spec=spec, skip_leading_tokens=6),
    )

    engine.store.assert_called_once()
    assert engine.store.call_args.kwargs["transfer_spec"] is spec
    # min(6, 5) aligned down to chunk size 4.
    assert engine.store.call_args.kwargs["offset"] == 4
