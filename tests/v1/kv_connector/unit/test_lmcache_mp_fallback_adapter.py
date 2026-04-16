# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import enum
import importlib.util
import inspect
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]


class _DummyFuture:
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, result=None):
        self._result = result

    def query(self):
        return True

    def result(self):
        return self._result

    def to_cuda_future(self):
        return self


class _DummyEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self):
        pass

    def ipc_handle(self):
        return b"ipc"


def _ensure_pkg(name: str):
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_fallback_modules():
    for pkg in [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "lmcache",
        "lmcache.integration",
        "lmcache.integration.vllm",
        "lmcache.v1",
        "lmcache.v1.multiprocess",
    ]:
        _ensure_pkg(pkg)

    torch = types.ModuleType("torch")
    torch.Tensor = object

    class _DummyCuda:
        Event = _DummyEvent

        @staticmethod
        def current_stream():
            return object()

        @staticmethod
        def stream(_):
            return contextlib.nullcontext()

    torch.cuda = _DummyCuda()
    sys.modules["torch"] = torch

    zmq = types.ModuleType("zmq")
    zmq.Context = type("Context", (), {})
    sys.modules["zmq"] = zmq

    lmcache_utils = types.ModuleType("lmcache.utils")

    class _Logger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    lmcache_utils.init_logger = lambda *args, **kwargs: _Logger()
    lmcache_utils._lmcache_nvtx_annotate = lambda f: f
    sys.modules["lmcache.utils"] = lmcache_utils

    lmcache_iv_utils = types.ModuleType("lmcache.integration.vllm.utils")
    lmcache_iv_utils.mla_enabled = lambda *args, **kwargs: False
    sys.modules["lmcache.integration.vllm.utils"] = lmcache_iv_utils

    custom_types = types.ModuleType("lmcache.v1.multiprocess.custom_types")

    class CudaIPCWrapper:
        def __init__(self, *args, **kwargs):
            pass

    class IPCCacheEngineKey:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def no_worker_id_version(self):
            return self

    class BlockAllocationRecord:
        pass

    custom_types.CudaIPCWrapper = CudaIPCWrapper
    custom_types.IPCCacheEngineKey = IPCCacheEngineKey
    custom_types.KVCache = list
    custom_types.BlockAllocationRecord = BlockAllocationRecord
    sys.modules["lmcache.v1.multiprocess.custom_types"] = custom_types

    mq = types.ModuleType("lmcache.v1.multiprocess.mq")

    class MessageQueueClient:
        def __init__(self, *args, **kwargs):
            pass

        def submit_request(self, *args, **kwargs):
            return _DummyFuture()

    mq.MessageQueueClient = MessageQueueClient
    mq.MessagingFuture = _DummyFuture
    sys.modules["lmcache.v1.multiprocess.mq"] = mq

    protocol = types.ModuleType("lmcache.v1.multiprocess.protocol")

    class RequestType(enum.Enum):
        GET_CHUNK_SIZE = 1
        LOOKUP = 2
        RETRIEVE = 3
        STORE = 4
        END_SESSION = 5

    protocol.RequestType = RequestType
    protocol.get_response_class = lambda request_type: object
    sys.modules["lmcache.v1.multiprocess.protocol"] = protocol

    config_mod = types.ModuleType("vllm.config")
    config_mod.VllmConfig = type("VllmConfig", (), {})
    sys.modules["vllm.config"] = config_mod

    base_mod = types.ModuleType(
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    )

    class KVConnectorBase_V1:
        pass

    class KVConnectorMetadata:
        def __init__(self):
            pass

    class KVConnectorRole(enum.Enum):
        WORKER = 1
        SCHEDULER = 2

    base_mod.KVConnectorBase_V1 = KVConnectorBase_V1
    base_mod.KVConnectorMetadata = KVConnectorMetadata
    base_mod.KVConnectorRole = KVConnectorRole
    sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    ] = base_mod

    backend_mod = types.ModuleType("vllm.v1.attention.backend")
    backend_mod.AttentionMetadata = type("AttentionMetadata", (), {})
    sys.modules["vllm.v1.attention.backend"] = backend_mod

    sched_output_mod = types.ModuleType("vllm.v1.core.sched.output")
    sched_output_mod.SchedulerOutput = type("SchedulerOutput", (), {})
    sys.modules["vllm.v1.core.sched.output"] = sched_output_mod

    outputs_mod = types.ModuleType("vllm.v1.outputs")
    outputs_mod.KVConnectorOutput = type("KVConnectorOutput", (), {})
    sys.modules["vllm.v1.outputs"] = outputs_mod

    request_mod = types.ModuleType("vllm.v1.request")

    class RequestStatus(enum.Enum):
        PREEMPTED = "PREEMPTED"

    request_mod.RequestStatus = RequestStatus
    sys.modules["vllm.v1.request"] = request_mod

    utils_mod = types.ModuleType("vllm.v1.utils")

    class ConstantList(list):
        pass

    utils_mod.ConstantList = ConstantList
    sys.modules["vllm.v1.utils"] = utils_mod

    multi_process_adapter = _load_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter",
        ROOT
        / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/multi_process_adapter.py",
    )

    integration_pkg = sys.modules[
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
    ]
    integration_pkg.LMCacheMPSchedulerAdapter = (
        multi_process_adapter.LMCacheMPSchedulerAdapter
    )
    integration_pkg.LMCacheMPWorkerAdapter = (
        multi_process_adapter.LMCacheMPWorkerAdapter
    )
    integration_pkg.LoadStoreOp = multi_process_adapter.LoadStoreOp
    integration_pkg.ParallelStrategy = multi_process_adapter.ParallelStrategy

    lmcache_mp_connector = _load_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector",
        ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py",
    )

    return multi_process_adapter, lmcache_mp_connector


@pytest.fixture
def fallback_modules():
    tracked_prefixes = ("vllm", "lmcache", "torch", "zmq")
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == tracked_prefixes
        or any(name.startswith(f"{prefix}.") for prefix in tracked_prefixes)
        or name in tracked_prefixes
    }

    try:
        yield _load_fallback_modules()
    finally:
        for name in list(sys.modules):
            if name in tracked_prefixes or any(
                name.startswith(f"{prefix}.") for prefix in tracked_prefixes
            ):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)


def test_fallback_adapter_signatures_accept_cache_salt_keywords(fallback_modules):
    multi_process_adapter, lmcache_mp_connector = fallback_modules

    scheduler_init_sig = inspect.signature(
        multi_process_adapter.LMCacheMPSchedulerAdapter.__init__
    )
    worker_init_sig = inspect.signature(
        multi_process_adapter.LMCacheMPWorkerAdapter.__init__
    )

    scheduler_sig = inspect.signature(
        multi_process_adapter.LMCacheMPSchedulerAdapter.maybe_submit_lookup_request
    )
    store_sig = inspect.signature(
        multi_process_adapter.LMCacheMPWorkerAdapter.batched_submit_store_requests
    )
    retrieve_sig = inspect.signature(
        multi_process_adapter.LMCacheMPWorkerAdapter.batched_submit_retrieve_requests
    )

    assert "cache_salt" in scheduler_sig.parameters
    assert "cache_salts" in store_sig.parameters
    assert "cache_salts" in retrieve_sig.parameters
    assert "mq_timeout" in scheduler_init_sig.parameters
    assert "heartbeat_interval" in scheduler_init_sig.parameters
    assert "mq_timeout" in worker_init_sig.parameters
    assert "heartbeat_interval" in worker_init_sig.parameters


def test_fallback_connector_paths_accept_cache_salt_keywords(fallback_modules):
    multi_process_adapter, lmcache_mp_connector = fallback_modules

    multi_process_adapter.send_lmcache_request = (
        lambda *args, **kwargs: _DummyFuture(result=1)
    )

    dummy_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model="dummy"),
        cache_config=types.SimpleNamespace(block_size=1),
        parallel_config=types.SimpleNamespace(
            world_size=1,
            rank=0,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
    )

    scheduler = lmcache_mp_connector.create_scheduler_adapter(
        server_url="ipc://dummy",
        zmq_context=object(),
        vllm_config=dummy_config,
        mq_timeout=1.0,
        heartbeat_interval=1.0,
    )
    request = types.SimpleNamespace(
        request_id="req-lookup",
        all_token_ids=[1, 2, 3],
        status="RUNNING",
        cache_salt="salt",
        block_hashes=[],
    )
    tracker = types.SimpleNamespace(
        cache_salt="salt",
        increase_num_stored_blocks=lambda num_blocks: None,
    )
    connector_self = types.SimpleNamespace(
        _get_or_create_request_tracker=lambda request: tracker,
        scheduler_adapter=scheduler,
        vllm_block_size=1,
    )
    assert lmcache_mp_connector.LMCacheMPConnector.get_num_new_matched_tokens(
        connector_self, request, 0
    ) == (1, True)
    assert "req-lookup" in scheduler.lookup_futures

    worker = lmcache_mp_connector.create_worker_adapter(
        server_url="ipc://dummy",
        zmq_context=object(),
        vllm_config=dummy_config,
        mq_timeout=1.0,
        heartbeat_interval=1.0,
    )
    retrieve_meta = lmcache_mp_connector.LMCacheMPConnectorMetadata()
    retrieve_meta.add_request_metadata(
        lmcache_mp_connector.LMCacheMPRequestMetadata(
            request_id="req-retrieve",
            direction="RETRIEVE",
            op=multi_process_adapter.LoadStoreOp(
                block_ids=[1], token_ids=[1], start=0, end=1
            ),
            cache_salt="salt",
        )
    )
    retrieve_self = types.SimpleNamespace(
        worker_adapter=worker,
        _get_connector_metadata=lambda: retrieve_meta,
    )
    lmcache_mp_connector.LMCacheMPConnector.start_load_kv(retrieve_self, None)
    assert "req-retrieve" in worker.retrieve_futures

    store_meta = lmcache_mp_connector.LMCacheMPConnectorMetadata()
    store_meta.add_request_metadata(
        lmcache_mp_connector.LMCacheMPRequestMetadata(
            request_id="req-store",
            direction="STORE",
            op=multi_process_adapter.LoadStoreOp(
                block_ids=[1], token_ids=[1], start=0, end=1
            ),
            cache_salt="salt",
        )
    )
    store_self = types.SimpleNamespace(
        worker_adapter=worker,
        _get_connector_metadata=lambda: store_meta,
    )
    lmcache_mp_connector.LMCacheMPConnector.wait_for_save(store_self)
    assert "req-store" in worker.store_futures
