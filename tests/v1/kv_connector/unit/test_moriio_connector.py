# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import queue
import socket
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch
import zmq

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOConstants,
    MoRIIOMode,
    ReqMeta,
    resolve_host_ip,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    KVConnectorRole,
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_ip,
    make_zmq_path,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from .utils import create_request, create_scheduler


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _make_test_kv_cache_config() -> KVCacheConfig:
    layer_names = ["layer0", "layer1", "layer2"]
    return KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[KVCacheTensor(size=0, shared_by=layer_names)],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )


aiter_available = importlib.util.find_spec("aiter") is not None
mori_available = importlib.util.find_spec("mori") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and mori_available),
    reason="MoRIIOs are only available on ROCm with aiter package installed",
)


@pytest.fixture
def mock_parallel_groups():
    """Mock tensor/data parallel group functions for single-rank tests."""
    mock_group = MagicMock()
    mock_group.rank = 0
    mock_group.local_rank = 0
    mock_group.world_size = 1

    with (
        patch.multiple(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common",
            get_tensor_model_parallel_rank=MagicMock(return_value=0),
            get_tensor_model_parallel_world_size=MagicMock(return_value=0),
        ),
        patch.multiple(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector",
            get_tensor_model_parallel_world_size=MagicMock(return_value=0),
            get_world_group=MagicMock(return_value=mock_group),
            get_tp_group=MagicMock(return_value=mock_group),
        ),
    ):
        yield mock_group


def _setup_kv_transfer_request(
    request, remote_host="127.0.0.1", fake_port=4789, fake_transfer_id="0"
):
    """Setup KV transfer parameters for a request."""
    request.kv_transfer_params.update(
        {
            "transfer_id": fake_transfer_id,
            "remote_notify_port": fake_port,
            "remote_block_ids": None,
            "remote_host": remote_host,
            "remote_port": fake_port,
            "remote_handshake_port": fake_port,
            "remote_engine_id": "test_engine",
        }
    )
    zmq_addr = f"host:{remote_host},handshake:{fake_port},notify:{fake_port}"
    fake_uuid = uuid.uuid4().hex
    request.request_id = (
        f"___prefill_addr_{zmq_addr}___decode_addr_{zmq_addr}_{fake_uuid}"
    )
    return request


def _write_consumer_scheduler_for_finished_request(tp_size: int = 2):
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.is_producer = False
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = tp_size
    scheduler._reqs_need_recv = {}
    scheduler.unmap_request_id = MagicMock()
    return scheduler


class FakeMoRIIOWrapper:
    # A fake MoRIIOWrapper for testing purposes
    def __init__(self, *args, **kwargs):
        pass

    def set_moriio_engine(self, moriio_engine):
        pass

    def set_backend_type(self, backend_type):
        pass

    def get_agent_metadata(self):
        pass

    def register_remote_engine(self, remote_packed_engine_metadata):
        pass

    def register_local_tensor(self, tensor: torch.Tensor):
        pass

    def get_unpack_memory_metadata(self, packed_memory_metadata):
        pass

    def build_session(self, local_memory_metadata, remote_memory_metadata):
        pass

    def read_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        pass

    def write_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        pass

    def write_remote_data_single(
        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0
    ):
        pass

    def waiting_for_transfer_complete(self):
        pass

    def async_wait_reqid(self):
        pass

    def _handle_message(self, msg: bytes):
        pass

    def _handle_structured_message(self, data: dict):
        pass

    def _handle_completion_message(self, msg: str):
        pass

    def send_notify(
        self,
        req_ids,
        remote_ip,
        remote_port,
        message_type=None,
        message_fields=None,
    ):
        pass

    def pop_finished_req_ids(self):
        pass

    def pop_finished_write_req_ids(self):
        pass

    def shutdown(self):
        pass


class FakeMoRIIOConnectorWorker(MoRIIOConnectorWorker):
    # Define a fake remote engine id for testing
    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(
        self,
        vllm_config,
        engine_id,
        *args,
        hand_shake_latency: float = 1.8,
        kv_cache_layout="HND",
        kv_cache_config=None,
        **kwargs,
    ):
        super().__init__(
            vllm_config, engine_id, kv_cache_config or _make_test_kv_cache_config()
        )


def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
    role="kv_consumer",
    read_mode: bool = False,
) -> VllmConfig:
    """Initialize VllmConfig for testing."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=False,
    )
    model_config = ModelConfig(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    # These tests exercise connector setup, not real RDMA transfer (MoRI wrapper is
    # mocked), so we can use any backend without affecting test validity. Use xGMI to
    # avoid requiring RNICs in CI.
    kv_transfer_config = KVTransferConfig(
        kv_connector="MoRIIOConnector",
        kv_role=role,
        enable_permute_local_kv=enable_permute_local_kv,
        kv_connector_extra_config={
            "read_mode": read_mode,
            "backend": "xgmi",
            "trusted_remote_hosts": ["127.0.0.1"],
        },
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def test_write_mode_saves_local_block_ids():
    """Write mode records local block ids in MoRIIOConnectorMetadata.reqs_to_save."""

    # Setup Scheduler and Request
    vllm_config = create_vllm_config(role="kv_producer")
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
        do_remote_prefill=False,
    )

    # Setup KV transfer params and embed ZMQ addrs in request_id before
    # adding to scheduler so the ID is consistent everywhere.
    request = _setup_kv_transfer_request(request)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill, triggers MoRIIOConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None, "kv_connector_metadata is None"
    assert isinstance(kv_connector_metadata, MoRIIOConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_save) == 1, (
        "Unexpected number of reqs_to_save"
    )
    assert len(kv_connector_metadata.reqs_to_recv) == 0, (
        "Unexpected number of reqs_to_recv"
    )
    assert len(kv_connector_metadata.reqs_to_send) == 0, (
        "Unexpected number of reqs_to_send"
    )
    assert request_id in kv_connector_metadata.reqs_to_save, (
        "Request ID not in reqs_to_save"
    )
    req_meta = kv_connector_metadata.reqs_to_save[request_id]

    for block_id, block in zip(
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id, f"{block_id} != {block.block_id}"


def test_write_mode_with_chunked_prefill_saves_local_block_ids():
    """Write mode with chunked prefill still records correct local block ids."""
    # Setup Scheduler and Request
    MAX_NUM_BATCHED_TOKENS = 64
    NUM_TOKENS = MAX_NUM_BATCHED_TOKENS * 2 + MAX_NUM_BATCHED_TOKENS // 2

    vllm_config = create_vllm_config(
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS, role="kv_producer"
    )
    BLOCK_SIZE = vllm_config.cache_config.block_size

    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=True,
        do_remote_prefill=False,
    )

    # Setup KV transfer params and embed ZMQ addrs in request_id before
    # adding to scheduler so the ID is consistent everywhere.
    request = _setup_kv_transfer_request(request)
    request_id = request.request_id

    scheduler.add_request(request)

    # Remote Prefill with chunked prefill, triggers multiple schedules.
    expected_counts = [(0, 0, 0), (0, 0, 0), (1, 0, 0)]
    kv_connector_metadata = None
    for _, (expected_save, expected_recv, expected_send) in enumerate(expected_counts):
        scheduler_output = scheduler.schedule()
        kv_connector_metadata = scheduler_output.kv_connector_metadata

        assert len(kv_connector_metadata.reqs_to_save) == expected_save
        assert len(kv_connector_metadata.reqs_to_recv) == expected_recv
        assert len(kv_connector_metadata.reqs_to_send) == expected_send
    assert kv_connector_metadata is not None, "kv_connector_metadata is None"
    assert request_id in kv_connector_metadata.reqs_to_save, (
        "Request ID not in reqs_to_save"
    )
    req_meta = kv_connector_metadata.reqs_to_save[request_id]

    for block_id, block in zip(
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id, f"{block_id} != {block.block_id}"


def test_read_mode_loads_remote_block_ids():
    """Read mode loads remote block ids into local cache mapping."""

    # Setup Scheduler and Request
    vllm_config = create_vllm_config(role="kv_consumer", read_mode=True)
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_decode=False,
        do_remote_prefill=True,
    )

    # Setup KV transfer params and embed ZMQ addrs in request_id before
    # adding to scheduler so the ID is consistent everywhere.
    request = _setup_kv_transfer_request(request)
    request_id = request.request_id

    scheduler.add_request(request)
    block_list = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_id]

    # Set remote block ids to be fetched.
    request.kv_transfer_params["remote_block_ids"] = block_list

    # Remote Prefill, triggers MoRIIOConnectorMetadata.

    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None, "kv_connector_metadata is None"
    assert isinstance(kv_connector_metadata, MoRIIOConnectorMetadata), (
        "kv_connector_metadata is not MoRIIOConnectorMetadata"
    )
    assert len(kv_connector_metadata.reqs_to_save) == 0, (
        "Unexpected number of reqs_to_save"
    )
    assert len(kv_connector_metadata.reqs_to_recv) == 1, (
        "Unexpected number of reqs_to_recv"
    )
    assert len(kv_connector_metadata.reqs_to_send) == 0, (
        "Unexpected number of reqs_to_send"
    )
    assert request_id in kv_connector_metadata.reqs_to_recv, (
        "Request ID not in reqs_to_recv"
    )
    req_meta = kv_connector_metadata.reqs_to_recv[request_id]

    for block_id, block in zip(
        req_meta.local_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id, f"{block_id} != {block.block_id}"


@pytest.mark.parametrize(
    ("transfer_id", "extra_params", "expected_notifications"),
    [
        pytest.param(
            "xfer-7",
            {"remote_host": "127.0.0.1", "remote_notify_port": 7000},
            [
                ("xfer-7", "127.0.0.1", 7000),
                ("xfer-7", "127.0.0.1", 7001),
            ],
            id="address-available",
        ),
        pytest.param("xfer-8", {}, [], id="address-unavailable-plain-id"),
    ],
)
def test_write_mode_finished_before_alloc_releases_prefill_blocks(
    transfer_id, extra_params, expected_notifications
):
    scheduler = _write_consumer_scheduler_for_finished_request(tp_size=2)
    notifications = []
    scheduler._send_transfer_release = lambda transfer_id, host, port: (
        notifications.append((transfer_id, host, port))
    )
    request = create_request(request_id=7, do_remote_prefill=True)
    request.request_id = "plain-decode-id"
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "transfer_id": transfer_id,
    } | extra_params

    delay_free, new_params = scheduler.request_finished(request, block_ids=[])

    assert not delay_free
    assert new_params is None
    assert request.kv_transfer_params["do_remote_prefill"] is False
    assert scheduler._reqs_need_recv == {}
    assert notifications == expected_notifications


def test_send_transfer_release_sends_structured_release_message():
    scheduler = _write_consumer_scheduler_for_finished_request()
    path = make_zmq_path("tcp", "127.0.0.1", 7000)
    sock = MagicMock()
    scheduler.paths = {path: sock}

    scheduler._send_transfer_release("xfer-7", "127.0.0.1", 7000)

    payload = sock.send.call_args.args[0]
    assert msgspec.msgpack.decode(payload) == {
        "type": "release",
        "transfer_id": "xfer-7",
    }


def test_read_mode_trims_remote_blocks_not_local_blocks():
    """Read mode rejects mismatched block counts and preserves block id sides."""

    class Blocks:
        def __init__(self, block_ids):
            self._block_ids = block_ids

        def get_block_ids(self):
            return [self._block_ids]

    def build_read_metadata(remote_block_ids):
        scheduler = object.__new__(MoRIIOConnectorScheduler)
        scheduler.mode = MoRIIOMode.READ
        scheduler.tp_size = 1
        scheduler.transfer_id_to_request_id = {}
        scheduler.request_id_to_transfer_id = {}
        scheduler._pending_transfer_id_to_request_id = {}
        scheduler._transfer_ids_to_forget = set()
        scheduler._reqs_need_recv = {}
        scheduler._reqs_need_save = {}
        scheduler._reqs_need_pending_save = {}
        scheduler._reqs_need_send = {}
        scheduler.trusted_remote_hosts = ["127.0.0.1"]

        request_id = "req0"
        local_block_ids = [10, 11, 12]
        request = SimpleNamespace(
            request_id=request_id,
            num_prompt_tokens=48,
            kv_transfer_params={
                "transfer_id": "tx0",
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "remote_engine_id": "prefill:6301",
                "remote_block_ids": remote_block_ids,
                "remote_hosts": ["127.0.0.1"],
                "remote_zmq_address": "host:127.0.0.1,handshake:4789,notify:61005",
            },
        )
        scheduler.update_state_after_alloc(
            request, Blocks(local_block_ids), num_external_tokens=48
        )
        metadata = scheduler.build_connector_meta(None)
        return metadata.reqs_to_recv[request_id], local_block_ids

    with pytest.raises(AssertionError):
        build_read_metadata([1000, 1001, 1002, 1003])

    req_meta, local_block_ids = build_read_metadata([2000, 2001, 2002])
    assert req_meta.local_block_ids == local_block_ids
    assert req_meta.remote_block_ids == [2000, 2001, 2002]
    assert req_meta.remote_block_ids != local_block_ids


def test_requires_piecewise_for_cudagraph_by_default():
    for extra_config in (
        {},
        {"allow_full_cudagraph": False},
        {"allow_full_cudagraph": "0"},
    ):
        assert MoRIIOConnector.requires_piecewise_for_cudagraph(extra_config) is True


def test_requires_piecewise_for_cudagraph_rejects_allow_full():
    for extra_config in (
        {"allow_full_cudagraph": True},
        {"allow_full_cudagraph": "true"},
        {"allow_full_cudagraph": "1"},
    ):
        with pytest.raises(ValueError, match="allow_full_cudagraph"):
            MoRIIOConnector.requires_piecewise_for_cudagraph(extra_config)


def test_read_mode_start_load_kv_drains_all_ready_requests():
    class NoopLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    worker = object.__new__(MoRIIOConnectorWorker)
    worker.transfer_id_to_request_id = {}
    worker.request_id_to_transfer_id = {}
    worker._pending_unmapped_done_tids = {}
    worker.is_producer = False
    worker.mode = MoRIIOMode.READ
    worker._reqs_to_send = {}
    worker._ready_requests = queue.Queue()
    worker.load_ready_flag = {"127.0.0.1:4789": True}
    worker._remote_agents = {"127.0.0.1:4789_dp0": {"agent"}}
    worker._unmatched_write_completions = set()
    worker._handshake_lock = NoopLock()
    worker.moriio_config = SimpleNamespace(transfer_timeout=0.01)
    worker._eager_handshake_all_dp_ranks = lambda _metadata: None

    def make_meta(transfer_id):
        return ReqMeta(
            transfer_id=transfer_id,
            local_block_ids=[1],
            remote_block_ids=[2],
            remote_host="127.0.0.1",
            remote_port=4789,
            remote_handshake_port=4789,
            remote_notify_port=4789,
            remote_engine_id="127.0.0.1:4789",
            tp_size=1,
            remote_dp_size=1,
        )

    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_recv["direct"] = make_meta("transfer-direct")
    worker._ready_requests.put(("ready-1", make_meta("transfer-ready-1")))
    worker._ready_requests.put(("ready-2", make_meta("transfer-ready-2")))

    read_calls = []

    def read_blocks_for_req(req_id, meta):
        read_calls.append(req_id)

    worker._read_blocks_for_req = read_blocks_for_req

    worker.start_load_kv(metadata)

    assert read_calls == ["direct", "ready-1", "ready-2"]
    assert worker._ready_requests.empty()


def _bare_worker():
    """Build a worker without opening sockets or initializing MoRIIO."""
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = 0
    return worker


def test_pick_host_for_dp_rank_cross_node():
    worker = _bare_worker()
    meta = MagicMock(
        remote_hosts=["10.0.0.1", "10.0.0.2"],
        remote_dp_size=16,
        remote_host="10.0.0.1",
        tp_size=1,
    )

    assert worker._pick_host_for_dp_rank(meta, 0) == "10.0.0.1"
    assert worker._pick_host_for_dp_rank(meta, 7) == "10.0.0.1"
    assert worker._pick_host_for_dp_rank(meta, 8) == "10.0.0.2"
    assert worker._pick_host_for_dp_rank(meta, 15) == "10.0.0.2"


def test_pick_host_for_dp_rank_single_node_fallback():
    worker = _bare_worker()
    meta_none = MagicMock(
        remote_hosts=None, remote_dp_size=8, remote_host="10.0.0.1", tp_size=1
    )
    assert worker._pick_host_for_dp_rank(meta_none, 5) == "10.0.0.1"

    meta_single = MagicMock(
        remote_hosts=["10.0.0.1"],
        remote_dp_size=8,
        remote_host="10.0.0.1",
        tp_size=1,
    )
    assert worker._pick_host_for_dp_rank(meta_single, 5) == "10.0.0.1"


def _request_id_with_zmq(host: str = "10.0.0.1") -> str:
    zmq_addr = f"host:{host},handshake:4789,notify:61005"
    return f"___prefill_addr_{zmq_addr}___decode_addr_{zmq_addr}_{uuid.uuid4().hex}"


def test_add_new_req_populates_remote_dp_rank():
    base_kv = {
        "transfer_id": "tx-1",
        "remote_block_ids": [0, 1, 2],
        "remote_engine_id": "engine-A",
    }

    metadata = MoRIIOConnectorMetadata()
    metadata.add_new_req(
        request_id=_request_id_with_zmq(),
        local_block_ids=[10, 11],
        kv_transfer_params=dict(base_kv, remote_dp_rank=7),
        write_mode=True,
    )
    assert next(iter(metadata.reqs_to_save.values())).remote_dp_rank == 7

    metadata = MoRIIOConnectorMetadata()
    metadata.add_new_req(
        request_id=_request_id_with_zmq(),
        local_block_ids=[0],
        kv_transfer_params=dict(base_kv),
        write_mode=True,
    )
    assert next(iter(metadata.reqs_to_save.values())).remote_dp_rank == 0


def test_add_new_req_uses_remote_zmq_address_when_request_id_has_no_peer():
    metadata = MoRIIOConnectorMetadata()

    metadata.add_new_req(
        request_id="plain-request-id",
        local_block_ids=[10, 11],
        kv_transfer_params={
            "transfer_id": "tx-1",
            "remote_block_ids": [0, 1],
            "remote_engine_id": "engine-A",
            "remote_zmq_address": "host:prefill.example,handshake:1234,notify:2345",
            "remote_hosts": ["prefill-a", "prefill-b"],
        },
        trusted_remote_hosts={"prefill.example", "prefill-a", "prefill-b"},
    )

    req_meta = next(iter(metadata.reqs_to_recv.values()))
    assert req_meta.remote_host == "prefill.example"
    assert req_meta.remote_handshake_port == 1234
    assert req_meta.remote_notify_port == 2345
    assert req_meta.remote_hosts == ["prefill-a", "prefill-b"]


def test_add_new_req_rejects_untrusted_remote_zmq_address():
    metadata = MoRIIOConnectorMetadata()

    with pytest.raises(ValueError, match="trusted peer hosts"):
        metadata.add_new_req(
            request_id="plain-request-id",
            local_block_ids=[10, 11],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [0, 1],
                "remote_engine_id": "engine-A",
                "remote_zmq_address": (
                    "host:untrusted.example,handshake:1234,notify:2345"
                ),
                "remote_hosts": ["prefill-a"],
            },
            trusted_remote_hosts={"prefill-a"},
        )


def test_add_new_req_rejects_untrusted_request_id_host():
    metadata = MoRIIOConnectorMetadata()

    with pytest.raises(ValueError, match="request_id host"):
        metadata.add_new_req(
            request_id=_request_id_with_zmq(host="untrusted.example"),
            local_block_ids=[10, 11],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [0, 1],
                "remote_engine_id": "engine-A",
            },
            trusted_remote_hosts={"prefill-a"},
        )


def test_add_new_req_rejects_untrusted_explicit_remote_hosts():
    metadata = MoRIIOConnectorMetadata()

    with pytest.raises(ValueError, match="remote_hosts"):
        metadata.add_new_req(
            request_id="plain-request-id",
            local_block_ids=[10, 11],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [0, 1],
                "remote_engine_id": "engine-A",
                "remote_host": "prefill.example",
                "remote_handshake_port": 1234,
                "remote_notify_port": 2345,
                "remote_hosts": ["untrusted.example"],
            },
            trusted_remote_hosts={"prefill.example"},
        )


def test_add_new_req_rejects_remote_hosts_without_zmq_ports():
    metadata = MoRIIOConnectorMetadata()

    with pytest.raises(ValueError, match="remote_zmq_address"):
        metadata.add_new_req(
            request_id="plain-request-id",
            local_block_ids=[10, 11],
            kv_transfer_params={
                "transfer_id": "tx-1",
                "remote_block_ids": [0, 1],
                "remote_engine_id": "engine-A",
                "remote_hosts": ["prefill-a", "prefill-b"],
            },
        )


def test_write_remote_blocks_uses_remote_zmq_address_without_embedded_request_id():
    class Blocks:
        def get_block_ids(self):
            return [[1, 2, 3]]

    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.tp_size = 1
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler._pending_transfer_id_to_request_id = {}
    scheduler.transfer_id_to_remote_tp_size = {}
    scheduler._pending_transfer_id_to_remote_tp_size = {}
    scheduler.send_notify_block = MagicMock()
    scheduler._trim_block_ids_to_token_span = lambda block_ids, _tokens: block_ids
    scheduler.trusted_remote_hosts = ["prefill.example", "prefill-a", "prefill-b"]

    request = SimpleNamespace(
        request_id="plain-request-id",
        num_prompt_tokens=3,
        kv_transfer_params={
            "transfer_id": "tx-1",
            "do_remote_prefill": True,
            "remote_zmq_address": "host:prefill.example,handshake:1234,notify:2345",
            "remote_tp_size": 2,
            "remote_dp_size": 1,
            "remote_hosts": ["prefill-a", "prefill-b"],
        },
    )

    scheduler.update_state_after_alloc(request, Blocks(), num_external_tokens=3)

    assert scheduler.send_notify_block.call_count == 2
    first_call = scheduler.send_notify_block.call_args_list[0].kwargs
    second_call = scheduler.send_notify_block.call_args_list[1].kwargs
    assert first_call["host"] == "prefill-a"
    assert first_call["port"] == 2345
    assert second_call["host"] == "prefill-b"
    assert second_call["port"] == 2346


def test_build_connector_meta_snapshots_transfer_mapping_without_delta_state():
    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.READ
    scheduler.transfer_id_to_request_id = {"tx0": "req0"}
    scheduler._transfer_ids_to_forget = set()
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_pending_save = {}
    scheduler._reqs_need_send = {}

    metadata = scheduler.build_connector_meta(None)

    scheduler.transfer_id_to_request_id["tx1"] = "req1"

    assert metadata.transfer_id_to_request_id == {"tx0": "req0"}
    assert metadata.transfer_id_to_request_id is not scheduler.transfer_id_to_request_id


def test_build_connector_meta_sends_transfer_mapping_delta_once():
    scheduler = object.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.READ
    scheduler.transfer_id_to_request_id = {"tx0": "req0"}
    scheduler._pending_transfer_id_to_request_id = {"tx0": "req0"}
    scheduler._transfer_ids_to_forget = set()
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_pending_save = {}
    scheduler._reqs_need_send = {}

    first_metadata = scheduler.build_connector_meta(None)
    second_metadata = scheduler.build_connector_meta(None)

    assert first_metadata.transfer_id_to_request_id == {"tx0": "req0"}
    assert second_metadata.transfer_id_to_request_id == {}


def _save_kv_worker(registered_dp_rank: int | None) -> MoRIIOConnectorWorker:
    worker = _bare_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.WRITE
    worker.moriio_config = MagicMock(transfer_timeout=0.01)
    worker._handshake_lock = MagicMock()
    worker._ready_requests = queue.Queue()
    bare_engine_id = "10.0.0.1:4789"
    worker.write_ready_flags = {bare_engine_id: True}
    worker._remote_agents = {}
    if registered_dp_rank is not None:
        worker._remote_agents[f"{bare_engine_id}_dp{registered_dp_rank}"] = {"agent"}
    worker._write_blocks_for_req = MagicMock()
    worker._background_moriio_handshake = MagicMock()
    return worker


def _req_meta_for_dp(remote_dp_rank: int) -> ReqMeta:
    return ReqMeta(
        transfer_id="tx-1",
        local_block_ids=[0, 1],
        remote_block_ids=[0, 1],
        remote_host="10.0.0.1",
        remote_port=4789,
        remote_handshake_port=4789,
        remote_notify_port=61005,
        remote_engine_id="placeholder",
        tp_size=1,
        remote_dp_size=8,
        remote_dp_rank=remote_dp_rank,
    )


def test_save_kv_layer_routes_to_target_dp_rank():
    worker = _save_kv_worker(registered_dp_rank=7)
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save = {"r1": _req_meta_for_dp(7)}

    worker.save_kv_layer(metadata, "layer0", torch.zeros(1), None)

    worker._write_blocks_for_req.assert_called_once()
    worker._background_moriio_handshake.assert_not_called()


def test_save_kv_layer_triggers_handshake_when_target_dp_missing():
    worker = _save_kv_worker(registered_dp_rank=0)
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save = {"r1": _req_meta_for_dp(7)}

    worker.save_kv_layer(metadata, "layer0", torch.zeros(1), None)

    worker._background_moriio_handshake.assert_called_once()
    worker._write_blocks_for_req.assert_not_called()


@pytest.mark.skipif(
    not aiter_available, reason="Requires aiter package for ROCm FlashAttention backend"
)
def test_register_kv_caches(mock_parallel_groups):
    """Test that MoRIIOConnector.register_kv_caches correctly registers kv caches."""
    ROLE = "kv_consumer"
    IP = get_ip()
    vllm_config = create_vllm_config(role=ROLE)
    DEFAULT_PORT = 6301
    TP_RANK = 0
    DP_RANK = 0
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    backend_cls = AiterFlashAttentionBackend

    # Create test kv cache tensors using proper backend shape
    kv_cache_shape = backend_cls.get_kv_cache_shape(
        num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
    )
    shared_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    unique_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    kv_caches = {
        "layer0": shared_tensor,
        "layer1": unique_tensor,
        "layer2": shared_tensor,
    }

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector.threading.Thread"
        ),
    ):
        # Create connector
        vllm_config.kv_transfer_config.kv_connector_extra_config.update(
            {
                "proxy_ip": "127.0.0.1",
                "proxy_ping_port": 12345,
                "http_port": 12346,
            }
        )

        with set_current_vllm_config(vllm_config):
            connector = MoRIIOConnector(
                vllm_config,
                KVConnectorRole.WORKER,
                _make_test_kv_cache_config(),
            )
            connector.connector_worker = FakeMoRIIOConnectorWorker(
                vllm_config, connector.engine_id, hand_shake_latency=0
            )

        from mori.io import (
            MemoryDesc,
        )

        # Execute register_kv_caches
        connector.register_kv_caches(kv_caches)

        # Verify that the MemoryDesc stored in layer_name_to_local_kv_cache_metadata
        assert (
            shared_tensor.data_ptr()
            == MemoryDesc.unpack(
                connector.connector_worker.layer_name_to_local_kv_cache_metadata[
                    "layer0"
                ][0]
            ).data
        )
        assert (
            unique_tensor.data_ptr()
            == MemoryDesc.unpack(
                connector.connector_worker.layer_name_to_local_kv_cache_metadata[
                    "layer1"
                ][0]
            ).data
        )
        assert (
            shared_tensor.data_ptr()
            == MemoryDesc.unpack(
                connector.connector_worker.layer_name_to_local_kv_cache_metadata[
                    "layer2"
                ][0]
            ).data
        )

        # Verify engine keys
        expected_engine_key = f"{ROLE[3:]}:{IP}:{DEFAULT_PORT}:tp{TP_RANK}:dp{DP_RANK}"
        assert (
            MemoryDesc.unpack(
                connector.connector_worker.layer_name_to_local_kv_cache_metadata[
                    "layer0"
                ][0]
            ).engine_key
            == expected_engine_key
        )


@pytest.mark.skipif(
    not aiter_available, reason="Requires aiter package for ROCm FlashAttention backend"
)
def test_moriio_handshake_returns_metadata(mock_parallel_groups):
    """MoRIIO handshake socket returns valid agent metadata over ZMQ."""

    ROLE = "kv_consumer"
    vllm_config = create_vllm_config(role=ROLE)
    from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionBackend

    backend_cls = AiterFlashAttentionBackend

    # Create test kv cache tensors using proper backend shape
    kv_cache_shape = backend_cls.get_kv_cache_shape(
        num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
    )
    shared_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    unique_tensor = torch.zeros(*kv_cache_shape, dtype=torch.float16)
    kv_caches = {
        "layer0": shared_tensor,
        "layer1": unique_tensor,
        "layer2": shared_tensor,
    }

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine.MoRIIOWrapper",
            FakeMoRIIOWrapper,
        ),
    ):
        handshake_port = _find_free_port()
        # Create connector
        vllm_config.kv_transfer_config.kv_connector_extra_config.update(
            {
                "proxy_ip": "127.0.0.1",
                "proxy_ping_port": 12345,
                "http_port": 12346,
                "handshake_port": handshake_port,
            }
        )
        with set_current_vllm_config(vllm_config):
            connector = MoRIIOConnector(
                vllm_config,
                KVConnectorRole.WORKER,
                _make_test_kv_cache_config(),
            )

        # Execute register_kv_caches
        connector.register_kv_caches(kv_caches)

        # Connect to handshake socket and request metadata
        path = make_zmq_path("tcp", "127.0.0.1", handshake_port)
        with zmq_ctx(zmq.DEALER, path) as sock:
            sock.send(MoRIIOConstants.GET_META_MSG)
            received_frame = sock.recv_multipart()

            if len(received_frame) != 2 or received_frame[0] != b"":
                raise ValueError(f"Unexpected frame! {received_frame = }")

            metadata_bytes = received_frame[1]
            decoder = msgspec.msgpack.Decoder(MoRIIOAgentMetadata)
            metadata = decoder.decode(metadata_bytes)
            assert isinstance(metadata, MoRIIOAgentMetadata), (
                "Decoded metadata is not MoRIIOAgentMetadata"
            )


def test_resolve_host_ip_prefers_extra_config():
    """An explicit ``host_ip`` in kv_connector_extra_config overrides get_ip()
    (so an external router can advertise a routable/internal address); an
    absent or empty value falls back to get_ip()."""
    assert resolve_host_ip({"host_ip": "10.0.0.7"}) == "10.0.0.7"

    fallback = get_ip()
    assert resolve_host_ip({}) == fallback
    assert resolve_host_ip({"host_ip": ""}) == fallback
