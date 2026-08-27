# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import socket
import uuid
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
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
    compute_layer_kv_cache_shape_bytes,
)

from .utils import create_request, create_scheduler


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _make_test_kv_cache_config() -> KVCacheConfig:
    layer_names = ["layer0", "layer1", "layer2"]
    num_blocks = 2
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
    )
    page = spec.page_size_bytes
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=len(layer_names) * num_blocks * page,
                layers=layer_names,
                layer_stride=num_blocks * page,
                block_stride=page,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
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
        kv_cache_layout="LBHNC",
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
        kv_connector_extra_config={"read_mode": read_mode, "backend": "xgmi"},
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
        req_meta.local_block_ids[0],
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
        req_meta.local_block_ids[0],
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
    request.kv_transfer_params["remote_block_ids"] = [block_list]

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
        req_meta.local_block_ids[0],
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
    # WRITE-mode release advertises the consumer (decode) TP size so the prefill
    # side counts the right number of ACKs via get_moriio_expected_ack_count,
    # mirroring the READ-mode release in _pop_done_transfers (see
    # test_read_completion_sends_structured_release_with_consumer_tp_size). The
    # fixture's tp_size is 2, so consumer_tp_size == 2.
    assert msgspec.msgpack.decode(payload) == {
        "type": "release",
        "transfer_id": "xfer-7",
        "consumer_tp_size": 2,
    }


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
    # Create test kv cache tensors using KVCacheSpec layout
    shape = compute_layer_kv_cache_shape_bytes(
        FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=64, dtype=torch.float16
        ),
        2,
    )
    shared_tensor = torch.zeros(*shape, dtype=torch.int8).view(torch.float16)
    unique_tensor = torch.zeros(*shape, dtype=torch.int8).view(torch.float16)
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
    # Create test kv cache tensors using KVCacheSpec layout
    shape = compute_layer_kv_cache_shape_bytes(
        FullAttentionSpec(
            block_size=16, num_kv_heads=4, head_size=64, dtype=torch.float16
        ),
        2,
    )
    shared_tensor = torch.zeros(*shape, dtype=torch.int8).view(torch.float16)
    unique_tensor = torch.zeros(*shape, dtype=torch.int8).view(torch.float16)
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


def _make_hybrid_kv_cache_config() -> KVCacheConfig:
    """One full-attention group + one sliding-window group (Gemma-like)."""
    full_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=64, dtype=torch.float16
    )
    sw_spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        sliding_window=32,
    )
    num_blocks = 2
    page = full_spec.page_size_bytes
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * page,
                layers=["full0", "sw0"],
                layer_stride=num_blocks * page,
                block_stride=page,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["full0"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["sw0"], kv_cache_spec=sw_spec),
        ],
    )


def _read_scheduler(
    kv_cache_config: KVCacheConfig, disable_hma: bool = False
) -> MoRIIOConnectorScheduler:
    vllm_config = create_vllm_config(role="kv_producer", read_mode=True)
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hma
    with set_current_vllm_config(vllm_config):
        connector = MoRIIOConnector(
            vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
        )
    assert connector.connector_scheduler is not None
    return connector.connector_scheduler


def test_hma_blocks_per_sw_two_groups():
    """A Full + SlidingWindow config runs with HMA and computes block budgets
    correctly"""
    scheduler = _read_scheduler(_make_hybrid_kv_cache_config())
    assert scheduler._is_hma_required is True
    # cdiv(32, 16) + 1 == 3 for the sliding-window group, 0 for full attention.
    assert scheduler.blocks_per_sw == [0, 3]


@pytest.mark.parametrize(
    "swa_enabled, disable_hma, expected_is_hma",
    [
        (True, False, True),  # sliding-window group present, HMA enabled
        (True, True, False),  # sliding-window group present but HMA disabled
        (False, False, False),  # full-attention only, HMA not needed
    ],
)
def test_is_hma_required(swa_enabled, disable_hma, expected_is_hma):
    """_is_hma_required tracks both the KV cache groups and the
    --disable-hybrid-kv-cache-manager flag. When HMA is off,
    get_exchange_clipped_blocks must be a no-op."""
    config = (
        _make_hybrid_kv_cache_config() if swa_enabled else _make_test_kv_cache_config()
    )
    scheduler = _read_scheduler(config, disable_hma=disable_hma)
    assert scheduler._is_hma_required is expected_is_hma
    if not expected_is_hma:
        blocks = [[1, 2, 3, 4, 5]]
        assert scheduler.get_exchange_clipped_blocks(blocks) == blocks


def test_non_sliding_window_hybrid_is_rejected():
    """A hybrid group that is not sliding-window (e.g. chunked-local
    attention) must fail closed rather than be silently mistransferred."""
    full_spec = FullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=64, dtype=torch.float16
    )
    local_spec = ChunkedLocalAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        attention_chunk_size=32,
    )
    num_blocks = 2
    page = full_spec.page_size_bytes
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * num_blocks * page,
                layers=["full0", "local0"],
                layer_stride=num_blocks * page,
                block_stride=page,
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["full0"], kv_cache_spec=full_spec),
            KVCacheGroupSpec(layer_names=["local0"], kv_cache_spec=local_spec),
        ],
    )
    with pytest.raises(NotImplementedError, match="sliding-window hybrid"):
        _read_scheduler(config)


def test_token_count_basis_uses_full_attention_group():
    """Chunked-prefill token counting must use an unclipped full-attention
    group (blocks_per_sw == 0), not a clipped sliding-window group."""
    scheduler = _read_scheduler(_make_hybrid_kv_cache_config())
    # Group 0 is the full-attention group
    assert scheduler._full_attn_group_idx == 0
    assert scheduler._full_attn_block_size == 16


def test_get_exchange_clipped_blocks_clips_only_sw_group():
    """get_exchange_clipped_blocks keeps the full attn group intact and clips the
    sliding-window group to its window tail."""
    scheduler = _read_scheduler(_make_hybrid_kv_cache_config())
    full = [10, 11, 12, 13, 14]
    sw = [20, 21, 22, 23, 24]
    clipped = scheduler.get_exchange_clipped_blocks([full, sw])
    assert clipped[0] == full
    assert clipped[1] == [22, 23, 24]


def test_metadata_hma_block_ids_preserved_per_group():
    """add_new_req stores per-group (BlockIds) block lists unchanged for both
    read and write, so the hybrid group structure is retained."""
    metadata = MoRIIOConnectorMetadata()

    # Assume:
    # - Full-attention group (6 blocks) +
    # - sliding-window group already clipped to its window tail (3 blocks).
    fa_blocks = [0, 1, 2, 3, 4, 5]
    sw_blocks = [10, 11, 12]
    local_block_ids = [fa_blocks, sw_blocks]
    remote_block_ids = [[100, 101, 102, 103, 104, 105], [200, 201, 202]]

    base_params = {
        "remote_engine_id": "remote-engine",
        "remote_host": "127.0.0.1",
        "remote_handshake_port": 6301,
        "remote_notify_port": 61005,
        "remote_block_ids": remote_block_ids,
    }

    # Read mode: both local and remote ids stay per-group.
    metadata.add_new_req(
        request_id="recv-req",
        local_block_ids=local_block_ids,
        kv_transfer_params={**base_params, "transfer_id": "recv-req"},
    )
    recv_meta = metadata.reqs_to_recv["recv-req"]
    assert recv_meta.local_block_ids == [fa_blocks, sw_blocks]
    assert recv_meta.remote_block_ids == remote_block_ids

    # Write mode: the decode peer allocates its own blocks, so #remote_block_ids may
    # be empty, Local group structure is kept.
    metadata.add_new_req(
        request_id="save-req",
        local_block_ids=local_block_ids,
        kv_transfer_params={
            **base_params,
            "transfer_id": "save-req",
            "remote_block_ids": [],
        },
        write_mode=True,
    )
    save_meta = metadata.reqs_to_save["save-req"]
    assert save_meta.local_block_ids == [fa_blocks, sw_blocks]
    assert save_meta.remote_block_ids == []


def test_single_group_path_unchanged():
    """A full attn config does not enable HMA and never clips blocks."""
    scheduler = _read_scheduler(_make_test_kv_cache_config())
    assert scheduler._is_hma_required is False
    assert scheduler.blocks_per_sw == [0]
    blocks = [[1, 2, 3, 4, 5]]
    assert scheduler.get_exchange_clipped_blocks(blocks) == blocks


def test_hybrid_write_mode_rejected():
    """Hybrid KV cache groups are unsupported in WRITE mode and fail closed."""
    vllm_config = create_vllm_config(role="kv_producer", read_mode=False)
    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(NotImplementedError),
    ):
        MoRIIOConnector(
            vllm_config,
            KVConnectorRole.SCHEDULER,
            _make_hybrid_kv_cache_config(),
        )


def test_worker_layer_to_group_routing(mock_parallel_groups):
    """The worker maps every layer to its KV cache group correctly."""
    vllm_config = create_vllm_config(role="kv_consumer", read_mode=True)
    # Building the worker directly bypasses MoRIIOConnector._set_port_defaults,
    # so provide the ports manually.
    vllm_config.kv_transfer_config.kv_connector_extra_config.update(
        {
            "http_port": 12346,
            "handshake_port": 12347,
            "notify_port": 12348,
        }
    )
    with set_current_vllm_config(vllm_config):
        worker = FakeMoRIIOConnectorWorker(
            vllm_config,
            "engine0",
            hand_shake_latency=0,
            kv_cache_config=_make_hybrid_kv_cache_config(),
        )
    assert worker.layer_to_group == {"full0": 0, "sw0": 1}
