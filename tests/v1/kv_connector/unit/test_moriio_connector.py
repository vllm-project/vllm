# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import os
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch
import zmq

from tests.conftest import _find_free_port
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOConstants,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    KVConnectorRole,
    MoRIIOConnector,
    MoRIIOConnectorWorker,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_ip,
    make_zmq_path,
)

from .utils import create_request, create_scheduler

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


def _setup_kv_transfer_request(request, remote_host="127.0.0.1", fake_port=4789):
    """Setup KV transfer parameters for a request."""
    request.kv_transfer_params.update(
        {
            "remote_notify_port": fake_port,
            "remote_block_ids": None,
            "remote_host": remote_host,
            "remote_port": fake_port,
            "remote_handshake_port": fake_port,
            "remote_engine_id": "test_engine",
        }
    )
    return request


class FakeMorIIOWrapper:
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

    def send_notify(self, req_ids, remote_ip, remote_port):
        pass

    def pop_finished_req_ids(self):
        pass

    def pop_finished_write_req_ids(self):
        pass

    def shutdown(self):
        pass


class FakeMorIIOConnectorWorker(MoRIIOConnectorWorker):
    # Define a fake remote engine id for testing
    REMOTE_ENGINE_ID = "remote_engine"

    def __init__(
        self, *args, hand_shake_latency: float = 1.8, kv_cache_layout="HND", **kwargs
    ):
        super().__init__(*args, **kwargs)


def create_vllm_config(
    model: str = "facebook/opt-125m",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
    role="kv_consumer",
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
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="MoRIIOConnector",
        kv_role=role,
        enable_permute_local_kv=enable_permute_local_kv,
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


@pytest.fixture
def moriio_read_mode():
    """Force the connector into read mode via env for tests."""
    os.environ["VLLM_MORIIO_CONNECTOR_READ_MODE"] = "True"
    yield
    # Cleanup after test
    os.environ.pop("VLLM_MORIIO_CONNECTOR_READ_MODE", None)


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
    request_id = request.request_id

    scheduler.add_request(request)

    # Fake Config
    request = _setup_kv_transfer_request(request)

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
    request_id = request.request_id

    scheduler.add_request(request)

    # Fake Config
    request = _setup_kv_transfer_request(request)

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


def test_read_mode_loads_remote_block_ids(moriio_read_mode):
    """Read mode loads remote block ids into local cache mapping."""

    # Setup Scheduler and Request
    vllm_config = create_vllm_config(role="kv_consumer")
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
    request_id = request.request_id

    scheduler.add_request(request)
    block_list = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0
    ].req_to_blocks[request_id]

    request = _setup_kv_transfer_request(request)

    # Set remote block ids to be fetched.
    request.kv_transfer_params["remote_block_ids"] = block_list

    # Remote Prefill, triggers MorIIOConnectorMetadata.

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


@pytest.mark.skipif(
    not aiter_available, reason="Requires aiter package for ROCm FlashAttention backend"
)
def test_register_kv_caches(default_vllm_config, mock_parallel_groups):
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

        connector = MoRIIOConnector(vllm_config, KVConnectorRole.WORKER)
        connector.connector_worker = FakeMorIIOConnectorWorker(
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
def test_moriio_handshake_returns_metadata(default_vllm_config, mock_parallel_groups):
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
            FakeMorIIOWrapper,
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
        connector = MoRIIOConnector(vllm_config, KVConnectorRole.WORKER)

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
