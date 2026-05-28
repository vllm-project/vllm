# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import subprocess
import threading
import uuid
from collections import defaultdict
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
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOConstants,
    MoRIIOMode,
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
from vllm.v1.kv_cache_interface import KVCacheConfig

from .utils import create_request, create_scheduler


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])


aiter_available = importlib.util.find_spec("aiter") is not None
mori_available = importlib.util.find_spec("mori") is not None


def _rdma_available() -> bool:
    """Check if RDMA devices are available."""
    try:
        result = subprocess.run(["ibv_devinfo"], capture_output=True, text=True)
        return "No IB devices found" not in result.stderr
    except FileNotFoundError:
        return False


rdma_available = _rdma_available()

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

    def send_notify(self, req_ids, remote_ip, remote_port):
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
    kv_transfer_config = KVTransferConfig(
        kv_connector="MoRIIOConnector",
        kv_role=role,
        enable_permute_local_kv=enable_permute_local_kv,
        kv_connector_extra_config={"read_mode": read_mode},
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


@pytest.mark.skipif(
    not aiter_available, reason="Requires aiter package for ROCm FlashAttention backend"
)
@pytest.mark.skipif(not rdma_available, reason="No RDMA devices available")
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
@pytest.mark.skipif(not rdma_available, reason="No RDMA devices available")
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


def _make_scheduler(role: str = "kv_producer") -> "MoRIIOConnectorScheduler":
    """Scheduler with the minimal kv_connector_extra_config required by __init__."""
    vllm_config = create_vllm_config(role=role)
    vllm_config.kv_transfer_config.kv_connector_extra_config.update(
        {
            "proxy_ip": "127.0.0.1",
            "proxy_ping_port": 36367,
            "http_port": 8100,
            "handshake_port": 6300,
            "notify_port": 6100,
        }
    )
    return MoRIIOConnectorScheduler(vllm_config, "test_engine")


def _make_worker() -> MoRIIOConnectorWorker:
    """Worker via __new__ so we set only the attributes get_finished touches."""
    w = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    w.transfer_id_to_request_id = {}
    w._read_sibs = {}
    w._read_done_tids = set()
    w._read_finished_seen = set()
    w._read_req_tid = {}
    w._read_seen_tid_sibcount = {}
    w._recving_transfers = defaultdict(list)
    w._recving_transfers_callback_addr = {}
    w._recving_transfer_id = {}
    return w


def test_best_of_n_map_accumulates_siblings():
    """FIX(best_of-1many) scheduler-side: map_request_id accumulates sibling
    req_ids under a shared transfer_id (was 1:1 dict that clobbered all but the
    last)."""
    sched = _make_scheduler(role="kv_producer")
    tid = "tx-d1f3abc456789012"
    siblings = [
        "0_cmpl-fb4bc0e0a1234567",
        "1_cmpl-fb4bc0e0a1234567",
        "2_cmpl-fb4bc0e0a1234567",
        "3_cmpl-fb4bc0e0a1234567",
        "4_cmpl-fb4bc0e0a1234567",
        "5_cmpl-fb4bc0e0a1234567",
        "6_cmpl-fb4bc0e0a1234567",
        "7_cmpl-fb4bc0e0a1234567",
    ]

    for s in siblings:
        sched.map_request_id(s, tid)

    assert sched.transfer_id_to_request_id[tid] == siblings, (
        "all 8 siblings should be stored under the shared transfer_id"
    )
    assert all(sched.request_id_to_transfer_id[s] == tid for s in siblings)

    # unmap one at a time: tid key persists until the last sibling is gone.
    for s in siblings[:-1]:
        sched.unmap_request_id(s)
        assert tid in sched.transfer_id_to_request_id, (
            f"tid evicted too early after unmap({s})"
        )
        assert s not in sched.transfer_id_to_request_id[tid]
    sched.unmap_request_id(siblings[-1])
    assert tid not in sched.transfer_id_to_request_id, (
        "tid should be evicted when its last sibling is unmapped"
    )


def test_consumer_write_mode_fans_out_completion():
    """FIX(best_of-1many) consumer-side: get_finished in WRITE mode maps ONE shared
    transfer_id to ALL its best_of siblings (was 1:1, leaving n-1 stuck Deferred)."""
    worker = _make_worker()
    worker.is_producer = False
    worker.mode = MoRIIOMode.WRITE
    tid = "tx-35dba48109876543"
    siblings = [
        "0_cmpl-35dba481b9876543",
        "1_cmpl-35dba481b9876543",
        "2_cmpl-35dba481b9876543",
    ]
    worker.transfer_id_to_request_id = {tid: siblings}

    fake_wrapper = MagicMock()
    fake_wrapper.pop_finished_write_req_ids.return_value = {tid}
    worker.moriio_wrapper = fake_wrapper

    done_sending, done_recving = worker.get_finished()
    assert done_sending == set()
    assert done_recving == set(siblings), (
        "one shared transfer_id must release ALL its best_of siblings"
    )


def test_consumer_read_mode_discards_done_recving():
    """READ-mode consumer is synchronous (load_kv_async=False) so reqs never enter
    WAITING_FOR_REMOTE_KVS. _pop_done_transfers's return is consumed internally for
    send_notify; passing it on would trip the scheduler's is_finished assert."""
    worker = _make_worker()
    worker.is_producer = False
    worker.mode = MoRIIOMode.READ
    worker.moriio_wrapper = MagicMock()
    # Pretend the internal pull-complete path returned a non-empty set.
    worker._pop_done_transfers = MagicMock(return_value={"0_cmpl-a1b2c3d4e5f60718"})

    done_sending, done_recving = worker.get_finished()
    assert done_sending == set()
    assert done_recving == set(), "READ-mode consumer must discard done_recving"


def test_producer_read_mode_finished_gated_release():
    """FIX(read-release): producer READ-mode get_finished releases siblings only
    after vLLM marks them finished. Step 1: notify arrives, no finished ->
    no release. Step 2/3: each finished sibling is released individually; tid
    is pruned only when the last is gone."""
    worker = _make_worker()
    worker.is_producer = True
    worker.mode = MoRIIOMode.READ
    tid = "tx-7d3e9a2b54c1f680"
    sib0 = "0_cmpl-7d3e9a2bc1f68054"
    sib1 = "1_cmpl-7d3e9a2bc1f68054"
    worker._read_sibs = {tid: {sib0, sib1}}
    worker._read_seen_tid_sibcount = {tid: 2}

    fake_wrapper = MagicMock()
    worker.moriio_wrapper = fake_wrapper

    # Step 1: decode read-complete notify arrives (shared tid), no finished yet.
    fake_wrapper.pop_finished_req_ids.return_value = {tid}
    done_sending, _ = worker.get_finished(finished_req_ids=set())
    assert done_sending == set(), "release must be gated on finished_req_ids"
    assert worker._read_sibs == {tid: {sib0, sib1}}, "no sibling freed yet"
    assert tid in worker._read_done_tids, "tid persisted for later release"

    # Step 2: vLLM marks sib0 finished -> only sib0 is freed; sib1 still held.
    fake_wrapper.pop_finished_req_ids.return_value = set()
    done_sending, _ = worker.get_finished(finished_req_ids={sib0})
    assert done_sending == {sib0}
    assert worker._read_sibs == {tid: {sib1}}
    assert tid in worker._read_done_tids

    # Step 3: vLLM marks sib1 finished -> released; tid fully drained from all state.
    done_sending, _ = worker.get_finished(finished_req_ids={sib1})
    assert done_sending == {sib1}
    assert tid not in worker._read_sibs
    assert tid not in worker._read_done_tids
    assert tid not in worker._read_seen_tid_sibcount


def test_pop_done_transfers_sends_shared_transfer_id():
    """FIX(read-release) decode-side: notify the prefill with the SHARED
    transfer_id (captured at read setup), not the local sibling req_id (which has
    the per-sibling prefix the prefill doesn't own). Bookkeeping cleaned up after."""
    worker = _make_worker()
    fake_wrapper = MagicMock()
    fake_wrapper.lock = threading.Lock()
    worker.moriio_wrapper = fake_wrapper

    shared_tid = "tx-aabbccdd11223344"
    req_id = "0_cmpl-aabbccdd11223344"
    succeeded = MagicMock()
    succeeded.Succeeded.return_value = True
    worker._recving_transfers = defaultdict(list, {req_id: [succeeded]})
    worker._recving_transfers_callback_addr = {req_id: ("127.0.0.1", "6100")}
    worker._recving_transfer_id = {req_id: shared_tid}

    done = worker._pop_done_transfers()
    assert done == {req_id}
    fake_wrapper.send_notify.assert_called_once_with(shared_tid, "127.0.0.1", "6100")
    assert req_id not in worker._recving_transfers
    assert req_id not in worker._recving_transfers_callback_addr
    assert req_id not in worker._recving_transfer_id, (
        "_recving_transfer_id must be cleaned up alongside the other dicts"
    )
