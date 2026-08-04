# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cache controller message serialization and deserialization."""

# Standard
import time

# Third Party
import msgspec

# First Party
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    ClearWorkerMsg,
    ClearWorkerRetMsg,
    CompressMsg,
    CompressRetMsg,
    CompressWorkerMsg,
    CompressWorkerRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    DecompressWorkerMsg,
    DecompressWorkerRetMsg,
    DeRegisterMsg,
    ErrorMsg,
    HealthMsg,
    HealthRetMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    HeartbeatMsg,
    KVAdmitMsg,
    KVEvictMsg,
    KVOpEvent,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    MoveWorkerMsg,
    MoveWorkerRetMsg,
    Msg,
    OpType,
    PinMsg,
    PinRetMsg,
    PinWorkerMsg,
    PinWorkerRetMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    RegisterMsg,
)
from lmcache.v1.cache_controller.utils import WorkerInfo


class TestWorkerMessages:
    """Test worker message serialization."""

    def test_register_msg_serialization(self):
        """Test RegisterMsg encode/decode."""
        msg = RegisterMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
            peer_init_url="tcp://127.0.0.1:6000",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, RegisterMsg)
        assert decoded.instance_id == "test_instance"
        assert decoded.worker_id == 0
        assert decoded.ip == "127.0.0.1"
        assert decoded.port == 5000
        assert decoded.peer_init_url == "tcp://127.0.0.1:6000"

    def test_register_msg_without_peer_url(self):
        """Test RegisterMsg without peer_init_url."""
        msg = RegisterMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
            peer_init_url=None,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, RegisterMsg)
        assert decoded.peer_init_url is None

    def test_deregister_msg_serialization(self):
        """Test DeRegisterMsg encode/decode."""
        msg = DeRegisterMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, DeRegisterMsg)
        assert decoded.instance_id == "test_instance"
        assert decoded.worker_id == 0

    def test_kv_admit_msg_serialization(self):
        """Test KVAdmitMsg encode/decode."""
        msg = KVAdmitMsg(
            instance_id="test_instance",
            worker_id=0,
            key=12345,
            location="LocalCPUBackend",
            seq_num=1,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, KVAdmitMsg)
        assert decoded.instance_id == "test_instance"
        assert decoded.worker_id == 0
        assert decoded.key == 12345
        assert decoded.location == "LocalCPUBackend"
        assert decoded.seq_num == 1

    def test_kv_evict_msg_serialization(self):
        """Test KVEvictMsg encode/decode."""
        msg = KVEvictMsg(
            instance_id="test_instance",
            worker_id=0,
            key=12345,
            location="LocalCPUBackend",
            seq_num=2,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, KVEvictMsg)
        assert decoded.key == 12345
        assert decoded.seq_num == 2

    def test_batched_kv_operation_msg(self):
        """Test BatchedKVOperationMsg encode/decode."""
        operations = [
            KVOpEvent(op_type=OpType.ADMIT, key=100, seq_num=1),
            KVOpEvent(op_type=OpType.EVICT, key=200, seq_num=2),
            KVOpEvent(op_type=OpType.ADMIT, key=300, seq_num=3),
        ]

        msg = BatchedKVOperationMsg(
            instance_id="test_instance",
            worker_id=0,
            location="LocalCPUBackend",
            operations=operations,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, BatchedKVOperationMsg)
        assert decoded.instance_id == "test_instance"
        assert decoded.worker_id == 0
        assert decoded.location == "LocalCPUBackend"
        assert len(decoded.operations) == 3
        assert decoded.operations[0].op_type == OpType.ADMIT
        assert decoded.operations[0].key == 100
        assert decoded.operations[1].op_type == OpType.EVICT
        assert decoded.operations[1].key == 200

    def test_heartbeat_msg_serialization(self):
        """Test HeartbeatMsg encode/decode."""
        msg = HeartbeatMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
            peer_init_url="tcp://127.0.0.1:6000",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HeartbeatMsg)
        assert decoded.instance_id == "test_instance"

    def test_batched_p2p_lookup_msg(self):
        """Test BatchedP2PLookupMsg encode/decode."""
        msg = BatchedP2PLookupMsg(
            hashes=[100, 200, 300],
            instance_id="test_instance",
            worker_id=0,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, BatchedP2PLookupMsg)
        assert decoded.hashes == [100, 200, 300]
        assert decoded.instance_id == "test_instance"
        assert decoded.worker_id == 0


class TestControlMessages:
    """Test control message serialization."""

    def test_clear_worker_msg_serialization(self):
        """Test ClearWorkerMsg encode/decode."""
        msg = ClearWorkerMsg(
            worker_event_id="event_123",
            location="LocalCPUBackend",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, ClearWorkerMsg)
        assert decoded.worker_event_id == "event_123"
        assert decoded.location == "LocalCPUBackend"

    def test_pin_worker_msg_serialization(self):
        """Test PinWorkerMsg encode/decode."""
        msg = PinWorkerMsg(
            worker_event_id="event_123",
            location="LocalCPUBackend",
            tokens=[1, 2, 3, 4, 5],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, PinWorkerMsg)
        assert decoded.tokens == [1, 2, 3, 4, 5]

    def test_compress_worker_msg_serialization(self):
        """Test CompressWorkerMsg encode/decode."""
        msg = CompressWorkerMsg(
            worker_event_id="event_123",
            method="gzip",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CompressWorkerMsg)
        assert decoded.method == "gzip"
        assert decoded.tokens == [1, 2, 3]

    def test_compress_worker_msg_without_tokens(self):
        """Test CompressWorkerMsg without tokens (compress all)."""
        msg = CompressWorkerMsg(
            worker_event_id="event_123",
            method="gzip",
            location="LocalCPUBackend",
            tokens=None,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CompressWorkerMsg)
        assert decoded.tokens is None

    def test_decompress_worker_msg_serialization(self):
        """Test DecompressWorkerMsg encode/decode."""
        msg = DecompressWorkerMsg(
            worker_event_id="event_123",
            method="gzip",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, DecompressWorkerMsg)
        assert decoded.method == "gzip"

    def test_move_worker_msg_serialization(self):
        """Test MoveWorkerMsg encode/decode."""
        msg = MoveWorkerMsg(
            worker_event_id="event_123",
            old_position="LocalDiskBackend",
            new_position=("tcp://127.0.0.1:5000", "LocalCPUBackend"),
            tokens=[1, 2, 3],
            copy=True,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, MoveWorkerMsg)
        assert decoded.old_position == "LocalDiskBackend"
        assert decoded.new_position == ("tcp://127.0.0.1:5000", "LocalCPUBackend")
        assert decoded.copy is True

    def test_health_worker_msg_serialization(self):
        """Test HealthWorkerMsg encode/decode."""
        msg = HealthWorkerMsg(worker_event_id="event_123")

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HealthWorkerMsg)
        assert decoded.worker_event_id == "event_123"


class TestOrchestrationMessages:
    """Test orchestration message serialization."""

    def test_lookup_msg_serialization(self):
        """Test LookupMsg encode/decode."""
        msg = LookupMsg(
            event_id="event_123",
            tokens=[1, 2, 3, 4, 5],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, LookupMsg)
        assert decoded.event_id == "event_123"
        assert decoded.tokens == [1, 2, 3, 4, 5]

    def test_clear_msg_serialization(self):
        """Test ClearMsg encode/decode."""
        msg = ClearMsg(
            event_id="event_123",
            instance_id="test_instance",
            location="LocalCPUBackend",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, ClearMsg)
        assert decoded.instance_id == "test_instance"
        assert decoded.location == "LocalCPUBackend"

    def test_pin_msg_serialization(self):
        """Test PinMsg encode/decode."""
        msg = PinMsg(
            event_id="event_123",
            instance_id="test_instance",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, PinMsg)
        assert decoded.tokens == [1, 2, 3]

    def test_compress_msg_serialization(self):
        """Test CompressMsg encode/decode."""
        msg = CompressMsg(
            event_id="event_123",
            instance_id="test_instance",
            method="gzip",
            location="LocalCPUBackend",
            tokens=[1, 2, 3],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CompressMsg)
        assert decoded.method == "gzip"

    def test_decompress_msg_serialization(self):
        """Test DecompressMsg encode/decode."""
        msg = DecompressMsg(
            event_id="event_123",
            instance_id="test_instance",
            method="gzip",
            location="LocalCPUBackend",
            tokens=None,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, DecompressMsg)
        assert decoded.tokens is None

    def test_move_msg_serialization(self):
        """Test MoveMsg encode/decode."""
        msg = MoveMsg(
            event_id="event_123",
            old_position=("instance1", "LocalCPUBackend"),
            new_position=("instance2", "LocalCPUBackend"),
            tokens=[1, 2, 3],
            copy=False,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, MoveMsg)
        assert decoded.old_position == ("instance1", "LocalCPUBackend")
        assert decoded.new_position == ("instance2", "LocalCPUBackend")
        assert decoded.copy is False

    def test_health_msg_serialization(self):
        """Test HealthMsg encode/decode."""
        msg = HealthMsg(
            event_id="event_123",
            instance_id="test_instance",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HealthMsg)
        assert decoded.instance_id == "test_instance"

    def test_check_finish_msg_serialization(self):
        """Test CheckFinishMsg encode/decode."""
        msg = CheckFinishMsg(event_id="event_123")

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CheckFinishMsg)
        assert decoded.event_id == "event_123"

    def test_query_inst_msg_serialization(self):
        """Test QueryInstMsg encode/decode."""
        msg = QueryInstMsg(
            event_id="event_123",
            ip="127.0.0.1",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, QueryInstMsg)
        assert decoded.ip == "127.0.0.1"

    def test_query_worker_info_msg_serialization(self):
        """Test QueryWorkerInfoMsg encode/decode."""
        msg = QueryWorkerInfoMsg(
            event_id="event_123",
            instance_id="test_instance",
            worker_ids=[0, 1, 2],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, QueryWorkerInfoMsg)
        assert decoded.worker_ids == [0, 1, 2]


class TestReturnMessages:
    """Test return message serialization."""

    def test_lookup_ret_msg_serialization(self):
        """Test LookupRetMsg encode/decode."""
        msg = LookupRetMsg(
            event_id="event_123",
            layout_info={"instance1": ("LocalCPUBackend", 100)},
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, LookupRetMsg)
        assert decoded.layout_info == {"instance1": ("LocalCPUBackend", 100)}

    def test_clear_ret_msg_serialization(self):
        """Test ClearRetMsg encode/decode."""
        msg = ClearRetMsg(
            event_id="event_123",
            num_tokens=100,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, ClearRetMsg)
        assert decoded.num_tokens == 100

    def test_pin_ret_msg_serialization(self):
        """Test PinRetMsg encode/decode."""
        msg = PinRetMsg(
            event_id="event_123",
            num_tokens=50,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, PinRetMsg)
        assert decoded.num_tokens == 50

    def test_compress_ret_msg_serialization(self):
        """Test CompressRetMsg encode/decode."""
        msg = CompressRetMsg(
            event_id="event_123",
            num_tokens=75,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CompressRetMsg)
        assert decoded.num_tokens == 75

    def test_decompress_ret_msg_serialization(self):
        """Test DecompressRetMsg encode/decode."""
        msg = DecompressRetMsg(
            event_id="event_123",
            num_tokens=75,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, DecompressRetMsg)
        assert decoded.num_tokens == 75

    def test_move_ret_msg_serialization(self):
        """Test MoveRetMsg encode/decode."""
        msg = MoveRetMsg(
            event_id="event_123",
            num_tokens=200,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, MoveRetMsg)
        assert decoded.num_tokens == 200

    def test_health_ret_msg_serialization(self):
        """Test HealthRetMsg encode/decode."""
        msg = HealthRetMsg(
            event_id="event_123",
            error_codes={0: 0, 1: 0, 2: -1},
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HealthRetMsg)
        assert decoded.error_codes == {0: 0, 1: 0, 2: -1}

    def test_check_finish_ret_msg_serialization(self):
        """Test CheckFinishRetMsg encode/decode."""
        msg = CheckFinishRetMsg(status="completed")

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CheckFinishRetMsg)
        assert decoded.status == "completed"

    def test_query_inst_ret_msg_serialization(self):
        """Test QueryInstRetMsg encode/decode."""
        msg = QueryInstRetMsg(
            event_id="event_123",
            instance_id="test_instance",
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, QueryInstRetMsg)
        assert decoded.instance_id == "test_instance"

    def test_query_inst_ret_msg_none_instance(self):
        """Test QueryInstRetMsg with None instance_id."""
        msg = QueryInstRetMsg(
            event_id="event_123",
            instance_id=None,
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, QueryInstRetMsg)
        assert decoded.instance_id is None

    def test_query_worker_info_ret_msg_serialization(self):
        """Test QueryWorkerInfoRetMsg encode/decode."""
        worker_info = WorkerInfo(
            instance_id="test_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
            peer_init_url="tcp://127.0.0.1:6000",
            registration_time=time.time(),
            last_heartbeat_time=time.time(),
        )

        msg = QueryWorkerInfoRetMsg(
            event_id="event_123",
            worker_infos=[worker_info],
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, QueryWorkerInfoRetMsg)
        assert len(decoded.worker_infos) == 1
        assert decoded.worker_infos[0].instance_id == "test_instance"

    def test_batched_p2p_lookup_ret_msg(self):
        """Test BatchedP2PLookupRetMsg encode/decode."""
        msg = BatchedP2PLookupRetMsg(
            layout_info=[
                ("instance1", "LocalCPUBackend", 10, "tcp://127.0.0.1:5000"),
                ("instance2", "LocalCPUBackend", 5, "tcp://127.0.0.1:5001"),
            ]
        )

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, BatchedP2PLookupRetMsg)
        assert len(decoded.layout_info) == 2
        assert decoded.layout_info[0][0] == "instance1"
        assert decoded.layout_info[0][2] == 10

    def test_error_msg_serialization(self):
        """Test ErrorMsg encode/decode."""
        msg = ErrorMsg(error="Something went wrong")

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, ErrorMsg)
        assert decoded.error == "Something went wrong"


class TestWorkerReturnMessages:
    """Test worker return message serialization."""

    def test_clear_worker_ret_msg(self):
        """Test ClearWorkerRetMsg encode/decode."""
        msg = ClearWorkerRetMsg(num_tokens=100)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, ClearWorkerRetMsg)
        assert decoded.num_tokens == 100

    def test_pin_worker_ret_msg(self):
        """Test PinWorkerRetMsg encode/decode."""
        msg = PinWorkerRetMsg(num_tokens=50)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, PinWorkerRetMsg)
        assert decoded.num_tokens == 50

    def test_compress_worker_ret_msg(self):
        """Test CompressWorkerRetMsg encode/decode."""
        msg = CompressWorkerRetMsg(num_tokens=75)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, CompressWorkerRetMsg)
        assert decoded.num_tokens == 75

    def test_decompress_worker_ret_msg(self):
        """Test DecompressWorkerRetMsg encode/decode."""
        msg = DecompressWorkerRetMsg(num_tokens=75)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, DecompressWorkerRetMsg)
        assert decoded.num_tokens == 75

    def test_move_worker_ret_msg(self):
        """Test MoveWorkerRetMsg encode/decode."""
        msg = MoveWorkerRetMsg(num_tokens=200)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, MoveWorkerRetMsg)
        assert decoded.num_tokens == 200

    def test_health_worker_ret_msg(self):
        """Test HealthWorkerRetMsg encode/decode."""
        msg = HealthWorkerRetMsg(error_code=0)

        encoded = msgspec.msgpack.encode(msg)
        decoded = msgspec.msgpack.decode(encoded, type=Msg)

        assert isinstance(decoded, HealthWorkerRetMsg)
        assert decoded.error_code == 0


class TestMessageDescribe:
    """Test message describe() methods."""

    def test_register_msg_describe(self):
        """Test RegisterMsg describe method."""
        msg = RegisterMsg(
            instance_id="test_instance",
            worker_id=0,
            ip="127.0.0.1",
            port=5000,
            peer_init_url="tcp://127.0.0.1:6000",
        )
        description = msg.describe()
        assert "test_instance" in description
        assert "worker" in description.lower()

    def test_lookup_msg_describe(self):
        """Test LookupMsg describe method."""
        msg = LookupMsg(event_id="event_123", tokens=[1, 2, 3])
        description = msg.describe()
        assert "lookup" in description.lower() or "tokens" in description.lower()

    def test_error_msg_describe(self):
        """Test ErrorMsg describe method."""
        msg = ErrorMsg(error="Test error")
        description = msg.describe()
        assert "error" in description.lower() or "Test error" in description
