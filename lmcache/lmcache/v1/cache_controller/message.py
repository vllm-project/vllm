# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

# Third Party
import msgspec

# First Party
from lmcache.v1.cache_controller.commands.base import HeartbeatCommand
from lmcache.v1.cache_controller.commands.full_sync import FullSyncCommand

# Type alias for all command types - msgspec needs Union for tagged unions
AnyCommand = Union[FullSyncCommand, HeartbeatCommand]


class MsgBase(msgspec.Struct, tag=True):  # type: ignore
    """Base class for all messages"""

    def describe(self) -> str:
        return ""


# NOTE: The additional layer of abstraction is to
# differentiate among
# (1) WorkerMsg: push-pull (lmcache->controller)
# (2) WorkerReqMsg: req-reply (lmcache->controller)
# (3) ControlMessage: req-reply (controller->lmcache)
# (4) OrchMsg: req-reply (ochestrator->controller)


"""Message from LMCache to Controller"""


class WorkerMsg(MsgBase):
    """Message between LMCache and Controller"""

    def describe(self) -> str:
        return ""


"""Worker Request (requiring a reply) Message from LMCache to Controller"""


class WorkerReqMsg(MsgBase):
    def describe(self) -> str:
        return ""


class RegisterMsg(WorkerReqMsg):
    """Message for Registration (REQ-REP mode)"""

    instance_id: str
    worker_id: int
    ip: str
    port: int
    # URL for actual KV cache transfer, only useful when p2p is enabled
    peer_init_url: Optional[str]

    def describe(self) -> str:
        return (
            f"Registering instance {self.instance_id}, "
            f"worker {self.worker_id} "
            f"at {self.ip}:{self.port}"
            f" with peer init URL {self.peer_init_url}"
        )


class DeRegisterMsg(WorkerMsg):
    """Message for Deregistration"""

    instance_id: str
    worker_id: int
    ip: str
    port: int

    def describe(self) -> str:
        return (
            f"Deregistering instance {self.instance_id}, "
            f"worker {self.worker_id} "
            f"at {self.ip}:{self.port}"
        )


class KVOperationMsg(WorkerMsg):
    """Base class for KV operation messages (admit/evict) with full context"""

    instance_id: str
    worker_id: int
    key: int
    location: str
    seq_num: int = 0


class KVAdmitMsg(KVOperationMsg):
    """Message for KV chunk admission"""

    def describe(self) -> str:
        return f"kv_admit {self.key} to {self.instance_id}"


class KVEvictMsg(KVOperationMsg):
    """Message for KV chunk eviction"""

    def describe(self) -> str:
        return f"kv_evict {self.key} from {self.instance_id}"


@dataclass
class WorkerInfo:
    """Information about a worker for external API consumption."""

    instance_id: str
    worker_id: int
    ip: str
    port: int
    peer_init_url: Optional[str]
    registration_time: float
    last_heartbeat_time: float


class OpType(Enum):
    """Enum for KV operation types"""

    ADMIT = "admit"
    EVICT = "evict"


class KVOpEvent(msgspec.Struct):
    """Lightweight KV operation event for queue storage (without common fields)"""

    op_type: OpType
    key: int
    seq_num: int


class BatchedKVOperationMsg(WorkerMsg):
    """Batched KV operation message with common fields and lightweight operations

    Design: Common fields (instance_id, worker_id, location) are stored once
    at the batch level, while individual operations only contain the varying
    fields (op_type, key, seq_num). This reduces memory and network overhead.
    """

    instance_id: str
    worker_id: int
    location: str
    operations: list[KVOpEvent]

    def describe(self) -> str:
        return (
            f"Batched KV operations with {len(self.operations)} messages "
            f"from {self.instance_id}:{self.worker_id}"
        )


# ============= Full Sync Messages (PUSH mode) =============


class FullSyncBatchMsg(WorkerMsg):
    """Full sync batch message (PUSH mode, no confirmation needed)

    Used to send a batch of chunk hashes during full sync.
    """

    instance_id: str
    worker_id: int
    location: str
    sync_id: str
    batch_id: int  # Current batch number (0-indexed)
    keys: list[int]  # List of chunk_hash in this batch

    def describe(self) -> str:
        return (
            f"FullSyncBatch sync_id={self.sync_id} batch_id={self.batch_id} "
            f"keys_count={len(self.keys)} from {self.instance_id}:{self.worker_id}"
        )


class FullSyncEndMsg(WorkerMsg):
    """Full sync end message (PUSH mode)

    Sent after all batches are sent to signal completion.
    """

    instance_id: str
    worker_id: int
    location: str
    sync_id: str
    actual_total_keys: int  # Actual total keys sent (for verification)

    def describe(self) -> str:
        return (
            f"FullSyncEnd sync_id={self.sync_id} "
            f"total_keys={self.actual_total_keys} from "
            f"{self.instance_id}:{self.worker_id}"
        )


class HeartbeatMsg(WorkerReqMsg):
    """Message for heartbeat (REQ-REP mode), include register info for re-register"""

    instance_id: str
    worker_id: int
    ip: str
    port: int
    peer_init_url: Optional[str]

    def describe(self) -> str:
        return f"Heartbeat from instance {self.instance_id}, worker {self.worker_id}"


class BatchedP2PLookupMsg(WorkerReqMsg):
    """Batched P2P lookup message"""

    hashes: list[int]
    instance_id: str
    worker_id: int  # TP rank

    def describe(self) -> str:
        return (
            f"Batched P2P lookup for {len(self.hashes)} keys from "
            f"instance id {self.instance_id} and "
            f"worker id {self.worker_id}"
        )


class FullSyncStartMsg(WorkerReqMsg):
    """Full sync start message (REQ-REP mode, needs confirmation)

    Sent before starting full sync to notify controller and get confirmation.
    """

    instance_id: str
    worker_id: int
    location: str
    sync_id: str  # Sync session ID
    total_keys: int  # Expected total key count
    batch_count: int  # Expected batch count

    def describe(self) -> str:
        return (
            f"FullSyncStart sync_id={self.sync_id} "
            f"total_keys={self.total_keys} batches={self.batch_count} "
            f"from {self.instance_id}:{self.worker_id}"
        )


class FullSyncStatusMsg(WorkerReqMsg):
    """Full sync status query message (REQ-REP mode)

    Used to query sync progress and check if freeze mode can be exited.
    """

    instance_id: str
    worker_id: int
    sync_id: str

    def describe(self) -> str:
        return (
            f"FullSyncStatus query sync_id={self.sync_id} "
            f"from {self.instance_id}:{self.worker_id}"
        )


"""Worker Request Return Message from Controller back to LMCache"""


class WorkerReqRetMsg(MsgBase):
    def describe(self) -> str:
        return ""


class RegisterRetMsg(WorkerReqRetMsg):
    """Register response message with controller configuration

    Returns configuration information that the worker needs for initialization,
    such as dedicated heartbeat URL.
    """

    # Extra configuration from controller to worker
    # e.g., {"heartbeat_url": "tcp://...:8082"}
    extra_config: dict[str, str] = msgspec.field(default_factory=dict)

    def describe(self) -> str:
        return f"RegisterRet extra_config={self.extra_config}"


class HeartbeatRetMsg(WorkerReqRetMsg):
    """Heartbeat response message with optional commands

    The controller can use this to send commands to workers through
    the heartbeat mechanism. This provides a general-purpose way to
    trigger actions on workers without adding new message types.

    The commands field uses msgspec's tagged union for polymorphic
    serialization - each command subclass is identified by its tag.
    Commands are executed sequentially by the worker.
    """

    commands: List[AnyCommand] = []

    def describe(self) -> str:
        if not self.commands:
            return "HeartbeatRet (no commands)"
        cmd_descs = [cmd.describe() for cmd in self.commands]
        return f"HeartbeatRet commands=[{', '.join(cmd_descs)}]"

    def has_commands(self) -> bool:
        """Check if there are commands to execute"""
        return len(self.commands) > 0


class BatchedP2PLookupRetMsg(WorkerReqRetMsg):
    """Batched P2P lookup return message"""

    # (instance_id, location, num_hit_chunks, peer_init_url)
    layout_info: list[tuple[str, str, int, str]]

    def describe(self) -> str:
        return f"The layout info is {self.layout_info}"


class FullSyncStartRetMsg(WorkerReqRetMsg):
    """Full sync start response message"""

    sync_id: str
    accepted: bool  # Whether sync request is accepted
    error_msg: Optional[str] = None

    def describe(self) -> str:
        return f"FullSyncStartRet sync_id={self.sync_id} accepted={self.accepted}"


class FullSyncStatusRetMsg(WorkerReqRetMsg):
    """Full sync status response message"""

    sync_id: str
    is_complete: bool  # Whether current worker sync is complete
    global_progress: float  # Global progress (0.0 - 1.0)
    can_exit_freeze: bool  # Whether freeze mode can be exited
    missing_batches: list[int] = []  # List of missing batch IDs that need to be resent

    def describe(self) -> str:
        missing_info = (
            f" missing={self.missing_batches}" if self.missing_batches else ""
        )
        return (
            f"FullSyncStatusRet sync_id={self.sync_id} "
            f"complete={self.is_complete} progress={self.global_progress:.2%} "
            f"can_exit_freeze={self.can_exit_freeze}{missing_info}"
        )


"""Control Message from Controller to LMCache"""


class ControlMsg(MsgBase):
    def describe(self) -> str:
        return ""


class ClearWorkerMsg(ControlMsg):
    """Clear message for a single lmcache worker"""

    worker_event_id: str
    location: str

    def describe(self) -> str:
        return f"Clear tokens in location {self.location}"


class PinWorkerMsg(ControlMsg):
    """Pin message for a single lmcache worker"""

    worker_event_id: str
    location: str
    tokens: list[int]

    def describe(self) -> str:
        return f"Pin tokens {self.tokens} in location {self.location}"


class CompressWorkerMsg(ControlMsg):
    """Compress message for a single lmcache worker"""

    worker_event_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class DecompressWorkerMsg(ControlMsg):
    """Decompress message for a single lmcache worker"""

    worker_event_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None

    def describe(self) -> str:
        return (
            f"Decompress tokens {self.tokens} in "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class MoveWorkerMsg(ControlMsg):
    """Move message for a single lmcache worker"""

    worker_event_id: str
    old_position: str  # location (storage backend name)
    new_position: Tuple[str, str]  # (target_url, location (storage backend name) )
    tokens: Optional[list[int]] = None
    copy: Optional[bool] = True

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )


class HealthWorkerMsg(ControlMsg):
    """Health message for a single lmcache worker"""

    worker_event_id: str

    def describe(self) -> str:
        return "Health check"


class CheckFinishWorkerMsg(ControlMsg):
    """Check finish message for a single lmcache worker"""

    worker_event_id: str

    def describe(self) -> str:
        return f"Checking finish for worker event {self.worker_event_id}"


class ControlRetMsg(MsgBase):
    """Return message from LMCache to Controller"""

    def describe(self) -> str:
        return ""


class ClearWorkerRetMsg(ControlRetMsg):
    """Return message for a ClearWorkerMsg"""

    num_tokens: int

    def describe(self) -> str:
        return f"Number of cleared tokens: {self.num_tokens}"


class PinWorkerRetMsg(ControlRetMsg):
    """Pin return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Number of pinned tokens: {self.num_tokens}"


class CompressWorkerRetMsg(ControlRetMsg):
    """Compress return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Compress success: {self.num_tokens}"


class DecompressWorkerRetMsg(ControlRetMsg):
    """Decompress return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Decompress success: {self.num_tokens}"


class MoveWorkerRetMsg(ControlRetMsg):
    """Move return message for a single lmcache worker"""

    num_tokens: int

    def describe(self) -> str:
        return f"Moving {self.num_tokens} tokens"


class HealthWorkerRetMsg(ControlRetMsg):
    """Health return message for a single lmcache worker"""

    error_code: int

    def describe(self) -> str:
        return f"Health check error code: {self.error_code}"


class CheckFinishWorkerRetMsg(ControlRetMsg):
    """Check finish return message for a single lmcache worker"""

    status: str

    def describe(self) -> str:
        return f"Check finish status: {self.status}"


"""Orchestration Message from Ochestrator to LMCache"""


class OrchMsg(MsgBase):
    """Message from Ochestrator to Controller"""

    def describe(self) -> str:
        return ""


class QueryInstMsg(OrchMsg):
    """Query instance message"""

    event_id: str
    ip: str

    def describe(self) -> str:
        return f"Query instance id of ip {self.ip}"


class LookupMsg(OrchMsg):
    """Lookup message"""

    event_id: str
    tokens: list[int]

    def describe(self) -> str:
        return f"Lookup tokens {self.tokens}"


class ClearMsg(OrchMsg):
    """Clear message"""

    event_id: str
    instance_id: str
    location: str

    def describe(self) -> str:
        return (
            f"Clear tokens in instance {self.instance_id} and locations {self.location}"
        )


class PinMsg(OrchMsg):
    """Pin message"""

    event_id: str
    instance_id: str
    location: str
    tokens: list[int]

    def describe(self) -> str:
        return (
            f"Pin tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"location {self.location}"
        )


class CompressMsg(OrchMsg):
    """Compress message"""

    event_id: str
    instance_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None  # `None` means compress all tokens

    def describe(self) -> str:
        return (
            f"Compress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class DecompressMsg(OrchMsg):
    """Decompress message"""

    event_id: str
    instance_id: str
    method: str
    location: str
    tokens: Optional[list[int]] = None  # `None` means compress all tokens

    def describe(self) -> str:
        return (
            f"Decompress tokens {self.tokens} in instance "
            f"{self.instance_id} and "
            f"locations {self.location} with "
            f"method {self.method}"
        )


class MoveMsg(OrchMsg):
    """Move message"""

    event_id: str
    old_position: Tuple[str, str]
    new_position: Tuple[str, str]
    tokens: Optional[list[int]] = None
    copy: Optional[bool] = False

    def describe(self) -> str:
        return (
            f"Move tokens {self.tokens} from {self.old_position} to {self.new_position}"
        )


class HealthMsg(OrchMsg):
    """Health message"""

    event_id: str
    instance_id: str

    def describe(self) -> str:
        return f"Health check for instance {self.instance_id}"


class CheckFinishMsg(OrchMsg):
    """Check finish message"""

    event_id: str

    def describe(self) -> str:
        return f"Checking finish for event {self.event_id}"


class QueryWorkerInfoMsg(OrchMsg):
    """Query worker info message"""

    event_id: str
    instance_id: str
    worker_ids: Optional[list[int]]

    def describe(self) -> str:
        return f"Query worker info of {self.instance_id} : {self.worker_ids}"


class OrchRetMsg(MsgBase):
    """Return message from Controller to Ochestrator"""

    def describe(self) -> str:
        return ""


class QueryInstRetMsg(OrchRetMsg):
    """Query instance return message"""

    event_id: str
    instance_id: Optional[str]

    def describe(self) -> str:
        return f"The instance id is {self.instance_id}"


class LookupRetMsg(OrchRetMsg):
    """Lookup return message"""

    event_id: str
    layout_info: Dict[str, Tuple[str, int]]

    def describe(self) -> str:
        return f"The layout info is {self.layout_info}"


class ClearRetMsg(OrchRetMsg):
    """Clear return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Number of cleared tokens: {self.num_tokens}"


class PinRetMsg(OrchRetMsg):
    """Pin return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Number of pinned tokens: {self.num_tokens}"


class CompressRetMsg(OrchRetMsg):
    """Compress return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Compressed {self.num_tokens} tokens"


class DecompressRetMsg(OrchRetMsg):
    """Decompress return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Decompressed {self.num_tokens} tokens"


class MoveRetMsg(OrchRetMsg):
    """Move return message"""

    event_id: str
    num_tokens: int

    def describe(self) -> str:
        return f"Moving {self.num_tokens} tokens"


class HealthRetMsg(OrchRetMsg):
    """Health return message"""

    event_id: str
    # worker_id -> error_code
    error_codes: Dict[int, int]

    def describe(self) -> str:
        return f"error_codes: {self.error_codes}"


class CheckFinishRetMsg(OrchRetMsg):
    """Check finish return message"""

    status: str

    def describe(self) -> str:
        return f"Event status: {self.status}"


class QueryWorkerInfoRetMsg(OrchRetMsg):
    """Query worker info return message"""

    event_id: str
    worker_infos: list[WorkerInfo]

    def describe(self) -> str:
        return f"worker infos: {self.worker_infos}"


class ErrorMsg(WorkerReqRetMsg):
    """Control Error Message"""

    error: str

    def describe(self) -> str:
        return f"Error: {self.error}"


Msg = Union[
    RegisterMsg,
    RegisterRetMsg,
    DeRegisterMsg,
    KVAdmitMsg,
    KVEvictMsg,
    BatchedKVOperationMsg,
    ClearWorkerMsg,
    ClearWorkerRetMsg,
    PinWorkerMsg,
    PinWorkerRetMsg,
    CompressWorkerMsg,
    CompressWorkerRetMsg,
    DecompressWorkerMsg,
    DecompressWorkerRetMsg,
    MoveWorkerMsg,
    MoveWorkerRetMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    CheckFinishWorkerMsg,
    CheckFinishWorkerRetMsg,
    LookupMsg,
    LookupRetMsg,
    ClearMsg,
    ClearRetMsg,
    PinMsg,
    PinRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    MoveMsg,
    MoveRetMsg,
    HealthMsg,
    HealthRetMsg,
    CheckFinishMsg,
    CheckFinishRetMsg,
    ErrorMsg,
    QueryInstMsg,
    QueryInstRetMsg,
    HeartbeatMsg,
    HeartbeatRetMsg,
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    QueryWorkerInfoMsg,
    QueryWorkerInfoRetMsg,
    FullSyncStartMsg,
    FullSyncStartRetMsg,
    FullSyncBatchMsg,
    FullSyncEndMsg,
    FullSyncStatusMsg,
    FullSyncStatusRetMsg,
]
