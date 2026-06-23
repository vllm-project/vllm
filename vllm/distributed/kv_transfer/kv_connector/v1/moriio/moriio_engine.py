# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
from collections import OrderedDict, defaultdict
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any
from weakref import ref as weakref_ref

import msgpack
import torch
import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import (
    make_zmq_path,
    make_zmq_socket,
)

if TYPE_CHECKING:
    from mori.io import BackendType

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    HandshakeError,
    LayerTransferPlan,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    MoRIIOError,
    RemoteAllocInfo,
    TransferError,
    TransferId,
    WriteTask,
    get_port_offset,
    get_role,
    zmq_ctx,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
        MoRIIOConnectorWorker,
    )

logger = init_logger(__name__)
try:
    from mori.io import (
        BackendType,
        EngineDesc,
        IOEngine,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig,
        XgmiBackendConfig,
    )

    logger.info("MoRIIO is available")
except ImportError:
    logger.error("MoRIIO is not available")


"""Write task execution logic for MoRIIO connector."""


_MAX_TERMINAL_TRANSFER_IDS = 4096


WriteGeometryKey = tuple[tuple[int, ...], tuple[int, ...], torch.dtype]


def _get_write_geometry_key(kv_cache: torch.Tensor) -> WriteGeometryKey:
    return (tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.dtype)


class MoRIIOWriter:
    """Handles write operations for KV cache transfers.

    WRITE mode state machine:
    D sends destination block allocation, P schedules one write per layer
    after the layer CUDA event, P seals the scheduled write count after
    forward, then P notifies D and releases P blocks after all scheduled
    writes complete.
    """

    def __init__(self, worker: "MoRIIOConnectorWorker"):
        """Initialize the writer.

        Args:
            worker: Reference to the parent worker
        """
        self._worker_ref: weakref_ref[MoRIIOConnectorWorker] = weakref_ref(worker)
        self._write_task_q: Queue[WriteTask] = Queue()
        self._write_worker_started = False
        self._write_worker_lock = threading.Lock()
        self._write_state_lock = threading.Lock()
        self._deferred_tasks: list[WriteTask] = []
        self._scheduled_writes: dict[TransferId, int] = defaultdict(int)
        self._scheduled_layers: dict[TransferId, set[str]] = defaultdict(set)
        self._sealed_writes: dict[TransferId, int] = {}
        self._defer_timeout = worker.moriio_config.defer_timeout

    @property
    def worker(self) -> "MoRIIOConnectorWorker":
        """Get the worker instance.

        Returns:
            The parent worker instance

        Raises:
            RuntimeError: If worker has been garbage collected
        """
        worker = self._worker_ref()
        if worker is None:
            raise RuntimeError("Parent worker has been garbage collected")
        return worker

    def ensure_worker_started(self) -> None:
        """Ensure the background write worker is running."""
        if self._write_worker_started:
            return
        self._write_worker_started = True
        with self._write_worker_lock:
            thread = threading.Thread(
                target=self._write_worker_loop, daemon=True, name="moriio-write-worker"
            )
            thread.start()
            logger.info("Started MoRIIO write worker thread")

    def schedule_write(self, task: WriteTask) -> bool:
        """Schedule a write task.

        Args:
            task: The write task to schedule
        """
        self.ensure_worker_started()
        if self._is_transfer_terminal(task.transfer_id):
            return False

        with self._write_state_lock:
            if self._is_transfer_terminal(task.transfer_id):
                return False
            if task.layer_name in self._scheduled_layers[task.transfer_id]:
                return False
            self._scheduled_layers[task.transfer_id].add(task.layer_name)
            self._scheduled_writes[task.transfer_id] += 1
        self._write_task_q.put(task)
        return True

    def is_scheduled(self, transfer_id: TransferId, layer_name: str) -> bool:
        with self._write_state_lock:
            return layer_name in self._scheduled_layers.get(transfer_id, set())

    def seal_pending_transfers(self) -> None:
        """Seal expected WRITE counts after the model forward has run.

        `save_kv_layer` is only invoked for attention layers whose backend uses
        the standard KV connector hook. Hybrid models can register more KV
        cache tensors than the number of hooks that fire in a forward, so WRITE
        completion must be based on the tasks actually queued for the transfer.
        """
        pending: list[tuple[TransferId, RemoteAllocInfo]] = []
        with self._write_state_lock:
            for transfer_id, write_count in self._scheduled_writes.items():
                if transfer_id in self._sealed_writes:
                    continue
                self._sealed_writes[transfer_id] = write_count
                request_info = (
                    self.worker.moriio_wrapper.done_remote_allocate_req_dict.get(
                        transfer_id
                    )
                )
                if request_info is not None:
                    request_info.writes_expected = write_count
                    pending.append((transfer_id, request_info))

        for transfer_id, request_info in pending:
            self._finalize_if_complete(transfer_id, request_info)

    def _write_worker_loop(self) -> None:
        """Main loop for the write worker thread."""

        while True:
            # Process deferred tasks first
            self._process_deferred_tasks()

            # Get new task
            try:
                task = self._write_task_q.get(timeout=0.01)
            except Empty:
                continue

            if self._is_transfer_terminal(task.transfer_id):
                continue

            # Check if remote blocks are ready
            if not self._is_remote_ready(task):
                # task.retry_count += 1
                self._deferred_tasks.append(task)
                # logger.debug(
                #     "Deferred task for request %s (retry %d)",
                #     task.request_id, task.retry_count
                # )
                continue

            # Execute the task
            try:
                self._execute_write_task(task)
            except Exception:
                logger.exception(
                    "Write task failed for request %s, marking done",
                    task.request_id,
                )
                self._mark_request_done(task.transfer_id)

    def _process_deferred_tasks(self) -> None:
        """Process tasks that were previously deferred."""
        if not self._deferred_tasks:
            return

        defer_timeout = self._defer_timeout
        now = time.perf_counter()
        still_deferred: list[WriteTask] = []

        for task in self._deferred_tasks:
            if self._is_transfer_terminal(task.transfer_id):
                continue
            if now - task.enqueue_time > defer_timeout:
                logger.error(
                    "Deferred write task for request %s expired after %.1fs "
                    "(remote blocks never arrived), marking done",
                    task.request_id,
                    now - task.enqueue_time,
                )
                self._mark_request_done(task.transfer_id)
                continue
            if self._is_remote_ready(task):
                try:
                    self._execute_write_task(task)
                except Exception:
                    logger.exception(
                        "Deferred write task failed for request %s, marking done",
                        task.request_id,
                    )
                    self._mark_request_done(task.transfer_id)
            else:
                still_deferred.append(task)

        self._deferred_tasks = still_deferred

    def _clear_transfer_state(self, transfer_id: TransferId) -> None:
        with self._write_state_lock:
            self._scheduled_writes.pop(transfer_id, None)
            self._scheduled_layers.pop(transfer_id, None)
            self._sealed_writes.pop(transfer_id, None)

    def _is_transfer_terminal(self, transfer_id: TransferId) -> bool:
        wrapper = self.worker.moriio_wrapper
        with wrapper.lock:
            return wrapper._is_transfer_terminal_locked(transfer_id)

    def _mark_request_done(self, transfer_id: str) -> None:
        """Mark a request done so its blocks are freed, even on transfer failure."""
        wrapper = self.worker.moriio_wrapper
        with wrapper.lock:
            wrapper.done_req_ids.append(transfer_id)
            wrapper.done_remote_allocate_req_dict.pop(transfer_id, None)
            wrapper._mark_transfer_terminal_locked(transfer_id)
        self._clear_transfer_state(transfer_id)

    def _is_remote_ready(self, task: WriteTask) -> bool:
        """Check if remote blocks are allocated for this task.

        Args:
            task: The write task

        Returns:
            True if remote blocks are ready
        """
        return (
            task.transfer_id in self.worker.moriio_wrapper.done_remote_allocate_req_dict
        )

    def _get_remote_alloc_info(self, transfer_id: str) -> RemoteAllocInfo:
        """Get remote allocation info for a request.

        Args:
            transfer_id:TransferId The request ID

        Returns:
            Remote allocation information

        Raises:
            KeyError: If allocation info is missing
        """
        try:
            return self.worker.moriio_wrapper.done_remote_allocate_req_dict[transfer_id]
        except KeyError as e:
            raise KeyError(
                f"Remote allocation info missing for transfer {transfer_id}"
            ) from e

    def _execute_write_task(self, task: WriteTask) -> None:
        """Execute a single write task.

        Args:
            task: The write task to execute

        """
        # Get remote allocation info
        request_info = self._get_remote_alloc_info(task.transfer_id)
        with self._write_state_lock:
            request_info.completion_request_id = task.request_id
            request_info.completion_remote_notify_port = task.remote_notify_port
            request_info.completion_remote_ip = task.remote_ip
            if task.transfer_id in self._sealed_writes:
                request_info.writes_expected = self._sealed_writes[task.transfer_id]

        if request_info.block_ids is None:
            logger.debug(
                "Request remote block IDs not ready:request_id = %s, transfer_id = %s",
                task.request_id,
                task.transfer_id,
            )
            return

        # Wait for CUDA event
        # The attention computation of the current layer cannot
        # overlap with the kv transfer task,
        # otherwise it will cause precision issues.
        # This event is used to synchronize the kv transfer and computation tasks.
        task.event.synchronize()

        # Update engine ID with DP rank
        task.dst_engine_id = self.worker.get_engine_name_with_dp(
            task.dst_engine_id, request_info.decode_dp_rank
        )

        # Get or create sessions
        sessions, remote_moriio_meta = self.worker._get_built_session(
            task.dst_engine_id
        )

        # Prepare transfer plan
        plan = self._prepare_transfer_plan(task, request_info, remote_moriio_meta)

        # Execute transfer
        transfer_statuses = self._do_layer_write(plan, sessions)
        with self._write_state_lock:
            request_info.transfer_statuses.extend(transfer_statuses)

        # Finalize if all layers complete
        self._mark_write_done(task.transfer_id, request_info)

    def _prepare_transfer_plan(
        self,
        task: WriteTask,
        request_info: RemoteAllocInfo,
        remote_moriio_meta: MoRIIOAgentMetadata,
    ) -> LayerTransferPlan:
        """Prepare the transfer plan for a layer.

        Args:
            task: The write task
            request_info: Remote allocation information

        Returns:
            The transfer plan
        """
        layer_cache = self.worker.kv_caches[task.layer_name]
        geometry_key = _get_write_geometry_key(layer_cache)
        offsets = request_info.transfer_offsets.get(geometry_key)
        if offsets is None:
            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                task.local_block_ids,
                request_info.block_ids,
                remote_moriio_meta,
            )
            request_info.transfer_offsets[geometry_key] = offsets

        # Get session index
        layer_names = list(self.worker.layer_name_to_local_kv_cache_metadata.keys())
        sess_idx = layer_names.index(task.layer_name)

        local_off, remote_off, sizes = offsets

        return LayerTransferPlan(
            request_id=task.request_id,
            transfer_id=task.transfer_id,
            layer_name=task.layer_name,
            sess_idx=sess_idx,
            transfer_local_offsets=local_off,
            transfer_remote_offsets=remote_off,
            transfer_sizes=sizes,
            use_batch=True,
        )

    def _do_layer_write(self, plan: LayerTransferPlan, sessions: list) -> list[Any]:
        """Perform the actual layer write.

        Args:
            plan: The transfer plan
            sessions: List of transfer sessions
        """
        if plan.use_batch:
            return [
                self.worker.moriio_wrapper.write_remote_data(
                    plan.transfer_sizes,
                    plan.transfer_local_offsets,
                    plan.transfer_remote_offsets,
                    sessions[plan.sess_idx],
                )
            ]

        transfer_statuses: list[Any] = []
        for i in range(len(plan.transfer_local_offsets)):
            transfer_statuses.append(
                self.worker.moriio_wrapper.write_remote_data_single(
                    plan.transfer_sizes[i],
                    plan.transfer_local_offsets[i],
                    plan.transfer_remote_offsets[i],
                    plan.sess_idx,
                )
            )
        return transfer_statuses

    def _mark_write_done(
        self, transfer_id: TransferId, request_info: RemoteAllocInfo
    ) -> None:
        """Record one completed WRITE task and finalize if sealed."""
        with self._write_state_lock:
            request_info.writes_done += 1
        self._finalize_if_complete(transfer_id, request_info)

    def _finalize_if_complete(
        self, transfer_id: TransferId, request_info: RemoteAllocInfo
    ) -> None:
        """Finalize transfer if all scheduled writes are complete."""
        with self._write_state_lock:
            expected = request_info.writes_expected
            if expected is None or request_info.writes_done < expected:
                return
            if request_info.completion_notified:
                return
            request_id = request_info.completion_request_id
            remote_notify_port = request_info.completion_remote_notify_port
            remote_ip = request_info.completion_remote_ip
            if request_id is None or remote_notify_port is None or remote_ip is None:
                return
            transfer_statuses = list(request_info.transfer_statuses)
            request_info.transfer_statuses.clear()
            request_info.completion_notified = True

        # Wait for this request's transfers to complete.
        self.worker.moriio_wrapper.waiting_for_transfer_complete(transfer_statuses)

        remote_port = remote_notify_port + get_port_offset(
            request_info.decode_dp_rank, self.worker.tp_rank
        )
        # Consider using RDMA immediate data in decode side
        # to eliminate the need for this notification.
        # Consider including the first gen token from prefill in the notification

        # Send completion notification
        self.worker.moriio_wrapper.send_notify(
            transfer_id, remote_ip, remote_port, message_type="write_done"
        )
        # mark request as done, then we can free the blocks
        with self.worker.moriio_wrapper.lock:
            self.worker.moriio_wrapper.done_req_ids.append(transfer_id)
            self.worker.moriio_wrapper.done_remote_allocate_req_dict.pop(
                transfer_id, None
            )
            self.worker.moriio_wrapper._mark_transfer_terminal_locked(transfer_id)
        self._clear_transfer_state(transfer_id)
        logger.debug(
            "Completed transfer for (request, transfer) %s, %s, notified port %d",
            request_id,
            transfer_id,
            remote_port,
        )


class MoRIIOWrapper:
    """Wrapper for MoRIIO engine operations.

    Handles both producer and consumer roles for KV cache transfers.

    Args:
        moriio_engine:  MoRIIO engine instance
        tp_rank: Tensor parallel rank
        dp_rank: Data parallel rank
    """

    def __init__(
        self,
        moriio_engine: "IOEngine | None" = None,
        tp_rank: int = 0,
        dp_rank: int = 0,
        transfer_timeout: float = MoRIIOConstants.DEFAULT_TRANSFER_TIMEOUT,
    ):
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moriio_engine = moriio_engine
        self.remote_memory_metadata = None
        self.local_memory_registered = False
        self.local_memory_metadata = None
        self.transfer_status: list[Any] = []
        self.remote_engine_ip: str | None = None
        self.notify_port: int | None = None
        self.lock = threading.Lock()
        self.done_req_ids: list[str] = []
        self.done_remote_allocate_req_dict: dict[TransferId, RemoteAllocInfo] = {}
        self.done_write_cache_req_ids: list[str] = []
        self._terminal_transfer_ids: OrderedDict[TransferId, None] = OrderedDict()
        self._transfer_timeout = transfer_timeout
        self.notify_thread: threading.Thread | None = None
        self.sessions: list[IOEngine.Session] = []
        self.paths: dict[str, zmq.Socket] = {}

    def set_moriio_engine(self, moriio_engine):
        assert moriio_engine is not None, (
            "You Cannot pass None engine to MoRIIOWrapper!"
        )
        self.moriio_engine = moriio_engine

    def set_backend_type(
        self,
        backend_type: "BackendType",
        qp_per_transfer: int = 1,
        post_batch_size: int = -1,
        num_workers: int = 1,
    ) -> None:
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        if backend_type == BackendType.XGMI:
            logger.info("Using MoRIIO backend: XGMI")
            self.moriio_engine.create_backend(backend_type, XgmiBackendConfig())
        else:
            logger.info(
                "Using MoRIIO backend: RDMA "
                "(qp_per_transfer=%d, post_batch_size=%d, num_workers=%d)",
                qp_per_transfer,
                post_batch_size,
                num_workers,
            )
            rdma_cfg = RdmaBackendConfig(
                qp_per_transfer,
                post_batch_size,
                num_workers,
                PollCqMode.POLLING,
                # vLLM uses ZMQ for completion signaling
                # and never calls PopInboundTransferStatus.
                # With notifications enabled, ibv_post_send
                # ENOMEM at high concurrency permanently
                # poisons TransferStatus before the data WR
                # completes, hanging requests in
                # WAITING_FOR_REMOTE_KVS indefinitely.
                enable_notification=False,
            )
            self.moriio_engine.create_backend(backend_type, rdma_cfg)

    def get_agent_metadata(self):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        engine_metadata = self.moriio_engine.get_engine_desc()
        engine_metadata_packed = engine_metadata.pack()
        return engine_metadata_packed

    def register_remote_engine(self, remote_packed_engine_metadata):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        consumer_engine_metadata = EngineDesc.unpack(remote_packed_engine_metadata)
        self.moriio_engine.register_remote_engine(consumer_engine_metadata)
        return consumer_engine_metadata.key

    def register_local_tensor(self, tensor: torch.Tensor):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        try:
            self.local_memory_metadata = self.moriio_engine.register_torch_tensor(
                tensor
            )
            assert self.local_memory_metadata is not None, (
                "register_torch_tensor returned None"
            )
            local_memory_metadata_packed = self.local_memory_metadata.pack()
        except Exception as e:
            raise MoRIIOError(f"Failed to register local memory: {e}") from e
        self.local_memory_registered = True
        return local_memory_metadata_packed

    def get_unpack_memory_metadata(self, packed_memory_metadata):
        return MemoryDesc.unpack(packed_memory_metadata)

    def build_session(self, local_memory_metadata, remote_memory_metadata):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        return self.moriio_engine.create_session(
            local_memory_metadata, remote_memory_metadata
        )

    def read_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        transfer_status = session.batch_read(
            local_offset,
            remote_offset,
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid(),
        )

        return transfer_status

    def write_remote_data(
        self, transfer_size_byte, local_offset=0, remote_offset=0, session=None
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        write_uid = self.moriio_engine.allocate_transfer_uid()

        transfer_status = session.batch_write(
            local_offset, remote_offset, transfer_size_byte, write_uid
        )
        return transfer_status

    def write_remote_data_single(
        self, transfer_size_byte, local_offset=0, remote_offset=0, sess_idx=0
    ):
        assert self.local_memory_registered, "You have not register local memory data!"
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        transfer_status = self.sessions[sess_idx].write(
            local_offset,
            remote_offset,
            transfer_size_byte,
            self.moriio_engine.allocate_transfer_uid(),
        )
        return transfer_status

    def waiting_for_transfer_complete(self, transfer_statuses: list[Any] | None = None):
        if transfer_statuses is None:
            with self.lock:
                transfers_to_wait = self.transfer_status[:]
                self.transfer_status.clear()
        else:
            transfers_to_wait = list(transfer_statuses)

        if not transfers_to_wait:
            return

        timeout = self._transfer_timeout
        deadline = time.monotonic() + timeout
        remaining = list(transfers_to_wait)
        errors: list[str] = []

        while remaining:
            timed_out = time.monotonic() > deadline
            still_waiting = []
            for status in remaining:
                if status.Succeeded():
                    continue
                if status.Failed():
                    errors.append(
                        f"RDMA transfer failed: {status.Message()} "
                        f"(code={status.Code()})"
                    )
                    continue
                if timed_out:
                    errors.append(
                        f"RDMA transfer timed out after {timeout:.0f}s. "
                        f"Check NIC queue depth or reduce concurrency; "
                        f"adjust with kv_connector_extra_config.transfer_timeout."
                    )
                else:
                    still_waiting.append(status)
            remaining = still_waiting
            if remaining:
                time.sleep(0.001)
        if errors:
            raise TransferError(
                f"{len(errors)}/{len(transfers_to_wait)} transfers failed:\n"
                + "\n".join(errors)
            )

    def async_wait_reqid(self):
        assert self.notify_port is not None, "Notify port cannot be None"

        if self.notify_thread is not None:
            return

        def _async_wait():
            host = "*"
            path = make_zmq_path("tcp", host, self.notify_port)
            logger.info("Node starting to listen notify from path = %s", path)

            with zmq_ctx(zmq.ROUTER, path) as sock:
                while True:
                    try:
                        identity, msg = sock.recv_multipart()
                        self._handle_message(msg)
                    except Exception as e:
                        logger.error("Error processing message: %s", e)
                        raise HandshakeError(f"Error processing message: {e}") from e

        self.notify_thread = threading.Thread(
            target=_async_wait, daemon=True, name="moriio-notify-listener"
        )
        self.notify_thread.start()

    def _handle_message(self, msg: bytes):
        """Handles incoming messages from remote nodes."""
        # Handles incoming remote messages:
        # Prefill Role:
        #   [write] mode: receives block information (allocation)
        #   [read]  mode: receives block release messages from decode side
        # Decode Role:
        #   [write] mode: receives KV cache write completion notifications
        msg_str = repr(msg)
        handled = False
        try:
            data = msgpack.loads(msg)
            if isinstance(data, dict):
                self._handle_structured_message(data)
                return
        except (
            msgpack.exceptions.ExtraData,
            msgpack.exceptions.UnpackException,
            ValueError,
        ):
            logger.debug("Failed to decode msgpack message, will try as string")
            pass

        try:
            msg_str = msg.decode("UTF-8")
            if msg_str:
                self._handle_completion_message(msg_str)
                handled = True
        except UnicodeDecodeError:
            logger.warning("Received non-UTF8 message: %r", msg)
        if not handled:
            raise MoRIIOError(f"Unhandled message format: {msg_str}")

    def _handle_structured_message(self, data: dict):
        message_type = data.get("type")
        if message_type is None and "req_id" in data:
            message_type = "remote_blocks"

        if message_type == "remote_blocks":
            self._handle_remote_blocks_message(data)
        elif message_type == "write_done":
            self._handle_write_done_message(data)
        elif message_type == "release":
            self._handle_release_message(data)
        else:
            raise MoRIIOError(f"Unhandled structured message type: {message_type}")

    def _handle_remote_blocks_message(self, data: dict):
        assert get_role() == ROLE.PRODUCER, "Only prefill can get block messages"
        transfer_id = data["transfer_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)
        if not block_notify_list:
            raise MoRIIOError(
                "block_notify_list cannot be empty in remote allocate message"
            )

        with self.lock:
            if self._is_transfer_terminal_locked(transfer_id):
                logger.debug(
                    "Ignoring remote allocation for terminal transfer %s",
                    transfer_id,
                )
                return
            self.done_remote_allocate_req_dict[transfer_id] = RemoteAllocInfo(
                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank
            )

    def _handle_write_done_message(self, data: dict):
        assert get_role() != ROLE.PRODUCER, (
            "Only decode can get WRITE completion messages"
        )
        transfer_id = data["transfer_id"]
        with self.lock:
            self.done_write_cache_req_ids.append(transfer_id)

    def _handle_release_message(self, data: dict):
        assert get_role() == ROLE.PRODUCER, (
            "Only prefill can get transfer release messages"
        )
        transfer_id = data["transfer_id"]
        with self.lock:
            self.done_req_ids.append(transfer_id)
            self.done_remote_allocate_req_dict.pop(transfer_id, None)
            self._mark_transfer_terminal_locked(transfer_id)

    def _handle_completion_message(self, msg: str):
        with self.lock:
            if get_role() == ROLE.PRODUCER:
                self.done_req_ids.append(msg)
                self.done_remote_allocate_req_dict.pop(msg, None)
                self._mark_transfer_terminal_locked(msg)
            else:
                self.done_write_cache_req_ids.append(msg)

    def _is_transfer_terminal_locked(self, transfer_id: TransferId) -> bool:
        return transfer_id in self._terminal_transfer_ids

    def _mark_transfer_terminal_locked(self, transfer_id: TransferId) -> None:
        self._terminal_transfer_ids[transfer_id] = None
        self._terminal_transfer_ids.move_to_end(transfer_id)
        while len(self._terminal_transfer_ids) > _MAX_TERMINAL_TRANSFER_IDS:
            self._terminal_transfer_ids.popitem(last=False)

    def send_notify(
        self, req_ids, remote_ip, remote_port, message_type: str | None = None
    ):
        if not remote_ip or not remote_port:
            logger.warning("Missing remote_ip or remote_port for notification")
            return

        path = make_zmq_path("tcp", remote_ip, remote_port)

        if path not in self.paths:
            ctx = zmq.Context.instance()
            sock = make_zmq_socket(
                ctx=ctx, path=path, socket_type=zmq.DEALER, bind=False
            )
            self.paths[path] = sock

        req_list = req_ids if isinstance(req_ids, list) else [req_ids]

        sock = self.paths[path]
        try:
            for req_id in req_list:
                if not isinstance(req_id, str):
                    logger.warning(
                        "Invalid req_id type: %s, expected str", type(req_id)
                    )
                    continue
                if message_type is None:
                    sock.send(req_id.encode("utf-8"))
                else:
                    sock.send(
                        msgpack.dumps({"type": message_type, "transfer_id": req_id})
                    )
        except Exception as e:
            logger.error("Failed to send notification to %s: %s", path, e)
            self.paths.pop(path, None)
            raise

    def pop_finished_req_ids(self):
        # producer invocation: get the set of completed requests at the decode
        with self.lock:
            done_send = set(self.done_req_ids)
            self.done_req_ids = []
        return done_send

    def pop_finished_write_req_ids(self):
        # Call the consumer in write mode to get the collection after write completion
        with self.lock:
            done_write_cache = set(self.done_write_cache_req_ids)
            self.done_write_cache_req_ids = []
        return done_write_cache

    def shutdown(self):
        logger.debug("Closing MoRIIOWrapper and cleaning up ZMQ sockets")
        for path, sock in self.paths.items():
            try:
                sock.close(linger=0)
                logger.debug("Closed ZMQ socket for path: %s", path)
            except Exception as e:
                logger.warning("Error closing ZMQ socket for path %s: %s", path, e)
        self.paths.clear()
