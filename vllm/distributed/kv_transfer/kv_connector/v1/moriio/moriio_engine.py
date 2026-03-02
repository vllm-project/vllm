# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import TYPE_CHECKING, Any
from weakref import ref as weakref_ref

import msgpack
import torch
import zmq

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.network_utils import (
    make_zmq_path,
    make_zmq_socket,
)

if TYPE_CHECKING:
    pass

from queue import Empty, Queue

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    HandshakeError,
    LayerTransferPlan,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    MoRIIOError,
    RemoteAllocInfo,
    TransferError,
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
        EngineDesc,
        IOEngine,
        MemoryDesc,
        PollCqMode,
        RdmaBackendConfig,
    )

    logger.info("MoRIIO is available")
except ImportError:
    logger.error("MoRIIO is not available")


"""Write task execution logic for MoRIIO connector."""


class MoRIIOWriter:
    """Handles write operations for KV cache transfers.
    Implements distributed KV cache transfer using the MoRIIO library
    for RDMA-based communication between prefill and decode instances."""

    def __init__(self, worker: "MoRIIOConnectorWorker"):
        """Initialize the writer.

        Args:
            worker: Reference to the parent worker
        """
        self._worker_ref: weakref_ref[MoRIIOConnectorWorker] = weakref_ref(worker)
        self._write_task_q: Queue[WriteTask] = Queue()
        self._write_worker_started = False
        self._write_worker_lock = threading.Lock()
        self._deferred_tasks: list[WriteTask] = []

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

    def schedule_write(self, task: WriteTask) -> None:
        """Schedule a write task.

        Args:
            task: The write task to schedule
        """
        self.ensure_worker_started()
        self._write_task_q.put(task)

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

            self._execute_write_task(task)

    def _process_deferred_tasks(self) -> None:
        """Process tasks that were previously deferred."""
        if not self._deferred_tasks:
            return

        still_deferred: list[WriteTask] = []
        for task in self._deferred_tasks:
            if self._is_remote_ready(task):
                self._execute_write_task(task)
            else:
                still_deferred.append(task)

        self._deferred_tasks = still_deferred

    def _is_remote_ready(self, task: WriteTask) -> bool:
        """Check if remote blocks are allocated for this task.

        Args:
            task: The write task

        Returns:
            True if remote blocks are ready
        """
        return (
            task.request_id in self.worker.moriio_wrapper.done_remote_allocate_req_dict
        )

    def _get_remote_alloc_info(self, request_id: str) -> RemoteAllocInfo:
        """Get remote allocation info for a request.

        Args:
            request_id: The request ID

        Returns:
            Remote allocation information

        Raises:
            KeyError: If allocation info is missing
        """
        try:
            return self.worker.moriio_wrapper.done_remote_allocate_req_dict[request_id]
        except KeyError as e:
            raise KeyError(
                f"Remote allocation info missing for request {request_id}"
            ) from e

    def _execute_write_task(self, task: WriteTask) -> None:
        """Execute a single write task.

        Args:
            task: The write task to execute

        """
        # Get remote allocation info
        request_info = self._get_remote_alloc_info(task.request_id)

        if request_info.block_ids is None:
            logger.debug("Request %s remote block IDs not ready", task.request_id)
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
        self._do_layer_write(plan, sessions)

        # Finalize if all layers complete
        self._finalize_if_complete(task, request_info)

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
        # Compute offsets if not cached
        if request_info.transfer_offset is None:
            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                task.local_block_ids,
                request_info.block_ids,
                remote_moriio_meta,
            )
            request_info.transfer_offset = offsets

        # Get session index
        layer_names = list(self.worker.layer_name_to_local_kv_cache_metadata.keys())
        sess_idx = layer_names.index(task.layer_name)

        local_off, remote_off, sizes = request_info.transfer_offset

        return LayerTransferPlan(
            request_id=task.request_id,
            layer_name=task.layer_name,
            sess_idx=sess_idx,
            transfer_local_offsets=local_off,
            transfer_remote_offsets=remote_off,
            transfer_sizes=sizes,
            use_batch=True,
        )

    def _do_layer_write(self, plan: LayerTransferPlan, sessions: list) -> None:
        """Perform the actual layer write.

        Args:
            plan: The transfer plan
            sessions: List of transfer sessions
        """
        if plan.use_batch:
            self.worker.moriio_wrapper.write_remote_data(
                plan.transfer_sizes,
                plan.transfer_local_offsets,
                plan.transfer_remote_offsets,
                sessions[plan.sess_idx],
            )
        else:
            for i in range(len(plan.transfer_local_offsets)):
                self.worker.moriio_wrapper.write_remote_data_single(
                    plan.transfer_sizes[i],
                    plan.transfer_local_offsets[i],
                    plan.transfer_remote_offsets[i],
                    plan.sess_idx,
                )

    def _finalize_if_complete(
        self, task: WriteTask, request_info: RemoteAllocInfo
    ) -> None:
        """Finalize transfer if all layers are complete.

        Args:
            task: The write task
            request_info: Remote allocation information
        """
        request_info.writes_done += 1

        if request_info.writes_done >= self.worker.num_layers:
            # Wait for transfer to complete
            self.worker.moriio_wrapper.waiting_for_transfer_complete()

            remote_port = task.remote_notify_port + get_port_offset(
                request_info.decode_dp_rank, self.worker.tp_rank
            )
            # Consider using RDMA immediate data in decode side
            # to eliminate the need for this notification.
            # Consider including the first gen token from prefill in the notification

            # Send completion notification
            self.worker.moriio_wrapper.send_notify(
                task.request_id, task.remote_ip, remote_port
            )
            # mark request as done, then we can free the blocks
            with self.worker.moriio_wrapper.lock:
                self.worker.moriio_wrapper.done_req_ids.append(task.request_id)
            del self.worker.moriio_wrapper.done_remote_allocate_req_dict[
                task.request_id
            ]
            logger.debug(
                "Completed transfer for request %s, notified port %d",
                task.request_id,
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
        self.done_remote_allocate_req_dict: dict[str, RemoteAllocInfo] = {}
        self.done_write_cache_req_ids: list[str] = []
        self.notify_thread: threading.Thread | None = None
        self.sessions: list[IOEngine.Session] = []
        self.paths: dict[str, zmq.Socket] = {}

    def set_moriio_engine(self, moriio_engine):
        assert moriio_engine is not None, (
            "You Cannot pass None engine to MoRIIOWrapper!"
        )
        self.moriio_engine = moriio_engine

    def set_backend_type(self, backend_type):
        assert self.moriio_engine is not None, "MoRIIO engine must be set first"
        qp_per_transfer = envs.VLLM_MORIIO_QP_PER_TRANSFER
        post_batch_size = envs.VLLM_MORIIO_POST_BATCH_SIZE
        num_worker_threads = envs.VLLM_MORIIO_NUM_WORKERS
        poll_mode = PollCqMode.POLLING
        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
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
        with self.lock:
            self.transfer_status.append(transfer_status)

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
        with self.lock:
            self.transfer_status.append(transfer_status)

    def waiting_for_transfer_complete(self):
        if not self.transfer_status:
            return

        transfers_to_wait = []
        with self.lock:
            transfers_to_wait = self.transfer_status[:]
            self.transfer_status.clear()

        for status in transfers_to_wait:
            try:
                status.Wait()
                if not status.Succeeded():
                    logger.error(
                        "Transfer failed: %s, Code: %s", status.Message(), status.Code()
                    )
                    raise TransferError("MoRIIO transfer failed!")
            except Exception as e:
                logger.error("Transfer %s failed: %s", status, e)
                raise

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
        handled = False
        try:
            data = msgpack.loads(msg)
            if isinstance(data, dict) and "req_id" in data:
                self._handle_structured_message(data)

                return
        except (msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackException):
            logger.debug("Failed to decode msgpack message, will try as string")
            pass

        try:
            msg_str = msg.decode("UTF-8")
            if msg_str.startswith(MoRIIOConstants.COMPLETION_PREFIX):
                self._handle_completion_message(msg_str)
                handled = True
        except UnicodeDecodeError:
            logger.warning("Received non-UTF8 message: %s", msg_str)
        if not handled:
            raise MoRIIOError(f"Unhandled message format: {msg_str}")

    def _handle_structured_message(self, data: dict):
        assert get_role() == ROLE.PRODUCER, "Only prefill can get block messages"
        req_id = data["req_id"]
        block_notify_list = data.get("block_notify_list", [])
        decode_dp_rank = data.get("decode_rank", 0)
        assert len(block_notify_list) > 0, (
            "block_notify_list cannot be empty in remote allocate message"
        )

        with self.lock:
            self.done_remote_allocate_req_dict[req_id] = RemoteAllocInfo(
                block_ids=block_notify_list, decode_dp_rank=decode_dp_rank
            )

    def _handle_completion_message(self, msg: str):
        with self.lock:
            if get_role() == ROLE.PRODUCER:
                self.done_req_ids.append(msg)
            else:
                self.done_write_cache_req_ids.append(msg)

    def send_notify(self, req_ids, remote_ip, remote_port):
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
                sock.send(req_id.encode("utf-8"))
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
