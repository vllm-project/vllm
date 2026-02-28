# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import traceback
import uuid
from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import cast

import msgspec.msgpack
import torch
import zmq

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    GroupCoordinator,
    cleanup_dist_env_and_memory,
    get_all_model_groups,
    get_pp_group,
    get_tp_group,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.logger import init_logger
from vllm.utils.collection_utils import ThreadSafeDict
from vllm.utils.network_utils import close_sockets, get_open_port, make_zmq_socket
from vllm.v1.engine import (
    EngineCoreRequestType,
    EngineStatusType,
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.engine.exceptions import EngineLoopPausedError, FaultInfo
from vllm.v1.serial_utils import run_method
from vllm.v1.utils import get_engine_client_zmq_addr

logger = init_logger(__name__)
# Polling timeout in milliseconds for non-blocking message reception
POLL_TIMEOUT_MS = 100


class BaseSentinel:
    """
    Core functionalities of the sentinel covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        upstream_cmd_addr: str | None,
        downstream_cmd_addr: str | None,
        sentinel_identity: bytes | None,
        sentinel_tag: str | None,
    ):
        self.sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_tag = sentinel_tag
        self.logger = self._make_logger()
        self.vllm_config = vllm_config
        self.ft_config = vllm_config.fault_tolerance_config
        self.upstream_cmd_socket = None
        self.downstream_cmd_socket = None
        if upstream_cmd_addr is not None:
            assert sentinel_identity is not None
            self.upstream_cmd_socket = make_zmq_socket(
                self.ctx,
                upstream_cmd_addr,
                zmq.DEALER,
                bind=False,
                identity=sentinel_identity,
            )
        if downstream_cmd_addr is not None:
            self.downstream_cmd_socket = make_zmq_socket(
                ctx=self.ctx,
                path=downstream_cmd_addr,
                socket_type=zmq.ROUTER,
                bind=True,
            )

    def _make_logger(self):
        def log(msg, *args, level="info", **kwargs):
            """
            msg: log message
            """
            prefix = self.sentinel_name
            getattr(logger, level)(prefix + msg, *args, **kwargs)

        return log

    @property
    def sentinel_name(self) -> str:
        if self.sentinel_tag is None:
            return f"[{self.__class__.__name__}] "
        return f"[{self.__class__.__name__}_{self.sentinel_tag}] "

    @abstractmethod
    def run(self) -> None:
        """
        The run() method is launched as a separate thread when a Sentinel
        instance is created.

        This background thread typically runs persistently to ensure real-time
        detection of errors and timely reception of fault tolerance instructions
        from upstream sentinels.
        """
        raise NotImplementedError

    def poll_and_execute_upstream_cmd(self):
        """
        Receive and execute a command from upstream sentinel and send back
        the execution result.
        """
        poll_timeout_ms = 100
        assert self.upstream_cmd_socket is not None
        try:
            # Polls the upstream command socket
            poller = zmq.Poller()
            poller.register(self.upstream_cmd_socket, zmq.POLLIN)
            events = poller.poll(timeout=poll_timeout_ms)
            if events:
                _, ft_request_bytes = self.upstream_cmd_socket.recv_multipart()
                ft_request = msgspec.msgpack.decode(
                    ft_request_bytes, type=FaultToleranceRequest
                )
                ft_result = self._execute_cmd(ft_request)
                msg_bytes = msgspec.msgpack.encode(ft_result)
                self.upstream_cmd_socket.send_multipart([b"", msg_bytes])
        except zmq.ZMQError:
            self.logger(
                "Socket closed, terminating %s", self.sentinel_name, level="info"
            )
            self.sentinel_dead = True

    def _execute_cmd(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        method = ft_request.instruction
        self.logger("Executing command: %s", ft_request, level="info")
        try:
            success: bool = run_method(self, method, args=(), kwargs=ft_request.params)
            self.logger("Command (%s) succeeded: %s", method, success, level="info")
            reason = None
        except Exception as e:
            self.logger(
                "Error executing ft request: %s",
                ft_request,
                level="error",
            )
            success = False
            reason = f"{type(e).__name__}: {e}"
        return FaultToleranceResult(
            success=success, request_id=ft_request.request_id, reason=reason
        )

    @abstractmethod
    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """
        Pause the vLLM instance to enter fault-tolerance mode.
        This method should be called when a fault is detected. It pauses the
        execution, allowing the system to wait for fault-tolerance instructions
        (e.g., retry, scale-down, or other control commands).
        """
        raise NotImplementedError

    @abstractmethod
    def retry(self, timeout: int = 1, **kwargs) -> bool:
        """
        Retry execution after a transient recoverable fault.
        """
        raise NotImplementedError

    def _execute_command_on_downstreams(
        self,
        method_name: str,
        target_downstream_sentinels: set[bytes],
        timeout: int = 5,
        **kwargs,
    ) -> tuple[bool, dict[bytes, FaultToleranceResult]]:
        """
        Broadcast a command to downstream sentinels and collect responses.
        """
        assert self.downstream_cmd_socket is not None
        # Create fault tolerance request
        kwargs["timeout"] = timeout
        ft_request = FaultToleranceRequest(
            request_id=str(uuid.uuid4()),
            instruction=method_name,
            params=kwargs,
        )
        # Broadcast the instruction
        msg_bytes = msgspec.msgpack.encode(ft_request)
        for identity in target_downstream_sentinels:
            self.downstream_cmd_socket.send_multipart([identity, b"", msg_bytes])
        # Wait for responses
        responses = self._wait_for_execution_result(
            target_downstream_sentinels,
            timeout,
            ft_request,
        )
        # check the execution results
        for sentinel_identity in target_downstream_sentinels:
            response = responses.get(sentinel_identity)
            if response is None:
                self.logger(
                    'Downstream sentinels timed out on "%s".',
                    method_name,
                    level="error",
                )
                return False, responses
            elif not response.success:
                self.logger(
                    'Downstream sentinels failed to "%s" (reason: %s)',
                    method_name,
                    response.reason or "unknown",
                    level="error",
                )
                return False, responses

        return True, responses

    def _wait_for_execution_result(
        self,
        target_identities: set[bytes] | list[bytes],
        timeout: int,
        ft_request: "FaultToleranceRequest",
    ) -> dict[bytes, "FaultToleranceResult"]:
        """Collect responses for the given request.
        Returns partial results on timeout or error.
        """
        assert self.downstream_cmd_socket is not None
        deadline = time.monotonic() + timeout
        responses: dict[bytes, FaultToleranceResult] = {}
        pending = set(target_identities)
        poller = zmq.Poller()
        poller.register(self.downstream_cmd_socket, zmq.POLLIN)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                events = poller.poll(int(remaining * 1000))
                if not events:
                    break
                identity, _, payload = self.downstream_cmd_socket.recv_multipart()
                result = msgspec.msgpack.decode(payload, type=FaultToleranceResult)
                # Ignore unrelated responses
                if result.request_id != ft_request.request_id:
                    logger.debug("Discarding outdated response: %s", result)
                    continue

                responses[identity] = result
                pending.discard(identity)
            except Exception as e:
                logger.error("Error while processing engine response: %s", e)
                break

        return responses

    def shutdown(self):
        close_sockets([self.upstream_cmd_socket, self.downstream_cmd_socket])
        self.ctx.term()
        self.sentinel_dead = True


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_fault_socket_addr: str,
        client_sentinel_request_addr: str,
        engine_core_sentinel_cmd_addr: str,
        engine_core_sentinel_identities: dict[int, bytes],
        fault_state_pub_socket_addr: str,
    ):
        dp_rank = vllm_config.parallel_config.data_parallel_index
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_local_size = vllm_config.parallel_config.data_parallel_size_local
        # Client manages local+remote EngineCores in pure internal LB case.
        # Client manages local EngineCores in hybrid and external LB case.
        num_dp_managed = (
            dp_local_size if vllm_config.parallel_config.local_engines_only else dp_size
        )
        host = vllm_config.parallel_config.data_parallel_master_ip

        super().__init__(
            upstream_cmd_addr=None,
            downstream_cmd_addr=engine_core_sentinel_cmd_addr,
            sentinel_identity=None,
            sentinel_tag=None,
            vllm_config=vllm_config,
        )

        self.fault_tolerance_req_socket = make_zmq_socket(
            ctx=self.ctx,
            path=client_sentinel_request_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.fault_receiver_socket = make_zmq_socket(
            ctx=self.ctx,
            path=engine_fault_socket_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.fault_state_pub_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_state_pub_socket_addr,
            socket_type=zmq.PUB,
            bind=True,
        )
        # Queue to serialize fault tolerance instruction execution:
        # (client_identity, FaultToleranceRequest)
        self.ft_request_queue: queue.Queue[
            tuple[bytes | None, FaultToleranceRequest]
        ] = queue.Queue(
            maxsize=2
        )  # allow one pause command and another fault tolerance command
        self.ft_result_queue: queue.Queue[tuple[bytes | None, FaultToleranceResult]] = (
            queue.Queue(maxsize=2)
        )
        inproc_comm_addr = get_engine_client_zmq_addr(local_only=True, host=host)
        self.inproc_res_recv_socket = make_zmq_socket(
            ctx=self.ctx, path=inproc_comm_addr, socket_type=zmq.PAIR, bind=True
        )
        self.inproc_res_send_socket = make_zmq_socket(
            ctx=self.ctx, path=inproc_comm_addr, socket_type=zmq.PAIR, bind=False
        )

        self.is_faulted = threading.Event()
        self.engine_status_dict: ThreadSafeDict[int, dict[str, EngineStatusType]] = (
            ThreadSafeDict()
        )
        self.engine_status_dict.update(
            {
                engine_index: {"status": EngineStatusType.HEALTHY}
                for engine_index in range(dp_rank, dp_rank + num_dp_managed)
            }.items()
        )
        # todo: use identities as the key, indexes as the value as the index may
        # change in dp scale down and up
        self.engine_core_sentinel_identities = engine_core_sentinel_identities

        threading.Thread(
            target=self.run, daemon=True, name="ClientSentinelCmdAndFaultReceiverThread"
        ).start()

        threading.Thread(
            target=self._process_ft_requests_loop,
            daemon=True,
            name="ClientSentinelFtRequestsLoopThread",
        ).start()

    def _process_ft_requests_loop(self) -> None:
        """
        Worker loop to process Fault Tolerance (FT) requests
        """
        try:
            while not self.sentinel_dead:
                try:
                    identity, ft_request = self.ft_request_queue.get(timeout=1)
                    ft_result = self._execute_cmd(ft_request)
                    self.ft_result_queue.put((identity, ft_result))
                    self.inproc_res_send_socket.send(msgspec.msgpack.encode(ft_result))
                except queue.Empty:
                    pass
        except zmq.ZMQError:
            # Socket is closed.
            pass

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        for engine_status in self.engine_status_dict.values():
            if engine_status["status"] == EngineStatusType.DEAD:
                self.logger(
                    "Engine core is dead; retry won't work.",
                    level="warning",
                )
                return False

        target_engines = set(self.engine_core_sentinel_identities.values())
        new_stateless_dp_group_port = get_open_port()
        success, _ = self._execute_command_on_downstreams(
            "retry",
            target_engines,
            new_stateless_dp_group_port=new_stateless_dp_group_port,
            timeout=timeout,
        )

        for engine_index, _ in self.engine_status_dict.items():
            self.engine_status_dict[engine_index] = {"status": EngineStatusType.HEALTHY}

        if success:
            self.is_faulted.clear()
            self._pub_engine_status()
        return success

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """Pause engine cores, return True if successful. Best-effort operation."""
        self.logger(
            "Pause operation is best-effort only. Due to the complexity of "
            "collective communications (e.g., timing dependencies and "
            "synchronization barriers), pausing may not always succeed. If "
            "the process remains unresponsive or collective operations "
            "cannot be interrupted, consider shutting down and restarting "
            "the instance.",
            level="warning",
        )
        exclude_engine_index = kwargs.get("exclude_engine_index")
        soft_pause = kwargs.get("soft_pause", False)
        alive_engines = {
            identity
            for index, identity in self.engine_core_sentinel_identities.items()
            if self.engine_status_dict[index]["status"] != EngineStatusType.DEAD
            and (exclude_engine_index is None or index not in exclude_engine_index)
        }
        success, responses = self._execute_command_on_downstreams(
            "pause",
            alive_engines,
            timeout=timeout,
            soft_pause=soft_pause,
        )
        identity_to_index = {
            identity: index
            for index, identity in self.engine_core_sentinel_identities.items()
        }
        for engine_identity, ft_result in responses.items():
            if ft_result.success:
                i = identity_to_index[engine_identity]
                engine_status = self.engine_status_dict[i]["status"]
                if engine_status == EngineStatusType.HEALTHY:
                    self.engine_status_dict[i] = {"status": EngineStatusType.PAUSED}
        return success

    def _pub_engine_status(self):
        engine_status = self.engine_status_dict.to_dict()
        topic = self.ft_config.fault_state_pub_topic.encode()
        self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(engine_status))
        )

    def _alert_and_pause(self):
        """Receive fault info from engine and pause engines if first fault."""
        try:
            identity, _, message = self.fault_receiver_socket.recv_multipart()
            fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
            engine_status = (
                EngineStatusType.DEAD
                if "dead" in fault_info.type
                else EngineStatusType.UNHEALTHY
            )
            self.engine_status_dict[int(fault_info.engine_id)] = {
                "status": engine_status
            }
            self._pub_engine_status()
            if not self.is_faulted.is_set():
                self.is_faulted.set()
                pause_request = FaultToleranceRequest(
                    request_id=str(uuid.uuid4()),
                    instruction="pause",
                    params={
                        "timeout": self.ft_config.gloo_comm_timeout + 5,
                        "soft_pause": False,
                    },
                )
                while not self._dispatch_fault_tolerance_request(pause_request, None):
                    # If the queue is full, it means another fault tolerance
                    # command is being executed.
                    # Wait and retry until we can add the pause command to
                    # the queue.
                    time.sleep(0.1)

        except zmq.ZMQError:
            # Socket was closed during polling, exit loop.
            self.logger("Fault receiver socket closed, stopping thread.", level="info")
            raise

    def _dispatch_fault_tolerance_request(
        self, ft_request: FaultToleranceRequest, identity: bytes | None
    ) -> bool:
        """Add fault tolerance request to queue, return False if busy."""
        try:
            self.ft_request_queue.put_nowait((identity, ft_request))
        except queue.Full:
            return False
        return True

    def run(self):
        """Poll for fault messages and commands, dispatch to handlers."""
        poller = zmq.Poller()
        poller.register(self.fault_receiver_socket, zmq.POLLIN)
        poller.register(self.fault_tolerance_req_socket, zmq.POLLIN)
        poller.register(self.inproc_res_recv_socket, zmq.POLLIN)
        try:
            while not self.sentinel_dead:
                events = poller.poll(timeout=5000)
                if not events:
                    continue
                events = dict(events)
                if self.fault_receiver_socket in events:
                    # If a fault message is received, alert and attempt to pause.
                    self._alert_and_pause()
                if self.fault_tolerance_req_socket in events:
                    # Received fault tolerance command from client.
                    # Add corresponding command to the queue.
                    parts = self.fault_tolerance_req_socket.recv_multipart()
                    identity = parts[0]
                    msg_bytes = parts[-1]
                    ft_request = msgspec.msgpack.decode(
                        msg_bytes, type=FaultToleranceRequest
                    )
                    success = self._dispatch_fault_tolerance_request(
                        ft_request, identity
                    )
                    if not success:
                        # If we're busy, reply with a busy message.
                        msg = (
                            "System busy, vLLM is executing another fault "
                            "tolerance instruction."
                        )
                        res = FaultToleranceResult(ft_request.request_id, False, msg)
                        resp = msgspec.msgpack.encode(res)
                        self.fault_tolerance_req_socket.send_multipart(
                            [identity, b"", resp]
                        )
                if self.inproc_res_recv_socket in events:
                    # Send FT execution result back to client.
                    msg = self.inproc_res_recv_socket.recv()
                    recv_res = msgspec.msgpack.decode(msg, type=FaultToleranceResult)
                    identity, ft_result = self.ft_result_queue.get_nowait()
                    assert recv_res.request_id == ft_result.request_id
                    if identity is not None:
                        self.fault_tolerance_req_socket.send_multipart(
                            [identity, b"", msg]
                        )
                    self._pub_engine_status()
        except zmq.ZMQError:
            # Context terminated, exit thread cleanly.
            pass

    def shutdown(self):
        close_sockets(
            [
                self.fault_receiver_socket,
                self.fault_state_pub_socket,
                self.inproc_res_send_socket,
                self.inproc_res_recv_socket,
                self.fault_tolerance_req_socket,
            ]
        )
        super().shutdown()


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for:
      1. Receiving fault signals (exceptions raised in EngineCore busy loop)
      2. Receiving and executing commands from ClientSentinel
      3. Reporting execution results or faults back to the ClientSentinel
    """

    def __init__(
        self,
        engine_index: int,
        fault_signal_q: queue.Queue,
        cmd_q: queue.Queue,
        busy_loop_active: threading.Event,
        engine_input_q: queue.Queue,
        upstream_cmd_addr: str,
        downstream_cmd_addr: str,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        vllm_config: VllmConfig,
    ):
        self.engine_index = engine_index
        super().__init__(
            upstream_cmd_addr=upstream_cmd_addr,
            downstream_cmd_addr=downstream_cmd_addr,
            sentinel_identity=sentinel_identity,
            sentinel_tag=f"DP_{engine_index}",
            vllm_config=vllm_config,
        )

        self.fault_signal_q = fault_signal_q
        self.cmd_q = cmd_q
        self.busy_loop_active = busy_loop_active
        self.engine_input_q = engine_input_q
        parallel_config = vllm_config.parallel_config
        self.tp_size = parallel_config.tensor_parallel_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.dp_size = parallel_config.data_parallel_size

        # Client <-> EngineCoreSentinel sockets
        self.engine_fault_socket = make_zmq_socket(
            self.ctx,
            engine_fault_socket_addr,
            zmq.DEALER,
            bind=False,
            identity=sentinel_identity,
        )

        self.poller = zmq.Poller()
        self.communicator_aborted = False
        self.engine_running = True
        threading.Thread(
            target=self.run, daemon=True, name="EngineCoreSentinelMonitorThread"
        ).start()

    def run(self):
        """
        Continuously poll for fault signals and commands.
        """
        while not self.sentinel_dead:
            # Check for engine fault signals
            self.poll_and_report_fault_events()
            # Check for commands from ClientSentinel
            self.poll_and_execute_upstream_cmd()

    def poll_and_report_fault_events(self):
        try:
            engine_exception = self.fault_signal_q.get_nowait()
            self.engine_running = False
            if isinstance(engine_exception, EngineLoopPausedError):
                self.logger("Engine paused", level="info")
            else:
                self.logger(
                    "Detected exception %s: %s\n Call Stack:\n%s",
                    type(engine_exception).__name__,
                    engine_exception,
                    "".join(traceback.format_tb(engine_exception.__traceback__)),
                    level="error",
                )
                msg = FaultInfo.from_exception(engine_exception, self.engine_index)
                msg_bytes = msgspec.msgpack.encode(msg)
                self.engine_fault_socket.send_multipart([b"", msg_bytes])
        except queue.Empty:
            pass

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """
        Pause the busy loop of engine core safely.
        """
        self.logger("Start pausing EngineCore", level="info")
        soft_pause = kwargs.get("soft_pause", False)
        deadline = time.monotonic() + timeout
        # Clear the flag to signal busy loop should pause
        self.busy_loop_active.clear()
        success, _ = self._execute_command_on_downstreams(
            "pause",
            self._get_target_worker_identity(),
            timeout=timeout,
            soft_pause=soft_pause,
        )
        if self.engine_running:
            # Put a sentinel (empty request) to unblock the busy loop
            # if it's blocked on input_queue.get()
            self.engine_input_q.put((EngineCoreRequestType.PAUSE, None))
            remaining_timeout = max(0, deadline - time.monotonic())
            if success:
                try:
                    # Wait for engine to acknowledge the pause via fault_signal_q
                    exception = self.fault_signal_q.get(timeout=remaining_timeout)
                    self.fault_signal_q.put(exception)
                    self.engine_running = False
                except queue.Empty:
                    # Timeout waiting for pause acknowledgment
                    success = False
        return success

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        """
        Handle the retry instruction from the ClientSentinel.
        This instruction tells the EngineCore to continue its busy loop
        after being suspended due to an exception.
        """
        if self.engine_running:
            return True
        new_stateless_dp_group_port = kwargs.get("new_stateless_dp_group_port")
        deadline = time.monotonic() + timeout
        identities = self._get_target_worker_identity()
        success, _ = self._execute_command_on_downstreams(
            "retry", identities, timeout=timeout
        )
        if not success:
            return success

        if self.dp_size > 1:
            # If the Gloo communication times out,
            # the data parallel group (dp_group) needs to be reinitialized
            reinit_request = FaultToleranceRequest(
                instruction="reinit_dp_group_on_fault_tolerance",
                request_id=str(uuid.uuid4()),
                params={"new_stateless_dp_group_port": new_stateless_dp_group_port},
            )
            self.cmd_q.put(reinit_request)
        else:
            self.cmd_q.put(None)

        # Ensure busy loop has been recovered.
        remaining_timeout = max(0, deadline - time.monotonic())
        self.engine_running = self.busy_loop_active.wait(timeout=remaining_timeout)
        return self.engine_running

    def _get_target_worker_identity(self):
        identities = set()
        for tp_rank in range(self.tp_size):
            for pp_rank in range(self.pp_size):
                identity = f"PP{pp_rank}_TP{tp_rank}".encode()
                identities.add(identity)
        return identities

    def shutdown(self):
        close_sockets([self.engine_fault_socket])
        super().shutdown()


class WorkerSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        pause_event: threading.Event,
        init_distributed_env_callback: Callable,
        clear_input_batch_callback: Callable,
        device: torch.cuda.device,
    ):
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(
            upstream_cmd_addr=vllm_config.fault_tolerance_config.worker_cmd_addr,
            downstream_cmd_addr=None,
            sentinel_identity=identity.encode(),
            sentinel_tag=f"{dp_rank}_{identity}",
            vllm_config=vllm_config,
        )
        self.init_distributed_env_callback = init_distributed_env_callback
        self.clear_input_batch_callback = clear_input_batch_callback
        self.device = device

        self.pause_event = pause_event
        self.communicator_aborted = False
        torch.cuda.set_device(self.device)
        threading.Thread(
            target=self.run, daemon=True, name="WorkerSentinelMonitorThread"
        ).start()

    def run(self):
        # Wait for fault tolerance instructions from EngineCoreSentinel
        while not self.sentinel_dead:
            self.poll_and_execute_upstream_cmd()

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        soft_pause = kwargs.get("soft_pause", False)
        if soft_pause:
            self._set_device_communicator_status(False)
            self.pause_event.set()
            self.logger("Pause signal sent.")
            return True
        # Abort all NCCL communicators and
        # process groups in parallel using a thread pool.
        if self.communicator_aborted:
            return True
        self.pause_event.set()
        self._set_device_communicator_status(False)
        torch.cuda.set_device(self.device)
        model_groups = get_all_model_groups()
        futures = []

        def _abort_nccl_comm(group: GroupCoordinator):
            if group.device_communicator is not None:
                device_comm = cast(CudaCommunicator, group.device_communicator)
                nccl_comm = device_comm.pynccl_comm
                assert nccl_comm is not None
                nccl_comm.nccl_abort_comm()

        def _abort_process_group(group: GroupCoordinator):
            backend = group.device_group._get_backend(self.device)
            backend.abort()

        executor = ThreadPoolExecutor(max_workers=len(model_groups) * 2)
        try:
            for group in model_groups:
                futures.append(executor.submit(_abort_nccl_comm, group))
                futures.append(executor.submit(_abort_process_group, group))

            done, not_done = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
            if not_done:
                self.logger(
                    "%d abort calls did not finish in total %s seconds",
                    len(not_done),
                    timeout,
                    level="warning",
                )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        exception_count = sum(1 for f in done if f.exception() is not None)
        self.communicator_aborted = len(not_done) == 0 and exception_count == 0
        if self.communicator_aborted:
            cleanup_dist_env_and_memory()
            self.logger("Communicators are aborted.")
        else:
            self.logger(
                "Communicator abort failed: %d NCCL comm abort calls timed out,"
                " %d tasks threw exceptions. This may leave NCCL communicators "
                "or process groups in an inconsistent state. Subsequent "
                "distributed operations could be unsafe.",
                len(not_done),
                exception_count,
                level="error",
            )
        return self.communicator_aborted

    def _set_device_communicator_status(self, active: bool):
        model_groups = get_all_model_groups()
        for group in model_groups:
            if group.device_communicator is not None:
                device_comm = cast(CudaCommunicator, group.device_communicator)
                nccl_comm = device_comm.pynccl_comm
                assert nccl_comm is not None
                nccl_comm.available = active
                nccl_comm.disabled = not active

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        if self.communicator_aborted:
            torch.cuda.set_device(self.device)
            with set_current_vllm_config(self.vllm_config):
                self.init_distributed_env_callback()
                self.communicator_aborted = False
            torch.cuda.synchronize()
        self.clear_input_batch_callback()
        self.pause_event.clear()
        return True
