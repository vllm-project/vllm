# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import traceback
import uuid

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.exceptions import EngineLoopPausedError
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.serial_utils import run_method


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
        downstream_cmd_addr: str,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        vllm_config: VllmConfig,
    ):
        self.engine_index = engine_index
        super().__init__(
            sentinel_tag=f"DP_{engine_index}",
            vllm_config=vllm_config,
            identity=sentinel_identity,
        )

        self.fault_signal_q = fault_signal_q
        self.cmd_q = cmd_q
        self.busy_loop_active = busy_loop_active
        self.engine_input_q = engine_input_q
        parallel_config = vllm_config.parallel_config
        self.tp_size = parallel_config.tensor_parallel_size
        self.pp_size = parallel_config.pipeline_parallel_size
        self.dp_size = parallel_config.data_parallel_size

        self.worker_cmd_socket = make_zmq_socket(
            ctx=self.ctx,
            path=downstream_cmd_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self.worker_cmd_poller = zmq.Poller()
        self.worker_cmd_poller.register(self.worker_cmd_socket, zmq.POLLIN)

        # Client <-> EngineCoreSentinel sockets
        self.engine_fault_socket = make_zmq_socket(
            self.ctx,
            engine_fault_socket_addr,
            zmq.DEALER,
            bind=False,
            identity=sentinel_identity,
        )

        self.communicator_aborted = False
        self.engine_paused = threading.Event()
        threading.Thread(
            target=self.run, daemon=True, name="EngineCoreSentinelMonitorThread"
        ).start()

    def run(self):
        """Continuously poll for fault signals and report to client sentinel."""
        while not self.sentinel_dead:
            # poll and report fault events
            engine_exception = self.fault_signal_q.get()
            self.engine_paused.set()
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

    def handle_fault(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        return self._execute_cmd(ft_request)

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """
        Pause the busy loop of engine core safely.
        """
        self.logger("Start pausing EngineCore", level="info")
        soft_pause = kwargs.get("soft_pause", False)
        deadline = time.monotonic() + timeout
        # Clear the flag to signal busy loop should pause
        self.busy_loop_active.clear()
        success, _ = self._execute_command_on_workers(
            "pause",
            self._get_target_worker_identity(),
            timeout=timeout,
            soft_pause=soft_pause,
        )
        if success and not self.engine_paused.is_set():
            # Put a sentinel (empty request) to unblock the busy loop
            # if it's blocked on input_queue.get()
            self.engine_input_q.put((EngineCoreRequestType.PAUSE, None))
            remaining_timeout = max(0, deadline - time.monotonic())
            success = self.engine_paused.wait(remaining_timeout)
        return success

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        """
        Handle the retry instruction from the ClientSentinel.
        This instruction tells the EngineCore to continue its busy loop
        after being suspended due to an exception.
        """
        if not self.engine_paused.is_set():
            return True
        new_stateless_dp_group_port = kwargs.get("new_stateless_dp_group_port")
        deadline = time.monotonic() + timeout
        identities = self._get_target_worker_identity()
        success, _ = self._execute_command_on_workers(
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
        success = self.busy_loop_active.wait(timeout=remaining_timeout)
        if success:
            self.engine_paused.clear()
        return success

    def _get_target_worker_identity(self):
        return {
            f"PP{pp_rank}_TP{tp_rank}".encode()
            for tp_rank in range(self.tp_size)
            for pp_rank in range(self.pp_size)
        }

    def _execute_command_on_workers(
        self,
        method_name: str,
        target_worker_sentinels: set[bytes],
        timeout: int = 5,
        **kwargs,
    ) -> tuple[bool, dict[bytes, FaultToleranceResult]]:
        """
        Broadcast a command to worker sentinels and collect responses.
        """
        # Create fault tolerance request
        kwargs["timeout"] = timeout
        ft_request = FaultToleranceRequest(
            request_id=str(uuid.uuid4()),
            instruction=method_name,
            params=kwargs,
        )
        # Broadcast the instruction
        msg_bytes = msgspec.msgpack.encode(ft_request)
        for identity in target_worker_sentinels:
            self.worker_cmd_socket.send_multipart([identity, b"", msg_bytes])
        # Wait for responses
        responses = self._wait_for_execution_result(
            target_worker_sentinels,
            timeout,
            ft_request,
        )
        # check the execution results
        for sentinel_identity in target_worker_sentinels:
            response = responses.get(sentinel_identity)
            if response is None:
                self.logger(
                    'Worker sentinels timed out on "%s".',
                    method_name,
                    level="error",
                )
                return False, responses
            elif not response.success:
                self.logger(
                    'Worker sentinels failed to "%s" (reason: %s)',
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
        assert self.worker_cmd_socket is not None
        deadline = time.monotonic() + timeout
        responses: dict[bytes, FaultToleranceResult] = {}
        pending = set(target_identities)
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                events = self.worker_cmd_poller.poll(int(remaining * 1000))
                if not events:
                    break
                identity, _, payload = self.worker_cmd_socket.recv_multipart()
                result = msgspec.msgpack.decode(payload, type=FaultToleranceResult)
                # Ignore unrelated responses
                if result.request_id != ft_request.request_id:
                    self.logger(
                        "Discarding outdated response: %s", result, level="warning"
                    )
                    continue

                responses[identity] = result
                pending.discard(identity)
            except Exception as e:
                self.logger(
                    "Error while processing engine response: %s", e, level="error"
                )
                break

        return responses

    def shutdown(self):
        close_sockets([self.engine_fault_socket, self.worker_cmd_socket])
        super().shutdown()


def busy_loop_wrapper(busy_loop_func):
    """
    Wrap the busy loop function to perform fault tolerance.
    """
    from vllm.v1.engine.core import logger

    def run_with_fault_tolerance(self):
        while True:
            try:
                if self.enable_fault_tolerance:
                    self.busy_loop_active.set()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as original_exc:
                if self.enable_fault_tolerance:
                    self.busy_loop_active.clear()
                    self.fault_signal_q.put(original_exc)
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore busy loop raised an exception. "
                        "Suspended and waiting for fault tolerance "
                        "instructions."
                    )

                    # Put running requests into waiting list.
                    timestamp = time.monotonic()
                    while self.scheduler.running:
                        request = self.scheduler.running.pop()
                        self.scheduler.preempt_request(request, timestamp)
                    self.scheduler.prev_step_scheduled_req_ids.clear()
                    if self.batch_queue is not None:
                        self.batch_queue.clear()

                    try:
                        # Block until recovery command received
                        ft_request = self.cmd_q.get(
                            timeout=self.engine_recovery_timeout_sec
                        )

                        if ft_request is not None:
                            logger.debug(
                                "[BusyLoopWrapper] Received fault tolerance "
                                "command: %s",
                                ft_request.instruction,
                            )
                            method, params = (ft_request.instruction, ft_request.params)
                            run_method(self, method, args=(), kwargs=params)
                        # recovery succeeded; restart the busy loop
                        continue
                    except queue.Empty:
                        # No handling instruction received within predefined
                        # timeout period.
                        logger.error(
                            "[BusyLoopWrapper] Fault tolerance instruction not received"
                            " within timeout. Proceeding with default exception "
                            "handling."
                        )
                    except Exception as cmd_exc:
                        raise RuntimeError(
                            "Fault tolerance execution failed."
                        ) from cmd_exc

                # Fault tolerance not enabled OR no instruction received
                # before timeout. Re-raise the original exception
                # for upper level handling.
                raise original_exc

    return run_with_fault_tolerance
