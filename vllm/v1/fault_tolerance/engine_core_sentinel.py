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
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceRequest
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
                            timeout=self.engine_recovery_timeout
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
