# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineCoreRequestType, EngineStatusType
from vllm.v1.engine.exceptions import EngineLoopPausedError
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
)

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

logger = init_logger(__name__)


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for:
      1. Receiving fault signals (exceptions raised in EngineCore busy loop)
      2. Receiving and executing commands from ClientSentinel
      3. Reporting execution results or faults back to the ClientSentinel
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        engine: "EngineCoreProc",
        worker_cmd_addr: str,
    ):
        self.engine_index = engine.engine_index
        super().__init__(
            f"DP_{self.engine_index}",
            sentinel_identity,
            engine,
        )

        self.data_parallel_size = parallel_config.data_parallel_size
        self.engine_recovery_timeout_sec = (
            parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
        )
        # flag to indicate if busy_loop should run.
        self.run_busy_loop = threading.Event()
        self.run_busy_loop.set()
        # flag to indicate the status of busy loop.
        self.busy_loop_paused = threading.Event()
        self.worker_responsive = threading.Event()
        self.worker_responsive.set()
        self.worker_cmd_socket = make_zmq_socket(
            ctx=self.ctx,
            path=worker_cmd_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self.worker_cmd_poller = zmq.Poller()
        self.worker_cmd_poller.register(self.worker_cmd_socket, zmq.POLLIN)
        self.worker_identities = [
            f"PP{pp_rank}_TP{tp_rank}".encode()
            for tp_rank in range(parallel_config.tensor_parallel_size)
            for pp_rank in range(parallel_config.pipeline_parallel_size)
        ]

        # Client <-> EngineCoreSentinel sockets
        self.engine_fault_socket = make_zmq_socket(
            self.ctx,
            engine_fault_socket_addr,
            zmq.DEALER,
            bind=False,
            identity=sentinel_identity,
        )

    @property
    def engine(self) -> "EngineCoreProc":
        return self.host

    def report_fault_events(self, engine_exception, engine_status: EngineStatusType):
        self.run_busy_loop.clear()
        msg = FaultInfo.from_exception(
            engine_exception, self.engine_index, engine_status
        )
        msg_bytes = msgspec.msgpack.encode(msg)
        self.engine_fault_socket.send_multipart([b"", msg_bytes])

    def handle_fault(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        return self._execute_cmd(ft_request)

    def check_worker_responsive(self) -> bool:
        # Check if workers are responsive. Should only be called in busy_loop thread.
        try:
            self.engine.model_executor.check_health()
            return True
        except Exception:
            self.worker_responsive.clear()
            logger.exception("Executor check_health() failed; worker may not recover.")
            return False

    def pause(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """Pause the busy loop of engine core safely."""
        logger.info("Start pausing EngineCore")
        timeout = ft_request.params["timeout"]
        deadline = time.monotonic() + timeout
        # set the flag to signal busy loop should pause
        self.run_busy_loop.clear()
        # Put a wakeup request to unblock the busy loop
        # if it's blocked on input_queue.get()
        self.engine.input_queue.put((EngineCoreRequestType.WAKEUP, None))
        self._execute_command_on_workers(
            FaultToleranceRequest(str(uuid.uuid4()), "pause", ft_request.params),
            self.worker_identities,
            timeout=timeout,
        )
        remaining_timeout = max(0, deadline - time.monotonic())
        # Wait for the busy loop to acknowledge the pause signal and pause itself.
        if success := self.busy_loop_paused.wait(remaining_timeout):
            remaining_timeout = max(0, deadline - time.monotonic())
            # Ensure the workers are responsive now.
            success = self.worker_responsive.wait(remaining_timeout)
        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=success,
            reason=None if success else "The engine did not pause within timeout.",
        )

    def retry(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """
        Handle the retry instruction from the ClientSentinel.
        This instruction tells the EngineCore to continue its busy loop
        after being suspended due to an exception.
        """
        timeout = ft_request.params["timeout"]
        if not self.worker_responsive.wait(timeout=timeout):
            return FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason="Worker is not responsive yet.",
            )
        if not self.busy_loop_paused.is_set():
            return FaultToleranceResult(ft_request.request_id, True)

        parallel_config = self.engine.vllm_config.parallel_config
        parallel_config._coord_store_port = ft_request.params["coord_store_port"]
        self._execute_command_on_workers(
            FaultToleranceRequest(str(uuid.uuid4()), "retry", ft_request.params),
            self.worker_identities,
            timeout=timeout,
        )
        self.clean_engine_state()
        self.run_busy_loop.set()
        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=True,
        )

    def clean_engine_state(self):
        # Put running requests into waiting list.
        scheduler = self.engine.scheduler
        timestamp = time.monotonic()
        while scheduler.running:  # type: ignore[attr-defined]
            request = scheduler.running.pop()  # type: ignore[attr-defined]
            scheduler.preempt_request(request, timestamp)  # type: ignore[attr-defined]
        scheduler.prev_step_scheduled_req_ids.clear()  # type: ignore[attr-defined]
        if self.engine.batch_queue is not None:
            self.engine.batch_queue.clear()

        if self.data_parallel_size > 1:
            # If the Gloo communication times out,
            # the data parallel group (dp_group) needs to be reinitialized
            dp_engine = cast("DPEngineCoreProc", self.engine)
            stateless_destroy_torch_distributed_process_group(dp_engine.dp_group)
            dp_engine.dp_group = (
                self.engine.vllm_config.parallel_config.stateless_init_dp_group()
            )
            self.engine.step_counter = 0

        executor = self.engine.model_executor
        # Drain all stale futures and their pending responses
        if isinstance(executor, MultiprocExecutor):
            num_stale = len(executor.futures_queue)
            executor.futures_queue.clear()
            if num_stale == 0:
                return
            logger.info("Draining %d stale response(s) from response queue", num_stale)
            if executor.kv_output_aggregator is not None:
                mqs = executor.response_mqs
            else:
                mqs = [executor.response_mqs[executor.output_rank]]
            for mq in mqs:
                for _ in range(num_stale):
                    try:
                        mq.dequeue(timeout=30)
                    except Exception:
                        break

    def _execute_command_on_workers(
        self,
        ft_request: FaultToleranceRequest,
        target_worker_sentinels: list[bytes],
        timeout: int = 5,
    ) -> FaultToleranceResult:
        request_bytes = msgspec.msgpack.encode(ft_request)
        for identity in target_worker_sentinels:
            self.worker_cmd_socket.send_multipart([identity, b"", request_bytes])

        results: dict[bytes, FaultToleranceResult] = {}
        pending = set(target_worker_sentinels)
        deadline = time.monotonic() + timeout

        while pending and (remaining := deadline - time.monotonic()) > 0:
            events = dict(self.worker_cmd_poller.poll(timeout=int(remaining * 1000)))
            if self.worker_cmd_socket not in events:
                continue

            identity, _, msg = self.worker_cmd_socket.recv_multipart()
            res = msgspec.msgpack.decode(msg, type=FaultToleranceResult)
            if identity in pending and res.request_id == ft_request.request_id:
                results[identity] = res
                pending.remove(identity)

        # For any workers that did not respond within the timeout, mark them as failed.
        for identity in pending:
            results[identity] = FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason=f"did not respond within {timeout}s",
            )

        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=all(result.success for result in results.values()),
            reason="\n".join(
                f"Worker {identity.decode()}: {result.reason}"
                for identity, result in results.items()
                if not result.success
            )
            or None,
        )

    def shutdown(self):
        close_sockets([self.engine_fault_socket, self.worker_cmd_socket])
        super().shutdown()


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """
    Wrap the busy loop function to perform fault tolerance.
    """
    from vllm.v1.engine.core import logger

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                if self.enable_fault_tolerance:
                    self.sentinel.busy_loop_paused.clear()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as e:
                if self.enable_fault_tolerance:
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore %s: %s\n Call Stack:\n%s",
                        type(e).__name__,
                        e,
                        "".join(traceback.format_tb(e.__traceback__)),
                    )
                    self.sentinel.busy_loop_paused.set()
                    if isinstance(e, EngineLoopPausedError):
                        # In async scheduling, treat worker state as temporarily HUNG
                        # until health check completes.
                        self.sentinel.worker_responsive.clear()
                        self.sentinel.report_fault_events(e, EngineStatusType.HUNG)
                        if self.sentinel.check_worker_responsive():
                            self.sentinel.worker_responsive.set()
                            self.sentinel.report_fault_events(
                                e, EngineStatusType.PAUSED
                            )
                    else:
                        self.sentinel.report_fault_events(e, EngineStatusType.UNHEALTHY)
                    logger.warning(
                        "[BusyLoopWrapper] Busy loop Suspended and "
                        "waiting for fault tolerance instructions.",
                    )
                    recovered = self.sentinel.run_busy_loop.wait(
                        timeout=self.engine_recovery_timeout_sec
                    )
                    if recovered:
                        continue
                    else:
                        logger.error(
                            "[BusyLoopWrapper] EngineCore did not recover in time."
                        )

                raise e

    return run_with_fault_tolerance
