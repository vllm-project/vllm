# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCoreProc


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for receiving
    fault signals (exceptions raised in EngineCore busy loop) and reporting them to
    ClientSentinel.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        engine_index: int,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
    ):
        self.engine_index = engine_index
        super().__init__(
            f"DP_{engine_index}",
            sentinel_identity,
        )

        self.fault_signal_q: queue.Queue[Exception] = queue.Queue()
        self.engine_recovery_timeout_sec = (
            parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
        )

        # Client <-> EngineCoreSentinel sockets
        self.engine_fault_socket = make_zmq_socket(
            self.ctx,
            engine_fault_socket_addr,
            zmq.DEALER,
            bind=False,
            identity=sentinel_identity,
        )

        threading.Thread(
            target=self.run, daemon=True, name="EngineCoreSentinelMonitorThread"
        ).start()

    def run(self):
        """Continuously poll for fault signals and report to client sentinel."""
        while not self.sentinel_dead:
            # Check for engine fault signals
            self.poll_and_report_fault_events()

    def poll_and_report_fault_events(self):
        try:
            engine_exception = self.fault_signal_q.get(timeout=1)
            self.logger(
                "Detected exception %s: %s\n Call Stack:\n%s",
                type(engine_exception).__name__,
                engine_exception,
                "".join(traceback.format_tb(engine_exception.__traceback__)),
                level="error",
            )
            msg = FaultInfo.from_exception(
                engine_exception, self.engine_index, EngineStatusType.UNHEALTHY
            )
            msg_bytes = msgspec.msgpack.encode(msg)
            self.engine_fault_socket.send_multipart([b"", msg_bytes])
        except queue.Empty:
            pass

    def shutdown(self):
        self.engine_fault_socket.close(linger=0)
        super().shutdown()


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """
    Wrap the busy loop function to perform fault tolerance.
    """
    from vllm.v1.engine.core import logger

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as original_exc:
                if self.enable_fault_tolerance:
                    self.engine_core_sentinel.fault_signal_q.put(original_exc)
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore busy loop raised a %s exception.",
                        type(original_exc).__name__,
                    )
                    # todo: Currently only wait a certain time before shutting
                    #  down the engine. Will implement fault tolerance methods
                    #  in the upcoming PRs.
                    time.sleep(self.engine_core_sentinel.engine_recovery_timeout_sec)

                # Fault tolerance not enabled OR no instruction received
                # before timeout. Re-raise the original exception
                # for upper level handling.
                raise

    return run_with_fault_tolerance
