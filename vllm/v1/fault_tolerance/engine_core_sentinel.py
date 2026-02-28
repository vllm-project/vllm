# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import traceback

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for receiving
    fault signals (exceptions raised in EngineCore busy loop) and reporting them to
    ClientSentinel.
    """

    def __init__(
        self,
        engine_index: int,
        fault_signal_q: queue.Queue,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        vllm_config: VllmConfig,
    ):
        self.engine_index = engine_index
        super().__init__(
            sentinel_tag=f"DP_{engine_index}",
            vllm_config=vllm_config,
        )

        self.fault_signal_q = fault_signal_q

        # Client <-> EngineCoreSentinel sockets
        self.engine_fault_socket = make_zmq_socket(
            self.ctx,
            engine_fault_socket_addr,
            zmq.DEALER,
            bind=False,
            identity=sentinel_identity,
        )

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
            self._report_exception_to_client_sentinel(engine_exception)
            self.engine_running = False
        except queue.Empty:
            pass

    def _report_exception_to_client_sentinel(self, exception: Exception) -> None:
        msg = FaultInfo.from_exception(exception, self.engine_index)
        msg_bytes = msgspec.msgpack.encode(msg)
        self.engine_fault_socket.send_multipart([b"", msg_bytes])

    def shutdown(self):
        self.engine_fault_socket.close(linger=0)
        super().shutdown()


def busy_loop_wrapper(busy_loop_func):
    """
    Wrap the busy loop function to perform fault tolerance.
    """
    from vllm.v1.engine.core import logger

    def run_with_fault_tolerance(self):
        while True:
            try:
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as original_exc:
                if self.enable_fault_tolerance:
                    self.fault_signal_q.put(original_exc)
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore busy loop raised an exception. "
                    )
                    # todo: Currently only wait a certain time before shutting
                    #  down the engine. Will implement fault tolerance methods
                    #  in the upcoming PRs.
                    time.sleep(self.engine_recovery_timeout)

                # Fault tolerance not enabled OR no instruction received
                # before timeout. Re-raise the original exception
                # for upper level handling.
                raise original_exc

    return run_with_fault_tolerance
