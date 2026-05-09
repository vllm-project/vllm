# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCoreProc

logger = init_logger(__name__)


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for receiving
    fault signals (exceptions raised in EngineCore busy loop) and reporting them to
    ClientSentinel.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        engine: "EngineCoreProc",
    ):
        self.engine_index = engine.engine_index
        super().__init__(
            f"DP_{self.engine_index}",
            sentinel_identity,
            engine,
        )

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

    @property
    def engine(self) -> "EngineCoreProc":
        return self.host

    def report_fault_events(self, engine_exception):
        msg = FaultInfo.from_exception(
            engine_exception, self.engine_index, EngineStatusType.UNHEALTHY
        )
        msg_bytes = msgspec.msgpack.encode(msg)
        self.engine_fault_socket.send_multipart([b"", msg_bytes])

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
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore %s: %s\n Call Stack:\n%s",
                        type(original_exc).__name__,
                        original_exc,
                        "".join(traceback.format_tb(original_exc.__traceback__)),
                    )
                    self.engine_core_sentinel.report_fault_events(original_exc)
                    # todo: Currently only wait a certain time before shutting
                    #  down the engine. Will implement fault tolerance methods
                    #  in the upcoming PRs.
                    time.sleep(self.engine_core_sentinel.engine_recovery_timeout_sec)

                # Fault tolerance not enabled OR no instruction received
                # before timeout. Re-raise the original exception
                # for upper level handling.
                raise

    return run_with_fault_tolerance
