# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import msgspec
import torch
import zmq

from vllm.config import ParallelConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.fault_tolerance import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult

logger = init_logger(__name__)


class WorkerSentinel(BaseSentinel):
    def __init__(
        self,
        parallel_config: ParallelConfig,
        pause_event: threading.Event,
        device: torch.device,
        worker_cmd_addr: str,
    ):
        dp_rank = parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity_str = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(f"{dp_rank}_{identity_str}", identity_str.encode())
        self.device = device
        self.pause_event = pause_event
        torch.accelerator.set_device_index(self.device)

        self.engine_core_cmd_socket = make_zmq_socket(
            self.ctx,
            worker_cmd_addr,
            zmq.DEALER,
            bind=False,
            identity=self.identity,
        )

        threading.Thread(
            target=self.run, daemon=True, name="WorkerSentinelThread"
        ).start()

    def run(self):
        # Wait for fault tolerance instructions from EngineCoreSentinel
        while not self.sentinel_dead:
            self.poll_and_execute_upstream_cmd()

    def poll_and_execute_upstream_cmd(self):
        """
        Receive and execute a command from upstream sentinel and send back
        the execution result.
        """
        try:
            _, msg = self.engine_core_cmd_socket.recv_multipart()
            ft_request = msgspec.msgpack.decode(msg, type=FaultToleranceRequest)
            ft_result = self._execute_cmd(ft_request)
            msg_bytes = msgspec.msgpack.encode(ft_result)
            self.engine_core_cmd_socket.send_multipart([b"", msg_bytes])
        except zmq.ZMQError:
            logger.info("Socket closed, terminating.")
            self.sentinel_dead = True

    def pause(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        self.pause_event.set()
        return FaultToleranceResult(ft_request.request_id, True)

    def shutdown(self):
        close_sockets([self.engine_core_cmd_socket])
        super().shutdown()
