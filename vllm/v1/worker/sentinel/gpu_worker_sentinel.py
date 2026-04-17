# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import threading
import msgspec
import torch
import zmq

from vllm.distributed import get_ep_group, get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.fault_tolerance import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_GLOBAL_PAUSE_EVENT = threading.Event()


def get_pause_event() -> threading.Event:
    global _GLOBAL_PAUSE_EVENT
    return _GLOBAL_PAUSE_EVENT


class WorkerSentinel(BaseSentinel):
    def __init__(
        self,
        worker: "Worker",
        device: torch.device,
        worker_cmd_addr: str,
    ):
        dp_rank = worker.parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity_str = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(f"{dp_rank}_{identity_str}", identity_str.encode(), worker)
        self.device = device
        torch.accelerator.set_device_index(self.device)

        self.engine_core_cmd_socket = make_zmq_socket(
            self.ctx,
            worker_cmd_addr,
            zmq.DEALER,
            bind=False,
            identity=self.identity,
        )

        # Currently, only deepep_ll and nixl_ep backends support fault tolerance.
        ft_backend_set = {"deepep_low_latency", "nixl_ep"}
        parallel_config = worker.parallel_config
        self.use_ft_backend = (
            parallel_config.all2all_backend in ft_backend_set
            and parallel_config.data_parallel_size > 1
        )
        if self.use_ft_backend:
            world_size = get_ep_group().world_size
            self.mask = torch.zeros((world_size,), device=self.device, dtype=torch.int)
            # todo: last_mask is prepared and should be updated in scaling down.
            self.last_mask = torch.zeros_like(self.mask)

        threading.Thread(
            target=self.run, daemon=True, name="WorkerSentinelThread"
        ).start()

    @property
    def worker(self) -> "Worker":
        return self.host

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
        get_pause_event().set()
        return FaultToleranceResult(ft_request.request_id, True)

    def shutdown(self):
        close_sockets([self.engine_core_cmd_socket])
        super().shutdown()
