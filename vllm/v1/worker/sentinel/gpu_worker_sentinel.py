# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import msgspec
import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.fault_tolerance import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult


class WorkerSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        pause_event: threading.Event,
        device: torch.device,
    ):
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity_str = f"PP{pp_rank}_TP{tp_rank}"
        super().__init__(
            sentinel_tag=f"{dp_rank}_{identity_str}",
            vllm_config=vllm_config,
            identity=identity_str.encode(),
        )
        self.device = device
        self.pause_event = pause_event
        torch.accelerator.set_device_index(self.device)

        assert vllm_config.fault_tolerance_config.worker_cmd_addr is not None
        self.engine_core_cmd_socket = make_zmq_socket(
            self.ctx,
            vllm_config.fault_tolerance_config.worker_cmd_addr,
            zmq.DEALER,
            bind=False,
            identity=self.identity,
        )
        self.engine_core_cmd_socket.setsockopt(zmq.RCVTIMEO, 100)

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
        except zmq.Again:
            pass
        except zmq.ZMQError:
            self.logger("Socket closed, terminating.")
            self.sentinel_dead = True

    def pause(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        self.pause_event.set()
        return FaultToleranceResult(ft_request.request_id, True)

    def shutdown(self):
        close_sockets([self.engine_core_cmd_socket])
        super().shutdown()
