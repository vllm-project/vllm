# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Callable

import msgspec.msgpack
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceZmqAddresses


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        shutdown_callback: Callable,
    ):
        super().__init__(
            vllm_config=vllm_config, sentinel_tag=None, identity=b"client_sentinel"
        )

        self.ctx_async = zmq.asyncio.Context()
        self.instance_shutdown_callback = shutdown_callback
        self.sentinel_dead = False
        self.logger = self._make_logger()
        self._shutdown_task: asyncio.Task | None = None

        self.fault_receiver_socket = make_zmq_socket(
            ctx=self.ctx_async,
            path=fault_tolerance_addresses.engine_fault_socket_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.fault_state_pub_socket = make_zmq_socket(
            ctx=self.ctx_async,
            path=fault_tolerance_addresses.fault_state_pub_socket_addr,
            socket_type=zmq.PUB,
            bind=True,
        )

        self.start_rank = vllm_config.parallel_config.data_parallel_index
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_local_size = vllm_config.parallel_config.data_parallel_size_local
        num_dp_managed = (
            dp_local_size if vllm_config.parallel_config.local_engines_only else dp_size
        )
        self.engine_status_dict: dict[int, dict[str, str]] = {
            engine_index: {"status": "healthy"}
            for engine_index in range(self.start_rank, self.start_rank + num_dp_managed)
        }
        asyncio.create_task(self.run())

    async def _pub_engine_status(self):
        engine_status = self.engine_status_dict.copy()
        pub_msg = {
            "total_engines": len(engine_status),
            "engines": [
                {"id": i, "status": status["status"]}
                for i, status in engine_status.items()
            ],
        }
        topic = self.ft_config.fault_state_pub_topic.encode()
        await self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(pub_msg))
        )

    async def run(self):
        """Receive fault info from engine and pause engines if happened."""
        try:
            while not self.sentinel_dead:
                _, _, message = await self.fault_receiver_socket.recv_multipart()
                fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
                # Update engine status
                status_enum = EngineStatusType(fault_info.engine_status)
                self.engine_status_dict[int(fault_info.engine_id)] = {
                    "status": status_enum.name.lower()
                }
                await self._pub_engine_status()
                if self._shutdown_task is None:
                    self._shutdown_task = asyncio.create_task(
                        self._shutdown_after_timeout()
                    )

        except zmq.ZMQError:
            self.logger("Fault receiver socket closed, stopping async monitor.")

    async def _shutdown_after_timeout(self):
        await asyncio.sleep(self.ft_config.engine_recovery_timeout_sec)
        self.instance_shutdown_callback()

    def shutdown(self):
        close_sockets([self.fault_receiver_socket, self.fault_state_pub_socket])
        self.ctx_async.term()
        super().shutdown()
