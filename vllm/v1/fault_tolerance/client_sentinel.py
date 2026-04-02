# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Callable

import msgspec.msgpack
import zmq.asyncio

from vllm.config import ParallelConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FAULT_STATE_PUB_TOPIC,
    FaultInfo,
    FaultToleranceZmqAddresses,
)


class ClientSentinel(BaseSentinel):
    """
    Client-side sentinel for fault tolerance monitoring.
    Monitors EngineCore health status via ZMQ sockets, publishes engine state
    to upper-level orchestration frameworks (LLMD, K8s, Aibrix), and triggers
    instance shutdown on unrecoverable faults.

    Connects to:
    - EngineCore fault reporting sockets (receives exceptions & process exit events)
    - Fault state publisher socket (broadcasts engine health status)
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        shutdown_callback: Callable,
    ):
        super().__init__(None, b"client_sentinel")
        self.ft_config = parallel_config.fault_tolerance_config
        self.ctx_async = zmq.asyncio.Context()
        self.instance_shutdown_callback = shutdown_callback
        self.sentinel_dead = False
        self.logger = self._make_logger()
        self._shutdown_task: asyncio.Task | None = None

        # Port for receiving fault signals:
        # 1. Exceptions caught by fault_tolerant_wrapper in EngineCore
        # 2. Exit notifications from EngineCoreProc monitored by CoreEngineActorManager
        #    or CoreEngineProcManager
        self.fault_receiver_socket = make_zmq_socket(
            ctx=self.ctx_async,
            path=fault_tolerance_addresses.engine_fault_socket_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        # Port for reporting EngineCore status to service frameworks (LLMD, K8s, Aibrix)
        self.fault_state_pub_socket = make_zmq_socket(
            ctx=self.ctx_async,
            path=fault_tolerance_addresses.fault_state_pub_socket_addr,
            socket_type=zmq.PUB,
            bind=True,
        )

        self.start_rank = parallel_config.data_parallel_index
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_dp_managed = (
            dp_local_size if parallel_config.local_engines_only else dp_size
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
        topic = FAULT_STATE_PUB_TOPIC.encode()
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

    async def scale_elastic_ep(self, new_data_parallel_size: int):
        # Update the engine status dict and publish the new status.
        for engine, status in self.engine_status_dict.items():
            if status["status"] != "healthy":
                msg = f"Cannot scale elastic EP because engine {engine} is not healthy."
                self.logger(msg, level="error")
                raise RuntimeError(msg)

        # TODO: Elastic EP currently supports only Ray + internal LB.
        # Refresh behavior should be revisited after MP and other LB modes are supported
        self.engine_status_dict = {
            engine_index: {"status": "healthy"}
            for engine_index in range(new_data_parallel_size)
        }
        await self._pub_engine_status()

    def shutdown(self):
        close_sockets([self.fault_receiver_socket, self.fault_state_pub_socket])
        self.ctx_async.term()
        super().shutdown()
