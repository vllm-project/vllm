# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import uuid
from collections.abc import Callable

import msgspec.msgpack
import zmq.asyncio
from torch.distributed import default_pg_timeout

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs as FTUtilityOutputs
from vllm.v1.engine import EngineStatusType, UtilityOutput
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FAULT_STATE_PUB_TOPIC,
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
    FaultToleranceZmqAddresses,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

logger = init_logger(__name__)
DEEP_EP_KERNEL_TIMEOUT = 100  # seconds (currently fixed)


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
        call_utility_async: Callable,
        core_engines: list[bytes],
    ):
        self.ctx = zmq.asyncio.Context()
        super().__init__(None, b"client_sentinel")
        self.engine_identities = core_engines
        self.call_utility_async = call_utility_async

        self.ft_config = parallel_config.fault_tolerance_config
        self.gloo_timeout_seconds: int = (
            parallel_config.gloo_timeout_seconds
            if parallel_config.gloo_timeout_seconds is not None
            else int(default_pg_timeout.total_seconds())
        )
        if parallel_config.gloo_timeout_seconds is None:
            logger.warning(
                "Gloo timeout not set, using default_pg_timeout:%s",
                int(default_pg_timeout.total_seconds()),
            )
        # Gloo collective timeout and all2all kernel timeout (DeepEP / NIXL-EP)
        # must both be shorter than the engine recovery timeout.
        # Otherwise execution may block inside communication (CPU or GPU),
        # and the recovery logic will not get a chance to run.
        if (
            max(self.gloo_timeout_seconds, DEEP_EP_KERNEL_TIMEOUT)
            > self.ft_config.engine_recovery_timeout_sec
        ):
            raise ValueError(
                "Engine recovery timeout must be greater than both Gloo timeout and "
                "all2all kernel timeout (DeepEP / NIXL-EP) to ensure recovery can run."
            )

        self.sentinel_dead = False
        self._shutdown_task: asyncio.Task | None = None

        # Port for receiving fault signals:
        # 1. Exceptions caught by fault_tolerant_wrapper in EngineCore
        # 2. Exit notifications of EngineCoreProc monitored by CoreEngineActorManager
        #    or CoreEngineProcManager
        self.fault_receiver_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.engine_fault_socket_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        # Port for reporting EngineCore status to service frameworks (LLMD, K8s, Aibrix)
        self.fault_state_pub_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.fault_state_pub_socket_addr,
            socket_type=zmq.PUB,
            bind=True,
        )

        # sockets to receive fault tolerance request from clients
        self.ft_request_sockets = [
            make_zmq_socket(self.ctx, addr, zmq.DEALER, False, self.identity)
            for addr in fault_tolerance_addresses.ft_request_addresses
        ]
        # sockets to send fault tolerance execution results back to clients
        self.ft_result_sockets = [
            make_zmq_socket(
                self.ctx,
                addr,
                zmq.PUSH,
                linger=4000,
            )
            for addr in fault_tolerance_addresses.ft_result_addresses
        ]

        self.is_faulted = asyncio.Event()
        self._utility_encoder = MsgpackEncoder()

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
        self.engine_identity_to_index = {
            identity: index
            for index, identity in zip(
                range(self.start_rank, self.start_rank + num_dp_managed),
                self.engine_identities,
            )
        }
        asyncio.create_task(self.run())
        asyncio.create_task(self.poll_and_execute_cmd())

    async def _send_utility_result(
        self,
        client_index: int,
        call_id: int,
        result: FaultToleranceResult,
    ) -> None:
        # Return the fault-tolerance execution result to the originating client.
        uo = UtilityOutput(call_id=call_id)
        uo.result = UtilityResult(result)
        outputs = FTUtilityOutputs(utility_output=uo)
        buffers = self._utility_encoder.encode(outputs)
        await self.ft_result_sockets[client_index].send_multipart(buffers, copy=False)

    async def pause(self, ft_request: FaultToleranceRequest):  # type: ignore[override]
        """Expected params: timeout, exclude_engine_index (optional)."""
        exclude_engine_index = ft_request.params.get("exclude_engine_index")

        # Pause all engines except ones already marked dead or being excluded.
        target_engines = [
            self.engine_identities[i - self.start_rank]
            for i, status in self.engine_status_dict.items()
            if status["status"] != "dead"
            and (exclude_engine_index is None or i not in exclude_engine_index)
        ]
        res = await self._execute_cmd_on_engines(ft_request, target_engines)
        if res.success:
            logger.info("vLLM instance is paused and waiting for recovery commands.")
        return res

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

    async def _execute_cmd_on_engines(
        self, ft_request: FaultToleranceRequest, target_engines: list[bytes]
    ) -> FaultToleranceResult:
        coroutines = []
        # dispatch commands to target engines
        for core_engine in target_engines:
            coro = self.call_utility_async(
                "handle_fault", ft_request, engine=core_engine
            )
            coroutines.append(coro)

        timeout = ft_request.params["timeout"]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*coroutines),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason=f"Timed out after {timeout}s waiting for engine responses.",
            )

        results = [FaultToleranceResult(**res) for res in results]
        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=all(res.success for res in results),
            reason="\n".join(
                f"Engine {self.engine_identity_to_index[engine]}: {res.reason}"
                for engine, res in zip(target_engines, results)
                if not res.success
            )
            or None,
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
                if (
                    not self.is_faulted.is_set()
                    and status_enum != EngineStatusType.HEALTHY
                ):
                    self.is_faulted.set()
                    # todo: Timeout for DeepEP/nixl-ep kernel is fixed to 100 seconds
                    timeout = max(DEEP_EP_KERNEL_TIMEOUT, self.gloo_timeout_seconds) + 5
                    pause_request = FaultToleranceRequest.builder(
                        request_id=str(uuid.uuid4()),
                        instruction="pause",
                        params={"timeout": timeout},
                    )
                    asyncio.create_task(self.pause(pause_request))

        except zmq.ZMQError:
            logger.info("Fault receiver socket closed, stopping async monitor.")

    async def refresh_engine_status(self, new_data_parallel_size: int):
        # Update the engine status dict and publish the new status.
        for engine, status in self.engine_status_dict.items():
            if status["status"] != "healthy":
                msg = f"Cannot scale elastic EP because engine {engine} is not healthy."
                logger.error(msg)
                raise RuntimeError(msg)

        # TODO: Elastic EP currently supports only Ray + internal LB.
        # The current refresh behavior assumes that ranks have been reassigned to start
        # from 0 and be contiguous after scale down. This is true for Ray + internal LB
        # but this logic may need to be revisited when support for MP and other LB modes
        # are added.
        self.engine_status_dict = {
            engine_index: {"status": "healthy"}
            for engine_index in range(new_data_parallel_size)
        }
        await self._pub_engine_status()

    async def poll_and_execute_cmd(self):
        """Poll and execute fault tolerance commands."""
        generic_decoder = MsgpackDecoder()
        # Initialize request sockets.
        for request_socket in self.ft_request_sockets:
            await request_socket.send(b"")

        poller = zmq.asyncio.Poller()
        for sock in self.ft_request_sockets:
            poller.register(sock, zmq.POLLIN)

        while not self.sentinel_dead:
            try:
                events = await poller.poll(timeout=100)
                if not events:
                    continue
                for sock, event in events:
                    # Receive a client FT request, execute it, and route the result back
                    _, *msg = await sock.recv_multipart(copy=False)
                    client_index, call_id, _, ft_args = generic_decoder.decode(msg)
                    ft_request = FaultToleranceRequest(**ft_args[0])
                    ft_result = await getattr(self, ft_request.instruction)(ft_request)
                    await self._send_utility_result(client_index, call_id, ft_result)
            except zmq.ZMQError:
                logger.info("Sockets closed, terminating.")
                self.sentinel_dead = True

    def shutdown(self):
        self.sentinel_dead = True
        close_sockets([self.fault_receiver_socket, self.fault_state_pub_socket])
        close_sockets(self.ft_request_sockets + self.ft_result_sockets)
        super().shutdown()
