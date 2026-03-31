# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import uuid
from collections.abc import Callable

import msgspec.msgpack
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs as FTUtilityOutputs
from vllm.v1.engine import EngineStatusType, UtilityOutput
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
    FaultToleranceZmqAddresses,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        call_utility_async: Callable,
        core_engines: list[bytes],
    ):
        self.engine_identities = core_engines
        super().__init__(
            vllm_config=vllm_config, sentinel_tag=None, identity=b"client_sentinel"
        )

        self.call_utility_async = call_utility_async
        self.ctx_async = zmq.asyncio.Context()
        self.sentinel_dead = False
        self.logger = self._make_logger()

        # sockets to receive fault tolerance request from clients
        self.ft_request_sockets = [
            make_zmq_socket(self.ctx_async, addr, zmq.DEALER, False, self.identity)
            for addr in fault_tolerance_addresses.ft_request_addresses
        ]
        # sockets to send fault tolerance execution results back to clients
        self.ft_result_sockets = [
            make_zmq_socket(
                self.ctx_async,
                addr,
                zmq.PUSH,
                linger=4000,
            )
            for addr in fault_tolerance_addresses.ft_result_addresses
        ]

        self.is_faulted = asyncio.Event()
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

        self._utility_encoder = MsgpackEncoder()

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
            self.logger("vLLM instance is paused and waiting for recovery commands.")
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
        topic = self.ft_config.fault_state_pub_topic.encode()
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
                    # todo: Timeout for DeepEP kernel is fixed to 100 seconds
                    timeout = max(100, self.ft_config.gloo_comm_timeout_sec) + 5
                    pause_request = FaultToleranceRequest.builder(
                        request_id=str(uuid.uuid4()),
                        instruction="pause",
                        params={"timeout": timeout},
                    )
                    asyncio.create_task(self.pause(pause_request))

        except zmq.ZMQError:
            self.logger("Fault receiver socket closed, stopping async monitor.")

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
                self.logger("Sockets closed, terminating.")
                self.sentinel_dead = True

    def shutdown(self):
        self.sentinel_dead = True
        close_sockets([self.fault_receiver_socket, self.fault_state_pub_socket])
        close_sockets(self.ft_request_sockets + self.ft_result_sockets)
        self.ctx_async.term()
        super().shutdown()
