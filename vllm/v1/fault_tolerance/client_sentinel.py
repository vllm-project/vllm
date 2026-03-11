# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import traceback
import uuid
from collections.abc import Callable

import msgspec.msgpack
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, get_open_port, make_zmq_socket
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

        self.input_sockets = [
            make_zmq_socket(
                self.ctx_async,
                input_address,
                zmq.DEALER,
                identity=self.identity,
                bind=False,
            )
            for input_address in fault_tolerance_addresses.all_client_input_addresses
        ]

        self.output_sockets = [
            make_zmq_socket(self.ctx_async, output_address, zmq.PUSH, linger=4000)
            for output_address in fault_tolerance_addresses.all_client_output_addresses
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
        self.engine_status_dict: dict[int, dict[str, EngineStatusType]] = {
            engine_index: {"status": EngineStatusType.HEALTHY}
            for engine_index in range(self.start_rank, self.start_rank + num_dp_managed)
        }

        asyncio.create_task(self.run())
        asyncio.create_task(self._monitor_and_pause_on_fault())

    async def _send_utility_result(
        self,
        client_index: int,
        call_id: int,
        result: FaultToleranceResult,
    ) -> None:
        uo = UtilityOutput(call_id=call_id)
        uo.result = UtilityResult(result)
        outputs = FTUtilityOutputs(utility_output=uo)
        buffers = self._utility_encoder.encode(outputs)
        await self.output_sockets[client_index].send_multipart(buffers, copy=False)

    async def retry(self, timeout: int = 1, **kwargs) -> bool:  # type: ignore[override]
        for engine_status in self.engine_status_dict.values():
            if engine_status["status"] == EngineStatusType.DEAD:
                self.logger(
                    "Engine core is dead; retry won't work.",
                    level="warning",
                )
                return False
        kwargs["timeout"] = timeout
        if "new_stateless_dp_group_port" not in kwargs:
            kwargs["new_stateless_dp_group_port"] = get_open_port()
        ft_results = await self._execute_cmd_on_engines(
            ft_request=FaultToleranceRequest(
                request_id=str(uuid.uuid4()), instruction="retry", params=kwargs
            ),
            target_engines=self.engine_identities,
        )

        for i, res in enumerate(ft_results):
            if res.success:
                self.engine_status_dict[i + self.start_rank] = {
                    "status": EngineStatusType.HEALTHY
                }

        success = all([res.success for res in ft_results])
        if success:
            self.is_faulted.clear()
        await self._pub_engine_status()
        return success

    async def pause(self, timeout: int = 1, **kwargs) -> bool:  # type: ignore[override]
        """Pause engine cores, return True if successful. Best-effort operation."""
        self.logger(
            "Pause operation is best-effort only. Due to the complexity of "
            "collective communications (e.g., timing dependencies and "
            "synchronization barriers), pausing may not always succeed. If "
            "the process remains unresponsive or collective operations "
            "cannot be interrupted, consider shutting down and restarting "
            "the instance.",
            level="warning",
        )
        exclude_engine_index = kwargs.get("exclude_engine_index")
        soft_pause = kwargs.get("soft_pause", False)
        alive_engines = [
            i
            for i, status in self.engine_status_dict.items()
            if status["status"] != EngineStatusType.DEAD
            and (exclude_engine_index is None or i not in exclude_engine_index)
        ]

        kwargs["timeout"] = timeout
        kwargs["soft_pause"] = soft_pause

        ft_results = await self._execute_cmd_on_engines(
            ft_request=FaultToleranceRequest(
                request_id=str(uuid.uuid4()),
                instruction="pause",
                params=kwargs,
            ),
            target_engines=[
                self.engine_identities[i - self.start_rank] for i in alive_engines
            ],
        )

        return all([res.success for res in ft_results])

    async def _pub_engine_status(self):
        # No lock needed since we're single-threaded in asyncio
        engine_status = self.engine_status_dict.copy()
        topic = self.ft_config.fault_state_pub_topic.encode()
        await self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(engine_status))
        )

    async def _execute_cmd_on_engines(
        self, ft_request: FaultToleranceRequest, target_engines: list[bytes]
    ) -> list[FaultToleranceResult]:
        coroutines = []
        for core_engine in target_engines:
            coro = self.call_utility_async(
                "handle_fault", ft_request, engine=core_engine
            )
            coroutines.append(coro)
        results = await asyncio.gather(*coroutines)
        results = [FaultToleranceResult(**res) for res in results]
        return results

    async def _monitor_and_pause_on_fault(self):
        """Receive fault info from engine and pause engines if happened."""
        try:
            while not self.sentinel_dead:
                _, _, message = await self.fault_receiver_socket.recv_multipart(
                    copy=False
                )  # type: ignore
                fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
                if fault_info.type == "EngineLoopPausedError":
                    engine_status = EngineStatusType.PAUSED
                elif fault_info.type == "EngineDeadError":
                    engine_status = EngineStatusType.DEAD
                else:
                    engine_status = EngineStatusType.UNHEALTHY
                # Update engine status
                self.engine_status_dict[int(fault_info.engine_id)] = {
                    "status": engine_status
                }

                await self._pub_engine_status()
                if not self.is_faulted.is_set():
                    self.is_faulted.set()
                    await self.pause(timeout=self.ft_config.gloo_comm_timeout + 5)

        except zmq.ZMQError:
            self.logger("Fault receiver socket closed, stopping async monitor.")

    async def run(self):
        """Poll and execute fault tolerance commands."""
        generic_decoder = MsgpackDecoder()
        # Initialize input sockets
        for input_socket in self.input_sockets:
            await input_socket.send(b"")

        poller = zmq.asyncio.Poller()
        for sock in self.input_sockets:
            poller.register(sock, zmq.POLLIN)

        while not self.sentinel_dead:
            try:
                events = await poller.poll(timeout=100)
                if not events:
                    continue
                for sock, event in events:
                    _, *msg = await sock.recv_multipart(copy=False)
                    client_index, call_id, _, ft_args = generic_decoder.decode(msg)
                    ft_request = FaultToleranceRequest(**ft_args[0])
                    ft_result = await self._execute_cmd(ft_request)
                    await self._send_utility_result(client_index, call_id, ft_result)
            except zmq.ZMQError:
                self.logger("Sockets closed, terminating.")
                self.sentinel_dead = True

    async def _execute_cmd(self, ft_request):
        success, reason = False, None
        try:
            success = await getattr(self, ft_request.instruction)(**ft_request.params)
        except Exception as e:
            self.logger(f"Error processing command: {e}", level="error")
            reason = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        return FaultToleranceResult(
            success=success,
            request_id=ft_request.request_id,
            reason=reason if not success else None,
        )

    def shutdown(self):
        self.sentinel_dead = True
        close_sockets([self.fault_receiver_socket, self.fault_state_pub_socket])
        close_sockets(self.input_sockets + self.output_sockets)
        self.ctx_async.term()
        super().shutdown()
