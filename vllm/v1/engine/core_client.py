from typing import List, Optional, Type

import msgspec
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreProfile, EngineCoreRequest,
                            EngineCoreRequestType, EngineCoreRequestUnion)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import PickleEncoder
from vllm.v1.utils import BackgroundProcHandle

logger = init_logger(__name__)


class EngineCoreClient:
    """
    EngineCoreClient: subclasses handle different methods for pushing 
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    @staticmethod
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool = False,
    ) -> "EngineCoreClient":

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported.")

        if multiprocess_mode and asyncio_mode:
            return AsyncMPClient(vllm_config, executor_class, log_stats)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    def shutdown(self):
        pass

    def get_output(self) -> List[EngineCoreOutput]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: List[str]) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> List[EngineCoreOutput]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(self, is_start: bool = True) -> None:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended 
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore

        TODO: support asyncio-mode for debugging.
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> List[EngineCoreOutput]:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: List[str]) -> None:
        self.engine_core.abort_requests(request_ids)

    def shutdown(self):
        self.engine_core.shutdown()

    def __del__(self):
        self.shutdown()

    def profile(self, is_start: bool = True) -> None:
        self.engine_core.profile(is_start)


class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    
        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: Type[Executor],
        log_stats: bool = False,
    ):
        # Serialization setup.
        self.encoder = PickleEncoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # ZMQ setup.
        if asyncio_mode:
            self.ctx = zmq.asyncio.Context()
        else:
            self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Paths and sockets for IPC.
        output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()
        self.output_socket = make_zmq_socket(self.ctx, output_path,
                                             zmq.constants.PULL)
        self.input_socket = make_zmq_socket(self.ctx, input_path,
                                            zmq.constants.PUSH)

        # Start EngineCore in background process.
        self.proc_handle: Optional[BackgroundProcHandle]
        self.proc_handle = BackgroundProcHandle(
            input_path=input_path,
            output_path=output_path,
            process_name="EngineCore",
            target_fn=EngineCoreProc.run_engine_core,
            process_kwargs={
                "vllm_config": vllm_config,
                "executor_class": executor_class,
                "log_stats": log_stats,
            })

    def shutdown(self):
        # Shut down the zmq context.
        self.ctx.destroy(linger=0)

        if hasattr(self, "proc_handle") and self.proc_handle:
            self.proc_handle.shutdown()
            self.proc_handle = None


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: Type[Executor],
                 log_stats: bool = False):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

    def get_output(self) -> List[EngineCoreOutput]:

        (frame, ) = self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frame.buffer).outputs
        return engine_core_outputs

    def _send_input(self, request_type: EngineCoreRequestType,
                    request: EngineCoreRequestUnion) -> None:

        # (RequestType, SerializedRequest)
        msg = (request_type.value, self.encoder.encode(request))
        self.input_socket.send_multipart(msg, copy=False)

    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: List[str]) -> None:
        self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True) -> None:
        self._send_input(EngineCoreRequestType.PROFILE,
                         EngineCoreProfile(is_start))


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: Type[Executor],
                 log_stats: bool = False):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

    async def get_output_async(self) -> List[EngineCoreOutput]:

        frames = await self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        return engine_core_outputs

    async def _send_input(self, request_type: EngineCoreRequestType,
                          request: EngineCoreRequestUnion) -> None:

        msg = (request_type.value, self.encoder.encode(request))
        await self.input_socket.send_multipart(msg, copy=False)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self._send_input(EngineCoreRequestType.ADD, request)

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self._send_input(EngineCoreRequestType.PROFILE,
                               EngineCoreProfile(is_start))
