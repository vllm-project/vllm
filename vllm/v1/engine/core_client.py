import multiprocessing
import time
from typing import List, Union

import msgspec
import zmq
import zmq.asyncio

from vllm.logger import init_logger
from vllm.utils import get_open_zmq_ipc_path
from vllm.v1.engine import (EngineCoreOutput, EngineCoreOutputs,
                            EngineCoreProfile, EngineCoreRequest,
                            EngineCoreRequestType)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.serial_utils import PickleEncoder

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
        *args,
        multiprocess_mode: bool,
        asyncio_mode: bool,
        **kwargs,
    ) -> "EngineCoreClient":

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported.")

        if multiprocess_mode and asyncio_mode:
            return AsyncMPClient(*args, **kwargs)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(*args, **kwargs)

        return InprocClient(*args, **kwargs)

    def shutdown(self):
        pass

    def get_output(self) -> List[EngineCoreOutput]:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile(self, is_start=True) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: List[str]) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> List[EngineCoreOutput]:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
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

    async def profile(self, is_start=True) -> None:
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
        *args,
        asyncio_mode: bool,
        **kwargs,
    ):
        # Serialization setup.
        self.encoder = PickleEncoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # ZMQ setup.
        self.ctx = (zmq.asyncio.Context() if asyncio_mode else zmq.Context())

        # Path for IPC.
        ready_path = get_open_zmq_ipc_path()
        output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()

        # Get output (EngineCoreOutput) from EngineCore.
        self.output_socket = self.ctx.socket(zmq.constants.PULL)
        self.output_socket.connect(output_path)

        # Send input (EngineCoreRequest) to EngineCore.
        self.input_socket = self.ctx.socket(zmq.constants.PUSH)
        self.input_socket.bind(input_path)

        # Start EngineCore in background process.
        self.should_shutdown = multiprocessing.Value('b', False, lock=False)
        self.proc = EngineCoreProc.make_engine_core_process(
            *args,
            input_path=input_path,
            output_path=output_path,
            ready_path=ready_path,
            should_shutdown=self.should_shutdown,
            **kwargs,
        )

    def shutdown(self):
        # Send shutdown signal to background process.
        self.should_shutdown = True

        # Shut down the zmq context.
        self.ctx.destroy(linger=0)

        # Shutdown the process if needed.
        if hasattr(self, "proc") and self.proc.is_alive():
            self.proc.terminate()

            time.sleep(5)
            if self.proc.is_alive():
                self.proc.kill()

    def __del__(self):
        self.shutdown()


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, asyncio_mode=False, **kwargs)

    def get_output(self) -> List[EngineCoreOutput]:

        (frame, ) = self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frame.buffer).outputs
        return engine_core_outputs

    def _send_input(
        self, request_type: EngineCoreRequestType,
        request: Union[EngineCoreRequest, EngineCoreProfile,
                       List[str]]) -> None:

        # (RequestType, SerializedRequest)
        msg = (request_type.value, self.encoder.encode(request))
        self.input_socket.send_multipart(msg, copy=False)

    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: List[str]) -> None:
        self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile(self, is_start=True) -> None:
        self._send_input(EngineCoreRequestType.PROFILE,
                         EngineCoreProfile(is_start))


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, asyncio_mode=True, **kwargs)

    async def get_output_async(self) -> List[EngineCoreOutput]:

        frames = await self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        return engine_core_outputs

    async def _send_input(
        self, request_type: EngineCoreRequestType,
        request: Union[EngineCoreRequest, EngineCoreProfile,
                       List[str]]) -> None:

        msg = (request_type.value, self.encoder.encode(request))
        await self.input_socket.send_multipart(msg, copy=False)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self._send_input(EngineCoreRequestType.ADD, request)

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile(self, is_start=True) -> None:
        await self._send_input(EngineCoreRequestType.PROFILE,
                               EngineCoreProfile(is_start))
