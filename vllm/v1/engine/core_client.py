from abc import ABC, abstractmethod
from typing import List

import msgspec
import zmq
import zmq.asyncio

from vllm.logger import init_logger
from vllm.utils import get_open_zmq_ipc_path

from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine import (POLLING_TIMEOUT_MS, EngineCoreOutput,
                            EngineCoreOutputs, EngineCoreRequest)

class EngineCoreClient:

    def __init__(
        self,
        *args,
        multiprocess_mode: bool,
        asyncio_mode: bool,
        **kwargs,
    ):
        self.client = ClientProtocol.make_client(
            *args,
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=asyncio_mode,
            **kwargs,
        )
    
    def get_output(self) -> List[EngineCoreOutput]:
        """Get EngineCoreOutput from the EngineCore."""

        return self.client.get_output()
    
    async def get_output_async(self) -> List[EngineCoreOutput]:
        """Get EngineCoreOutput from EngineCore in asyncio."""

        await self.client.get_output_async()

    def add_request(self, request: EngineCoreRequest) -> None:
        """Add EngineCoreRequest to EngineCore."""

        return self.client.add_request(request)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        """Add EngineCoreRequest to EngineCore in asyncio."""

        return await self.client.add_request_async(request)


class ClientProtocol(ABC):

    @staticmethod
    def make_client(
        *args,
        multiprocess_mode: bool,
        asyncio_mode: bool,
        **kwargs,
    ) -> "ClientProtocol":

        if asyncio_mode and not multiprocess_mode:
                raise NotImplementedError(
                    "Running EngineCore in asyncio without multiprocessing "
                    "is not currently supported."
                )
        
        if asyncio_mode:
            return AsyncMPClient(*args, **kwargs)

        if multiprocess_mode:
            return MPClient(*args, **kwargs)
        
        return InprocClient(*args, **kwargs)

    @abstractmethod
    def get_output(self) -> List[EngineCoreOutput]:
        ...

    @abstractmethod
    def add_request(self, request: EngineCoreRequest) -> None:
        ...

    async def get_output_async(self) -> List[EngineCoreOutput]:
        raise NotImplementedError(
            "Running in asyncio requires using the AsyncMQClient")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError(
            "Running in asyncio requires using the AsyncMQClient")


class InprocClient(ClientProtocol):
    """In process client for the EngineCore."""

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)
    
    def get_output(self) -> List[EngineCoreOutput]:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)


class MPClient(ClientProtocol):
    """Multiprocess client for the EngineCore."""

    def __init__(
        self,
        *args,
        use_zmq_asyncio: bool = False,
        **kwargs,
    ):
        # Serialization setup.
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # IPC Setup
        self.ctx = zmq.asyncio.Context() if use_zmq_asyncio else zmq.Context()

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
        self.proc = EngineCoreProc.make_engine_core_process(
            *args,
            input_path=input_path,
            output_path=output_path,
            ready_path=ready_path,
            **kwargs,
        )
        self.proc.start()
        EngineCoreProc.wait_for_startup(self.proc, ready_path)

    def __del__(self):
        # Hack.
        if hasattr(self, "proc"):
            self.proc.kill()

    def get_output(self) -> List[EngineCoreOutput]:
        """Get EngineCoreOutput from the EngineCore (non-blocking)."""

        while self.output_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
            logger.debug("Waiting for output from EngineCore.")

        frames = self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        return engine_core_outputs

    def add_request(self, request: EngineCoreRequest) -> None:
        """Add EngineCoreRequest to EngineCore (non-blocking)."""

        self.input_socket.send_multipart((self.encoder.encode(request), ),
                                         copy=False,
                                         flags=zmq.NOBLOCK)
        
    async def get_output_async(self) -> List[EngineCoreOutput]:
        raise NotImplementedError(
            "Use the AsyncMQClient for this coroutine.")

    async def get_output_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError(
            "Use the AsyncMQClient for this coroutine.")


class AsyncMPClient(MPClient):
    """Asyncio-compatible multiprocess client for the EngineCore."""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, 
                         use_zmq_asyncio=True,
                         **kwargs)

    def get_output(self) -> List[EngineCoreOutput]:
        raise NotImplementedError(
            "AsyncMQClient uses the get_output_async coroutine for this method.")

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError(
            "AsyncMQClient uses the add_request_async coroutine for this method.")
        
    async def get_output_async(self) -> List[EngineCoreOutput]:
        """Get EngineCoreOutput from EngineCore (non-blocking) in asyncio."""

        while await self.output_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
            logger.debug("Waiting for output from EngineCore.")

        frames = await self.output_socket.recv_multipart(copy=False)
        engine_core_outputs = self.decoder.decode(frames[0].buffer).outputs

        return engine_core_outputs

    async def get_output_async(self, request: EngineCoreRequest) -> None:
        """Add EngineCoreRequest to EngineCore (non-blocking) in asyncio."""

        await self.input_socket.send_multipart(
            (self.encoder.encode(request), ), copy=False, flags=zmq.NOBLOCK)
