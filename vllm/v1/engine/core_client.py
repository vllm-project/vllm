import asyncio
import os
import signal
import weakref
from abc import ABC, abstractmethod
from typing import List, Optional, Type

import msgspec
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import (get_open_zmq_ipc_path, kill_process_tree,
                        make_zmq_socket)
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreProfile,
                            EngineCoreRequest, EngineCoreRequestType,
                            EngineCoreRequestUnion, EngineCoreResetPrefixCache)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import PickleEncoder
from vllm.v1.utils import BackgroundProcHandle

logger = init_logger(__name__)


class EngineCoreClient(ABC):
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
    ) -> "EngineCoreClient":

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported.")

        if multiprocess_mode and asyncio_mode:
            return AsyncMPClient(vllm_config, executor_class)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class)

        return InprocClient(vllm_config, executor_class)

    @abstractmethod
    def shutdown(self):
        ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True) -> None:
        raise NotImplementedError

    def reset_prefix_cache(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: List[str]) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(self, is_start: bool = True) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(self) -> None:
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
    """

    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True) -> None:
        self.engine_core.profile(is_start)

    def reset_prefix_cache(self) -> None:
        self.engine_core.reset_prefix_cache()


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
        log_stats: bool,
    ):
        # The child processes will send SIGUSR1 when unrecoverable
        # errors happen. We kill the process tree here so that the
        # stack trace is very evident.
        # TODO(rob): rather than killing the main process, we should
        # figure out how to raise an AsyncEngineDeadError and
        # handle at the API server level so we can return a better
        # error code to the clients calling VLLM.
        def sigusr1_handler(signum, frame):
            logger.fatal("Got fatal signal from worker processes, shutting "
                         "down. See stack trace above for root cause issue.")
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        # Serialization setup.
        self.encoder = PickleEncoder()
        self.decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        # ZMQ setup.
        self.ctx = (
            zmq.asyncio.Context()  # type: ignore[attr-defined]
            if asyncio_mode else zmq.Context())  # type: ignore[attr-defined]

        # Note(rob): shutdown function cannot be a bound method,
        # else the gc cannot collect the object.
        self._finalizer = weakref.finalize(self, lambda x: x.destroy(linger=0),
                                           self.ctx)

        # Paths and sockets for IPC.
        output_path = get_open_zmq_ipc_path()
        input_path = get_open_zmq_ipc_path()
        self.output_socket = make_zmq_socket(self.ctx, output_path,
                                             zmq.constants.PULL)
        self.input_socket = make_zmq_socket(self.ctx, input_path,
                                            zmq.constants.PUSH)

        # Start EngineCore in background process.
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
        """Clean up background resources."""
        if hasattr(self, "proc_handle"):
            self.proc_handle.shutdown()

        self._finalizer()


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig,
                 executor_class: Type[Executor]):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
        )

    def get_output(self) -> EngineCoreOutputs:

        (frame, ) = self.output_socket.recv_multipart(copy=False)
        return self.decoder.decode(frame.buffer)

    def _send_input(self, request_type: EngineCoreRequestType,
                    request: EngineCoreRequestUnion) -> None:

        # (RequestType, SerializedRequest)
        msg = (request_type.value, self.encoder.encode(request))
        self.input_socket.send_multipart(msg, copy=False)

    def add_request(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True) -> None:
        self._send_input(EngineCoreRequestType.PROFILE,
                         EngineCoreProfile(is_start))

    def reset_prefix_cache(self) -> None:
        self._send_input(EngineCoreRequestType.RESET_PREFIX_CACHE,
                         EngineCoreResetPrefixCache())


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig,
                 executor_class: Type[Executor]):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

        self.outputs_queue: Optional[asyncio.Queue[bytes]] = None
        self.queue_task: Optional[asyncio.Task] = None

    async def get_output_async(self) -> EngineCoreOutputs:
        if self.outputs_queue is None:
            # Perform IO in separate task to parallelize as much as possible
            self.outputs_queue = asyncio.Queue()

            async def process_outputs_socket():
                assert self.outputs_queue is not None
                while True:
                    (frame, ) = await self.output_socket.recv_multipart(
                        copy=False)
                    self.outputs_queue.put_nowait(frame.buffer)

            self.queue_task = asyncio.create_task(process_outputs_socket())

        return self.decoder.decode(await self.outputs_queue.get())

    async def _send_input(self, request_type: EngineCoreRequestType,
                          request: EngineCoreRequestUnion) -> None:

        msg = (request_type.value, self.encoder.encode(request))
        await self.input_socket.send_multipart(msg, copy=False)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # NOTE: text prompt is not needed in the core engine as it has been
        # tokenized.
        request.prompt = None
        await self._send_input(EngineCoreRequestType.ADD, request)

    async def abort_requests_async(self, request_ids: List[str]) -> None:
        if len(request_ids) > 0:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self._send_input(EngineCoreRequestType.PROFILE,
                               EngineCoreProfile(is_start))

    async def reset_prefix_cache_async(self) -> None:
        await self._send_input(EngineCoreRequestType.RESET_PREFIX_CACHE,
                               EngineCoreResetPrefixCache())
