# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
import weakref
from abc import ABC, abstractmethod
from typing import List, Optional, Type

import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreProfile,
                            EngineCoreRequest, EngineCoreRequestType,
                            EngineCoreRequestUnion, EngineCoreResetPrefixCache)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import MsgpackDecoder, PickleEncoder
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
        # Serialization setup.
        self.encoder = PickleEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

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
        self.engine_core_errored = False
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
        self.proc_handle.wait_for_startup()

    def shutdown(self):
        """Clean up background resources."""

        self.proc_handle.shutdown()
        self._finalizer()

    def _sigusr1_handler(self):
        """
        EngineCoreProc sends SIGUSR1 if it encounters an Exception.
        Set self in errored state and begin shutdown.
        """
        logger.fatal("Got fatal signal from EngineCore, shutting down.")
        self.engine_core_errored = True
        self.shutdown()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""

        return (EngineDeadError(
            "EngineCore encountered an issue. See stack trace "
            "for the root cause.",
            suppress_context=True) if self.engine_core_errored else e)


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig,
                 executor_class: Type[Executor]):

        # Setup EngineCore signal handler.
        def sigusr1_handler(signum, frame):
            self._sigusr1_handler()

        signal.signal(signal.SIGUSR1, sigusr1_handler)

        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=False,
        )

    def get_output(self) -> EngineCoreOutputs:

        try:
            (frame, ) = self.output_socket.recv_multipart(copy=False)
            engine_core_outputs = self.decoder.decode(frame.buffer)
            return engine_core_outputs
        except Exception as e:
            raise self._format_exception(e) from None

    def _send_input(self, request_type: EngineCoreRequestType,
                    request: EngineCoreRequestUnion) -> None:
        try:
            # (RequestType, SerializedRequest)
            msg = (request_type.value, self.encoder.encode(request))
            self.input_socket.send_multipart(msg, copy=False)
        except Exception as e:
            raise self._format_exception(e) from None

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

        # EngineCore sends SIGUSR1 when it gets an Exception.
        # NOTE: super().__init__ blocks the event loop until
        # background procs are setup. This handler allows us
        # to catch  issues during startup (e.g. OOM). We switch
        # to a signal handler in the event loop __init__.
        def sigusr1_handler(signum, frame):
            self._sigusr1_handler()

        # signal.signal(signal.SIGUSR1, sigusr1_handler)

        # Initialize EngineCore + all background processes.
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

        self.queue_task: Optional[asyncio.Task] = None

    async def get_output_async(self) -> EngineCoreOutputs:
        try:
            (frame, ) = await self.output_socket.recv_multipart(copy=False)
            return self.decoder.decode(frame.buffer)
        except Exception as e:
            raise self._format_exception(e) from None

    async def _send_input(self, request_type: EngineCoreRequestType,
                          request: EngineCoreRequestUnion) -> None:
        try:
            msg = (request_type.value, self.encoder.encode(request))
            await self.input_socket.send_multipart(msg, copy=False)
        except Exception as e:
            raise self._format_exception(e) from None

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
