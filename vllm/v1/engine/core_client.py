# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import queue
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from threading import Thread
from typing import Any, Callable, Optional, TypeVar, Union

import msgspec
import zmq
import zmq.asyncio

from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import (get_open_port, get_open_zmq_inproc_path,
                        get_open_zmq_ipc_path, get_tcp_uri, make_zmq_socket)
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.executor.abstract import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr
from vllm.v1.utils import CoreEngineProcManager

logger = init_logger(__name__)

AnyFuture = Union[asyncio.Future[Any], Future[Any]]

_R = TypeVar('_R')  # Return type for collective_rpc

STARTUP_POLL_PERIOD_MS = 10000


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
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":

        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported.")

        if multiprocess_mode and asyncio_mode:
            if vllm_config.parallel_config.data_parallel_size > 1:
                return DPAsyncMPClient(vllm_config, executor_class, log_stats)

            return AsyncMPClient(vllm_config, executor_class, log_stats)

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @abstractmethod
    def shutdown(self):
        ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def profile(self, is_start: bool = True) -> None:
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        raise NotImplementedError

    def reset_prefix_cache(self) -> None:
        raise NotImplementedError

    def sleep(self, level: int = 1) -> None:
        raise NotImplementedError

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self) -> None:
        raise NotImplementedError

    async def execute_dummy_batch_async(self) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def save_sharded_state(self,
                           path: str,
                           pattern: Optional[str] = None,
                           max_size: Optional[int] = None) -> None:
        raise NotImplementedError

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    async def profile_async(self, is_start: bool = True) -> None:
        raise NotImplementedError

    async def reset_mm_cache_async(self) -> None:
        raise NotImplementedError

    async def reset_prefix_cache_async(self) -> None:
        raise NotImplementedError

    async def sleep_async(self, level: int = 1) -> None:
        raise NotImplementedError

    async def wake_up_async(self, tags: Optional[list[str]] = None) -> None:
        raise NotImplementedError

    async def is_sleeping_async(self) -> bool:
        raise NotImplementedError

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    async def remove_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def list_loras_async(self) -> set[int]:
        raise NotImplementedError

    async def pin_lora_async(self, lora_id: int) -> bool:
        raise NotImplementedError

    async def save_sharded_state_async(self,
                                       path: str,
                                       pattern: Optional[str] = None,
                                       max_size: Optional[int] = None) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
            self,
            method: Union[str, Callable[..., _R]],
            timeout: Optional[float] = None,
            args: tuple = (),
            kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
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
        outputs, _ = self.engine_core.step()
        return outputs

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    def shutdown(self) -> None:
        self.engine_core.shutdown()

    def profile(self, is_start: bool = True) -> None:
        self.engine_core.profile(is_start)

    def reset_mm_cache(self) -> None:
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(self) -> None:
        self.engine_core.reset_prefix_cache()

    def sleep(self, level: int = 1) -> None:
        self.engine_core.sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self) -> None:
        self.engine_core.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(self,
                           path: str,
                           pattern: Optional[str] = None,
                           max_size: Optional[int] = None) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)


class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.index = index
        self.identity = index.to_bytes(length=2, byteorder="little")

        self.state = CoreEngineState.NEW
        self.num_reqs_in_flight = 0


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: Union[zmq.Context]
    local_engine_manager: Optional[CoreEngineProcManager] = None
    output_socket: Optional[Union[zmq.Socket, zmq.asyncio.Socket]] = None
    input_socket: Optional[Union[zmq.Socket, zmq.asyncio.Socket]] = None
    output_queue_task: Optional[asyncio.Task] = None
    shutdown_path: Optional[str] = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        self.engine_dead = True
        if self.local_engine_manager is not None:
            self.local_engine_manager.close()

        if self.output_queue_task is not None:
            self.output_queue_task.cancel()

        # ZMQ context termination can hang if the sockets
        # aren't explicitly closed first.
        if self.output_socket is not None:
            self.output_socket.close(linger=0)
        if self.input_socket is not None:
            self.input_socket.close(linger=0)
        if self.shutdown_path is not None:
            # We must ensure that the sync output socket is
            # closed cleanly in its own thread.
            with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                shutdown_sender.connect(self.shutdown_path)
                # Send shutdown signal.
                shutdown_sender.send(b'')

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer
                                 == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


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
        executor_class: type[Executor],
        log_stats: bool,
    ):
        self.vllm_config = vllm_config
        # Serialization setup.
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup.
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            parallel_config = vllm_config.parallel_config
            local_engine_count = parallel_config.data_parallel_size_local
            start_index = parallel_config.data_parallel_rank
            local_start_index = parallel_config.data_parallel_rank_local

            # SPMD mode is where there is an LLM instance per DP rank and
            # one core engine per LLM, see
            # examples/offline_inference/data_parallel.py.
            spmd_mode = local_start_index is not None
            if spmd_mode:
                assert local_engine_count == 1
                self.core_engines = [
                    CoreEngine(index=local_start_index, local=True)
                ]
            else:
                assert start_index == 0
                local_start_index = 0
                self.core_engines = [
                    CoreEngine(index=i, local=(i < local_engine_count))
                    for i in range(parallel_config.data_parallel_size)
                ]

            input_address, output_address = self._get_zmq_addresses(
                parallel_config, spmd_mode)

            # Create input and output sockets.
            self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True)

            self.resources.output_socket = make_zmq_socket(
                self.ctx, output_address, zmq.constants.PULL)
            # Start local engines.
            if local_engine_count:
                # In server mode, start_index and local_start_index will
                # both be 0.
                self.resources.local_engine_manager = CoreEngineProcManager(
                    EngineCoreProc.run_engine_core,
                    vllm_config=vllm_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    input_address=input_address,
                    on_head_node=True,
                    local_engine_count=local_engine_count,
                    start_index=start_index,
                    local_start_index=local_start_index)

            self.core_engine = self.core_engines[0]

            # Wait for engine core process(es) to start.
            self._wait_for_engine_startup(output_address, parallel_config)

            self.utility_results: dict[int, AnyFuture] = {}

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            success = True
        finally:
            if not success:
                self._finalizer()

    @staticmethod
    def _get_zmq_addresses(parallel_config: ParallelConfig,
                           spmd_mode: bool) -> tuple[str, str]:
        """Returns (input_address, output_address)."""
        dp_size = parallel_config.data_parallel_size
        local_engine_count = parallel_config.data_parallel_size_local

        if local_engine_count == dp_size or spmd_mode:
            input_address = get_open_zmq_ipc_path()
            output_address = get_open_zmq_ipc_path()
        else:
            host = parallel_config.data_parallel_master_ip
            input_port = parallel_config.data_parallel_rpc_port
            output_port = get_open_port()
            input_address = get_tcp_uri(host, input_port)
            output_address = get_tcp_uri(host, output_port)

        return input_address, output_address

    def _wait_for_engine_startup(self, output_address: str,
                                 parallel_config: ParallelConfig):
        # Get a sync handle to the socket which can be sync or async.
        sync_input_socket = zmq.Socket.shadow(self.input_socket)

        # Wait for engine core process(es) to send ready messages.
        local_count = parallel_config.data_parallel_size_local
        remote_count = len(self.core_engines) - local_count
        # [local, remote] counts
        conn_pending, start_pending = [local_count, remote_count], [0, 0]

        poller = zmq.Poller()
        poller.register(sync_input_socket, zmq.POLLIN)
        proc_manager = self.resources.local_engine_manager
        if proc_manager is not None:
            for sentinel in proc_manager.sentinels():
                poller.register(sentinel, zmq.POLLIN)
        while any(conn_pending) or any(start_pending):
            events = poller.poll(STARTUP_POLL_PERIOD_MS)
            if not events:
                if any(conn_pending):
                    logger.debug(
                        "Waiting for %d local, %d remote core engine proc(s) "
                        "to connect.", *conn_pending)
                if any(start_pending):
                    logger.debug(
                        "Waiting for %d local, %d remote core engine proc(s) "
                        "to start.", *start_pending)
                continue
            if len(events) > 1 or events[0][0] != sync_input_socket:
                # One of the local core processes exited.
                finished = proc_manager.finished_procs(
                ) if proc_manager else {}
                raise RuntimeError("Engine core initialization failed. "
                                   "See root cause above. "
                                   f"Failed core proc(s): {finished}")

            # Receive HELLO and READY messages from the input socket.
            eng_identity, ready_msg_bytes = sync_input_socket.recv_multipart()
            eng_index = int.from_bytes(eng_identity, byteorder="little")
            engine = next(
                (e for e in self.core_engines if e.identity == eng_identity),
                None)
            if engine is None:
                raise RuntimeError(f"Message from engine with unexpected data "
                                   f"parallel rank: {eng_index}")
            msg = msgspec.msgpack.decode(ready_msg_bytes)
            status, local = msg["status"], msg["local"]
            if local != engine.local:
                raise RuntimeError(f"{status} message from "
                                   f"{'local' if local else 'remote'} "
                                   f"engine {eng_index}, expected it to be "
                                   f"{'local' if engine.local else 'remote'}")

            if status == "HELLO" and engine.state == CoreEngineState.NEW:

                # Send init message with DP config info.
                init_message = self.encoder.encode({
                    "output_socket_address": output_address,
                    "parallel_config": {
                        "data_parallel_master_ip":
                        parallel_config.data_parallel_master_ip,
                        "data_parallel_master_port":
                        parallel_config.data_parallel_master_port,
                        "data_parallel_size":
                        parallel_config.data_parallel_size,
                    },
                })
                sync_input_socket.send_multipart((eng_identity, *init_message),
                                                 copy=False)
                conn_pending[0 if local else 1] -= 1
                start_pending[0 if local else 1] += 1
                engine.state = CoreEngineState.CONNECTED
            elif status == "READY" and (engine.state
                                        == CoreEngineState.CONNECTED):
                # Setup KV cache config with initialization state from
                # engine core process. Sum values from all engines in DP case.
                cache_config = self.vllm_config.cache_config
                num_gpu_blocks = cache_config.num_gpu_blocks or 0
                num_gpu_blocks += msg['num_gpu_blocks']
                cache_config.num_gpu_blocks = num_gpu_blocks

                start_pending[0 if local else 1] -= 1
                engine.state = CoreEngineState.READY
            else:
                raise RuntimeError(f"Unexpected {status} message for "
                                   f"{'local' if local else 'remote'} engine "
                                   f"{eng_index} in {engine.state} state.")

            logger.debug("%s from %s core engine process %s.", status,
                         "local" if local else "remote", eng_index)

    def shutdown(self):
        # Terminate background resources.
        self._finalizer()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return EngineDeadError(
            suppress_context=True) if self.resources.engine_dead else e

    def ensure_alive(self):
        if self.resources.engine_dead:
            raise EngineDeadError()

    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    def free_pending_messages(self):
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()


def _process_utility_output(output: UtilityOutput,
                            utility_results: dict[int, AnyFuture]):
    """Set the result from a utility method in the waiting future"""
    future = utility_results.pop(output.call_id)
    if output.failure_message is not None:
        future.set_exception(Exception(output.failure_message))
    else:
        future.set_result(output.result)


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.outputs_queue = queue.Queue[Union[EngineCoreOutputs, Exception]]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        out_socket = self.resources.output_socket
        assert out_socket is not None
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket)
                poller.register(out_socket)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output,
                                                utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(target=process_outputs_socket,
                                          name="EngineCoreOutputQueueThread",
                                          daemon=True)
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        self.free_pending_messages()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine.identity, request_type.value,
               *self.encoder.encode(request))

        if len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            self.input_socket.send_multipart(msg, copy=False)
            return

        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        self.add_pending_message(tracker, request)

    def call_utility(self, method: str, *args) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: Future[Any] = Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY,
                         (call_id, method, args))

        return future.result()

    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def profile(self, is_start: bool = True) -> None:
        self.call_utility("profile", is_start)

    def reset_mm_cache(self) -> None:
        self.call_utility("reset_mm_cache")

    def reset_prefix_cache(self) -> None:
        self.call_utility("reset_prefix_cache")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.call_utility("add_lora", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.call_utility("remove_lora", lora_id)

    def list_loras(self) -> set[int]:
        return self.call_utility("list_loras")

    def pin_lora(self, lora_id: int) -> bool:
        return self.call_utility("pin_lora", lora_id)

    def sleep(self, level: int = 1) -> None:
        self.call_utility("sleep", level)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self) -> None:
        self.call_utility("execute_dummy_batch")

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args,
                                 kwargs)

    def save_sharded_state(self,
                           path: str,
                           pattern: Optional[str] = None,
                           max_size: Optional[int] = None) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.outputs_queue = asyncio.Queue[Union[EngineCoreOutputs,
                                                 Exception]]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        output_handler: Optional[Callable[[AsyncMPClient, EngineCoreOutputs],
                                          Awaitable[None]]] = getattr(
                                              self.__class__,
                                              "process_engine_outputs", None)
        _self_ref = weakref.ref(self) if output_handler else None
        output_socket = resources.output_socket
        assert output_socket is not None

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output,
                                                utility_results)
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask")

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(self,
                    request_type: EngineCoreRequestType,
                    request: Any,
                    engine: Optional[CoreEngine] = None) -> Awaitable[Any]:
        self.ensure_alive()
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    def _send_input_message(self, message: tuple[bytestr,
                                                 ...], engine: CoreEngine,
                            objects: Any) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine.identity, ) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    async def call_utility_async(self, method: str, *args) -> Any:
        return await self._call_utility_async(method,
                                              *args,
                                              engine=self.core_engine)

    async def _call_utility_async(self, method: str, *args,
                                  engine: CoreEngine) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (EngineCoreRequestType.UTILITY.value, *self.encoder.encode(
            (call_id, method, args)))
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    async def profile_async(self, is_start: bool = True) -> None:
        await self.call_utility_async("profile", is_start)

    async def reset_mm_cache_async(self) -> None:
        await self.call_utility_async("reset_mm_cache")

    async def reset_prefix_cache_async(self) -> None:
        await self.call_utility_async("reset_prefix_cache")

    async def sleep_async(self, level: int = 1) -> None:
        await self.call_utility_async("sleep", level)

    async def wake_up_async(self, tags: Optional[list[str]] = None) -> None:
        await self.call_utility_async("wake_up", tags)

    async def is_sleeping_async(self) -> bool:
        return await self.call_utility_async("is_sleeping")

    async def execute_dummy_batch_async(self) -> None:
        await self.call_utility_async("execute_dummy_batch")

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return await self.call_utility_async("add_lora", lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("remove_lora", lora_id)

    async def list_loras_async(self) -> set[int]:
        return await self.call_utility_async("list_loras")

    async def pin_lora_async(self, lora_id: int) -> bool:
        return await self.call_utility_async("pin_lora", lora_id)

    async def save_sharded_state_async(self,
                                       path: str,
                                       pattern: Optional[str] = None,
                                       max_size: Optional[int] = None) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern,
                                      max_size)

    async def collective_rpc_async(
            self,
            method: Union[str, Callable[..., _R]],
            timeout: Optional[float] = None,
            args: tuple = (),
            kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return await self.call_utility_async("collective_rpc", method, timeout,
                                             args, kwargs)


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore."""

    def __init__(self, vllm_config: VllmConfig, executor_class: type[Executor],
                 log_stats: bool):

        self.current_wave = 0
        self.engines_running = False
        self.reqs_in_flight: dict[str, CoreEngine] = {}

        super().__init__(vllm_config, executor_class, log_stats)

        assert len(self.core_engines) > 1

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (await asyncio.gather(*[
            self._call_utility_async(method, *args, engine=engine)
            for engine in self.core_engines
        ]))[0]

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.current_wave = self.current_wave

        chosen_engine = self.get_core_engine_for_request()
        self.reqs_in_flight[request.request_id] = chosen_engine
        chosen_engine.num_reqs_in_flight += 1

        to_await = self._send_input(EngineCoreRequestType.ADD, request,
                                    chosen_engine)
        if not self.engines_running:
            # Send request to chosen engine and dp start loop
            # control message to all other engines.
            self.engines_running = True
            to_await = asyncio.gather(
                to_await,  # type: ignore[assignment]
                *self._start_wave_coros(exclude_index=chosen_engine.index))

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self) -> CoreEngine:
        return min(self.core_engines, key=lambda e: e.num_reqs_in_flight)

    @staticmethod
    async def process_engine_outputs(self: "DPAsyncMPClient",
                                     outputs: EngineCoreOutputs):
        if self.reqs_in_flight:
            for req_id in outputs.finished_requests or ():
                if engine := self.reqs_in_flight.pop(req_id, None):
                    engine.num_reqs_in_flight -= 1

        if outputs.wave_complete is not None:
            # Current wave is complete, move to next wave number
            # and mark engines as paused.
            if self.current_wave <= outputs.wave_complete:
                self.current_wave = outputs.wave_complete + 1
                self.engines_running = False

        elif outputs.start_wave is not None and (
                outputs.start_wave > self.current_wave or
            (outputs.start_wave == self.current_wave
             and not self.engines_running)):
            # Engine received request for a non-current wave so we must ensure
            # that other engines progress to the next wave.
            self.current_wave = outputs.start_wave
            self.engines_running = True
            await asyncio.gather(*self._start_wave_coros(
                exclude_index=outputs.engine_index))

    def _start_wave_coros(self, exclude_index: int) -> list[Awaitable[None]]:
        logger.debug("Sending start DP wave %d.", self.current_wave)
        return [
            self._send_input(EngineCoreRequestType.START_DP_WAVE,
                             self.current_wave, engine)
            for engine in self.core_engines if engine.index != exclude_index
        ]

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine: dict[CoreEngine, list[str]] = {}
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine.setdefault(engine, []).append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(self, request_ids: list[str],
                              engine: CoreEngine) -> None:
        if not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids,
                                   engine)
