# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import multiprocessing
import queue
import sys
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, TypeAlias, TypeVar

import msgspec.msgpack
import zmq
import zmq.asyncio

from vllm.config import VllmConfig
from vllm.config.parallel import FaultToleranceMode
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.utils.async_utils import in_loop
from vllm.utils.network_utils import (
    close_sockets,
    get_open_port,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
)
from vllm.v1.engine.coordinator import DPCoordinator
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    launch_core_engines,
)
from vllm.v1.executor import Executor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # Return type for collective_rpc

EngineIdentity = bytes


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
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    @staticmethod
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "MPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)

    @abstractmethod
    def shutdown(self): ...

    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
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

    def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    def is_sleeping(self) -> bool:
        raise NotImplementedError

    def execute_dummy_batch(self, cpu_only: bool = False) -> None:
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

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        raise NotImplementedError

    def dp_engines_running(self) -> bool:
        """Returns True id data parallel engines are collectively in a
        running state."""
        raise NotImplementedError

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        raise NotImplementedError

    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
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

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
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

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
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
        outputs, _ = self.engine_core.step_fn()
        return outputs and outputs.get(0) or EngineCoreOutputs()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.engine_core.get_supported_tasks()

    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

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

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.engine_core.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def execute_dummy_batch(self, cpu_only: bool = False) -> None:
        self.engine_core.execute_dummy_batch(cpu_only=cpu_only)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.engine_core.pin_lora(lora_id)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.engine_core.save_sharded_state(path, pattern, max_size)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def dp_engines_running(self) -> bool:
        return False


@dataclass
class BackgroundResources:
    """Used as a finalizer for clean shutdown, avoiding
    circular reference back to the client object."""

    ctx: zmq.Context
    # If CoreEngineProcManager, it manages local engines;
    # if CoreEngineActorManager, it manages all engines.
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None
    coordinator: DPCoordinator | None = None
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
    first_req_send_socket: zmq.asyncio.Socket | None = None
    first_req_rcv_socket: zmq.asyncio.Socket | None = None
    stats_update_socket: zmq.asyncio.Socket | None = None
    output_queue_task: asyncio.Task | None = None
    stats_update_task: asyncio.Task | None = None
    shutdown_path: str | None = None

    # Set if any of the engines are dead. Here so that the output
    # processing threads can access it without holding a ref to the client.
    engine_dead: bool = False

    def __call__(self):
        """Clean up background resources."""

        self.engine_dead = True
        if self.engine_manager is not None:
            self.engine_manager.close()
        if self.coordinator is not None:
            self.coordinator.close()

        if isinstance(self.output_socket, zmq.asyncio.Socket):
            # Async case.
            loop = self.output_queue_task._loop if self.output_queue_task else None

            sockets = (
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
            )

            tasks = (self.output_queue_task, self.stats_update_task)

            def close_sockets_and_tasks():
                close_sockets(sockets)
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            if loop is not None:
                if in_loop(loop):
                    close_sockets_and_tasks()
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)
            else:
                # Loop has been closed, try to clean up directly.
                del tasks
                del close_sockets_and_tasks
                close_sockets(sockets)
                del self.output_queue_task
                del self.stats_update_task
        else:
            # Sync case.

            # ZMQ context termination can hang if the sockets
            # aren't explicitly closed first.
            close_sockets((self.output_socket, self.input_socket))

            if self.shutdown_path is not None:
                # We must ensure that the sync output socket is
                # closed cleanly in its own thread.
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self.shutdown_path)
                    # Send shutdown signal.
                    shutdown_sender.send(b"")

    def validate_alive(self, frames: Sequence[zmq.Frame]):
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            self.engine_dead = True
            logger.error(
                "[EngineDeadError] Detected ENGINE_CORE_DEAD message from engine. "
                "One or more engine core processes have crashed."
            )
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
        client_addresses: dict[str, str] | None = None,
    ):
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.log_stats = log_stats
        
        # Debug: Log fault tolerance config at initialization
        logger.info(
            "[FT] MPClient init: dp_size=%d, enable_eplb=%s, fault_tolerance=%s",
            vllm_config.parallel_config.data_parallel_size,
            vllm_config.parallel_config.enable_eplb,
            getattr(vllm_config.parallel_config, 'fault_tolerance', 'NOT_SET')
        )
        
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
            # State used for data parallel.
            self.engines_running = False

            # Initialize engines and coordinator (refactored for reuse)
            # This also resets utility_results and pending_messages
            self._initialize_engines(client_addresses, vllm_config, executor_class, log_stats)

            # Start monitoring engine core processes for unexpected failures
            self.start_engine_core_monitor()

            success = True
        finally:
            if not success:
                self._finalizer()

    def shutdown(self):
        # Terminate background resources.
        self._finalizer()

    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        if self.resources.engine_dead:
            logger.error(
                "[EngineDeadError] Formatting exception as EngineDeadError. "
                "Original exception type: %s, message: %s",
                type(e).__name__, str(e)
            )
            return EngineDeadError(suppress_context=True)
        return e

    def ensure_alive(self):
        if self.resources.engine_dead:
            logger.error(
                "[EngineDeadError] Engine is dead, cannot proceed with operation"
            )
            raise EngineDeadError()

    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    def free_pending_messages(self):
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()

    def dp_engines_running(self) -> bool:
        return self.engines_running

    def _initialize_engines(
        self,
        client_addresses: dict[str, str] | None,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        explicit_local_dp_ranks: list[int] | None = None,
    ):
        """
        Initialize or reinitialize engines and coordinator.
        
        Args:
            client_addresses: External addresses if engines managed externally
            vllm_config: vLLM configuration
            executor_class: Executor class to use
            log_stats: Whether to log stats
            explicit_local_dp_ranks: Optional explicit GPU assignments for FT full restart
        """
        self.stats_update_address: str | None = None
        
        if client_addresses:
            # Engines are managed externally to this client.
            input_address = client_addresses["input_address"]
            output_address = client_addresses["output_address"]
            self.stats_update_address = client_addresses.get("stats_update_address")
        else:
            # Engines are managed by this client.
            with launch_core_engines(
                vllm_config,
                executor_class,
                log_stats,
                explicit_local_dp_ranks=explicit_local_dp_ranks,
            ) as (
                engine_manager,
                coordinator,
                addresses,
            ):
                self.resources.coordinator = coordinator
                self.resources.engine_manager = engine_manager

            (input_address,) = addresses.inputs
            (output_address,) = addresses.outputs
            self.stats_update_address = addresses.frontend_stats_publish_address
            if coordinator is not None:
                assert self.stats_update_address == (
                    coordinator.get_stats_publish_address()
                )

        # For full restart, recreate ZMQ context for complete isolation
        if explicit_local_dp_ranks is not None:
            logger.info("[FT Full Restart] Recreating ZMQ context for complete isolation")
            
            # Close old sockets first
            if hasattr(self, 'input_socket') and self.input_socket:
                self.input_socket.close()
            if hasattr(self.resources, 'output_socket') and self.resources.output_socket:
                self.resources.output_socket.close()
            if hasattr(self.resources, 'stats_update_socket') and self.resources.output_socket:
                self.resources.stats_update_socket.close()
            
            # Recreate context
            sync_ctx = zmq.Context(io_threads=2)
            self.ctx = zmq.asyncio.Context(sync_ctx)
            
            # Update resources
            self.resources.ctx = sync_ctx
            
            logger.info("[FT Full Restart] ZMQ context recreated")
        
        # Create input and output sockets (using potentially new context)
        self.input_socket = self.resources.input_socket = make_zmq_socket(
            self.ctx, input_address, zmq.ROUTER, bind=True
        )
        self.resources.output_socket = make_zmq_socket(
            self.ctx, output_address, zmq.PULL
        )

        parallel_config = vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_rank = parallel_config.data_parallel_rank
        dp_local_size = parallel_config.data_parallel_size_local
        offline_mode = parallel_config.data_parallel_rank_local is not None
        # Client manages local+remote EngineCores in pure internal LB case.
        # Client manages local EngineCores in hybrid and external LB case.
        local_engines_only = (
            parallel_config.data_parallel_hybrid_lb
            or parallel_config.data_parallel_external_lb
        )

        num_ranks = dp_local_size if local_engines_only else dp_size
        self.engine_ranks_managed = (
            [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
        )
        assert parallel_config.data_parallel_size_local <= len(
            self.engine_ranks_managed
        )
        
        # Track GPU assignments for fault tolerance
        # Maps dp_rank → local_dp_rank (physical GPU assignment)
        self.gpu_mapping: dict[int, int] = {}
        self._initialize_gpu_mapping(explicit_local_dp_ranks)

        self.core_engines: list[EngineIdentity] = [
            i.to_bytes(2, "little") for i in self.engine_ranks_managed
        ]
        
        # Reset state for new engines (important for full restart)
        self.utility_results: dict[int, AnyFuture] = {}
        self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

        logger.info(
            f"Initialized {len(self.core_engines)} engines: "
            f"{self.engine_ranks_managed} (offline mode: {offline_mode})"
        )

    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        engine_manager = self.resources.engine_manager
        if (
            engine_manager is None
            or not hasattr(engine_manager, "processes")
            or not engine_manager.processes
        ):
            # No engine processes to monitor
            return

        engine_processes = engine_manager.processes
        self_ref = weakref.ref(self)
        
        # Check if fault tolerance is enabled
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        enable_eplb = self.vllm_config.parallel_config.enable_eplb
        ft_flag = getattr(self.vllm_config.parallel_config, 'fault_tolerance', False)
        
        fault_tolerance_enabled = (
            dp_size > 1 and
            enable_eplb and
            ft_flag
        )
        
        logger.info(
            "[FT] Fault tolerance check: dp_size=%d, enable_eplb=%s, "
            "fault_tolerance=%s, enabled=%s",
            dp_size, enable_eplb, ft_flag, fault_tolerance_enabled
        )
        
        # Create dedicated sync socket for FT notifications (if FT enabled)
        ft_notification_socket = None
        if fault_tolerance_enabled:
            # Create separate address for FT notifications
            self.ft_sock_addr = get_open_zmq_inproc_path()
            
            # IMPORTANT: For inproc://, sender and receiver MUST use same ZMQ context!
            # Use sync context for PUSH socket (monitor thread is sync)
            # PULL socket will use async wrapper of same context
            ft_notification_socket = make_zmq_socket(
                self.resources.ctx,  # Sync context
                self.ft_sock_addr,
                zmq.PUSH,
                bind=True
            )
            
            logger.info(
                "[FT] Created dedicated sync PUSH socket for FT notifications at %s",
                self.ft_sock_addr
            )

        # Monitor engine core process liveness. If any die unexpectedly,
        # either trigger fault tolerance or shutdown depending on config.
        def monitor_engine_cores():
            sentinels = [proc.sentinel for proc in engine_processes]
            
            while sentinels:
                died = multiprocessing.connection.wait(sentinels)
                _self = self_ref()
                if not _self or _self.resources.engine_dead:
                    return
                
                # Find which rank died
                failed_proc = next(
                    proc for proc in engine_processes 
                    if proc.sentinel == died[0]
                )
                failed_rank = _self._get_rank_from_process(failed_proc)
                
                logger.error(
                    "[FT] Engine core proc %s (rank %d, PID %d) died unexpectedly. "
                    "Exit code: %s",
                    failed_proc.name, failed_rank, failed_proc.pid,
                    failed_proc.exitcode
                )
                
                if fault_tolerance_enabled:
                    # Fault tolerance: handle based on configured mode
                    ft_mode = _self.vllm_config.parallel_config.fault_tolerance_mode
                    
                    logger.info(
                        "[FT] Fault tolerance enabled (mode=%s), handling failure of rank %d",
                        ft_mode, failed_rank
                    )
                    
                    try:
                        # Update GPU mapping and check if this is first detection
                        surviving_gpus, was_removed = _self._update_gpu_mapping_after_failure(failed_rank)
                        
                        if not was_removed:
                            # Duplicate detection - rank already handled
                            logger.info(
                                "[FT] Rank %d failure already processed. Skipping notification.",
                                failed_rank
                            )
                            # Update sentinels but don't send duplicate notification
                            engine_processes[:] = [p for p in engine_processes if p.is_alive()]
                            sentinels = [p.sentinel for p in engine_processes]
                            continue
                        
                        if ft_mode == FaultToleranceMode.FULL_RESTART or ft_mode == "full_restart":
                            # Full restart: Client manages everything directly
                            # No coordinator involvement - just shutdown all and recreate
                            logger.info(
                                "[FT Full Restart] Rank %d died. Shutting down ALL processes. "
                                "Will restart on GPUs: %s",
                                failed_rank, surviving_gpus
                            )
                            
                            # Trigger full restart (shutdown all + recreate)
                            _self._trigger_full_restart(failed_rank, surviving_gpus, engine_processes)
                            
                            # Full restart completes, exit monitor loop
                            # New processes will be monitored by new monitor thread
                            return
                        
                        else:
                            # Lightweight: Coordinate with engines via coordinator
                            logger.info(
                                "[FT %s] Notifying coordinator about rank %d failure. "
                                "Surviving GPUs: %s",
                                ft_mode, failed_rank, surviving_gpus
                            )
                            
                            # Send notification to coordinator
                            # Coordinator broadcasts FT_SCALE_DOWN
                            # Engines call _execute_ft_reconfiguration (preserves KV)
                            _self._send_ft_notification_to_coordinator(failed_rank, ft_mode, ft_notification_socket)
                        
                    except Exception as e:
                        logger.error(
                            "[FT] Failed to handle failure: %s. "
                            "Falling back to shutdown.", e, exc_info=True
                        )
                        _self.resources.engine_dead = True
                        _self.shutdown()
                        return
                    
                    # Remove dead process from monitoring
                    engine_processes[:] = [p for p in engine_processes if p.is_alive()]
                    sentinels = [p.sentinel for p in engine_processes]
                    
                    # DON'T set engine_dead flag - let recovery complete!
                else:
                    # No fault tolerance: shutdown everything (old behavior)
                    _self.resources.engine_dead = True
                    _self.shutdown()
                    return

        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()
    
    def _get_rank_from_process(self, proc: multiprocessing.Process) -> int:
        """Extract rank number from process name like 'EngineCore_DP2'."""
        import re
        match = re.search(r'EngineCore_DP(\d+)', proc.name)
        if match:
            return int(match.group(1))
        raise ValueError(f"Cannot parse rank from process name: {proc.name}")
    
    def _initialize_gpu_mapping(self, explicit_local_dp_ranks: list[int] | None = None):
        """
        Initialize GPU mapping from process configuration.
        Maps dp_rank → local_dp_rank (physical GPU assignment).
        
        Args:
            explicit_local_dp_ranks: Optional explicit GPU assignments (for full restart).
                If provided, uses these instead of sequential assignment.
                Example: [0, 1, 3] creates mapping {0: 0, 1: 1, 2: 3} (skips GPU 2)
        """
        parallel_config = self.vllm_config.parallel_config
        
        # For local processes
        local_start_index = parallel_config.data_parallel_rank_local or 0
        start_index = parallel_config.data_parallel_rank or 0
        local_engine_count = parallel_config.data_parallel_size_local or parallel_config.data_parallel_size
        
        for i in range(local_engine_count):
            dp_rank = start_index + i
            
            # Use explicit GPU assignment if provided (for FT full restart)
            # Otherwise use sequential assignment (normal case)
            if explicit_local_dp_ranks is not None:
                local_dp_rank = explicit_local_dp_ranks[i]
            else:
                local_dp_rank = local_start_index + i
                
            self.gpu_mapping[dp_rank] = local_dp_rank
        
        logger.info("[FT] Initial GPU mapping: %s", self.gpu_mapping)
    
    def _get_current_gpu_mapping(self) -> dict[int, int]:
        """Get current GPU mapping for all ranks."""
        return self.gpu_mapping.copy()
    
    def _update_gpu_mapping_after_failure(self, failed_rank: int) -> tuple[list[int], bool]:
        """
        Update GPU mapping after a rank failure and return surviving GPUs in order.
        
        Args:
            failed_rank: The rank that failed
            
        Returns:
            Tuple of (surviving_gpus, was_removed):
            - surviving_gpus: List of GPU indices for surviving ranks
            - was_removed: True if this was first detection, False if duplicate
        """
        # Check if rank was already removed (duplicate detection)
        was_removed = failed_rank in self.gpu_mapping
        
        if was_removed:
            failed_gpu = self.gpu_mapping.pop(failed_rank)
            logger.info(
                "[FT] Removed rank %d (GPU %d) from mapping",
                failed_rank, failed_gpu
            )
        else:
            logger.info(
                "[FT] Rank %d already removed (duplicate detection, skipping)",
                failed_rank
            )
        
        # Get surviving ranks and their GPUs (in order)
        surviving_ranks = sorted(self.gpu_mapping.keys())
        surviving_gpus = [self.gpu_mapping[rank] for rank in surviving_ranks]
        
        logger.info(
            "[FT] Current GPU mapping: ranks %s → GPUs %s",
            surviving_ranks, surviving_gpus
        )
        
        return surviving_gpus, was_removed
    
    def _trigger_full_restart(
        self,
        failed_rank: int,
        surviving_gpus: list[int],
        engine_processes: list,
    ):
        """
        Trigger full restart: shutdown all processes and recreate with explicit GPUs.
        Called from monitor thread (sync context).
        
        Args:
            failed_rank: The rank that failed
            surviving_gpus: GPU indices for surviving ranks (e.g., [0, 1, 3])
            engine_processes: List of engine processes to shutdown
        """
        logger.info(
            "[FT Full Restart] Starting full restart procedure. "
            "Failed rank: %d, Surviving GPUs: %s",
            failed_rank, surviving_gpus
        )
        
        # Step 1: Shutdown ALL engine processes (including healthy ones)
        logger.info("[FT Full Restart] Shutting down all %d engine processes...", len(engine_processes))
        
        for proc in engine_processes:
            if proc.is_alive():
                logger.info("[FT Full Restart] Terminating process %s (PID %s)", proc.name, proc.pid)
                proc.terminate()  # SIGTERM for clean shutdown
        
        # Step 2: Wait for all processes to exit (with timeout)
        timeout = 10  # seconds
        start_time = time.time()
        while any(p.is_alive() for p in engine_processes):
            if time.time() - start_time > timeout:
                logger.warning("[FT Full Restart] Timeout waiting for clean shutdown, force killing...")
                for proc in engine_processes:
                    if proc.is_alive():
                        logger.warning("[FT Full Restart] Force killing %s", proc.name)
                        proc.kill()  # SIGKILL
                break
            time.sleep(0.1)
        
        logger.info("[FT Full Restart] All engine processes shut down")
        
        # Step 3: Close old engine manager and coordinator
        if self.resources.engine_manager:
            self.resources.engine_manager.close()
            logger.info("[FT Full Restart] Closed old engine manager")
        
        if self.resources.coordinator:
            coordinator_pid = self.resources.coordinator.proc.pid if hasattr(self.resources.coordinator, 'proc') else None
            self.resources.coordinator.close()
            logger.info("[FT Full Restart] Closed coordinator (PID: %s)", coordinator_pid)
            
            # Wait for coordinator process to actually terminate to avoid port conflicts
            if coordinator_pid:
                timeout = 5
                start = time.time()
                while self.resources.coordinator.proc.is_alive():
                    if time.time() - start > timeout:
                        logger.warning("[FT Full Restart] Coordinator didn't exit cleanly, force killing")
                        self.resources.coordinator.proc.kill()
                        break
                    time.sleep(0.1)
                logger.info("[FT Full Restart] Coordinator process fully terminated")
        
        # Step 4: Update EPLB config to maintain expert-per-GPU ratio
        # This mimics lightweight FT behavior by marking lost experts as redundant
        original_dp_size = self.vllm_config.parallel_config.data_parallel_size
        new_size = len(surviving_gpus)
        num_failed_ranks = original_dp_size - new_size
        
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.enable_eplb:
            eplb_config = parallel_config.eplb_config
            model_config = self.vllm_config.model_config
            
            # Get logical number of experts from model config
            # In EPLB: global_num_experts = logical_experts + redundant_experts
            # After failure: we add lost experts to redundant_experts to maintain expert-per-GPU ratio
            logical_experts = model_config.get_num_experts()
            current_redundant = eplb_config.num_redundant_experts
            total_physical_experts = logical_experts + current_redundant
            
            # Calculate experts per rank based on total physical experts
            # (must be evenly divisible for consistent distribution)
            if total_physical_experts % original_dp_size != 0:
                error_msg = (
                    f"[FT Full Restart] Cannot maintain expert distribution: "
                    f"{total_physical_experts} physical experts (logical={logical_experts} + "
                    f"redundant={current_redundant}) not evenly divisible by {original_dp_size} ranks. "
                    f"Full restart requires even expert distribution."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            experts_per_rank = total_physical_experts // original_dp_size
            lost_experts = num_failed_ranks * experts_per_rank
            
            new_redundant_experts = current_redundant - lost_experts
            # Calculate remaining physical experts after loss
            remaining_physical = total_physical_experts - lost_experts
            
            # Validate: new_redundant_experts must be non-negative
            # If negative, it means we have fewer physical experts than logical experts,
            # which violates the requirement that every logical expert needs a physical implementation
            if new_redundant_experts < 0:
                error_msg = (
                    f"[FT Full Restart] Configuration invalid after failure: "
                    f"Insufficient physical experts remaining. "
                    f"logical_experts={logical_experts}, remaining_physical={remaining_physical}, "
                    f"new_redundant={new_redundant_experts}. "
                    f"Lost too many experts ({lost_experts}) - cannot maintain minimal coverage."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            
            
            # Validate: remaining physical experts must be evenly divisible by new DP size
            if remaining_physical % new_size != 0:
                error_msg = (
                    f"[FT Full Restart] Configuration invalid after failure: "
                    f"{remaining_physical} remaining physical experts not evenly divisible by "
                    f"{new_size} surviving ranks. Cannot maintain consistent expert-per-GPU ratio."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            logger.info(
                "[FT Full Restart] Maintaining expert-per-GPU ratio: "
                "logical_experts=%d, physical_before=%d, experts_per_rank=%d, "
                "lost_experts=%d, physical_after=%d, redundant: %d→%d",
                logical_experts, total_physical_experts, experts_per_rank,
                lost_experts, remaining_physical, current_redundant, new_redundant_experts
            )
            
            # Update EPLB configuration
            eplb_config.num_redundant_experts = new_redundant_experts
        
        # Step 5: Update config for new size
        self.vllm_config.parallel_config.data_parallel_size = new_size
        self.vllm_config.parallel_config.data_parallel_size_local = new_size
        
        # CRITICAL: Get new master port for process group initialization
        # Without this, new engines will try to use old port which may have stale NCCL state
        from vllm.utils.network_utils import get_open_port
        self.vllm_config.parallel_config.data_parallel_master_port = get_open_port()
        
        logger.info(
            "[FT Full Restart] Recreating %d engines on GPUs %s (new master port: %d)...",
            new_size, surviving_gpus, 
            self.vllm_config.parallel_config.data_parallel_master_port
        )
        
        # Step 5: Reinitialize engines using refactored method
        # This reuses the same initialization code as __init__
        self._initialize_engines(
            client_addresses=None,  # Manage engines internally
            vllm_config=self.vllm_config,
            executor_class=self.executor_class,
            log_stats=self.log_stats,
            explicit_local_dp_ranks=surviving_gpus,  # [0, 1, 3] - Skip GPU 2!
        )
        
        # Step 6: Reset engine_dead flag since new engines are healthy
        self.resources.engine_dead = False
        logger.info("[FT Full Restart] Reset engine_dead flag - new engines are operational")
        
        # Step 7: Restart output_queue_task with new ZMQ sockets
        # CRITICAL: The old output_queue_task captured OLD socket references in its closure (line 1354)
        # Even though we created NEW sockets, the old task still reads from the OLD socket reference!
        # We MUST cancel it so _ensure_output_queue_task() creates a NEW task that captures NEW sockets
        # 
        # Note: We're called from sync monitor thread, so we can't create async tasks here.
        # Instead, set to None and the task will be recreated lazily on next get_output_async() call.
        # AsyncLLM.output_handler doesn't need restarting - it will automatically use the new task.
        if self.resources.output_queue_task is not None:
            logger.info("[FT Full Restart] Cancelling stale output_queue_task (has old socket refs)")
            self.resources.output_queue_task.cancel()
            self.resources.output_queue_task = None
            logger.info(
                "[FT Full Restart] output_queue_task cancelled. "
                "Will be recreated with new sockets on next get_output_async() call"
            )
        
        # Step 8: Start new monitor thread for new processes
        self.start_engine_core_monitor()
        
        logger.info(
            "[FT Full Restart] Complete! Successfully restarted %d processes on GPUs %s",
            new_size, surviving_gpus
        )
    
    def _send_ft_notification_to_coordinator(
        self, 
        failed_rank: int, 
        ft_mode: str | None = None,
        ft_socket: Any = None
    ):
        """
        Send fault tolerance notification to coordinator.
        Called from monitor thread (sync context).
        
        Args:
            failed_rank: The rank that failed
            ft_mode: Fault tolerance mode ("lightweight" or "full_restart")
                    If None, uses lightweight by default
            ft_socket: Dedicated sync PUSH socket for FT notifications
        """
        if ft_mode is None:
            ft_mode = "lightweight"
        try:
            import msgspec.msgpack
            
            # Build notification message with mode information
            # Format: ("FT_RANK_DIED", failed_rank, mode)
            ft_msg = msgspec.msgpack.encode(("FT_RANK_DIED", failed_rank, str(ft_mode)))
            
            if ft_socket is None:
                raise RuntimeError(
                    f"FT notification socket not available. "
                    "Cannot send notification for failed rank {failed_rank}."
                )
            
            # Use dedicated sync PUSH socket - simple blocking send
            ft_socket.send(ft_msg)
            
            logger.info(
                "[FT] Sent notification for rank %d (mode=%s) to coordinator",
                failed_rank, ft_mode
            )
            
        except Exception as e:
            logger.error(
                "[FT] Failed to send FT notification for rank %d: %s",
                failed_rank, e, exc_info=True
            )
            raise


def _process_utility_output(
    output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        self.is_dp = self.vllm_config.parallel_config.data_parallel_size > 1
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        ctx = self.ctx
        out_socket = self.resources.output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue

        shutdown_path = get_open_zmq_inproc_path()
        resources = self.resources
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()
        if isinstance(outputs, Exception):
            logger.error(
                "[EngineDeadError] Exception received from output queue (sync). "
                "Exception type: %s, message: %s",
                type(outputs).__name__, str(outputs)
            )
            raise self._format_exception(outputs) from None
        if outputs.wave_complete is not None:
            self.engines_running = False
        return outputs

    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        self.free_pending_messages()
        # (Identity, RequestType, SerializedRequest)
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

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
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        return future.result()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.call_utility("get_supported_tasks")

    def add_request(self, request: EngineCoreRequest) -> None:
        if self.is_dp:
            self.engines_running = True
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

    def wake_up(self, tags: list[str] | None = None) -> None:
        self.call_utility("wake_up", tags)

    def is_sleeping(self) -> bool:
        return self.call_utility("is_sleeping")

    def execute_dummy_batch(self, cpu_only: bool = False) -> None:
        self.call_utility("execute_dummy_batch", cpu_only)

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.call_utility("save_sharded_state", path, pattern, max_size)


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
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
        output_handler: (
            Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
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
                        _process_utility_output(outputs.utility_output, utility_results)
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
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            logger.error(
                "[EngineDeadError] Exception received from output queue (async). "
                "Exception type: %s, message: %s",
                type(outputs).__name__, str(outputs)
            )
            raise self._format_exception(outputs) from None
        return outputs

    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)

    def _send_input_message(
        self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        self.ensure_alive()
        self.free_pending_messages()

        msg = (engine,) + message
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
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    async def _call_utility_async(
        self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        call_id = uuid.uuid1().int >> 64
        future = asyncio.get_running_loop().create_future()
        self.utility_results[call_id] = future
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        await self._send_input_message(message, engine, args)
        self._ensure_output_queue_task()
        return await future

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return await self.call_utility_async("get_supported_tasks")

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
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

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
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

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.current_wave = 0

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running] pair per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _ensure_stats_update_task(self):
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0
        # NOTE: running and waiting counts are all global from
        # the Coordinator include all global EngineCores. This
        # slice includes just the cores managed by this client.
        count_slice = slice(
            self.engine_ranks_managed[0], self.engine_ranks_managed[-1] + 1
        )

        async def run_engine_stats_update_task():
            # Build socket context managers
            socket_managers = [
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0),
                make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0),
            ]
            
            # Add FT notification receiver if FT enabled
            # Uses self.ctx which wraps the same sync context as PUSH socket
            if hasattr(self, 'ft_sock_addr'):
                socket_managers.append(
                    make_zmq_socket(self.ctx, self.ft_sock_addr, zmq.PULL, bind=False, linger=0)
                )
            
            with contextlib.ExitStack() as stack:
                sockets = [stack.enter_context(sm) for sm in socket_managers]
                socket = sockets[0]
                first_req_rcv_socket = sockets[1]
                ft_rcv_socket = sockets[2] if len(sockets) > 2 else None
                
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)
                if ft_rcv_socket is not None:
                    poller.register(ft_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    
                    # Check FT notification socket first (if enabled)
                    if ft_rcv_socket is not None and any(sock == ft_rcv_socket for sock, _ in events):
                        # Received FT notification from monitor thread
                        ft_buf = await ft_rcv_socket.recv()
                        ft_decoded = msgspec.msgpack.decode(ft_buf)
                        
                        if (
                            isinstance(ft_decoded, (list, tuple))
                            and len(ft_decoded) >= 2
                            and ft_decoded[0] == "FT_RANK_DIED"
                        ):
                            # Forward to coordinator via stats socket (XSUB→XPUB)
                            # The coordinator's publish_front (XPUB) receives this.
                            # 
                            # NOTE: Cannot use first_req_send_socket because:
                            # - first_req_send_socket is a PAIR socket connected to first_req_rcv_socket
                            # - PAIR sockets are internal to the client (client talks to itself)
                            # - Coordinator receives on publish_front (XPUB), not first_req_rcv_socket
                            # - Must use stats socket (XSUB) which is subscribed to coordinator's XPUB
                            logger.info(
                                "[FT] Received notification from monitor, forwarding to coordinator: %s",
                                ft_decoded
                            )
                            await socket.send(ft_buf)  # Use stats_update_socket (XSUB)
                    
                    if (
                        not self.engines_running
                        and len(events) >= 2
                        or any(sock == first_req_rcv_socket for sock, _ in events)
                    ):
                        # Check if this is a regular request notification or scale notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) >= 2
                            and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # Extract new engine count and optional failed rank
                            new_engine_count = decoded[1]
                            failed_rank = decoded[2] if len(decoded) > 2 else None
                            
                            logger.info(
                                "[Elastic EP] Received scale notification: new_size=%d, failed_rank=%s",
                                new_engine_count, failed_rank
                            )
                            
                            # Trigger scale operation
                            # If failed_rank is provided (FT full restart), pass it to scale down
                            if failed_rank is not None:
                                # FT full restart case - include failed rank info
                                logger.info(
                                    "[FT Full Restart] Triggering scale down for failed rank %d",
                                    failed_rank
                                )
                                await self.scale_elastic_ep(new_engine_count, failed_rank=failed_rank)
                            else:
                                # Normal scale case
                                await self.scale_elastic_ep(new_engine_count)
                            
                            # Send notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue
                        
                        if (
                            isinstance(decoded, (list, tuple))
                            and len(decoded) == 2
                            and decoded[0] == "FT_RANK_DIED"
                        ):
                            # Fault tolerance: rank failed, notify coordinator
                            failed_rank = decoded[1]
                            logger.warning(
                                "[FT] Client forwarding rank %d failure to coordinator",
                                failed_rank
                            )
                            
                            # Forward to coordinator
                            ft_msg = msgspec.msgpack.encode(
                                ("FT_RANK_DIED", failed_rank)
                            )
                            await socket.send(ft_msg)
                            
                            # Handle client-side cleanup for failed rank
                            await self._handle_rank_failure_client_side(failed_rank)
                            
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ", f"Unexpected message: {decoded}"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    decoded = msgspec.msgpack.decode(buf)
                    
                    # Handle both old format (3 items) and new format (4 items)
                    if len(decoded) == 4:
                        counts, wave, running, masked_ranks = decoded
                        
                        # Update masked ranks tracking
                        old_masked = self.masked_ranks.copy()
                        self.masked_ranks = set(masked_ranks)
                        
                        # Log when masked ranks change
                        if masked_ranks != list(old_masked):
                            logger.info(
                                "[GPU_FT] Load balancer: Updated masked ranks: %s "
                                "(will not route requests to these ranks)",
                                masked_ranks if masked_ranks else "none"
                            )
                    else:
                        # Old format compatibility
                        counts, wave, running = decoded
                    
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}
        
        # Track which DP ranks are masked (GPU unhealthy)
        # Updated from SchedulerStats published by engines
        self.masked_ranks: set[int] = set()

        super().__init__(
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1

        self.eng_start_index = (
            len(self.core_engines) * self.client_index
        ) // client_count

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                
                # Skip masked ranks (GPU unhealthy)
                if idx in self.masked_ranks:
                    logger.debug(
                        "[GPU_FT] Skipping masked DP rank %d in load balancing",
                        idx
                    )
                    continue
                
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    @staticmethod
    async def process_engine_outputs(
        self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
        self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)
    
    async def _handle_rank_failure_client_side(self, failed_rank: int):
        """
        Handle client-side cleanup when a rank fails.
        Aborts in-flight requests sent to the failed rank.
        """
        logger.info(
            "[FT] Client handling failure of rank %d (client-side cleanup)",
            failed_rank
        )
        
        # Find requests sent to failed rank
        if failed_rank >= len(self.core_engines):
            logger.warning(
                "[FT] Failed rank %d out of range (we have %d engines)",
                failed_rank, len(self.core_engines)
            )
            return
        
        failed_engine = self.core_engines[failed_rank]
        requests_to_abort = [
            req_id for req_id, engine in self.reqs_in_flight.items()
            if engine == failed_engine
        ]
        
        if requests_to_abort:
            logger.warning(
                "[FT] Found %d in-flight requests to failed rank %d, aborting",
                len(requests_to_abort), failed_rank
            )
            
            # Clean up tracking (engine is dead, can't send abort message)
            for req_id in requests_to_abort:
                self.reqs_in_flight.pop(req_id, None)
            
            # Abort at output processor level (will send FinishReason.ABORT to clients)
            # Note: We access output_processor from the parent AsyncLLM if available
            # For now, just clean up local state
            logger.info(
                "[FT] Removed %d requests to failed rank %d from tracking",
                len(requests_to_abort), failed_rank
            )
        
        # Update client state
        logger.info(
            "[FT] Updating client state: removing rank %d",
            failed_rank
        )
        
        # Update core_engines list
        self.core_engines.pop(failed_rank)
        
        # Update load balancer state
        if hasattr(self, 'lb_engines'):
            self.lb_engines.pop(failed_rank)
        
        # Clear masked ranks
        if hasattr(self, 'masked_ranks'):
            self.masked_ranks.clear()
        
        # Update config
        new_size = len(self.core_engines)
        self.vllm_config.parallel_config.data_parallel_size = new_size
        
        logger.info(
            "[FT] Client state updated: now tracking %d engines",
            new_size
        )

    async def scale_elastic_ep(
        self, 
        new_data_parallel_size: int,
        failed_rank: int | None = None
    ) -> None:
        """
        Scale elastic EP data parallel size.
        
        Args:
            new_data_parallel_size: Target DP size
            failed_rank: Optional rank that failed (for FT full restart).
                        If provided, this rank will be excluded from surviving processes.
        """
        cur_data_parallel_size = len(self.core_engines)

        assert new_data_parallel_size != cur_data_parallel_size, (
            f"new_data_parallel_size {new_data_parallel_size} must be "
            f"different from cur_data_parallel_size {cur_data_parallel_size}"
        )

        assert self.vllm_config.parallel_config.data_parallel_backend == "ray", (
            "Only ray DP backend supports scaling elastic EP"
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size

        if scale_up:
            await self._scale_up_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
        else:
            # For scale down with failed rank (FT full restart), calculate removed ranks
            if failed_rank is not None:
                removed_ranks = [failed_rank]
                logger.info(
                    "[FT Full Restart] Scale down excluding failed rank %d", 
                    failed_rank
                )
            else:
                # Normal scale down: remove last ranks
                removed_ranks = list(range(new_data_parallel_size, cur_data_parallel_size))
            
            await self._scale_down_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size, removed_ranks
            )

    async def _scale_up_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        cur_data_parallel_size = len(self.core_engines)

        # Phase 1: Send reconfigure messages to all existing engines and wait
        # for them to be sent
        reconfig_futures = []
        self.vllm_config.parallel_config.data_parallel_master_port = get_open_port()
        for engine in self.core_engines:
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=self.vllm_config.parallel_config.data_parallel_master_ip,
                new_data_parallel_master_port=self.vllm_config.parallel_config.data_parallel_master_port,
            )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        logger.info("All reconfigure messages sent, starting engine creation")

        # Phase 2: Create new engines now that reconfig messages have been sent
        # self.resources.engine_manager is guaranteed to be
        # CoreEngineActorManager for RayDPClient
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.scale_up_elastic_ep(
            self.vllm_config, new_data_parallel_size
        )

        # Create new CoreEngine objects for the new engines
        new_engine_identities = set()
        for i in range(cur_data_parallel_size, new_data_parallel_size):
            new_engine = i.to_bytes(2, "little")
            self.core_engines.append(new_engine)
            new_engine_identities.add(new_engine)

        # Wait for ready messages from new engines on the input socket
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        while new_engine_identities:
            if not sync_input_socket.poll(timeout=600_000):
                raise TimeoutError(
                    "Timed out waiting for new engines to send initial "
                    "message on input socket."
                )
            identity, _ = sync_input_socket.recv_multipart()
            new_engine_identities.discard(identity)

        # Phase 3: Wait for all existing engines to complete reconfiguration
        logger.info("Waiting for existing engines to complete reconfiguration")
        await asyncio.gather(*reconfig_futures)

        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        # Update the parallel config
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )

    async def _scale_down_elastic_ep(
        self, 
        cur_data_parallel_size: int, 
        new_data_parallel_size: int,
        removed_ranks: list[int] | None = None,
    ) -> None:
        """
        Scale down the data parallel size by shutting down and reconfiguring existing engine cores.
        
        Args:
            cur_data_parallel_size: Current DP size
            new_data_parallel_size: Target DP size  
            removed_ranks: Specific ranks to remove. If None, removes last ranks.
                          For FT: [2] means remove rank 2, keep 0,1,3
                          Normal: None means remove rank 3, keep 0,1,2
        """
        if removed_ranks is None:
            # Normal elastic scale down: remove last ranks
            removed_ranks = list(range(new_data_parallel_size, cur_data_parallel_size))
        logger.info(
            "[Scale Down] Removing ranks: %s (normal removes last, FT removes failed)",
            removed_ranks
        )
        
        cur_data_parallel_size = len(self.core_engines)

        self.vllm_config.parallel_config.data_parallel_master_port = get_open_port()

        # Build rank mapping
        rank_mapping = {}
        new_rank = 0
        for old_rank in range(cur_data_parallel_size):
            if old_rank in removed_ranks:
                rank_mapping[old_rank] = -1  # Remove
            else:
                rank_mapping[old_rank] = new_rank
                new_rank += 1
        
        logger.info("[Scale Down] Rank mapping: %s", rank_mapping)

        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(self.core_engines):
            new_rank_for_this_process = rank_mapping.get(cur_dp_rank, -1)
            
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK if new_rank_for_this_process != -1 else ReconfigureRankType.SHUTDOWN_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=self.vllm_config.parallel_config.data_parallel_master_ip,
                new_data_parallel_master_port=self.vllm_config.parallel_config.data_parallel_master_port,
            )
            
            # Set specific new rank if not shutting down
            if new_rank_for_this_process != -1:
                reconfig_request.new_data_parallel_rank = new_rank_for_this_process
            
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # Remove engines in removed_ranks (reverse order to maintain indices)
        for rank in sorted(removed_ranks, reverse=True):
            self.core_engines.pop(rank)

        await asyncio.gather(*reconfig_futures)

        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        self.resources.engine_manager.scale_down_elastic_ep(
            cur_data_parallel_size, new_data_parallel_size
        )

        self._ensure_stats_update_task()
        scale_down_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_down_marker)

        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size
        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
    
