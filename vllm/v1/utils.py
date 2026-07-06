# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import json
import multiprocessing
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
    overload,
)

import torch
import uvloop
from torch.autograd.profiler import record_function

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils.network_utils import get_open_zmq_ipc_path, get_tcp_uri
from vllm.utils.system_utils import decorate_logs, kill_process_tree, set_process_title
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    import numpy as np

    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import CoreEngineActorManager, CoreEngineProcManager

logger = init_logger(__name__)

T = TypeVar("T")


class ConstantList(Generic[T], Sequence):
    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise TypeError("Cannot append to a constant list")

    def extend(self, item):
        raise TypeError("Cannot extend a constant list")

    def insert(self, item):
        raise TypeError("Cannot insert into a constant list")

    def pop(self, item):
        raise TypeError("Cannot pop from a constant list")

    def remove(self, item):
        raise TypeError("Cannot remove from a constant list")

    def clear(self):
        raise TypeError("Cannot clear a constant list")

    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...

    @overload
    def __setitem__(self, s: slice, value: T, /): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise TypeError("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise TypeError("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"

    def copy(self) -> list[T]:
        return self._x.copy()


class CpuGpuBuffer:
    """Buffer to easily copy tensors between CPU and GPU."""

    def __init__(
        self,
        *size: int | torch.SymInt,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = PIN_MEMORY,
        with_numpy: bool = True,
    ) -> None:
        # these buffers are mutable runtime state, so allocate them as normal
        with torch.inference_mode(False):
            self.cpu = torch.zeros(
                *size, dtype=dtype, device="cpu", pin_memory=pin_memory
            )
            self.gpu = torch.zeros_like(self.cpu, device=device)
        self.np: np.ndarray
        # To keep type hints simple (avoiding generics and subclasses), we
        # only conditionally create the numpy array attribute. This can cause
        # AttributeError if `self.np` is accessed when `with_numpy=False`.
        if with_numpy:
            if dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call CpuGpuBuffer with with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        """NOTE: Because this method is non-blocking, explicit synchronization
        is needed to ensure the data is copied to CPU."""
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=True)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


def get_engine_client_zmq_addr(
    local_only: bool,
    host: str,
    port: int = 0,
) -> str:
    """Return an IPC path (``local_only=True``) or ``tcp://host:port``.

    ``port=0`` lets the kernel assign the port at ``bind()`` time; the
    caller must recover it via ``getsockopt(zmq.LAST_ENDPOINT)``."""
    if local_only:
        return get_open_zmq_ipc_path()
    return get_tcp_uri(host, port)


class APIServerProcessManager:
    """Manages a group of API server processes.

    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.
    """

    def __init__(
        self,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        target_server_fn: Callable | None = None,
        stats_update_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        """Initialize and start API server worker processes.

        ``input_addresses``/``output_addresses`` may contain
        ``tcp://host:0`` placeholders; each child must report the actual
        bound endpoint over its ``actual_address_pipe`` in ``client_config``
        and the parent collects them via
        :py:meth:`gather_actual_addresses`.

        Args:
            target_server_fn: Override function to call for each API server process
            listen_address: Address to listen for client connections
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address
            tensor_queue: Optional tensor IPC queue for sharing MM tensors
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = []
        self._address_pipes: list[connection.Connection] = []

        snapshot_barrier = (
            spawn_context.Barrier(num_servers)
            if num_servers > 1 and getattr(args, "enable_snapshot_post_startup", False)
            else None
        )

        for i, in_addr, out_addr in zip(
            range(num_servers), input_addresses, output_addresses
        ):
            client_config: dict[str, Any] = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_count": num_servers,
                "client_index": i,
            }
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address
            if tensor_queue is not None:
                client_config["tensor_queue"] = tensor_queue
            if snapshot_barrier is not None:
                client_config["snapshot_barrier"] = snapshot_barrier

            parent_recv, child_send = spawn_context.Pipe(duplex=False)
            self._address_pipes.append(parent_recv)
            client_config["actual_address_pipe"] = child_send

            proc = spawn_context.Process(
                target=target_server_fn or run_api_server_worker_proc,
                name=f"ApiServer_{i}",
                args=(listen_address, sock, args, client_config),
            )
            self.processes.append(proc)
            proc.start()

            # Drop parent's write end so reader sees EOF on child death.
            child_send.close()

        logger.info("Started %d API server processes", len(self.processes))

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def gather_actual_addresses(
        self,
        timeout: float = envs.VLLM_ENGINE_READY_TIMEOUT_S,
    ) -> tuple[list[str], list[str]]:
        """Return (inputs, outputs) reported by each child, indexed by
        ``client_index``. Raises ``RuntimeError`` on timeout or premature
        child exit."""
        n = len(self._address_pipes)
        inputs: list[str | None] = [None] * n
        outputs: list[str | None] = [None] * n
        pending: dict[connection.Connection, int] = {
            pipe: i for i, pipe in enumerate(self._address_pipes)
        }
        sentinel_to_idx: dict[Any, int] = {
            proc.sentinel: i for i, proc in enumerate(self.processes)
        }

        deadline = time.monotonic() + timeout
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [self.processes[i].name for i in pending.values()]
                    raise RuntimeError(
                        f"Timed out after {timeout:.1f}s waiting for "
                        f"API server(s) to report bound ZMQ addresses: "
                        f"{missing}"
                    )
                waitables: list[Any] = list(pending.keys()) + list(
                    sentinel_to_idx.keys()
                )
                ready = connection.wait(waitables, timeout=remaining)
                # Drain pipes before checking sentinels: a child that sent
                # its message and then exited can surface both events in
                # the same poll, and we must record the success first.
                for item in ready:
                    if isinstance(item, connection.Connection) and item in pending:
                        idx = pending.pop(item)
                        try:
                            msg: dict[str, str] = item.recv()
                        except EOFError as e:
                            raise RuntimeError(
                                f"API server {self.processes[idx].name} "
                                f"closed its address pipe without "
                                f"reporting its bound ZMQ addresses"
                            ) from e
                        inputs[idx] = msg["input_address"]
                        outputs[idx] = msg["output_address"]
                        item.close()
                for item in ready:
                    if item in sentinel_to_idx:
                        idx = sentinel_to_idx.pop(item)
                        pipe = self._address_pipes[idx]
                        if pipe in pending:
                            proc = self.processes[idx]
                            raise RuntimeError(
                                f"API server process {proc.name} exited "
                                f"(code={proc.exitcode}) before reporting "
                                f"its bound ZMQ addresses"
                            )
        finally:
            for pipe in pending:
                with contextlib.suppress(Exception):
                    pipe.close()

        return inputs, outputs  # type: ignore[return-value]

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown API server processes with configurable timeout"""
        for pipe in self._address_pipes:
            with contextlib.suppress(Exception):
                pipe.close()
        self._address_pipes = []

        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)


class RustFrontendProcessManager:
    """Manages a single Rust frontend subprocess.

    Launches the Rust vllm-rs binary in 'frontend' mode, passing the
    listening socket fd and ZMQ transport addresses. Provides the same
    interface as APIServerProcessManager for process monitoring.
    """

    def __init__(
        self,
        binary_path: str,
        sock: Any,
        args: argparse.Namespace,
        input_address: str,
        output_address: str,
        engine_start_index: int,
        engine_count: int,
        stats_update_address: str | None = None,
    ):
        import os
        import subprocess

        fd = sock.fileno()
        os.set_inheritable(fd, True)

        cmd = [
            binary_path,
            "frontend",
            "--listen-fd",
            str(fd),
            "--input-address",
            input_address,
            "--output-address",
            output_address,
            "--engine-start-index",
            str(engine_start_index),
            "--engine-count",
            str(engine_count),
        ]
        if stats_update_address is not None:
            cmd.extend(["--coordinator-address", stats_update_address])
        from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

        args_dict = jsonify_non_default_args(
            args,
            exclude={
                "api_server_count",
                # Python passes the bootstrapped engine range explicitly.
                "data_parallel_rank",
                "data_parallel_external_lb",
                "data_parallel_hybrid_lb",
            },
        )
        # The Rust `frontend` subcommand parses --args-json via serde_json,
        # which bypasses clap and therefore ignores any `#[arg(env = ...)]`
        # declarations on SharedRuntimeArgs fields. Forward the env-driven
        # values explicitly so VLLM_ENGINE_READY_TIMEOUT_S and
        # VLLM_HTTP_TIMEOUT_KEEP_ALIVE behave the same on both Python and Rust
        # frontends.
        args_dict["engine_ready_timeout_secs"] = envs.VLLM_ENGINE_READY_TIMEOUT_S
        args_dict["http_timeout_keep_alive"] = envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE
        args_json = json.dumps(args_dict, sort_keys=True)
        cmd.extend(["--args-json", args_json])

        logger.info("Launching Rust frontend: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, pass_fds=(fd,))

        # Create a process wrapper with a sentinel fd for monitoring
        self.processes: list[_SubprocessWrapper] = [
            _SubprocessWrapper(self._proc, "RustFrontend")
        ]

        self._finalizer = weakref.finalize(self, _shutdown_subprocesses, self.processes)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._finalizer.detach() is not None:
            _shutdown_subprocesses(self.processes, timeout=timeout)


class _SubprocessWrapper:
    """Wraps subprocess.Popen to provide the BaseProcess-like interface
    needed by wait_for_completion_or_failure."""

    def __init__(self, proc, name: str):
        self._proc = proc
        self.name = name
        self.pid = proc.pid
        self._sentinel_conn: connection.Connection | None = None
        self._sentinel_send: connection.Connection | None = None

        # Use a Pipe-based sentinel so subprocess monitoring works uniformly
        # across platforms with multiprocessing.connection.wait().
        recv, send = connection.Pipe(duplex=False)
        self._sentinel_conn = recv
        self._sentinel_send = send

        def monitor_subprocess() -> None:
            try:
                proc.wait()
            finally:
                with contextlib.suppress(Exception):
                    send.close()

        threading.Thread(
            target=monitor_subprocess, daemon=True, name=f"{name}Monitor"
        ).start()

    @property
    def sentinel(self):
        return self._sentinel_conn

    @property
    def exitcode(self) -> int | None:
        return self._proc.returncode if self._proc.poll() is not None else None

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def terminate(self):
        self._proc.terminate()

    def join(self, timeout=None):
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=timeout)

    def __del__(self):
        with contextlib.suppress(Exception):
            if self._sentinel_conn is not None:
                self._sentinel_conn.close()
            if self._sentinel_send is not None:
                self._sentinel_send.close()


def _shutdown_subprocesses(
    procs: list[_SubprocessWrapper], timeout: float | None = None
) -> None:
    """Shutdown subprocess wrappers (mirrors the shutdown() function)."""
    if timeout is None:
        timeout = 0.0
    timeout = max(timeout, 5.0)

    logger.debug(
        "[shutdown] Subprocess manager: start process_count=%d timeout=%ss",
        len(procs),
        timeout,
    )

    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    remaining_pids = [
        proc.pid for proc in procs if proc.is_alive() and proc.pid is not None
    ]
    if remaining_pids:
        logger.warning(
            "[shutdown] Subprocess manager: force killing remaining processes count=%d",
            len(remaining_pids),
        )
    for pid in remaining_pids:
        kill_process_tree(pid)

    logger.debug_once("[shutdown] Subprocess manager: complete")


def run_api_server_worker_proc(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Entrypoint for individual API server worker processes."""

    from vllm.entrypoints.openai.api_server import run_server_worker

    client_config = client_config or {}
    server_index = client_config.get("client_index", 0)

    # Set process title and add process-specific prefix to stdout and stderr.
    set_process_title("APIServer", str(server_index))
    decorate_logs()

    uvloop.run(
        run_server_worker(listen_address, sock, args, client_config, **uvicorn_kwargs)
    )


def wait_for_completion_or_failure(
    api_server_manager: "APIServerProcessManager | RustFrontendProcessManager",
    engine_manager: Union["CoreEngineProcManager", "CoreEngineActorManager"]
    | None = None,
    coordinator: "DPCoordinator | None" = None,
) -> None:
    """Wait for all processes to complete or detect if any fail.

    Raises an exception if any process exits with a non-zero status.

    Args:
        api_server_manager: The manager for API servers.
        engine_manager: The manager for engine processes.
            If CoreEngineProcManager, it manages local engines;
            if CoreEngineActorManager, it manages all engines.
        coordinator: The coordinator for data parallel.
    """

    try:
        logger.info("Waiting for API servers to complete ...")
        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        sentinel_to_proc: dict[Any, BaseProcess | _SubprocessWrapper | None] = {
            proc.sentinel: proc for proc in api_server_manager.processes
        }

        if coordinator:
            sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

        if engine_manager:
            core_shutdown_recv, core_shutdown_send = connection.Pipe(duplex=False)

            def monitor_engines():
                try:
                    engine_manager.monitor_engine_liveness()
                finally:
                    core_shutdown_send.close()
                    core_shutdown_recv.close()

            # start monitor for engine liveness
            threading.Thread(target=monitor_engines, daemon=True).start()
            sentinel_to_proc[core_shutdown_recv] = None  # type: ignore[assignment]

        # Check if any process terminates
        while sentinel_to_proc:
            # Wait for any process to terminate (or engine shutdown signal)
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc)

            # Process any terminated processes
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                if proc is not None and proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}"
                    )
                if engine_manager and engine_manager.failed_proc_name is not None:
                    raise RuntimeError(
                        f"Engine core process {engine_manager.failed_proc_name} "
                        "died unexpectedly."
                    )

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s", str(e))
        raise


# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
    """Shutdown processes with timeout.

    Args:
        procs: List of processes to shutdown
        timeout: Maximum time in seconds to wait for graceful shutdown
    """
    if timeout is None:
        # Keep a small grace period for best-effort cleanup paths that do not
        # have a user-configured shutdown timeout.
        timeout = 5.0

    logger.debug(
        "[shutdown] Process manager: start process_count=%d timeout=%ss",
        len(procs),
        timeout,
    )

    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # Allow time for remaining procs to terminate.
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    remaining_pids = [
        proc.pid for proc in procs if proc.is_alive() and proc.pid is not None
    ]
    if remaining_pids:
        logger.warning(
            "[shutdown] Process manager: force killing remaining processes count=%d",
            len(remaining_pids),
        )
    for pid in remaining_pids:
        kill_process_tree(pid)

    logger.debug_once("[shutdown] Process manager: complete")


def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


def report_usage_stats(
    vllm_config, usage_context: UsageContext = UsageContext.ENGINE_CONTEXT
) -> None:
    """Report usage statistics if enabled."""

    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    parallel_config = vllm_config.parallel_config
    attention_config = vllm_config.attention_config
    compilation_config = vllm_config.compilation_config
    speculative_config = vllm_config.speculative_config

    # Prepare KV connector string if applicable
    kv_connector = None
    if vllm_config.kv_transfer_config is not None:
        kv_connector = vllm_config.kv_transfer_config.kv_connector

    # Attention backend is None when set to "auto" (resolved at runtime per platform).
    attention_backend = (
        attention_config.backend.name if attention_config.backend is not None else None
    )

    # CompilationMode is an IntEnum; report the name for readability in dashboards.
    compilation_mode = (
        compilation_config.mode.name if compilation_config.mode is not None else None
    )

    # Speculative decoding fields default to None when spec decode is disabled.
    spec_decode_method = (
        speculative_config.method if speculative_config is not None else None
    )
    num_speculative_tokens = (
        speculative_config.num_speculative_tokens
        if speculative_config is not None
        else None
    )

    if model_config.using_transformers_backend():
        backend_cls = model_config._model_info.architecture
        # Show what was wrapped e.g. TransformersForCausalLM(Starcoder2ForCausalLM)
        architecture = f"{backend_cls}({model_config.architectures[0]})"
    else:
        architecture = get_architecture_class_name(model_config)

    usage_message.report_usage(
        architecture,
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype": str(model_config.dtype),
            "block_size": vllm_config.cache_config.block_size,
            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,
            "kv_cache_memory_bytes": vllm_config.cache_config.kv_cache_memory_bytes,
            # Quantization
            "quantization": model_config.quantization,
            "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
            # Feature flags
            "enable_lora": bool(vllm_config.lora_config),
            "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager": model_config.enforce_eager,
            "disable_custom_all_reduce": parallel_config.disable_custom_all_reduce,
            # Distributed parallelism settings
            "tensor_parallel_size": parallel_config.tensor_parallel_size,
            "data_parallel_size": parallel_config.data_parallel_size,
            "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
            "enable_expert_parallel": parallel_config.enable_expert_parallel,
            # All2All backend for MoE expert parallel
            "all2all_backend": parallel_config.all2all_backend,
            # KV connector used
            "kv_connector": kv_connector,
            # Batching limits — tuning knobs operators commonly override
            "max_model_len": model_config.max_model_len,
            "max_num_seqs": scheduler_config.max_num_seqs,
            "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
            # Attention backend (user-requested; None = auto-selected at runtime)
            "attention_backend": attention_backend,
            # torch.compile mode (e.g. NONE, STOCK_TORCH_COMPILE, VLLM_COMPILE)
            "compilation_mode": compilation_mode,
            # Speculative decoding configuration
            "spec_decode_method": spec_decode_method,
            "num_speculative_tokens": num_speculative_tokens,
            # Wide expert parallel: load balancer + redundant/total expert counts
            "enable_eplb": parallel_config.enable_eplb,
            "num_redundant_experts": parallel_config.eplb_config.num_redundant_experts,
            "num_experts": model_config.get_num_experts(),
        },
    )


_PROFILER_FUNC = None


def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    global _PROFILER_FUNC

    # fast path assume it is set
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    func = contextlib.nullcontext
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        import nvtx

        func = nvtx.annotate

    _PROFILER_FUNC = func
    return func(name)


def tensor_data(tensor: torch.Tensor) -> memoryview:
    """Get the raw data of a tensor as a uint8 memoryview, useful for
    serializing and hashing.

    Args:
        tensor: The input tensor.

    Returns:
        A memoryview of the tensor data as uint8.
    """
    return tensor.flatten().cpu().contiguous().view(torch.uint8).numpy().data


@dataclass
class IterationDetails:
    num_ctx_requests: int
    num_ctx_tokens: int
    num_generation_requests: int
    num_generation_tokens: int

    def __repr__(self) -> str:
        return f"IterationDetails(num_ctx_requests={self.num_ctx_requests},\
                 num_ctx_tokens={self.num_ctx_tokens}, \
                 num_generation_requests={self.num_generation_requests}, \
                 num_generation_tokens={self.num_generation_tokens})"


def compute_iteration_details(scheduler_output: SchedulerOutput) -> IterationDetails:
    """
    Compute the number of context/generation requests and tokens
    for the current iteration's scheduler output. A requests is regarded
    as a context request if its output tokens are still 0, an extended chunk
    of chunked prefill falls into this category.

    Args:
        scheduler_output: The scheduler output for the current iteration.

    Returns:
        An IterationDetails object containing the number of
        context/generation requests and tokens.
    """
    num_context_requests = 0
    num_context_tokens = 0
    num_generation_requests = 0
    num_generation_tokens = 0
    new_req_ids = {new_req.req_id for new_req in scheduler_output.scheduled_new_reqs}
    for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
        if scheduler_output.scheduled_cached_reqs.is_context_phase(req_id) or (
            req_id in new_req_ids
        ):
            num_context_requests += 1
            num_context_tokens += num_tokens
        else:
            num_generation_requests += 1
            num_generation_tokens += num_tokens
    return IterationDetails(
        num_context_requests,
        num_context_tokens,
        num_generation_requests,
        num_generation_tokens,
    )
