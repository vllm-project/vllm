# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import contextlib
import multiprocessing
import time
import weakref
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    Union,
    overload,
)

import torch
from torch.autograd.profiler import record_function

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.utils.network_utils import get_open_port, get_open_zmq_ipc_path, get_tcp_uri
from vllm.utils.system_utils import kill_process_tree
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
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
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


def get_engine_client_zmq_addr(local_only: bool, host: str, port: int = 0) -> str:
    """Assign a new ZMQ socket address.

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP
    address (port == 0 means assign an available port)."""

    return (
        get_open_zmq_ipc_path()
        if local_only
        else (get_tcp_uri(host, port or get_open_port()))
    )


class APIServerProcessManager:
    """Manages a group of API server processes.

    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.
    """

    def __init__(
        self,
        target_server_fn: Callable,
        listen_address: str,
        engine_registry_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        stats_update_address: str | None = None,
    ):
        """Initialize and start API server worker processes.

        Args:
            target_server_fn: Function to call for each API server process
            listen_address: Address to listen for client connections
            engine_registry_address: Address of the engine registry used by
            the engine process manager, which can be used to notify it of
            events such as elastic EP scaling.
            sock: Socket for client connections
            args: Command line arguments
            num_servers: Number of API server processes to start
            input_addresses: Input addresses for each API server
            output_addresses: Output addresses for each API server
            stats_update_address: Optional stats update address
        """
        self.listen_address = listen_address
        self.sock = sock
        self.args = args

        # Start API servers
        spawn_context = multiprocessing.get_context("spawn")
        self.processes: list[BaseProcess] = []

        for i, in_addr, out_addr in zip(
            range(num_servers), input_addresses, output_addresses
        ):
            client_config = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_count": num_servers,
                "client_index": i,
                "engine_registry_address": engine_registry_address,
            }
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address

            proc = spawn_context.Process(
                target=target_server_fn,
                name=f"ApiServer_{i}",
                args=(listen_address, sock, args, client_config),
            )
            self.processes.append(proc)
            proc.start()

        logger.info("Started %d API server processes", len(self.processes))

        # Shutdown only the API server processes on garbage collection
        # The extra processes are managed by their owners
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

    def close(self) -> None:
        self._finalizer()


def wait_for_completion_or_failure(
    api_server_manager: APIServerProcessManager,
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

    from vllm.v1.engine.utils import CoreEngineActorManager, CoreEngineProcManager

    try:
        logger.info("Waiting for API servers to complete ...")

        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        def create_sentinel_to_proc():
            sentinel_to_proc: dict[Any, BaseProcess] = {
                proc.sentinel: proc for proc in api_server_manager.processes
            }

            if coordinator:
                sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

            if isinstance(engine_manager, CoreEngineProcManager):
                for proc in engine_manager.processes:
                    sentinel_to_proc[proc.sentinel] = proc

            return sentinel_to_proc

        actor_run_refs = []
        if isinstance(engine_manager, CoreEngineActorManager):
            actor_run_refs = engine_manager.get_run_refs()

        # Check if any process terminates
        sentinel_to_proc = create_sentinel_to_proc()
        while sentinel_to_proc or actor_run_refs:
            # Wait for any process to terminate
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc, timeout=5)

            # Rebuild the sentinel_to_process mapping because processes can
            # exit during scale-down, which is expected.
            sentinel_to_proc = create_sentinel_to_proc()

            # Process any terminated processes
            for sentinel in ready_sentinels:
                if sentinel not in sentinel_to_proc:
                    # Process may have exited during scale-down; ignore stale
                    # sentinel.
                    continue
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                if proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}"
                    )

            if actor_run_refs:
                import ray

                _, actor_run_refs = ray.wait(actor_run_refs, timeout=5)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s", str(e))
        raise
    finally:
        logger.info("Terminating remaining processes ...")
        api_server_manager.close()
        if coordinator:
            coordinator.close()
        if engine_manager:
            engine_manager.close()


# Note(rob): shutdown function cannot be a bound method,
# else the gc cannot collect the object.
def shutdown(procs: list[BaseProcess]):
    # Shutdown the process.
    for proc in procs:
        if proc.is_alive():
            proc.terminate()

    # Allow 5 seconds for remaining procs to terminate.
    deadline = time.monotonic() + 5
    for proc in procs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if proc.is_alive():
            proc.join(remaining)

    for proc in procs:
        if proc.is_alive() and (pid := proc.pid) is not None:
            kill_process_tree(pid)


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

    parallel_config = vllm_config.parallel_config

    # Prepare KV connector string if applicable
    kv_connector = None
    if vllm_config.kv_transfer_config is not None:
        kv_connector = vllm_config.kv_transfer_config.kv_connector

    usage_message.report_usage(
        get_architecture_class_name(vllm_config.model_config),
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype": str(vllm_config.model_config.dtype),
            "block_size": vllm_config.cache_config.block_size,
            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,
            "kv_cache_memory_bytes": vllm_config.cache_config.kv_cache_memory_bytes,
            # Quantization
            "quantization": vllm_config.model_config.quantization,
            "kv_cache_dtype": str(vllm_config.cache_config.cache_dtype),
            # Feature flags
            "enable_lora": bool(vllm_config.lora_config),
            "enable_prefix_caching": vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager": vllm_config.model_config.enforce_eager,
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
    return tensor.flatten().contiguous().view(torch.uint8).numpy().data


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
