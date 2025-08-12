# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import multiprocessing
import time
import weakref
from collections import defaultdict
from collections.abc import Sequence
from multiprocessing import connection
from multiprocessing.process import BaseProcess
from typing import (TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar,
                    Union, overload)

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled,
                                  usage_message)
from vllm.utils import (get_open_port, get_open_zmq_ipc_path, get_tcp_uri,
                        kill_process_tree)

if TYPE_CHECKING:
    from vllm.attention.layer import Attention
    from vllm.v1.engine.coordinator import DPCoordinator
    from vllm.v1.engine.utils import (CoreEngineActorManager,
                                      CoreEngineProcManager)

logger = init_logger(__name__)

T = TypeVar("T")


class ConstantList(Generic[T], Sequence):

    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise Exception("Cannot append to a constant list")

    def extend(self, item):
        raise Exception("Cannot extend a constant list")

    def insert(self, item):
        raise Exception("Cannot insert into a constant list")

    def pop(self, item):
        raise Exception("Cannot pop from a constant list")

    def remove(self, item):
        raise Exception("Cannot remove from a constant list")

    def clear(self):
        raise Exception("Cannot clear a constant list")

    def index(self,
              item: T,
              start: int = 0,
              stop: Optional[int] = None) -> int:
        return self._x.index(item, start,
                             stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]:
        ...

    def __getitem__(self, item: Union[int, slice]) -> Union[T, list[T]]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T):
        ...

    @overload
    def __setitem__(self, s: slice, value: T, /):
        ...

    def __setitem__(self, item: Union[int, slice], value: Union[T, list[T]]):
        raise Exception("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise Exception("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"


def get_engine_client_zmq_addr(local_only: bool,
                               host: str,
                               port: int = 0) -> str:
    """Assign a new ZMQ socket address.

    If local_only is True, participants are colocated and so a unique IPC
    address will be returned.

    Otherwise, the provided host and port will be used to construct a TCP
    address (port == 0 means assign an available port)."""

    return get_open_zmq_ipc_path() if local_only else (get_tcp_uri(
        host, port or get_open_port()))


class APIServerProcessManager:
    """Manages a group of API server processes.
    
    Handles creation, monitoring, and termination of API server worker
    processes. Also monitors extra processes to check if they are healthy.
    """

    def __init__(
        self,
        target_server_fn: Callable,
        listen_address: str,
        sock: Any,
        args: argparse.Namespace,
        num_servers: int,
        input_addresses: list[str],
        output_addresses: list[str],
        stats_update_address: Optional[str] = None,
    ):
        """Initialize and start API server worker processes.
        
        Args:
            target_server_fn: Function to call for each API server process
            listen_address: Address to listen for client connections
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

        for i, in_addr, out_addr in zip(range(num_servers), input_addresses,
                                        output_addresses):
            client_config = {
                "input_address": in_addr,
                "output_address": out_addr,
                "client_index": i
            }
            if stats_update_address is not None:
                client_config["stats_update_address"] = stats_update_address

            proc = spawn_context.Process(target=target_server_fn,
                                         name=f"ApiServer_{i}",
                                         args=(listen_address, sock, args,
                                               client_config))
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
        engine_manager: Optional[Union["CoreEngineProcManager",
                                       "CoreEngineActorManager"]] = None,
        coordinator: Optional["DPCoordinator"] = None) -> None:
    """Wait for all processes to complete or detect if any fail.
    
    Raises an exception if any process exits with a non-zero status.

    Args:
        api_server_manager: The manager for API servers.
        engine_manager: The manager for engine processes.
            If CoreEngineProcManager, it manages local engines;
            if CoreEngineActorManager, it manages all engines.
        coordinator: The coordinator for data parallel.
    """

    from vllm.v1.engine.utils import (CoreEngineActorManager,
                                      CoreEngineProcManager)

    try:
        logger.info("Waiting for API servers to complete ...")
        # Create a mapping of sentinels to their corresponding processes
        # for efficient lookup
        sentinel_to_proc: dict[Any, BaseProcess] = {
            proc.sentinel: proc
            for proc in api_server_manager.processes
        }

        if coordinator:
            sentinel_to_proc[coordinator.proc.sentinel] = coordinator.proc

        actor_run_refs = []
        if isinstance(engine_manager, CoreEngineProcManager):
            for proc in engine_manager.processes:
                sentinel_to_proc[proc.sentinel] = proc
        elif isinstance(engine_manager, CoreEngineActorManager):
            actor_run_refs = engine_manager.get_run_refs()

        # Check if any process terminates
        while sentinel_to_proc or actor_run_refs:
            # Wait for any process to terminate
            ready_sentinels: list[Any] = connection.wait(sentinel_to_proc,
                                                         timeout=5)

            # Process any terminated processes
            for sentinel in ready_sentinels:
                proc = sentinel_to_proc.pop(sentinel)

                # Check if process exited with error
                if proc.exitcode != 0:
                    raise RuntimeError(
                        f"Process {proc.name} (PID: {proc.pid}) "
                        f"died with exit code {proc.exitcode}")

            if actor_run_refs:
                import ray
                _, actor_run_refs = ray.wait(actor_run_refs, timeout=5)

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down API servers...")
    except Exception as e:
        logger.exception("Exception occurred while running API servers: %s",
                         str(e))
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


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, "Attention"],
    runner_kv_caches: list[torch.Tensor],
) -> None:
    """
    Bind the allocated KV cache to both ModelRunner and forward context so
    that the KV cache can be used in the forward pass.

    This function:
      1) Fills the ModelRunner's kv cache list (`runner_kv_caches`) with
         kv_caches.
      2) Associates each attention layer in the `forward_context` with its 
         corresponding KV cache in kv_caches.

    Args:
        kv_caches: The allocated kv_caches with layer names as keys.
        forward_context: The global forward context containing all Attention 
        layers with layer names as keys.
        runner_kv_caches: The kv_cache declared by ModelRunner.
    """
    # Bind kv_caches to ModelRunner
    assert len(runner_kv_caches) == 0

    # Convert kv_caches dict to a list of tensors in the order of layer_index.
    index2name = defaultdict(list)
    for layer_name in kv_caches:
        index2name[extract_layer_index(layer_name)].append(layer_name)

    for layer_index in sorted(index2name.keys()):
        layer_names = index2name[layer_index]
        if len(layer_names) > 1:
            # One typical case is encoder-decoder model, e.g., bart.
            # The cross attention and self attention in the same decoder layer
            # has different layer_name but the same layer_index.
            raise NotImplementedError
        layer_name = layer_names[0]
        runner_kv_caches.append(kv_caches[layer_name])

    # Bind kv_caches to forward context
    for layer_name, kv_cache in kv_caches.items():
        # NOTE: Use list because of v0 PP virtual engine.
        forward_context[layer_name].kv_cache = [kv_cache]


def copy_slice(from_tensor: torch.Tensor, to_tensor: torch.Tensor,
               length: int) -> torch.Tensor:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


def report_usage_stats(
        vllm_config,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT) -> None:
    """Report usage statistics if enabled."""

    if not is_usage_stats_enabled():
        return

    from vllm.model_executor.model_loader import get_architecture_class_name

    usage_message.report_usage(
        get_architecture_class_name(vllm_config.model_config),
        usage_context,
        extra_kvs={
            # Common configuration
            "dtype":
            str(vllm_config.model_config.dtype),
            "tensor_parallel_size":
            vllm_config.parallel_config.tensor_parallel_size,
            "block_size":
            vllm_config.cache_config.block_size,
            "gpu_memory_utilization":
            vllm_config.cache_config.gpu_memory_utilization,

            # Quantization
            "quantization":
            vllm_config.model_config.quantization,
            "kv_cache_dtype":
            str(vllm_config.cache_config.cache_dtype),

            # Feature flags
            "enable_lora":
            bool(vllm_config.lora_config),
            "enable_prompt_adapter":
            bool(vllm_config.prompt_adapter_config),
            "enable_prefix_caching":
            vllm_config.cache_config.enable_prefix_caching,
            "enforce_eager":
            vllm_config.model_config.enforce_eager,
            "disable_custom_all_reduce":
            vllm_config.parallel_config.disable_custom_all_reduce,
        })
