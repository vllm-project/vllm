# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, TypeVar

from vllm.config import VllmConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.executor.executor_base import ExecutorBase
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import UniProcExecutor as _UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

_R = TypeVar("_R", default=Any)

FailureCallback = Callable[[], None]

# For backwards compatibility.
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher


class Executor(ExecutorBase, ABC):
    """Abstract base class for vLLM Executors."""

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            from vllm.v1.executor.ray_distributed_executor import (  # noqa
                RayDistributedExecutor,
            )

            executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """
        Initialize the KV caches and begin the model execution loop of the
        underlying workers.
        """
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
        self.collective_rpc("compile_or_warm_up_model")

    def register_failure_callback(self, callback: FailureCallback):
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
    ) -> list[_R] | list[Future[_R]]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        raise NotImplementedError

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        output = self.collective_rpc(
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        output = self.collective_rpc("take_draft_token_ids")
        return output[0]

    @property
    def max_concurrent_batches(self) -> int:
        return 1

    def profile(self, is_start: bool = True):
        self.collective_rpc("profile", args=(is_start,))
