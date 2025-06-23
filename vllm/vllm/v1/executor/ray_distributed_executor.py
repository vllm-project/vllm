# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from typing import Union

from vllm.executor.ray_distributed_executor import (  # noqa
    RayDistributedExecutor as RayDistributedExecutorV0)
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput


class FutureWrapper(Future):
    """A wrapper around a Ray output reference to meet the interface
    of .execute_model().
    """

    def __init__(self, ref):
        super().__init__()
        self.ref = ref

    def result(self, timeout=None):
        if timeout is not None:
            raise NotImplementedError("timeout is not supported")
        return self.ref.get()


class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    """Ray distributed executor using Ray Compiled Graphs."""

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        return self.parallel_config.pipeline_parallel_size

    def execute_model(
        self,
        scheduler_output,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Execute the model on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute(scheduler_output)  # type: ignore

        # When PP is not used, we block here until the result is available.
        if self.max_concurrent_batches == 1:
            return refs[0].get()

        # When PP is used, we return a FutureWrapper immediately so that
        # the scheduler can yield to the next batch.
        return FutureWrapper(refs[0])
