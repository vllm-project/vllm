# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

import ray

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.executor.ray_distributed_executor import (  # noqa
    RayDistributedExecutor as RayDistributedExecutorV0)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


class FutureWrapper(Future):
    """A wrapper around Ray output reference to meet the interface
    of .execute_model(): The top level (core busy loop) expects .result() api 
    to block and return a single output.
    
    If aggregator is provided, the outputs from all workers are aggregated upon 
    the result() call. If not only the first worker's output is returned.
    """

    def __init__(self, refs, aggregator: Optional[KVOutputAggregator] = None):
        super().__init__()
        self.refs = refs
        self.aggregator = aggregator

    def result(self, timeout=None):
        if timeout is not None:
            raise NotImplementedError("timeout is not supported")

        if self.aggregator is None:
            return self.refs[0].get()

        outputs = [ref.get() for ref in self.refs]
        return self.aggregator.aggregate(outputs, output_rank=0)


class KVCacheFutureWrapper(Future):
    """A wrapper around concurrent.futures.Future objects to handle KV cache operations.
    
    This class is used when pulling KV cache from workers, where the futures
    are ThreadPoolExecutor futures that resolve Ray object references.
    """

    def __init__(self, futures, parallel_config, aggregator=None):
        super().__init__()
        self.futures = futures
        self.parallel_config = parallel_config
        self.aggregator = aggregator
        self._result_cache = None
        self._completed = False

    def result(self, timeout=None):
        if timeout is not None:
            raise NotImplementedError("timeout is not supported")

        if self._completed:
            return self._result_cache

        # Wait for all futures to complete and get their results
        outputs = [future.result() for future in self.futures]

        # Only returns ModelRunnerOutput from TP rank=0 and PP rank=-1
        out_rank = self.parallel_config.world_size - self.parallel_config.tensor_parallel_size
        if self.aggregator is None:
            # No aggregator means no connector, so return the first output
            self._result_cache = outputs[out_rank]
        else:
            # Use aggregator to combine outputs from all workers
            self._result_cache = self.aggregator.aggregate(
                outputs, output_rank=out_rank)
        
        self._completed = True
        return self._result_cache

    def get(self, timeout=None):
        """Implement Ray object reference interface for compatibility.
        
        This is needed because the engine core sometimes expects .get() method
        when dealing with Ray object references.
        """
        return self.result(timeout)

    def done(self):
        """Check if all underlying futures are completed."""
        if self._completed:
            return True
        return all(future.done() for future in self.futures)

    def cancel(self):
        """Cancel all underlying futures."""
        success = True
        for future in self.futures:
            if not future.cancel():
                success = False
        return success

    def cancelled(self):
        """Check if any underlying future is cancelled."""
        return any(future.cancelled() for future in self.futures)


class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    """Ray distributed executor using Ray Compiled Graphs."""

    supports_pp: bool = True

    def _init_executor(self) -> None:
        super()._init_executor()

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        """Execute the model on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        Returns:
            The model runner output.
        """
        if not scheduler_output.total_num_scheduled_tokens:
            return self.pull_kvcache(scheduler_output)

        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute(scheduler_output)  # type: ignore

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if self.max_concurrent_batches == 1:
                return refs[0].get()

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs)

        # Get output from all workers when connector is present
        if self.max_concurrent_batches == 1:
            # Block and get results from all workers
            outputs = [ref.get() for ref in refs]
            return self.kv_output_aggregator.aggregate(outputs)

        # Return a future that will aggregate outputs from all workers
        return FutureWrapper(refs, self.kv_output_aggregator)

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self._run_workers("reinitialize_distributed", reconfig_request)
        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            self.shutdown()
        return

    def pull_kvcache(
        self,
        scheduler_output,
    ) :
        """Pull KVCache on the Ray workers.

        Args:
            scheduler_output: The scheduler output to execute.

        """
        output_refs = [
            (worker.pull_kvcache_ray.remote(scheduler_output)
             for worker in workers)
            for workers in self.pp_tp_workers]
        all_object_refs = []
        for generator in output_refs:
            for obj_ref in generator:
                all_object_refs.append(obj_ref)
        
        def create_future_for_ref(obj_ref):
            return executor.submit(ray.get, obj_ref)

        executor = ThreadPoolExecutor()
        futures = [
            create_future_for_ref(obj_ref) for obj_ref in all_object_refs
        ]

        return KVCacheFutureWrapper(
            futures, self.parallel_config,
            self.kv_output_aggregator if self.has_connector else None)

