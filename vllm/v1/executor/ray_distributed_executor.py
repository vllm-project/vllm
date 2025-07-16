# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from concurrent.futures import CancelledError, Future
from typing import Optional, Sequence, Union, cast

from vllm.executor.ray_distributed_executor import (  # noqa
    RayDistributedExecutor as RayDistributedExecutorV0)
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.logger import init_logger, logger

init_logger("vllm")

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

    def _init_executor(self) -> None:
        super()._init_executor()
        
        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        # Complete transfer tracker. Used by to track finished requests
        # [req_id -> n_finished_workers]
        self._recv_remaining_count = defaultdict[str,
                                                 int](lambda: self.parallel_config.world_size)
        self._send_remaining_count = defaultdict[str,
                                                 int](lambda: self.parallel_config.world_size)

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        if self.scheduler_config.async_scheduling:
            return 2
        return self.parallel_config.pipeline_parallel_size

    def _aggregate_workers_output(
            self, outputs: list[ModelRunnerOutput]) -> ModelRunnerOutput:
        # aggregate finished_sending, finished_recving from all workers

        finished_sending = set[str]()
        finished_recving = set[str]()
        for output in outputs:
            # update finished_sending
            for req_id in output.finished_sending or []:
                new_count = self._send_remaining_count[req_id] - 1
                if new_count == 0:
                    # got response from all workers, report back to scheduler
                    finished_sending.add(req_id)
                    del self._send_remaining_count[req_id]
                else:
                    self._send_remaining_count[req_id] = new_count

            # update finished_recving
            for req_id in output.finished_recving or []:
                new_count = self._recv_remaining_count[req_id] - 1
                if new_count == 0:
                    # got response from all workers, report back to scheduler
                    finished_recving.add(req_id)
                    del self._recv_remaining_count[req_id]
                else:
                    self._recv_remaining_count[req_id] = new_count

        # select output of the worker specified by output_rank
        output = outputs[0]

        # set the aggregated finished_sending / finished_recving
        if finished_sending:
            output.finished_sending = finished_sending
        if finished_recving:
            output.finished_recving = finished_recving

        return output

    def _async_aggregate_workers_output(
        self, output_futures: Sequence[Union[Future[ModelRunnerOutput], FutureWrapper]]
    ) -> Future[ModelRunnerOutput]:
        """Takes a list of futures and returns a single future which resolves
        to the respective list of outputs."""
        result_future: Future[ModelRunnerOutput] = Future()

        outputs: list[Optional[ModelRunnerOutput]] = [None
                                                      ] * len(output_futures)

        def make_callback(idx):

            def callback(fut):
                if result_future.done():
                    return

                try:
                    outputs[idx] = fut.result()
                except CancelledError:
                    result_future.cancel()
                except Exception as e:
                    result_future.set_exception(e)

                # Check if all outputs are ready
                if all(outputs):
                    result_future.set_result(
                        self._aggregate_workers_output(
                            cast(list[ModelRunnerOutput], outputs)))

            return callback

        for i, output_future in enumerate(output_futures):
            output_future.add_done_callback(make_callback(i))

        return result_future

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

        if not self.has_connector:
            # get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if self.max_concurrent_batches == 1:
                return refs[0].get()

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs[0])

        # get output from all workers when connector is present
        if self.max_concurrent_batches == 1:
            # Block and get results from all workers
            outputs = [ref.get() for ref in refs]
            return self._aggregate_workers_output(outputs)
        else:
            # Return a future that will aggregate outputs from all workers
            output_futures = [FutureWrapper(ref) for ref in refs]
            return self._async_aggregate_workers_output(output_futures)
