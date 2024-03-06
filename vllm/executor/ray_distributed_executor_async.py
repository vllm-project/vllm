import asyncio
from functools import partial
from typing import Any, Dict, List, Optional

from vllm.executor.ray_distributed_executor import (
    RayDistributedModelExecutor, USE_RAY_COMPILED_DAG)
from vllm.sequence import SequenceGroupMetadata


class RayDistributedModelExecutorAsync(RayDistributedModelExecutor):

    async def _run_workers_async(
        self,
        method: str,
        *args,
        driver_args: Optional[List[Any]] = None,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        coros = []

        if driver_args is None:
            driver_args = args
        if driver_kwargs is None:
            driver_kwargs = kwargs

        # Run the driver worker asynchronously.
        driver_executor = getattr(self.driver_worker, method)
        coros.append(asyncio.get_event_loop().run_in_executor(
            None, partial(driver_executor, *driver_args, **driver_kwargs)))

        # Run the ray workers asynchronously.
        for worker in self.workers:
            coros.append(worker.execute_method.remote(method, *args, **kwargs))

        all_outputs = await asyncio.gather(*coros)
        return all_outputs

    async def execute_model_async(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            blocks_to_swap_in: Dict[int, int], blocks_to_swap_out: Dict[int,
                                                                        int],
            blocks_to_copy: Dict[int, List[int]]):
        all_outputs = await self._run_workers_async(
            "execute_model",
            driver_kwargs={
                "seq_group_metadata_list": seq_group_metadata_list,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            },
            use_ray_compiled_dag=USE_RAY_COMPILED_DAG)

        # Only the driver worker returns the sampling results.
        output = all_outputs[0]
        return output

    async def check_health_async(self):
        """Raises an error if engine is unhealthy."""
        self._check_if_any_actor_is_dead()
