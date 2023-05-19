from typing import List, Optional, Tuple, Union

try:
    import ray
except ImportError:
    ray = None

from cacheflow.core.scheduler import Scheduler
from cacheflow.worker.worker import Worker


DeviceID = Tuple[int, str, int] # rank, node resource (node IP), device id


class Controller:

    def __init__(
        self,
        stage_id: int,
        stage_devices: List[DeviceID],
        world_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        distributed_init_method: str,
        model_name: str,
        dtype: str,
        seed: int,
        cache_dir: Optional[str],
        use_dummy_weights: bool,
        use_np_cache: bool,
        max_num_batched_tokens: int,
        max_num_sequences: int,
        use_ray: bool,
    ) -> None:
        self.stage_id = stage_id
        self.stage_devices = stage_devices
        self.model_name = model_name
        self.use_ray = use_ray

        # Which pipeline stage is this node assigned to?
        self.is_first_stage = stage_id == 0
        self.is_last_stage = False

        self.workers: List[Worker] = []
        for rank, node_resource, device_id in stage_devices:
            if self.use_ray:
                worker_cls = ray.remote(num_cpus=0,
                                        num_gpus=1,
                                        resources={node_resource: 1e-5})(Worker).remote
            else:
                worker_cls = Worker
            worker = worker_cls(
                model_name=model_name,
                dtype=dtype,
                seed=seed,
                distributed_init_method=distributed_init_method,
                rank=rank,
                world_size=world_size,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                cache_dir=cache_dir,
                use_dummy_weights=use_dummy_weights,
                use_np_cache=use_np_cache,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_sequences=max_num_sequences,
            )
            self.workers.append(worker)

    def get_num_available_blocks(self, block_size: int, cpu_swap_space: int,
                                 gpu_memory_utilization: float) -> List[Tuple[int, int]]:
        all_worker_results = []
        for worker in self.workers:
            executor = worker.get_num_available_blocks
            if self.use_ray:
                executor = executor.remote

            result = executor(
                block_size,
                cpu_swap_space,
                gpu_memory_utilization,
            )
            all_worker_results.append(result)
        if self.use_ray:
            all_worker_results = ray.get(all_worker_results)
        return all_worker_results

    def init_cache_engine(self, block_size: int, num_gpu_blocks: int,
                          num_cpu_blocks: int):
        all_worker_futures = []
        for worker in self.workers:
            executor = worker.init_cache_engine
            if self.use_ray:
                executor = executor.remote
            future = executor(
                block_size,
                num_gpu_blocks,
                num_cpu_blocks,
            )
            all_worker_futures.append(future)
        if self.use_ray:
            ray.get(all_worker_futures)

    def set_next(
        self,
        next_node: Union['Controller', 'Scheduler'],
    ) -> None:
        self.next_node = next_node
        self.is_last_stage = isinstance(next_node, Scheduler)

    def execute_stage(self, *args, **kwargs) -> None:
        all_outputs = []
        for worker in self.workers:
            executor = (worker.execute_stage.remote
                        if self.use_ray else worker.execute_stage)
            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.use_ray:
            all_outputs = ray.get(all_outputs)

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output

        if self.is_last_stage:
            self.next_node.post_step(output)
        else:
            # TODO: Support pipeline parallelism.
            assert False
