from typing import List, Optional, Tuple

try:
    import ray
except ImportError:
    ray = None

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
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: str,
        seed: int,
        cache_dir: Optional[str],
        use_dummy_weights: bool,
        use_np_cache: bool,
        max_num_batched_tokens: int,
        use_ray: bool,
    ) -> None:
        self.stage_id = stage_id
        self.stage_devices = stage_devices
        self.model_name = model_name
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.use_ray = use_ray

        # Which pipeline stage is this node assigned to?
        self.is_first_stage = stage_id == 0
        self.is_last_stage = False

        self.workers: List[Worker] = []
        for rank, node_resource, _ in stage_devices:
            if self.use_ray:
                worker_cls = ray.remote(num_cpus=0,
                                        num_gpus=1,
                                        resources={node_resource: 1e-5})(Worker).remote
            else:
                worker_cls = Worker
            worker = worker_cls(
                model_name=model_name,
                block_size=block_size,
                num_gpu_blocks=num_gpu_blocks,
                num_cpu_blocks=num_cpu_blocks,
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
            )
            self.workers.append(worker)

    def execute_stage(self, *args, **kwargs):
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

        # TODO: Support pipeline parallelism.
        return output
