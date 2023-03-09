from typing import Dict, List, Union, Tuple

import ray

from cacheflow.master.scheduler import Scheduler
from cacheflow.worker.worker import Worker


class Controller:

    def __init__(
        self,
        stage_id: int,
        stage_devices: List[Tuple[int, 'str', int]],
        world_size: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        distributed_init_method: str,
        model_name: str,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: str = 'half',
    ) -> None:
        self.stage_id = stage_id
        self.stage_devices = stage_devices
        self.model_name = model_name
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        # Which pipeline stage is this node assigned to?
        self.is_first_stage = stage_id == 0
        self.is_last_stage = False

        self.workers: List[Worker] = []
        for rank, node_resource, device_id in stage_devices:
            worker_cls = ray.remote(num_cpus=0,
                                    num_gpus=1,
                                    resources={node_resource: 1e-5})(Worker)
            worker = worker_cls.remote(
                model_name=model_name,
                block_size=block_size,
                num_gpu_blocks=num_gpu_blocks,
                num_cpu_blocks=num_cpu_blocks,
                dtype=dtype,
                distributed_init_method=distributed_init_method,
                rank=rank,
                local_rank=device_id,
                world_size=world_size,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
            )
            self.workers.append(worker)

    def set_next(
        self,
        next_node: Union['Controller', 'Scheduler'],
    ) -> None:
        self.next_node = next_node
        self.is_last_stage = isinstance(next_node, Scheduler)

    def execute_stage(
        self,
        prompt_tokens: Dict[int, List[int]],
        generation_tokens: Dict[int, int],
        context_lens: Dict[int, int],
        block_tables: Dict[int, List[int]],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, int],
    ) -> None:
        for worker in self.workers:
            output = worker.execute_stage.remote(
                prompt_tokens,
                generation_tokens,
                context_lens,
                block_tables,
                blocks_to_swap_in,
                blocks_to_swap_out,
                blocks_to_copy,
            )
            output = ray.get(output)

        if self.is_last_stage:
            self.next_node.post_step(output)
        else:
            # TODO: Support pipeline parallelism.
            assert False
