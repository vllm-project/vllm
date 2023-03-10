from typing import Dict, List, Union

from cacheflow.master.scheduler import Scheduler
from cacheflow.sequence import SequenceGroupInputs
from cacheflow.worker.worker import Worker


class Controller:

    def __init__(
        self,
        node_id: int,
        num_workers: int,
        model_name: str,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        dtype: str,
        seed: int,
    ) -> None:
        self.node_id = node_id
        self.num_workers = num_workers
        self.model_name = model_name
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks

        # Which pipeline stage is this node assigned to?
        self.is_first_stage = node_id == 0
        self.is_last_stage = False

        self.workers: List[Worker] = []
        for i in range(num_workers):
            worker = Worker(
                worker_id=node_id + i,
                gpu_id=i,
                model_name=model_name,
                block_size=block_size,
                num_gpu_blocks=num_gpu_blocks,
                num_cpu_blocks=num_cpu_blocks,
                dtype=dtype,
                seed=seed,
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
        input_seq_groups: List[SequenceGroupInputs],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # FIXME: Support tensor parallelism.
        assert len(self.workers) == 1
        worker = self.workers[0]
        output = worker.execute_stage(
            input_seq_groups,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
        )

        if self.is_last_stage:
            self.next_node.post_step(output)
        else:
            # TODO: Support pipeline parallelism.
            assert False
