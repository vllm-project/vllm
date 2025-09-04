from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaseUpdator(ABC):
    @abstractmethod
    def step(self, model, is_dummy, is_profile, log_stats):
        """
        Performs a single synchronous step of the updator's main operation.
        """
        pass

    @abstractmethod
    def step_async(self):
        """
        Performs a single asynchronous step of the updator's main operation.
        This method should return immediately, allowing other operations to
        proceed concurrently.
        """
        pass

    @abstractmethod
    def step_before_forward(self):
        """
        Executes operations that need to occur before the forward pass
        of a model.
        """
        pass

    @abstractmethod
    def step_after_forward(self):
        """
        Executes operations that need to occur after the forward pass
        of a model.
        """

        pass

    @staticmethod
    def _node_count_with_rank_mapping(
            pg: Union[ProcessGroup, StatelessProcessGroup],
            rank_mapping: dict[int, int],
    ) -> int:
        """
        Calculates the number of distinct physical nodes involved in a process group,
        considering a given rank mapping.

        Args:
            pg: The PyTorch distributed ProcessGroup or a custom StatelessProcessGroup.
            rank_mapping: A dictionary mapping global ranks to their logical ranks
                          or a special value like -1 (for pending shutdown).

        Returns:
            The total number of distinct physical nodes.
        """
        if isinstance(pg, ProcessGroup):
            world_size = torch.distributed.get_world_size(group=pg)
        else:
            world_size = pg.world_size

        if world_size == 1:
            return 1

        # Build node assignment map
        node_assignment = [0] * world_size  # rank -> node_id
        next_node_id = 0

        for current_rank in range(world_size):
            if node_assignment[current_rank] != 0:
                continue  # Already assigned to a node

            assert current_rank in rank_mapping
            if rank_mapping[current_rank] == -1:
                continue  # Pending shutdown

            # Assign current rank to a new node
            next_node_id += 1
            node_assignment[current_rank] = next_node_id

            # Find all ranks on the same node as current_rank
            same_node_flags = in_the_same_node_as(pg, current_rank)
            for other_rank, is_same_node in enumerate(same_node_flags):
                if is_same_node and node_assignment[other_rank] == 0:
                    node_assignment[other_rank] = next_node_id

        return next_node_id
