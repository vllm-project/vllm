from typing import List, Optional

import torch


class FakeGroupCoordinator:
    rank: int = 0
    ranks: List[int] = [0]
    world_size: int = 1
    local_rank: int = 0
    rank_in_group: int = 0

    def destroy(self):
        pass

    @property
    def first_rank(self):
        return self.ranks[0]

    @property
    def last_rank(self):
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        return self.rank == self.last_rank

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return input_

    def gather(self,
               input_: torch.Tensor,
               dst: int = 0,
               dim: int = -1) -> Optional[torch.Tensor]:
        return input_


def fix_distributed_environment():
    # This dirty_fix can make ParallelLinear etc. work properly.
    # Why should tp and model layers be coupled together?

    import vllm.distributed.parallel_state

    fake_parallel_group = FakeGroupCoordinator()
    vllm.distributed.parallel_state._TP = fake_parallel_group
    vllm.distributed.parallel_state._PP = fake_parallel_group
