# Implementation of the Coordinator interface based on
# PyTorch's distributed package.

import os
from typing import List, Optional

import torch
import torch.distributed as dist

from vllm.interfaces.coordinator import Coordinator


class TorchDistributedCoordinator(Coordinator):

    def __init__(self, groups: Optional[List[List[int]]] = None):
        assert 'RANK' in os.environ, \
            'RANK not found in environment'
        assert 'WORLD_SIZE' in os.environ, \
            'WORLD_SIZE not found in environment'
        assert 'LOCAL_RANK' in os.environ, \
            'LOCAL_RANK not found in environment'
        assert 'LOCAL_WORLD_SIZE' in os.environ, \
            'LOCAL_WORLD_SIZE not found in environment'
        assert 'MASTER_ADDR' in os.environ, \
            'MASTER_ADDR not found in environment'
        assert 'MASTER_PORT' in os.environ, \
            'MASTER_PORT not found in environment'
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        groups = groups or [list(range(world_size))]
        super().__init__(rank=rank,
                         world_size=world_size,
                         local_rank=local_rank,
                         local_world_size=local_world_size,
                         groups=groups)
        self.process_group = None

    def initialize(self):
        # in `torch.distributed`, we can only initialize the process group once
        # if the process group is already initialized, we should not initialize
        # it again, but need to use `new_group` to create a new group.
        # in either case, `self.group` contains all the processes. It's just we
        # use `gloo` backend ourselves inside this coordinator,
        # to avoid interfering with other process groups.
        if not dist.is_initialized():
            dist.init_process_group(backend='gloo')
            self.process_group = dist.group.WORLD
        else:
            # each time we create a new group, we need **all the processes**
            # to call `new_group`. For example, when we have 4 processes,
            # and we want to create two groups, [0, 1] and [2, 3],
            # 1. [0, 1, 2, 3] should call `new_group` with ranks=[0, 1]
            # 2. [0, 1, 2, 3] should call `new_group` with ranks=[2, 3]
            for group in self.groups:
                result_group = dist.new_group(ranks=group, backend='gloo')
                if self.rank in group:
                    self.process_group = result_group
        super().initialize()

    def barrier(self):
        dist.barrier(group=self.process_group)

    def broadcast(self, message: bytearray, src: int = 0):
        tensor = torch.tensor(list(message), dtype=torch.uint8)
        dist.broadcast(tensor, src=src, group=self.process_group)
        data = tensor.tolist()
        for i in range(len(message)):
            message[i] = data[i]

    def __del__(self):
        # `dist` module might have been already destroyed
        if hasattr(dist, 'destroy_process_group'):
            dist.destroy_process_group(self.process_group)
