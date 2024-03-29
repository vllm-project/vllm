# Implementation of the Coordinator interface based on
# PyTorch's distributed package.

import os

import torch
import torch.distributed as dist

from vllm.interfaces.coordinator import Coordinator


class TorchDistributedCoordinator(Coordinator):

    def __init__(self):
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
        super().__init__(rank=rank,
                         world_size=world_size,
                         local_rank=local_rank,
                         local_world_size=local_world_size)

    def initialize(self):
        assert not dist.is_initialized()
        dist.init_process_group(backend='gloo')
        super().initialize()

    def barrier(self):
        dist.barrier()

    def broadcast(self, message: bytearray, src: int = 0):
        tensor = torch.tensor(list(message), dtype=torch.uint8)
        dist.broadcast(tensor, src=src)
        data = tensor.tolist()
        for i in range(len(message)):
            message[i] = data[i]
