# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch.distributed as dist

from vllm.distributed.parallel_state import _node_count
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils import get_ip, get_open_port

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        port = get_open_port()
        ip = get_ip()
        dist.broadcast_object_list([ip, port], src=0)
    else:
        recv = [None, None]
        dist.broadcast_object_list(recv, src=0)
        ip, port = recv

    stateless_pg = StatelessProcessGroup.create(ip, port, rank, world_size)

    for pg in [dist.group.WORLD, stateless_pg]:
        test_result = _node_count(pg)

        # Expected node count based on environment variable)
        expected = int(os.environ.get("NUM_NODES", "1"))

        assert test_result == expected, \
            f"Expected {expected} nodes, got {test_result}"

        if pg == dist.group.WORLD:
            print(f"Node count test passed! Got {test_result} nodes "
                  f"when using torch distributed!")
        else:
            print(f"Node count test passed! Got {test_result} nodes "
                  f"when using StatelessProcessGroup!")
