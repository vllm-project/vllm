import os

import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.distributed.utils import StatelessProcessGroup
from vllm.utils import get_ip, get_open_port

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    if rank == 0:
        port = get_open_port()
        ip = get_ip()
        dist.broadcast_object_list([ip, port], src=0)
    else:
        recv = [None, None]
        dist.broadcast_object_list(recv, src=0)
        ip, port = recv

    stateless_pg = StatelessProcessGroup.create(ip, port, rank,
                                                dist.get_world_size())

    for pg in [dist.group.WORLD, stateless_pg]:
        test_result = all(in_the_same_node_as(pg, source_rank=0))

        expected = os.environ.get("VLLM_TEST_SAME_HOST", "1") == "1"
        assert test_result == expected, \
            f"Expected {expected}, got {test_result}"
        if pg == dist.group.WORLD:
            print("Same node test passed! when using torch distributed!")
        else:
            print("Same node test passed! when using StatelessProcessGroup!")
