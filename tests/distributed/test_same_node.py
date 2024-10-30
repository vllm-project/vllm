import os

import torch.distributed as dist

from vllm.distributed.parallel_state import in_the_same_node_as

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    test_result = all(in_the_same_node_as(dist.group.WORLD, source_rank=0))

    expected = os.environ.get("VLLM_TEST_SAME_HOST", "1") == "1"
    assert test_result == expected, f"Expected {expected}, got {test_result}"
    print("Same node test passed!")
