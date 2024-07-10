import os

import torch

from vllm.distributed.parallel_state import in_the_same_node_as

torch.distributed.init_process_group(backend="gloo")
test_result = all(
    in_the_same_node_as(torch.distributed.group.WORLD, source_rank=0))

expected = os.environ.get("VLLM_TEST_SAME_HOST", "1") == "1"
assert test_result == expected, f"Expected {expected}, got {test_result}"
