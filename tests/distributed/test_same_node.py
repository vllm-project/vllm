import os

import torch

from vllm.distributed.parallel_state import is_in_the_same_node

torch.distributed.init_process_group(backend="gloo")
test_result = is_in_the_same_node(torch.distributed.group.WORLD)

expected = os.environ.get("VLLM_TEST_SAME_HOST", "1") == "1"
assert test_result == expected, f"Expected {expected}, got {test_result}"
