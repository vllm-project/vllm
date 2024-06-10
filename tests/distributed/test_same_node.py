import torch

from vllm.distributed.parallel_state import is_in_the_same_node

torch.distributed.init_process_group(backend="gloo")
test_result = is_in_the_same_node(torch.distributed.group.WORLD)
if test_result:
    exit(0)
else:
    exit(1)
