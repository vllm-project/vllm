from vllm.distributed.parallel_state import is_in_the_same_node

torch.distributed.init_process_group(backend="gloo")
ans = is_in_the_same_node(torch.distributed.group.WORLD)
if ans:
    exit(0)
else:
    exit(1)
