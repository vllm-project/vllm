import ray

from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.utils import get_open_port


def init_test_distributed_environment(
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
    local_rank: int = -1,
) -> None:
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    init_distributed_environment(
        world_size=pp_size * tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
        local_rank=local_rank)
    ensure_model_parallel_initialized(tp_size, pp_size)


def multi_process_tensor_parallel(
    tp_size: int,
    pp_size: int,
    test_target,
) -> None:
    # Using ray helps debugging the error when it failed
    # as compared to multiprocessing.
    ray.init()

    distributed_init_port = get_open_port()
    refs = []
    for rank in range(tp_size * pp_size):
        refs.append(
            test_target.remote(tp_size, pp_size, rank, distributed_init_port))
    ray.get(refs)

    ray.shutdown()
