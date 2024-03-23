# this script is not run with `pytest`.
# It is run with `torchrun`.
import os
import multiprocessing
import pytest
import torch
from vllm.model_executor.parallel_utils.pynccl import (
    NCCLCommunicator,
    ncclGetUniqueId,
)


def worker_fn(env):
    import os
    os.environ.update(env)

    # when environments are properly set, the usage is simple
    comm = NCCLCommunicator()
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32).cuda(comm.rank)
    comm.all_reduce(tensor)
    result = tensor.mean().cpu().item()
    assert result == comm.world_size


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl():
    number_of_processes = 2
    processes = []
    for i in range(number_of_processes):
        env = os.environ.copy()
        env['RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        env['MASTER_PORT'] = '12345'
        p = multiprocessing.Process(target=worker_fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def test_ncclGetUniqueId():
    unique_id = ncclGetUniqueId()
    # `list(unique_id.internal)` is something like this:
    # [34, -16, 23, 83, 109, -19, 59, 95, 2, 0, -86, 55, 10, -128, 0, 29, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # as long as the function doesn't raise an exception, we're good
    assert unique_id is not None
