# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.quantized_allreduce import (
    two_shot_quantized_allreduce,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

NUMEL = 4 * 1024 * 1024


def quantized_allreduce_worker(
    local_rank: int, world_size: int, use_fp8: bool, group_size: int
):
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12399",
        }
    )
    dist.init_process_group("nccl", device_id=device)

    msg = torch.full((NUMEL,), local_rank + 1, dtype=torch.bfloat16, device=device)
    golden = msg.clone()
    dist.all_reduce(golden)

    output = two_shot_quantized_allreduce(
        msg.clone(), use_fp8=use_fp8, group_size=group_size
    )
    torch.accelerator.synchronize()
    # Constant-per-rank inputs are exactly representable after
    # per-group quantization, so the result should match NCCL.
    torch.testing.assert_close(output, golden)

    dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Quantized allreduce is only available for CUDA.",
)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda"],
    reason="Only test on CUDA",
)
@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("use_fp8", [False, True], ids=["int8", "fp8"])
@pytest.mark.parametrize("group_size", [128, 256])
def test_quantized_allreduce(tp_size, use_fp8, group_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    mp.spawn(
        quantized_allreduce_worker,
        args=(tp_size, use_fp8, group_size),
        nprocs=tp_size,
    )
    cleanup_dist_env_and_memory()
