# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.collective_fusion import AllGatherDecomposePass
from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

from ...utils import multi_gpu_test
from ..backend import TestBackend


class AllGatherModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        gathered = tensor_model_parallel_all_gather(x, dim=-1)
        variance = gathered.pow(2).mean(-1, keepdim=True)
        return gathered * torch.rsqrt(variance + 1e-6)

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.all_gather_raw.default]


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA"
)
def test_all_gather_decompose_pass():
    torch.multiprocessing.spawn(
        _test_worker,
        args=(2,),
        nprocs=2,
    )


def _test_worker(local_rank: int, world_size: int):
    set_random_seed(0)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    update_environment_variables({
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12346",
    })

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig()
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))

    decompose_pass = AllGatherDecomposePass(vllm_config)

    with set_current_vllm_config(vllm_config):
        backend = TestBackend(decompose_pass)
        model = AllGatherModel()

        x = torch.randn((8, 16, 16), dtype=torch.float16)

        # Get eager reference output
        with torch.no_grad():
            eager_out = model(x)

        # Compile and run
        compiled_model = torch.compile(model, backend=backend)
        compiled_out = compiled_model(x)

        # Check correctness
        torch.testing.assert_close(compiled_out, eager_out, rtol=1e-2, atol=1e-2)

        # Check ops were transformed
        backend.check_before_ops(model.ops_in_model_before())
        backend.check_after_ops(model.ops_in_model_after())
