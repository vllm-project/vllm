# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the ROCm AITER AllReduce + RMSNorm fusion pass.

Mirrors test_fusion_all_reduce.py but targets the ROCm/AITER path.
Requires 2 ROCm GPUs and the AITER library.
"""

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.utils import multi_gpu_test
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed


class TestAllReduceRMSNormModel(torch.nn.Module):
    """
    Simple model with 4 allreduce+rmsnorm pairs that should each
    be fused into a single rocm_aiter_fused_allreduce_rmsnorm op.
    """

    def __init__(self, hidden_size=16, token_num=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = torch.nn.ModuleList([RMSNorm(hidden_size, eps) for _ in range(4)])
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.rand(hidden_size, hidden_size)) for _ in range(3)]
        )

    def forward(self, x):
        z = torch.relu(x)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)

        z3 = torch.mm(y2, self.w[1])
        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)

        z4 = torch.mm(y3, self.w[2])
        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)
        return y4

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm.default]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("test_model", [TestAllReduceRMSNormModel])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["rocm"],
    reason="Only test on ROCm",
)
def test_rocm_aiter_allreduce_fusion_pass(
    test_model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op: bool,
):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                enable_rms_norm_custom_op,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(_worker_fn, num_processes)


def _worker_fn(
    local_rank: int,
    world_size: int,
    test_model_cls: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op: bool,
):
    set_random_seed(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12346",
        }
    )

    init_distributed_environment()

    custom_ops = []
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=custom_ops
        )
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.parallel_config.rank = local_rank

    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        from vllm.compilation.passes.fusion.rocm_aiter_allreduce_fusion import (
            RocmAiterAllReduceFusionPass,
        )

        allreduce_fusion_pass = RocmAiterAllReduceFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, allreduce_fusion_pass, func_pass, cleanup_pass)

        token_num = batch_size * seq_len
        model = test_model_cls(hidden_size, token_num)

        hidden_states = torch.randn((token_num, hidden_size), requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        results_unfused = model(hidden_states)
        results_fused = compiled_model(hidden_states)
        torch.testing.assert_close(results_unfused, results_fused, atol=1e-2, rtol=1e-2)

        assert allreduce_fusion_pass.matched_count == 4, (
            f"{allreduce_fusion_pass.matched_count=}"
        )
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        del allreduce_fusion_pass
