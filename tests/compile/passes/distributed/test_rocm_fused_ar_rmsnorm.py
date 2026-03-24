# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU compiler-pass integration test for fused AR+RMSNorm on ROCm.

Tests that RocmAiterRMSNormQuantFusionPass correctly handles both paths:

FP4/BF16 path (no FP8 quant consumers):
  - fused_allreduce_rmsnorm is preserved as-is in the compiled graph.
  - At runtime, AITER's fused AR+RMSNorm kernel handles the operation.

FP8 path (FP8 quant consumer follows the normed output):
  - fused_allreduce_rmsnorm is decomposed into all_reduce + rmsnorm_with_add.
  - Then rmsnorm_with_add + fp8_quant are fused into a single AITER op.
"""

import pytest
import torch

from vllm._aiter_ops import IS_AITER_FOUND
from vllm.platforms import current_platform

from tests.utils import multi_gpu_test

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not IS_AITER_FOUND,
    reason="ROCm with AITER required",
)


class TestFusedARRMSNormNoQuantModel(torch.nn.Module):
    """Model with fused_allreduce_rmsnorm but NO FP8 quant consumers.

    Simulates the FP4/BF16 path: the decomposition pass should NOT
    decompose these nodes, preserving the AITER fused AR+RMSNorm kernel.
    """

    def __init__(self, hidden_size=64, eps=1e-5):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.norm = torch.nn.ModuleList([
            RMSNorm(hidden_size, eps, fused_allreduce=False),
            RMSNorm(hidden_size, eps, fused_allreduce=True),
            RMSNorm(hidden_size, eps, fused_allreduce=True),
        ])
        self.w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(hidden_size, hidden_size))
            for _ in range(2)
        ])

    def forward(self, x):
        z = torch.relu(x)
        resid = z
        y = self.norm[0](z)

        z2 = torch.mm(y, self.w[0])
        y2, resid = self.norm[1](z2, resid)

        z3 = torch.mm(y2, self.w[1])
        y3, resid = self.norm[2](z3, resid)
        return y3

    def ops_in_model_before(self):
        return [torch.ops.vllm.fused_allreduce_rmsnorm.default]


class TestFusedARRMSNormFP8Model(torch.nn.Module):
    """Model with fused_allreduce_rmsnorm AND FP8 per-token quant consumer.

    Simulates the FP8 path: the decomposition pass should decompose
    fused_allreduce_rmsnorm into all_reduce + rmsnorm_with_add, then the
    pattern matcher fuses rmsnorm_with_add + fp8_quant into one AITER op.
    """

    def __init__(self, hidden_size=64, eps=1e-5):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm

        self.norm = torch.nn.ModuleList([
            RMSNorm(hidden_size, eps, fused_allreduce=False),
            RMSNorm(hidden_size, eps, fused_allreduce=True),
        ])
        self.w = torch.nn.Parameter(torch.rand(hidden_size, hidden_size))

    def forward(self, x):
        z = torch.relu(x)
        resid = z
        y = self.norm[0](z)

        z2 = torch.mm(y, self.w)
        y2, resid = self.norm[1](z2, resid)

        quant_out, scale = torch.ops.vllm.rocm_aiter_per_token_quant(
            y2, torch.float8_e4m3fnuz,
        )
        return quant_out.to(x.dtype) * scale

    def ops_in_model_before(self):
        return [torch.ops.vllm.fused_allreduce_rmsnorm.default]


def _run_rocm_fused_ar_test(
    local_rank: int,
    world_size: int,
    test_model_cls: type,
    hidden_size: int,
    dtype: torch.dtype,
    expect_decomposition: bool,
):
    """Worker process for the multi-GPU test."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
        RocmAiterRMSNormQuantFusionPass,
    )
    from vllm.compilation.passes.utility.noop_elimination import (
        NoOpEliminationPass,
    )
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
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.system_utils import update_environment_variables
    from vllm.utils.torch_utils import set_random_seed

    from tests.compile.backend import TestBackend

    set_random_seed(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12346",
            "VLLM_ROCM_USE_AITER": "1",
        }
    )

    rocm_aiter_ops.refresh_env_variables()
    init_distributed_environment()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["+rms_norm"],
        )
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_norm_quant=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.parallel_config.rank = local_rank
    vllm_config.model_config = ModelConfig(dtype=dtype)

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        fusion_pass = RocmAiterRMSNormQuantFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(noop_pass, fusion_pass, cleanup_pass)

        model = test_model_cls(hidden_size)
        token_num = 64
        hidden_states = torch.randn(
            (token_num, hidden_size), requires_grad=False
        )

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        results_unfused = model(hidden_states)
        results_fused = compiled_model(hidden_states)
        torch.testing.assert_close(
            results_unfused, results_fused, atol=1e-2, rtol=1e-2
        )

        fused_ar_op = torch.ops.vllm.fused_allreduce_rmsnorm.default
        from vllm.compilation.passes.fx_utils import find_op_nodes

        fused_count_before = len(
            list(find_op_nodes(fused_ar_op, backend.graph_pre_pass))
        )
        fused_count_after = len(
            list(find_op_nodes(fused_ar_op, backend.graph_post_pass))
        )

        if expect_decomposition:
            assert fused_count_before > 0, (
                "Expected fused_allreduce_rmsnorm in pre-pass graph"
            )
            assert fused_count_after < fused_count_before, (
                "Expected decomposition to remove some "
                "fused_allreduce_rmsnorm nodes"
            )
        else:
            assert fused_count_after == fused_count_before, (
                "fused_allreduce_rmsnorm should be preserved "
                "when no FP8 consumer"
            )

        del fusion_pass


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_rocm_fused_ar_rmsnorm_no_quant_preserved(
    hidden_size: int,
    dtype: torch.dtype,
):
    """FP4/BF16 path: fused_allreduce_rmsnorm preserved when no FP8 quant."""
    num_processes = 2
    torch.multiprocessing.spawn(
        _run_rocm_fused_ar_test,
        args=(
            num_processes,
            TestFusedARRMSNormNoQuantModel,
            hidden_size,
            dtype,
            False,
        ),
        nprocs=num_processes,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_rocm_fused_ar_rmsnorm_fp8_decomposed(
    hidden_size: int,
    dtype: torch.dtype,
):
    """FP8 path: fused_allreduce_rmsnorm decomposed when FP8 quant follows."""
    num_processes = 2
    torch.multiprocessing.spawn(
        _run_rocm_fused_ar_test,
        args=(
            num_processes,
            TestFusedARRMSNormFP8Model,
            hidden_size,
            dtype,
            True,
        ),
        nprocs=num_processes,
    )
