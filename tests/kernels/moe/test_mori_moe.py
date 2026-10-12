# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-rank MoRI/AITER numerical counterpart to the DeepEP v2 MoE matrix."""

from dataclasses import replace

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    parallel_launch_with_config,
)
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.utils import torch_experts
from tests.utils import multi_gpu_test
from vllm.config import ParallelConfig, SchedulerConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_mori
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

MNKS = [
    (1, 256, 256),
    (2, 256, 512),
    (3, 1024, 2048),
    (32, 256, 1024),
    (45, 512, 2048),
    (64, 1024, 1024),
    (222, 1024, 2048),
]


def _mori_worker(pgi, vllm_config, cpu_group, m, n, k, fp8, graph):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import AiterExperts
    from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
        MoriPrepareAndFinalize,
    )

    rocm_aiter_ops.refresh_env_variables()
    init_workspace_manager(pgi.device)
    set_random_seed(47)
    dtype = current_platform.fp8_dtype() if fp8 else None
    (_, w1, s1, _), (_, w2, s2, _) = make_test_weights(
        32, n, k, quant_dtype=dtype, per_out_ch_quant=fp8
    )
    # Share global expert weights, but give each rank different token inputs.
    set_random_seed(101 + pgi.rank)
    x = torch.randn(m, k, device=pgi.device, dtype=torch.bfloat16) / 10
    routing = torch.randn(m, 32, device=pgi.device)
    # Guarantee a route to both ranks, including the one-token case.
    routing[:, 0] = routing[:, 16] = float("inf")
    ids = routing.topk(6, dim=-1).indices.to(torch.int32)
    weights = torch.randn(m, 6, device=pgi.device, dtype=torch.float32)
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        dp_size=2,
        dp_rank=pgi.rank,
        ep_size=2,
        ep_rank=pgi.rank,
        use_ep=True,
        all2all_backend="mori_high_throughput",
    )
    config = replace(
        make_dummy_moe_config(
            num_experts=32,
            num_local_experts=16,
            experts_per_token=6,
            hidden_dim=k,
            intermediate_size=n,
            max_num_tokens=256,
        ),
        moe_parallel_config=parallel,
    )
    start, end = pgi.rank * 16, (pgi.rank + 1) * 16
    quant = FusedMoEQuantConfig.make(
        dtype,
        per_act_token_quant=fp8,
        per_out_ch_quant=fp8,
        w1_scale=None if s1 is None else s1[start:end].contiguous(),
        w2_scale=None if s2 is None else s2[start:end].contiguous(),
    )
    prepare = maybe_make_prepare_finalize(config, quant)
    assert isinstance(prepare, MoriPrepareAndFinalize)
    assert prepare.use_fp8_dispatch == fp8
    assert prepare.num_dispatchers() == 2
    experts = AiterExperts(config, quant)
    assert not experts.expects_unquantized_inputs
    kernel = FusedMoEKernel(prepare, experts)
    local_w1, local_w2 = rocm_aiter_ops.shuffle_weights(
        w1[start:end].contiguous(), w2[start:end].contiguous()
    )
    local_w1.is_shuffled = local_w2.is_shuffled = True
    expert_mask = torch.zeros(32, dtype=torch.int32, device=pgi.device)
    expert_mask[start:end] = 1

    def reference(route_weights):
        return torch_experts(
            x,
            w1,
            w2,
            route_weights,
            ids,
            w1_scale=s1,
            w2_scale=s2,
            quant_dtype=dtype,
            per_act_token_quant=fp8,
        )

    def run():
        with set_forward_context(None, vllm_config):
            return kernel.apply(
                x,
                local_w1,
                local_w2,
                weights,
                ids,
                activation=MoEActivation.SILU,
                global_num_experts=32,
                expert_map=expert_mask,
                apply_router_weight_on_input=False,
            )

    def check(actual):
        expected = reference(weights)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        local_weights = weights * ((ids >= start) & (ids < end))
        missing_remote = reference(local_weights)
        norm = expected.float().norm()
        assert norm > 0
        norm_budget = 0.2 if fp8 else 0.06
        assert (actual.float() - expected.float()).norm() / norm < norm_budget
        # The norm check also rejects low-signal zero or rank-local results.
        for incorrect in (torch.zeros_like(expected), missing_remote):
            assert (incorrect.float() - expected.float()).norm() / norm > norm_budget
        if fp8:
            assert (
                torch.isclose(actual, expected, atol=0.2, rtol=0.2)
                .float()
                .mean()
                .item()
                > 0.99
            )
        else:
            torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.06)
        assert torch.isfinite(actual).all()

    actual = run()
    check(actual)
    if graph:
        saved = actual.clone()
        run()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            captured = run()
        for factor in (-0.5, 1.5):
            x.mul_(factor)
            weights.mul_(-0.75)
            capture.replay()
            torch.accelerator.synchronize()
            check(captured)
            torch.testing.assert_close(actual, saved, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not has_mori(), reason="Requires ROCm MoRI"
)
@pytest.mark.parametrize("m,n,k", MNKS)
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8_token_channel"])
@multi_gpu_test(num_gpus=2)
def test_mori_moe(m, n, k, fp8):
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="mori_high_throughput",
        ),
        scheduler_config=SchedulerConfig.default_factory(max_num_batched_tokens=256),
    )
    env = {
        "VLLM_ROCM_USE_AITER": "1",
        "VLLM_ROCM_USE_AITER_MOE": "1",
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0",
    }
    parallel_launch_with_config(2, _mori_worker, config, env, m, n, k, fp8, m == 32)
