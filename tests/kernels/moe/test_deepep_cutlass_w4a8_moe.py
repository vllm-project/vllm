# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration coverage for DeepEP LL with CUTLASS W4A8 experts."""

import dataclasses

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.test_cutlass_w4a8_moe import (
    GROUP_SIZE,
    make_batched_pipeline_weight,
)
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassBatchedExpertsW4A8Fp8,
    run_cutlass_moe_w4a8_fp8,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.oracle.w4a8 import (
    make_w4a8_moe_quant_config,
)
from vllm.utils.import_utils import has_deep_ep
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test
from .parallel_utils import (
    DeepEPLLArgs,
    ProcessGroupInfo,
    make_deepep_a2a,
    parallel_launch,
)

requires_deep_ep = pytest.mark.skipif(
    not has_deep_ep(),
    reason="Requires DeepEP kernels",
)

WORLD_SIZE = 8
TOKENS_PER_RANK = 4
MAX_TOKENS_PER_RANK = 16
NUM_EXPERTS = 8
TOPK = 2
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 1024


def _make_strides(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a_strides1 = torch.full(
        (num_experts,),
        hidden_size,
        dtype=torch.int64,
        device=device,
    )
    a_strides2 = torch.full(
        (num_experts,),
        intermediate_size,
        dtype=torch.int64,
        device=device,
    )
    c_strides1 = torch.full(
        (num_experts,),
        intermediate_size * 2,
        dtype=torch.int64,
        device=device,
    )
    c_strides2 = a_strides1
    return a_strides1, a_strides2, c_strides1, c_strides2


def _make_scale_strides(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_strides1 = torch.zeros((num_experts, 2), dtype=torch.int64, device=device)
    s_strides1[:, 0] = intermediate_size * 2
    s_strides2 = torch.zeros_like(s_strides1)
    s_strides2[:, 0] = hidden_size
    return s_strides1, s_strides2


def _run_flat_reference(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_chan_scale: torch.Tensor,
    w2_chan_scale: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    a_strides1, a_strides2, c_strides1, c_strides2 = _make_strides(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device,
    )
    s_strides1, s_strides2 = _make_scale_strides(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device,
    )
    a1q, a1q_scale = ops.scaled_fp8_quant(
        hidden_states,
        use_per_token_if_dynamic=True,
    )
    workspace13 = torch.empty(
        (hidden_states.shape[0] * TOPK, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    workspace2 = torch.empty_like(workspace13)
    output = torch.empty_like(hidden_states)

    run_cutlass_moe_w4a8_fp8(
        output=output,
        hidden_states=a1q,
        w1=w1,
        w2=w2,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=NUM_EXPERTS,
        expert_map=None,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1q_scale=a1q_scale,
        a2_scale=None,
        w1_chan_scale=w1_chan_scale,
        w2_chan_scale=w2_chan_scale,
        a_strides1=a_strides1,
        a_strides2=a_strides2,
        b_strides1=b_strides1,
        b_strides2=b_strides2,
        c_strides1=c_strides1,
        c_strides2=c_strides2,
        s_strides1=s_strides1,
        s_strides2=s_strides2,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_num_tokens=None,
        out_dtype=torch.bfloat16,
        per_act_token=True,
        per_out_ch=True,
        use_batched_format=False,
        topk_weights=topk_weights,
        group_size=GROUP_SIZE,
        permute_scratch=None,
    )
    return output


def _run_deepep_cutlass_w4a8(
    pgi: ProcessGroupInfo,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_chan_scale: torch.Tensor,
    w2_chan_scale: torch.Tensor,
    b_strides1: torch.Tensor,
    b_strides2: torch.Tensor,
) -> torch.Tensor:
    pg = torch.distributed.new_group(list(range(pgi.world_size)))
    prepare_finalize = make_deepep_a2a(
        pg=pg,
        pgi=pgi,
        dp_size=1,
        q_dtype=torch.float8_e4m3fn,
        block_shape=None,
        deepep_ht_args=None,
        deepep_ll_args=DeepEPLLArgs(
            max_tokens_per_rank=MAX_TOKENS_PER_RANK,
            hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            use_fp8_dispatch=False,
        ),
    )

    expert_start = pgi.rank
    expert_end = expert_start + 1
    # Materialize rank-local storage as it would be loaded in an EP worker.
    # Packed group-scale expert views are not sufficiently aligned for CUTLASS.
    local_w1 = w1[expert_start:expert_end].clone()
    local_w2 = w2[expert_start:expert_end].clone()
    local_w1_scale = (
        w1_scale.view(NUM_EXPERTS, -1)[expert_start:expert_end].clone().view(-1)
    )
    local_w2_scale = (
        w2_scale.view(NUM_EXPERTS, -1)[expert_start:expert_end].clone().view(-1)
    )
    local_w1_chan_scale = w1_chan_scale[expert_start:expert_end].clone()
    local_w2_chan_scale = w2_chan_scale[expert_start:expert_end].clone()
    local_b_strides1 = b_strides1[expert_start:expert_end].clone()
    local_b_strides2 = b_strides2[expert_start:expert_end].clone()
    moe_config = make_dummy_moe_config(
        num_experts=NUM_EXPERTS,
        num_local_experts=1,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        max_num_tokens=MAX_TOKENS_PER_RANK,
    )
    moe_config.moe_parallel_config = dataclasses.replace(
        moe_config.moe_parallel_config,
        dp_size=WORLD_SIZE,
        ep_size=WORLD_SIZE,
        dp_rank=pgi.rank,
        ep_rank=pgi.rank,
        use_ep=True,
        all2all_backend="deepep_low_latency",
    )
    quant_config = make_w4a8_moe_quant_config(
        w1_scale=local_w1_scale,
        w2_scale=local_w2_scale,
        g1_alphas=local_w1_chan_scale,
        g2_alphas=local_w2_chan_scale,
    )
    experts = CutlassBatchedExpertsW4A8Fp8(
        moe_config=moe_config,
        quant_config=quant_config,
        b_strides1=local_b_strides1,
        b_strides2=local_b_strides2,
        group_size=GROUP_SIZE,
        max_num_tokens=MAX_TOKENS_PER_RANK,
        num_dispatchers=WORLD_SIZE,
    )
    kernel = FusedMoEKernel(
        prepare_finalize=prepare_finalize,
        fused_experts=experts,
    )
    prepare_finalize.buffer.clean_low_latency_buffer(
        MAX_TOKENS_PER_RANK,
        HIDDEN_SIZE,
        NUM_EXPERTS,
    )
    expert_map = torch.full(
        (NUM_EXPERTS,),
        -1,
        dtype=torch.int32,
        device=pgi.device,
    )
    expert_map[pgi.rank] = 0
    return kernel.apply(
        hidden_states=hidden_states,
        w1=local_w1,
        w2=local_w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=NUM_EXPERTS,
        expert_map=expert_map,
        apply_router_weight_on_input=False,
    )


def _test_deepep_cutlass_w4a8_worker(pgi: ProcessGroupInfo) -> None:
    set_random_seed(7)
    init_workspace_manager(pgi.device)
    hidden_states = torch.randn(
        (TOKENS_PER_RANK, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=pgi.device,
    )
    topk_ids = torch.randint(
        0,
        NUM_EXPERTS,
        (TOKENS_PER_RANK, TOPK),
        dtype=torch.int64,
        device=pgi.device,
    )
    topk_weights = torch.softmax(
        torch.randn(
            (TOKENS_PER_RANK, TOPK),
            dtype=torch.float32,
            device=pgi.device,
        ),
        dim=-1,
    )
    w1, w1_scale, w1_chan_scale, b_strides1 = make_batched_pipeline_weight(
        NUM_EXPERTS,
        INTERMEDIATE_SIZE * 2,
        HIDDEN_SIZE,
    )
    w2, w2_scale, w2_chan_scale, b_strides2 = make_batched_pipeline_weight(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
    )

    with set_current_vllm_config(VllmConfig()):
        expected = _run_flat_reference(
            hidden_states,
            topk_weights,
            topk_ids,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_chan_scale,
            w2_chan_scale,
            b_strides1,
            b_strides2,
        )
        actual = _run_deepep_cutlass_w4a8(
            pgi,
            hidden_states,
            topk_weights,
            topk_ids,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_chan_scale,
            w2_chan_scale,
            b_strides1,
            b_strides2,
        )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=5e-2)


@multi_gpu_test(num_gpus=WORLD_SIZE)
@requires_deep_ep
def test_deepep_ll_cutlass_w4a8_integration(workspace_init):
    parallel_launch(WORLD_SIZE, _test_deepep_cutlass_w4a8_worker)
