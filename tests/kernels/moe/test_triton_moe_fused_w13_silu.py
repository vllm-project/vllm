# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
)
from vllm.platforms import current_platform


def _interleave_w13(w13: torch.Tensor) -> torch.Tensor:
    gate, up = w13.chunk(2, dim=1)
    return torch.stack((gate, up), dim=2).flatten(1, 2)


def _run_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    fuse_w13_silu: bool,
    global_num_experts: int | None = None,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    num_local_experts, w13_size, _ = w13.shape
    if global_num_experts is None:
        global_num_experts = num_local_experts
    topk = topk_ids.shape[1]
    intermediate_size = w13_size // 2
    moe_config = make_dummy_moe_config(
        num_experts=global_num_experts,
        num_local_experts=num_local_experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        max_num_tokens=num_tokens,
    )
    moe_config.w13_swiglu_interleaved = fuse_w13_silu
    experts = TritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)
    workspace13_shape, workspace2_shape, _ = experts.workspace_shapes(
        num_tokens,
        w13_size,
        hidden_size,
        topk,
        global_num_experts,
        num_local_experts,
        None,
        MoEActivation.SILU,
    )
    workspace13 = torch.empty(
        workspace13_shape,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    workspace2 = torch.empty(
        workspace2_shape,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = torch.zeros_like(hidden_states)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return output


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 13, 128])
@torch.inference_mode()
def test_fused_w13_silu_matches_standalone_activation_bitwise(num_tokens: int):
    """The epilogue must preserve both CUDA fast math and its BF16 rounding."""
    torch.manual_seed(0)
    num_experts, hidden_size, intermediate_size, topk = 16, 256, 128, 4
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    w13 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / (hidden_size**0.5)
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / (intermediate_size**0.5)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device="cuda")[:topk] for _ in range(num_tokens)]
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", dtype=torch.float32), dim=-1
    )

    reference = _run_experts(hidden_states, w13, w2, topk_weights, topk_ids, False)
    actual = _run_experts(
        hidden_states, _interleave_w13(w13), w2, topk_weights, topk_ids, True
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@torch.inference_mode()
def test_fused_w13_silu_expert_parallel_zero_fill_bitwise():
    """Remote experts must zero the half-width fused output, not adjacent rows."""
    torch.manual_seed(1)
    num_tokens, hidden_size, intermediate_size, topk = 13, 256, 128, 4
    num_global_experts, num_local_experts = 8, 4
    hidden_states = torch.randn(
        num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / (hidden_size**0.5)
    w2 = torch.randn(
        num_local_experts,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / (intermediate_size**0.5)
    topk_ids = torch.randint(num_global_experts, (num_tokens, topk), device="cuda")
    topk_weights = torch.softmax(
        torch.randn(num_tokens, topk, device="cuda", dtype=torch.float32), dim=-1
    )
    expert_map = torch.tensor(
        [0, 1, 2, 3, -1, -1, -1, -1], device="cuda", dtype=torch.int32
    )

    reference = _run_experts(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        False,
        num_global_experts,
        expert_map,
    )
    actual = _run_experts(
        hidden_states,
        _interleave_w13(w13),
        w2,
        topk_weights,
        topk_ids,
        True,
        num_global_experts,
        expert_map,
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_unquantized_kernel_conversion_interleaves_eligible_weights():
    torch.manual_seed(0)
    w13 = torch.randn(4, 64, 32, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(4, 32, 32, device="cuda", dtype=torch.bfloat16)
    moe_config = make_dummy_moe_config(
        num_experts=4, hidden_dim=32, intermediate_size=32
    )

    converted, _ = convert_to_unquantized_kernel_format(
        UnquantizedMoeBackend.TRITON, moe_config, w13, w2
    )
    assert moe_config.w13_swiglu_interleaved
    torch.testing.assert_close(converted, _interleave_w13(w13), rtol=0, atol=0)

    moe_config.activation = MoEActivation.GELU
    converted, _ = convert_to_unquantized_kernel_format(
        UnquantizedMoeBackend.TRITON, moe_config, w13, w2
    )
    assert not moe_config.w13_swiglu_interleaved
    assert converted is w13
