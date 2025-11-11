# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10))
    + [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32],  #  torch.float16, torch.bfloat16 - aren't working correctly yet
)
@pytest.mark.parametrize(
    "params",
    [
        # (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        # (512, 16, 8, 16),
    ],
)
@pytest.mark.parametrize(
    "num_fused_shared_experts",
    [
        0,
        1,
    ],
)
def test_moe_fused_gate_combined(
    seq_length, dtype, params, num_fused_shared_experts, monkeypatch
):
    num_experts, num_expert_group, topk_group, topk = params
    topk += 1 if num_fused_shared_experts > 0 else 0

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts)).to(dtype).cuda()
    scores = tensor.clone()
    bias = torch.rand(num_experts).to(dtype).cuda()
    routed_scaling_factor = 2.5

    output, indices = ops.moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )

    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    ref_vllm_output, ref_vllm_indices = grouped_topk(
        hidden_states=scores,
        gating_output=scores,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=routed_scaling_factor,
    )

    if num_fused_shared_experts > 0:
        original_indices = indices.clone()
        original_ref_indices = ref_vllm_indices.clone()
        indices = indices[:, :-1]
        ref_vllm_indices = ref_vllm_indices[:, :-1]

        valid_min = num_experts
        valid_max = num_experts + num_fused_shared_experts
        shared_indices = original_indices[:, -1]
        shared_ref_indices = original_ref_indices[:, -1]
        if shared_indices is not None:
            assert torch.all(
                (shared_indices >= valid_min) & (shared_indices < valid_max)
            ), (
                "Shared expert indices out of range: ",
                f"found values outside [{valid_min}, {valid_max})",
            )
        if shared_ref_indices is not None:
            assert torch.all(
                (shared_ref_indices >= valid_min) & (shared_ref_indices < valid_max)
            ), (
                "Shared expert reference indices out of range: ",
                f"found values outside [{valid_min}, {valid_max})",
            )

    vllm_idx_check = torch.allclose(
        ref_vllm_indices.sort()[0].to(torch.int32),
        indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    vllm_output_check = torch.allclose(
        ref_vllm_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-04,
        atol=1e-03,
    )

    assert vllm_idx_check, (
        f"Indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )
    assert vllm_output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}, num_fused_shared_experts {num_fused_shared_experts}"
    )
