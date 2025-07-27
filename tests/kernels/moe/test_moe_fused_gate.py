# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (
    grouped_topk as vllm_compiled_grouped_topk)


def ref_non_compiled_vllm_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    if hidden_states.shape[0] == 0:
        # Happens when using DeepEP and number of tokens per dp rank < dp size.
        return torch.zeros((0, 0),
                           device=hidden_states.device,
                           dtype=torch.float32), torch.zeros(
                               (0, 0),
                               device=hidden_states.device,
                               dtype=torch.int32)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.shape[0]
    num_experts = scores.shape[1]

    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (scores.view(num_token, num_expert_group,
                                    -1).topk(2, dim=-1)[0].sum(dim=-1))
    else:
        group_scores = scores.view(num_token, num_expert_group,
                                   -1).max(dim=-1).values  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        scores.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(),
                                    float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores,
                                            k=topk,
                                            dim=-1,
                                            sorted=False)

    if num_fused_shared_experts > 0:
        assert routed_scaling_factor is not None
        # [NOTE] randint is used here to load-balance replicated
        # shared experts on each EP rank.
        topk_ids[:, -1] = torch.randint(low=num_experts,
                                        high=num_experts +
                                        num_fused_shared_experts,
                                        size=(topk_ids.size(0), ),
                                        dtype=topk_ids.dtype,
                                        device=topk_ids.device)

        # [NOTE] To keep the numerical correctness, we need to devide the
        # routed_scaling_factor here first for the shared experts,
        # so that later on in the experts accumulation, we can apply
        # routed_scaling_factor to all experts.
        topk_weights[:, -1] = (topk_weights[:, :-1].sum(dim=-1) /
                               routed_scaling_factor)

    if renormalize:
        topk_weights_sum = (topk_weights.sum(dim=-1, keepdim=True)
                            if num_fused_shared_experts == 0 else
                            topk_weights[:, :-1].sum(dim=-1, keepdim=True))
        topk_weights = topk_weights / topk_weights_sum

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10)) +
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize("dtype",
                         [torch.float16, torch.float32, torch.bfloat16])
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
    [0, 1, 8],
)
def test_moe_fused_gate_combined(seq_length, dtype, params,
                                 num_fused_shared_experts):
    num_experts, num_expert_group, topk_group, topk = params
    topk += 1 if num_fused_shared_experts > 0 else 0

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts)).to(dtype).cuda()
    scores = tensor.clone()
    bias = torch.rand(num_experts).to(dtype).cuda()

    output, indices = ops.moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=1.0,
    )

    ref_vllm_output, ref_vllm_indices = ref_non_compiled_vllm_grouped_topk(
        hidden_states=scores,
        gating_output=scores,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
        num_fused_shared_experts=num_fused_shared_experts,
        routed_scaling_factor=1.0,
    )

    if num_fused_shared_experts < 2:
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
    else:
        vllm_idx_check = torch.allclose(
            ref_vllm_indices[:, :-1].sort()[0].to(torch.int32),
            indices[:, :-1].sort()[0].to(torch.int32),
            rtol=1e-04,
            atol=1e-05,
        ) and indices[:, -1].ge(256).all() and indices[:, -1].lt(
            256 + num_fused_shared_experts).all()

        vllm_output_check = torch.allclose(
            ref_vllm_output[:, :-1].sort()[0].to(torch.float32),
            output[:, :-1].sort()[0].to(torch.float32),
            rtol=1e-04,
            atol=1e-03,
        )

    assert vllm_idx_check, (
        f"Indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}")
    assert vllm_output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}")


@pytest.mark.skip(
    reason="Compiled output expert indices are not matching when seq_len > 16")
@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 10)) +
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "params",
    [
        # (128, 4, 2, 4),
        (256, 8, 4, 8),  # deepseek v3
        # (512, 16, 8, 16),
    ],
)
def test_compiled_vllm_grouped_topk(seq_length, dtype, params):
    num_experts, num_expert_group, topk_group, topk = params

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts)).to(dtype).cuda()
    scores = tensor.clone()
    bias = torch.rand(num_experts).to(dtype).cuda()

    compiled_output, compiled_indices = vllm_compiled_grouped_topk(
        hidden_states=scores,
        gating_output=scores,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    ref_vllm_output, ref_vllm_indices = ref_non_compiled_vllm_grouped_topk(
        hidden_states=scores,
        gating_output=scores,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )

    idx_check = torch.allclose(
        ref_vllm_indices.sort()[0].to(torch.int32),
        compiled_indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    output_check = torch.allclose(
        ref_vllm_output.sort()[0].to(torch.float32),
        compiled_output.sort()[0].to(torch.float32),
        rtol=1e-04,
        atol=1e-05,
    )

    assert idx_check, (
        f"Indices mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}")
    assert output_check, (
        f"Output mismatch at seq_length {seq_length}, dtype {dtype}, "
        f"params {params}")
