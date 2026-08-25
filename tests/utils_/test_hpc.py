# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.hpc_bf16_moe import HPCBf16Experts
from vllm.utils.hpc import hpc_fuse_moe_bf16


def test_hpc_fuse_moe_bf16_uses_keyword_control_arguments(monkeypatch):
    tensors = [torch.empty(0) for _ in range(8)]
    (
        x,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_scale,
        shared_output,
        output,
        workspace,
    ) = tensors
    result = torch.empty(0)
    captured_args = None
    captured_kwargs = None

    def fuse_moe_bf16(*args, **kwargs):
        nonlocal captured_args, captured_kwargs
        captured_args = args
        captured_kwargs = kwargs
        return result

    monkeypatch.setitem(
        sys.modules, "hpc", SimpleNamespace(fuse_moe_bf16=fuse_moe_bf16)
    )

    actual = hpc_fuse_moe_bf16(
        x,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_scale,
        rank_ep=3,
        num_expert_total=256,
        shared_output=shared_output,
        output=output,
        workspace=workspace,
    )

    assert actual is result
    assert captured_args == (
        x,
        gate_up_weight,
        down_weight,
        topk_ids,
        topk_scale,
    )
    assert captured_kwargs == {
        "rank_ep": 3,
        "num_expert_total": 256,
        "shared_output": shared_output,
        "output": output,
        "workspace": workspace,
    }


def test_hpc_bf16_experts_passes_vllm_owned_workspace(monkeypatch):
    captured = {}

    def fake_hpc_fuse_moe_bf16(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.hpc_bf16_moe.hpc_fuse_moe_bf16",
        fake_hpc_fuse_moe_bf16,
    )

    experts = object.__new__(HPCBf16Experts)
    experts.ep_rank = 0
    workspace = torch.empty(1024, dtype=torch.bfloat16)
    output = torch.empty(2, 8, dtype=torch.bfloat16)
    hidden_states = torch.empty_like(output)
    w1 = torch.empty(4, 16, 8, dtype=torch.bfloat16)
    w2 = torch.empty(4, 8, 8, dtype=torch.bfloat16)
    topk_ids = torch.zeros(2, 2, dtype=torch.int32)
    topk_weights = torch.zeros(2, 2, dtype=torch.float32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=workspace,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert captured["output"] is output
    assert captured["workspace"].dtype == torch.uint8
    assert captured["workspace"].data_ptr() == workspace.data_ptr()


def test_hpc_bf16_workspace_shape_is_conservative():
    experts = object.__new__(HPCBf16Experts)
    workspace13, workspace2, output = experts.workspace_shapes(
        M=16,
        N=384,
        K=2048,
        topk=8,
        global_num_experts=256,
        local_num_experts=32,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )

    assert workspace13 == (16, 2048)
    assert output == (16, 2048)
    assert workspace2[0] * 2 >= 16 * 8 * (4 * 2048 + 3 * 384 + 64)
