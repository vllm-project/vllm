# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import ocp_mx_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    rocm_aiter_fused_experts,
)


def test_minimax_aiter_call_uses_bf16_activations(monkeypatch):
    captured = {}

    def fused_moe(hidden_states, *args, q_dtype_a=None, **kwargs):
        captured["input_dtype"] = hidden_states.dtype
        captured["q_dtype_a"] = q_dtype_a
        return hidden_states

    monkeypatch.setattr(rocm_aiter_ops, "fused_moe", fused_moe)
    scales = torch.ones((1, 1), dtype=torch.uint8)
    quant_config = ocp_mx_moe_quant_config(
        quant_dtype="mxfp4",
        w1_scale=scales,
        w2_scale=scales,
    )
    moe_config = SimpleNamespace(
        hidden_dim_unpadded=32,
        intermediate_size_per_partition=16,
        intermediate_size_per_partition_unpadded=16,
        intermediate_pad=None,
        tp_size=4,
        use_mxfp4_w4a4_dsv4=False,
        swiglu_limit=7.0,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    hidden_states = torch.randn((1, 32), dtype=torch.bfloat16)

    rocm_aiter_fused_experts(
        hidden_states=hidden_states,
        w1=torch.empty((1, 32, 16), dtype=torch.float4_e2m1fn_x2),
        w2=torch.empty((1, 32, 8), dtype=torch.float4_e2m1fn_x2),
        topk_weights=torch.ones((1, 1)),
        topk_ids=torch.zeros((1, 1), dtype=torch.int32),
        moe_config=moe_config,
        quant_config=quant_config,
        activation=MoEActivation.SWIGLUOAI,
    )

    assert captured == {
        "input_dtype": torch.bfloat16,
        "q_dtype_a": None,
    }
