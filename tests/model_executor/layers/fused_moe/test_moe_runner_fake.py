# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner


def test_moe_forward_fake_uses_trtllm_mxfp4_unpadded_dim() -> None:
    hidden_states = torch.empty(2, 3072, device="meta")

    out = moe_runner._moe_forward_fake(
        hidden_states,
        torch.empty(2, 128, device="meta"),
        None,
        None,
        "model.layers.0.mlp.experts",
        2880,
    )

    assert out.shape == (2, 2880)


def test_moe_forward_fake_keeps_default_hidden_dim() -> None:
    hidden_states = torch.empty(2, 3072, device="meta")

    out = moe_runner._moe_forward_fake(
        hidden_states,
        torch.empty(2, 128, device="meta"),
        None,
        None,
        "model.layers.0.mlp.experts",
        0,
    )

    assert out.shape == hidden_states.shape


def test_moe_forward_shared_fake_uses_trtllm_mxfp4_unpadded_dim() -> None:
    hidden_states = torch.empty(2, 3072, device="meta")
    shared_input = torch.empty(2, 3072, device="meta")

    shared_out, fused_out = moe_runner._moe_forward_shared_fake(
        hidden_states,
        torch.empty(2, 128, device="meta"),
        shared_input,
        None,
        "model.layers.0.mlp.experts",
        2880,
    )

    assert shared_out.shape == shared_input.shape
    assert fused_out.shape == (2, 2880)
