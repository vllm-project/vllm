# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.runner import moe_runner


class FakeTrtLlmMxfp4Experts:
    __module__ = "vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe"


class FakeOtherExperts:
    pass


def test_moe_forward_fake_uses_trtllm_mxfp4_unpadded_dim(
    monkeypatch,
) -> None:
    experts = FakeTrtLlmMxfp4Experts()
    experts.hidden_dim_unpadded = 2880
    layer = SimpleNamespace(
        runner=SimpleNamespace(
            _quant_method=SimpleNamespace(
                moe_kernel=SimpleNamespace(fused_experts=experts)
            )
        )
    )
    monkeypatch.setattr(moe_runner, "get_layer_from_name", lambda _: layer)

    hidden_states = torch.empty(2, 3072, device="meta")
    out = moe_runner._moe_forward_fake(
        hidden_states,
        torch.empty(2, 128, device="meta"),
        None,
        None,
        "model.layers.0.mlp.experts",
    )

    assert out.shape == (2, 2880)


def test_moe_forward_fake_keeps_default_hidden_dim(monkeypatch) -> None:
    layer = SimpleNamespace(
        runner=SimpleNamespace(
            _quant_method=SimpleNamespace(
                moe_kernel=SimpleNamespace(fused_experts=FakeOtherExperts())
            )
        )
    )
    monkeypatch.setattr(moe_runner, "get_layer_from_name", lambda _: layer)

    hidden_states = torch.empty(2, 3072, device="meta")
    out = moe_runner._moe_forward_fake(
        hidden_states,
        torch.empty(2, 128, device="meta"),
        None,
        None,
        "model.layers.0.mlp.experts",
    )

    assert out.shape == hidden_states.shape


def test_moe_forward_shared_fake_uses_trtllm_mxfp4_unpadded_dim(
    monkeypatch,
) -> None:
    experts = FakeTrtLlmMxfp4Experts()
    experts.hidden_dim_unpadded = 2880
    layer = SimpleNamespace(
        runner=SimpleNamespace(
            _quant_method=SimpleNamespace(
                moe_kernel=SimpleNamespace(fused_experts=experts)
            )
        )
    )
    monkeypatch.setattr(moe_runner, "get_layer_from_name", lambda _: layer)

    hidden_states = torch.empty(2, 3072, device="meta")
    shared_input = torch.empty(2, 3072, device="meta")
    shared_out, fused_out = moe_runner._moe_forward_shared_fake(
        hidden_states,
        torch.empty(2, 128, device="meta"),
        shared_input,
        None,
        "model.layers.0.mlp.experts",
    )

    assert shared_out.shape == shared_input.shape
    assert fused_out.shape == (2, 2880)
