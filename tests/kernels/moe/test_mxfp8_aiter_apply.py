# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MXFP8 expert application and expert-map behavior."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (  # noqa: E402
    AiterMxfp8Experts,
)
from vllm.model_executor.layers.fused_moe.routed_experts import (  # noqa: E402
    RoutedExperts,
)
from vllm.platforms.rocm import on_gfx950  # noqa: E402


def _capture_expert_mask(
    expert_mask: torch.Tensor,
    *,
    global_num_experts: int,
) -> torch.Tensor:
    """Capture the expert mask forwarded by the real AITER MXFP8 apply path."""
    experts = object.__new__(AiterMxfp8Experts)
    experts.quant_config = SimpleNamespace(gemm1_clamp_limit=None)
    experts.w1_scale_val = None
    experts.w2_scale_val = None
    experts.shared_w1 = None
    experts.shared_w2 = None
    experts.shared_w1_scale = None
    experts.shared_w2_scale = None
    experts.shared_expert_id = -1

    captured: dict[str, torch.Tensor] = {}

    def _fake_fused_moe(hidden_states, w1, w2, tw, ti, *, expert_mask, **kwargs):
        captured["expert_mask"] = expert_mask
        assert kwargs["gate_mode"] == "interleave"
        return torch.zeros_like(hidden_states)

    w1 = torch.zeros(1)
    w2 = torch.zeros(1)
    output = torch.zeros(4, 8, dtype=torch.bfloat16)
    hidden_states = torch.zeros(4, 8, dtype=torch.bfloat16)
    topk_weights = torch.ones(4, 2)
    topk_ids = torch.zeros(4, 2, dtype=torch.int32)

    with patch.object(rocm_aiter_ops, "fused_moe", side_effect=_fake_fused_moe):
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=global_num_experts,
            expert_map=expert_mask,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
    return captured["expert_mask"]


def test_apply_forwards_expert_mask_and_interleaved_layout():
    """AITER receives the precomputed mask and declared preshuffle layout."""
    expert_mask = torch.tensor(
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        dtype=torch.int32,
    )

    result = _capture_expert_mask(expert_mask, global_num_experts=8)

    assert torch.equal(result.to(torch.int32), expert_mask.to(torch.int32))
    assert result.sum().item() == 4


def test_apply_forwards_unclamped_shared_expert():
    experts = object.__new__(AiterMxfp8Experts)
    experts.quant_config = SimpleNamespace(gemm1_clamp_limit=10.0)
    experts.w1_scale_val = None
    experts.w2_scale_val = None
    experts.shared_w1 = torch.empty(1)
    experts.shared_w2 = torch.empty(1)
    experts.shared_w1_scale = torch.empty(1)
    experts.shared_w2_scale = torch.empty(1)
    experts.shared_expert_id = 8
    captured = {}

    def fake_fused_moe(hidden_states, *_args, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(hidden_states)

    hidden_states = torch.zeros(2, 8, dtype=torch.bfloat16)
    with patch.object(rocm_aiter_ops, "fused_moe", side_effect=fake_fused_moe):
        experts.apply(
            output=torch.empty_like(hidden_states),
            hidden_states=hidden_states,
            w1=torch.empty(1),
            w2=torch.empty(1),
            topk_weights=torch.ones(2, 2),
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            activation=MoEActivation.SILU,
            global_num_experts=8,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    assert captured["shared_expert_id"] == 8
    assert captured["clamp_shared"] is False
    assert captured["swiglu_limit"] == 10.0


def test_custom_op_forwards_clamp_shared_to_aiter(monkeypatch):
    import importlib

    import vllm._aiter_ops as aiter_ops

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    captured = {}

    def fake_fused_moe(*_args, **kwargs):
        captured.update(kwargs)
        return torch.empty((2, 8), dtype=torch.bfloat16)

    monkeypatch.setattr(fused_moe_module, "fused_moe", fake_fused_moe)
    hidden_states = torch.empty((2, 8), dtype=torch.bfloat16)
    shared = torch.empty(1)
    aiter_ops._rocm_aiter_fused_moe_impl(
        hidden_states,
        torch.empty(1),
        torch.empty(1),
        torch.ones((2, 2)),
        torch.zeros((2, 2), dtype=torch.int32),
        shared_w1=shared,
        shared_w2=shared,
        shared_w1_scale=shared,
        shared_w2_scale=shared,
        shared_expert_id=8,
        clamp_shared=False,
    )

    assert captured["clamp_shared"] is False


def test_custom_op_omits_default_clamp_for_legacy_aiter(monkeypatch):
    import importlib

    import vllm._aiter_ops as aiter_ops

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    captured = {}

    def legacy_fused_moe(*_args, **kwargs):
        captured.update(kwargs)
        return torch.empty((2, 8), dtype=torch.bfloat16)

    monkeypatch.setattr(fused_moe_module, "fused_moe", legacy_fused_moe)
    hidden_states = torch.empty((2, 8), dtype=torch.bfloat16)
    shared = torch.empty(1)
    aiter_ops._rocm_aiter_fused_moe_impl(
        hidden_states,
        torch.empty(1),
        torch.empty(1),
        torch.ones((2, 2)),
        torch.zeros((2, 2), dtype=torch.int32),
        shared_w1=shared,
        shared_w2=shared,
        shared_w1_scale=shared,
        shared_w2_scale=shared,
        shared_expert_id=8,
    )

    assert "clamp_shared" not in captured


def test_custom_op_rejects_unclamped_dispatch_without_shared_expert():
    import vllm._aiter_ops as aiter_ops

    with pytest.raises(
        ValueError,
        match="clamp_shared=False requires FHMoE shared-expert dispatch",
    ):
        aiter_ops._rocm_aiter_fused_moe_impl(
            torch.empty((2, 8), dtype=torch.bfloat16),
            torch.empty(1),
            torch.empty(1),
            torch.ones((2, 2)),
            torch.zeros((2, 2), dtype=torch.int32),
            clamp_shared=False,
        )


def test_hy4_capability_probe_forwards_latest_exact_contract(monkeypatch):
    import aiter.fhmoe as fhmoe
    from aiter.ops.flydsl.moe_common import GateMode

    captured: dict[str, object] = {}

    def supports_mxfp8_fhmoe_config(
        max_tokens,
        *,
        model_dim,
        inter_dim,
        experts,
        topk,
        dtype,
        hidden_pad,
        intermediate_pad,
        gate_mode,
        doweight_stage1,
    ):
        captured.update(
            max_tokens=max_tokens,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            dtype=dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            gate_mode=gate_mode,
            doweight_stage1=doweight_stage1,
        )
        return True

    monkeypatch.setattr(
        fhmoe,
        "supports_mxfp8_fhmoe_config",
        supports_mxfp8_fhmoe_config,
    )

    assert rocm_aiter_ops._probe_hy4_mxfp8_fhmoe_capability(32768)
    assert captured == {
        "max_tokens": 32768,
        "model_dim": 6144,
        "inter_dim": 256,
        "experts": 257,
        "topk": 9,
        "dtype": torch.bfloat16,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "gate_mode": GateMode.INTERLEAVE,
        "doweight_stage1": False,
    }


def test_hy4_capability_probe_requires_clamp_shared(monkeypatch):
    import importlib

    fused_moe_module = importlib.import_module("aiter.fused_moe")

    def incompatible_fused_moe(
        *,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        shared_expert_id,
    ):
        pass

    monkeypatch.setattr(fused_moe_module, "fused_moe", incompatible_fused_moe)
    assert not rocm_aiter_ops._probe_hy4_mxfp8_fhmoe_capability(32)


def test_hy4_capability_probe_requires_generic_shuffle_api(monkeypatch):
    import aiter.ops.shuffle as shuffle

    monkeypatch.setattr(shuffle, "moe_shuffle_weight", lambda tensor: tensor)
    assert not rocm_aiter_ops._probe_hy4_mxfp8_fhmoe_capability(32)


def test_routed_experts_expert_map_delegates_to_kernel():
    """Only kernels that consume the expert mask receive its 0/1 layout."""
    canonical = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
    mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int32)

    def resolve(consumes_mask, *, has_moe_kernel=True):
        moe_kernel = (
            SimpleNamespace(
                fused_experts=SimpleNamespace(consumes_expert_mask=consumes_mask)
            )
            if has_moe_kernel
            else None
        )
        layer = SimpleNamespace(
            _expert_map=canonical,
            expert_mask=mask,
            quant_method=SimpleNamespace(moe_kernel=moe_kernel),
        )
        return RoutedExperts.expert_map.fget(layer)

    assert torch.equal(resolve(True), mask)
    assert torch.equal(resolve(False), canonical)
    assert torch.equal(resolve(False, has_moe_kernel=False), canonical)


@pytest.mark.skipif(not on_gfx950(), reason="legacy layout is gfx950-specific")
def test_moe_shuffle_wrappers_match_legacy_layout():
    """Generic MoE wrappers preserve the existing gfx950 preshuffle layout."""

    def fp8(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )

    w1 = fp8((1, 64, 128))
    w2 = fp8((1, 128, 256))
    s1 = torch.ones((64, 4), device="cuda", dtype=torch.float8_e8m0fnu)
    s2 = torch.ones((128, 8), device="cuda", dtype=torch.float8_e8m0fnu)

    pairs = (
        (
            rocm_aiter_ops.shuffle_weight_a16w4(w1, 16, True),
            rocm_aiter_ops.moe_shuffle_weight(
                w1,
                experts_cnt=1,
                is_guinterleave=True,
                gate_up=True,
            ),
        ),
        (
            rocm_aiter_ops.shuffle_weight_a16w4(w2, 16, False),
            rocm_aiter_ops.moe_shuffle_weight(
                w2,
                experts_cnt=1,
                is_guinterleave=True,
                gate_up=False,
            ),
        ),
        (
            rocm_aiter_ops.shuffle_scale_a16w4(s1, 1, True),
            rocm_aiter_ops.moe_shuffle_scale(
                s1,
                experts_cnt=1,
                is_guinterleave=True,
                gate_up=True,
            ),
        ),
        (
            rocm_aiter_ops.shuffle_scale_a16w4(s2, 1, False),
            rocm_aiter_ops.moe_shuffle_scale(
                s2,
                experts_cnt=1,
                is_guinterleave=True,
                gate_up=False,
            ),
        ),
    )

    assert all(torch.equal(legacy, generic) for legacy, generic in pairs)
