# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmSm100Fp8MegaMoEBackend,
    requant_block_fp8_to_ue8m0,
)
from vllm.models.glm5next.nvidia.mega_moe import Glm5NextMegaMoEExperts

BLOCK = (128, 128)


def _block_fp8(real: torch.Tensor, power_of_two: bool = False):
    """Quantize ``(..., mn, k)`` to E4M3 with one float32 scale per 128x128."""
    *lead, mn, k = real.shape
    blocks = real.reshape(*lead, mn // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(-3, -1)) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    full = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return (real / full).to(torch.float8_e4m3fn), scale


def _dequant_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * scale.repeat_interleave(128, -2).repeat_interleave(128, -1)


def test_requant_gives_power_of_two_scale_per_32_columns():
    torch.manual_seed(0)
    real = torch.randn(3, 256, 384) * 0.02
    weight, scale = _block_fp8(real)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale, chunk=2)

    assert requant.dtype == torch.float8_e4m3fn
    assert requant.shape == weight.shape
    assert sf.shape == (3, 256, 384 // 32)
    assert torch.equal(torch.log2(sf), torch.round(torch.log2(sf)))
    assert requant.float().abs().max() <= 448.0
    original = _dequant_block(weight, scale)
    restored = requant.float() * sf.repeat_interleave(32, -1)
    # One more E4M3 rounding of every element: a few percent, never garbage.
    rel = ((restored - original).norm() / original.norm()).item()
    assert 0 < rel < 0.05


def test_requant_is_exact_for_power_of_two_block_scales():
    torch.manual_seed(1)
    real = torch.randn(256, 256) * 0.05
    weight, scale = _block_fp8(real, power_of_two=True)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale)

    # A finer power-of-two scale only shifts exponents, so nothing rounds.
    assert torch.equal(
        requant.float() * sf.repeat_interleave(32, -1), _dequant_block(weight, scale)
    )


def _experts(**kwargs) -> Glm5NextMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return Glm5NextMegaMoEExperts(
        vllm_config,  # type: ignore[arg-type]
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=256,
        intermediate_size=128,
        weight_block_size=BLOCK,
        num_shared_experts=1,
        prefix="model.layers.3.mlp.experts",
        **kwargs,
    )


def test_loader_params_match_the_block_fp8_expert_mapping():
    experts = _experts()

    names = dict(experts.named_parameters())

    assert names["routed_experts.w13_weight"].shape == (2, 256, 256)
    assert names["routed_experts.w13_weight"].dtype == torch.float8_e4m3fn
    assert names["routed_experts.w13_weight_scale_inv"].shape == (2, 2, 2)
    assert names["routed_experts.w2_weight"].shape == (2, 256, 128)
    assert names["routed_experts.w2_weight_scale_inv"].shape == (2, 2, 1)


def test_weight_loader_places_gate_then_up_for_local_experts_only():
    experts = _experts()
    loader = experts.routed_experts
    assert loader is not None
    gate = torch.full((128, 256), 2.0).to(torch.float8_e4m3fn)
    up = torch.full((128, 256), 3.0).to(torch.float8_e4m3fn)
    up_scale = torch.full((1, 2), 0.5)

    def load(param, tensor, name, shard, expert):
        return experts.weight_loader(
            param, tensor, name, shard_id=shard, expert_id=expert, return_success=True
        )

    # Experts 0 and 1 live on another rank.
    assert not load(
        loader.w13_weight, gate, "experts.routed_experts.w13_weight", "w1", 1
    )
    assert load(loader.w13_weight, gate, "experts.routed_experts.w13_weight", "w1", 3)
    assert load(loader.w13_weight, up, "experts.routed_experts.w13_weight", "w3", 3)
    assert load(
        loader.w13_weight_scale_inv,
        up_scale,
        "experts.routed_experts.w13_weight_scale_inv",
        "w3",
        3,
    )

    w13 = loader.w13_weight.data.float()
    assert torch.all(w13[1, :128] == 2.0) and torch.all(w13[1, 128:] == 3.0)
    assert torch.all(w13[0] == 0.0)
    scale = loader.w13_weight_scale_inv.data
    assert torch.all(scale[1, 1] == 0.5) and torch.all(scale[1, 0] == 0.0)


class _FakeDeepGemm:
    """Identity layout transforms, so the test sees what the backend passes."""

    sf_calls: list[tuple] = []

    @classmethod
    def transform_sf_into_required_layout(cls, sf, mn, k, gran, num_groups):
        cls.sf_calls.append((tuple(sf.shape), mn, k, gran, num_groups))
        return sf

    @staticmethod
    def transform_weights_for_mega_moe(l1, l2):
        return l1, l2


def test_fp8_backend_requantizes_routed_weights_to_1x32(monkeypatch):
    torch.manual_seed(2)
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: _FakeDeepGemm)
    _FakeDeepGemm.sf_calls = []
    w13, s13 = _block_fp8(torch.randn(2, 256, 256) * 0.02)
    w2, s2 = _block_fp8(torch.randn(2, 256, 128) * 0.02)

    (l1, l1_sf), (l2, l2_sf) = DeepGemmSm100Fp8MegaMoEBackend().transform_weights(
        w13_weight=w13,
        w13_weight_scale=s13,
        w2_weight=w2,
        w2_weight_scale=s2,
        num_local_experts=2,
        hidden_size=256,
        intermediate_size=128,
    )

    assert _FakeDeepGemm.sf_calls == [
        ((2, 256, 8), 256, 256, (1, 32), 2),
        ((2, 256, 4), 256, 128, (1, 32), 2),
    ]
    for q, sf, w, s in ((l1, l1_sf, w13, s13), (l2, l2_sf, w2, s2)):
        assert q.dtype == torch.float8_e4m3fn
        restored = q.float() * sf.repeat_interleave(32, -1)
        original = _dequant_block(w, s)
        assert ((restored - original).norm() / original.norm()).item() < 0.05


def test_finalize_hands_the_checkpoint_weights_to_the_backend(monkeypatch):
    experts = _experts()
    seen = {}

    class Backend:
        mma_type = "fp8xfp8"

        def transform_weights(self, **kwargs):
            seen.update(kwargs)
            return ("l1", "l1_sf"), ("l2", "l2_sf")

    monkeypatch.setattr(experts, "_ensure_backend", lambda: Backend())
    monkeypatch.setattr("vllm.utils.deep_gemm._import_deep_gemm", lambda: object())

    experts.finalize_weights()

    assert seen["w13_weight"].dtype == torch.float8_e4m3fn
    assert seen["w13_weight_scale"].shape == (2, 2, 2)
    assert seen["w2_weight_scale"].shape == (2, 2, 1)
    assert experts._transformed_l1_weights == ("l1", "l1_sf")
    # Only the kernel's tensors remain.
    assert experts.routed_experts is None
    assert dict(experts.named_parameters()) == {}
