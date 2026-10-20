# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MegaMoE: block-FP8 checkpoint shards requantized to the
MXFP8 MegaMoE kernel layout."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.models.glm5next.nvidia.mega_moe import (
    Glm5NextMegaMoEExperts,
    requant_block_fp8_to_ue8m0,
)

BLOCK = (128, 128)
HIDDEN, INTER = 256, 128
FP8 = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def sm100_mega_moe_layout(monkeypatch):
    """Build the SM100 kernel layout on any device."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )
    monkeypatch.setattr(
        DeepGemmMegaMoEExperts, "_kernel_fuses_shared_experts", lambda self, spec: True
    )


def _block_fp8(real: torch.Tensor, power_of_two: bool = False):
    """Quantize ``(..., mn, k)`` to E4M3 with one float32 scale per 128x128."""
    *lead, mn, k = real.shape
    blocks = real.reshape(*lead, mn // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(-3, -1)) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    full = scale.repeat_interleave(128, -2).repeat_interleave(128, -1)
    return (real / full).to(FP8), scale


def _dequant_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return weight.float() * scale.repeat_interleave(128, -2).repeat_interleave(128, -1)


def test_requant_gives_power_of_two_scale_per_32_columns():
    torch.manual_seed(0)
    real = torch.randn(3, 256, 384) * 0.02
    weight, scale = _block_fp8(real)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale, chunk=2)

    assert requant.dtype == FP8
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


def _shared_mlp(shared_size: int):
    def linear(n, k):
        return SimpleNamespace(
            weight=torch.empty(n, k, dtype=FP8),
            weight_scale_inv=torch.empty(n // BLOCK[0], k // BLOCK[1]),
        )

    return SimpleNamespace(
        gate_up_proj=linear(2 * shared_size, HIDDEN),
        down_proj=linear(HIDDEN, shared_size),
    )


def _experts(**kwargs) -> Glm5NextMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    kwargs.setdefault("shared_experts", _shared_mlp(INTER))
    return Glm5NextMegaMoEExperts(
        vllm_config,  # type: ignore[arg-type]
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        weight_block_size=BLOCK,
        activation=MegaMoEActivation("swiglu", clamp=10.0),
        prefix="model.layers.3.mlp.experts",
        **kwargs,
    )


def _checkpoint(num_experts: int, seed: int) -> dict[str, torch.Tensor]:
    """Block-FP8 experts as stored: gate rows, then up rows, for w13."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    w13, s13 = _block_fp8(rand(num_experts, 2 * INTER, HIDDEN))
    w2, s2 = _block_fp8(rand(num_experts, HIDDEN, INTER))
    return {
        "w13_weight": w13,
        "w13_weight_scale_inv": s13,
        "w2_weight": w2,
        "w2_weight_scale_inv": s2,
    }


def _load_shuffled(experts, ckpt, seed) -> None:
    shards = []
    for expert_id in range(ckpt["w13_weight"].shape[0]):
        for name, tensor in ckpt.items():
            if name.startswith("w2_"):
                shards.append((name, tensor[expert_id], "w2", expert_id))
            else:
                half = tensor.shape[1] // 2
                shards.append((name, tensor[expert_id, :half], "w1", expert_id))
                shards.append((name, tensor[expert_id, half:], "w3", expert_id))
    generator = torch.Generator().manual_seed(seed)
    for i in torch.randperm(len(shards), generator=generator).tolist():
        name, loaded_weight, shard_id, expert_id = shards[i]
        loaded = experts.weight_loader(
            getattr(experts, name),
            loaded_weight,
            f"experts.{name}",
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        # Physical experts 2 and 3 are this rank's.
        assert loaded == (expert_id >= 2)


def _ue8m0(sf: torch.Tensor) -> torch.Tensor:
    return (sf.view(torch.int32) >> 23).to(torch.uint8)


def _reference_mega_moe_layout(w13, s13, w2, s2):
    """CPU copy of DeepGEMM's transform_weights_for_mega_moe on packed scales."""

    def interleave(t):
        g, n, k = t.shape
        halves = t.view(g, 2, n // 16, 8, k)
        return torch.stack([halves[:, 0], halves[:, 1]], dim=2).reshape(g, n, k)

    def utccp(sf):
        g, mn, k = sf.shape
        return sf.reshape(g, -1, 4, 32, k).transpose(2, 3).reshape(g, mn, k)

    return (
        interleave(w13.view(torch.uint8)),
        utccp(interleave(s13.view(torch.int32))),
        w2.view(torch.uint8),
        utccp(s2.view(torch.int32)),
    )


def _expected_routed(ckpt, local=slice(2, 4)):
    w13, s13 = requant_block_fp8_to_ue8m0(
        ckpt["w13_weight"][local], ckpt["w13_weight_scale_inv"][local]
    )
    w2, s2 = requant_block_fp8_to_ue8m0(
        ckpt["w2_weight"][local], ckpt["w2_weight_scale_inv"][local]
    )
    return _reference_mega_moe_layout(w13, _ue8m0(s13), w2, _ue8m0(s2))


def _assert_layout(actual, expected):
    for tensor, reference in zip(actual, expected):
        assert torch.equal(tensor.view(reference.dtype), reference)


def _routed(experts):
    return (*experts.l1_weights, *experts.l2_weights)


def test_block_fp8_shards_requantize_into_the_kernel_layout():
    experts = _experts()
    ckpt = _checkpoint(4, seed=0)
    _load_shuffled(experts, ckpt, seed=0)
    assert torch.equal(experts.w13_weight_scale_inv, ckpt["w13_weight_scale_inv"][2:])

    experts.requantize_block_fp8()

    expected = _expected_routed(ckpt)
    _assert_layout(_routed(experts), expected)
    # Already requantized shards are left alone by a second call.
    experts.requantize_block_fp8()
    _assert_layout(_routed(experts), expected)
    # A reload writes raw block-FP8 values again and requantizes them in place.
    ptr = experts.w13_weight.data_ptr()
    ckpt = _checkpoint(4, seed=1)
    _load_shuffled(experts, ckpt, seed=1)
    experts.requantize_block_fp8()
    assert experts.w13_weight.data_ptr() == ptr
    _assert_layout(_routed(experts), _expected_routed(ckpt))


def test_block_fp8_shared_expert_is_fused_and_requantized():
    experts = _experts()
    assert experts.has_fused_shared_experts
    w13, s13 = _block_fp8(torch.randn(1, 2 * INTER, HIDDEN) * 0.02)
    w2, s2 = _block_fp8(torch.randn(1, HIDDEN, INTER) * 0.02)
    params = {f"mlp.experts.{n}" for n, _ in experts.named_parameters()}
    for name, data in [
        ("down_proj.weight_scale_inv", s2[0]),
        ("gate_proj.weight", w13[0, :INTER]),
        ("up_proj.weight_scale_inv", s13[0, 1:]),
        ("down_proj.weight", w2[0]),
        ("up_proj.weight", w13[0, INTER:]),
        ("gate_proj.weight_scale_inv", s13[0, :1]),
    ]:
        loaded = experts.load_shared_expert_weight(f"mlp.shared_experts.{name}", data)
        assert loaded in params

    experts.requantize_block_fp8()

    q13, sf13 = requant_block_fp8_to_ue8m0(w13, s13)
    q2, sf2 = requant_block_fp8_to_ue8m0(w2, s2)
    expected = _reference_mega_moe_layout(q13, _ue8m0(sf13), q2, _ue8m0(sf2))
    actual = (
        experts.shared_l1_weight,
        experts.shared_l1_weight_scale,
        experts.shared_l2_weight,
        experts.shared_l2_weight_scale,
    )
    _assert_layout(actual, (t[0] for t in expected))


def test_shared_expert_with_other_scales_is_not_fused():
    mlp = _shared_mlp(INTER)
    mlp.down_proj.weight_scale_inv = torch.empty(HIDDEN, INTER // 32)
    experts = _experts(shared_experts=mlp)

    assert not experts.has_fused_shared_experts
    assert not any(n.startswith("shared_") for n, _ in experts.named_parameters())


def test_requantize_rejects_missing_block_scales():
    experts = _experts(shared_experts=None)
    ckpt = _checkpoint(4, seed=2)
    for shard_id, rows in (("w1", slice(0, INTER)), ("w3", slice(INTER, None))):
        experts.weight_loader(
            experts.w13_weight,
            ckpt["w13_weight"][2, rows],
            "experts.w13_weight",
            shard_id=shard_id,
            expert_id=2,
        )

    with pytest.raises(ValueError, match="block scales"):
        experts.requantize_block_fp8()
