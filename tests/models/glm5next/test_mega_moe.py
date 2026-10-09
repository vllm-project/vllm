# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.glm5next.nvidia.mega_moe import (
    Glm5NextMegaMoEExperts,
    requant_block_fp8_to_ue8m0,
)

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

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale, BLOCK, chunk=2)

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

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale, BLOCK)

    # A finer power-of-two scale only shifts exponents, so nothing rounds.
    assert torch.equal(
        requant.float() * sf.repeat_interleave(32, -1), _dequant_block(weight, scale)
    )


def _experts(**kwargs) -> Glm5NextMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64)
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
