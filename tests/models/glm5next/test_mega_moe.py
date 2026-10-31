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
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.glm5next.nvidia.mega_moe import (
    Glm5NextMegaMoEExperts,
    requant_block_fp8_to_ue8m0,
)

BLOCK = (128, 128)
HIDDEN, INTER = 256, 128
FP8 = torch.float8_e4m3fn
# Global ids of the experts _experts() places on this rank.
LOCAL_EXPERTS = slice(2, 4)


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


def _dequant_ue8m0(weight: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return weight.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, -1)


def test_requant_gives_power_of_two_scale_per_32_columns():
    torch.manual_seed(0)
    real = torch.randn(3, 256, 384) * 0.02
    weight, scale = _block_fp8(real)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale)

    assert requant.dtype == FP8
    assert requant.shape == weight.shape
    assert sf.dtype == torch.uint8
    assert sf.shape == (3, 256, 384 // 32)
    assert requant.float().abs().max() <= 448.0
    original = _dequant_block(weight, scale)
    restored = _dequant_ue8m0(requant, sf)
    # One more E4M3 rounding of every element: a few percent, never garbage.
    rel = ((restored - original).norm() / original.norm()).item()
    assert 0 < rel < 0.05


def test_requant_is_exact_for_power_of_two_block_scales():
    torch.manual_seed(1)
    real = torch.randn(1, 256, 256) * 0.05
    weight, scale = _block_fp8(real, power_of_two=True)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale)

    # A finer power-of-two scale only shifts exponents, so nothing rounds.
    assert torch.equal(_dequant_ue8m0(requant, sf), _dequant_block(weight, scale))


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


def _load_shuffled(experts, ckpt, seed, suffix="") -> None:
    """Load the checkpoint tensors whose names end in ``suffix``, shuffled."""
    shards = []
    for expert_id in range(ckpt["w13_weight"].shape[0]):
        for name, tensor in ckpt.items():
            if not name.endswith(suffix):
                continue
            if name.startswith("w2_"):
                shards.append((name, tensor[expert_id], "w2", expert_id))
            else:
                half = tensor.shape[1] // 2
                shards.append((name, tensor[expert_id, :half], "w1", expert_id))
                shards.append((name, tensor[expert_id, half:], "w3", expert_id))
    generator = torch.Generator().manual_seed(seed)
    for i in torch.randperm(len(shards), generator=generator).tolist():
        name, loaded_weight, shard_id, expert_id = shards[i]
        param = getattr(experts, name)
        loaded = param.weight_loader(
            param,
            loaded_weight,
            f"experts.{name}",
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        # Physical experts 2 and 3 are this rank's.
        assert loaded == (expert_id >= 2)


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


def _expected_routed(ckpt):
    local = LOCAL_EXPERTS
    w13, s13 = requant_block_fp8_to_ue8m0(
        ckpt["w13_weight"][local], ckpt["w13_weight_scale_inv"][local]
    )
    w2, s2 = requant_block_fp8_to_ue8m0(
        ckpt["w2_weight"][local], ckpt["w2_weight_scale_inv"][local]
    )
    return _reference_mega_moe_layout(w13, s13, w2, s2)


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


@pytest.mark.parametrize("first", ["weight", "weight_scale_inv"])
def test_weight_update_split_across_calls(first):
    """A weight update may send weights and block scales in separate
    load_weights calls. The finish step rejects a shard still missing its
    other half and requantizes each shard once both are in."""
    experts = _experts()
    _load_shuffled(experts, _checkpoint(4, seed=0), seed=0)
    experts.requantize_block_fp8()

    ckpt = _checkpoint(4, seed=1)
    second = "weight_scale_inv" if first == "weight" else "weight"
    _load_shuffled(experts, ckpt, seed=1, suffix=first)
    with pytest.raises(ValueError, match="without their weight or block scale"):
        experts.requantize_block_fp8()
    _load_shuffled(experts, ckpt, seed=2, suffix=second)
    experts.requantize_block_fp8()

    _assert_layout(_routed(experts), _expected_routed(ckpt))


def _shared_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w13, s13 = _block_fp8(torch.randn(1, 2 * INTER, HIDDEN, generator=generator) * 0.02)
    w2, s2 = _block_fp8(torch.randn(1, HIDDEN, INTER, generator=generator) * 0.02)
    return {"w13": w13, "s13": s13, "w2": w2, "s2": s2}


def _load_shared(experts, ckpt) -> None:
    w13, s13, w2, s2 = ckpt["w13"], ckpt["s13"], ckpt["w2"], ckpt["s2"]
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


def _expected_shared(ckpt):
    q13, sf13 = requant_block_fp8_to_ue8m0(ckpt["w13"], ckpt["s13"])
    q2, sf2 = requant_block_fp8_to_ue8m0(ckpt["w2"], ckpt["s2"])
    return [t[0] for t in _reference_mega_moe_layout(q13, sf13, q2, sf2)]


def _shared(experts):
    return (
        experts.shared_l1_weight,
        experts.shared_l1_weight_scale,
        experts.shared_l2_weight,
        experts.shared_l2_weight_scale,
    )


def test_block_fp8_shared_expert_is_fused_and_requantized():
    experts = _experts()
    assert experts.has_fused_shared_experts
    ckpt = _shared_checkpoint(seed=0)
    _load_shared(experts, ckpt)

    experts.requantize_block_fp8()

    _assert_layout(_shared(experts), _expected_shared(ckpt))


def test_layerwise_reload_requantizes_into_the_kernel_tensors():
    """Reload and weight transfer restore the layer on meta, buffer every load,
    replay it once the layer is complete and finish with the model's
    process_weights_after_loading; the kernel tensors keep their storage."""
    experts = _experts()
    record_metadata_for_reloading(experts)
    _load_shuffled(experts, _checkpoint(4, seed=0), seed=0)
    _load_shared(experts, _shared_checkpoint(seed=0))
    experts.requantize_block_fp8()
    storage = {n: p.data_ptr() for n, p in experts.named_parameters()}

    ckpt, shared_ckpt = _checkpoint(4, seed=1), _shared_checkpoint(seed=1)
    initialize_layerwise_reload(experts)
    _load_shuffled(experts, ckpt, seed=1)
    _load_shared(experts, shared_ckpt)
    # Every loaded tensor is in, so the layer was replayed without waiting for
    # finalize: the derived kernel scales do not count as missing loads.
    assert not experts.w13_weight.is_meta
    finalize_layerwise_reload(experts, None)  # type: ignore[arg-type]
    experts.requantize_block_fp8()

    _assert_layout(_routed(experts), _expected_routed(ckpt))
    _assert_layout(_shared(experts), _expected_shared(shared_ckpt))
    assert {n: p.data_ptr() for n, p in experts.named_parameters()} == storage


def test_shared_expert_with_other_scales_is_not_fused():
    mlp = _shared_mlp(INTER)
    mlp.down_proj.weight_scale_inv = torch.empty(HIDDEN, INTER // 32)
    experts = _experts(shared_experts=mlp)

    assert not experts.has_fused_shared_experts
    assert not any(n.startswith("shared_") for n, _ in experts.named_parameters())


def test_mxfp8_shared_expert_is_not_fused():
    """Only a block-FP8 shared expert is fused; the GLM loader cannot place an
    MXFP8 one, so it stays a separate MLP."""
    mlp = _shared_mlp(INTER)
    for linear in (mlp.gate_up_proj, mlp.down_proj):
        n, k = linear.weight.shape
        del linear.weight_scale_inv
        linear.weight_scale = torch.empty(n, k // 32, dtype=torch.uint8)
    experts = _experts(shared_experts=mlp)

    assert not experts.has_fused_shared_experts


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

    with pytest.raises(ValueError, match="without their weight or block scale"):
        experts.requantize_block_fp8()
