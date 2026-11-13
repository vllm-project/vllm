# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 MegaMoE: block-FP8 checkpoint shards requantized to the MXFP8
MegaMoE kernel layout (kept as block-FP8 on SM90), and BF16 shards loaded as
they are."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_FP8_BLOCK_SM90,
    MegaMoEActivation,
    check_deep_gemm_mega_moe_supported,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    requant_block_fp8_to_ue8m0,
)
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.glm5next.nvidia.mega_moe import Glm5NextMegaMoEExperts
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import transform_fp8_block_sf_for_sm90_mega_moe
from vllm.utils.import_utils import has_deep_gemm

BLOCK = (128, 128)
HIDDEN, INTER = 256, 128
FP8 = torch.float8_e4m3fn
NUM_EXPERTS = 4
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
        current_platform, "is_device_capability", lambda *args, **kwargs: False
    )


def _block_fp8(real: torch.Tensor, power_of_two: bool = False):
    """Quantize `(..., mn, k)` to E4M3 with one float32 scale per 128x128."""
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


def test_requant_is_exact_for_power_of_two_block_scales():
    torch.manual_seed(1)
    real = torch.randn(1, 256, 256) * 0.05
    weight, scale = _block_fp8(real, power_of_two=True)

    requant, sf = requant_block_fp8_to_ue8m0(weight, scale)

    # A finer power-of-two scale only shifts exponents, so nothing rounds.
    assert torch.equal(_dequant_ue8m0(requant, sf), _dequant_block(weight, scale))


def _experts(weight_block_size=BLOCK) -> Glm5NextMegaMoEExperts:
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return Glm5NextMegaMoEExperts(
        vllm_config,  # type: ignore[arg-type]
        num_experts=NUM_EXPERTS,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        weight_block_size=weight_block_size,
        activation=MegaMoEActivation("swiglu", clamp=10.0),
        num_shared_experts=1,
        prefix="model.layers.3.mlp.experts",
    )


def _checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """Block-FP8 experts as stored: gate rows, then up rows, for w13."""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    w13, s13 = _block_fp8(rand(NUM_EXPERTS, 2 * INTER, HIDDEN))
    w2, s2 = _block_fp8(rand(NUM_EXPERTS, HIDDEN, INTER))
    return {
        "w13_weight": w13,
        "w13_weight_scale_inv": s13,
        "w2_weight": w2,
        "w2_weight_scale_inv": s2,
    }


def _load_shuffled(experts, ckpt, seed, suffix="") -> None:
    """Load the checkpoint tensors whose names end in `suffix`, shuffled."""
    shards = []
    for expert_id in range(NUM_EXPERTS):
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


def _interleave(t):
    """Gate and up rows interleaved by 8, as DeepGEMM's MegaMoE L1 takes them."""
    g, n, k = t.shape
    halves = t.view(g, 2, n // 16, 8, k)
    return torch.stack([halves[:, 0], halves[:, 1]], dim=2).reshape(g, n, k)


def _reference_mega_moe_layout(w13, s13, w2, s2):
    """CPU copy of DeepGEMM's transform_weights_for_mega_moe on packed scales."""

    def utccp(sf):
        g, mn, k = sf.shape
        return sf.reshape(g, -1, 4, 32, k).transpose(2, 3).reshape(g, mn, k)

    return (
        _interleave(w13.view(torch.uint8)),
        utccp(_interleave(s13.view(torch.int32))),
        w2.view(torch.uint8),
        utccp(s2.view(torch.int32)),
    )


def _sm90_layout(w13, s13, w2, s2):
    """Block-FP8 weights as loaded, block scales laid out by DeepGEMM."""
    return (
        _interleave(w13.view(torch.uint8)),
        transform_fp8_block_sf_for_sm90_mega_moe(w13, s13, is_l1=True),
        w2.view(torch.uint8),
        transform_fp8_block_sf_for_sm90_mega_moe(w2, s2, is_l1=False),
    )


def _expected(w13, s13, w2, s2, sm90):
    if sm90:
        return _sm90_layout(w13, s13, w2, s2)
    w13, s13 = requant_block_fp8_to_ue8m0(w13, s13)
    w2, s2 = requant_block_fp8_to_ue8m0(w2, s2)
    return _reference_mega_moe_layout(w13, s13, w2, s2)


def _expected_routed(ckpt, sm90=False):
    w13, s13, w2, s2 = (t[LOCAL_EXPERTS] for t in ckpt.values())
    return _expected(w13, s13, w2, s2, sm90)


def _assert_layout(actual, expected, sm90=False):
    for tensor, reference in zip(actual, expected):
        assert torch.equal(tensor.view(reference.dtype), reference)
        # The SM90 kernel reads the scales with DeepGEMM's strides.
        assert not sm90 or tensor.stride() == reference.stride()


def _routed(experts):
    return (*experts.l1_weights, *experts.l2_weights)


@pytest.mark.parametrize("first", ["weight", "weight_scale_inv"])
def test_weight_update_split_across_calls(first):
    """A weight update may send weights and block scales in separate
    load_weights calls. The finish step rejects a shard still missing its
    other half and requantizes each shard once both are in."""
    experts = _experts()
    _load_shuffled(experts, _checkpoint(seed=0), seed=0)
    experts.requantize_block_fp8()

    ckpt = _checkpoint(seed=1)
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


def _expected_shared(ckpt, sm90=False):
    expected = _expected(ckpt["w13"], ckpt["s13"], ckpt["w2"], ckpt["s2"], sm90)
    return [t[0] for t in expected]


def _shared(experts):
    return (
        experts.shared_l1_weight,
        experts.shared_l1_weight_scale,
        experts.shared_l2_weight,
        experts.shared_l2_weight_scale,
    )


class _Model(torch.nn.Module):
    """Finishes a load the way the GLM model, wrapper and MTP drafter do."""

    def __init__(self):
        super().__init__()
        self.experts = _experts()

    def process_weights_after_loading(self):
        self.experts.finalize_block_fp8_weights()


@pytest.mark.parametrize(
    "sm90",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not has_deep_gemm(), reason="no DeepGEMM")
        ),
    ],
)
@pytest.mark.parametrize("mode", ["layerwise", "direct"])
def test_reload_requantizes_into_the_kernel_tensors(monkeypatch, mode, sm90):
    """A reload writes the block-FP8 shards and finishes with the model's
    process_weights_after_loading; the kernel tensors keep their storage."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap, **_: sm90 and cap == 90
    )
    model_config = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False, quantization=None
    )
    device = torch.get_default_device()
    model = _Model()
    experts = model.experts
    record_metadata_for_reloading(model)
    ckpt, shared_ckpt = _checkpoint(seed=0), _shared_checkpoint(seed=0)
    _load_shuffled(experts, ckpt, seed=0)
    _load_shared(experts, shared_ckpt)
    process_weights_after_loading(model, model_config, device)  # type: ignore[arg-type]
    # A second call leaves the kernel tensors as they are.
    experts.finalize_block_fp8_weights()
    _assert_layout(_routed(experts), _expected_routed(ckpt, sm90), sm90)
    _assert_layout(_shared(experts), _expected_shared(shared_ckpt, sm90), sm90)
    storage = {n: p.data_ptr() for n, p in experts.named_parameters()}

    ckpt, shared_ckpt = _checkpoint(seed=1), _shared_checkpoint(seed=1)
    start_reload(model)
    _load_shuffled(experts, ckpt, seed=1)
    _load_shared(experts, shared_ckpt)
    # Every loaded tensor is in, so the layer is not meta any more: the derived
    # kernel scales do not count as missing loads.
    assert not experts.w13_weight.is_meta
    finish_reload(model, model_config, device)  # type: ignore[arg-type]

    _assert_layout(_routed(experts), _expected_routed(ckpt, sm90), sm90)
    _assert_layout(_shared(experts), _expected_shared(shared_ckpt, sm90), sm90)
    assert {n: p.data_ptr() for n, p in experts.named_parameters()} == storage


def test_sm90_eplb_views_cover_padded_block_scales(monkeypatch):
    """EPLB moves experts as rows of these views. The SM90 block scales pad
    their MN stride to 4 (N/128 = 2 here), and each row must still be one
    expert of the parameter."""
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda cap, **_: cap == 90
    )
    experts = _experts()
    names = [n for n, _ in experts.named_parameters() if not n.startswith("shared_")]
    for name, view in zip(names, experts.get_expert_weights(), strict=True):
        param = getattr(experts, name)
        view[1].fill_(7)
        assert (param[1].float() == 7).all(), name
        assert (param[0].float() == 0).all(), name


def test_sm90_format_rejected_on_sm100(monkeypatch):
    monkeypatch.setattr(
        current_platform, "is_device_capability", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda *args, **kwargs: True
    )
    with pytest.raises(NotImplementedError, match="requires SM90"):
        check_deep_gemm_mega_moe_supported(
            torch.device("cpu"), HIDDEN, INTER, MEGA_MOE_FP8_BLOCK_SM90
        )


def test_bf16_checkpoint_loads_as_is():
    """Without a weight block size the experts take BF16 shards straight into
    the BF16 kernel layout, the shared expert included, and stage no scales."""
    experts = _experts(weight_block_size=None)
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator=generator).bfloat16()

    w13, w2 = rand(NUM_EXPERTS, 2 * INTER, HIDDEN), rand(NUM_EXPERTS, HIDDEN, INTER)
    _load_shuffled(experts, {"w13_weight": w13, "w2_weight": w2}, seed=0)
    shared13, shared2 = rand(1, 2 * INTER, HIDDEN), rand(HIDDEN, INTER)
    for name, data in [
        ("down_proj.weight", shared2),
        ("up_proj.weight", shared13[0, INTER:]),
        ("gate_proj.weight", shared13[0, :INTER]),
    ]:
        experts.load_shared_expert_weight(f"mlp.shared_experts.{name}", data)

    assert not [n for n, _ in experts.named_parameters() if "scale" in n]
    assert torch.equal(experts.w13_weight, _interleave(w13[LOCAL_EXPERTS]))
    assert torch.equal(experts.w2_weight, w2[LOCAL_EXPERTS])
    assert torch.equal(experts.shared_l1_weight, _interleave(shared13)[0])
    assert torch.equal(experts.shared_l2_weight, shared2)
