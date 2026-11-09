# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Qwen4Exp routed experts loaded for MegaMoE."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_BF16,
    MEGA_MOE_MXFP4,
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.models.qwen4_exp.nvidia.mega_moe import (
    Qwen4ExpMegaMoEExperts,
    flush_mega_moe_requantize,
    mega_moe_expert_formats,
    requantize_experts,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

HIDDEN, INTER = 512, 256
SCALE_NAMES = {
    "fp8_block": "weight_scale_inv",
    "mxfp8": "weight_scale",
    "mxfp4": "weight_scale",
}
pytestmark = pytest.mark.skipif(
    not is_deep_gemm_supported(), reason="Requires DeepGEMM"
)


def _cast(fmt):
    from vllm.third_party.deep_gemm.utils.math import (
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    return per_token_cast_to_fp4 if fmt is MEGA_MOE_MXFP4 else per_token_cast_to_fp8


def _exponent(sf: torch.Tensor) -> torch.Tensor:
    return (sf.view(torch.int32) >> 23).to(torch.uint8)


def _cast_back(values: torch.Tensor, exponent: torch.Tensor, fmt) -> torch.Tensor:
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp4,
        cast_back_from_fp8,
    )

    sf = (exponent.to(torch.int32) << 23).view(torch.float32)
    cast_back = cast_back_from_fp4 if fmt is MEGA_MOE_MXFP4 else cast_back_from_fp8
    return cast_back(values, sf, gran_k=32)


def _shard(checkpoint_format: str, rows: int, cols: int, generator):
    """One projection of one expert as the checkpoint stores it."""
    if checkpoint_format == "fp8_block":
        weight = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
        scale = torch.rand(rows // 128, cols // 128, generator=generator) * 1e-2
        return weight, scale + 1e-3
    x = torch.randn(rows, cols, generator=generator) * 1e-2
    if checkpoint_format == "bf16":
        return x.bfloat16(), None
    fmt = MEGA_MOE_MXFP4 if checkpoint_format == "mxfp4" else MEGA_MOE_MXFP8
    values, sf = _cast(fmt)(x, use_ue8m0=True, gran_k=32)
    # MXFP4 checkpoints store packed uint8 values.
    if fmt is MEGA_MOE_MXFP4:
        values = values.view(torch.uint8)
    return values, _exponent(sf)


def _checkpoint(checkpoint_format: str, num_experts: int, generator):
    shapes = {
        "gate_proj": (INTER, HIDDEN),
        "up_proj": (INTER, HIDDEN),
        "down_proj": (HIDDEN, INTER),
    }
    return {
        (e, proj): _shard(checkpoint_format, *shape, generator)
        for e in range(num_experts)
        for proj, shape in shapes.items()
    }


def _names(checkpoint_format, ckpt, generator) -> list[tuple[str, torch.Tensor]]:
    """Per-expert checkpoint tensors in a random order."""
    names = [(f"{e}.{p}.weight", w) for (e, p), (w, _) in ckpt.items()]
    if checkpoint_format != "bf16":
        scale_name = SCALE_NAMES[checkpoint_format]
        names += [(f"{e}.{p}.{scale_name}", s) for (e, p), (_, s) in ckpt.items()]
    return [names[i] for i in torch.randperm(len(names), generator=generator)]


def _fused_names(ckpt, num_experts) -> list[tuple[str, torch.Tensor]]:
    """The official BF16 checkpoint's fused [E, N, K] experts."""
    experts = range(num_experts)
    gate_up = [
        torch.cat([ckpt[e, "gate_proj"][0], ckpt[e, "up_proj"][0]]) for e in experts
    ]
    down = [ckpt[e, "down_proj"][0] for e in experts]
    return [("gate_up_proj", torch.stack(gate_up)), ("down_proj", torch.stack(down))]


def _experts(cls, fmt, num_local, start, num_logical, checkpoint_format=None):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    kwargs = dict(
        weight_format=fmt,
        num_experts=num_logical,
        num_local_experts=num_local,
        experts_start_idx=start,
        num_logical_experts=num_logical,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
    )
    if cls is DeepGemmMegaMoEExperts:
        kwargs["activation"] = MegaMoEActivation()
    else:
        kwargs["checkpoint_format"] = checkpoint_format
    with torch.device("cuda"):
        return cls(vllm_config, **kwargs)


def _expected(checkpoint_format, fmt, ckpt, num_local, start, num_logical):
    """Experts written shard by shard through the generic MegaMoE loader."""
    expected = _experts(DeepGemmMegaMoEExperts, fmt, num_local, start, num_logical)
    shard_ids = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for (e, proj), (weight, scale) in ckpt.items():
        if checkpoint_format != fmt.name:
            scale = None if scale is None else scale.cuda()
            weight, scale = requantize_experts(
                weight.cuda(), scale, checkpoint_format, fmt
            )
        prefix = "w2" if proj == "down_proj" else "w13"
        for suffix, value in (("weight", weight), ("weight_scale", scale)):
            if value is not None:
                name = f"{prefix}_{suffix}"
                expected.weight_loader(
                    getattr(expected, name), value, name, shard_ids[proj], e
                )
    return expected


def _assert_same_experts(a, b) -> None:
    for name, param in b.named_parameters():
        assert torch.equal(getattr(a, name), param), name


def _load(experts, names) -> None:
    experts.load_weights(iter(names))
    flush_mega_moe_requantize(experts)


@pytest.fixture
def any_device_mega_moe(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


def _with_online(config, **online) -> SimpleNamespace:
    config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(**online)
    )
    return SimpleNamespace(quant_config=config)


def test_expert_formats_follow_checkpoint_and_quantization_config():
    prefix, other = "model.layers.3.mlp.experts", "model.layers.47.mlp.experts"
    fp8 = Fp8Config(weight_block_size=[128, 128])
    # Block FP8 has no MegaMoE kernel: MXFP8 unless a format is forced.
    assert mega_moe_expert_formats(SimpleNamespace(quant_config=fp8), prefix) == (
        "fp8_block",
        MEGA_MOE_MXFP8,
    )
    config = _with_online(fp8, moe="mxfp4", ignore=["*.layers.47.*"])
    assert mega_moe_expert_formats(config, prefix)[1] is MEGA_MOE_MXFP4
    assert mega_moe_expert_formats(config, other)[1] is MEGA_MOE_MXFP8
    # Only the forced layer shows in the online quantization summary.
    assert list(config.quant_config.online_quantization_config.quantized_layers) == [
        prefix
    ]
    config = _with_online(Fp8Config(weight_block_size=[128, 128]), moe="mxfp8")
    assert mega_moe_expert_formats(config, prefix) == ("fp8_block", MEGA_MOE_MXFP8)

    mxfp8 = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    assert mega_moe_expert_formats(SimpleNamespace(quant_config=mxfp8), prefix) == (
        "mxfp8",
        MEGA_MOE_MXFP8,
    )
    config = _with_online(mxfp8, moe="mxfp4")
    assert mega_moe_expert_formats(config, prefix) == ("mxfp8", MEGA_MOE_MXFP4)

    bf16 = SimpleNamespace(quant_config=None)
    assert mega_moe_expert_formats(bf16, prefix) == ("bf16", MEGA_MOE_BF16)
    online = OnlineQuantizationConfig(QuantizationConfigArgs(moe="mxfp4"))
    online_bf16 = SimpleNamespace(quant_config=online)
    assert mega_moe_expert_formats(online_bf16, prefix) == ("bf16", MEGA_MOE_MXFP4)


@pytest.mark.parametrize(
    "checkpoint_format,fmt",
    [
        ("fp8_block", MEGA_MOE_MXFP4),
        ("mxfp8", MEGA_MOE_MXFP4),
        ("mxfp4", MEGA_MOE_MXFP8),
        ("bf16", MEGA_MOE_MXFP4),
    ],
)
def test_requantize_matches_deep_gemm_cast(checkpoint_format, fmt):
    """Requantization dequantizes exactly and rounds as DeepGEMM's casts."""
    weight, scale = _shard(checkpoint_format, INTER, HIDDEN, torch.Generator())
    weight = weight.cuda()
    scale = None if scale is None else scale.cuda()
    if checkpoint_format == "fp8_block":
        assert scale is not None
        scale_full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        dequantized = weight.float() * scale_full
    elif checkpoint_format == "mxfp4":
        dequantized = _cast_back(weight, scale, MEGA_MOE_MXFP4)
    elif checkpoint_format == "mxfp8":
        dequantized = _cast_back(weight, scale, MEGA_MOE_MXFP8)
    else:
        dequantized = weight.float()
    values, exponent = requantize_experts(weight, scale, checkpoint_format, fmt)
    ref_values, ref_sf = _cast(fmt)(dequantized, use_ue8m0=True, gran_k=32)
    assert torch.equal(values.view(torch.uint8), ref_values.view(torch.uint8))
    assert torch.equal(exponent, _exponent(ref_sf))


@pytest.mark.parametrize(
    "checkpoint_format,fmt",
    [
        ("fp8_block", MEGA_MOE_MXFP8),
        ("mxfp8", MEGA_MOE_MXFP8),
        ("mxfp4", MEGA_MOE_MXFP4),
        ("bf16", MEGA_MOE_BF16),
        ("fused_bf16", MEGA_MOE_BF16),
        ("fused_bf16", MEGA_MOE_MXFP4),
    ],
)
def test_experts_load_checkpoint_into_kernel_layout(
    any_device_mega_moe, monkeypatch, checkpoint_format, fmt
):
    """Shards in any order, weight and scale apart, only owned experts kept."""
    # One expert per requantize call, so batching spans several calls.
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.mega_moe._REQUANTIZE_CHUNK_ELEMENTS",
        INTER * HIDDEN,
    )
    fused = checkpoint_format == "fused_bf16"
    checkpoint_format = "bf16" if fused else checkpoint_format
    generator = torch.Generator().manual_seed(2)
    ckpt = _checkpoint(checkpoint_format, 4, generator)
    names = (
        _fused_names(ckpt, 4) if fused else _names(checkpoint_format, ckpt, generator)
    )
    # EP rank 1 of 2 owns logical experts 2 and 3.
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 2, 4, checkpoint_format)
    loaded = experts.load_weights(iter(names))
    assert loaded == {n for n, _ in experts.named_parameters()}
    experts.quant_method.process_weights_after_loading(experts)
    _assert_same_experts(experts, _expected(checkpoint_format, fmt, ckpt, 2, 2, 4))


def test_weight_and_scale_may_arrive_in_separate_calls(any_device_mega_moe):
    """Chunked weight updates may split an expert's weight from its scale and
    a rank's experts across calls; a pair still incomplete fails the load."""
    generator = torch.Generator().manual_seed(3)
    ckpt = _checkpoint("fp8_block", 2, generator)
    names = _names("fp8_block", ckpt, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, MEGA_MOE_MXFP8, 2, 0, 2, "fp8_block")
    first = [n for n in names if n[0].startswith("0.") or n[0].endswith(".weight")]
    _load(experts, first)
    with pytest.raises(ValueError, match="without both weight and scale"):
        experts.quant_method.process_weights_after_loading(experts)
    _load(experts, [n for n in names if n not in first])
    experts.quant_method.process_weights_after_loading(experts)
    _assert_same_experts(experts, _expected("fp8_block", MEGA_MOE_MXFP8, ckpt, 2, 0, 2))


@pytest.mark.parametrize(
    "checkpoint_format,fmt",
    [("fp8_block", MEGA_MOE_MXFP4), ("mxfp8", MEGA_MOE_MXFP8)],
    ids=["requantized", "as_stored"],
)
def test_layerwise_reload_replaces_routed_experts(
    any_device_mega_moe, checkpoint_format, fmt
):
    """reload_weights / weight transfer put the experts on meta and wrap their
    loaders; the new experts must land in the original kernel tensors."""
    generator = torch.Generator().manual_seed(5)
    old = _checkpoint(checkpoint_format, 2, generator)
    new = _checkpoint(checkpoint_format, 2, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 0, 2, checkpoint_format)
    with torch.device("cuda"):
        record_metadata_for_reloading(experts)
    _load(experts, _names(checkpoint_format, old, generator))
    experts.quant_method.process_weights_after_loading(experts)
    pointers = {n: p.data_ptr() for n, p in experts.named_parameters()}

    initialize_layerwise_reload(experts)
    _load(experts, _names(checkpoint_format, new, generator))
    finalize_layerwise_reload(experts, None)

    _assert_same_experts(experts, _expected(checkpoint_format, fmt, new, 2, 0, 2))
    # CUDA graphs keep pointing at the same storage.
    assert {n: p.data_ptr() for n, p in experts.named_parameters()} == pointers


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="DeepGEMM MegaMoE kernels require SM100",
)
@pytest.mark.parametrize("fmt", [MEGA_MOE_MXFP8, MEGA_MOE_MXFP4, MEGA_MOE_BF16])
def test_loaded_experts_match_dequantized_reference(dist_init, fmt):
    """Each kernel format against a dequantized reference. BF16 experts load
    as stored; MX experts are requantized from block FP8."""
    checkpoint_format = "bf16" if fmt is MEGA_MOE_BF16 else "fp8_block"
    num_experts, top_k, num_tokens = 8, 2, 48
    generator = torch.Generator().manual_seed(4)
    ckpt = _checkpoint(checkpoint_format, num_experts, generator)
    experts = _experts(
        Qwen4ExpMegaMoEExperts, fmt, num_experts, 0, num_experts, checkpoint_format
    )
    _load(experts, _names(checkpoint_format, ckpt, generator))

    def dequant(e: int, proj: str) -> torch.Tensor:
        weight, scale = ckpt[e, proj]
        if fmt is MEGA_MOE_BF16:
            return weight.cuda().float()
        values, exponent = requantize_experts(
            weight.cuda(), scale.cuda(), checkpoint_format, fmt
        )
        return _cast_back(values, exponent, fmt)

    def round_activation(x: torch.Tensor) -> torch.Tensor:
        # MX experts take MXFP8 activations; BF16 experts take BF16.
        if fmt is MEGA_MOE_BF16:
            return x.bfloat16().float()
        values, sf = _cast(MEGA_MOE_MXFP8)(x.bfloat16(), use_ue8m0=True, gran_k=32)
        return _cast_back(values, _exponent(sf), MEGA_MOE_MXFP8)

    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, HIDDEN, device="cuda").bfloat16()
    scores = torch.randn(num_tokens, num_experts, device="cuda")
    topk_weights, topk_ids = torch.topk(scores.softmax(-1), top_k, dim=-1)
    output = experts(hidden_states, topk_weights, topk_ids)

    x = round_activation(hidden_states.float())
    expected = torch.zeros(num_tokens, HIDDEN, device="cuda")
    for e in range(num_experts):
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        gate = x[token] @ dequant(e, "gate_proj").t()
        up = x[token] @ dequant(e, "up_proj").t()
        gate, up = gate.bfloat16().float(), up.bfloat16().float()
        act = F.silu(gate) * up * topk_weights[token, slot, None]
        expected.index_add_(
            0, token, round_activation(act) @ dequant(e, "down_proj").t()
        )

    assert output.dtype == torch.bfloat16
    x, y = output.double(), expected.double()
    assert (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item() < 1e-3
    assert F.cosine_similarity(output.float(), expected, dim=-1).min() > 0.99
