# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Qwen4Exp block-FP8 experts requantized for MegaMoE."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
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
    mega_moe_weight_format,
    requantize_fp8_block,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

HIDDEN, INTER = 512, 256
FORMATS = [
    pytest.param(MEGA_MOE_MXFP8, id="mxfp8"),
    pytest.param(MEGA_MOE_MXFP4, id="mxfp4"),
]
requires_deep_gemm = pytest.mark.skipif(
    not is_deep_gemm_supported(), reason="Requires DeepGEMM"
)


def _block_fp8(rows: int, cols: int, generator):
    weight = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(rows // 128, cols // 128, generator=generator) * 1e-2 + 1e-3
    return weight, scale


def _dequant_block(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    return weight.float() * scale.to(weight.device)


def _dequant_mx(values: torch.Tensor, exponent: torch.Tensor, fmt) -> torch.Tensor:
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp4,
        cast_back_from_fp8,
    )

    sf = (exponent.to(torch.int32) << 23).view(torch.float32)
    cast_back = cast_back_from_fp4 if fmt is MEGA_MOE_MXFP4 else cast_back_from_fp8
    return cast_back(values, sf, gran_k=32)


def _quant_config(ignored_layers=None, **online) -> SimpleNamespace:
    config = Fp8Config(weight_block_size=[128, 128], ignored_layers=ignored_layers)
    if online:
        config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(**online)
        )
    return SimpleNamespace(quant_config=config)


def test_weight_format_follows_quantization_config():
    prefix, other = "model.layers.3.mlp.experts", "model.layers.47.mlp.experts"
    assert mega_moe_weight_format(_quant_config(), prefix) is MEGA_MOE_MXFP8
    config = _quant_config(moe="mxfp4", ignore=["*.layers.47.*"])
    assert mega_moe_weight_format(config, prefix) is MEGA_MOE_MXFP4
    assert mega_moe_weight_format(config, other) is MEGA_MOE_MXFP8
    # Only the MXFP4 layer shows in the online quantization summary.
    assert list(config.quant_config.online_quantization_config.quantized_layers) == [
        prefix
    ]
    config = _quant_config(targets={"*.layers.3.mlp.experts": "mxfp4"})
    assert mega_moe_weight_format(config, prefix) is MEGA_MOE_MXFP4
    assert mega_moe_weight_format(config, other) is MEGA_MOE_MXFP8
    config = _quant_config(moe="mxfp8")
    assert mega_moe_weight_format(config, prefix) is MEGA_MOE_MXFP8
    with pytest.raises(NotImplementedError, match="mxfp8 or mxfp4"):
        mega_moe_weight_format(_quant_config(moe="fp8_per_tensor"), prefix)
    with pytest.raises(NotImplementedError, match="block-FP8"):
        mega_moe_weight_format(SimpleNamespace(quant_config=None), prefix)
    with pytest.raises(NotImplementedError, match="leaves .* unquantized"):
        mega_moe_weight_format(_quant_config(ignored_layers=[prefix]), prefix)


@requires_deep_gemm
@pytest.mark.parametrize("fmt", FORMATS)
def test_requantize_matches_deep_gemm_cast(fmt):
    from vllm.third_party.deep_gemm.utils.math import (
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    weight, scale = _block_fp8(INTER, HIDDEN, torch.Generator().manual_seed(1))
    weight, scale = weight.cuda(), scale.cuda()
    cast = per_token_cast_to_fp4 if fmt is MEGA_MOE_MXFP4 else per_token_cast_to_fp8
    values, exponent = requantize_fp8_block(weight, scale, fmt)
    ref_values, ref_sf = cast(_dequant_block(weight, scale), use_ue8m0=True, gran_k=32)
    assert torch.equal(values.view(torch.uint8), ref_values.view(torch.uint8))
    assert torch.equal(exponent, (ref_sf.view(torch.int32) >> 23).to(torch.uint8))


@pytest.fixture
def any_device_mega_moe(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


def _experts(cls, fmt, num_local_experts, experts_start_idx, num_logical_experts):
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    kwargs = dict(
        weight_format=fmt,
        num_experts=num_logical_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=experts_start_idx,
        num_logical_experts=num_logical_experts,
        top_k=2,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
    )
    if cls is DeepGemmMegaMoEExperts:
        kwargs["activation"] = MegaMoEActivation()
    with torch.device("cuda"):
        return cls(vllm_config, **kwargs)


def _checkpoint(num_experts: int, generator):
    shapes = {
        "gate_proj": (INTER, HIDDEN),
        "up_proj": (INTER, HIDDEN),
        "down_proj": (HIDDEN, INTER),
    }
    return {
        (e, proj): _block_fp8(*shape, generator)
        for e in range(num_experts)
        for proj, shape in shapes.items()
    }


def _shuffled_names(ckpt, generator) -> list[tuple[int, str, str]]:
    names = [(e, p, k) for (e, p) in ckpt for k in ("weight", "weight_scale_inv")]
    return [names[i] for i in torch.randperm(len(names), generator=generator)]


@requires_deep_gemm
@pytest.mark.parametrize("fmt", FORMATS)
def test_experts_load_block_fp8_checkpoint_into_kernel_layout(any_device_mega_moe, fmt):
    """Shards in any order, weight and scale apart, only owned experts kept."""
    generator = torch.Generator().manual_seed(2)
    ckpt = _checkpoint(4, generator)
    # EP rank 1 of 2 owns logical experts 2 and 3.
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 2, 4)
    loaded = experts.load_weights(
        (f"{e}.{p}.{k}", ckpt[e, p][k == "weight_scale_inv"])
        for e, p, k in _shuffled_names(ckpt, generator)
    )
    assert loaded == {"w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"}
    experts.quant_method.process_weights_after_loading(experts)

    expected = _experts(DeepGemmMegaMoEExperts, fmt, 2, 2, 4)
    shard_ids = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for (e, proj), (weight, scale) in ckpt.items():
        values, exponent = requantize_fp8_block(weight.cuda(), scale.cuda(), fmt)
        prefix = "w2" if proj == "down_proj" else "w13"
        for suffix, value in (("weight", values), ("weight_scale", exponent)):
            name = f"{prefix}_{suffix}"
            expected.weight_loader(
                getattr(expected, name), value, name, shard_ids[proj], e
            )
    for name in loaded:
        assert torch.equal(getattr(experts, name), getattr(expected, name)), name
    assert torch.count_nonzero(experts.w13_weight.view(torch.uint8)) > 0


def _load(experts, ckpt, names) -> None:
    experts.load_weights(
        (f"{e}.{p}.{k}", ckpt[e, p][k == "weight_scale_inv"]) for e, p, k in names
    )
    flush_mega_moe_requantize(experts)


def _assert_same_experts(a, b) -> None:
    for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name


@requires_deep_gemm
def test_weight_and_scale_may_arrive_in_separate_calls(any_device_mega_moe):
    """Chunked weight updates may split an expert's weight from its scale and
    a rank's experts across calls; a pair still incomplete fails the load."""
    generator = torch.Generator().manual_seed(3)
    ckpt = _checkpoint(2, generator)
    names = _shuffled_names(ckpt, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, MEGA_MOE_MXFP8, 2, 0, 2)
    first = [n for n in names if n[0] == 0 or n[2] == "weight"]
    _load(experts, ckpt, first)
    with pytest.raises(ValueError, match="without both weight and scale"):
        experts.quant_method.process_weights_after_loading(experts)
    _load(experts, ckpt, [n for n in names if n not in first])
    experts.quant_method.process_weights_after_loading(experts)

    expected = _experts(Qwen4ExpMegaMoEExperts, MEGA_MOE_MXFP8, 2, 0, 2)
    _load(expected, ckpt, names)
    _assert_same_experts(experts, expected)


@requires_deep_gemm
@pytest.mark.parametrize("fmt", FORMATS)
def test_layerwise_reload_replaces_routed_experts(any_device_mega_moe, fmt):
    """reload_weights / weight transfer put the experts on meta and wrap their
    loaders; the new experts must land in the original kernel tensors."""
    generator = torch.Generator().manual_seed(5)
    old, new = _checkpoint(2, generator), _checkpoint(2, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 0, 2)
    with torch.device("cuda"):
        record_metadata_for_reloading(experts)
    _load(experts, old, _shuffled_names(old, generator))
    experts.quant_method.process_weights_after_loading(experts)
    pointers = {n: p.data_ptr() for n, p in experts.named_parameters()}

    initialize_layerwise_reload(experts)
    _load(experts, new, _shuffled_names(new, generator))
    finalize_layerwise_reload(experts, None)

    expected = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 0, 2)
    _load(expected, new, _shuffled_names(new, generator))
    _assert_same_experts(experts, expected)
    # CUDA graphs keep pointing at the same storage.
    assert {n: p.data_ptr() for n, p in experts.named_parameters()} == pointers


@pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(100) and is_deep_gemm_supported()
    ),
    reason="DeepGEMM MegaMoE kernels require SM100 and DeepGEMM",
)
@pytest.mark.parametrize("fmt", FORMATS)
def test_loaded_experts_match_dequantized_reference(dist_init, fmt):
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp8,
        per_token_cast_to_fp8,
    )

    num_experts, top_k, num_tokens = 8, 2, 48
    generator = torch.Generator().manual_seed(4)
    ckpt = _checkpoint(num_experts, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, num_experts, 0, num_experts)
    _load(experts, ckpt, _shuffled_names(ckpt, generator))

    def dequant(e: int, proj: str) -> torch.Tensor:
        weight, scale = (t.cuda() for t in ckpt[e, proj])
        values, exponent = requantize_fp8_block(weight, scale, fmt)
        return _dequant_mx(values, exponent, fmt)

    def mxfp8_round_trip(x: torch.Tensor) -> torch.Tensor:
        q = per_token_cast_to_fp8(x.bfloat16(), use_ue8m0=True, gran_k=32)
        return cast_back_from_fp8(*q, gran_k=32)

    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, HIDDEN, device="cuda").bfloat16()
    scores = torch.randn(num_tokens, num_experts, device="cuda")
    topk_weights, topk_ids = torch.topk(scores.softmax(-1), top_k, dim=-1)
    output = experts(hidden_states, topk_weights, topk_ids)

    x = mxfp8_round_trip(hidden_states.float())
    expected = torch.zeros(num_tokens, HIDDEN, device="cuda")
    for e in range(num_experts):
        token, slot = (topk_ids == e).nonzero(as_tuple=True)
        gate = x[token] @ dequant(e, "gate_proj").t()
        up = x[token] @ dequant(e, "up_proj").t()
        gate, up = gate.bfloat16().float(), up.bfloat16().float()
        act = F.silu(gate) * up * topk_weights[token, slot, None]
        expected.index_add_(
            0, token, mxfp8_round_trip(act) @ dequant(e, "down_proj").t()
        )

    assert output.dtype == torch.bfloat16
    x, y = output.double(), expected.double()
    assert (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item() < 1e-3
    assert F.cosine_similarity(output.float(), expected, dim=-1).min() > 0.99
