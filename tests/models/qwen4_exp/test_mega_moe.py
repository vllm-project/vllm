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
    dequantize_experts,
    flush_mega_moe_requantize,
    mega_moe_expert_formats,
    requantize_experts,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

HIDDEN, INTER = 512, 256
# (checkpoint format, MegaMoE format): native loads and every requantization.
CASES = [
    pytest.param("fp8_block", MEGA_MOE_MXFP8, id="fp8_block-mxfp8"),
    pytest.param("fp8_block", MEGA_MOE_MXFP4, id="fp8_block-mxfp4"),
    pytest.param("mxfp8", MEGA_MOE_MXFP8, id="mxfp8-native"),
    pytest.param("mxfp8", MEGA_MOE_MXFP4, id="mxfp8-mxfp4"),
    pytest.param("mxfp4", MEGA_MOE_MXFP4, id="mxfp4-native"),
    pytest.param("mxfp4", MEGA_MOE_MXFP8, id="mxfp4-mxfp8"),
    pytest.param("bf16", MEGA_MOE_BF16, id="bf16-native"),
    pytest.param("bf16", MEGA_MOE_MXFP8, id="bf16-mxfp8"),
    pytest.param("bf16", MEGA_MOE_MXFP4, id="bf16-mxfp4"),
]
SCALE_NAMES = {
    "fp8_block": "weight_scale_inv",
    "mxfp8": "weight_scale",
    "mxfp4": "weight_scale",
}
requires_deep_gemm = pytest.mark.skipif(
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


def _shard(checkpoint_format: str, rows: int, cols: int, generator):
    """One projection of one expert as the checkpoint stores it."""
    x = torch.randn(rows, cols, generator=generator) * 1e-2
    if checkpoint_format == "bf16":
        return x.bfloat16(), None
    if checkpoint_format == "fp8_block":
        weight = torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
        scale = torch.rand(rows // 128, cols // 128, generator=generator) * 1e-2
        return weight, scale + 1e-3
    fmt = MEGA_MOE_MXFP4 if checkpoint_format == "mxfp4" else MEGA_MOE_MXFP8
    values, sf = _cast(fmt)(x, use_ue8m0=True, gran_k=32)
    # Checkpoints store MXFP4 as uint8 and the scales as UE8M0 bytes.
    return values.view(torch.uint8) if fmt is MEGA_MOE_MXFP4 else values, _exponent(sf)


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
    gate_up = torch.stack(
        [
            torch.cat([ckpt[e, "gate_proj"][0], ckpt[e, "up_proj"][0]])
            for e in range(num_experts)
        ]
    )
    down = torch.stack([ckpt[e, "down_proj"][0] for e in range(num_experts)])
    return [("gate_up_proj", gate_up), ("down_proj", down)]


def _quant_config(ignored_layers=None, **online) -> SimpleNamespace:
    config = Fp8Config(weight_block_size=[128, 128], ignored_layers=ignored_layers)
    if online:
        config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(**online)
        )
    return SimpleNamespace(quant_config=config)


def _modelopt_mxfp8_config(**online) -> SimpleNamespace:
    config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    if online:
        config.online_quantization_config = OnlineQuantizationConfig(
            QuantizationConfigArgs(**online)
        )
    return SimpleNamespace(quant_config=config)


@requires_deep_gemm
def test_expert_formats_follow_checkpoint_and_quantization_config():
    prefix, other = "model.layers.3.mlp.experts", "model.layers.47.mlp.experts"
    # Block FP8 has no MegaMoE kernel: MXFP8 unless forced.
    assert mega_moe_expert_formats(_quant_config(), prefix) == (
        "fp8_block",
        MEGA_MOE_MXFP8,
    )
    config = _quant_config(moe="mxfp4", ignore=["*.layers.47.*"])
    assert mega_moe_expert_formats(config, prefix)[1] is MEGA_MOE_MXFP4
    assert mega_moe_expert_formats(config, other)[1] is MEGA_MOE_MXFP8
    # Only the forced layer shows in the online quantization summary.
    assert list(config.quant_config.online_quantization_config.quantized_layers) == [
        prefix
    ]
    config = _quant_config(targets={"*.layers.3.mlp.experts": "mxfp4"})
    assert mega_moe_expert_formats(config, prefix)[1] is MEGA_MOE_MXFP4
    assert mega_moe_expert_formats(config, other)[1] is MEGA_MOE_MXFP8
    with pytest.raises(NotImplementedError, match="mxfp8 or mxfp4"):
        mega_moe_expert_formats(_quant_config(moe="fp8_per_tensor"), prefix)
    # Experts the FP8 checkpoint leaves unquantized are BF16.
    assert mega_moe_expert_formats(_quant_config(ignored_layers=[prefix]), prefix) == (
        "bf16",
        MEGA_MOE_BF16,
    )
    # MXFP8 checkpoints load as stored, or requantize when forced.
    config = _modelopt_mxfp8_config()
    assert mega_moe_expert_formats(config, prefix) == ("mxfp8", MEGA_MOE_MXFP8)
    config = _modelopt_mxfp8_config(moe="mxfp8")
    assert mega_moe_expert_formats(config, prefix) == ("mxfp8", MEGA_MOE_MXFP8)
    assert not config.quant_config.online_quantization_config.quantized_layers
    config = _modelopt_mxfp8_config(moe="mxfp4")
    assert mega_moe_expert_formats(config, prefix) == ("mxfp8", MEGA_MOE_MXFP4)
    # BF16 checkpoints, with and without online quantization.
    bf16 = SimpleNamespace(quant_config=None)
    assert mega_moe_expert_formats(bf16, prefix) == ("bf16", MEGA_MOE_BF16)
    online = SimpleNamespace(
        quant_config=OnlineQuantizationConfig(QuantizationConfigArgs(moe="mxfp4"))
    )
    assert mega_moe_expert_formats(online, prefix) == ("bf16", MEGA_MOE_MXFP4)
    with pytest.raises(NotImplementedError, match="128x128 block"):
        mega_moe_expert_formats(SimpleNamespace(quant_config=Fp8Config()), prefix)


def test_mxfp4_checkpoint_packing_matches_deep_gemm():
    """MXFP4 checkpoints (OCP) pack element 2i in the low nibble, as DeepGEMM."""
    x = torch.tensor([[1.0, 6.0] + [0.0] * 30])
    values, sf = _cast(MEGA_MOE_MXFP4)(x, use_ue8m0=True, gran_k=32)
    # E2M1 codes: 1.0 -> 0b0010, 6.0 -> 0b0111; the block scale is 1.
    assert values.view(torch.uint8)[0, 0].item() == 0x72
    assert _exponent(sf)[0, 0].item() == 127


@requires_deep_gemm
@pytest.mark.parametrize("checkpoint_format,fmt", CASES)
def test_requantize_matches_deep_gemm_cast(checkpoint_format, fmt):
    if checkpoint_format == fmt.name:
        pytest.skip("Experts in a MegaMoE format load as stored")
    weight, scale = _shard(
        checkpoint_format, INTER, HIDDEN, torch.Generator().manual_seed(1)
    )
    weight = weight.cuda()
    scale = None if scale is None else scale.cuda()
    values, exponent = requantize_experts(weight, scale, checkpoint_format, fmt)
    ref_values, ref_sf = _cast(fmt)(
        dequantize_experts(weight, scale, checkpoint_format), use_ue8m0=True, gran_k=32
    )
    assert torch.equal(values.view(torch.uint8), ref_values.view(torch.uint8))
    assert torch.equal(exponent, _exponent(ref_sf))


@pytest.fixture
def any_device_mega_moe(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


def _experts(
    cls,
    fmt,
    num_local_experts,
    experts_start_idx,
    num_logical_experts,
    checkpoint_format="fp8_block",
):
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
        tensors = [("weight", weight)] + (
            [] if scale is None else [("weight_scale", scale)]
        )
        for suffix, value in tensors:
            name = f"{prefix}_{suffix}"
            expected.weight_loader(
                getattr(expected, name), value, name, shard_ids[proj], e
            )
    return expected


def _assert_same_experts(a, b) -> None:
    names = [n for n, _ in b.named_parameters()]
    assert names
    for name in names:
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def _load(experts, names) -> None:
    experts.load_weights(iter(names))
    flush_mega_moe_requantize(experts)


@requires_deep_gemm
@pytest.mark.parametrize("fused", [False, True], ids=["per_expert", "fused"])
@pytest.mark.parametrize("checkpoint_format,fmt", CASES)
def test_experts_load_checkpoint_into_kernel_layout(
    any_device_mega_moe, monkeypatch, checkpoint_format, fmt, fused
):
    """Shards in any order, weight and scale apart, only owned experts kept."""
    if fused and checkpoint_format != "bf16":
        pytest.skip("Only BF16 checkpoints fuse the experts")
    # One expert per requantize call, so batching spans several calls.
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.mega_moe._REQUANTIZE_CHUNK_ELEMENTS",
        INTER * HIDDEN,
    )
    generator = torch.Generator().manual_seed(2)
    ckpt = _checkpoint(checkpoint_format, 4, generator)
    # EP rank 1 of 2 owns logical experts 2 and 3.
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 2, 4, checkpoint_format)
    names = (
        _fused_names(ckpt, 4) if fused else _names(checkpoint_format, ckpt, generator)
    )
    loaded = experts.load_weights(iter(names))
    assert loaded == {n for n, _ in experts.named_parameters()}
    experts.quant_method.process_weights_after_loading(experts)

    expected = _expected(checkpoint_format, fmt, ckpt, 2, 2, 4)
    _assert_same_experts(experts, expected)
    assert torch.count_nonzero(experts.w13_weight.view(torch.uint8)) > 0


@requires_deep_gemm
def test_weight_and_scale_may_arrive_in_separate_calls(any_device_mega_moe):
    """Chunked weight updates may split an expert's weight from its scale and
    a rank's experts across calls; a pair still incomplete fails the load."""
    generator = torch.Generator().manual_seed(3)
    ckpt = _checkpoint("fp8_block", 2, generator)
    names = _names("fp8_block", ckpt, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, MEGA_MOE_MXFP8, 2, 0, 2)
    first = [n for n in names if n[0].startswith("0.") or n[0].endswith(".weight")]
    _load(experts, first)
    with pytest.raises(ValueError, match="without both weight and scale"):
        experts.quant_method.process_weights_after_loading(experts)
    _load(experts, [n for n in names if n not in first])
    experts.quant_method.process_weights_after_loading(experts)
    _assert_same_experts(experts, _expected("fp8_block", MEGA_MOE_MXFP8, ckpt, 2, 0, 2))


@requires_deep_gemm
def test_unexpected_expert_tensor_fails_loudly(any_device_mega_moe):
    experts = _experts(Qwen4ExpMegaMoEExperts, MEGA_MOE_MXFP8, 2, 0, 2)
    with pytest.raises(ValueError, match="unexpected fp8_block expert tensor"):
        experts.load_weights(iter([("0.gate_proj.input_scale", torch.ones(1))]))


@requires_deep_gemm
@pytest.mark.parametrize(
    "checkpoint_format,fmt",
    [
        pytest.param("fp8_block", MEGA_MOE_MXFP8, id="fp8_block-mxfp8"),
        pytest.param("fp8_block", MEGA_MOE_MXFP4, id="fp8_block-mxfp4"),
        pytest.param("mxfp8", MEGA_MOE_MXFP8, id="mxfp8-native"),
        pytest.param("bf16", MEGA_MOE_BF16, id="bf16-native"),
    ],
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
    not (
        current_platform.is_device_capability_family(100) and is_deep_gemm_supported()
    ),
    reason="DeepGEMM MegaMoE kernels require SM100 and DeepGEMM",
)
@pytest.mark.parametrize("checkpoint_format,fmt", CASES)
def test_loaded_experts_match_dequantized_reference(dist_init, checkpoint_format, fmt):
    from vllm.third_party.deep_gemm.utils.math import (
        cast_back_from_fp4,
        cast_back_from_fp8,
    )

    num_experts, top_k, num_tokens = 8, 2, 48
    generator = torch.Generator().manual_seed(4)
    ckpt = _checkpoint(checkpoint_format, num_experts, generator)
    experts = _experts(
        Qwen4ExpMegaMoEExperts, fmt, num_experts, 0, num_experts, checkpoint_format
    )
    _load(experts, _names(checkpoint_format, ckpt, generator))

    def dequant(e: int, proj: str) -> torch.Tensor:
        weight, scale = ckpt[e, proj]
        weight = weight.cuda()
        if fmt is MEGA_MOE_BF16:
            return weight.float()
        scale = None if scale is None else scale.cuda()
        values, exponent = (
            (weight, scale)
            if checkpoint_format == fmt.name
            else requantize_experts(weight, scale, checkpoint_format, fmt)
        )
        assert exponent is not None
        sf = (exponent.to(torch.int32) << 23).view(torch.float32)
        cast_back = cast_back_from_fp4 if fmt is MEGA_MOE_MXFP4 else cast_back_from_fp8
        return cast_back(values, sf, gran_k=32)

    def round_activation(x: torch.Tensor) -> torch.Tensor:
        # MX experts take MXFP8 activations; BF16 experts take BF16.
        if fmt is MEGA_MOE_BF16:
            return x.bfloat16().float()
        values, sf = _cast(MEGA_MOE_MXFP8)(x.bfloat16(), use_ue8m0=True, gran_k=32)
        return cast_back_from_fp8(values, sf, gran_k=32)

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
