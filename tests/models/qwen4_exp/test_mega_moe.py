# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for Qwen4Exp routed experts loaded for MegaMoE."""

from types import SimpleNamespace

import pytest
import torch

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
    _checkpoint_shared_expert_format,
    mega_moe_expert_formats,
)
from vllm.utils.deep_gemm import is_deep_gemm_supported

HIDDEN, INTER = 512, 256
pytestmark = pytest.mark.skipif(
    not is_deep_gemm_supported(), reason="Requires DeepGEMM"
)
dg_math = pytest.importorskip("vllm.third_party.deep_gemm.utils.math")
CASTS = {
    "mxfp8": dg_math.per_token_cast_to_fp8,
    "mxfp4": dg_math.per_token_cast_to_fp4,
}


def _checkpoint(checkpoint_format: str, num_experts: int, generator):
    """{(expert, proj): (weight, scale)} as the checkpoint stores them."""
    shapes = {
        "gate_proj": (INTER, HIDDEN),
        "up_proj": (INTER, HIDDEN),
        "down_proj": (HIDDEN, INTER),
    }
    ckpt = {}
    for e in range(num_experts):
        for proj, (rows, cols) in shapes.items():
            if checkpoint_format == "fp8_block":
                weight = torch.randn(rows, cols, generator=generator)
                scale = torch.rand(rows // 128, cols // 128, generator=generator)
                ckpt[e, proj] = weight.to(torch.float8_e4m3fn), scale * 1e-2 + 1e-3
                continue
            x = torch.randn(rows, cols, generator=generator) * 1e-2
            if checkpoint_format == "bf16":
                ckpt[e, proj] = x.bfloat16(), None
                continue
            values, sf = CASTS[checkpoint_format](x, use_ue8m0=True, gran_k=32)
            # MXFP4 checkpoints store packed uint8 values; scales are UE8M0 bytes.
            if checkpoint_format == "mxfp4":
                values = values.view(torch.uint8)
            ckpt[e, proj] = values, (sf.view(torch.int32) >> 23).to(torch.uint8)
    return ckpt


def _names(checkpoint_format, ckpt, generator) -> list[tuple[str, torch.Tensor]]:
    """Per-expert checkpoint tensors in a random order."""
    names = [(f"{e}.{p}.weight", w) for (e, p), (w, _) in ckpt.items()]
    if checkpoint_format != "bf16":
        scale_name = "weight_scale" if "mx" in checkpoint_format else "weight_scale_inv"
        names += [(f"{e}.{p}.{scale_name}", s) for (e, p), (_, s) in ckpt.items()]
    return [names[i] for i in torch.randperm(len(names), generator=generator)]


def _experts(
    cls,
    fmt,
    num_local,
    start,
    num_logical,
    checkpoint_format=None,
    shared_size=0,
    shared_checkpoint_format=None,
):
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
        kwargs["shared_expert_intermediate_size"] = shared_size
        kwargs["shared_checkpoint_format"] = shared_checkpoint_format
    with torch.device("cuda"):
        return cls(vllm_config, **kwargs)


def _cast(checkpoint_format, fmt_name, weight, scale):
    """A checkpoint tensor requantized with DeepGEMM's own casts."""
    if checkpoint_format == fmt_name:
        return weight, scale
    weight = weight.cuda()
    if checkpoint_format == "bf16":
        x = weight.float()
    elif checkpoint_format == "fp8_block":
        block = scale.cuda().repeat_interleave(128, 0).repeat_interleave(128, 1)
        x = weight.float() * block
    else:
        sf = (scale.cuda().to(torch.int32) << 23).view(torch.float32)
        cast_back = (
            dg_math.cast_back_from_fp4
            if checkpoint_format == "mxfp4"
            else dg_math.cast_back_from_fp8
        )
        x = cast_back(weight, sf, gran_k=32)
    weight, sf = CASTS[fmt_name](x, use_ue8m0=True, gran_k=32)
    return weight, (sf.view(torch.int32) >> 23).to(torch.uint8)


def _expected(checkpoint_format, fmt, ckpt, num_local, start, num_logical):
    """Experts written shard by shard through the generic MegaMoE loader,
    requantized with DeepGEMM's own casts."""
    expected = _experts(DeepGemmMegaMoEExperts, fmt, num_local, start, num_logical)
    shard_ids = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for (e, proj), (weight, scale) in ckpt.items():
        weight, scale = _cast(checkpoint_format, fmt.name, weight, scale)
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
    config = _with_online(
        Fp8Config(weight_block_size=[128, 128]), targets={prefix: "mxfp4"}
    )
    assert mega_moe_expert_formats(config, prefix)[1] is MEGA_MOE_MXFP4
    assert mega_moe_expert_formats(config, other)[1] is MEGA_MOE_MXFP8
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
        ("fp8_block", MEGA_MOE_MXFP8),
        ("mxfp8", MEGA_MOE_MXFP8),
        ("mxfp4", MEGA_MOE_MXFP4),
        ("mxfp8", MEGA_MOE_MXFP4),
        ("mxfp4", MEGA_MOE_MXFP8),
        ("fused_bf16", MEGA_MOE_BF16),
        ("fused_bf16", MEGA_MOE_MXFP4),
    ],
)
def test_experts_load_checkpoint_into_kernel_layout(
    any_device_mega_moe, checkpoint_format, fmt
):
    """Shards in any order, weight and scale apart, only owned experts kept."""
    fused = checkpoint_format == "fused_bf16"
    checkpoint_format = "bf16" if fused else checkpoint_format
    generator = torch.Generator().manual_seed(2)
    ckpt = _checkpoint(checkpoint_format, 4, generator)
    if fused:
        # The official BF16 checkpoint's [E, N, K] experts, gate rows first.
        gate_up = [
            torch.cat([ckpt[e, "gate_proj"][0], ckpt[e, "up_proj"][0]])
            for e in range(4)
        ]
        down = [ckpt[e, "down_proj"][0] for e in range(4)]
        names = [
            ("gate_up_proj", torch.stack(gate_up)),
            ("down_proj", torch.stack(down)),
        ]
    else:
        names = _names(checkpoint_format, ckpt, generator)
    # EP rank 1 of 2 owns logical experts 2 and 3.
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 2, 4, checkpoint_format)
    loaded = experts.load_weights(names)
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
    experts.load_weights(first)
    with pytest.raises(ValueError, match="without both weight and scale"):
        experts.quant_method.process_weights_after_loading(experts)
    experts.load_weights([n for n in names if n not in first])
    experts.quant_method.process_weights_after_loading(experts)
    _assert_same_experts(experts, _expected("fp8_block", MEGA_MOE_MXFP8, ckpt, 2, 0, 2))


def test_layerwise_reload_replaces_routed_experts(any_device_mega_moe):
    """reload_weights / weight transfer put the experts on meta and wrap their
    loaders; the new experts must land in the original kernel tensors."""
    checkpoint_format, fmt = "fp8_block", MEGA_MOE_MXFP4
    generator = torch.Generator().manual_seed(5)
    old = _checkpoint(checkpoint_format, 2, generator)
    new = _checkpoint(checkpoint_format, 2, generator)
    experts = _experts(Qwen4ExpMegaMoEExperts, fmt, 2, 0, 2, checkpoint_format)
    with torch.device("cuda"):
        record_metadata_for_reloading(experts)
    experts.load_weights(_names(checkpoint_format, old, generator))
    experts.quant_method.process_weights_after_loading(experts)
    pointers = {n: p.data_ptr() for n, p in experts.named_parameters()}

    initialize_layerwise_reload(experts)
    experts.load_weights(_names(checkpoint_format, new, generator))
    finalize_layerwise_reload(experts, None)

    _assert_same_experts(experts, _expected(checkpoint_format, fmt, new, 2, 0, 2))
    # CUDA graphs keep pointing at the same storage.
    assert {n: p.data_ptr() for n, p in experts.named_parameters()} == pointers


@pytest.mark.parametrize(
    "checkpoint_format,shared_checkpoint_format,fmt",
    [
        # The official FP8 checkpoint keeps the shared expert in BF16.
        ("fp8_block", "bf16", MEGA_MOE_MXFP8),
        ("fp8_block", "fp8_block", MEGA_MOE_MXFP4),
        ("mxfp8", "mxfp8", MEGA_MOE_MXFP4),
        ("bf16", "bf16", MEGA_MOE_BF16),
        ("bf16", "bf16", MEGA_MOE_MXFP4),
    ],
)
def test_fused_shared_expert_loads_into_kernel_layout(
    any_device_mega_moe, checkpoint_format, shared_checkpoint_format, fmt
):
    """The gated shared expert fuses at the precision of a routed slice (MXFP8
    next to MX experts, else BF16), requantized from its own checkpoint format."""
    generator = torch.Generator().manual_seed(7)
    ckpt = _checkpoint(checkpoint_format, 2, generator)
    shared_ckpt = _checkpoint(shared_checkpoint_format, 1, generator)
    shared = {p: v for (_, p), v in shared_ckpt.items()}
    names = _names(checkpoint_format, ckpt, generator)
    names += [
        (n.replace("0.", "shared.", 1), t)
        for n, t in _names(shared_checkpoint_format, shared_ckpt, generator)
    ]
    names = [names[i] for i in torch.randperm(len(names), generator=generator)]
    experts = _experts(
        Qwen4ExpMegaMoEExperts,
        fmt,
        2,
        0,
        2,
        checkpoint_format,
        shared_size=INTER,
        shared_checkpoint_format=shared_checkpoint_format,
    )
    assert experts.has_fused_shared_experts
    loaded = experts.load_weights(names)
    assert loaded == {n for n, _ in experts.named_parameters()}
    experts.quant_method.process_weights_after_loading(experts)

    expected = _expected(checkpoint_format, fmt, ckpt, 2, 0, 2)
    for name, param in expected.named_parameters():
        assert torch.equal(getattr(experts, name), param), name
    # Expected shared parameters through the generic shared-expert loader.
    shared_fmt = "bf16" if fmt is MEGA_MOE_BF16 else "mxfp8"
    reference = _experts(
        Qwen4ExpMegaMoEExperts,
        fmt,
        2,
        0,
        2,
        checkpoint_format,
        shared_size=INTER,
        shared_checkpoint_format=shared_checkpoint_format,
    )
    shard_ids = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
    for proj, (weight, scale) in shared.items():
        weight, scale = _cast(shared_checkpoint_format, shared_fmt, weight, scale)
        layer = "l2" if proj == "down_proj" else "l1"
        for suffix, value in (("weight", weight), ("weight_scale", scale)):
            if value is not None:
                param = getattr(reference, f"shared_{layer}_{suffix}")
                DeepGemmMegaMoEExperts.shared_weight_loader(
                    reference,
                    param,
                    value.cuda(),
                    f"shared_experts.{shard_ids[proj]}.{suffix}",
                )
    for name, param in reference.named_parameters():
        if name.startswith("shared_"):
            assert torch.equal(getattr(experts, name), param), name


def test_shared_expert_format_follows_checkpoint():
    """FP8 checkpoints may leave the shared expert unquantized."""
    prefix = "model.layers.3.mlp.shared_expert"
    fp8 = Fp8Config(weight_block_size=[128, 128])
    assert _checkpoint_shared_expert_format(fp8, prefix) == "fp8_block"
    # As the official FP8 checkpoint's `modules_to_not_convert` lists it.
    ignored = [f"{prefix}.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
    fp8 = Fp8Config(weight_block_size=[128, 128], ignored_layers=ignored)
    assert _checkpoint_shared_expert_format(fp8, prefix) == "bf16"
    # Partly quantized: not read by the fused loader.
    fp8 = Fp8Config(weight_block_size=[128, 128], ignored_layers=ignored[2:])
    assert _checkpoint_shared_expert_format(fp8, prefix) is None
    assert _checkpoint_shared_expert_format(None, prefix) == "bf16"
    mxfp8 = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    # Not read by the fused loader: the shared expert stays unfused.
    assert _checkpoint_shared_expert_format(mxfp8, prefix) is None


def test_block_routes_stacked_shared_expert_names():
    """The model's weights mapper stacks gate/up into `gate_up_proj` with a
    `shard_id`; a fused block hands them to the experts per projection."""
    block = SimpleNamespace(shared_expert=None)
    seen: list[str] = []

    class _Loader:
        def __init__(self, module):
            pass

        def load_weights(self, weights):
            seen.extend(name for name, _ in weights)
            return set()

    import vllm.models.qwen4_exp.nvidia.mega_moe as mega_moe

    gate, up = torch.zeros(1), torch.zeros(1)
    gate.shard_id, up.shard_id = 0, 1
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mega_moe, "AutoWeightsLoader", _Loader)
        mega_moe.Qwen4ExpMegaMoeBlock.load_weights(
            block,
            [
                ("shared_expert.gate_up_proj.weight", gate),
                ("shared_expert.gate_up_proj.weight_scale_inv", up),
                ("shared_expert.down_proj.weight", torch.zeros(1)),
                ("shared_expert_gate.weight", torch.zeros(1)),
                ("experts.0.gate_proj.weight", torch.zeros(1)),
            ],
        )
    assert seen == [
        "experts.shared.gate_proj.weight",
        "experts.shared.up_proj.weight_scale_inv",
        "experts.shared.down_proj.weight",
        "shared_expert_gate.weight",
        "experts.0.gate_proj.weight",
    ]
