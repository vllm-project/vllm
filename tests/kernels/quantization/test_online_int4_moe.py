# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online `int4_per_group_32` MoE (`Int4PerGroup32OnlineMoEMethod`), which is
ROCm-only: block-FP8, MXFP4 (Kimi-K3) and BF16 checkpoint experts quantized for
the selected WNA16 backend, AITER's FlyDSL a16wi4 kernels on gfx942 and Triton
(or int4 emulation for SiTU) otherwise. The quantizer and the AITER backend
checks run on any platform."""

import dataclasses
import importlib.util
import json
import math

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.load import LoadConfig
from vllm.config.model import ModelConfig
from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.oracle import int_wna16
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    WNA16WeightFormat,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.int4 import (
    Int4PerGroup32OnlineMoEMethod,
    quantize_int4_per_group_sym,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    dequant_block_fp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequantize_mxfp4_e8m0,
)
from vllm.model_executor.model_loader import weight_utils
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


def _on_aiter_gfx942() -> bool:
    if not current_platform.is_rocm() or importlib.util.find_spec("aiter") is None:
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


AITER_GFX942 = _on_aiter_gfx942()
requires_rocm = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="online int4 MoE is ROCm-only"
)
requires_aiter_gfx942 = pytest.mark.skipif(
    not AITER_GFX942, reason="requires AITER on gfx942"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

E2M1_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
FP8_BLOCK = [128, 128]
INT4_ONLINE = {"moe": {"weight": "int4_per_group_32"}}
# Mistral Large 4 routed experts per rank under TP4.
ML4_EXPERTS, ML4_HIDDEN, ML4_INTER, ML4_TOPK = 128, 6144, 1536, 4
# Kimi-K3's intermediate size per rank under TP8, with fewer and narrower experts.
K3_EXPERTS, K3_HIDDEN, K3_INTER, K3_TOPK = 64, 2048, 384, 8
SITU_BETA, SITU_LINEAR_BETA = 1.5, 2.0


def _expected_backend(aiter: bool, activation: str = "silu") -> WNA16MoEBackend:
    if aiter and AITER_GFX942:
        return WNA16MoEBackend.AITER
    # Triton WNA16 has no SiTU.
    return WNA16MoEBackend.TRITON if activation == "silu" else WNA16MoEBackend.EMULATION


def _mxfp4_experts(
    num_experts: int, rows: int, cols: int, scale_range: tuple[int, int] = (117, 123)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random packed MXFP4 experts, every E2M1 code included, with E8M0 block
    scales drawn from `scale_range`."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randint(
        0,
        256,
        (num_experts, rows, cols // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    scale = torch.randint(
        *scale_range,
        (num_experts, rows, cols // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=gen,
    )
    return weight, scale


def _ocp_mxfp4_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode MXFP4 as the OCP MX spec defines it."""
    *batch, rows, packed_cols = weight.shape
    cols = packed_cols * 2
    # Element 2i is in the low nibble and element 2i + 1 in the high nibble.
    codes = torch.stack((weight & 0xF, weight >> 4), dim=-1).view(*batch, rows, cols)
    values = torch.tensor(E2M1_VALUES, device=weight.device)[codes.long()]
    block_scales = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32), scale.int() - 127
    )
    return (
        (values.view(*batch, rows, cols // 32, 32) * block_scales.unsqueeze(-1))
        .view(*batch, rows, cols)
        .to(torch.bfloat16)
    )


def _unpack_uint4b8(packed: torch.Tensor) -> torch.Tensor:
    """int32 words of 8 uint4b8 codes along K, low nibble first -> int8."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    codes = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (codes.flatten(-2) - 8).to(torch.int8)


def _int4_reference_quant(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int4 with one scale per 32 input channels, in the weight dtype:
    scale = amax / 7 and code = round(w / scale), in [-7, 7]."""
    groups = weight.unflatten(-1, (-1, 32))
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7
    codes = torch.round(groups / scale).clamp(-7, 7).to(torch.int8)
    return codes.flatten(-2), scale.squeeze(-1)


def _int4_reference_weight(weight: torch.Tensor) -> torch.Tensor:
    """What the int4 kernels multiply by: the int4 codes times their scales."""
    codes, scale = _int4_reference_quant(weight)
    return codes.float() * scale.float().repeat_interleave(32, dim=-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_quantizer_matches_torch_reference(dtype):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    weight = torch.randn(4, 64, 512, device=DEVICE, generator=gen).to(dtype)
    # An all-zero group exercises the scale clamp.
    weight[0, 0, :32] = 0

    packed, scale = quantize_int4_per_group_sym(weight, 32)

    ref_codes, ref_scale = _int4_reference_quant(weight)
    assert packed.dtype == torch.int32 and packed.shape == (4, 64, 512 // 8)
    assert scale.dtype == dtype and torch.equal(scale, ref_scale)
    codes = _unpack_uint4b8(packed)
    assert torch.equal(codes, ref_codes)
    assert not codes[0, 0, :32].any()
    # Round to nearest, up to the rounding of the weight dtype's math.
    step = scale.float().repeat_interleave(32, dim=-1)
    error = (codes.float() * step - weight.float()).abs()
    assert (error <= step * (0.5 + 2**-5)).all()


def test_online_int4_moe_is_rocm_only(monkeypatch):
    moe = make_dummy_moe_config()
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    with pytest.raises(NotImplementedError, match="only supported on ROCm"):
        Int4PerGroup32OnlineMoEMethod(moe=moe)


@requires_aiter_gfx942
def test_quantizer_matches_aiter_per_1x32_i4_quant():
    from aiter.ops.quant import per_1x32_i4_quant

    gen = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(
        8, 256, 1024, dtype=torch.bfloat16, device="cuda", generator=gen
    )
    # An all-zero group exercises the scale clamp.
    weight[0, 0, :32] = 0

    packed, scale = quantize_int4_per_group_sym(weight, 32)

    ref_codes, ref_scale = per_1x32_i4_quant(weight)
    assert packed.dtype == torch.int32 and packed.shape == (8, 256, 1024 // 8)
    assert torch.equal(_unpack_uint4b8(packed), ref_codes.view(torch.int8))
    assert torch.equal(scale, ref_scale.transpose(-1, -2))


@requires_aiter_gfx942
def test_mxfp4_dequant_matches_aiter_fp4_utils():
    from aiter.utility import fp4_utils

    # Every finite E8M0 scale; 255 is NaN.
    weight, scale = _mxfp4_experts(4, 256, 1024, scale_range=(0, 255))

    out = dequantize_mxfp4_e8m0(weight, scale, torch.bfloat16)

    ref = (
        fp4_utils.mxfp4_to_f32(weight.view(torch.float4_e2m1fn_x2)).view(4, 256, 32, 32)
        * fp4_utils.e8m0_to_f32(scale.view(torch.float8_e8m0fnu)).unsqueeze(-1)
    ).view(4, 256, 1024)
    assert torch.equal(out, ref.to(torch.bfloat16))


def _reference_kimi_requant(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The previous in-method Kimi-K3 gfx942 conversion: AITER quantizes the
    dequantized experts, then shuffles and packs them for a16wi4."""
    from aiter.ops.quant import per_1x32_i4_quant
    from aiter.ops.shuffle import (
        pack_int8_to_packed_int4,
        shuffle_scale_for_int4,
        shuffle_weight,
    )

    num_experts, rows, packed_cols = weight.shape
    weight_int4, weight_scale = per_1x32_i4_quant(_ocp_mxfp4_to_bf16(weight, scale))
    packed = pack_int8_to_packed_int4(shuffle_weight(weight_int4, (16, 16)))
    packed_scale = shuffle_scale_for_int4(weight_scale, group_size=32)
    return (
        packed.view(torch.uint8).view(num_experts, rows, packed_cols),
        packed_scale.reshape(-1),
    )


@requires_aiter_gfx942
@pytest.mark.parametrize(
    "experts_per_chunk",
    [1, 5, 16],
    ids=["one_expert", "uneven_last_chunk", "single_chunk"],
)
def test_mxfp4_requant_matches_previous_kimi_path(monkeypatch, experts_per_chunk):
    num_experts, rows, cols = 16, 256, 2048
    monkeypatch.setattr(
        int_wna16, "_AITER_REPACK_CHUNK_VALUES", experts_per_chunk * rows * cols
    )
    weight, scale = _mxfp4_experts(num_experts, rows, cols)

    packed, int4_scale = quantize_int4_per_group_sym(
        dequantize_mxfp4_e8m0(weight, scale, torch.bfloat16), 32
    )
    out_weight, out_scale = int_wna16._aiter_int4_repack(packed, int4_scale)

    ref_weight, ref_scale = _reference_kimi_requant(weight, scale)
    assert out_scale.shape[0] == num_experts
    assert torch.equal(out_weight.view(torch.uint8), ref_weight)
    assert torch.equal(out_scale.reshape(-1), ref_scale)


@pytest.mark.parametrize(
    ("quant_config", "hidden", "reason"),
    [
        (WNA16WeightFormat(num_bits=4, group_size=32), 2048, None),
        (WNA16WeightFormat(num_bits=8, group_size=32), 2048, "group size 32"),
        (WNA16WeightFormat(num_bits=4, group_size=128), 2048, "group size 32"),
        (WNA16WeightFormat(num_bits=4, group_size=32), 2880, "multiples of 128"),
    ],
)
def test_aiter_backend_takes_only_online_int4_layouts(quant_config, hidden, reason):
    moe = make_dummy_moe_config(num_experts=8, hidden_dim=hidden, intermediate_size=384)
    out = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.AITER, moe, quant_config, False, False, True
    )
    if reason is None:
        assert out is None
    else:
        assert out is not None and reason in out


def test_aiter_backend_rejects_bias_and_fp16():
    weight_format = WNA16WeightFormat(num_bits=4, group_size=32)
    moe = make_dummy_moe_config(num_experts=8, hidden_dim=2048, intermediate_size=384)
    bias_reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.AITER, moe, weight_format, False, True, True
    )
    assert bias_reason is not None and "bias" in bias_reason
    fp16_moe = make_dummy_moe_config(
        num_experts=8, hidden_dim=2048, intermediate_size=384, in_dtype=torch.float16
    )
    dtype_reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.AITER, fp16_moe, weight_format, False, False, True
    )
    assert dtype_reason is not None and "bfloat16" in dtype_reason


@pytest.mark.parametrize(
    ("activation", "pad", "supported"),
    [
        (MoEActivation.SILU, {}, True),
        (MoEActivation.SILU, {"intermediate_size_per_partition_unpadded": 300}, False),
        (MoEActivation.SILU, {"intermediate_pad": 128}, False),
        (MoEActivation.SILU, {"hidden_dim_unpadded": 1920}, False),
        (MoEActivation.SITU, {"intermediate_size_per_partition_unpadded": 300}, True),
        (MoEActivation.SITU, {"hidden_dim_unpadded": 1920}, True),
    ],
    ids=[
        "silu",
        "silu_padded_intermediate",
        "silu_intermediate_pad_override",
        "silu_padded_hidden",
        "situ_padded_intermediate",
        "situ_padded_hidden",
    ],
)
def test_aiter_backend_rejects_padded_non_situ_layers(activation, pad, supported):
    """At TP1 rocm_aiter_fused_experts would double a non-SiTU int4 layer's
    intermediate pad, so such padded layers fall back to another backend."""
    moe = dataclasses.replace(
        make_dummy_moe_config(
            num_experts=8, hidden_dim=2048, intermediate_size=384, activation=activation
        ),
        **pad,
    )
    reason = int_wna16._backend_incompatibility_reason(
        WNA16MoEBackend.AITER,
        moe,
        WNA16WeightFormat(num_bits=4, group_size=32),
        False,
        False,
        True,
    )
    if supported:
        assert reason is None
    else:
        assert reason is not None and "padded" in reason


def _fp8_block_experts(
    num_experts: int, rows: int, cols: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random experts stored like a block-FP8 checkpoint: e4m3fn values with
    fp32 128x128 block scales (`weight_scale_inv`)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bn, bk = FP8_BLOCK
    weight = torch.empty(
        num_experts, rows, cols, dtype=torch.float8_e4m3fn, device="cuda"
    )
    scale_inv = torch.empty(
        num_experts,
        math.ceil(rows / bn),
        math.ceil(cols / bk),
        dtype=torch.float32,
        device="cuda",
    )
    for e in range(num_experts):
        w = torch.randn(rows, cols, device="cuda", generator=gen) / math.sqrt(cols)
        padded = F.pad(w, (0, -cols % bk, 0, -rows % bn))
        blocks = padded.view(padded.shape[0] // bn, bn, padded.shape[1] // bk, bk)
        scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
        scale_inv[e] = scale
        expanded = scale.repeat_interleave(bn, 0).repeat_interleave(bk, 1)
        weight[e] = (w / expanded[:rows, :cols]).to(torch.float8_e4m3fn)
    return weight, scale_inv


def _bf16_experts(num_experts: int, rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(num_experts, rows, cols, device="cuda", generator=gen)
    return (weight / math.sqrt(cols)).to(torch.bfloat16)


def _fp8_checkpoint_quant_config(tmp_path, vllm_config) -> Fp8Config:
    """The quant config the model loader builds for a block-FP8 checkpoint
    served with `--quantization-config.moe.weight int4_per_group_32`."""
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": FP8_BLOCK,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(hf_config))
    vllm_config.model_config = ModelConfig(
        model=str(tmp_path),
        quantization_config=resolve_quantization_config(None, INT4_ONLINE),
    )
    quant_config = weight_utils.get_quant_config(vllm_config.model_config, LoadConfig())
    assert isinstance(quant_config, Fp8Config)
    return quant_config


def _mxfp4_checkpoint_quant_config() -> Mxfp4Config:
    """An MXFP4 checkpoint config (Kimi-K3's experts) composed with
    `--quantization-config.moe.weight int4_per_group_32`, as the loader does."""
    quant_config = Mxfp4Config()
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        resolve_quantization_config(None, INT4_ONLINE)
    )
    return quant_config


def _moe_runner(
    quant_config: QuantizationConfig,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str = "silu",
):
    situ = (
        {
            "activation_situ_beta": SITU_BETA,
            "activation_situ_linear_beta": SITU_LINEAR_BETA,
        }
        if activation == "situ"
        else {}
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        return FusedMoEFactory(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="model.layers.0.mlp.experts",
            activation=activation,
            **situ,
        )


def _load_experts(runner, w13, w2, scales=None) -> None:
    """Load stacked `[E, 2I, ...]` / `[E, H, ...]` experts as checkpoint shards;
    `scales` is `(name, w13_scale, w2_scale)` for a quantized checkpoint."""
    tensors = {"weight": (w13, w2)}
    if scales is not None:
        name, w13_s, w2_s = scales
        tensors[name] = (w13_s, w2_s)
    checkpoint = []
    for e in range(w13.shape[0]):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for name, (t13, t2) in tensors.items():
                gate, up = t13[e].chunk(2)
                shard = {"gate_proj": gate, "up_proj": up, "down_proj": t2[e]}[proj]
                checkpoint.append((f"{e}.{proj}.{name}", shard))
    list(runner.load_weights(checkpoint))


def _reference_moe(x, w13, w2, topk_weights, topk_ids, activation: str):
    """Float32 MoE over the int4 weights, dequantizing each routed expert on
    use. `w13`/`w2` map an expert to its bf16 source weight."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in topk_ids.unique().tolist():
        tokens, slots = (topk_ids == e).nonzero(as_tuple=True)
        gate_up = x[tokens].float() @ _int4_reference_weight(w13(e)).t()
        gate, up = gate_up.chunk(2, -1)
        if activation == "situ":
            gate = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
            up = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
        else:
            gate = F.silu(gate)
        y = (gate * up) @ _int4_reference_weight(w2(e)).t()
        out.index_add_(0, tokens, y * topk_weights[tokens, slots, None])
    return out


def _routed_inputs(num_tokens: int, hidden: int, num_experts: int, top_k: int):
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(
        num_tokens, hidden, device="cuda", generator=gen, dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, num_experts, device="cuda", generator=gen)
    topk_weights, topk_ids = torch.topk(logits.softmax(-1), top_k, dim=-1)
    return x, topk_weights / topk_weights.sum(-1, keepdim=True), topk_ids


def _assert_matches(runner, ref_fn, x, topk_weights, topk_ids) -> None:
    layer = runner.routed_experts
    out = runner._quant_method.apply(layer, x, topk_weights, topk_ids, None, None)
    assert isinstance(out, torch.Tensor) and out.shape == x.shape
    ref = ref_fn(x, topk_weights, topk_ids)
    assert ((out.float() - ref).norm() / ref.norm()).item() < 0.02


@pytest.fixture
def set_aiter_moe(monkeypatch):
    def set_enabled(enabled: bool) -> None:
        for name in ("VLLM_ROCM_USE_AITER", "VLLM_ROCM_USE_AITER_MOE"):
            monkeypatch.setenv(name, "1" if enabled else "0")
        rocm_aiter_ops.refresh_env_variables()

    yield set_enabled
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@requires_rocm
@pytest.mark.parametrize(
    ("aiter", "num_tokens"),
    [(True, 1), (True, 16), (True, 256), (True, 4096), (False, 16)],
)
def test_fp8_block_checkpoint_moe_matches_reference(
    aiter,
    num_tokens,
    tmp_path,
    default_vllm_config,
    dist_init,
    workspace_init,
    set_aiter_moe,
):
    """Mistral Large 4's TP4 routed experts (SiLU, top-4 of 128) loaded from a
    block-FP8 checkpoint, against a reference using the same int4 weights."""
    set_aiter_moe(aiter)
    quant_config = _fp8_checkpoint_quant_config(tmp_path, default_vllm_config)
    runner = _moe_runner(quant_config, ML4_EXPERTS, ML4_TOPK, ML4_HIDDEN, ML4_INTER)
    method = runner._quant_method
    assert type(method) is Int4PerGroup32OnlineMoEMethod
    assert method.wna16_backend == _expected_backend(aiter)
    w13, w13_s = _fp8_block_experts(ML4_EXPERTS, 2 * ML4_INTER, ML4_HIDDEN, seed=1)
    w2, w2_s = _fp8_block_experts(ML4_EXPERTS, ML4_HIDDEN, ML4_INTER, seed=2)
    _load_experts(runner, w13, w2, ("weight_scale_inv", w13_s, w2_s))

    # The layerwise loader re-quantizes the experts once all of them are loaded.
    layer = runner.routed_experts
    assert not hasattr(layer, "w13_weight_scale_inv")
    if method.wna16_backend == WNA16MoEBackend.AITER:
        assert layer.w13_weight.dtype == torch.int4
        assert layer.w13_weight.shape == (ML4_EXPERTS, 2 * ML4_INTER, ML4_HIDDEN // 2)

    def ref_fn(x, topk_weights, topk_ids):
        return _reference_moe(
            x,
            lambda e: dequant_block_fp8(w13[e], w13_s[e], FP8_BLOCK, torch.bfloat16),
            lambda e: dequant_block_fp8(w2[e], w2_s[e], FP8_BLOCK, torch.bfloat16),
            topk_weights,
            topk_ids,
            "silu",
        )

    _assert_matches(
        runner,
        ref_fn,
        *_routed_inputs(num_tokens, ML4_HIDDEN, ML4_EXPERTS, ML4_TOPK),
    )


@requires_rocm
@pytest.mark.parametrize("aiter", [True, False])
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("num_tokens", [1, 256])
def test_mxfp4_checkpoint_moe_matches_reference(
    aiter,
    activation,
    num_tokens,
    default_vllm_config,
    dist_init,
    workspace_init,
    set_aiter_moe,
):
    """MXFP4 checkpoint experts at Kimi-K3's TP8 intermediate size, with
    Kimi-K3's SiTU and with SiLU."""
    set_aiter_moe(aiter)
    runner = _moe_runner(
        _mxfp4_checkpoint_quant_config(),
        K3_EXPERTS,
        K3_TOPK,
        K3_HIDDEN,
        K3_INTER,
        activation=activation,
    )
    method = runner._quant_method
    assert type(method) is Int4PerGroup32OnlineMoEMethod
    assert method.wna16_backend == _expected_backend(aiter, activation)
    w13, w13_s = _mxfp4_experts(K3_EXPERTS, 2 * K3_INTER, K3_HIDDEN)
    w2, w2_s = _mxfp4_experts(K3_EXPERTS, K3_HIDDEN, K3_INTER)
    _load_experts(runner, w13, w2, ("weight_scale", w13_s, w2_s))

    def ref_fn(x, topk_weights, topk_ids):
        return _reference_moe(
            x,
            lambda e: _ocp_mxfp4_to_bf16(w13[e], w13_s[e]),
            lambda e: _ocp_mxfp4_to_bf16(w2[e], w2_s[e]),
            topk_weights,
            topk_ids,
            activation,
        )

    _assert_matches(
        runner, ref_fn, *_routed_inputs(num_tokens, K3_HIDDEN, K3_EXPERTS, K3_TOPK)
    )


@requires_rocm
@pytest.mark.parametrize(("aiter", "num_tokens"), [(True, 1), (True, 256), (False, 16)])
def test_bf16_checkpoint_moe_matches_reference(
    aiter, num_tokens, default_vllm_config, dist_init, workspace_init, set_aiter_moe
):
    """BF16 checkpoint experts, quantized with
    `--quantization-config.moe.weight int4_per_group_32` alone."""
    set_aiter_moe(aiter)
    quant_config = OnlineQuantizationConfig(
        resolve_quantization_config(None, INT4_ONLINE)
    )
    runner = _moe_runner(quant_config, K3_EXPERTS, K3_TOPK, K3_HIDDEN, K3_INTER)
    method = runner._quant_method
    assert type(method) is Int4PerGroup32OnlineMoEMethod
    assert method.wna16_backend == _expected_backend(aiter)
    w13 = _bf16_experts(K3_EXPERTS, 2 * K3_INTER, K3_HIDDEN, seed=1)
    w2 = _bf16_experts(K3_EXPERTS, K3_HIDDEN, K3_INTER, seed=2)
    _load_experts(runner, w13, w2)

    def ref_fn(x, topk_weights, topk_ids):
        return _reference_moe(
            x, lambda e: w13[e], lambda e: w2[e], topk_weights, topk_ids, "silu"
        )

    _assert_matches(
        runner, ref_fn, *_routed_inputs(num_tokens, K3_HIDDEN, K3_EXPERTS, K3_TOPK)
    )


@requires_rocm
def test_fp8_requant_env_conflicts_with_online_int4(
    tmp_path, monkeypatch, default_vllm_config, dist_init, set_aiter_moe
):
    set_aiter_moe(True)
    monkeypatch.setenv("VLLM_FP8_MOE_REQUANT", "mxfp4")
    quant_config = _fp8_checkpoint_quant_config(tmp_path, default_vllm_config)
    with pytest.raises(ValueError, match="conflicts"):
        _moe_runner(quant_config, 8, 2, 256, 256)
