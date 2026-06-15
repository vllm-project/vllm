# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-vs-optimized unit tests for the MiniMax-M3 AMD/ROCm fused kernels.

Each optimized kernel added for the ROCm port has a slow PyTorch reference; the
tests assert the two agree within tolerance:

  * Gemma RMSNorm (plain + fused-add-residual)  -> fp32 PyTorch normalize
  * SwiGLU-OAI (split layout)                    -> fp32 PyTorch elementwise
  * Fused MXFP8 activation quant (Triton)        -> _mxfp8_e4m3_quantize_torch
  * Native MXFP8 linear (CDNA4 dot_scaled)       -> dequant-to-bf16 @ matmul
  * Fused MXFP8 MoE (grouped GEMM)               -> dequant-to-bf16 MoE math

CDNA4 exercises Triton ``dot_scaled`` for linear and MoE. CDNA3 converts MoE
checkpoint bytes to E4M3FNUZ, executes native FP8 partial dots for each 1x32
block, and applies the E8M0 scale product in-register. Dense linear layers
continue to use the faster BF16-at-load hipBLASLt path on CDNA3.

Hardware scope: the whole module is ROCm-only (these are the AMD path; NVIDIA
uses the FlashInfer kernels). Fused MXFP8 MoE tests run on CDNA3 gfx94x and
CDNA4 gfx95x; native MXFP8 linear tests remain CDNA4-only.

Run:  pytest tests/kernels/test_minimax_m3_amd_ops.py -v
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MiniMax-M3 AMD fused ops require ROCm.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (  # noqa: E402
    _mxfp8_e4m3_quantize_torch,
    _mxfp8_e4m3_quantize_triton,
    dequant_mxfp8_to_bf16,
    normalize_mxfp8_e4m3fn_to_e4m3fnuz,
)
from vllm.models.minimax_m3.amd.ops import (  # noqa: E402
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    swiglu_oai_split,
)
from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import _num_warps  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6


def _gcn_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # pragma: no cover - no device / non-AMD
        return ""


# The fused MoE path is selected on MI300/MI325 (gfx94x) and MI350 (gfx95x).
requires_mi3xx = pytest.mark.skipif(
    not any(arch in _gcn_arch() for arch in ("gfx94", "gfx95")),
    reason="fused ROCm MXFP8 requires gfx94x or gfx95x.",
)

requires_gfx950 = pytest.mark.skipif(
    "gfx95" not in _gcn_arch(),
    reason="native MXFP8 linear requires CDNA4 (gfx95x).",
)


@requires_mi3xx
def test_mxfp8_fused_moe_backend_supported():
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        Mxfp8NativeTritonExperts,
    )

    assert Mxfp8NativeTritonExperts._supports_current_device()


@pytest.mark.parametrize(
    ("m_routed", "n", "k", "block_m", "is_gemm2", "expected"),
    [
        (64, 768, 6144, 16, False, (32, 128, 1)),
        (64, 6144, 384, 16, True, (64, 64, 2)),
        (64, 6144, 384, 16, False, (32, 128, 1)),
        (64, 6144, 3072, 16, True, (32, 128, 1)),
        (64, 6144, 384, 32, True, (128, 64, 4)),
        (4096, 6144, 6144, 64, True, (64, 64, 2)),
    ],
)
def test_mxfp8_gfx94x_grouped_gemm_config(
    m_routed,
    n,
    k,
    block_m,
    is_gemm2,
    expected,
):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _gfx94x_grouped_gemm_config,
    )

    assert _gfx94x_grouped_gemm_config(m_routed, n, k, block_m, is_gemm2) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"max_model_len": 10240}, True),
        ({"max_model_len": 0}, False),
        ({"ep_size": 8}, False),
        ({"has_shared_experts": False}, False),
        ({"experts_per_token": 8}, False),
        ({"hidden_dim": 4096}, False),
    ],
)
def test_mxfp8_bf16_decode_fallback_scope(monkeypatch, overrides, expected):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _should_use_bf16_decode_fallback,
    )

    monkeypatch.setattr(
        type(current_platform),
        "is_fp8_fnuz",
        classmethod(lambda cls: True),
    )
    values = {
        "ep_size": 1,
        "has_shared_experts": True,
        "num_experts": 128,
        "experts_per_token": 4,
        "hidden_dim": 6144,
        "intermediate_size": 3072,
        "max_model_len": 2304,
    }
    values.update(overrides)
    assert _should_use_bf16_decode_fallback(SimpleNamespace(**values)) is expected


def test_mxfp8_bf16_decode_fallback_disabled_on_gfx950(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _should_use_bf16_decode_fallback,
    )

    monkeypatch.setattr(
        type(current_platform),
        "is_fp8_fnuz",
        classmethod(lambda cls: False),
    )
    moe_config = SimpleNamespace(
        ep_size=1,
        has_shared_experts=True,
        num_experts=128,
        experts_per_token=4,
        hidden_dim=6144,
        intermediate_size=3072,
        max_model_len=2304,
    )

    assert not _should_use_bf16_decode_fallback(moe_config)


@pytest.mark.parametrize(
    ("num_tokens", "native_weights_available", "expected"),
    [
        (1, True, True),
        (8, True, True),
        (9, True, False),
        (128, True, False),
        (831, True, False),
        (832, True, True),
        (1023, True, True),
        (128, False, True),
    ],
)
def test_mxfp8_bf16_expert_dispatch(
    num_tokens,
    native_weights_available,
    expected,
):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _should_use_bf16_experts,
    )

    assert _should_use_bf16_experts(num_tokens, native_weights_available) is expected


@pytest.mark.parametrize(
    ("max_model_len", "layer_index", "expected"),
    [
        (2304, 0, False),
        (9472, 0, True),
        (9472, 1, False),
        (9472, 5, True),
    ],
)
def test_mxfp8_bf16_only_storage_policy(max_model_len, layer_index, expected):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        _should_store_bf16_only,
    )

    assert _should_store_bf16_only(max_model_len, layer_index) is expected


@pytest.mark.skipif(
    not current_platform.is_fp8_fnuz(),
    reason="E4M3FN checkpoint normalization is specific to gfx94x.",
)
@torch.inference_mode()
def test_mxfp8_e4m3fn_to_fnuz_normalization():
    value_bits = torch.zeros((1, 32), dtype=torch.int8, device=DEVICE)
    value_bits[0, :3] = torch.tensor([-128, 56, -72], device=DEVICE)
    values = value_bits.view(torch.float8_e4m3fn)
    scales = torch.tensor([[127]], dtype=torch.uint8, device=DEVICE)
    expected = dequant_mxfp8_to_bf16(values, scales)

    converted, converted_scales = normalize_mxfp8_e4m3fn_to_e4m3fnuz(
        values.clone(), scales.clone()
    )

    assert converted.dtype == torch.float8_e4m3fnuz
    assert converted.view(torch.int8)[0, 0].item() == 0
    assert converted_scales.item() == 128
    assert torch.equal(dequant_mxfp8_to_bf16(converted, converted_scales), expected)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


# --------------------------------------------------------------------------- #
# Gemma RMSNorm
# --------------------------------------------------------------------------- #
def _ref_gemma_rmsnorm(x, w, eps, residual=None):
    orig_dtype = x.dtype
    xf = x.float()
    res_out = None
    if residual is not None:
        xf = xf + residual.float()
        res_out = xf.to(orig_dtype)
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    xf = xf * (1.0 + w.float())
    out = xf.to(orig_dtype)
    return out if residual is None else (out, res_out)


@pytest.mark.parametrize("shape", [(1, 4096), (37, 6144), (128, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 1234])
@torch.inference_mode()
def test_gemma_rmsnorm(shape, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE, dtype=dtype) * 0.1
    got = gemma_rmsnorm(x, w, EPS)
    ref = _ref_gemma_rmsnorm(x, w, EPS)
    assert got.shape == x.shape
    assert _relerr(got, ref) < 5e-3


@pytest.mark.parametrize("shape", [(1, 6144), (64, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gemma_fused_add_rmsnorm(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    res = torch.randn(*shape, device=DEVICE, dtype=dtype)
    w = torch.randn(shape[-1], device=DEVICE, dtype=dtype) * 0.1
    got_out, got_res = gemma_fused_add_rmsnorm(x, res, w, EPS)
    ref_out, ref_res = _ref_gemma_rmsnorm(x, w, EPS, residual=res)
    assert _relerr(got_out, ref_out) < 5e-3
    # residual_out is the pre-norm sum (x + res): bit-for-bit identical cast.
    assert torch.equal(got_res, ref_res)


@torch.inference_mode()
def test_gemma_rmsnorm_per_head_strided():
    """q_norm/k_norm normalize a non-contiguous ``qkv.split`` slice over head_dim."""
    torch.manual_seed(0)
    T, H, D, kv = 7, 48, 128, 8
    total = (H + 2 * kv) * D
    qkv = torch.randn(T, total, device=DEVICE, dtype=torch.bfloat16)
    q = qkv[..., : H * D]  # non-contiguous view (row stride == total)
    q_by_head = q.view(T, H, D)
    assert not q_by_head.is_contiguous()
    w = torch.randn(D, device=DEVICE, dtype=torch.bfloat16) * 0.1
    got = gemma_rmsnorm(q_by_head, w, EPS)
    ref = _ref_gemma_rmsnorm(q_by_head, w, EPS)
    assert got.shape == q_by_head.shape
    assert _relerr(got, ref) < 5e-3


def test_num_warps_monotonic():
    assert _num_warps(128) <= _num_warps(2048) <= _num_warps(8192)


# --------------------------------------------------------------------------- #
# SwiGLU-OAI (split layout)
# --------------------------------------------------------------------------- #
def _ref_swiglu(gate_up, alpha, beta, limit):
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(gate_up.dtype)


@pytest.mark.parametrize("m,inter", [(1, 768), (64, 1536), (128, 1024)])
@pytest.mark.parametrize("limit", [7.0, None])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_swiglu_oai_split(m, inter, limit, dtype):
    torch.manual_seed(0)
    gate_up = torch.randn(m, 2 * inter, device=DEVICE, dtype=dtype)
    got = swiglu_oai_split(gate_up, alpha=1.702, beta=1.0, limit=limit)
    ref = _ref_swiglu(gate_up, 1.702, 1.0, limit)
    assert got.shape == (m, inter)
    assert _relerr(got, ref) < 5e-3


# --------------------------------------------------------------------------- #
# Fused MXFP8 activation quant (Triton vs torch reference)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(64, 4096), (1, 6144), (333, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_mxfp8_quant_triton_matches_torch(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, device=DEVICE, dtype=dtype)
    xq_t, s_t = _mxfp8_e4m3_quantize_torch(
        x,
        is_sf_swizzled_layout=False,
        value_dtype=current_platform.fp8_dtype(),
    )
    xq_k, s_k = _mxfp8_e4m3_quantize_triton(x)
    assert s_k.shape == s_t.shape == (shape[0], shape[1] // 32)
    # E8M0 block exponents share the floor(log2(amax))+127 algorithm; allow at
    # most a 1-step difference at exact powers of two.
    assert (s_k.int() - s_t.int()).abs().max().item() <= 1
    # Dequantized values agree to fp8 granularity.
    deq_t = dequant_mxfp8_to_bf16(xq_t, s_t)
    deq_k = dequant_mxfp8_to_bf16(xq_k, s_k)
    assert _relerr(deq_k, deq_t) < 1e-2


# --------------------------------------------------------------------------- #
# Native CDNA4 MXFP8 linear vs dequant-to-bf16 matmul
# --------------------------------------------------------------------------- #
@requires_gfx950
@pytest.mark.parametrize("m,n,k", [(64, 256, 128), (37, 512, 256), (1, 6144, 4096)])
@torch.inference_mode()
def test_mxfp8_native_linear(m, n, k):
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        _mxfp8_dot_scaled_linear,
    )

    torch.manual_seed(0)
    w_bf16 = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16, is_sf_swizzled_layout=False)
    x = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16) * 0.5

    got = _mxfp8_dot_scaled_linear(x, w_fp8, w_scale)
    # Reference: consume the SAME quantized weights (isolates activation-quant
    # noise) -> dequant to bf16, plain matmul.
    w_deq = dequant_mxfp8_to_bf16(w_fp8, w_scale)
    ref = torch.nn.functional.linear(x, w_deq).to(x.dtype)
    assert got.shape == (m, n)
    # Only the activation is re-quantized inside the kernel -> small MX noise.
    assert _relerr(got, ref) < 5e-2


# --------------------------------------------------------------------------- #
# Fused MXFP8 MoE grouped GEMM vs dequant-to-bf16 MoE math
# --------------------------------------------------------------------------- #
def _ref_moe(x, w13, w2, topk_weights, topk_ids, alpha, beta, limit):
    T, H = x.shape
    inter = w2.shape[-1]
    top_k = topk_ids.shape[1]
    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            g1 = x[t].float() @ w13[e].float().T  # [2I]
            gate = g1[:inter]
            up = g1[inter:]
            if limit is not None:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            act = gate * torch.sigmoid(alpha * gate) * (up + beta)
            g2 = act @ w2[e].float().T  # [H]
            out[t] += topk_weights[t, j].float() * g2
    return out.to(x.dtype)


@requires_mi3xx
@pytest.mark.parametrize(
    "T,H,inter,E,top_k",
    [
        (8, 256, 512, 8, 2),
        (1, 512, 256, 16, 4),
        (1, 2048, 256, 4, 2),
    ],
)
@pytest.mark.parametrize("pack_scales", [False, True])
@torch.inference_mode()
def test_mxfp8_native_moe(T, H, inter, E, top_k, pack_scales):
    from vllm.model_executor.layers.fused_moe.experts.mxfp8_native_moe import (
        fused_moe_mxfp8_native,
    )

    torch.manual_seed(0)
    alpha, beta, limit = 1.702, 1.0, 7.0
    w13_bf16 = torch.randn(E, 2 * inter, H, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w2_bf16 = torch.randn(E, H, inter, device=DEVICE, dtype=torch.bfloat16) * 0.1
    w13_fp8, w13_scale = _mxfp8_e4m3_quantize_torch(
        w13_bf16, is_sf_swizzled_layout=False
    )
    w2_fp8, w2_scale = _mxfp8_e4m3_quantize_torch(w2_bf16, is_sf_swizzled_layout=False)
    if current_platform.is_fp8_fnuz():
        w13_fp8, w13_scale = normalize_mxfp8_e4m3fn_to_e4m3fnuz(w13_fp8, w13_scale)
        w2_fp8, w2_scale = normalize_mxfp8_e4m3fn_to_e4m3fnuz(w2_fp8, w2_scale)
    w13_deq = dequant_mxfp8_to_bf16(w13_fp8, w13_scale)
    w2_deq = dequant_mxfp8_to_bf16(w2_fp8, w2_scale)
    if pack_scales:
        w13_scale = w13_scale.transpose(1, 2).contiguous()
        w2_scale = w2_scale.transpose(1, 2).contiguous()

    x = torch.randn(T, H, device=DEVICE, dtype=torch.bfloat16) * 0.5
    logits = torch.randn(T, E, device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = logits.softmax(dim=-1).topk(top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    got = fused_moe_mxfp8_native(
        x,
        w13_fp8,
        w13_scale,
        w2_fp8,
        w2_scale,
        topk_weights,
        topk_ids,
        alpha=alpha,
        beta=beta,
        limit=limit,
        global_num_experts=E,
        expert_map=None,
    )
    # Reference consumes the dequantized weights (same bits the kernel reads).
    ref = _ref_moe(x, w13_deq, w2_deq, topk_weights, topk_ids, alpha, beta, limit)
    assert got.shape == (T, H)
    assert _relerr(got, ref) < 5e-2


# --------------------------------------------------------------------------- #
# MXFP8 linear emulation: BF16-at-load (default) vs per-step dequant + switch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(512, 2048), (1, 6144)])
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dequant_at_load", [True, False])
@torch.inference_mode()
def test_mxfp8_linear_emulation_bf16_at_load(
    shape, act_dtype, dequant_at_load, monkeypatch
):
    """EmulationMxfp8LinearKernel load-time BF16 dequant (default) and the
    ``VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD=0`` per-step fallback must produce the
    same result; the dtype-match (BF16/FP16 activations) must also hold."""
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )

    monkeypatch.setenv(
        "VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD", "1" if dequant_at_load else "0"
    )
    N, K = shape
    torch.manual_seed(0)
    w_bf16 = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
    w_fp8, w_scale = _mxfp8_e4m3_quantize_torch(w_bf16, is_sf_swizzled_layout=False)
    assert w_scale.shape == (N, K // 32)

    # Reference: dequant once, plain linear in the activation dtype.
    w_ref = dequant_mxfp8_to_bf16(w_fp8, w_scale).to(act_dtype)
    x = torch.randn(7, K, device=DEVICE, dtype=act_dtype)
    out_ref = torch.nn.functional.linear(x, w_ref)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_fp8.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(w_scale.clone(), requires_grad=False)

    kernel = EmulationMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)

    if dequant_at_load:
        # weights converted to BF16 at load (>= 2-byte)
        assert layer.weight.element_size() >= 2
    else:
        # opt-out: weights stay 1-byte MXFP8, dequant happens per-step
        assert layer.weight.element_size() == 1

    out = kernel.apply_weights(layer, x)
    assert out.dtype == act_dtype  # dtype-match preserved (no tl.dot/F.linear crash)
    assert _relerr(out.float(), out_ref.float()) < 2e-2
