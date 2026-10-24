# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CDNA4 MXFP8 linear (``RocmDotScaledMxfp8LinearKernel``).

DeepSeek V4.1 routes every attention linear through this kernel. It used to
carry two BF16-dequantize workarounds on ROCm (one for all ``.attn.`` linears,
one for the row-parallel ``wo_b``, whose comment blamed NaNs at small token
counts). These tests pin the two properties those workarounds stood in for --
finite output and MXFP8-level accuracy -- across the V4.1 attention shapes at
TP=4 and the small token counts decode actually runs.

They also cover the tiny-M fused quant+GEMM fastpath of
``rocm_mxfp8_block32_gemm_quant`` (at M <= 8 the activation quant runs inside
the packed GEMM's K loop): bit-exactness with the stock two-launch
``mxfp8_e4m3_quantize`` + ``rocm_mxfp8_block32_gemm`` path at the serving
shapes and under CUDA-graph capture/replay, agreement with a high-precision
QDQ fp32 reference, and bit-identical fallbacks for ineligible inputs and the
``VLLM_DEBUG_DISABLE_FUSED_MXFP8_QUANT_GEMM=1`` kill switch.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    _mxfp8_e4m3_quantize_triton,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not current_platform.supports_mx(),
    reason="native MX linear requires CDNA4 (gfx95x)",
)

from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (  # noqa: E402
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import (  # noqa: E402
    rocm_mxfp8_block32_gemm,
    rocm_mxfp8_block32_gemm_quant,
)
from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (  # noqa: E402
    RocmDotScaledMxfp8LinearKernel,
)

# (name, N, K) for the DeepSeek V4.1 attention linears at tensor_parallel=4:
# fused wq_a|wkv is replicated, wq_b/wo_a are column-parallel, wo_b is
# row-parallel (so K, not N, is sharded).
V41_ATTN_SHAPES = [
    ("fused_wqa_wkv", 1280 + 512, 5120),
    ("wq_b", 64 * 512 // 4, 1280),
    ("wo_a", 8 * 1024 // 4, 4096),
    ("wo_b", 5120, 8192 // 4),
]


def _make_layer(n: int, k: int, device: str) -> torch.nn.Module:
    weight_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    return layer


@pytest.mark.parametrize("name,n,k", V41_ATTN_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 16, 31, 32, 128, 1024])
@torch.inference_mode()
def test_rocm_mxfp8_linear_matches_dequant_reference(name, n, k, num_tokens):
    """Output is finite and matches an exact MXFP8 GEMM of the same inputs.

    The reference quantizes the activation with the same quantizer and then
    contracts in FP32, so the tolerance covers only kernel/accumulation error,
    not the MXFP8 quantization error both sides share.
    """
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    torch.manual_seed(num_tokens)
    device = "cuda"
    layer = _make_layer(n, k, device)
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(num_tokens, k, device=device, dtype=torch.bfloat16) * 0.5
    out = kernel.apply_weights(layer, x)

    assert out.shape == (num_tokens, n)
    assert torch.isfinite(out).all(), f"{name}: non-finite output"

    x_q, x_scale = mxfp8_e4m3_quantize(x)
    expected = (
        dequant_mxfp8_to_bf16(x_q, x_scale).float()
        @ dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).float().T
    )
    rel = (out.float() - expected).norm() / expected.norm()
    assert rel < 5e-3, f"{name}: relative error {rel:.4f}"


@torch.inference_mode()
def test_rocm_mxfp8_linear_handles_extreme_magnitudes():
    """Rows that are all-zero or near the E4M3 range must not produce NaNs.

    A degenerate block makes the dynamic activation quantizer pick an extreme
    E8M0 scale; a zero row makes it pick the tiny-clamped one. Both feed
    ``tl.dot_scaled`` directly.
    """
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    torch.manual_seed(0)
    device = "cuda"
    n, k = 5120, 2048  # wo_b at TP=4
    layer = _make_layer(n, k, device)
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    x = torch.randn(8, k, device=device, dtype=torch.bfloat16)
    x[0] = 0.0
    x[1] = 1e-30
    x[2] = 400.0
    x[3, ::2] = 0.0

    out = kernel.apply_weights(layer, x)
    assert torch.isfinite(out).all()
    assert (out[0] == 0).all(), "an all-zero activation row must give a zero row"


# (name, N, K) of the DeepSeek-V4.1-Flash MXFP8 linears at TP=2 and TP=4. The
# checkpoint stores their scales in 32x32 blocks; shared_down at TP=4 has
# K % 128 != 0, which only the block-scaled GEMM runs natively.
V41_BLOCK32_SHAPES = [
    ("fused_wqa_wkv", 1280 + 512, 5120),
    ("wq_b_tp2", 64 * 512 // 2, 1280),
    ("indexer_wq_b", 32 * 128, 1280),
    ("wo_b_tp2", 5120, 8192 // 2),
    ("shared_gate_up_tp2", 2 * 2304 // 2, 5120),
    ("shared_down_tp2", 5120, 2304 // 2),
    ("shared_down_tp4", 5120, 2304 // 4),
]


def _make_block32_layer(n: int, k: int, device: str) -> torch.nn.Module:
    """An MXFP8 layer with 32x32 block scales, expanded per row as loaded."""
    blocks = (torch.randn(n, k, device=device) * 0.05).view(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=torch.finfo(torch.float32).tiny)
    bits = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0, 254)
    weight = (blocks * torch.exp2(127.0 - bits)[:, None, :, None]).view(n, k)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        weight.to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        bits.to(torch.uint8).repeat_interleave(32, dim=0), requires_grad=False
    )
    return layer


@pytest.mark.parametrize("name,n,k", V41_BLOCK32_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("num_tokens", [1, 7, 32, 33, 192, 1024])
@torch.inference_mode()
def test_rocm_mxfp8_linear_block32_matches_dequant_reference(name, n, k, num_tokens):
    """32x32 block scales stay compact and the block-scaled GEMM is exact.

    The token counts cover the packed, tiled and split-K configurations.
    """
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    torch.manual_seed(num_tokens)
    device = "cuda"
    layer = _make_block32_layer(n, k, device)
    row_scale = layer.weight_scale.data.clone()
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.shape == (n // 32, k // 32)

    x = torch.randn(num_tokens, k, device=device, dtype=torch.bfloat16) * 0.5
    out = kernel.apply_weights(layer, x)

    x_q, x_scale = mxfp8_e4m3_quantize(x)
    expected = (
        dequant_mxfp8_to_bf16(x_q, x_scale).float()
        @ dequant_mxfp8_to_bf16(layer.weight, row_scale).float().T
    )
    assert out.shape == (num_tokens, n)
    rel = (out.float() - expected).norm() / expected.norm()
    assert rel < 5e-3, f"{name}: relative error {rel:.4f}"


@torch.inference_mode()
def test_rocm_mxfp8_linear_keeps_per_row_scales():
    """Scales that differ within a 32-row block are not compacted."""
    from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
        Mxfp8LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
        RocmDotScaledMxfp8LinearKernel,
    )

    n, k = 64, 256
    layer = _make_block32_layer(n, k, "cuda")
    layer.weight_scale.data[1, 0] += 1
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)
    assert layer.weight_scale.shape == (n, k // 32)


def _split_k_case(packed: bool):
    """A 3-way in-launch split-K on shared_down TP4, with K and M tails."""
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_block32_gemm as g

    n, k, m = 5120, 576, 19
    layer = _make_block32_layer(n, k, "cuda")
    x_q, x_scale = mxfp8_e4m3_quantize(
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    )
    expected = (
        dequant_mxfp8_to_bf16(x_q, x_scale).float()
        @ dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).float().T
    )
    if packed:
        cfg = g._packed(16, 32, 128, 1, 2, 2, 16, 0, 3)
    else:
        cfg = g._tiled(32, 64, 128, 1, True, 4, 2, 16, 0, 3)
    args = (x_q, x_scale, layer.weight, layer.weight_scale.data[::32].contiguous())
    return g, cfg, args, expected


def _rel_err(out: torch.Tensor, expected: torch.Tensor) -> float:
    return ((out.float() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("packed", [True, False])
@torch.inference_mode()
def test_rocm_mxfp8_block32_split_k_in_launch(packed):
    """Splits summed by the tile's last program are exact and repeatable."""
    torch.manual_seed(0)
    g, cfg, args, expected = _split_k_case(packed)
    out = torch.empty_like(expected, dtype=torch.bfloat16)
    g._launch(*args, out, cfg)
    assert _rel_err(out, expected) < 5e-3
    first = out.clone()
    for _ in range(20):
        g._launch(*args, out, cfg)
        assert torch.equal(out, first)


@pytest.mark.parametrize("packed", [True, False])
@torch.inference_mode()
def test_rocm_mxfp8_block32_split_k_first_use_in_capture(packed):
    """A stream whose first split-K launch is being captured has no counters
    yet; it must fall back to an unfused split instead of failing."""
    torch.manual_seed(0)
    g, cfg, args, expected = _split_k_case(packed)
    out = torch.zeros_like(expected, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g._launch(*args, out, cfg)
    graph.replay()
    torch.accelerator.synchronize()
    assert _rel_err(out, expected) < 5e-3


@pytest.mark.parametrize("num_rows", [1, 64, 65])
@torch.inference_mode()
def test_rocm_mxfp8_quantizer_matches_torch_on_degenerate_blocks(num_rows):
    """The fused ROCm quantizer must agree with the torch one, zeros included.

    An all-zero 32-element block clamps the E8M0 scale to 0, i.e. 2**-127 --
    subnormal in fp32 and flushed to zero on CDNA. Dividing by it turned the
    block into 0/0 == NaN, which then poisoned every output column of the GEMM
    that consumed it.
    """
    torch.manual_seed(num_rows)
    device = "cuda"
    x = torch.randn(num_rows, 128, device=device, dtype=torch.bfloat16)
    x[0] = 0.0  # all-zero row: every block clamps to scale byte 0
    x[-1, :32] = 0.0  # single zero block in an otherwise normal row
    x[-1, 32:64] = 1e-38  # denormal-ish block

    q_triton, s_triton = _mxfp8_e4m3_quantize_triton(x)
    q_torch, s_torch = _mxfp8_e4m3_quantize_torch(x)

    assert not torch.isnan(q_triton.float()).any()
    torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)
    torch.testing.assert_close(
        q_triton.view(torch.uint8), q_torch.view(torch.uint8), rtol=0, atol=0
    )


# --- tiny-M fused quant+GEMM (rocm_mxfp8_block32_gemm_quant) -----------------
#
# Serving block32 linears at TP=4 that hit the fused path at decode,
# covering the K-structure variety: the multi-band K=5120 shape with the
# in-launch split-K tier (shared_gate_up), a second K=5120 tier, the
# masked-K-tail K=1280 (EVEN_K=False), a small-K tile, and a draft-model
# shape absent from the tuned table (_default_config tile).
FUSED_QUANT_SERVING_SHAPES = [
    ("shared_gate_up_tp4", 1152, 5120),
    ("fused_wqa_wkv", 1792, 5120),
    ("wq_b_tp4", 8192, 1280),
    ("shared_down_tp4", 5120, 576),
    ("default_config", 1024, 2048),
]

# Decode single, the DSpark k=5 hot M, and the M <= 8 tier bound.
FUSED_QUANT_MS = [1, 5, 8]


def _make_block32_weight(n: int, k: int, device: str, seed: int = 0):
    """e4m3 weight [n, k] with compact [n//32, k//32] E8M0 block scales."""
    torch.manual_seed(seed)
    blocks = (torch.randn(n, k, device=device) * 0.05).view(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=torch.finfo(torch.float32).tiny)
    bits = (torch.ceil(torch.log2(amax / 448.0)) + 127.0).clamp(0, 254)
    weight = (blocks * torch.exp2(127.0 - bits)[:, None, :, None]).view(n, k)
    return (
        weight.to(torch.float8_e4m3fn).contiguous(),
        bits.to(torch.uint8).contiguous(),
    )


def _stock(x, weight, weight_scale, out_dtype=torch.bfloat16):
    """The pre-fusion two-launch path, verbatim."""
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    return rocm_mxfp8_block32_gemm(x_q, x_scale, weight, weight_scale, out_dtype)


def _fused_quant_hit(m, n, k) -> bool:
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_block32_gemm as g

    return (m, n, k) in g._FUSED_QUANT_HIT_SHAPES


@pytest.mark.parametrize("name,n,k", FUSED_QUANT_SERVING_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("m", FUSED_QUANT_MS)
@torch.inference_mode()
def test_rocm_mxfp8_fused_quant_bit_exact_serving_shapes(name, n, k, m):
    """The fused output equals the stock two-launch path byte-for-byte at the
    serving shapes, and a high-precision QDQ fp32 reference within tolerance.

    The reference quantizes the activation with the same quantizer and then
    dequantizes both operands and contracts in fp32, so it is independent of
    the block32 GEMM kernel under test.
    """
    torch.manual_seed(m)
    weight, weight_scale = _make_block32_weight(n, k, "cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5

    expected = _stock(x, weight, weight_scale)
    out = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)
    out2 = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)

    assert _fused_quant_hit(m, n, k), f"{name}: fused fastpath did not engage at M={m}"
    assert torch.equal(out, expected), f"{name}: fused output differs from stock"
    assert torch.equal(out, out2), f"{name}: fused output not deterministic"

    x_q, x_scale = mxfp8_e4m3_quantize(x)
    row_scale = weight_scale.repeat_interleave(32, dim=0)
    reference = (
        dequant_mxfp8_to_bf16(x_q, x_scale).float()
        @ dequant_mxfp8_to_bf16(weight, row_scale).float().T
    )
    rel = (out.float() - reference).norm() / reference.norm()
    assert rel < 5e-3, f"{name}: fused output vs QDQ reference, rel err {rel:.4f}"


@torch.inference_mode()
def test_rocm_mxfp8_fused_quant_bit_exact_degenerate_blocks():
    """All-zero / denormal / huge / negative blocks quantize identically.

    An all-zero 32-block clamps the E8M0 scale to byte 0 (2**-127,
    subnormal in fp32): the reciprocal-form rescale must keep it NaN-free,
    exactly as the separate quantizer does.
    """
    n, k, m = 512, 1024, 6
    weight, weight_scale = _make_block32_weight(n, k, "cuda", seed=7)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    x[0] = 0.0  # all-zero row
    x[1, :32] = 0.0  # single zero block
    x[1, 32:64] = 1e-38  # denormal block
    x[2] = 3.0e30  # huge: scale byte clamps at 254
    x[3] = -x[3]  # negative mirror
    x[4, ::32] = 0.0  # zero at every block start

    expected = _stock(x, weight, weight_scale)
    out = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)
    assert _fused_quant_hit(m, n, k)
    assert torch.isfinite(out.float()).all(), "non-finite fused output"
    assert torch.equal(out, expected), "degenerate blocks differ from stock"


@torch.inference_mode()
def test_rocm_mxfp8_fused_quant_fallbacks_match_stock():
    """Ineligible inputs take the internal stock path, bit-identical."""
    n, k = 1024, 512
    weight, weight_scale = _make_block32_weight(n, k, "cuda", seed=11)

    # Non-contiguous bf16 x (stride-2 view).
    base = torch.randn(6, 2 * k, device="cuda", dtype=torch.bfloat16) * 0.5
    x_nc = base[:, ::2]
    assert not x_nc.is_contiguous()
    out = rocm_mxfp8_block32_gemm_quant(x_nc, weight, weight_scale, torch.bfloat16)
    assert not _fused_quant_hit(6, n, k), (
        "non-contiguous x must not take the fused path"
    )
    assert torch.equal(out, _stock(x_nc.contiguous(), weight, weight_scale))

    # fp32 x falls back (quantizer loads any float dtype, fused does not).
    x_f32 = torch.randn(6, k, device="cuda", dtype=torch.float32) * 0.5
    out = rocm_mxfp8_block32_gemm_quant(x_f32, weight, weight_scale, torch.float32)
    assert not _fused_quant_hit(6, n, k), "fp32 x must not take the fused path"
    assert torch.equal(out, _stock(x_f32, weight, weight_scale, torch.float32))

    # M > 8 (prefill-class) falls back.
    x_big = torch.randn(16, k, device="cuda", dtype=torch.bfloat16) * 0.5
    out = rocm_mxfp8_block32_gemm_quant(x_big, weight, weight_scale, torch.bfloat16)
    assert not _fused_quant_hit(16, n, k), "M=16 must not take the fused path"
    assert torch.equal(out, _stock(x_big, weight, weight_scale))

    # M = 0 returns an empty output.
    x0 = torch.randn(0, k, device="cuda", dtype=torch.bfloat16)
    out = rocm_mxfp8_block32_gemm_quant(x0, weight, weight_scale, torch.bfloat16)
    assert out.shape == (0, n)


def test_rocm_mxfp8_fused_quant_kill_switch(monkeypatch):
    """VLLM_DEBUG_DISABLE_FUSED_MXFP8_QUANT_GEMM=1 restores the two-launch
    path."""
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_block32_gemm as g

    n, k, m = 256, 512, 3  # unique shape for an unambiguous ledger check
    weight, weight_scale = _make_block32_weight(n, k, "cuda", seed=13)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    expected = _stock(x, weight, weight_scale)

    monkeypatch.setenv("VLLM_DEBUG_DISABLE_FUSED_MXFP8_QUANT_GEMM", "1")
    g._FUSED_QUANT_ENV_OK = None  # reset the cached decision
    try:
        with torch.inference_mode():
            out = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)
        assert not _fused_quant_hit(m, n, k), "kill switch did not disable the fastpath"
        assert torch.equal(out, expected), "disabled path differs from stock"
    finally:
        monkeypatch.delenv("VLLM_DEBUG_DISABLE_FUSED_MXFP8_QUANT_GEMM")
        g._FUSED_QUANT_ENV_OK = None


@torch.inference_mode()
def test_rocm_mxfp8_fused_quant_per_row_scale_unaffected():
    """Regression guard: the per-row-scale branch of
    RocDotScaledMxfp8LinearKernel.apply_weights (weight_scale [N, K//32],
    NOT compacted to block32) must keep working when the block32 branch
    dispatches to the fused op. The numerics of this branch are owned by
    test_rocm_mxfp8_linear_matches_dequant_reference above; this test pins
    the dispatch interaction only.
    """
    n, k, m = 1152, 5120, 7
    torch.manual_seed(23)
    weight_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    assert weight_scale.shape == (n, k // 32), "expected per-row scales"

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    kernel = RocmDotScaledMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)
    # Non-uniform per-row scales are not compacted to [N//32, K//32].
    assert layer.weight_scale.shape[0] == n

    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    out = kernel.apply_weights(layer, x)
    assert out.shape == (m, n)
    assert torch.isfinite(out.float()).all()
    # This branch never reaches the fused op.
    assert not _fused_quant_hit(m, n, k), (
        "per-row-scale layer must not hit the fused path"
    )


@pytest.mark.parametrize("warm_counters", [True, False])
@torch.inference_mode()
def test_rocm_mxfp8_fused_quant_cuda_graph_capture(warm_counters):
    """Capture+replay is bit-exact with and without warm split-K counters.

    shared_gate_up TP4 at M=6 uses the in-launch split-K config
    (fused_splits=3): with warmup the counters exist and the splits meet in
    one launch; captured first-use on a fresh stream has no counters and
    must fall back to the two-launch split-K (partials + reduce).
    """
    n, k, m = 1152, 5120, 6
    weight, weight_scale = _make_block32_weight(n, k, "cuda", seed=17)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    expected = _stock(x, weight, weight_scale)

    stream = torch.cuda.Stream()
    if warm_counters:
        with torch.cuda.stream(stream):
            warm = rocm_mxfp8_block32_gemm_quant(
                x, weight, weight_scale, torch.bfloat16
            )
        stream.synchronize()
        assert torch.equal(warm, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)
    for _ in range(3):
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out, expected), "replay differs from stock"
