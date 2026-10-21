# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Core contracts for the tiny-M fused MXFP8 quant+GEMM.

At M <= 8 (decode shapes) the block32 linear path pays a full
``_mxfp8_quant_triton_kernel`` launch just to produce the e4m3 tiles +
E8M0 scales the packed GEMM immediately re-reads; the fused kernel
computes that quant inside the GEMM's K loop, in registers, with the
exact same arithmetic. Essential tests only:

1. BIT-EXACTNESS: the fused output equals ``mxfp8_e4m3_quantize`` +
   ``rocm_mxfp8_block32_gemm`` byte-for-byte at the serving shapes
   (tuned tiers, the masked-K-tail config, the in-launch split-K tier,
   and the default-config tier), at degenerate activation blocks, and
   under CUDA-graph capture/replay in both counter states.
2. FALLBACKS: non-contiguous x, fp32 x, M > 8, and the kill switch
   (VLLM_DISABLE_FUSED_MXFP8_QUANT_GEMM=1) all take the stock path,
   bit-identically.
3. The per-row-scale dispatch branch of the linear kernel is unaffected.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
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

# Serving block32 linears at TP=4 that hit the fused path at decode,
# covering the K-structure variety: the multi-band K=5120 shape with the
# in-launch split-K tier (shared_gate_up), a second K=5120 tier, the
# masked-K-tail K=1280 (EVEN_K=False), a small-K tile, and a draft-model
# shape absent from the tuned table (_default_config tile).
SERVING_SHAPES = [
    ("shared_gate_up_tp4", 1152, 5120),
    ("fused_wqa_wkv", 1792, 5120),
    ("wq_b_tp4", 8192, 1280),
    ("shared_down_tp4", 5120, 576),
    ("default_config", 1024, 2048),
]

# Decode single, the DSpark k=5 hot M, and the M <= 8 tier bound.
CAPTURE_MS = [1, 5, 8]


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


def _hit(m, n, k) -> bool:
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_block32_gemm as g

    return (m, n, k) in g._FUSED_QUANT_HIT_SHAPES


@pytest.mark.parametrize("name,n,k", SERVING_SHAPES, ids=lambda v: str(v))
@pytest.mark.parametrize("m", CAPTURE_MS)
@torch.inference_mode()
def test_fused_bit_exact_serving_shapes(name, n, k, m):
    torch.manual_seed(m)
    weight, weight_scale = _make_block32_weight(n, k, "cuda")
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5

    expected = _stock(x, weight, weight_scale)
    out = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)
    out2 = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)

    assert _hit(m, n, k), f"{name}: fused fastpath did not engage at M={m}"
    assert torch.equal(out, expected), f"{name}: fused output differs from stock"
    assert torch.equal(out, out2), f"{name}: fused output not deterministic"


@torch.inference_mode()
def test_fused_bit_exact_degenerate_blocks():
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
    assert _hit(m, n, k)
    assert torch.isfinite(out.float()).all(), "non-finite fused output"
    assert torch.equal(out, expected), "degenerate blocks differ from stock"


@torch.inference_mode()
def test_fused_fallbacks_match_stock():
    """Ineligible inputs take the internal stock path, bit-identical."""
    n, k = 1024, 512
    weight, weight_scale = _make_block32_weight(n, k, "cuda", seed=11)

    # Non-contiguous bf16 x (stride-2 view).
    base = torch.randn(6, 2 * k, device="cuda", dtype=torch.bfloat16) * 0.5
    x_nc = base[:, ::2]
    assert not x_nc.is_contiguous()
    out = rocm_mxfp8_block32_gemm_quant(x_nc, weight, weight_scale, torch.bfloat16)
    assert not _hit(6, n, k), "non-contiguous x must not take the fused path"
    assert torch.equal(out, _stock(x_nc.contiguous(), weight, weight_scale))

    # fp32 x falls back (quantizer loads any float dtype, fused does not).
    x_f32 = torch.randn(6, k, device="cuda", dtype=torch.float32) * 0.5
    out = rocm_mxfp8_block32_gemm_quant(x_f32, weight, weight_scale, torch.float32)
    assert not _hit(6, n, k), "fp32 x must not take the fused path"
    assert torch.equal(out, _stock(x_f32, weight, weight_scale, torch.float32))

    # M > 8 (prefill-class) falls back.
    x_big = torch.randn(16, k, device="cuda", dtype=torch.bfloat16) * 0.5
    out = rocm_mxfp8_block32_gemm_quant(x_big, weight, weight_scale, torch.bfloat16)
    assert not _hit(16, n, k), "M=16 must not take the fused path"
    assert torch.equal(out, _stock(x_big, weight, weight_scale))

    # M = 0 returns an empty output.
    x0 = torch.randn(0, k, device="cuda", dtype=torch.bfloat16)
    out = rocm_mxfp8_block32_gemm_quant(x0, weight, weight_scale, torch.bfloat16)
    assert out.shape == (0, n)


def test_fused_kill_switch(monkeypatch):
    """VLLM_DISABLE_FUSED_MXFP8_QUANT_GEMM=1 restores the two-launch path."""
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_block32_gemm as g

    n, k, m = 256, 512, 3  # unique shape for an unambiguous ledger check
    weight, weight_scale = _make_block32_weight(n, k, "cuda", seed=13)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    expected = _stock(x, weight, weight_scale)

    monkeypatch.setenv("VLLM_DISABLE_FUSED_MXFP8_QUANT_GEMM", "1")
    g._FUSED_QUANT_ENV_OK = None  # reset the cached decision
    try:
        with torch.inference_mode():
            out = rocm_mxfp8_block32_gemm_quant(x, weight, weight_scale, torch.bfloat16)
        assert not _hit(m, n, k), "kill switch did not disable the fastpath"
        assert torch.equal(out, expected), "disabled path differs from stock"
    finally:
        monkeypatch.delenv("VLLM_DISABLE_FUSED_MXFP8_QUANT_GEMM")
        g._FUSED_QUANT_ENV_OK = None


@torch.inference_mode()
def test_per_row_scale_apply_weights_unaffected():
    """Regression guard: the per-row-scale branch of
    RocDotScaledMxfp8LinearKernel.apply_weights (weight_scale [N, K//32],
    NOT compacted to block32) must keep working when the block32 branch
    dispatches to the fused op. The numerics of this branch are owned by
    the pre-existing test_rocm_mxfp8_linear.py; this test pins the
    dispatch interaction only.
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )

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
    assert not _hit(m, n, k), "per-row-scale layer must not hit the fused path"


@pytest.mark.parametrize("warm_counters", [True, False])
@torch.inference_mode()
def test_fused_cuda_graph_capture(warm_counters):
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
