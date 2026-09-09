#!/usr/bin/env python3
# SPDX-LicenseFileCopyrightText: Copyright contributors to the vLLM project
"""Determinism + FP32-reference correctness tests for the RDNA3 W4A16 GEMM.

The split-K epilogues of ``gptq_gemm_rdna3`` (scalar path) and the WMMA
path previously accumulated per-split low-precision partials directly into
the output via CAS atomics. With multiple writers per output element the
accumulation order — and therefore the rounded result — varied per call:
repeated fixed-input calls produced a different output almost every time on
gfx1100 once more than a few split blocks contended.

Both paths now store FP32 split partials and reduce them in a fixed order
with a single final low-precision rounding, which is deterministic by
construction. When the split count is 1 every path stores directly
(single writer per output element).

These tests assert, through the public op path:

A. Repeatability — ``torch.equal`` across repeated identical calls.
B. Numerical correctness — against an FP32 dequantized reference.

(B) exists because repeatability alone cannot see the failure mode the
deterministic epilogue introduces: the FP32 partials scratch is
``at::empty`` (its coverage invariant is documented at
``alloc_wmma_partials`` in ``csrc/rocm/q_gemm_rdna3_wmma.cu``), and across
repeated calls the caching allocator hands back the same block — a slot
that ever went unwritten would be bitwise-repeatable *and* wrong.

Reference semantics (uint4b8, GPTQv1, synthesized zero points):
    W[k, n] = (q[k, n] - 8) * float(round_to_dtype(scale[g, n])),  g = k//G
    ref[m, n] = x[m, :] @ W[:, n]  (FP32 matmul)

Tolerances are derived from the kernel's rounding structure, not tuned to
pass (measured max errors on W7900/gfx1100 in parentheses):

* scalar bf16  — dequant keeps FP32 precision end-to-end (magic-value
  v_dot2 path); the only rounding is the final cast, so
  max_abs <= 0.5 * ulp_bf16(max|ref|) * 1.5 + 1e-3   (measured ~0.06).
* scalar fp16  — the classic exllama bit-trick rounds the per-(group,
  column) offset constant scale*(-1024-zero) to fp16 (~0.008 abs at
  scale≈0.02). That deterministic noise accumulates across the K/groups
  axis (sigma ≈ 0.24 at K=4096, measured max ≈ 1.0). This is pre-existing
  dequant behaviour shared with the exllama kernel — NOT epilogue error —
  so the tolerance is loose (2.5) and the tight fp16 check lives on the
  WMMA path below.
* WMMA bf16    — B is narrowed to bf16 per cell (one rounding,
  <= 0.5*ulp_bf16(|B|)), accumulated in FP32; measured max ≈ 0.09 at
  K=4096 — tolerance 0.25.
* WMMA fp16    — B narrowed to fp16 via the precise sub-then-mul variant
  (one rounding, ~1e-4); measured max ≈ 0.015 — tolerance 0.05.

All four are orders of magnitude below what an unwritten scratch slot
(caching-allocator garbage / NaN) or an indexing bug (errors of the same
scale as |ref| ≈ 7) would produce.

Test hygiene: the op path needs a single-process distributed group and a
current VLLM config; tests use the repo-standard ``dist_init`` fixture from
``tests/conftest.py`` (real context manager, temp-file rendezvous so no
hardcoded MASTER_PORT can collide in parallel CI, and the suite's standard
post-test cleanup).
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E402
    RDNA3W4A16LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (  # noqa: E402
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

device = "cuda"
WEIGHT_TYPE = scalar_types.uint4b8

gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
    ),
    reason="requires gfx1100 with the _rocm_C.gptq_gemm_rdna3 op built in",
)

REPEATS = 20
GROUP = 128
DTYPES = [torch.bfloat16, torch.float16]

# max_abs_error tolerances vs the FP32 reference, per (path, dtype).
# Derivation in the module docstring; ~2-3x headroom over measured maxima.
TOL = {
    ("scalar", torch.bfloat16): 0.10,
    ("scalar", torch.float16): 2.50,
    ("wmma", torch.bfloat16): 0.25,
    ("wmma", torch.float16): 0.05,
}


def _build_layer(k, n, seed, dtype):
    torch.manual_seed(seed)
    q_int4_kn = torch.randint(0, 16, (k, n), dtype=torch.int32)
    scales_gn = (torch.randn(k // GROUP, n) * 0.01 + 0.02).to(dtype)
    qweight = pack_quantized_values_into_int32(q_int4_kn, WEIGHT_TYPE,
                                               packed_dim=0)
    no_loader = lambda *a, **kw: None  # noqa: E731

    class DummyLayer(torch.nn.Module):
        pass

    layer = DummyLayer()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(data=qweight, weight_loader=no_loader, input_dim=0,
                            output_dim=1, packed_dim=0, packed_factor=8))
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(data=scales_gn, weight_loader=no_loader,
                                 input_dim=0, output_dim=1))
    layer.to(device)
    return layer, q_int4_kn, scales_gn


def _prepare(layer, dtype, k, n):
    cfg = MPLinearLayerConfig(
        full_weight_shape=(k, n), partition_weight_shape=(k, n),
        weight_type=WEIGHT_TYPE, act_type=dtype,
        group_size=GROUP, zero_points=False)
    kernel = RDNA3W4A16LinearKernel(cfg, w_q_param_name="qweight",
                                    w_s_param_name="scales",
                                    w_zp_param_name=None)
    kernel.process_weights_after_loading(layer)
    return kernel._get_weight_params(layer)


def _reference(m, k, dtype, seed, q_int4_kn, scales_gn):
    torch.manual_seed(seed + 1)
    x = torch.randn(m, k, device=device, dtype=dtype)
    # FP32 reference; see module docstring for the dequant semantics.
    w_f32 = (q_int4_kn.to(device).float() - 8.0) * scales_gn.to(
        device).repeat_interleave(GROUP, dim=0).float()
    return x, x.float() @ w_f32


def _run_op(x, w_q, w_zp, w_s):
    return torch.ops._rocm_C.gptq_gemm_rdna3(x, w_q, w_zp, w_s, False)


def _outputs_and_ref(M, K, N, seed, dtype, repeats=1):
    layer, q_int4_kn, scales_gn = _build_layer(K, N, seed, dtype)
    w_q, w_s, w_zp = _prepare(layer, dtype, K, N)
    x, ref = _reference(M, K, dtype, seed, q_int4_kn, scales_gn)
    outs = [_run_op(x, w_q, w_zp, w_s) for _ in range(repeats)]
    return outs, ref


def _path_for(dtype, m):
    """Public dispatch: bf16 reaches WMMA at M>=16, fp16 at M>=64."""
    if (dtype == torch.bfloat16 and m >= 16) or \
       (dtype == torch.float16 and m >= 64):
        return "wmma"
    return "scalar"


def _assert_repeatable(outs):
    for o in outs[1:]:
        assert torch.equal(o, outs[0]), (
            "RDNA3 W4A16 split-K produced differing outputs for identical "
            "inputs")


def _assert_close_to_ref(out, ref, path, dtype):
    err = (out.float() - ref).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    tol = TOL[(path, dtype)]
    assert torch.isfinite(out.float()).all(), "non-finite output"
    assert max_abs <= tol, (
        f"RDNA3 W4A16 {path} output deviates from the FP32 dequantized "
        f"reference: max_abs={max_abs:.4f} > tol={tol} (mean_abs="
        f"{mean_abs:.4f}). This catches values that are bitwise-repeatable "
        "but wrong (e.g. an unwritten FP32 scratch slot).")


# ---------------------------------------------------------------------------
# A. Repeatability (bit-exact across identical calls)
# ---------------------------------------------------------------------------

@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_splitk_bit_repeatable(dist_init, dtype):
    """Scalar path at production-like contention: K=4096 splits into
    ceil(4096/256) = 16 concurrent writers per output element (the old CAS
    bug's onset was between 2 and 4 writers, so this sits well inside the
    failure regime rather than at the empirical threshold)."""
    outs, _ = _outputs_and_ref(M=1, K=4096, N=4096, seed=1234,
                             dtype=dtype, repeats=REPEATS)
    _assert_repeatable(outs)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_small_m_bit_repeatable(dist_init, dtype):
    """Scalar M>1 tiles share the same deterministic epilogue."""
    outs, _ = _outputs_and_ref(M=8, K=4096, N=4096, seed=1235,
                             dtype=dtype, repeats=REPEATS)
    _assert_repeatable(outs)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_single_split_bit_repeatable(dist_init, dtype):
    """z_count == 1 (K=256 < BLOCK_KN_SIZE): direct-store epilogue, no
    scratch, no reduce pass."""
    outs, _ = _outputs_and_ref(M=1, K=256, N=4096, seed=1238,
                             dtype=dtype, repeats=REPEATS)
    _assert_repeatable(outs)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_splitk_bit_repeatable(dist_init, dtype):
    """WMMA path (bf16 M >= 16, fp16 M >= 64), split-K active.

    K=6656 gives K_SPLIT=4 under the upstream heuristic; the deterministic
    epilogue must be bit-repeatable regardless of the split count.
    """
    m = 16 if dtype == torch.bfloat16 else 64
    outs, _ = _outputs_and_ref(M=m, K=6656, N=4096, seed=1236,
                             dtype=dtype, repeats=REPEATS)
    _assert_repeatable(outs)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_large_m_bit_repeatable(dist_init, dtype):
    """Large-M WMMA tiles (128x64 kernels) with the deterministic epilogue."""
    outs, _ = _outputs_and_ref(M=128, K=6656, N=4096, seed=1237,
                             dtype=dtype, repeats=REPEATS)
    _assert_repeatable(outs)


# ---------------------------------------------------------------------------
# B. Numerical correctness against the FP32 dequantized reference
# ---------------------------------------------------------------------------

@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_matches_fp32_reference(dist_init, dtype):
    """Scalar split-K (K=4096, 16 writers) vs the FP32 reference."""
    m = 1
    outs, ref = _outputs_and_ref(M=m, K=4096, N=512, seed=1240,
                             dtype=dtype)
    _assert_close_to_ref(outs[0], ref, _path_for(dtype, m), dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_single_split_matches_fp32_reference(dist_init, dtype):
    """Scalar z_count == 1 direct-store path vs the FP32 reference."""
    m = 1
    outs, ref = _outputs_and_ref(M=m, K=256, N=512, seed=1241,
                             dtype=dtype)
    _assert_close_to_ref(outs[0], ref, _path_for(dtype, m), dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_matches_fp32_reference(dist_init, dtype):
    """WMMA split-K path (bf16 16x16_1w / fp16 64x64_4w) vs FP32 reference."""
    m = 16 if dtype == torch.bfloat16 else 64
    outs, ref = _outputs_and_ref(M=m, K=4096, N=512, seed=1242,
                             dtype=dtype)
    _assert_close_to_ref(outs[0], ref, _path_for(dtype, m), dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_single_split_matches_fp32_reference(dist_init, dtype):
    """WMMA k_split == 1 direct-store path (K=256 -> compute_wmma_k_split
    returns 1 for every dispatch level) vs the FP32 reference."""
    m = 16 if dtype == torch.bfloat16 else 64
    outs, ref = _outputs_and_ref(M=m, K=256, N=512, seed=1243,
                             dtype=dtype)
    _assert_close_to_ref(outs[0], ref, _path_for(dtype, m), dtype)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_large_m_matches_fp32_reference(dist_init, dtype):
    """Large-M WMMA (128x64_k32, row-tiled scratch + reduce) vs FP32."""
    m = 128
    outs, ref = _outputs_and_ref(M=m, K=4096, N=512, seed=1244,
                             dtype=dtype)
    _assert_close_to_ref(outs[0], ref, _path_for(dtype, m), dtype)


# ---------------------------------------------------------------------------
# C. V7/V8 (128x64) no-split path: k_split == 1 -> direct store, no scratch,
#    no reduce pass (partials == nullptr).
# ---------------------------------------------------------------------------

# The measured production-like shape from the W7900 benchmark campaign.
# Routing arithmetic (see _assert_v7v8_no_split_routing): with (128, 64)
# tiles, blocks_xy = ceil(25600/64) * ceil(512/128) = 400 * 4 = 1600 >=
# 1500, so compute_wmma_k_split_mn returns 1. Smaller no-split shapes are
# not practical: N would have to grow past ~24k at M=512 (or M past ~1k at
# N=12k) to reach the 1500-block threshold, so this is effectively the
# smallest production-realistic V7/V8 no-split shape.
V7V8_NOSPLIT = (512, 25600, 6656)  # (M, N, K)


def _compute_wmma_k_split_mn(m, n, k, m_tile, n_tile):
    """Faithful replica of compute_wmma_k_split_mn (C++ heuristic)."""
    blocks_xy = ((n + n_tile - 1) // n_tile) * ((m + m_tile - 1) // m_tile)
    if blocks_xy >= 1500:
        return 1
    if blocks_xy * 2 >= 1500 and k >= 512 and k % 32 == 0:
        return 2
    if blocks_xy * 4 >= 1500 and k >= 1024 and k % 64 == 0:
        return 4
    if k >= 1024 and k % 64 == 0:
        return 4
    if k >= 512 and k % 32 == 0:
        return 2
    return 1


def _assert_v7v8_no_split_routing(dtype, m, n, k):
    """Assert the shape deterministically routes to the V7/V8 128x64
    kernel with k_split == 1: WMMA dispatch (bf16 M>=16 / fp16 M>=64),
    the M >= 128 branch, and the no-split threshold."""
    assert (dtype == torch.bfloat16 and m >= 16) or \
           (dtype == torch.float16 and m >= 64), "not the WMMA path"
    assert m >= 128, "below the V7/V8 (128x64) branch"
    assert _compute_wmma_k_split_mn(m, n, k, 128, 64) == 1, \
        "shape does not route to the k_split == 1 no-split path"


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_v7v8_no_split_bit_repeatable(dist_init, dtype):
    """V7/V8 128x64_k32 with k_split == 1: single direct writer per output
    cell — no FP32 scratch, no reduce pass."""
    m, n, k = V7V8_NOSPLIT
    _assert_v7v8_no_split_routing(dtype, m, n, k)
    outs, _ = _outputs_and_ref(M=m, K=k, N=n, seed=1245,
                             dtype=dtype, repeats=REPEATS)
    _assert_repeatable(outs)


@gfx1100_only
@pytest.mark.parametrize("dtype", DTYPES)
def test_wmma_v7v8_no_split_matches_fp32_reference(dist_init, dtype):
    """V7/V8 128x64_k32 no-split direct-store path vs the FP32 reference."""
    m, n, k = V7V8_NOSPLIT
    _assert_v7v8_no_split_routing(dtype, m, n, k)
    outs, ref = _outputs_and_ref(M=m, K=k, N=n, seed=1246,
                             dtype=dtype)
    _assert_close_to_ref(outs[0], ref, _path_for(dtype, m), dtype)
