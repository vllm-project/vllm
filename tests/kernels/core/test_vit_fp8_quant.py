# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the stride-aware FP8 quantization kernel with head_dim padding."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.kernels.triton.qkv_padded_fp8_quant import (
        quantize_fp8_pad_head_dim_triton,
    )

HEAD_DIMS = [72, 80, 128]
SEQ_LENS = [64, 256]
NUM_HEADS = [16]
SCALES = [0.01, 0.1, 1.0]


def _naive_fp8_quantize(
    tensor: torch.Tensor, scale: torch.Tensor, skip_scale: bool
) -> torch.Tensor:
    """Reference FP8 quantization in PyTorch."""
    fp8_dtype = current_platform.fp8_dtype()
    fp8_min, fp8_max = get_fp8_min_max()

    x = tensor.float()
    if not skip_scale:
        x = x / scale.item()
    x = x.clamp(fp8_min, fp8_max)
    return x.to(fp8_dtype)


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("scale_val", SCALES)
def test_quantize_contiguous(
    head_dim: int, seq_len: int, num_heads: int, scale_val: float
) -> None:
    """Test quantization of contiguous 3D tensors."""
    torch.manual_seed(42)
    tensor = torch.randn(
        seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda").view(
        1, 1, 1, 1
    )

    result = quantize_fp8_pad_head_dim_triton(tensor, scale)

    padded_dim = (head_dim + 15) // 16 * 16
    assert result.shape == (seq_len, num_heads, padded_dim)
    assert result.is_contiguous()
    assert result.dtype == current_platform.fp8_dtype()

    # Compare unpadded portion against reference
    ref = _naive_fp8_quantize(tensor, scale, skip_scale=False)
    torch.testing.assert_close(result[:, :, :head_dim].float(), ref.float())

    # Padded region should be zero
    if padded_dim > head_dim:
        assert (result[:, :, head_dim:].float() == 0).all()


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
@pytest.mark.parametrize("head_dim", [72, 80])
def test_quantize_non_contiguous(head_dim: int) -> None:
    """Test quantization from non-contiguous QKV views (interleaved buffer)."""
    seq_len, num_heads = 64, 16
    # Simulate interleaved QKV buffer: shape (seq_len, 3 * num_heads, head_dim)
    qkv = torch.randn(
        seq_len, 3 * num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    # Q is every 3rd head slice - non-contiguous view
    q = qkv[:, 0::3, :]
    assert not q.is_contiguous()

    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda").view(1, 1, 1, 1)
    result = quantize_fp8_pad_head_dim_triton(q, scale)

    padded_dim = (head_dim + 15) // 16 * 16
    assert result.shape == (seq_len, num_heads, padded_dim)
    assert result.is_contiguous()

    # Compare against contiguous reference
    ref = _naive_fp8_quantize(q.contiguous(), scale, skip_scale=False)
    torch.testing.assert_close(result[:, :, :head_dim].float(), ref.float())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_skip_scale() -> None:
    """Test skip_scale=True produces cast-only output (no division)."""
    seq_len, num_heads, head_dim = 32, 8, 80
    tensor = torch.randn(
        seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda").view(1, 1, 1, 1)

    result_skip = quantize_fp8_pad_head_dim_triton(tensor, scale, skip_scale=True)
    result_noskip = quantize_fp8_pad_head_dim_triton(tensor, scale, skip_scale=False)

    # skip_scale should just cast, not divide
    ref_cast = _naive_fp8_quantize(tensor, scale, skip_scale=True)
    torch.testing.assert_close(result_skip[:, :, :head_dim].float(), ref_cast.float())

    # With scale != 1.0, skip and no-skip should differ
    assert not torch.equal(result_skip.float(), result_noskip.float())


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
def test_4d_input() -> None:
    """Test that 4D input (B, S, H, D) is handled correctly."""
    B, S, H, D = 2, 32, 8, 72
    tensor = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda").view(1, 1, 1, 1)

    result = quantize_fp8_pad_head_dim_triton(tensor, scale)
    padded_dim = (D + 15) // 16 * 16
    assert result.shape == (B, S, H, padded_dim)


# ---------------------------------------------------------------------------
# CUDA tests: validate that torch.ops._C.qkv_padded_fp8_quant
# matches the Triton reference bit-exactly.
# ---------------------------------------------------------------------------
_HAS_CUDA_OP = hasattr(torch.ops._C, "qkv_padded_fp8_quant")


@pytest.mark.skipif(
    not (current_platform.is_cuda() and _HAS_CUDA_OP and HAS_TRITON),
    reason="Requires CUDA + qkv_padded_fp8_quant op + Triton",
)
@pytest.mark.parametrize("head_dim", [72, 80, 96, 112, 120, 128])
@pytest.mark.parametrize("seq_len", [1, 64, 256, 4096])
@pytest.mark.parametrize("num_heads", [1, 16])
@pytest.mark.parametrize("scale_val", [0.01, 1.0, 100.0])
@pytest.mark.parametrize("skip_scale", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_matches_triton(
    head_dim: int,
    seq_len: int,
    num_heads: int,
    scale_val: float,
    skip_scale: bool,
    dtype: torch.dtype,
) -> None:
    """CUDA op must produce bit-identical output to the Triton kernel."""
    torch.manual_seed(0)
    x = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    scale = torch.tensor([scale_val], device="cuda", dtype=torch.float32)

    y_triton = quantize_fp8_pad_head_dim_triton(x, scale, skip_scale=skip_scale)
    y_cuda = torch.ops._C.qkv_padded_fp8_quant(x, scale, skip_scale)

    assert y_cuda.shape == y_triton.shape
    assert y_cuda.dtype == y_triton.dtype
    assert y_cuda.is_contiguous()
    # FP8 quantization is deterministic given the same scale; require an exact
    # match (this also catches sign/clamp/zero-padding regressions).
    torch.testing.assert_close(y_cuda.float(), y_triton.float(), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and _HAS_CUDA_OP and HAS_TRITON),
    reason="Requires CUDA + qkv_padded_fp8_quant op + Triton",
)
@pytest.mark.parametrize("head_dim", [72, 80])
def test_cuda_non_contiguous_qkv(head_dim: int) -> None:
    """Non-contiguous interleaved QKV slice (q = qkv[:, 0::3, :]).

    The slice keeps stride(-1) == 1, so the CUDA  applies.
    Result must still match the Triton reference bit-exactly.
    """
    seq_len, num_heads = 64, 16
    qkv = torch.randn(
        seq_len, 3 * num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    q = qkv[:, 0::3, :]
    assert q.stride(-1) == 1, "interleaved slice should keep contiguous head_dim"
    assert not q.is_contiguous()

    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    y_triton = quantize_fp8_pad_head_dim_triton(q, scale)
    y_cuda = torch.ops._C.qkv_padded_fp8_quant(q, scale, False)

    torch.testing.assert_close(y_cuda.float(), y_triton.float(), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and _HAS_CUDA_OP),
    reason="Requires CUDA + qkv_padded_fp8_quant op",
)
def test_cuda_padding_region_is_zero() -> None:
    """Padding lanes [D, padded_D) MUST be exact FP8 +0; cuDNN attention
    relies on this to avoid garbage out-of-range scores."""
    x = torch.randn(128, 16, 72, device="cuda", dtype=torch.bfloat16)
    # Use a tiny scale so unpadded lanes saturate to FP8 max -> we know they
    # are NOT zero, and the only zeros must come from the padding region.
    scale = torch.tensor([1e-5], dtype=torch.float32, device="cuda")
    y = torch.ops._C.qkv_padded_fp8_quant(x, scale, False)
    assert y.shape == (128, 16, 80)
    assert (y[:, :, 72:].float() == 0.0).all(), "padding region must be zero"


@pytest.mark.skipif(
    not (current_platform.is_cuda() and _HAS_CUDA_OP and HAS_TRITON),
    reason="Requires CUDA + qkv_padded_fp8_quant op + Triton",
)
@pytest.mark.parametrize(
    "shape",
    [(2304, 16, 72), (4096, 16, 72), (16384, 16, 72)],
)
def test_cuda_large_shapes_match_triton(shape) -> None:
    """Large ViT-style shapes (Qwen3-VL) — same answer as Triton."""
    S, H, D = shape
    x = torch.randn(S, H, D, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    y_triton = quantize_fp8_pad_head_dim_triton(x, scale)
    y_cuda = torch.ops._C.qkv_padded_fp8_quant(x, scale, False)
    torch.testing.assert_close(y_cuda.float(), y_triton.float(), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and _HAS_CUDA_OP and HAS_TRITON),
    reason="Requires CUDA + qkv_padded_fp8_quant op + Triton",
)
def test_cuda_4d_input_matches_triton() -> None:
    """4D (B, S, H, D) input is flattened to 3D in the dispatcher."""
    from vllm.kernels.triton.qkv_padded_fp8_quant import quantize_fp8_pad_head_dim

    x = torch.randn(2, 64, 16, 72, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    y_dispatch = quantize_fp8_pad_head_dim(x, scale)
    y_triton = quantize_fp8_pad_head_dim_triton(x, scale)
    assert y_dispatch.shape == y_triton.shape == (2, 64, 16, 80)
    torch.testing.assert_close(y_dispatch.float(), y_triton.float(), rtol=0.0, atol=0.0)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and _HAS_CUDA_OP and HAS_TRITON),
    reason="Requires CUDA + qkv_padded_fp8_quant op + Triton",
)
def test_cuda_correctness_edge_cases() -> None:
    """Comprehensive edge-case validation for the CUDA kernel.

    Covers: extreme values (inf/nan/saturation), all-zero input, aligned
    head_dim (no padding needed), tiny / huge scales (clamp boundary),
    odd num_heads, single-token shapes, multi-batch interleaved slices,
    and large head_dim that triggers the generic 2D kernel path.
    """
    triton_fn = quantize_fp8_pad_head_dim_triton
    cuda_op = torch.ops._C.qkv_padded_fp8_quant

    # ----- Case 1: extreme values (inf/nan) -----

    x = torch.zeros(64, 16, 72, device="cuda", dtype=torch.bfloat16)
    x[0, 0, 0] = float("inf")
    x[0, 0, 1] = float("-inf")
    x[1, 0, 0] = float("nan")
    x[2, 0, 0] = 1e30  # far above FP8 upper bound; must be clamped
    x[3, 0, 0] = -1e30
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    y_triton = triton_fn(x, scale)
    y_cuda = cuda_op(x, scale, False)
    # nan != nan, so use equal_nan=True so that nan slots match.
    torch.testing.assert_close(
        y_cuda.float(), y_triton.float(), rtol=0.0, atol=0.0, equal_nan=True
    )
    # Non-nan extreme values must fall inside [-448, 448].
    finite_mask = ~torch.isnan(y_cuda.float())
    finite_vals = y_cuda.float()[finite_mask]
    assert (finite_vals.abs() <= 448.0).all(), "non-nan values must be clamped"

    # ----- Case 2: all-zero input -> all-zero output -------------------
    # Sanity identity: 0 / scale == 0; clamp(0) == 0; fp8(0) == 0.
    x = torch.zeros(128, 16, 72, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    y_cuda = cuda_op(x, scale, False)
    assert (y_cuda.float() == 0.0).all(), "all-zero input -> all-zero output"
    assert y_cuda.shape == (128, 16, 80)

    # ----- Case 3: head_dim already aligned (no padding emitted) -------
    # D=128 is already covered by test_cuda_matches_triton; here we
    # just spot-check the small / mid BLOCK_M templates.
    for d in [16, 64]:
        x = torch.randn(64, 8, d, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(x, scale)
        y_cuda = cuda_op(x, scale, False)
        assert y_cuda.shape == (64, 8, d), f"D={d}: expected no padding"
        torch.testing.assert_close(y_cuda.float(), y_triton.float(), rtol=0.0, atol=0.0)

    # ----- Case 4: extreme scale values (clamp boundary) ---------------
    # Mid-range scales (0.01 / 100) are already in the parameterized
    # matrix; here we test only the most extreme ends to confirm that
    # under/overflow paths still bit-match Triton.
    x = torch.randn(64, 16, 72, device="cuda", dtype=torch.bfloat16)
    for scale_val in [1e-30, 1e10]:
        scale = torch.tensor([scale_val], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(x, scale)
        y_cuda = cuda_op(x, scale, False)
        torch.testing.assert_close(
            y_cuda.float(),
            y_triton.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"scale={scale_val} mismatch",
        )

    # ----- Case 5: non-power-of-two num_heads --------------------------
    # H=1 is already in the parameterized matrix; the remaining odd
    # values all exercise the same `s = row / H` integer-divide path,
    # so two representative values (small / mid) are sufficient.
    for h in [3, 17]:
        x = torch.randn(32, h, 72, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(x, scale)
        y_cuda = cuda_op(x, scale, False)
        torch.testing.assert_close(
            y_cuda.float(),
            y_triton.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"num_heads={h} mismatch",
        )

    # ----- Case 6: tiny seq_len (grid-edge early-return) ---------------
    # seq_len=1 is already in the parameterized matrix; pick S=2
    for s in [2, 17]:
        x = torch.randn(s, 16, 72, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(x, scale)
        y_cuda = cuda_op(x, scale, False)
        torch.testing.assert_close(
            y_cuda.float(),
            y_triton.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"seq_len={s} mismatch",
        )

    # ----- Case 7: padding path with row-aligned strides ----------------
    # Goal: exercise the dcoal kernel's "real-data + zero-padding" path
    # (D < padded_D) while keeping inputs on the CUDA  path.
    #
    # We pick D in {8, 24, 40, 56, 88, 104, 120}: each value is a multiple
    # of 8 (so the per-row stride stride_h * sizeof(bf16) is 16B aligned,
    # satisfying the uint4 vectorized load) but NOT a multiple of 16 (so
    # the kernel must produce padding_D - D zero lanes per row).
    #
    # Misaligned strides (e.g. D=70/78) are the dispatcher's responsibility
    # to route to Triton -- see Case 7b.
    # Three D values cover small / mid / large BLOCK_M classes:
    #   D=8   -> padded_D=16  (BLOCK_M=128)
    #   D=56  -> padded_D=64  (BLOCK_M=32)
    #   D=104 -> padded_D=112 (BLOCK_M=16)
    for d in [8, 56, 104]:
        padded_d = (d + 15) // 16 * 16
        x = torch.randn(64, 16, d, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(x, scale)
        y_cuda = cuda_op(x, scale, False)
        assert y_cuda.shape == (64, 16, padded_d)
        torch.testing.assert_close(
            y_cuda.float(),
            y_triton.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"D={d} -> padded_D={padded_d} mismatch",
        )
        # Strict check: padding zone must be zero.
        if padded_d > d:
            assert (y_cuda[:, :, d:].float() == 0).all(), (
                f"padding zone [D={d}, padded_D={padded_d}) must be zero"
            )

    # ----- Case 8: head_dim > 128 triggers the generic 2D kernel ------
    # Two values cover the boundary (just over 128) and a much larger
    # head_dim that forces multiple D-tiles per row.
    for d in [144, 256]:
        x = torch.randn(64, 8, d, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(x, scale)
        y_cuda = cuda_op(x, scale, False)
        torch.testing.assert_close(
            y_cuda.float(),
            y_triton.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"generic kernel D={d} mismatch",
        )

    # ----- Case 9: interleaved QKV slices (k = qkv[:, 1::3], v = 2::3) -
    # offset=0 is already covered by test_cuda_non_contiguous_qkv; we
    # add offset=1 and offset=2 to verify that non-zero base offsets
    # (which produce a different starting data_ptr) still work.
    seq_len, num_heads, head_dim = 64, 16, 72
    qkv = torch.randn(
        seq_len, 3 * num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    for offset in [1, 2]:
        sliced = qkv[:, offset::3, :]
        assert sliced.stride(-1) == 1
        scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
        y_triton = triton_fn(sliced, scale)
        y_cuda = cuda_op(sliced, scale, False)
        torch.testing.assert_close(
            y_cuda.float(),
            y_triton.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"qkv slice offset={offset} mismatch",
        )

    # ----- Case 10: device consistency ---------------------------------
    # The output tensor must be on the same CUDA device as the input
    # (matters for vLLM's multi-GPU scenarios).
    x = torch.randn(64, 16, 72, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    y_cuda = cuda_op(x, scale, False)
    assert y_cuda.device == x.device, "output must be on same device as input"

    # ----- Case 11: skip_scale must ignore the scale value -------------
    # When skip_scale=True the kernel must skip the divide-by-scale step.
    # Even with scale = 0 / inf / nan the result must be the cast-only
    # output (clamp + cast, no scale read).
    x = torch.randn(32, 16, 72, device="cuda", dtype=torch.bfloat16)
    scale_normal = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    scale_zero = torch.tensor([0.0], dtype=torch.float32, device="cuda")
    scale_inf = torch.tensor([float("inf")], dtype=torch.float32, device="cuda")
    y_normal = cuda_op(x, scale_normal, True)
    y_zero = cuda_op(x, scale_zero, True)
    y_inf = cuda_op(x, scale_inf, True)
    # The reference implementation does not pad, so only compare the
    # first 72 columns (D); the padding zone is checked separately.
    # All three CUDA outputs must equal the cast-only reference.
    ref = _naive_fp8_quantize(x, scale_normal, skip_scale=True)
    for tag, y in [("normal", y_normal), ("zero", y_zero), ("inf", y_inf)]:
        assert y.shape == (32, 16, 80), f"skip_scale[{tag}] shape mismatch"
        torch.testing.assert_close(
            y[:, :, :72].float(),
            ref.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"skip_scale[{tag}] must ignore scale value",
        )
        assert (y[:, :, 72:].float() == 0).all(), (
            f"skip_scale[{tag}]: padding zone must be zero"
        )

    # ----- Case 12: output is contiguous + dtype + stride ---------------
    x = torch.randn(64, 16, 72, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    y_cuda = cuda_op(x, scale, False)
    assert y_cuda.is_contiguous(), "output must be contiguous"
    assert y_cuda.dtype == current_platform.fp8_dtype()
    # cuDNN attention requires stride(-1) == 1; enforced strictly.
    assert y_cuda.stride(-1) == 1
    assert y_cuda.stride(-2) == 80
    assert y_cuda.stride(-3) == 16 * 80

    # NOTE: 4D batch input is covered by test_cuda_4d_input_matches_triton.
    # NOTE: fp16 dtype is covered by test_cuda_matches_triton (288 cases).

    # ----- Case 13: deterministic across repeated calls ----------------
    # Calling the kernel 5 times with the same input must produce the
    # exact same output (no hidden global state, no uninitialized
    # shared memory, no race conditions).
    x = torch.randn(64, 16, 72, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([0.1], dtype=torch.float32, device="cuda")
    y_first = cuda_op(x, scale, False).clone()
    for _ in range(5):
        y_again = cuda_op(x, scale, False)
        torch.testing.assert_close(
            y_first.float(),
            y_again.float(),
            rtol=0.0,
            atol=0.0,
            msg="repeated call must be deterministic",
        )
