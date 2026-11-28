# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import patch

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.quantization.utils import fp8_utils, int8_utils
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128),
        (1, 128),
        (3, 384),
        (5, 384),
        (7, 384),
        (15, 384),
        (31, 128),
        (32, 128),
        (63, 256),
        (64, 256),
        (16, 512),
        (3, 2688),
        (3, 5376),
        (333, 128),
        (512, 6144),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("tma_aligned", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Only test on CUDA/ROCm/XPU.",
)
def test_per_token_group_quant_fp8(
    shape,
    dtype: torch.dtype,
    column_major: bool,
    tma_aligned: bool,
    scale_ue8m0: bool,
    group_size: int,
):
    """Tail groups preserve FP8 values and scales in every supported layout."""
    device = current_platform.device_type

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=dtype) * 8

    # native kernel path
    out_q, scale = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        use_ue8m0=scale_ue8m0,
    )

    # triton ref
    with (
        patch("vllm.platforms.current_platform.is_cuda_alike", return_value=False),
        patch("vllm.platforms.current_platform.is_xpu", return_value=False),
    ):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=column_major,
            use_ue8m0=scale_ue8m0,
        )

    assert torch.allclose(out_q.float(), ref_q.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(scale, ref_s, atol=0.01, rtol=0.01)


def _misaligned_clone(x: torch.Tensor) -> torch.Tensor:
    """Contiguous copy of `x` whose `data_ptr` is not 16B aligned."""
    numel = x.numel()
    base = torch.empty(numel + 16 // x.element_size(), device=x.device, dtype=x.dtype)
    view = base[1 : 1 + numel]
    if view.data_ptr() % 16 == 0:
        pytest.skip("allocator did not yield a misaligned offset")
    return view.view(x.shape).copy_(x)


@pytest.mark.parametrize("shape", [(1, 128), (333, 128), (1024, 6144)])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="register-resident schedule is CUDA/ROCm only",
)
def test_per_token_group_quant_fp8_register_matches_smem(
    shape, column_major: bool, scale_ue8m0: bool, dtype: torch.dtype
):
    """group_size=128 with 16-bit input takes a register-resident schedule that
    needs 16B-aligned buffers; a misaligned input falls back to the
    shared-memory schedule. The two must agree bit-for-bit, which the
    tolerance-based comparison against Triton above cannot establish.
    """
    device = current_platform.device_type
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype) * 8

    reg_q, reg_s = fp8_utils.per_token_group_quant_fp8(
        x, 128, column_major_scales=column_major, use_ue8m0=scale_ue8m0
    )
    smem_q, smem_s = fp8_utils.per_token_group_quant_fp8(
        _misaligned_clone(x),
        128,
        column_major_scales=column_major,
        use_ue8m0=scale_ue8m0,
    )

    assert torch.equal(reg_q.view(torch.uint8), smem_q.view(torch.uint8))
    assert torch.equal(reg_s, smem_s)


@pytest.mark.parametrize(
    "num_tokens,hidden_dim,group_size",
    [
        # No padding: mn=4 (mult of 4), groups_per_row=56 (mult of 4)
        (4, 7168, 128),
        # MN padding only: mn=1, tma_aligned_mn=4
        (1, 7168, 128),
        # MN padding only: mn=3, tma_aligned_mn=4
        (3, 7168, 128),
        # K padding only: groups_per_row=5 (5%4=1)
        (4, 640, 128),
        # K padding only: groups_per_row=6 (6%4=2)
        (4, 768, 128),
        # Single packed column, no padding: k_num_packed=1, mn%4=0
        (4, 384, 128),
        # Both MN and K padding
        (1, 384, 128),
        (3, 640, 128),
        # Larger shapes with no padding
        (64, 7168, 128),
        (128, 14336, 128),
        # Larger shapes with padding
        (127, 7168, 128),
        (253, 640, 128),
        (1, 32, 32),
        (3, 64, 32),
        (7, 160, 32),
        (32, 512, 32),
        (127, 4096, 32),
    ],
)
@pytest.mark.parametrize("poisoned_scales", [False, True])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DeepGEMM not available on this platform",
)
def test_per_token_group_quant_fp8_packed(
    num_tokens, hidden_dim, group_size, poisoned_scales
):
    """Test the packed DeepGEMM quantization kernel against the Triton
    reference (row-major, UE8M0 scales)."""
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4
    num_scale_elems = mn + (k_num_packed - 1) * tma_aligned_mn

    if poisoned_scales:
        # Call the kernel with poisoned scale buffer to
        # ensure padded indices are correctly zeroed.
        fp8_dtype = current_platform.fp8_dtype()
        fp8_min, fp8_max = get_fp8_min_max()
        out_q = torch.empty_like(x, dtype=fp8_dtype)
        out_s_packed = torch.empty_strided(
            (mn, k_num_packed),
            (1, tma_aligned_mn),
            device=device,
            dtype=torch.int32,
        )
        torch.as_strided(out_s_packed, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)
        torch.ops._C.per_token_group_fp8_quant_packed(
            x,
            out_q,
            out_s_packed,
            group_size,
            1e-10,
            fp8_min,
            fp8_max,
        )
    else:
        out_q, out_s_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
            x,
            group_size=group_size,
            use_ue8m0=True,
        )

    # Triton reference (row-major float32 scales, UE8M0)
    with patch("vllm.platforms.current_platform.is_cuda_alike", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            use_ue8m0=True,
        )

    # Quantized values must match.
    assert torch.equal(out_q, ref_q), "Quantized output mismatch"

    # Verify packed scales (valid exponents + padding zeros).
    ref_s_flat = ref_s.reshape(mn, groups_per_row)
    ref_exponents = (ref_s_flat.view(torch.int32) >> 23) & 0xFF

    expected = torch.zeros(num_scale_elems, dtype=torch.int32, device="cpu")
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col = g // 4
            pos = g % 4
            idx = pack_col * tma_aligned_mn + row
            expected[idx] |= int(ref_exponents[row, g].item()) << (pos * 8)

    actual = torch.as_strided(out_s_packed, (num_scale_elems,), (1,)).cpu()
    assert torch.equal(actual, expected), (
        f"Packed scale storage mismatch.\n"
        f"First diff at index "
        f"{(actual != expected).nonzero(as_tuple=True)[0][0].item()}"
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DeepGEMM not available on this platform",
)
def test_per_token_group_quant_fp8_packed_all_zero():
    """All-zero input must produce well-defined UE8M0 scale bytes via the eps
    floor in the kernel's UE8M0 path. Locks down the all-zero behavior before
    optimization.

    The CUDA kernel computes:
        y_s = eps / fp8_max
        y_s = exp2(ceil(log2(fmax(y_s, 1e-10))))
    For all-zero input, eps/fp8_max < 1e-10, so the inner fmax clamps back to
    1e-10, giving exp2(ceil(log2(1e-10))) = exp2(-33) => UE8M0 byte 0x5E (94).
    """
    device = "cuda"
    num_tokens, hidden_dim, group_size = 4, 7168, 128
    x = torch.zeros((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16)

    out_q, out_s_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
        x,
        group_size=group_size,
        use_ue8m0=True,
    )

    # Quantized values must be all zero.
    assert torch.equal(
        out_q.view(torch.uint8),
        torch.zeros_like(out_q, dtype=torch.uint8),
    ), "All-zero input should produce all-zero FP8 output"

    # UE8M0 byte produced by the kernel for all-zero input.
    # The kernel's inner fmax(y_s, 1e-10) clamps eps/fp8_max back to 1e-10.
    # 1e-10 as float32 has biased exponent 0x5D and a non-zero mantissa, so
    # the kernel's bit-twiddle (exp_bits + (mant_bits != 0)) rounds up to
    # 0x5E. This matches exp2(ceil(log2(1e-10))) = exp2(-33).
    expected_exp_byte = 0x5E

    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4
    num_scale_elems = mn + (k_num_packed - 1) * tma_aligned_mn

    # All valid scale slots must contain the expected packed value.
    # Padding slots must be zero.
    actual = torch.as_strided(out_s_packed, (num_scale_elems,), (1,)).cpu()

    expected = torch.zeros(num_scale_elems, dtype=torch.int32, device="cpu")
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col = g // 4
            pos = g % 4
            idx = pack_col * tma_aligned_mn + row
            expected[idx] |= expected_exp_byte << (pos * 8)

    assert torch.equal(actual, expected), "All-zero scale bytes mismatch"


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DeepGEMM not available on this platform",
)
def test_per_token_group_quant_fp8_packed_mantissa_rounds_up():
    """Inputs whose absmax/max_8bit produces a non-power-of-2 force the
    mantissa-rounding-up branch (exp_byte += 1). Locks down this behavior
    before optimization."""
    device = "cuda"
    num_tokens, hidden_dim, group_size = 4, 7168, 128

    # Build a tensor whose per-group absmax = 1.5 * fp8_max * 2^k for various k.
    # fp8_max = torch.finfo(torch.float8_e4m3fn).max = 448.0.
    # Then absmax/fp8_max = 1.5 * 2^k -> non-zero mantissa, triggers ceil
    # rounding to 2^(k+1). Use k=0 for simplicity; the bf16 representation of
    # 1.5*448=672.0 is exact.
    x = torch.full(
        (num_tokens, hidden_dim),
        672.0,
        device=device,
        dtype=torch.bfloat16,
    )

    out_q, out_s_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
        x,
        group_size=group_size,
        use_ue8m0=True,
    )

    with patch("vllm.platforms.current_platform.is_cuda_alike", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            use_ue8m0=True,
        )

    assert torch.equal(out_q, ref_q), "Quantized output mismatch"

    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4
    num_scale_elems = mn + (k_num_packed - 1) * tma_aligned_mn

    ref_s_flat = ref_s.reshape(mn, groups_per_row)
    ref_exponents = (ref_s_flat.view(torch.int32) >> 23) & 0xFF
    expected = torch.zeros(num_scale_elems, dtype=torch.int32, device="cpu")
    for row in range(mn):
        for g in range(groups_per_row):
            pack_col = g // 4
            pos = g % 4
            idx = pack_col * tma_aligned_mn + row
            expected[idx] |= int(ref_exponents[row, g].item()) << (pos * 8)

    actual = torch.as_strided(out_s_packed, (num_scale_elems,), (1,)).cpu()
    assert torch.equal(actual, expected), "Scale bytes mismatch"


@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    [
        (1, 7168),  # mn padded 1 -> 4
        (2, 7168),  # mn padded 2 -> 4
        (3, 7168),  # mn padded 3 -> 4
        (5, 7168),  # mn padded 5 -> 8
        (127, 7168),  # mn padded 127 -> 128
        (253, 640),  # both mn and groups padded
        (1, 384),  # extreme: 1 group, 1 mn row -> both axes padded
    ],
)
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DeepGEMM not available on this platform",
)
def test_per_token_group_quant_fp8_packed_zero_fills_padded_output_q(
    num_tokens, hidden_dim
):
    """When output_q is allocated with shape (tma_aligned_mn, k) instead of
    (mn, k), the kernel must overwrite the padded mn rows with zeros so
    callers can use `torch.empty` instead of `torch.zeros`."""
    device = "cuda"
    group_size = 128
    torch.manual_seed(42)
    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((mn + 3) // 4) * 4

    fp8_dtype = current_platform.fp8_dtype()
    fp8_min, fp8_max = get_fp8_min_max()
    # Allocate output_q with the padded mn extent and pre-fill with 0xFF
    # so the kernel cannot rely on a clean buffer.
    out_q = torch.empty((tma_aligned_mn, hidden_dim), device=device, dtype=fp8_dtype)
    out_q.view(torch.uint8).fill_(0xFF)

    out_s_packed = torch.empty_strided(
        (mn, k_num_packed),
        (1, tma_aligned_mn),
        device=device,
        dtype=torch.int32,
    )

    torch.ops._C.per_token_group_fp8_quant_packed(
        x, out_q, out_s_packed, group_size, 1e-10, fp8_min, fp8_max
    )

    # Live rows must match the Triton reference.
    with patch("vllm.platforms.current_platform.is_cuda_alike", return_value=False):
        ref_q, _ = fp8_utils.per_token_group_quant_fp8(x, group_size, use_ue8m0=True)
    assert torch.equal(out_q[:mn], ref_q), "Live region mismatch"

    # Padded rows must be all-zero; without this, downstream TMA loads would
    # see uninitialised data.
    if tma_aligned_mn > mn:
        padded_bytes = out_q[mn:tma_aligned_mn].view(torch.uint8)
        assert padded_bytes.eq(0).all(), (
            f"Padded rows [{mn}, {tma_aligned_mn}) not zeroed; "
            f"{padded_bytes.ne(0).sum().item()} non-zero bytes"
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="packed FP8 per-token-group quant kernel requires a CUDA-alike GPU",
)
def test_per_token_group_quant_fp8_packed_large_mn():
    """Regression test for https://github.com/vllm-project/vllm/issues/45099.

    Some background: gridDim.x and gridDim.y have different limits of 2^31 - 1 and
    2^16 - 1, respectively.
    Prior code introduced a bug where it incorrectly assumed grid.x and y both have
    2^31 - 1 limits and mixed them up, which doesn't surface until the kernel is
    launched with a large mn that exceeds grid.y limit (2^16 - 1).

    This issue doesn't surface often because each forward pass only processes a
    bounded token batch, not the full context.
    Quantizing tensors with more rows than that will fail at launch with
    "CUDA error: invalid argument".
    This is a differential test that compares fp8 output against Triton output
    reference when token size sits just above the gridDim.y 2^16 - 1 limit.
    """
    device = "cuda"
    group_size = 128
    # hidden 2048 -> 2048/128 = 16 groups per row -> kx=16, ry=1: one grid row per mn
    # row, so any mn > 65535 overflowed grid.y before the fix.
    num_tokens, hidden_dim = 65537, 2048
    torch.manual_seed(42)
    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    out_q, out_s_packed = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
        x,
        group_size=group_size,
        use_ue8m0=True,
    )

    with patch("vllm.platforms.current_platform.is_cuda_alike", return_value=False):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x, group_size, use_ue8m0=True
        )

    assert torch.equal(out_q, ref_q), "Quantized output mismatch"

    # Vectorized packed-scale check; the per-element loop used by the smaller
    # tests is too slow at this size. groups_per_row is a multiple of 4 here,
    # so there is no K padding and the packed view lines up.
    mn = num_tokens
    groups_per_row = hidden_dim // group_size
    k_num_packed = (groups_per_row + 3) // 4
    assert groups_per_row % 4 == 0
    ref_exponents = (ref_s.reshape(mn, groups_per_row).view(torch.int32) >> 23) & 0xFF
    exp = ref_exponents.view(mn, k_num_packed, 4)
    expected = (
        exp[..., 0] | (exp[..., 1] << 8) | (exp[..., 2] << 16) | (exp[..., 3] << 24)
    )
    assert torch.equal(out_s_packed.cpu(), expected.cpu()), "Packed scale mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        (0, 128),
        (1, 128),
        (3, 384),
        (5, 384),
        (7, 384),
        (15, 384),
        (32, 128),
        (64, 256),
        (16, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Only test on CUDA/ROCm/XPU.",
)
def test_per_token_group_quant_int8(shape, dtype, group_size: int):
    """The shared launch policy also handles INT8 tail groups."""
    device = current_platform.device_type

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=dtype) * 8

    # cuda path
    out_q, scale = int8_utils.per_token_group_quant_int8(
        x,
        group_size,
    )

    # triton ref
    with patch("vllm.platforms.current_platform.is_cuda_alike", return_value=False):
        ref_q, ref_s = int8_utils.per_token_group_quant_int8(
            x,
            group_size,
        )

    assert torch.allclose(out_q.float(), ref_q.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(scale, ref_s, atol=0.01, rtol=0.01)


@pytest.mark.parametrize("shape", [(1, 128), (333, 128), (1024, 6144)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="register-resident schedule is CUDA/ROCm only",
)
def test_per_token_group_quant_int8_register_matches_smem(shape, dtype: torch.dtype):
    """int8 output shares the register-resident schedule with fp8, so it needs
    the same bit-exactness guard against the shared-memory schedule.
    """
    device = current_platform.device_type
    torch.manual_seed(42)
    x = torch.randn(shape, device=device, dtype=dtype) * 8

    reg_q, reg_s = int8_utils.per_token_group_quant_int8(x, 128)
    smem_q, smem_s = int8_utils.per_token_group_quant_int8(_misaligned_clone(x), 128)

    assert torch.equal(reg_q, smem_q)
    assert torch.equal(reg_s, smem_s)


# SM90 UE8M0 row quantization: native byte and scale contract.
_ROW_GROUP_SIZE = 128
_ROW_EPS = 1e-10
_ROW_FP8_DTYPE = torch.float8_e4m3fn
_requires_sm90_row_quant = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(90)),
    reason="SM90 UE8M0 row quantization",
)


def _row_alloc_scales(T, H, layout, device, group_size=_ROW_GROUP_SIZE):
    sf_k = H // group_size
    if layout == "row":
        return torch.empty((T, sf_k), device=device, dtype=torch.float32)
    if layout == "col":
        return torch.empty((sf_k, T), device=device, dtype=torch.float32).permute(
            -1, -2
        )
    m = fp8_utils.get_tma_aligned_size(T, 4)
    return torch.empty_strided((T, sf_k), (1, m), device=device, dtype=torch.float32)


def _row_run(
    x, layout, use_row_kernel, use_ue8m0=False, group_size=_ROW_GROUP_SIZE, out_q=None
):
    if use_row_kernel:
        return fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=layout != "row",
            tma_aligned_scales=layout == "tma",
            out_q=out_q,
            use_ue8m0=use_ue8m0,
        )
    fp8_min, fp8_max = torch.finfo(_ROW_FP8_DTYPE).min, torch.finfo(_ROW_FP8_DTYPE).max
    T, H = x.shape
    x_q = torch.empty_like(x, dtype=_ROW_FP8_DTYPE)
    x_s = _row_alloc_scales(T, H, layout, x.device, group_size)
    torch.ops._C.per_token_group_fp8_quant(
        x,
        x_q,
        x_s,
        group_size,
        _ROW_EPS,
        fp8_min,
        fp8_max,
        use_ue8m0,
        layout != "row",
        layout == "tma",
    )
    return x_q, x_s


def _row_assert_bits_equal(actual, expected):
    q, s = actual
    q_ref, s_ref = expected
    assert q.shape == q_ref.shape and q.dtype == q_ref.dtype
    assert s.shape == s_ref.shape and s.stride() == s_ref.stride()
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(
        s.contiguous().view(torch.int32), s_ref.contiguous().view(torch.int32)
    )


def _row_add_edge_groups(x):
    fixtures = torch.zeros((16, _ROW_GROUP_SIZE), device=x.device, dtype=x.dtype)
    fixtures[1] = -0.0
    fixtures[2] = torch.finfo(x.dtype).tiny / 2
    fixtures[3] = torch.finfo(x.dtype).max
    fixtures[4] = -torch.finfo(x.dtype).max
    fixtures[5] = float("nan")
    fixtures[6] = float("inf")
    fixtures[7] = -float("inf")
    fixtures[8] = torch.linspace(
        -448, 448, _ROW_GROUP_SIZE, device=x.device, dtype=x.dtype
    )
    fixtures[9] = fixtures[8] * (2**-20)
    fixtures[10] = fixtures[8] * (2**20)
    fixtures[11:13] = fixtures[8]
    fixtures[11, 0] = float("nan")
    fixtures[12, 0] = float("inf")
    for i, value in enumerate((446.0, 448.0, 450.0), 13):
        fixtures[i] = fixtures[8]
        fixtures[i, -1] = value
    x[:16] = fixtures.repeat(1, x.shape[1] // _ROW_GROUP_SIZE)


@pytest.mark.parametrize("T", [2047, 2048, 3187])
@pytest.mark.parametrize(
    "H,layout", [(5120, "row"), (6144, "col"), (17408, "tma"), (32768, "col")]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_ue8m0", [False, True])
@torch.inference_mode()
@_requires_sm90_row_quant
def test_padded_widths_preserve_cuda_bits(T, H, layout, dtype, use_ue8m0):
    """Masking and scale rounding agree at the dispatch boundary and odd M."""
    torch.manual_seed(0)
    x = torch.randn(T, H, device="cuda", dtype=dtype)
    _row_add_edge_groups(x)
    _row_assert_bits_equal(
        _row_run(x, layout, True, use_ue8m0), _row_run(x, layout, False, use_ue8m0)
    )


@torch.inference_mode()
@_requires_sm90_row_quant
def test_offset_views_preserve_outputs_and_guard_bytes():
    """Contiguous views need not have aligned storage offsets or own storage."""
    T, H = 2048, 6144
    storage = torch.randn(T * H + 2, device="cuda", dtype=torch.bfloat16)
    x = storage[1:-1].view(T, H)
    out_storage = torch.full((T * H + 2,), 0x55, device="cuda", dtype=torch.uint8)
    out_q = out_storage[1:-1].view(_ROW_FP8_DTYPE).view(T, H)
    result = _row_run(x, "col", True, True, out_q=out_q)
    assert result[0].data_ptr() == out_q.data_ptr()
    _row_assert_bits_equal(result, _row_run(x, "col", False, True))
    assert out_storage[0].item() == out_storage[-1].item() == 0x55


@pytest.mark.parametrize(
    "H,group_size,dtype,use_ue8m0",
    [
        (4096, 64, torch.float16, False),
        (5120, 64, torch.bfloat16, True),
        (33024, 128, torch.bfloat16, True),
        (5120, 128, torch.float32, True),
    ],
)
@torch.inference_mode()
@_requires_sm90_row_quant
def test_other_group_sizes_and_fallbacks_match_cuda(H, group_size, dtype, use_ue8m0):
    x = torch.randn(2048, H, device="cuda", dtype=dtype)
    _row_assert_bits_equal(
        _row_run(x, "col", True, use_ue8m0, group_size),
        _row_run(x, "col", False, use_ue8m0, group_size),
    )


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(90)),
    reason="new padded path is SM90",
)
@torch.inference_mode()
@_requires_sm90_row_quant
def test_ue8m0_dispatch_boundary_launches_row_kernel():
    """Output equality alone must not let an accidental all-CUDA fallback pass."""
    for T in (2047, 2048):
        x = torch.randn(T, 5120, device="cuda", dtype=torch.bfloat16)
        _row_run(x, "col", True, True)
        torch.accelerator.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            _row_run(x, "col", True, True)
            torch.accelerator.synchronize()
        names = [event.name for event in prof.events()]
        assert any(
            "_per_token_group_quant_fp8_ue8m0_row" in name for name in names
        ) == (T == 2048)
        assert any("per_token_group_quant_8bit" in name for name in names) == (
            T == 2047
        )


@torch.inference_mode()
@pytest.mark.parametrize(
    "backend_name,tma_aligned_scales",
    [("inductor", False), ("inductor", True), ("eager", False)],
)
@_requires_sm90_row_quant
def test_fullgraph_dynamic_rows_cross_dispatch_boundary(
    backend_name, tma_aligned_scales
):
    from torch._dynamo.testing import EagerAndRecordGraphs

    backend = EagerAndRecordGraphs() if backend_name == "eager" else backend_name

    def quant(x):
        return fp8_utils.per_token_group_quant_fp8(
            x,
            _ROW_GROUP_SIZE,
            column_major_scales=True,
            tma_aligned_scales=tma_aligned_scales,
            use_ue8m0=True,
        )

    compiled = torch.compile(quant, backend=backend, fullgraph=True, dynamic=True)
    for T in (2047, 2048, 3187):
        x = torch.randn(T, 6144, device="cuda", dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        layout = "tma" if tma_aligned_scales else "col"
        _row_assert_bits_equal(compiled(x), _row_run(x, layout, False, True))
    # Check Dynamo graph reuse separately from Inductor's compilation retries.
    if isinstance(backend, EagerAndRecordGraphs):
        assert len(backend.graphs) == 1
        op = torch.ops.vllm.per_token_group_fp8_quant_ue8m0_row
        assert (
            sum(
                node.op == "call_function" and node.target in (op, op.default)
                for node in backend.graphs[0].graph.nodes
            )
            == 1
        )


@_requires_sm90_row_quant
def test_ue8m0_row_opcheck():
    """Fusion's native keyword arguments and output mutations remain valid."""
    torch.manual_seed(0)
    x = torch.randn(2048, 5120, device="cuda", dtype=torch.bfloat16)
    x_q = torch.empty_like(x, dtype=_ROW_FP8_DTYPE)
    x_s = _row_alloc_scales(x.shape[0], x.shape[1], "col", x.device)
    fp8_min, fp8_max = torch.finfo(_ROW_FP8_DTYPE).min, torch.finfo(_ROW_FP8_DTYPE).max
    opcheck(
        torch.ops.vllm.per_token_group_fp8_quant_ue8m0_row,
        (),
        dict(
            input=x,
            output_q=x_q,
            output_s=x_s,
            group_size=_ROW_GROUP_SIZE,
            eps=_ROW_EPS,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            scale_ue8m0=True,
            dummy_is_scale_transposed=True,
            dummy_is_tma_aligned=False,
        ),
    )
