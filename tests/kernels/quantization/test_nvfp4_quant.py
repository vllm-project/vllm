# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import break_fp4_bytes
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import has_flashinfer_cutedsl_nvfp4_quant
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(128, 64), (128, 128), (256, 64), (256, 128)]
PAD_SHAPES = [
    (90, 64),
    (150, 64),
    (128, 48),
    (128, 80),
    (150, 80),
    (90, 48),
    (90, 128),
    (150, 128),
    (150, 48),
    (90, 80),
    (128, 512),
    (128, 1024),
    (128, 2048),
    (64, 7168),
    (64, 7152),
    (32, 14336),
]
PADDED_OUTPUT_SHAPES = [(128, 48), (128, 80), (150, 48), (150, 80), (64, 7152)]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = scalar_types.float4_e2m1f.max()
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# E2M1 to float
# 0111 -> 6
# 0110 -> 4
# 0101 -> 3
# 0100 -> 2
# 0011 -> 1.5
# 0010 -> 1
# 0001 -> 0.5
# 0000 -> 0
E2M1_TO_FLOAT32 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
BLOCK_SIZE = 16


def cast_from_fp4(x, m, n):
    # The fp4 values are packed in uint8 as [v_1st | v_2nd]
    v_2nd = x & 0xF
    v_1st = (x >> 4) & 0xF
    c = torch.stack((v_2nd, v_1st), dim=-1)
    out = torch.tensor([E2M1_TO_FLOAT32[x] for x in c.flatten()])
    out = out.reshape(m, n).to(torch.float32)
    return out


def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def get_reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.tensor(0.0, dtype=x.dtype), 1.0 / x)
    elif isinstance(x, (float, int)):
        return 0.0 if x == 0 else 1.0 / x
    else:
        raise TypeError("Input must be a float, int, or a torch.Tensor.")


def ref_nvfp4_quant(x, global_scale):
    assert global_scale.dtype == torch.float32
    assert x.ndim == 2
    m, n = x.shape
    x = torch.reshape(x, (m, n // BLOCK_SIZE, BLOCK_SIZE))
    vec_max = torch.max(torch.abs(x), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = global_scale * (vec_max * get_reciprocal(FLOAT4_E2M1_MAX))
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = get_reciprocal(scale * get_reciprocal(global_scale))

    scaled_x = x.to(torch.float32) * output_scale
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0).reshape(m, n)
    return cast_to_fp4(clipped_x), scale.squeeze(-1)


def recover_swizzled_scales(scale, m, n):
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // BLOCK_SIZE
    rounded_n = round_up(scale_n, 4)
    # Recover the swizzled scaling factor to linear layout
    tmp = torch.reshape(scale, (1, rounded_m // 128, rounded_n // 4, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    result = torch.reshape(tmp, (rounded_m, rounded_n)).to(torch.float32)
    return result[:m, :scale_n]


def round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


# Signed FP4 (E2M1) levels in ascending order; an index gap of 1 between two
# decoded values is a single representable-level (rounding-boundary) step.
_FP4_LEVELS_ASC = sorted(set(E2M1_TO_FLOAT32))


def assert_fp4_within_one_level(packed_a: torch.Tensor, packed_b: torch.Tensor) -> None:
    """Assert two packed-FP4 tensors decode to values differing by at most one FP4
    level per element. Unlike a packed-byte mismatch fraction this is size- and
    nibble-independent and bounds every element to a single level, so a larger
    systematic difference is still caught while an approximate-reciprocal boundary
    flip is tolerated."""
    a = break_fp4_bytes(packed_a, torch.float32).flatten()
    b = break_fp4_bytes(packed_b, torch.float32).flatten()
    levels = torch.tensor(_FP4_LEVELS_ASC, dtype=torch.float32, device=a.device)
    gap = (torch.searchsorted(levels, a) - torch.searchsorted(levels, b)).abs()
    max_gap = int(gap.max())
    assert max_gap <= 1, f"fp4 codes differ by up to {max_gap} levels"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_quantize_to_fp4(
    dtype: torch.dtype,
    shape: tuple[int, int],
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)

    m, n = shape

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = ops.scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)

    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.parametrize(
    "shape",
    [(32, 4096), (128, 4096), (1, 64), (127, 1024), (256, 16384)],
)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@torch.inference_mode()
def test_python_util_matches_cpp_allocation(
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
) -> None:
    """
    Verify that the Python utility (create_fp4_output_tensors) allocates
    tensors with the same shapes and dtypes as the C++ functional variant
    (scaled_fp4_quant_func).
    """
    from vllm._custom_ops import create_fp4_output_tensors

    torch.set_default_device("cuda:0")
    m, n = shape
    input_tensor = torch.randn((m, n), dtype=torch.bfloat16)
    input_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda:0")

    # C++ functional variant allocates internally
    cpp_out, cpp_scale = torch.ops._C.scaled_fp4_quant(
        input_tensor, input_scale, is_sf_swizzled_layout
    )

    # Python utility
    py_out, py_scale = create_fp4_output_tensors(
        m, n, torch.device("cuda:0"), is_sf_swizzled_layout
    )

    assert py_out.shape == cpp_out.shape, (
        f"Output shape mismatch: Python {py_out.shape} vs C++ {cpp_out.shape}"
    )
    assert py_out.dtype == cpp_out.dtype, (
        f"Output dtype mismatch: Python {py_out.dtype} vs C++ {cpp_out.dtype}"
    )
    assert py_scale.shape == cpp_scale.shape, (
        f"Scale shape mismatch: Python {py_scale.shape} vs C++ {cpp_scale.shape}"
    )
    assert py_scale.dtype == cpp_scale.dtype, (
        f"Scale dtype mismatch: Python {py_scale.dtype} vs C++ {cpp_scale.dtype}"
    )


@pytest.mark.parametrize("shape", PADDED_OUTPUT_SHAPES)
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
@torch.inference_mode()
def test_quantize_to_fp4_with_padded_output(
    shape: tuple[int, int],
    is_sf_swizzled_layout: bool,
) -> None:
    from vllm._custom_ops import create_fp4_output_tensors

    dtype = torch.float16
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = shape
    padded_n = round_up(n, 32)
    assert padded_n > n

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = ops.scaled_fp4_quant(
        x,
        global_scale,
        is_sf_swizzled_layout=is_sf_swizzled_layout,
        padded_n=padded_n,
    )
    py_out, py_scale = create_fp4_output_tensors(
        m,
        n,
        torch.device("cuda:0"),
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )

    assert out.shape == (m, padded_n // 2)
    assert out.shape == py_out.shape
    assert out_scale.shape == py_scale.view(torch.float8_e4m3fn).shape

    out_ans = cast_from_fp4(out[:, : n // 2], m, n)
    torch.testing.assert_close(out_ans, out_ref)
    assert torch.count_nonzero(out[:, n // 2 :]) == 0

    if is_sf_swizzled_layout:
        scale_ans = recover_swizzled_scales(out_scale, m, padded_n)
        torch.testing.assert_close(scale_ans[:, : n // BLOCK_SIZE], scale_ref)
        assert torch.count_nonzero(scale_ans[:, n // BLOCK_SIZE :]) == 0
    else:
        scale_ans = out_scale.to(torch.float32)
        torch.testing.assert_close(scale_ans[:, : n // BLOCK_SIZE], scale_ref)
        assert torch.count_nonzero(scale_ans[:, n // BLOCK_SIZE :]) == 0


@pytest.mark.parametrize("pad_shape", PAD_SHAPES)
@torch.inference_mode()
def test_quantize_to_fp4_padded(pad_shape: tuple[int, int]) -> None:
    dtype = torch.float16
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = pad_shape

    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = ops.scaled_fp4_quant(x, global_scale)
    scale_ans = recover_swizzled_scales(out_scale, m, n)
    out_ans = cast_from_fp4(out, m, n)
    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.parametrize("pad_shape", PAD_SHAPES)
@torch.inference_mode()
def test_quantize_to_fp4_padded_no_sf_swizzled(pad_shape: tuple[int, int]) -> None:
    dtype = torch.float16
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = pad_shape

    x = torch.randn((m, n), dtype=dtype)

    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax
    out_ref, scale_ref = ref_nvfp4_quant(x, global_scale)

    out, out_scale = ops.scaled_fp4_quant(x, global_scale, is_sf_swizzled_layout=False)
    scale_ans = out_scale.to(torch.float32)
    out_ans = cast_from_fp4(out, m, n)
    torch.testing.assert_close(out_ans, out_ref)
    torch.testing.assert_close(scale_ans, scale_ref)


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer NVFP4 quantization is not available.",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES + PAD_SHAPES)
@torch.inference_mode()
def test_flashinfer_nvfp4_quant_128x4_matches_vllm(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """FlashInfer CuTe-DSL 128x4 quant must match the vLLM C++ kernel it replaces
    (including M not a multiple of 128). CuTe-DSL uses an approximate reciprocal,
    so assert equivalence (at most one FP4 level per element + aggregate dequant
    error), not bit-exactness."""
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
    )

    # vLLM C++ 128x4 (the path FlashInfer replaces).
    ref_out, ref_scale = ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=True
    )
    # FlashInfer CuTe-DSL 128x4.
    fi_out, fi_scale = ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=True, quant_backend="flashinfer_cutedsl"
    )

    assert fi_out.shape == ref_out.shape
    assert fi_scale.shape == ref_scale.shape

    # An approximate reciprocal may flip a value to an adjacent FP4 level, but no
    # element may differ by more than one level.
    assert_fp4_within_one_level(fi_out, ref_out)

    # Aggregate dequant error catches systematic (scale/layout) errors while
    # tolerating those isolated single-level boundary flips.
    n_blocks = n // BLOCK_SIZE
    fi_s = recover_swizzled_scales(fi_scale, m, n)
    ref_s = recover_swizzled_scales(ref_scale, m, n)
    fi_fp4 = break_fp4_bytes(fi_out, torch.float32).reshape(m, n_blocks, BLOCK_SIZE)
    ref_fp4 = break_fp4_bytes(ref_out, torch.float32).reshape(m, n_blocks, BLOCK_SIZE)
    fi_deq = fi_fp4 * fi_s.unsqueeze(-1) / global_scale
    ref_deq = ref_fp4 * ref_s.unsqueeze(-1) / global_scale
    agg_rel_err = (fi_deq - ref_deq).abs().sum() / ref_deq.abs().sum().clamp_min(1e-6)
    assert agg_rel_err < 1e-2, f"aggregate rel err {float(agg_rel_err):.4f}"


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer NVFP4 quantization is not available.",
)
@pytest.mark.parametrize("shape", PADDED_OUTPUT_SHAPES)
@torch.inference_mode()
def test_flashinfer_nvfp4_quant_128x4_padded_output(
    shape: tuple[int, int],
) -> None:
    """The FlashInfer route emulates padded_n by zero-padding the input. The
    data region must equal the unpadded FlashInfer output and the padded
    columns (fp4 and scale) must be exactly zero."""
    dtype = torch.float16
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = shape
    padded_n = round_up(n, 32)
    assert padded_n > n

    x = torch.randn((m, n), dtype=dtype)
    global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
    )

    padded_out, padded_scale = ops.scaled_fp4_quant(
        x,
        global_scale,
        is_sf_swizzled_layout=True,
        padded_n=padded_n,
        quant_backend="flashinfer_cutedsl",
    )
    unpadded_out, unpadded_scale = ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=True, quant_backend="flashinfer_cutedsl"
    )

    assert padded_out.shape == (m, padded_n // 2)

    # Padding whole zero blocks leaves the real blocks unchanged.
    torch.testing.assert_close(padded_out[:, : n // 2], unpadded_out)
    assert torch.count_nonzero(padded_out[:, n // 2 :]) == 0

    padded_s = recover_swizzled_scales(padded_scale, m, padded_n)
    unpadded_s = recover_swizzled_scales(unpadded_scale, m, n)
    torch.testing.assert_close(padded_s[:, : n // BLOCK_SIZE], unpadded_s)
    assert torch.count_nonzero(padded_s[:, n // BLOCK_SIZE :]) == 0


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer NVFP4 quantization is not available.",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(90, 64), (150, 64), (90, 128), (150, 128)])
@torch.inference_mode()
def test_flashinfer_nvfp4_quant_128x4_zeros_scale_padding(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """The swizzled scale buffer is padded to round_up(m, 128) rows. This CuTe-DSL
    path bypasses create_fp4_scale_tensor's zero-init, so it relies on the
    FlashInfer kernel zeroing those padded rows itself. Guards against a FlashInfer
    change reintroducing the uninitialized-scale corruption fixed in PR #45739."""
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = shape
    padded_m = round_up(m, 128)
    assert padded_m > m and (n // BLOCK_SIZE) % 4 == 0  # row padding only

    x = torch.randn((m, n), dtype=dtype)
    global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
    )

    _, scale = ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=True, quant_backend="flashinfer_cutedsl"
    )

    # Un-swizzle the full padded buffer; the padded rows [m:padded_m) must be zero.
    full = recover_swizzled_scales(scale, padded_m, n)
    assert torch.count_nonzero(full[m:]) == 0


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer NVFP4 quantization is not available.",
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES + PAD_SHAPES)
@torch.inference_mode()
def test_flashinfer_nvfp4_quant_linear_matches_vllm(
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> None:
    """FlashInfer CuTe-DSL linear quant must match the vLLM C++ kernel it
    mirrors. Same approximate-reciprocal tolerance as the 128x4 test; the linear
    scale is a plain [m, n // 16] tensor, compared directly (no unswizzle)."""
    set_random_seed(42)
    torch.set_default_device("cuda:0")

    m, n = shape
    x = torch.randn((m, n), dtype=dtype)
    global_scale = (
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
    )

    # vLLM C++ linear (the path FlashInfer mirrors).
    ref_out, ref_scale = ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=False
    )
    # FlashInfer CuTe-DSL linear.
    fi_out, fi_scale = ops.scaled_fp4_quant(
        x, global_scale, is_sf_swizzled_layout=False, quant_backend="flashinfer_cutedsl"
    )

    assert fi_out.shape == ref_out.shape
    assert fi_scale.shape == ref_scale.shape

    # An approximate reciprocal may flip a value to an adjacent FP4 level, but no
    # element may differ by more than one level.
    assert_fp4_within_one_level(fi_out, ref_out)

    # Aggregate dequant error catches systematic (scale/layout) errors while
    # tolerating those isolated single-level boundary flips.
    n_blocks = n // BLOCK_SIZE
    fi_s = fi_scale.to(torch.float32)
    ref_s = ref_scale.to(torch.float32)
    fi_fp4 = break_fp4_bytes(fi_out, torch.float32).reshape(m, n_blocks, BLOCK_SIZE)
    ref_fp4 = break_fp4_bytes(ref_out, torch.float32).reshape(m, n_blocks, BLOCK_SIZE)
    fi_deq = fi_fp4 * fi_s.unsqueeze(-1) / global_scale
    ref_deq = ref_fp4 * ref_s.unsqueeze(-1) / global_scale
    agg_rel_err = (fi_deq - ref_deq).abs().sum() / ref_deq.abs().sum().clamp_min(1e-6)
    assert agg_rel_err < 1e-2, f"aggregate rel err {float(agg_rel_err):.4f}"


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer NVFP4 quantization is not available.",
)
@torch.inference_mode()
def test_nvfp4_quant_trtllm_8x4_layout_selection() -> None:
    """For the TRTLLM backend the SF layout is chosen by M: 8x4 at m <= 32 and
    128x4 above (distinguished by the scale row count). This holds with
    quant_backend="flashinfer_cutedsl", which routes to the cute-dsl 8x4 / 128x4
    kernels."""
    set_random_seed(42)
    torch.set_default_device("cuda:0")
    n = 64

    # m <= 32 with the TRTLLM backend: 8x4 layout (cute-dsl 8x4 here).
    m_small = 16
    x = torch.randn((m_small, n), dtype=torch.bfloat16)
    gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
    _, scale = ops.scaled_fp4_quant(
        x, gs, gemm_backend="flashinfer-trtllm", quant_backend="flashinfer_cutedsl"
    )
    assert scale.shape[0] == round_up(m_small, 8)

    # m > 32: the 8x4 path no longer applies, so it routes to the 128x4 CuTe-DSL
    # kernel.
    m_large = 64
    x = torch.randn((m_large, n), dtype=torch.bfloat16)
    gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
    _, scale = ops.scaled_fp4_quant(
        x, gs, gemm_backend="flashinfer-trtllm", quant_backend="flashinfer_cutedsl"
    )
    assert scale.shape[0] == round_up(m_large, 128)


@pytest.mark.skipif(
    not has_flashinfer_cutedsl_nvfp4_quant(),
    reason="FlashInfer NVFP4 quantization is not available.",
)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flashinfer_cutedsl_nvfp4_quant_8x4_matches_cuda(dtype: torch.dtype) -> None:
    """FlashInfer CuTe-DSL 8x4 quant (TRTLLM small-M) must match the CUDA 8x4
    kernel. There is no vLLM C++ 8x4 kernel, so both sides are FlashInfer; assert
    at most one FP4 level (and one e4m3 scale step) per element, not
    bit-exactness."""
    set_random_seed(42)
    torch.set_default_device("cuda:0")
    n = 128

    for m in (1, 16, 32):
        x = torch.randn((m, n), dtype=dtype)
        gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / torch.abs(x).max().to(torch.float32)
        cuda_out, cuda_scale = ops.scaled_fp4_quant(
            x, gs, gemm_backend="flashinfer-trtllm", quant_backend="auto"
        )
        fi_out, fi_scale = ops.scaled_fp4_quant(
            x, gs, gemm_backend="flashinfer-trtllm", quant_backend="flashinfer_cutedsl"
        )

        # Both use the 8x4 layout (scale rows rounded up to 8).
        assert fi_out.shape == cuda_out.shape
        assert fi_scale.shape == cuda_scale.shape
        assert fi_scale.shape[0] == round_up(m, 8)

        # At most one FP4 level and one e4m3 scale step may differ per element.
        assert_fp4_within_one_level(fi_out, cuda_out)
        fi_s = fi_scale.view(torch.float8_e4m3fn).to(torch.float32)
        cuda_s = cuda_scale.view(torch.float8_e4m3fn).to(torch.float32)
        torch.testing.assert_close(fi_s, cuda_s, rtol=0.13, atol=2**-9)
