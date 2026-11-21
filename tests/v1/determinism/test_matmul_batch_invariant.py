# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test batch-invariant matmul against torch.matmul for various shape combinations.

Tests correctness (matches torch.matmul) and batch invariance (result for one
item doesn't change based on other items in the batch).
"""

import pytest
import torch
from utils import skip_unsupported

import vllm.model_executor.determinism.batch_invariant as bi
from vllm.model_executor.determinism.batch_invariant import (
    enable_batch_invariant_mode,
    linear_batch_invariant,
    matmul_batch_invariant,
    matmul_persistent,
    mm_batch_invariant,
)
from vllm.model_executor.determinism.batch_invariant_configs import (
    _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS,
    _get_tuned_matmul_arch_family,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_batch_invariant_gemm,
    use_flashinfer_batch_invariant_gemm,
)

DEVICE_TYPE = current_platform.device_type

requires_flashinfer_gemm = pytest.mark.skipif(
    not has_flashinfer_batch_invariant_gemm(),
    reason="FlashInfer's batch-invariant GEMM needs FlashInfer >= 0.7.4 on "
    "SM100 / SM103 / SM107",
)

# (N, K) of Qwen3-8B's qkv / o / down projections, an R1-Distill-1.5B-class
# qkv, and an LM head
FLASHINFER_GEMM_SHAPES = [
    (6144, 4096),
    (4096, 4096),
    (4096, 12288),
    (2048, 1536),
    (151936, 1536),
]


@skip_unsupported
@pytest.mark.parametrize(
    "a_shape,b_shape",
    [
        # 2D x 2D
        ((32, 64), (64, 16)),
        # 2D x 3D
        ((64, 16), (4, 16, 32)),
        # 3D x 2D
        ((4, 32, 64), (64, 16)),
        # 4D x 2D
        ((1, 4, 32, 64), (64, 16)),
        # 3D x 3D
        ((4, 32, 64), (4, 64, 16)),
        # 3D x 4D
        ((2, 32, 64), (1, 2, 64, 16)),
        # 4D x 3D (Gemma4 pattern)
        ((1, 2, 32, 64), (2, 64, 16)),
        # 4D x 4D
        ((1, 2, 32, 64), (4, 2, 64, 16)),
        # 2D x 4D
        ((32, 64), (1, 2, 64, 16)),
        # 2D x 5D
        ((32, 64), (1, 2, 2, 64, 16)),
        # 5D x 2D
        ((1, 2, 2, 32, 64), (64, 16)),
        # 5D x 5D
        ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_correctness(a_shape, b_shape, dtype):
    """Compare matmul_batch_invariant against torch.matmul for various shapes."""
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a = torch.rand(a_shape, dtype=dtype, device=device)
    b = torch.rand(b_shape, dtype=dtype, device=device)

    # Standard implementation (CUDA ops)
    standard_output = torch.matmul(a, b)

    # Batch-invariant implementation (Triton)
    triton_output = matmul_batch_invariant(a, b)

    # Compare outputs
    # Use looser tolerance for bfloat16 due to its lower precision
    if dtype == torch.bfloat16:
        rtol, atol = 1e-1, 1e-1  # 10% relative tolerance for bfloat16
    else:
        rtol, atol = 1e-2, 1e-2  # 1% for float16/float32

    torch.testing.assert_close(
        triton_output,
        standard_output,
        rtol=rtol,
        atol=atol,
        msg=f"matmul mismatch for a ndim={a.ndim}, b ndim={b.ndim},",
    )


@skip_unsupported
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_batch_invariance(dtype):
    """Verify that the result for one item is bitwise identical regardless
    of what other items are in the batch.
    """
    device = torch.device(DEVICE_TYPE)

    torch.manual_seed(42)
    a_single = torch.rand((1, 64, 32), dtype=dtype, device=device)
    b = torch.rand((32, 128), dtype=dtype, device=device)

    standard_output = matmul_batch_invariant(a_single, b)

    a_batch = torch.rand((8, 64, 32), dtype=dtype, device=device)
    a_batch[3] = a_single[0]

    batch_output = matmul_batch_invariant(a_batch, b)
    batch_output_a = batch_output[3]

    assert torch.equal(standard_output[0], batch_output_a)


@skip_unsupported
@pytest.mark.parametrize("m", [8, 32, 256, 2048])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["contiguous", "transposed"])
def test_matmul_batch_invariance_across_tuned_m_buckets(m, transpose_b):
    # Tuned M buckets must preserve each row's K-reduction order.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family not in _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS:
        pytest.skip("No tuned persistent matmul config for this architecture")

    device = torch.device(DEVICE_TYPE)
    n = k = 2048
    torch.manual_seed(42)
    a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
    if transpose_b:
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device).t()
    else:
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

    single_output = matmul_batch_invariant(a[:1], b)
    batch_output = matmul_batch_invariant(a, b)

    assert torch.equal(single_output[0], batch_output[0])


@pytest.fixture
def batch_invariant_mode():
    # The overrides are process-global: remove them so later tests compare
    # against the stock aten kernels again.
    enable_batch_invariant_mode()
    yield
    assert bi._batch_invariant_LIB is not None
    bi._batch_invariant_LIB._destroy()
    bi._batch_invariant_MODE = False


@skip_unsupported
@pytest.mark.usefixtures("batch_invariant_mode")
@pytest.mark.parametrize("op", ["mm", "addmm", "bmm"])
def test_compiled_matmul_reaches_batch_invariant_out_overloads(op):
    """Inductor lowers mm/addmm/bmm to `extern_kernels.<op>(..., out=buf)`,
    i.e. the `.out` overload, which eager never dispatches."""
    if (
        op != "bmm"
        and current_platform.is_cuda()
        and not current_platform.is_device_capability_family(80)
    ):
        pytest.skip("mm/addmm are not overridden on SM90+")
    torch.manual_seed(0)
    dtype, k, n, needle = torch.bfloat16, 1024, 768, 37
    w = torch.randn(4, k, n, device=DEVICE_TYPE, dtype=dtype)
    bias = torch.randn(n, device=DEVICE_TYPE, dtype=dtype)
    fn = {
        "mm": lambda x: torch.mm(x[0], w[0]),
        # Bare addmm: a following pointwise op would make Inductor unfuse the bias.
        "addmm": lambda x: torch.addmm(bias, x[0], w[0]),
        "bmm": lambda x: torch.bmm(x, w),
    }[op]
    compiled = torch.compile(fn, fullgraph=True)
    x = torch.randn(4, 256, k, device=DEVICE_TYPE, dtype=dtype)
    row = None
    for m in (64, 256, 128):
        xs = x[:, :m].clone()
        # With dynamic=True, a specialization on M would silently recompile.
        torch._dynamo.mark_dynamic(xs, 1)
        out = compiled(xs)
        assert torch.equal(out, fn(xs)), f"compiled {op} bypassed the BI kernel (M={m})"
        row = out[..., needle, :] if row is None else row
        assert torch.equal(out[..., needle, :], row)


requires_matmul_overrides = pytest.mark.skipif(
    not (
        current_platform.is_rocm() or current_platform.is_device_capability_family(80)
    ),
    reason="Triton matmul overrides are only installed on ROCm and SM8x",
)


@skip_unsupported
@requires_matmul_overrides
@pytest.mark.usefixtures("batch_invariant_mode")
def test_mm_out_dtype_batch_invariance():
    device = torch.device(DEVICE_TYPE)
    torch.manual_seed(42)
    a = torch.randn((32, 4096), dtype=torch.bfloat16, device=device)
    w = torch.randn((8192, 4096), dtype=torch.bfloat16, device=device) * 0.02

    def head(x):
        return torch.mm(x, w.t(), out_dtype=torch.float32)

    batch_output = head(a)
    assert batch_output.dtype == torch.float32
    torch.testing.assert_close(head(a[:1])[0], batch_output[0], rtol=0, atol=0)
    assert torch.equal(
        batch_output, matmul_persistent(a, w.t(), out_dtype=torch.float32)
    )
    assert torch.equal(torch.compile(head)(a), batch_output)


@skip_unsupported
@requires_matmul_overrides
@pytest.mark.usefixtures("batch_invariant_mode")
def test_mm_out_dtype_keeps_aten_dtype_checks():
    a = torch.randn((4, 64), dtype=torch.bfloat16, device=DEVICE_TYPE)
    b = torch.randn((64, 32), dtype=torch.bfloat16, device=DEVICE_TYPE)
    with pytest.raises(RuntimeError, match="same as input dtype or fp32"):
        torch.mm(a, b, out_dtype=torch.float16)
    out = torch.empty((4, 32), dtype=torch.bfloat16, device=DEVICE_TYPE)
    with pytest.raises(RuntimeError, match="dtype of the provided out tensor"):
        torch.mm(a, b, out_dtype=torch.float32, out=out)


def _flashinfer_weights(n, k, with_bias, seed=0):
    torch.manual_seed(seed)
    w = torch.randn((n, k), dtype=torch.bfloat16, device=DEVICE_TYPE) / k**0.5
    b = torch.randn(n, dtype=torch.bfloat16, device=DEVICE_TYPE) if with_bias else None
    return w, b


@skip_unsupported
@requires_flashinfer_gemm
@pytest.mark.parametrize("n,k", FLASHINFER_GEMM_SHAPES)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_flashinfer_linear_matches_triton_bits(n, k, with_bias):
    """FlashInfer's batch-invariant GEMM computes the Triton persistent matmul's
    bits, so VLLM_BATCH_INVARIANT_GEMM_BACKEND changes speed, not outputs."""
    w, b = _flashinfer_weights(n, k, with_bias)
    for m in (1, 257):
        x = torch.randn((m, k), dtype=torch.bfloat16, device=DEVICE_TYPE)
        flashinfer_out = torch.ops.vllm.flashinfer_mm_bf16_batch_invariant(x, w, b)
        triton_out = mm_batch_invariant(x, w.t(), bias=b)
        assert torch.equal(flashinfer_out, triton_out), f"M={m}"


@skip_unsupported
@requires_flashinfer_gemm
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_flashinfer_linear_batch_invariance(with_bias):
    """linear_batch_invariant takes the FlashInfer path for every row count, and
    a row's bits do not depend on the rows around it."""
    n, k = 4096, 4096
    w, b = _flashinfer_weights(n, k, with_bias)
    x = torch.randn((2, 257, k), dtype=torch.bfloat16, device=DEVICE_TYPE)
    assert bi.flashinfer_gemm_enabled() and bi._fits_flashinfer_gemm(x, w, b)
    full = linear_batch_invariant(x, w, b)
    assert full.shape == (2, 257, n)
    flat = full.reshape(-1, n)
    for m in (1, 7, 64, 300):
        rows = linear_batch_invariant(x.reshape(-1, k)[:m], w, b)
        assert torch.equal(rows, flat[:m]), f"M={m}"


@skip_unsupported
@requires_flashinfer_gemm
def test_flashinfer_linear_under_torch_compile():
    """The custom op traces with a dynamic row count and keeps the eager bits."""
    n, k = 6144, 4096
    w, b = _flashinfer_weights(n, k, with_bias=True)

    def fn(x):
        return linear_batch_invariant(x, w, b)

    compiled = torch.compile(fn, fullgraph=True)
    x = torch.randn((256, k), dtype=torch.bfloat16, device=DEVICE_TYPE)
    for m in (64, 256, 128):
        xs = x[:m].clone()
        torch._dynamo.mark_dynamic(xs, 0)
        assert torch.equal(compiled(xs), fn(xs)), f"M={m}"


def test_batch_invariant_gemm_backend_env(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_GEMM_BACKEND", "triton")
    assert not use_flashinfer_batch_invariant_gemm()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_GEMM_BACKEND", "flashinfer")
    if has_flashinfer_batch_invariant_gemm():
        assert use_flashinfer_batch_invariant_gemm()
    else:
        with pytest.raises(RuntimeError, match="needs FlashInfer"):
            use_flashinfer_batch_invariant_gemm()
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_GEMM_BACKEND", "cublas")
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT_GEMM_BACKEND"):
        use_flashinfer_batch_invariant_gemm()
