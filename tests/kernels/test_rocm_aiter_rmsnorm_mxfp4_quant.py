# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the AITER MXFP4 ops: rocm_aiter_rmsnorm_mxfp4_quant and the fake of
gemm_with_dynamic_quant.

``rocm_aiter_rmsnorm_mxfp4_quant`` wraps the AITER HIP kernel ``add_rmsnorm_quant``: a
Gemma RMSNorm of ``x + residual`` quantized to MXFP4 (codes plus shuffled e8m0 scales),
out of place, so a fusion pass can put it in a compiled graph. Three groups of tests:

* the argument contract and the output allocation are plain functions of
  ``vllm._aiter_ops`` and run anywhere, on meta and fake tensors,
* the registered ops and their fakes need ROCm with AITER and run on fake tensors,
* the kernel tests compare the op with a direct call of the AITER kernel and need a
  gfx950 GPU.
"""

import inspect

import pytest
import regex as re
import torch

from vllm._aiter_ops import (
    ROCM_AITER_MXFP4_NORM_QUANT_MAX_K,
    _rocm_aiter_rmsnorm_mxfp4_quant_check_and_alloc,
    is_aiter_found_and_supported,
    rocm_aiter_rmsnorm_mxfp4_quant_args_error,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

GROUP = 32  # MXFP4 block size: one e8m0 scale per 32 values
SCALE_ROW_PAD = 256  # the shuffled scales fill whole 256-row tiles ...
SCALE_COL_PAD = 8  # ... of 8 group columns
HIDDEN = 5120  # a hidden size the norm sites of a real model have
EPS = 1e-6
BF16, FP16, FP32 = torch.bfloat16, torch.float16, torch.float32

requires_aiter = pytest.mark.skipif(
    not is_aiter_found_and_supported(), reason="ROCm with AITER only"
)


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_gfx950_gpu = pytest.mark.skipif(
    not (is_aiter_found_and_supported() and _on_gfx950() and torch.cuda.is_available()),
    reason="needs a gfx950 GPU with AITER",
)


def expected_shapes(m: int, k: int):
    """The documented output shapes: codes, shuffled scales, residual output."""
    return (
        (m, k // 2),
        (round_up(m, SCALE_ROW_PAD), round_up(k // GROUP, SCALE_COL_PAD)),
        (m, k),
    )


# --- the argument contract (runs anywhere, on meta tensors) --------------------------


def meta(*shape, dtype=BF16):
    """A meta tensor: shape, dtype and strides, no memory."""
    return torch.empty(shape, dtype=dtype, device="meta")


def call_args(m=1, k=HIDDEN, dtype=BF16, *, x=None, residual=None, weight=None):
    """(x, residual, weight) of an (m, k) call; a keyword replaces one of the three."""
    return (
        meta(m, k, dtype=dtype) if x is None else x,
        meta(m, k, dtype=dtype) if residual is None else residual,
        meta(k, dtype=dtype) if weight is None else weight,
    )


# Each case builds the arguments and gives a regex that the message must match; the
# message names the offending value.
BAD_INPUTS = [
    pytest.param(lambda: call_args(1, 9216), r"K=9216", id="wide_rows"),
    pytest.param(lambda: call_args(1, 1040), r"K=1040", id="k_not_a_multiple_of_32"),
    pytest.param(lambda: call_args(2, 0), r"K=0", id="zero_width"),
    pytest.param(lambda: call_args(dtype=FP16), r"got torch\.float16", id="fp16_input"),
    pytest.param(
        lambda: call_args(weight=meta(HIDDEN, dtype=FP32)),
        r"weight dtype torch\.float32",
        id="weight_dtype_differs",
    ),
    pytest.param(
        lambda: call_args(x=meta(2, 3, HIDDEN), residual=meta(2, 3, HIDDEN)),
        r"2D.*\(2, 3, 5120\)",
        id="three_dimensional",
    ),
    pytest.param(
        lambda: call_args(residual=meta(1, HIDDEN // 2)),
        r"residual.*got \(1, 2560\)",
        id="residual_width_differs",
    ),
    pytest.param(
        lambda: call_args(weight=meta(HIDDEN // 2)),
        r"weight of shape \(5120,\), got \(2560,\)",
        id="weight_half_as_long",
    ),
    pytest.param(
        lambda: call_args(33, x=meta(HIDDEN, 33).t()),
        r"unit stride.*x=\(1, 33\)",
        id="x_transposed",
    ),
    pytest.param(
        lambda: call_args(weight=meta(2 * HIDDEN)[::2]),
        r"unit stride.*weight=\(2,\)",
        id="weight_stride_two",
    ),
]


def test_the_max_hidden_size_is_8192():
    # The widest row of AITER's module_rmsnorm_quant. The constant is public so a
    # fusion pass skips a wider site instead of failing when it traces the op.
    assert ROCM_AITER_MXFP4_NORM_QUANT_MAX_K == 8192


@pytest.mark.parametrize("k", [32, HIDDEN, ROCM_AITER_MXFP4_NORM_QUANT_MAX_K])
def test_args_error_is_none_for_arguments_the_op_takes(k):
    # Input: well-formed arguments, K from one group to the limit, and a row-strided
    # x (rows 2 elements apart, inner stride 1). Expected: None for both. Why: None
    # means "the op takes this", and a fusion pass fuses exactly then.
    x, residual, weight = call_args(3, k)
    assert rocm_aiter_rmsnorm_mxfp4_quant_args_error(x, residual, weight) is None
    strided = meta(3, 2 * k)[:, :k]
    assert rocm_aiter_rmsnorm_mxfp4_quant_args_error(strided, residual, weight) is None


@pytest.mark.parametrize(("build_args", "message"), BAD_INPUTS)
def test_args_error_names_the_offending_value(build_args, message):
    # Input: one bad argument at a time. Expected: a message that matches the pattern
    # and names the value, and the allocation function raises it as a ValueError.
    # Why: the kernel reads and writes through raw pointers and trusts its arguments; a
    # bad one is a fault or silent garbage, and the fake raising the same text makes a
    # compile fail at trace time.
    args = build_args()
    error = rocm_aiter_rmsnorm_mxfp4_quant_args_error(*args)
    assert error is not None and re.search(message, error), error
    with pytest.raises(ValueError) as raised:
        _rocm_aiter_rmsnorm_mxfp4_quant_check_and_alloc(*args)
    assert str(raised.value) == error


def test_args_error_leaves_the_row_counts_to_the_kernel_call():
    # Input: x with 4 rows and a residual with 3. Expected: None. Why: in a trace x
    # and residual carry two different symbols for M, and comparing them would add a
    # guard on M that recompiles the graph per batch size, so only the op's impl
    # (concrete rows) compares them (see the kernel tests below).
    x, residual, weight = meta(4, HIDDEN), meta(3, HIDDEN), meta(HIDDEN)
    assert rocm_aiter_rmsnorm_mxfp4_quant_args_error(x, residual, weight) is None


@pytest.mark.parametrize("k", [HIDDEN, 1056])
@pytest.mark.parametrize("m", [0, 1, 257])
def test_alloc_returns_the_documented_shapes_and_dtypes(m, k):
    # Input: (m, k) bf16 arguments. Expected: codes (m, k // 2) float4_e2m1fn_x2,
    # scales (pad256(m), pad8(k // 32)) float8_e8m0fnu, residual output (m, k) bf16.
    # Why: the compiler plans memory from these shapes and the asm GEMM reads the
    # scales in this padded, shuffled layout.
    q, s, residual_out = _rocm_aiter_rmsnorm_mxfp4_quant_check_and_alloc(
        *call_args(m, k)
    )
    want_q, want_s, want_residual = expected_shapes(m, k)
    assert (tuple(q.shape), q.dtype) == (want_q, torch.float4_e2m1fn_x2)
    assert (tuple(s.shape), s.dtype) == (want_s, torch.float8_e8m0fnu)
    assert (tuple(residual_out.shape), residual_out.dtype) == (want_residual, BF16)


def test_alloc_with_symbolic_rows_is_exact_and_adds_no_guard():
    # Input: fake (M, k) tensors whose row counts are symbols: x and residual with two
    # different ones, as Dynamo gives two inputs marked dynamic. Expected: the outputs'
    # row counts are expressions of x's symbol, no guard was added to the shape
    # environment, and for each M substituted (crossing the 256-row tile edge) the
    # codes and residual have M rows and the scales pad256(M). Why: a guard on M
    # recompiles the graph for each batch size; a wrong padding formula would size the
    # scale buffer wrongly for some M.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        ShapeEnv,
        StatelessSymbolicContext,
    )

    shape_env = ShapeEnv()
    mode = FakeTensorMode(shape_env=shape_env)

    def symbolic_rows():
        return mode.from_tensor(
            meta(300, HIDDEN),
            symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
            ),
        )

    x, residual = symbolic_rows(), symbolic_rows()
    assert x.shape[0].node.expr != residual.shape[0].node.expr
    with mode:
        q, s, residual_out = _rocm_aiter_rmsnorm_mxfp4_quant_check_and_alloc(
            x, residual, meta(HIDDEN)
        )
    assert shape_env.guards == []
    symbol = x.shape[0].node.expr
    for m in [2, 255, 256, 257, 4096]:
        assert q.shape[0].node.expr.subs(symbol, m) == m
        assert residual_out.shape[0].node.expr.subs(symbol, m) == m
        assert s.shape[0].node.expr.subs(symbol, m) == round_up(m, SCALE_ROW_PAD)


# --- the registered ops on fake tensors (ROCm with AITER) ----------------------------


def fused_op(x, residual, weight, eps=EPS):
    """The op under test."""
    return torch.ops.vllm.rocm_aiter_rmsnorm_mxfp4_quant(x, residual, weight, eps)


@requires_aiter
@pytest.mark.parametrize("m", [0, 257])
def test_registered_fake_returns_the_documented_shapes(m):
    # Input: fake bf16 tensors through the registered op. Expected: the shapes and
    # dtypes the allocation function gives.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        device = current_platform.device_type
        x = torch.empty(m, HIDDEN, dtype=BF16, device=device)
        out = fused_op(
            x, torch.empty_like(x), torch.empty(HIDDEN, dtype=BF16, device=device)
        )
    want = expected_shapes(m, HIDDEN)
    assert [tuple(t.shape) for t in out] == list(want)
    assert [t.dtype for t in out] == [
        torch.float4_e2m1fn_x2,
        torch.float8_e8m0fnu,
        BF16,
    ]


@requires_aiter
@pytest.mark.parametrize(("build_args", "message"), BAD_INPUTS)
def test_registered_fake_raises_the_text_args_error_returns(build_args, message):
    # Input: the bad arguments through the registered op's fake. Expected: ValueError
    # with exactly the text of args_error. Why: the pass registers its patterns by
    # tracing the op, so what it cannot take must fail with a reason at trace time.
    args = build_args()
    error = rocm_aiter_rmsnorm_mxfp4_quant_args_error(*args)
    with pytest.raises(ValueError) as raised:
        fused_op(*args)
    assert str(raised.value) == error


@requires_aiter
def test_compile_fails_at_trace_time_with_the_ops_message():
    # Input: a (1, 9216) input through torch.compile(fullgraph=True, backend="eager").
    # Expected: Dynamo raises TorchRuntimeError, and the ValueError of the fake, naming
    # K=9216, is in its chain of causes. Why: the failure should show when the model
    # compiles, with the reason, not as a kernel fault on the first request.
    torch._dynamo.reset()
    f = torch.compile(fused_op, fullgraph=True, backend="eager")
    x = torch.zeros(1, 9216, dtype=BF16)
    with pytest.raises(torch._dynamo.exc.TorchRuntimeError) as err:
        f(x, torch.zeros_like(x), torch.zeros(9216, dtype=BF16))
    torch._dynamo.reset()
    chain, cause = [], err.value
    while cause is not None:
        chain.append(cause)
        cause = cause.__cause__ or cause.__context__
    assert any(isinstance(e, ValueError) and "K=9216" in str(e) for e in chain), chain


@requires_aiter
@pytest.mark.parametrize("out_dtype", [BF16, FP16])
def test_gemm_fake_takes_the_real_ops_arguments_in_the_real_order(out_dtype):
    # gemm_with_dynamic_quant gets the codes and scales of a fused quant through
    # x_scales and activation, after out_dtype. Input: the signatures of the op and of
    # its fake, then a call on fake tensors the way the norm fusion makes it. Expected:
    # equal parameters, and an (M, N) result of out_dtype. Why: the dispatcher passes
    # the arguments to the fake by position; with x_scales ahead of out_dtype in the
    # fake, the dtype was a tensor and tracing failed.
    from torch._subclasses.fake_tensor import FakeTensorMode

    import vllm.model_executor.kernels.linear.mxfp4.aiter as mxfp4_aiter

    def parameters(fn):
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(fn).parameters.values()
        ]

    assert parameters(mxfp4_aiter.gemm_with_dynamic_quant_fake) == parameters(
        mxfp4_aiter.gemm_with_dynamic_quant
    )

    device = current_platform.device_type
    m, n, k = 6, 32, 128
    with FakeTensorMode():
        x_q = torch.empty(m, k // 2, dtype=torch.float4_e2m1fn_x2, device=device)
        x_scales = torch.empty(256, 8, dtype=torch.float8_e8m0fnu, device=device)
        weight = torch.empty(n, k // 2, dtype=torch.uint8, device=device)
        weight_scale = torch.empty(n, k // 32, dtype=torch.uint8, device=device)
        out = torch.ops.vllm.gemm_with_dynamic_quant(
            x_q, weight, weight_scale, True, out_dtype, x_scales
        )
    assert (tuple(out.shape), out.dtype) == ((m, n), out_dtype)


# --- the kernel (gfx950 GPU) ---------------------------------------------------------


def norm_inputs(m: int, seed: int, k: int = HIDDEN):
    """Random bf16 x, residual and Gemma norm weight (the norm adds 1 itself)."""
    device = current_platform.device_type
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(m, k, dtype=BF16, device=device, generator=g)
    residual = torch.randn(m, k, dtype=BF16, device=device, generator=g)
    weight = torch.randn(k, dtype=BF16, device=device, generator=g) * 0.1
    return x, residual, weight


def direct_kernel(x, residual, weight, eps=EPS):
    """AITER's add_rmsnorm_quant called directly: the reference of the op."""
    from aiter import add_rmsnorm_quant, dtypes

    m, k = x.shape
    out = torch.empty(m, k // 2, dtype=dtypes.fp4x2, device=x.device)
    scale = torch.zeros(
        round_up(m, SCALE_ROW_PAD),
        round_up(k // GROUP, SCALE_COL_PAD),
        dtype=torch.uint8,
        device=x.device,
    )
    residual_out = torch.empty_like(x)
    add_rmsnorm_quant(
        out,
        x,
        residual,
        residual_out,
        scale,
        weight,
        eps,
        group_size=GROUP,
        shuffle_scale=True,
        gemma_norm=True,
    )
    return out, scale.view(dtypes.fp8_e8m0), residual_out


def assert_same_outputs(got, want, m: int, k: int) -> None:
    """Two results agree bit for bit on every position the kernel writes.

    Codes and the residual output are compared whole; scales only where the kernel
    writes them (aiter's own scale shuffle over zeros marks them: it fills the padding
    with 0x7F), the rest of that buffer is uninitialized memory.
    """
    from aiter.utility.fp4_utils import e8m0_shuffle

    zeros = torch.zeros(m, k // GROUP, dtype=torch.uint8, device=got[1].device)
    mask = e8m0_shuffle(zeros) == 0
    assert torch.equal(got[0].view(torch.uint8), want[0].view(torch.uint8)), "codes"
    assert torch.equal(
        got[1].view(torch.uint8)[mask], want[1].view(torch.uint8)[mask]
    ), "scales at the positions in use"
    assert torch.equal(got[2], want[2]), "residual_out"


@requires_gfx950_gpu
@pytest.mark.parametrize("k", [HIDDEN, ROCM_AITER_MXFP4_NORM_QUANT_MAX_K])
@pytest.mark.parametrize("m", [1, 33, 257])
def test_op_equals_the_direct_kernel(m, k):
    # Input: random bf16 x, residual and Gemma norm weight, m rows of k. Expected: the
    # op's codes, scales (where written) and residual output equal the kernel called
    # directly, and the shapes and dtypes are the documented ones. Why: a fusion pass
    # substitutes this op for the unfused norm and quant, so any difference from the
    # kernel is a numerics change.
    x, residual, weight = norm_inputs(m, 7, k)
    got = fused_op(x, residual, weight)
    assert [tuple(t.shape) for t in got] == list(expected_shapes(m, k))
    assert_same_outputs(got, direct_kernel(x, residual, weight), m, k)


@requires_gfx950_gpu
def test_op_leaves_its_inputs_alone_and_returns_a_fresh_residual():
    # Input: x, residual, weight. Expected: all three hold their old values after the
    # call, and the residual output shares no memory with x or residual. Why: the
    # graph node the pass replaces was out of place; the op must not turn it into an
    # in-place update of a tensor other nodes still read.
    x, residual, weight = norm_inputs(33, 9)
    before = [t.clone() for t in (x, residual, weight)]
    _, _, residual_out = fused_op(x, residual, weight)
    for tensor, old in zip((x, residual, weight), before):
        assert torch.equal(tensor, old)
    assert residual_out.data_ptr() not in (x.data_ptr(), residual.data_ptr())


@requires_gfx950_gpu
def test_op_with_no_rows_returns_empty_outputs_and_leaves_the_device_usable():
    # Input: an (0, K) x and residual. Expected: three empty tensors of the documented
    # shapes, and the next kernel launch works. Why: the kernel's empty grid is an
    # invalid launch that surfaces as an error in the NEXT kernel; the impl must skip
    # the launch.
    device = current_platform.device_type
    x = torch.zeros(0, HIDDEN, dtype=BF16, device=device)
    out = fused_op(
        x, torch.zeros_like(x), torch.zeros(HIDDEN, dtype=BF16, device=device)
    )
    assert [tuple(t.shape) for t in out] == list(expected_shapes(0, HIDDEN))
    assert torch.ones(1, device=device).sum().item() == 1  # a deferred error shows here


@requires_gfx950_gpu
@pytest.mark.parametrize(("x_rows", "residual_rows"), [(4, 3), (8192, 1)])
def test_op_rejects_a_residual_with_another_row_count(x_rows, residual_rows):
    # Input: x with x_rows rows, a residual with residual_rows. Expected: ValueError
    # naming both counts. Why: the kernel reads one residual row per x row, so a short
    # residual is read out of bounds. Only the impl, which sees concrete row counts,
    # can compare them.
    device = current_platform.device_type
    x = torch.zeros(x_rows, HIDDEN, dtype=BF16, device=device)
    residual = torch.zeros(residual_rows, HIDDEN, dtype=BF16, device=device)
    with pytest.raises(ValueError, match=rf"{x_rows} rows.*got {residual_rows}"):
        fused_op(x, residual, torch.zeros(HIDDEN, dtype=BF16, device=device))


@requires_gfx950_gpu
@pytest.mark.parametrize(("extra", "offset"), [(64, 0), (3, 1)])
def test_row_strided_views_give_the_contiguous_result(extra, offset):
    # Input: x and residual as views of wider buffers (row stride K + extra, inner
    # stride 1, starting `offset` elements into the row; the second case is odd
    # strided and misaligned). Expected: the outputs equal those on contiguous copies
    # of the same values, bit for bit. Why: the pass feeds the op tensors the graph
    # already had, which can be slices of a larger buffer; the kernel honours the row
    # stride and the op must neither refuse nor misread it.
    m = 33  # crosses the 32-row tile edge of the scale layout
    x_buffer, residual_buffer, _ = norm_inputs(m, 21, HIDDEN + extra)
    weight = norm_inputs(m, 21, HIDDEN)[2]
    x = x_buffer[:, offset : offset + HIDDEN]
    residual = residual_buffer[:, offset : offset + HIDDEN]
    assert x.stride() == residual.stride() == (HIDDEN + extra, 1)
    got = fused_op(x, residual, weight)
    want = fused_op(x.contiguous(), residual.contiguous(), weight)
    assert_same_outputs(got, want, m, HIDDEN)
