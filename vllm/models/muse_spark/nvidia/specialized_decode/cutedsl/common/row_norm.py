# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-tile RMS norm of one row, bitwise equal to the stock Triton norm
(``_rmsnorm_fwd_kernel``, single tile, 8 warps).

One CTA of ``THREADS`` threads per row, with the reference's thread mapping
(thread t holds 8 consecutive columns every ``THREADS * 8``) and its summation
order: per-thread tree (``thread_square_sum``), warp butterfly (``warp_sum``),
then the 8 warp sums (``warps_sum``). Every FP32 op is explicit PTX with its
rounding, so nothing contracts except where the reference does (``CONTRACTS``).
Rows are 8192 wide by default; the pooler's 4096-wide rows use the same order.
"""

import functools
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor

from .host import assert_not_capturing, check_operand
from .primitives import (
    add_rn,
    current_cu_stream,
    fma_rn,
    load_bf16x8_as_f32,
    load_f16x8_as_f32,
    load_f32x8,
    make_fake_dynamic_compact_tensor,
    mul_rn,
    regs,
    rsqrt_approx_ftz,
    store_bf16x8_rn,
    store_f32x8,
    to_cute,
)

H = 8192
WARP = 32
WARPS = 8  # the reference norm's num_warps
THREADS = WARPS * WARP
VEC = 8  # columns per fragment: 8 consecutive ones per thread
CHUNK = THREADS * VEC  # columns between one thread's fragments


# The weight forms of a norm: none, or a BF16, FP32 or FP16 weight applied
# after the norm or before it (input-scaled). Only the draft boundary takes
# FP16 weights.
WEIGHT_FORMS = (
    "none",
    "bf16",
    "fp32",
    "fp16",
    "bf16 input-scaled",
    "fp32 input-scaled",
    "fp16 input-scaled",
)

# The references the kernels reproduce.
# The stock norm (``_rmsnorm_fwd_kernel``, single tile, of fused_rmsnorm_fwd).
STOCK_NORM = "stock norm"
# The draft boundary's post-norm and pre-norm, with and without the stream
# stored (the pre-norm applies its weight after the norm).
DRAFT_POST_NORM = "draft boundary post-norm"
DRAFT_PRE_NORM = "draft boundary pre-norm"
DRAFT_PRE_NORM_NO_STREAM = "draft boundary pre-norm without the stream"
# The draft context K/V chain's hidden and layer norms.
CONTEXT_NORMS = "context norms"

# Whether the reference contracts the first level of a thread's sum of squares
# to ``fma(a, a, b * b)``, by (reference, weight form). This mirrors how the
# Triton reference is compiled; a test checks the stock rows against the
# installed Triton. A form a reference does not take has no entry.
CONTRACTS: dict[tuple[str, str], bool] = {
    (STOCK_NORM, "none"): False,
    (STOCK_NORM, "bf16"): False,
    (STOCK_NORM, "fp32"): True,
    (STOCK_NORM, "fp16"): False,
    (STOCK_NORM, "bf16 input-scaled"): True,
    (STOCK_NORM, "fp32 input-scaled"): True,
    (STOCK_NORM, "fp16 input-scaled"): True,
    (DRAFT_POST_NORM, "none"): False,
    (DRAFT_POST_NORM, "bf16"): False,
    (DRAFT_POST_NORM, "fp32"): False,
    (DRAFT_POST_NORM, "fp16"): False,
    (DRAFT_POST_NORM, "bf16 input-scaled"): True,
    (DRAFT_POST_NORM, "fp32 input-scaled"): True,
    (DRAFT_POST_NORM, "fp16 input-scaled"): True,
    (DRAFT_PRE_NORM, "none"): False,
    (DRAFT_PRE_NORM, "bf16"): False,
    (DRAFT_PRE_NORM, "fp32"): False,
    (DRAFT_PRE_NORM, "fp16"): False,
    (DRAFT_PRE_NORM, "bf16 input-scaled"): True,
    (DRAFT_PRE_NORM, "fp32 input-scaled"): True,
    (DRAFT_PRE_NORM, "fp16 input-scaled"): True,
    (DRAFT_PRE_NORM_NO_STREAM, "none"): False,
    (DRAFT_PRE_NORM_NO_STREAM, "bf16"): True,
    (DRAFT_PRE_NORM_NO_STREAM, "fp32"): True,
    (DRAFT_PRE_NORM_NO_STREAM, "fp16"): False,
    (DRAFT_PRE_NORM_NO_STREAM, "bf16 input-scaled"): True,
    (DRAFT_PRE_NORM_NO_STREAM, "fp32 input-scaled"): True,
    (DRAFT_PRE_NORM_NO_STREAM, "fp16 input-scaled"): True,
    (CONTEXT_NORMS, "none"): False,
    (CONTEXT_NORMS, "bf16"): False,
    (CONTEXT_NORMS, "fp32"): False,
    (CONTEXT_NORMS, "bf16 input-scaled"): True,
    (CONTEXT_NORMS, "fp32 input-scaled"): True,
}


def weight_form(
    has_weight: bool, input_scale: bool, weight_fp32: bool, weight_fp16: bool = False
) -> str:
    """A norm's entry in ``WEIGHT_FORMS`` (a BF16 weight unless FP32 or
    FP16)."""
    if not has_weight:
        return "none"
    form = "fp32" if weight_fp32 else "fp16" if weight_fp16 else "bf16"
    return f"{form} input-scaled" if input_scale else form


def contracts(
    reference: str,
    has_weight: bool,
    input_scale: bool,
    weight_fp32: bool,
    weight_fp16: bool = False,
) -> bool:
    """``CONTRACTS`` of ``reference`` for a norm of this weight form: the
    ``fused`` flag of ``thread_square_sum`` and ``chunk_pair``."""
    form = weight_form(has_weight, input_scale, weight_fp32, weight_fp16)
    return CONTRACTS[reference, form]


def _square_pair(a: Float32, b: Float32, fused: bool) -> Float32:
    """``a * a + b * b``, as ``fma(a, a, b * b)`` when ``fused``."""
    if fused:
        return fma_rn(a, a, mul_rn(b, b))
    return add_rn(mul_rn(a, a), mul_rn(b, b))


def _tree(values: list) -> Float32:
    """A balanced tree of add.rn over ``values`` (a power of two of them),
    neighbours first."""
    while len(values) > 1:
        values = [
            add_rn(values[2 * i], values[2 * i + 1]) for i in range(len(values) // 2)
        ]
    return values[0]


def chunk_pair(v: list, fused: bool) -> tuple:
    """The sum of squares of one 8-column fragment ``v`` as an (even, odd)
    pair, in the reference's order."""
    sums = [
        _square_pair(v[4 * half + lane], v[4 * half + 2 + lane], fused)
        for half in range(2)
        for lane in range(2)
    ]
    return add_rn(sums[0], sums[2]), add_rn(sums[1], sums[3])


def thread_sum(pairs: list) -> Float32:
    """A thread's sum from the (even, odd) pairs of its fragments: a balanced
    tree over each, then even + odd."""
    return add_rn(_tree([p[0] for p in pairs]), _tree([p[1] for p in pairs]))


def thread_square_sum(x: list, fused: bool) -> Float32:
    """A thread's sum of the squares of its values ``x`` (FP32, register order,
    whole fragments; traced inline): ``thread_sum`` of each fragment's
    ``chunk_pair``."""
    lanes = [
        _tree(
            [
                _square_pair(x[4 * i + lane], x[4 * i + 2 + lane], fused)
                for i in range(len(x) // 4)
            ]
        )
        for lane in range(2)
    ]
    return add_rn(lanes[0], lanes[1])


def warp_sum(total: Float32) -> Float32:
    """The reference's butterfly over the warp; every lane gets the sum."""
    for offset in (16, 8, 4, 2, 1):
        total = add_rn(total, cute.arch.shuffle_sync_bfly(total, offset=offset))
    return total


def warps_sum(w: list) -> Float32:
    """The row sum from the reference's ``WARPS`` warp sums, in its order."""
    return add_rn(
        add_rn(add_rn(w[0], w[4]), add_rn(w[2], w[6])),
        add_rn(add_rn(w[1], w[5]), add_rn(w[3], w[7])),
    )


@cute.jit
def row_sum(value: Float32, sums: cute.Pointer, warp: Int32, lane: Int32) -> Float32:
    """The row's sum of the threads' ``value`` in the reference order; every
    thread gets it. ``sums``: ``WARPS`` FP32 of shared memory, one buffer per
    call (a second call on the same buffer would race the first's reads)."""
    value = warp_sum(value)
    if lane == Int32(0):
        cute.arch.store((sums + warp).llvm_ptr, value)
    cute.arch.barrier()
    return warps_sum(
        [cute.arch.load((sums + i).llvm_ptr, Float32) for i in range(WARPS)]
    )


def rms_norm_row(
    x: list,
    gamma: list | None,
    eps: Float32,
    input_scale: bool,
    fused: bool,
    sums: cute.Pointer,
    warp: Int32,
    lane: Int32,
    width: int = H,
) -> list:
    """The RMS norm of one thread's values ``x`` (FP32, register order) before
    the output rounding (traced inline). ``gamma`` is the weight plus its gain
    center (None: no weight), applied before the norm with ``input_scale``,
    else after it."""
    if gamma is not None and input_scale:
        x = [mul_rn(x[i], gamma[i]) for i in range(len(x))]
    total = row_sum(thread_square_sum(x, fused), sums, warp, lane)
    r = rsqrt_approx_ftz(fma_rn(total, Float32(1.0 / width), eps))
    y = [mul_rn(v, r) for v in x]
    if gamma is not None and not input_scale:
        y = [mul_rn(y[i], gamma[i]) for i in range(len(y))]
    return y


def load_gamma(
    weight: cute.Tensor, column: Int64, center: float, fp32: bool, fp16: bool = False
) -> list:
    """The gains of the 8 columns from ``column``: the weight (FP32, FP16,
    else BF16) plus its gain center, added even when 0.0 as the reference does
    (traced inline)."""
    if fp32:
        raw = load_f32x8(weight, column)
    elif fp16:
        raw = load_f16x8_as_f32(weight, column)
    else:
        raw = load_bf16x8_as_f32(weight, column)
    return [add_rn(v, Float32(center)) for v in regs(raw)]


def _row_gains(
    weight: cute.Tensor | None, columns: list, center: float | None, fp32: bool
) -> list | None:
    """``load_gamma`` of each of a thread's fragments from ``columns``, in
    register order; None without a weight (traced inline)."""
    if weight is None or center is None:
        return None
    return [g for column in columns for g in load_gamma(weight, column, center, fp32)]


# The standalone row norm kernel: one CTA per row; optionally also stores the
# input rows as FP32 (``stream``, ``aux``).


class RowNormConstants(NamedTuple):
    """Compile-time constants of one row norm (bound at build)."""

    width: int
    eps: float
    # None: the norm has no weight.
    gain_center: float | None
    # The weight scales the rows before the norm (else the norm's output).
    input_scale: bool
    weight_fp32: bool
    enable_pdl: bool
    # FP32 input rows (else BF16).
    x_fp32: bool = False
    # Also store the input rows as FP32 (the residual stream), and with
    # ``aux`` a second copy (an aux layer's state).
    stream: bool = False
    aux: bool = False
    # Release the PDL dependents right after the PDL wait (else at the
    # grid's end).
    release_early: bool = False


def row_norm_constants(norm, *, width: int = H, enable_pdl: bool, **layout):
    """The constants of the stock norm ``norm`` (a MuseSpark1xRMSNorm or its
    input-scaled variant: the arguments its forward hands fused_rmsnorm_fwd)
    over rows of ``width``, with the ``layout`` fields of RowNormConstants;
    ValueError where the kernel does not reproduce it or does not take its
    weight."""
    if (
        getattr(norm, "variance_size_override", None) is not None
        or norm.hidden_size != width
    ):
        raise ValueError(f"the norm must reduce over the whole row of {width}")
    weight = norm.weight if norm.has_weight else None
    if weight is not None:
        if weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"the norm weight must be a contiguous BF16 or FP32 [{width}] tensor"
            )
        # Each thread loads 8 consecutive weights: 32 bytes in FP32, 16 in BF16.
        check_operand(
            weight, (width,), weight.dtype, 8 * weight.dtype.itemsize, "the norm weight"
        )
    return RowNormConstants(
        width=width,
        eps=float(norm.variance_epsilon),
        gain_center=float(norm.fused_gain_center) if weight is not None else None,
        input_scale=bool(norm.applies_weight_before_norm and weight is not None),
        weight_fp32=weight is not None and weight.dtype == torch.float32,
        enable_pdl=bool(enable_pdl),
        **layout,
    )


class _RowNormKernel:
    """Grid (m, 1, 1), THREADS threads: one row per CTA."""

    def __init__(self, constants: RowNormConstants) -> None:
        self.constants = constants

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor | None,
        out: cute.Tensor,
        stream_out: cute.Tensor | None,
        aux_out: cute.Tensor | None,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(x, weight, out, stream_out, aux_out).launch(
            grid=(m, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.constants.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor | None,
        out: cute.Tensor,
        stream_out: cute.Tensor | None,
        aux_out: cute.Tensor | None,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cu_utils.SmemAllocator()
        sums = smem.allocate_array(Float32, WARPS)
        has_weight = constants.gain_center is not None
        passes = constants.width // CHUNK
        columns = [Int64(tidx) * VEC + Int64(k * CHUNK) for k in range(passes)]

        # The weight does not change between calls: load before the PDL wait.
        gamma = _row_gains(
            weight, columns, constants.gain_center, constants.weight_fp32
        )

        if const_expr(constants.enable_pdl):
            cute.arch.griddepcontrol_wait()
            if const_expr(constants.release_early):
                cute.arch.griddepcontrol_launch_dependents()
        base = Int64(row) * constants.width
        values: list = []
        for k in cutlass.range_constexpr(passes):
            if const_expr(constants.x_fp32):
                chunk = load_f32x8(x, base + columns[k])
            else:
                chunk = load_bf16x8_as_f32(x, base + columns[k])
            if const_expr(constants.stream):
                store_f32x8(stream_out, base + columns[k], chunk)
            if const_expr(constants.aux):
                store_f32x8(aux_out, base + columns[k], chunk)
            values.extend(regs(chunk))

        y = rms_norm_row(
            values,
            gamma,
            Float32(constants.eps),
            constants.input_scale,
            contracts(
                STOCK_NORM,
                has_weight,
                constants.input_scale,
                constants.weight_fp32,
            ),
            sums,
            warp,
            lane,
            width=constants.width,
        )
        for k in cutlass.range_constexpr(passes):
            store_bf16x8_rn(out, base + columns[k], y[k * VEC : (k + 1) * VEC])


@functools.cache
def compile_row_norm(constants: RowNormConstants):
    """cute.compile one row norm; the row count is a runtime argument. Call:
    (x, weight, out, stream_out, aux_out, m, stream) with the rows dynamic
    views (``host.rows_view``) and the weight an [width] view, None where the
    constants have no weight, stream or aux. Call before CUDA graph
    capture."""
    assert_not_capturing("a row norm")
    width = constants.width
    rows_f32 = make_fake_dynamic_compact_tensor(
        Float32, alignment=16, divisibility=width
    )
    rows_bf16 = make_fake_dynamic_compact_tensor(
        BFloat16, alignment=16, divisibility=width
    )
    weight = None
    if constants.gain_center is not None:
        weight = make_fake_compact_tensor(
            Float32 if constants.weight_fp32 else BFloat16,
            (width,),
            assumed_align=32 if constants.weight_fp32 else 16,
        )
    return cute.compile(
        _RowNormKernel(constants),
        rows_f32 if constants.x_fp32 else rows_bf16,
        weight,
        rows_bf16,
        rows_f32 if constants.stream else None,
        rows_f32 if constants.aux else None,
        Int32(1),
        current_cu_stream(),
    )


class RowNorm:
    """One row norm, compiled and its weight bound at construction (before
    CUDA graph capture; one compile per process and constant set):
    ``launch(x, out, m, stream_out, aux_out)`` on rows views of at least
    ``m`` rows. Graph-capturable; allocates nothing."""

    def __init__(
        self, constants: RowNormConstants, weight: torch.Tensor | None
    ) -> None:
        if (weight is None) != (constants.gain_center is None):
            raise ValueError("the norm weight does not match the constants")
        self.constants = constants
        self.kernel = compile_row_norm(constants)
        # The tensor stays referenced for as long as its view.
        self.weight = weight
        self._weight_view = (
            None
            if weight is None
            else to_cute(weight, 32 if constants.weight_fp32 else 16)
        )

    def launch(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        m: int,
        stream_out: cute.Tensor | None = None,
        aux_out: cute.Tensor | None = None,
    ) -> None:
        self.kernel(
            x,
            self._weight_view,
            out,
            stream_out,
            aux_out,
            Int32(m),
            current_cu_stream(),
        )
