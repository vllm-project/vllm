# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft's residual boundary, bitwise equal to the eager chain.

One boundary is the eager post -> connection -> pre chain: ``post_norm(branch)``
rounded to BF16, the gated FP32 residual ``alpha * stream + beta * branch``
with each product rounded, then the next pre-norm of the stream rounded to
BF16. The norms use the stock single-tile norm's order and roundings
(``row_norm.py``). The branch is read as is (-0.0, subnormals, NaN and
infinities included).

``boundary_fragments`` is that arithmetic over one thread's 8-column
fragments, with the row sums supplied by the caller. Two kernels use it:

- ``DraftBoundary`` here, for an already all-reduced branch: one CTA of 256
  threads per row, each thread holding four fragments;
- the boundary with the branch's all-reduce fused in
  (``cutedsl/oproj/draft_all_reduce_boundary.py``), one fragment per thread
  over a cluster of 8 CTAs.

Gates and norm weights load before the PDL wait for up to one row per SM, and
after it above that (``late_gates``). With PDL, every CTA releases its
dependents right after the wait; the next kernel waits for this grid's end
before it reads the output.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor

from .host import assert_not_capturing, check_operand, pdl_supported
from .primitives import (
    VEC_BF16,
    add_rn,
    current_cu_stream,
    fma_rn,
    load_bf16x8_as_f32,
    load_f32x8,
    make_fake_dynamic_compact_tensor,
    mul_rn,
    pack_bf16x8_rn,
    regs,
    round_bf16,
    rsqrt_approx_ftz,
    store_f32x8,
    store_global_u32x4,
    to_cute,
    to_cute_dynamic,
)
from .row_norm import (
    CHUNK,
    DRAFT_POST_NORM,
    DRAFT_PRE_NORM,
    DRAFT_PRE_NORM_NO_STREAM,
    THREADS,
    VEC,
    WARPS,
    H,
    chunk_pair,
    contracts,
    load_gamma,
    row_sum,
    thread_sum,
)

FRAGMENTS = H // CHUNK  # per reference thread
INV_H = 1.0 / H

assert FRAGMENTS * CHUNK == H and VEC == VEC_BF16


class DraftBoundaryConstants(NamedTuple):
    """Compile-time constants of one draft boundary (bound at build)."""

    post_eps: float
    pre_eps: float
    # None: that norm has no weight.
    post_gain_center: float | None
    pre_gain_center: float | None
    # The post-norm weight scales the branch before the norm (else after it).
    post_input_scale: bool
    # The norm weights' dtypes: FP32 or FP16, else BF16.
    post_weight_fp32: bool = False
    pre_weight_fp32: bool = False
    # The pre-norm weight scales the stream before the norm (else after it).
    pre_input_scale: bool = False
    post_weight_fp16: bool = False
    pre_weight_fp16: bool = False


# The norm weight dtypes the kernels load.
WEIGHT_DTYPES = (torch.bfloat16, torch.float32, torch.float16)


def draft_boundary_constants(post, connection, pre) -> DraftBoundaryConstants:
    """The constants of the stock draft boundary ``post`` -> ``connection`` ->
    ``pre`` (MuseSpark1xRMSNorm or its input-scaled variant,
    MuseSpark1xNormPreservingResidualConnection), or ValueError where the
    kernels do not reproduce it."""
    if not getattr(connection, "_coefficients_ready", True):
        raise ValueError("residual coefficients are not materialized")
    for norm in (post, pre):
        if norm.has_weight and norm.weight.dtype not in WEIGHT_DTYPES:
            raise ValueError("the norm weights must be BF16, FP32 or FP16")

    def dtype_is(norm, dtype) -> bool:
        return norm.has_weight and norm.weight.dtype == dtype

    return DraftBoundaryConstants(
        post_eps=float(post.variance_epsilon),
        pre_eps=float(pre.variance_epsilon),
        post_gain_center=float(post.fused_gain_center) if post.has_weight else None,
        pre_gain_center=float(pre.fused_gain_center) if pre.has_weight else None,
        post_input_scale=bool(post.has_weight and post.applies_weight_before_norm),
        post_weight_fp32=dtype_is(post, torch.float32),
        pre_weight_fp32=dtype_is(pre, torch.float32),
        pre_input_scale=bool(pre.has_weight and pre.applies_weight_before_norm),
        post_weight_fp16=dtype_is(post, torch.float16),
        pre_weight_fp16=dtype_is(pre, torch.float16),
    )


class BoundaryGates(NamedTuple):
    """One fragment's gates and norm gains (``[]`` where a norm has no
    weight), each 8 FP32."""

    alpha: list
    beta: list
    post_gamma: list
    pre_gamma: list


def load_gates(
    constants: DraftBoundaryConstants,
    alpha: cute.Tensor,
    beta: cute.Tensor,
    post_weight: cute.Tensor,
    pre_weight: cute.Tensor,
    column: Int64,
) -> BoundaryGates:
    """The gates and gains of the 8 columns from ``column`` (traced inline)."""

    def gamma(weight, center, fp32, fp16) -> list:
        if center is None:
            return []
        return load_gamma(weight, column, center, fp32, fp16)

    return BoundaryGates(
        alpha=regs(load_f32x8(alpha, column)),
        beta=regs(load_f32x8(beta, column)),
        post_gamma=gamma(
            post_weight,
            constants.post_gain_center,
            constants.post_weight_fp32,
            constants.post_weight_fp16,
        ),
        pre_gamma=gamma(
            pre_weight,
            constants.pre_gain_center,
            constants.pre_weight_fp32,
            constants.pre_weight_fp16,
        ),
    )


# (a thread's fragments, whether the first level contracts, round: 0 for the
# post-norm, 1 for the pre-norm) -> the row's sum of squares.
RowSum = Callable[[list, bool, int], Float32]


def boundary_fragments(
    constants: DraftBoundaryConstants,
    x: list,
    stream: list,
    gates: list,
    row_sum: RowSum,
    store_stream: Callable[[int, Any], None] | None,
) -> list:
    """One thread's fragments of a boundary: ``x`` (the BF16 branch) and
    ``stream`` as lists of 8 FP32 per fragment, ``gates`` a ``BoundaryGates``
    per fragment. ``store_stream(k, values)`` stores fragment k's new stream
    (None: not stored). Returns per fragment the output as 4 BF16x2 words."""
    post_weight = constants.post_gain_center is not None
    pre_weight = constants.pre_gain_center is not None
    input_scaled = post_weight and constants.post_input_scale
    pre_input_scaled = pre_weight and constants.pre_input_scale
    if input_scaled:
        x = [
            [mul_rn(g.post_gamma[i], f[i]) for i in range(VEC_BF16)]
            for f, g in zip(x, gates)
        ]
    post_sum = row_sum(
        x,
        contracts(
            DRAFT_POST_NORM,
            post_weight,
            constants.post_input_scale,
            constants.post_weight_fp32,
            constants.post_weight_fp16,
        ),
        0,
    )
    pre_fused = contracts(
        DRAFT_PRE_NORM if store_stream is not None else DRAFT_PRE_NORM_NO_STREAM,
        pre_weight,
        constants.pre_input_scale,
        constants.pre_weight_fp32,
        constants.pre_weight_fp16,
    )
    r1 = rsqrt_approx_ftz(fma_rn(post_sum, Float32(INV_H), Float32(constants.post_eps)))
    z = []
    for k, (f, g) in enumerate(zip(x, gates)):
        y = [mul_rn(f[i], r1) for i in range(VEC_BF16)]
        if post_weight and not input_scaled:
            y = [mul_rn(y[i], g.post_gamma[i]) for i in range(VEC_BF16)]
        res = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        zk = []
        for i in range(VEC_BF16):
            res[i] = add_rn(
                mul_rn(g.alpha[i], stream[k][i]),
                mul_rn(g.beta[i], round_bf16(y[i])),
            )
            zk.append(round_bf16(res[i]))
        z.append(zk)
        if store_stream is not None:
            store_stream(k, res.load())
    if pre_input_scaled:
        z = [
            [mul_rn(f[i], g.pre_gamma[i]) for i in range(VEC_BF16)]
            for f, g in zip(z, gates)
        ]
    pre_sum = row_sum(z, pre_fused, 1)
    r2 = rsqrt_approx_ftz(fma_rn(pre_sum, Float32(INV_H), Float32(constants.pre_eps)))
    out = []
    for f, g in zip(z, gates):
        o = [mul_rn(f[i], r2) for i in range(VEC_BF16)]
        if pre_weight and not pre_input_scaled:
            o = [mul_rn(o[i], g.pre_gamma[i]) for i in range(VEC_BF16)]
        out.append(pack_bf16x8_rn(o))
    return out


@dataclass(frozen=True)
class _CtaRowSum:
    """``RowSum`` of a CTA that holds the row: ``sums`` 2 * WARPS FP32 of
    shared memory (one round each)."""

    sums: cute.Pointer
    warp: Int32
    lane: Int32

    def __call__(self, fragments: list, fused: bool, round: int) -> Float32:
        return row_sum(
            thread_sum([chunk_pair(f, fused) for f in fragments]),
            self.sums + round * WARPS,
            self.warp,
            self.lane,
        )


@dataclass(frozen=True)
class _StreamStore:
    """Stores fragment k of the new stream (FP32 rows) of row offset ``row``."""

    stream_out: cute.Tensor
    row: Int64
    columns: list

    def __call__(self, k: int, values) -> None:
        store_f32x8(self.stream_out, self.row + self.columns[k], values)


class _DraftBoundaryKernel:
    """Grid (m,), THREADS threads: CTA = row, thread t = reference thread t."""

    def __init__(
        self,
        constants: DraftBoundaryConstants,
        stream_bf16: bool,
        store_stream: bool,
        late_gates: bool,
        pdl: bool,
    ) -> None:
        self.constants = constants
        self.stream_bf16 = stream_bf16
        self.store_stream = store_stream
        self.late_gates = late_gates
        self.pdl = pdl

    @cute.jit
    def __call__(
        self,
        branch: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor | None,
        pre_weight: cute.Tensor | None,
        stream_out: cute.Tensor | None,
        out: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            branch, stream_in, alpha, beta, post_weight, pre_weight, stream_out, out
        ).launch(
            grid=(m, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        branch: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor | None,
        pre_weight: cute.Tensor | None,
        stream_out: cute.Tensor | None,
        out: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cu_utils.SmemAllocator()
        # One round of warp sums per norm.
        sums = smem.allocate_array(Float32, 2 * WARPS)
        columns = [Int64(tidx) * VEC_BF16 + Int64(k * CHUNK) for k in range(FRAGMENTS)]
        gates: list = []
        if const_expr(not self.late_gates):
            gates = [
                load_gates(constants, alpha, beta, post_weight, pre_weight, column)
                for column in columns
            ]
        if const_expr(self.pdl):
            # The next kernel may launch now: it waits for this grid's end
            # before it reads.
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        row = Int64(token) * H
        x = [regs(load_bf16x8_as_f32(branch, row + column)) for column in columns]
        if const_expr(self.stream_bf16):
            stream_values = [
                regs(load_bf16x8_as_f32(stream_in, row + column)) for column in columns
            ]
        else:
            stream_values = [
                regs(load_f32x8(stream_in, row + column)) for column in columns
            ]
        if const_expr(self.late_gates):
            gates = [
                load_gates(constants, alpha, beta, post_weight, pre_weight, column)
                for column in columns
            ]
        store = None
        if const_expr(self.store_stream):
            store = _StreamStore(stream_out, row, columns)
        words = boundary_fragments(
            constants, x, stream_values, gates, _CtaRowSum(sums, warp, lane), store
        )
        for k in cutlass.range_constexpr(FRAGMENTS):
            store_global_u32x4(
                Int64((out.iterator + (row + columns[k])).toint()), words[k].load()
            )


@functools.cache
def compile_draft_boundary(
    *,
    constants: DraftBoundaryConstants,
    stream_bf16: bool,
    store_stream: bool,
    pdl: bool,
    late_gates: bool = False,
):
    """cute.compile one kernel; the row count is a runtime argument. Call:
    (branch, stream_in, alpha, beta, post_weight, pre_weight, stream_out,
    out, m, stream), a weight None where its norm has none and
    ``stream_out`` None without ``store_stream``; ``pdl``: launched with
    programmatic dependent launch (``host.pdl_supported``). Call before CUDA
    graph capture."""
    assert_not_capturing("the draft boundary")
    gate = make_fake_compact_tensor(Float32, (H,), assumed_align=32)
    rows_f32 = make_fake_dynamic_compact_tensor(Float32, alignment=32, divisibility=H)
    rows_bf16 = make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=H)
    return cute.compile(
        _DraftBoundaryKernel(constants, stream_bf16, store_stream, late_gates, pdl),
        rows_bf16,
        rows_bf16 if stream_bf16 else rows_f32,
        gate,
        gate,
        *fake_weights(constants),
        rows_f32 if store_stream else None,
        rows_bf16,
        Int32(1),
        current_cu_stream(),
    )


def fake_weights(constants: DraftBoundaryConstants) -> tuple:
    """The compile-time post and pre norm weights of a boundary: an [8192]
    FP32, FP16 or BF16 tensor, None where that norm has no weight."""

    def weight(center, fp32, fp16):
        if center is None:
            return None
        if fp32:
            return make_fake_compact_tensor(Float32, (H,), assumed_align=32)
        dtype = Float16 if fp16 else BFloat16
        return make_fake_compact_tensor(dtype, (H,), assumed_align=16)

    return (
        weight(
            constants.post_gain_center,
            constants.post_weight_fp32,
            constants.post_weight_fp16,
        ),
        weight(
            constants.pre_gain_center,
            constants.pre_weight_fp32,
            constants.pre_weight_fp16,
        ),
    )


class BoundaryOperands:
    """A boundary's constant operands, checked and viewed once: the gates
    (``alpha``, ``beta`` FP32 [8192]) and the norm weights, where the
    constants have a gain center (else None: the kernels take none)."""

    def __init__(
        self,
        constants: DraftBoundaryConstants,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        pre_weight: torch.Tensor | None,
    ) -> None:
        # Each thread loads 8 consecutive gates and weights: 32 bytes in FP32,
        # 16 in BF16 and FP16.
        for gate, name in ((alpha, "alpha"), (beta, "beta")):
            check_operand(gate, (H,), torch.float32, 32, name)
        if (post_weight is None) != (constants.post_gain_center is None) or (
            pre_weight is None
        ) != (constants.pre_gain_center is None):
            raise ValueError("the norm weights do not match the constants")
        for weight in (post_weight, pre_weight):
            if weight is not None:
                check_operand(
                    weight,
                    (H,),
                    weight.dtype,
                    8 * weight.dtype.itemsize,
                    "the norm weights",
                )
        # The tensors stay referenced for as long as their views.
        self.tensors = (alpha, beta, post_weight, pre_weight)
        self.views = (
            to_cute(alpha, 32),
            to_cute(beta, 32),
            self._weight_view(post_weight),
            self._weight_view(pre_weight),
        )

    @staticmethod
    def _weight_view(weight: torch.Tensor | None) -> cute.Tensor | None:
        if weight is None:
            return None
        return to_cute(weight, 32 if weight.dtype == torch.float32 else 16)


class DraftBoundary:
    """One draft boundary of an all-reduced branch, bound at build: its
    constants and its constant operands (``BoundaryOperands``), and its
    kernel of each variant by (stream BF16, store stream, late gates: rows
    above the device's SM count), compiled once per process and variant
    (``prepare`` them before CUDA graph capture)."""

    def __init__(
        self,
        constants: DraftBoundaryConstants,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        pre_weight: torch.Tensor | None,
    ) -> None:
        self.constants = constants
        self.operands = BoundaryOperands(
            constants, alpha, beta, post_weight, pre_weight
        )
        self.pdl = pdl_supported()
        # Rows above one CTA per SM load the gates after the PDL wait.
        self.late_rows = (
            torch.cuda.get_device_properties(alpha.device).multi_processor_count
            if alpha.is_cuda
            else 0
        )
        # The kernel of each variant prepared (or run) so far.
        self._kernels: dict[tuple[bool, bool, bool], Any] = {}

    def prepare(self, stream_bf16: bool, store_stream: bool, late_gates: bool = False):
        """The kernel of one variant, kept for the calls that launch it
        (``compile_draft_boundary``: compiled on its first use in the
        process)."""
        key = (stream_bf16, store_stream, late_gates)
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = self._kernels[key] = compile_draft_boundary(
                constants=self.constants,
                stream_bf16=stream_bf16,
                store_stream=store_stream,
                pdl=self.pdl,
                late_gates=late_gates,
            )
        return kernel

    @staticmethod
    def takes(branch: torch.Tensor, stream: torch.Tensor) -> bool:
        """Whether the kernel takes these rows: a dense BF16 [m, 8192] branch
        (m >= 1) and an FP32 or BF16 stream of its shape."""
        return (
            branch.dim() == 2
            and branch.shape[0] >= 1
            and branch.shape[1] == H
            and branch.dtype == torch.bfloat16
            and branch.is_contiguous()
            and stream.shape == branch.shape
            and stream.dtype in (torch.float32, torch.bfloat16)
            and stream.is_contiguous()
        )

    def run(
        self, branch: torch.Tensor, stream: torch.Tensor, store_stream: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``(new_stream or None, out)``: the FP32 ``alpha * stream + beta *
        post_norm(branch)`` and the BF16 ``pre_norm(new_stream.to(bf16))`` of
        rows ``takes`` accepts. Graph-capturable once prepared."""
        if not self.takes(branch, stream):
            raise ValueError(
                f"the draft boundary takes a dense BF16 [m, {H}] branch and an "
                "FP32 or BF16 stream of its shape"
            )
        return self.launch(branch, stream, store_stream)

    def launch(
        self, branch: torch.Tensor, stream: torch.Tensor, store_stream: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``run`` without its operand check, for rows the caller has
        checked with ``takes``."""
        m = branch.shape[0]
        kernel = self.prepare(
            stream.dtype == torch.bfloat16, store_stream, m > self.late_rows
        )
        out = torch.empty((m, H), dtype=torch.bfloat16, device=branch.device)
        new_stream = (
            torch.empty((m, H), dtype=torch.float32, device=branch.device)
            if store_stream
            else None
        )
        kernel(
            to_cute_dynamic(branch.flatten(), 16, divisibility=H),
            to_cute_dynamic(
                stream.flatten(),
                16 if stream.dtype == torch.bfloat16 else 32,
                divisibility=H,
            ),
            *self.operands.views,
            (
                to_cute_dynamic(new_stream.flatten(), 32, divisibility=H)
                if new_stream is not None
                else None
            ),
            to_cute_dynamic(out.flatten(), 16, divisibility=H),
            Int32(m),
            current_cu_stream(),
        )
        return new_stream, out
