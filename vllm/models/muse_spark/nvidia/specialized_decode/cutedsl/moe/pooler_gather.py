# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 by FlashInfer team
# The gather + RMSNorm + NVFP4 quantization kernel is adapted from FlashInfer
# (flashinfer/comm/mnnvl_cutedsl/kernel_ll/all_gather_norm_quant.py) and its
# FP4 conversion helpers are copied from flashinfer/cute_dsl/fp4_common.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0; the mailbox polls are bounded here.
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The pooler of the Muse Spark specialized decode.

Each TP rank projects the replicated tokens onto its contiguous block of the
pooler's output columns, `[b, P / tp] = x [b, H] @ W[rank block].T` (the
pooler GEMM), multicasts that block into a Lamport mailbox and gathers the
full `[b, P]` rows from every rank straight into the expert input:

- NVFP4 layers (`ExpertsInput`): `pre_expert_norm` and NVFP4 quantization
  fused into the gather::

      y = bf16(x * rsqrt(mean(x**2) + eps) * (weight + gain_center))
      a, a_sf = nvfp4(y, global_scale, block 16)

  `a` is uint8 `[b, P / 2]` (two e2m1 codes per byte, the even column in
  the low nibble) and `a_sf` float8_e4m3fn `[b, P / 16]`, not swizzled.
  These are the bytes the stock all-gather + norm + quantize hands the routed
  experts, bit for bit.
- BF16 edge layers (`ExpertsInputBf16`): the gathered rows, then `pre_expert_norm`
  (`expert_norm.ExpertNorm`), bit for bit the stock gather + norm except for
  the sign of zero: publishers store BF16 -0.0 as +0.0 (-0.0 is the mailbox
  sentinel).

The pooler GEMM is the unpooler's cluster split-K GEMM (`gemm.unpooler`, BF16
in and out, FP32 accumulation) at the launch shape `POOLER`; its epilogue
publishes the block into the mailbox (zero-copy), the bytes and stage word the
mailbox's own publisher would write. One mailbox serves every layer; calls must
run in the same order on every rank.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from ..common.dual_norm import cluster_row_sum
from ..common.host import assert_not_capturing
from ..common.mailbox import LamportMailbox, poll_owned_fragment
from ..common.policy import MAILBOX_CAPACITY, POOLER
from ..common.primitives import (
    VEC_BF16,
    WARP_SIZE,
    current_cu_stream,
    load_bf16x8_as_f32,
    make_fake_dynamic_compact_tensor,
    packed_u32x4_to_bf16x8,
    store_global_u32,
    store_global_u32x4,
    to_cute,
    to_cute_dynamic,
)
from ..gemm.unpooler import Unpooler
from .expert_norm import ExpertNorm, rows_view

HIDDEN = 8192
POOLED = 4096
_THREADS = 128
_NVFP4_BLOCK = 16


# --------------------------------------------------------------------------
# FP4 helpers. The E4M3 widening of the output scale has no CuTe DSL form.


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """`a` as a saturated E4M3 byte (round to nearest even), zero-extended."""
    pair = vector.from_elements(
        ir.VectorType.get([2], T.f32(), loc=loc),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(0.0).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    fp8 = cute.TensorSSA(pair, 2, Float32).to(Float8E4M3FN)
    bits = llvm.bitcast(T.i16(), fp8.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    return Uint32(Uint16(bits))


@dsl_user_op
def nvfp4_compute_output_scale(
    fp8_val: Uint32, global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """The NVFP4 output scale `rcp(float(sf) * rcp(global_scale))`, 0 when the
    E4M3 scale is zero (the E4M3 byte widened through the hardware f16x2 path)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .b16 fp8_pair;
                .reg .b32 h2_32;
                .reg .b16 h_lo, h_hi;
                .reg .f32 scale_f32, rcp_gs, product, result;

                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
                mov.b32 {h_lo, h_hi}, h2_32;
                cvt.f32.f16 scale_f32, h_lo;

                rcp.approx.ftz.f32 rcp_gs, $2;
                mul.f32 product, scale_f32, rcp_gs;
                rcp.approx.ftz.f32 result, product;

                setp.eq.f32 p_zero, scale_f32, 0f00000000;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_e2m1x8_f32(*values: Float32, loc=None, ip=None) -> Uint32:
    """Eight FP32 values as saturated E2M1 codes (round to nearest even), the
    first in the low nibble."""
    vec = vector.from_elements(
        ir.VectorType.get([8], T.f32(), loc=loc),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in values],
        loc=loc,
        ip=ip,
    )
    codes = cute.TensorSSA(vec, 8, Float32).to(Float4E2M1FN)
    return Uint32(llvm.bitcast(T.i32(), codes.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


# --------------------------------------------------------------------------
# The gather consumers.


@dataclass(frozen=True, slots=True)
class _NormQuantConstants:
    eps: float
    gain_center: float


class _AllGatherNormQuantDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        constants: _NormQuantConstants,
        enable_pdl: bool,
    ) -> None:
        self.hidden = hidden
        self.tp = tp
        self.hidden_local = hidden // tp
        self.capacity_m = capacity_m
        self.cluster_size = hidden // (_THREADS * VEC_BF16)
        self.warps = _THREADS // WARP_SIZE
        self.constants = constants
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        weight: cute.Tensor,
        global_scale: cute.Tensor,
        packed_output: cute.Tensor,
        scale_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            weight,
            global_scale,
            packed_output,
            scale_output,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(_THREADS, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.cluster_size * self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        weight: cute.Tensor,
        global_scale: cute.Tensor,
        packed_output: cute.Tensor,
        scale_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        fragment = cluster_rank * _THREADS + tidx
        # Every peer must have started before cluster_row_sum stores into its
        # shared memory; the matching wait sits just before that first store.
        cute.arch.cluster_arrive_relaxed()

        gamma = load_bf16x8_as_f32(weight, Int64(fragment) * VEC_BF16) + Float32(
            self.constants.gain_center
        )
        scale = global_scale[0]

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        poll_owned_fragment(
            contribution_mailbox,
            stage_state,
            packed,
            token,
            fragment,
            self.tp,
            self.capacity_m,
            self.hidden_local,
            self.enable_pdl,
        )

        smem = cutlass.memory.SmemAllocator()
        slots = smem.allocate_array(Float32, self.cluster_size * self.warps)
        x = packed_u32x4_to_bf16x8(packed.load()).to(Float32)
        cute.arch.cluster_wait()
        sum_sq = cluster_row_sum(
            (x * x).reduce(
                cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
            ),
            slots,
            self.warps,
            self.cluster_size,
            cluster_rank,
            warp,
            lane,
        )
        inv_rms = cute.math.rsqrt(
            sum_sq * Float32(1.0 / self.hidden) + Float32(self.constants.eps),
            fastmath=True,
        )
        # The unfused norm rounds to BF16 before the quantization reads it.
        values = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
        values.store(((x * inv_rms) * gamma).to(BFloat16).to(Float32))

        # A 16-column scale block spans this lane and its neighbour.
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(VEC_BF16):
            amax = cute.arch.fmax(amax, cute.math.abs(values[i]))
        amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        sf = cvt_f32_to_e4m3(scale * (amax * cute.arch.rcp_approx(Float32(6.0))))
        output_scale = nvfp4_compute_output_scale(sf, scale)
        store_global_u32(
            packed_output.iterator
            + (Int64(token) * (self.hidden // 2) + Int64(fragment) * (VEC_BF16 // 2)),
            cvt_e2m1x8_f32(
                values[0] * output_scale,
                values[1] * output_scale,
                values[2] * output_scale,
                values[3] * output_scale,
                values[4] * output_scale,
                values[5] * output_scale,
                values[6] * output_scale,
                values[7] * output_scale,
            ),
        )
        if lane % 2 == 0:
            scale_output[token * (self.hidden // _NVFP4_BLOCK) + fragment // 2] = Uint8(
                sf & Uint32(0xFF)
            )


class _AllGatherDeviceKernel:
    """The plain column all-gather of the BF16 edge layers: one thread per 8
    global columns of one row, read from the rank that owns them and stored
    into `output` `[m, hidden]` BF16. Grid (m, hidden / (128 * 8))."""

    def __init__(
        self, *, hidden: int, tp: int, capacity_m: int, enable_pdl: bool
    ) -> None:
        self.hidden = hidden
        self.tp = tp
        self.hidden_local = hidden // tp
        self.capacity_m = capacity_m
        self.ctas = hidden // (_THREADS * VEC_BF16)
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(contribution_mailbox, output, stage_state).launch(
            grid=(m, self.ctas, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, cta, _ = cute.arch.block_idx()
        fragment = cta * _THREADS + tidx

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        poll_owned_fragment(
            contribution_mailbox,
            stage_state,
            packed,
            token,
            fragment,
            self.tp,
            self.capacity_m,
            self.hidden_local,
            self.enable_pdl,
        )
        element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
        store_global_u32x4(Int64((output.iterator + element).toint()), packed.load())


def _check_gather_shape(hidden: int, tp: int) -> None:
    if hidden % (_THREADS * VEC_BF16) or not 1 <= hidden // (_THREADS * VEC_BF16) <= 8:
        raise ValueError("hidden must be 1024 * n for n in [1, 8]")
    if hidden % (tp * VEC_BF16):
        raise ValueError("hidden must split into 8-element rank shards")


def compile_norm_quant(
    *,
    hidden: int,
    tp: int,
    capacity_m: int,
    eps: float,
    gain_center: float,
    enable_pdl: bool = True,
) -> Any:
    """The gather + norm + NVFP4 consumer over a `[3, tp, capacity_m,
    hidden / tp]` mailbox. Call: (mailbox, weight [hidden] BF16, global_scale
    [1] FP32, packed [m, hidden / 2] u8, scales [m, hidden / 16] u8, stage
    state, m, stream) as CuTe views."""
    _check_gather_shape(hidden, tp)
    assert_not_capturing("the pooler gather")
    device_kernel = _AllGatherNormQuantDeviceKernel(
        hidden=hidden,
        tp=tp,
        capacity_m=capacity_m,
        constants=_NormQuantConstants(float(eps), float(gain_center)),
        enable_pdl=enable_pdl,
    )
    return cute.compile(
        device_kernel,
        make_fake_compact_tensor(
            BFloat16, (3 * tp * capacity_m * (hidden // tp),), assumed_align=16
        ),
        make_fake_compact_tensor(BFloat16, (hidden,), assumed_align=16),
        make_fake_compact_tensor(Float32, (1,), assumed_align=4),
        make_fake_dynamic_compact_tensor(Uint8, alignment=16, divisibility=hidden // 2),
        make_fake_dynamic_compact_tensor(
            Uint8, alignment=4, divisibility=hidden // _NVFP4_BLOCK
        ),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int32(capacity_m),
        current_cu_stream(),
    )


def compile_gather(
    *, hidden: int, tp: int, capacity_m: int, enable_pdl: bool = True
) -> Any:
    """The plain gather consumer. Call: (mailbox, output [m, hidden] BF16,
    stage state, m, stream) as CuTe views."""
    _check_gather_shape(hidden, tp)
    assert_not_capturing("the pooler gather")
    device_kernel = _AllGatherDeviceKernel(
        hidden=hidden, tp=tp, capacity_m=capacity_m, enable_pdl=enable_pdl
    )
    return cute.compile(
        device_kernel,
        make_fake_compact_tensor(
            BFloat16, (3 * tp * capacity_m * (hidden // tp),), assumed_align=16
        ),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=hidden),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int32(capacity_m),
        current_cu_stream(),
    )


# --------------------------------------------------------------------------
# The provider piece.


class PoolerLayer(NamedTuple):
    """What the pooler reads from one stock MoE layer (views, no copies)."""

    # [P / tp, H] BF16: this rank's rows of pooler.weight.
    weight: torch.Tensor
    # [P] BF16 pre_expert_norm weight, and its eps and gain center.
    norm_weight: torch.Tensor
    eps: float
    gain_center: float
    # [1] FP32 NVFP4 input global scale of the experts; None for a BF16 edge
    # layer.
    global_scale: torch.Tensor | None


def pooler_layer(moe: Any, tp: int, rank: int) -> PoolerLayer:
    """The pooler of one stock MuseSparkMoE, checked against what the fused
    path assumes (the stock path's own conditions for its fused gather)."""
    weight = moe.pooler.weight
    if weight.shape != (POOLED, HIDDEN) or weight.dtype != torch.bfloat16:
        raise ValueError(f"pooler weight must be [{POOLED}, {HIDDEN}] BF16")
    norm = moe.pre_expert_norm
    if (
        norm is None
        or not norm.has_weight
        or norm.variance_size_override is not None
        or norm.applies_weight_before_norm
        or norm.weight.shape != (POOLED,)
        or norm.weight.dtype != torch.bfloat16
    ):
        raise ValueError("pre_expert_norm must apply a [P] BF16 weight after the norm")
    shard = POOLED // tp
    global_scale = moe.experts.nvfp4_input_global_scale
    if global_scale is not None:
        if global_scale.numel() != 1 or global_scale.dtype != torch.float32:
            raise ValueError("the NVFP4 input global scale must be one FP32 value")
        global_scale = global_scale.reshape(1)
    return PoolerLayer(
        weight=weight.narrow(0, rank * shard, shard),
        norm_weight=norm.weight,
        eps=float(norm.variance_epsilon),
        gain_center=float(norm.fused_gain_center),
        global_scale=global_scale,
    )


class Pooler:
    """The pooler for every MoE layer of one TP rank: the pooler GEMM, the
    publish of this rank's column block and the gather into the NVFP4 expert
    input (NVFP4 layers) or the normed BF16 rows (BF16 edge layers).

    Construction is collective over `group` (the mailbox rendezvous) and
    compiles every kernel; `warm_up` runs each form once and must run on
    every rank alike. `run` is graph-capturable and allocates nothing; its
    outputs are views of scratch the next call overwrites.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        tp: int,
        rank: int,
        device: torch.device,
        layers: Sequence[PoolerLayer],
        batches: Sequence[int] = (8,),
        capacity: int = MAILBOX_CAPACITY,
        enable_pdl: bool = True,
    ) -> None:
        if not 1 <= min(batches) <= max(batches) <= capacity:
            raise ValueError(f"batches must lie in [1, {capacity}]")
        self.group, self.tp, self.rank, self.capacity = group, tp, rank, capacity
        self.layers = tuple(layers)
        shard = POOLED // tp
        for layer in self.layers:
            if (
                layer.weight.shape != (shard, HIDDEN)
                or not layer.weight.is_contiguous()
            ):
                raise ValueError(
                    f"layer weights must be contiguous [{shard}, {HIDDEN}]"
                )

        self.mailbox = LamportMailbox(
            group, tp=tp, rank=rank, capacity=capacity, width=shard, device=device
        )
        self.batches = tuple(sorted(set(batches)))
        self._gemms = tuple(
            Unpooler(
                layer.weight,
                self.mailbox,
                enable_pdl=enable_pdl,
                shape=POOLER,
                rows=self.batches,
            )
            for layer in self.layers
        )
        nvfp4 = {
            (spec.eps, spec.gain_center)
            for spec in self.layers
            if spec.global_scale is not None
        }
        self._norm_quant = {
            constants: compile_norm_quant(
                hidden=POOLED,
                tp=tp,
                capacity_m=capacity,
                eps=constants[0],
                gain_center=constants[1],
                enable_pdl=enable_pdl,
            )
            for constants in nvfp4
        }
        # The BF16 edge layers' norms, by layer.
        self._edge_norms = {
            index: ExpertNorm(
                spec.norm_weight, spec.eps, spec.gain_center, enable_pdl=enable_pdl
            )
            for index, spec in enumerate(self.layers)
            if spec.global_scale is None
        }
        has_edge = bool(self._edge_norms)
        self._gather = (
            compile_gather(
                hidden=POOLED, tp=tp, capacity_m=capacity, enable_pdl=enable_pdl
            )
            if has_edge
            else None
        )

        def empty(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        self.shard = empty((capacity, shard), torch.bfloat16)
        self.packed = empty((capacity, POOLED // 2), torch.uint8)
        self.scales = empty((capacity, POOLED // _NVFP4_BLOCK), torch.uint8)
        self.rows = empty((capacity, POOLED), torch.bfloat16)
        self.normed = empty((capacity, POOLED), torch.bfloat16)
        # CuTe views bound once; the kernels read only the first m rows.
        self._mailbox_view = self.mailbox.view
        self._stage_view = self.mailbox.stage_view
        self._packed_view = to_cute_dynamic(
            self.packed.flatten(), 16, divisibility=POOLED // 2
        )
        self._scales_view = to_cute_dynamic(
            self.scales.flatten(), 4, divisibility=POOLED // _NVFP4_BLOCK
        )
        self._rows_view = rows_view(self.rows)
        self._normed_view = rows_view(self.normed)
        self._layer_views = tuple(
            (
                to_cute(layer.norm_weight, 16),
                None if layer.global_scale is None else to_cute(layer.global_scale, 4),
            )
            for layer in self.layers
        )

    def project(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        """The pooler GEMM: this rank's `[b, P / tp]` columns into scratch,
        multicast into the mailbox."""
        return self._gemms[layer].run(x, self.shard[: x.shape[0]])

    def gather(
        self, layer: int, b: int
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Consume the published blocks: the NVFP4 expert input `(a, a_sf)`
        for an NVFP4 layer, the normed BF16 rows for an edge layer."""
        spec = self.layers[layer]
        norm_weight, global_scale = self._layer_views[layer]
        stream = current_cu_stream()
        if global_scale is not None:
            self._norm_quant[(spec.eps, spec.gain_center)](
                self._mailbox_view,
                norm_weight,
                global_scale,
                self._packed_view,
                self._scales_view,
                self._stage_view,
                Int32(b),
                stream,
            )
            return self.packed[:b], self.scales[:b].view(torch.float8_e4m3fn)
        assert self._gather is not None
        self._gather(
            self._mailbox_view, self._rows_view, self._stage_view, Int32(b), stream
        )
        # pre_expert_norm, into scratch.
        self._edge_norms[layer].launch(self._rows_view, self._normed_view, b)
        return self.normed[:b]

    def run(
        self, layer: int, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """`x` BF16 [b, H] (the pre_ffn_norm output) -> the expert input."""
        self.project(layer, x)
        return self.gather(layer, x.shape[0])

    def warm_up(self) -> None:
        """One call of each form per batch on zero rows, before capture (every
        rank, same order)."""
        # No rank publishes before every rank has compiled: the gather's poll
        # traps after SPIN_TIMEOUT_NS, which a peer's compile could exceed.
        dist.barrier(group=self.group)
        forms: dict[tuple[float, float] | None, int] = {}
        for index, layer in enumerate(self.layers):
            key = None if layer.global_scale is None else (layer.eps, layer.gain_center)
            forms.setdefault(key, index)
        for b in self.batches:
            x = torch.zeros(b, HIDDEN, dtype=torch.bfloat16, device=self.shard.device)
            for index in forms.values():
                self.run(index, x)
        torch.accelerator.synchronize(self.shard.device)
