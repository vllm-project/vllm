# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark DSpark draft's context K/V chain around its K/V projection,
as two CuTe kernels.

``MuseSparkDSparkModel.precompute_and_store_context_kv`` runs the hidden norm,
one input norm per draft layer and a stack, then the batched K/V projection
(cuBLAS), then the K-norm, RoPE and one cache insert per layer. Here the chain
before the projection is one kernel (``ContextNorms``) and the chain after it
is another (``ContextKvStore``); the projection stays the stock ``torch.bmm``.
Both are bitwise equal to the stock chain on contiguous rows:

- the norms: one CTA of 256 threads per (row, layer), the row norm of
  ``common/row_norm.py`` (the stock single-tile ``fused_rmsnorm_fwd``), the
  hidden norm's output rounded to BF16 before the layer norm reads it, as the
  stock chain stores it;
- the store: one warp per (row, layer), lane j holding the GPT-J pair
  (2 j, 2 j + 1) of every head: the K-norm in the order of torch's CUDA mean
  over a 64-wide row (a torch that reduces in another order breaks the
  bitwise tests), the BF16 key, GPT-J RoPE as the CUDA rotary op computes
  it, and the cache insert.

Every FP32 op is explicit PTX with its rounding, so nothing contracts but
where the stock chain does.
"""

import functools
from collections.abc import Sequence
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr

from ..common.host import (
    assert_not_capturing,
    pdl_supported,
    rows_view,
    unread_operand,
)
from ..common.primitives import (
    VEC_BF16,
    add_rn,
    bf16x2_to_f32x2,
    current_cu_stream,
    fma_rn,
    load_bf16x8_as_f32,
    load_f32x8,
    load_global_bf16_as_f32,
    load_global_s64_address,
    load_global_u32_address,
    make_fake_dynamic_compact_tensor,
    mul_rn,
    pack_bf16x2_rn,
    regs,
    round_bf16,
    rsqrt_approx_ftz,
    store_bf16x8_rn,
    store_global_u32_address,
)
from ..common.row_norm import CONTEXT_NORMS, THREADS, contracts, rms_norm_row

# The norms' single tile, as the fused residual boundary's.
CONTEXT_HIDDEN_SIZE = 8192
# The layers (a cache and a slot pointer each) one launch of the store kernel takes.
MAX_LAYERS_PER_LAUNCH = 8
# The K-norm reproduces torch's mean over a 64-wide head (module docstring).
HEAD_DIM = 64

H = CONTEXT_HIDDEN_SIZE
# Columns between one norm thread's 8-column fragments.
CHUNK = THREADS * VEC_BF16
FRAGMENTS = H // CHUNK
WARP = 32
INV_HEAD_DIM = 1.0 / HEAD_DIM

assert FRAGMENTS * CHUNK == H
assert 2 * WARP == HEAD_DIM


class ContextNormConstants(NamedTuple):
    """Compile-time constants of the norms kernel."""

    # None: that norm has no weight; else whether its weight is FP32 (or BF16).
    hidden_weight_fp32: bool | None
    pre_weight_fp32: bool | None
    # The weight scales the input before the norm (else the norm's output).
    hidden_input_scale: bool
    pre_input_scale: bool
    enable_pdl: bool


class _ContextNormsKernel:
    """Grid (rows, layers), THREADS threads: one (row, layer) per CTA."""

    def __init__(self, constants: ContextNormConstants) -> None:
        self.constants = constants

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        hidden_weight: cute.Tensor,
        pre_weights: cute.Tensor,
        hidden_eps: Float32,
        pre_eps: Float32,
        hidden_gain_center: Float32,
        pre_gain_center: Float32,
        rows: Int32,
        layers: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            x,
            out,
            hidden_weight,
            pre_weights,
            hidden_eps,
            pre_eps,
            hidden_gain_center,
            pre_gain_center,
            rows,
        ).launch(
            grid=(rows, layers, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=self.constants.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        out: cute.Tensor,
        hidden_weight: cute.Tensor,
        pre_weights: cute.Tensor,
        hidden_eps: Float32,
        pre_eps: Float32,
        hidden_gain_center: Float32,
        pre_gain_center: Float32,
        rows: Int32,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        row, layer, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cu_utils.SmemAllocator()
        # One buffer of warp sums per norm.
        sums = smem.allocate_array(Float32, 16)
        columns = [Int64(tidx) * VEC_BF16 + k * CHUNK for k in range(FRAGMENTS)]

        # The weights load after the PDL wait, by choice (the row norm loads
        # its weight before it).
        if const_expr(constants.enable_pdl):
            cute.arch.griddepcontrol_wait()
        hidden_gamma = None
        if const_expr(constants.hidden_weight_fp32 is not None):
            hidden_gamma = []
            for k in cutlass.range_constexpr(FRAGMENTS):
                raw = regs(
                    load_f32x8(hidden_weight, columns[k])
                    if const_expr(constants.hidden_weight_fp32)
                    else load_bf16x8_as_f32(hidden_weight, columns[k])
                )
                hidden_gamma.extend([add_rn(v, hidden_gain_center) for v in raw])
        pre_gamma = None
        if const_expr(constants.pre_weight_fp32 is not None):
            pre_gamma = []
            first = Int64(layer) * H
            for k in cutlass.range_constexpr(FRAGMENTS):
                raw = regs(
                    load_f32x8(pre_weights, first + columns[k])
                    if const_expr(constants.pre_weight_fp32)
                    else load_bf16x8_as_f32(pre_weights, first + columns[k])
                )
                pre_gamma.extend([add_rn(v, pre_gain_center) for v in raw])

        values = []
        for k in cutlass.range_constexpr(FRAGMENTS):
            values.extend(regs(load_bf16x8_as_f32(x, Int64(row) * H + columns[k])))
        normed = rms_norm_row(
            values,
            hidden_gamma,
            hidden_eps,
            constants.hidden_input_scale,
            contracts(
                CONTEXT_NORMS,
                hidden_gamma is not None,
                constants.hidden_input_scale,
                bool(constants.hidden_weight_fp32),
            ),
            sums,
            warp,
            lane,
        )
        # The stock chain stores the hidden norm's output in BF16.
        values = [round_bf16(v) for v in normed]
        o = rms_norm_row(
            values,
            pre_gamma,
            pre_eps,
            constants.pre_input_scale,
            contracts(
                CONTEXT_NORMS,
                pre_gamma is not None,
                constants.pre_input_scale,
                bool(constants.pre_weight_fp32),
            ),
            sums + 8,
            warp,
            lane,
        )
        first = (Int64(layer) * Int64(rows) + Int64(row)) * H
        for k in cutlass.range_constexpr(FRAGMENTS):
            store_bf16x8_rn(
                out, first + columns[k], o[k * VEC_BF16 : (k + 1) * VEC_BF16]
            )


@functools.cache
def compile_context_norms(constants: ContextNormConstants):
    """cute.compile the norms kernel of one constant set (before CUDA graph
    capture); rows, layers, eps and gain centers are runtime arguments."""
    assert_not_capturing("the DSpark context norms")
    rows_bf16 = make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=H)
    rows_f32 = make_fake_dynamic_compact_tensor(Float32, alignment=32, divisibility=H)
    return cute.compile(
        _ContextNormsKernel(constants),
        rows_bf16,
        rows_bf16,
        rows_f32 if constants.hidden_weight_fp32 else rows_bf16,
        rows_f32 if constants.pre_weight_fp32 else rows_bf16,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Int32(1),
        Int32(1),
        current_cu_stream(),
    )


def _weight_fp32(weight: torch.Tensor | None) -> bool | None:
    if weight is None:
        return None
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("the norm weights must be BF16 or FP32")
    return weight.dtype == torch.float32


def context_norm_constants(
    hidden_weight: torch.Tensor | None,
    pre_weights: torch.Tensor | None,
    *,
    hidden_input_scale: bool = False,
    pre_input_scale: bool = False,
) -> ContextNormConstants:
    """The constants of the norms kernel, or ValueError where it does not
    take the weights."""
    return ContextNormConstants(
        hidden_weight_fp32=_weight_fp32(hidden_weight),
        pre_weight_fp32=_weight_fp32(pre_weights),
        hidden_input_scale=hidden_weight is not None and hidden_input_scale,
        pre_input_scale=pre_weights is not None and pre_input_scale,
        enable_pdl=pdl_supported(),
    )


def _rows_view(tensor: torch.Tensor):
    """``rows_view`` of BF16 or FP32 rows of ``H`` (32-byte aligned in FP32)."""
    return rows_view(tensor, H, 32 if tensor.dtype == torch.float32 else 16)


class ContextNorms:
    """``stack([pre_norm_l(hidden_norm(x)) for l])`` as one kernel, bound to
    its norms: the hidden norm's weight ``[D]`` and the layers' input-norm
    weights ``[L, D]`` (BF16 or FP32; None for weightless norms), eps, gain
    centers and whether each weight scales the input (else the norm's
    output); ``constants`` their ``context_norm_constants`` where the caller
    has them. Compiles at construction (before CUDA graph capture); a call
    is graph-capturable."""

    def __init__(
        self,
        hidden_weight: torch.Tensor | None,
        pre_weights: torch.Tensor | None,
        num_layers: int,
        hidden_eps: float,
        pre_eps: float,
        *,
        device: torch.device,
        hidden_gain_center: float = 0.0,
        pre_gain_center: float = 0.0,
        hidden_input_scale: bool = False,
        pre_input_scale: bool = False,
        constants: ContextNormConstants | None = None,
    ) -> None:
        if hidden_weight is not None:
            assert hidden_weight.shape == (H,) and hidden_weight.is_contiguous()
        if pre_weights is not None:
            assert pre_weights.shape == (num_layers, H) and pre_weights.is_contiguous()
        self.num_layers = num_layers
        if constants is None:
            constants = context_norm_constants(
                hidden_weight,
                pre_weights,
                hidden_input_scale=hidden_input_scale,
                pre_input_scale=pre_input_scale,
            )
        self.kernel = compile_context_norms(constants)
        # An absent weight takes an operand of its type the kernel never reads.
        unread = unread_operand(device, torch.bfloat16)[:H]
        # The tensors stay referenced for as long as their views.
        self._weights = (hidden_weight, pre_weights)
        self._operands = (
            _rows_view(hidden_weight if hidden_weight is not None else unread),
            _rows_view(pre_weights if pre_weights is not None else unread),
            Float32(hidden_eps),
            Float32(pre_eps),
            Float32(hidden_gain_center),
            Float32(pre_gain_center),
        )
        self._layers = Int32(num_layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The norms of ``x``, contiguous BF16 ``[T, D]`` with T >= 1 and D
        8192 (the norms' single tile; the caller checks it): BF16
        ``[L, T, D]``."""
        rows = x.shape[0]
        out = torch.empty(
            (self.num_layers, rows, H), dtype=torch.bfloat16, device=x.device
        )
        self.kernel(
            _rows_view(x),
            _rows_view(out),
            *self._operands,
            Int32(rows),
            self._layers,
            current_cu_stream(),
        )
        return out


def _sum_squares_64(k0: Float32, k1: Float32) -> Float32:
    """``x.pow(2).sum(-1)`` of one 64-wide head whose pair (2 j, 2 j + 1) lane
    j holds, in torch's CUDA mean order: the squares of elements i and i + 32
    added, then the 32 sums folded by halves (a lane butterfly, then even +
    odd). Every lane gets it (traced inline)."""
    s0, s1 = mul_rn(k0, k0), mul_rn(k1, k1)
    for offset in (16, 8, 4, 2, 1):
        s0 = add_rn(s0, cute.arch.shuffle_sync_bfly(s0, offset=offset))
        s1 = add_rn(s1, cute.arch.shuffle_sync_bfly(s1, offset=offset))
    return add_rn(s0, s1)


def _rope(x: Float32, y: Float32, c: Float32, s: Float32) -> tuple:
    """The CUDA rotary op's GPT-J pair (x * c - y * s, y * c + x * s) in its
    contraction (traced inline)."""
    return fma_rn(x, c, -mul_rn(y, s)), fma_rn(y, c, mul_rn(x, s))


class _ContextKvStoreKernel:
    """Grid (rows, layers), one warp: one (row, layer) per CTA."""

    def __init__(self, heads: int, rope: bool, enable_pdl: bool) -> None:
        self.heads, self.rope, self.enable_pdl = heads, rope, enable_pdl

    @cute.jit
    def __call__(
        self,
        k: Int64,
        v: Int64,
        positions: Int64,
        cos_sin: Int64,
        c0: Int64,
        c1: Int64,
        c2: Int64,
        c3: Int64,
        c4: Int64,
        c5: Int64,
        c6: Int64,
        c7: Int64,
        s0: Int64,
        s1: Int64,
        s2: Int64,
        s3: Int64,
        s4: Int64,
        s5: Int64,
        s6: Int64,
        s7: Int64,
        k_layer_stride: Int64,
        k_row_stride: Int64,
        v_layer_stride: Int64,
        v_row_stride: Int64,
        block_stride: Int64,
        page_stride: Int64,
        head_stride: Int64,
        block_size: Int64,
        eps: Float32,
        rows: Int32,
        layers: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            k,
            v,
            positions,
            cos_sin,
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            k_layer_stride,
            k_row_stride,
            v_layer_stride,
            v_row_stride,
            block_stride,
            page_stride,
            head_stride,
            block_size,
            eps,
        ).launch(
            grid=(rows, layers, 1),
            block=(WARP, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        k: Int64,
        v: Int64,
        positions: Int64,
        cos_sin: Int64,
        c0: Int64,
        c1: Int64,
        c2: Int64,
        c3: Int64,
        c4: Int64,
        c5: Int64,
        c6: Int64,
        c7: Int64,
        s0: Int64,
        s1: Int64,
        s2: Int64,
        s3: Int64,
        s4: Int64,
        s5: Int64,
        s6: Int64,
        s7: Int64,
        k_layer_stride: Int64,
        k_row_stride: Int64,
        v_layer_stride: Int64,
        v_row_stride: Int64,
        block_stride: Int64,
        page_stride: Int64,
        head_stride: Int64,
        block_size: Int64,
        eps: Float32,
    ) -> None:
        lane = cute.arch.lane_idx()
        t, layer, _ = cute.arch.block_idx()
        if const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        # Layer l's cache and slots (byte addresses).
        cache = c0
        slots = s0
        if layer == Int32(1):
            cache = c1
            slots = s1
        if layer == Int32(2):
            cache = c2
            slots = s2
        if layer == Int32(3):
            cache = c3
            slots = s3
        if layer == Int32(4):
            cache = c4
            slots = s4
        if layer == Int32(5):
            cache = c5
            slots = s5
        if layer == Int32(6):
            cache = c6
            slots = s6
        if layer == Int32(7):
            cache = c7
            slots = s7
        row = Int64(t)
        pair = Int64(lane) * 2
        k_row = k + (Int64(layer) * k_layer_stride + row * k_row_stride + pair) * 2
        v_row = v + (Int64(layer) * v_layer_stride + row * v_row_stride + pair) * 2
        # Every load first (none waits for another's use), then the math, then
        # one guarded block of stores.
        slot = load_global_s64_address(slots + row * 8)
        k_words = [
            load_global_u32_address(k_row + h * HEAD_DIM * 2) for h in range(self.heads)
        ]
        v_words = [
            load_global_u32_address(v_row + h * HEAD_DIM * 2) for h in range(self.heads)
        ]
        c = Float32(0.0)
        s = Float32(0.0)
        if const_expr(self.rope):
            table = cos_sin + load_global_s64_address(positions + row * 8) * (
                HEAD_DIM * 2
            )
            c = load_global_bf16_as_f32(table + Int64(lane) * 2)
            s = load_global_bf16_as_f32(table + (Int64(lane) + HEAD_DIM // 2) * 2)
        k_out = []
        for h in cutlass.range_constexpr(self.heads):
            k0, k1 = bf16x2_to_f32x2(k_words[h])
            # torch: x.pow(2).mean(-1) (the mean's factor 1/64 is exact), + eps,
            # rsqrt, x * rsqrt, then the bf16 copy.
            r = rsqrt_approx_ftz(
                fma_rn(_sum_squares_64(k0, k1), Float32(INV_HEAD_DIM), eps)
            )
            n0, n1 = round_bf16(mul_rn(k0, r)), round_bf16(mul_rn(k1, r))
            if const_expr(self.rope):
                n0, n1 = _rope(n0, n1, c, s)
            k_out.append(pack_bf16x2_rn(n0, n1))
        if slot >= Int64(0):
            # Slots of a cache fit 32 bits (checked on the host).
            page = Uint32(slot)
            block = page // Uint32(block_size)
            dst = (
                cache
                + (
                    Int64(block) * block_stride
                    + Int64(page - block * Uint32(block_size)) * page_stride
                    + pair
                )
                * 2
            )
            for h in cutlass.range_constexpr(self.heads):
                head = dst + Int64(h) * head_stride * 2
                store_global_u32_address(head, k_out[h])
                # V follows K in the cache's last dim.
                store_global_u32_address(head + HEAD_DIM * 2, v_words[h])


@functools.cache
def compile_context_kv_store(heads: int, rope: bool, enable_pdl: bool):
    """cute.compile the store kernel of ``heads`` KV heads (before CUDA graph
    capture); rows, layers, strides, the block size and eps are runtime
    arguments."""
    assert_not_capturing("the DSpark context K/V store")
    zero = Int64(0)
    return cute.compile(
        _ContextKvStoreKernel(heads, rope, enable_pdl),
        *([zero] * 28),
        Float32(0.0),
        Int32(1),
        Int32(1),
        current_cu_stream(),
    )


class ContextKvStore:
    """The K-norm with ``eps``, GPT-J RoPE (with ``rope``) and cache insert of
    ``num_heads`` KV heads per layer, as the stock chain after the K/V
    projection. Compiles at construction (before CUDA graph capture); a call
    is graph-capturable."""

    def __init__(self, num_heads: int, rope: bool, eps: float) -> None:
        self.rope = rope
        self.kernel = compile_context_kv_store(num_heads, rope, pdl_supported())
        self._eps = Float32(eps)

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        slots: Sequence[torch.Tensor],
        cos_sin_cache: torch.Tensor | None,
        caches: Sequence[torch.Tensor],
    ) -> None:
        """Insert ``L = len(caches)`` layers' context K/V rows into their
        caches, the operands as ``ContextKV.store`` checks them.

        ``k`` and ``v`` are bf16 ``[L, T, H, 64]`` views of the K/V
        projection's output, unit stride over the last dim, 64 over the one
        before, even strides and 4-byte aligned. ``slots`` holds each
        layer's contiguous int64 slot mapping, all of one length; the first
        ``len(slots[0])`` rows are inserted, as ``reshape_and_cache_flash``
        does; a negative slot skips its row. ``positions`` is contiguous
        int64. ``k`` gets the weightless K-norm, then GPT-J RoPE at
        ``positions`` from ``cos_sin_cache`` (contiguous bf16
        ``[max_position, 64]``, cos then sin; None without ``rope``).
        ``caches`` holds each layer's key-cache view ``[num_blocks,
        block_size, H, 64]`` of the attention backend's bf16 cache
        ``[num_blocks, H, block_size, 128]`` (K then V in the last dim), of
        one shape and stride, even strides, 4-byte aligned and fewer than
        2^31 slots. Up to ``MAX_LAYERS_PER_LAUNCH`` layers take one launch.
        """
        assert (cos_sin_cache is not None) == self.rope
        rows = slots[0].shape[0]
        if rows == 0:
            return
        ref = caches[0]
        table = cos_sin_cache.data_ptr() if cos_sin_cache is not None else 0
        for start in range(0, len(caches), MAX_LAYERS_PER_LAUNCH):
            chunk = caches[start : start + MAX_LAYERS_PER_LAUNCH]
            chunk_slots = slots[start : start + MAX_LAYERS_PER_LAUNCH]
            pad = MAX_LAYERS_PER_LAUNCH - len(chunk)
            self.kernel(
                Int64(k[start].data_ptr()),
                Int64(v[start].data_ptr()),
                Int64(positions.data_ptr()),
                Int64(table),
                *(Int64(c.data_ptr()) for c in chunk),
                *([Int64(chunk[0].data_ptr())] * pad),
                *(Int64(s.data_ptr()) for s in chunk_slots),
                *([Int64(chunk_slots[0].data_ptr())] * pad),
                Int64(k.stride(0)),
                Int64(k.stride(1)),
                Int64(v.stride(0)),
                Int64(v.stride(1)),
                Int64(ref.stride(0)),
                Int64(ref.stride(1)),
                Int64(ref.stride(2)),
                Int64(ref.shape[1]),
                self._eps,
                Int32(rows),
                Int32(len(chunk)),
                current_cu_stream(),
            )
