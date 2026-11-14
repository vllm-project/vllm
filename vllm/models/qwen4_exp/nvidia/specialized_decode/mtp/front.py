# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two input norms of the MTP draft model in one CuTe DSL kernel.

Every draft pass of Qwen4Exp's MTP model (`../../mtp.py`) starts with two
zero-centred RMSNorms (`GemmaRMSNorm`) of one row each per token:

* `pre_fc_norm_embedding` of the new token's embedding, `[rows, 2560]`;
* `pre_fc_norm_hidden` of the four hyper-connection streams the target (or
  the previous draft pass) produced, one norm over all `[rows, 10240]`
  values.

`GemmaRMSNorm` passes an FP32 `1 + w` to `rms_norm`, which vLLM's fused
kernel does not take (it needs the input's dtype), so each norm runs as about
ten PyTorch kernels: 17-20 us per norm at decode sizes. `front_norm`
computes both in one kernel with stock's arithmetic, step by step in FP32 of
the BF16 inputs::

    mean = sum(x * x) * (1 / n)  # PyTorch's MeanOps: sum, then factor
    r = rsqrt(mean + eps)
    y = bf16((x * r) * (w + 1))  # the weight widened, then + 1

with `mul.rn`/`add.rn` (no contraction) and `rsqrt.approx` (`_ptx`).
Only the order of the sum of squares differs from stock's reduction kernel, so
an output can differ from stock's in its last bit.

Work split. One CTA of 320 threads (ten warps) per token row: thread `t`
owns the eight embedding values `[8 t, 8 t + 8)` and the hidden values
`[8 c, 8 c + 8)` for `c = t + 320 j`, `j = 0..3`. It sums its squares in
that order, the warps reduce with butterfly shuffles (every lane ends with the
same bits) and every thread adds the ten warp sums in warp order, so a row's
result does not depend on the other rows.

Ordering. The norm weights are immutable after loading, so each thread loads
its share before `griddepcontrol.wait` (the kernel launches with
programmatic dependent launch and may start while the previous kernel runs).
The activations are read after the wait, which returns once every prerequisite
grid has completed and its writes are visible. The kernel does not trigger its
dependents early: the next kernel launches when this one completes.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from .. import HC_HIDDEN, HC_STREAMS
from ..hc._ptx import load_shared_f32, store_shared_f32
from ._ptx import (
    add_rn,
    load_bf16x8,
    load_weight_bf16x8,
    mul_rn,
    rsqrt_approx,
    store_bf16x8,
)

EMBED_WIDTH = HC_HIDDEN
HIDDEN_WIDTH = HC_STREAMS * HC_HIDDEN
THREADS = 320
WARPS = THREADS // 32
VECTOR = 8  # BF16 values of one 16-byte access
HIDDEN_CHUNKS = HIDDEN_WIDTH // (VECTOR * THREADS)  # 4
assert EMBED_WIDTH == VECTOR * THREADS
assert HIDDEN_CHUNKS * VECTOR * THREADS == HIDDEN_WIDTH
# The largest draft pass the specialized decode captures: 64 requests of four
# rows at TP1.
MAX_ROWS = 256


def _warp_sum(value: Float32) -> Float32:
    """The sum over a warp's lanes, the same bits in every lane."""
    for offset in (16, 8, 4, 2, 1):
        value = add_rn(value, cute.arch.shuffle_sync_bfly(value, offset))
    return value


def _norm(values: list[Float32], weights: list[Float32], r: Float32) -> list[Float32]:
    """`(x * r) * (w + 1)` per value: stock's order of operations."""
    return [
        mul_rn(mul_rn(x, r), add_rn(w, Float32(1.0))) for x, w in zip(values, weights)
    ]


class FrontNorm:
    """Both input norms of an MTP draft pass (module docstring)."""

    @cute.jit
    def __call__(
        self,
        embedded: Int64,
        embedded_stride: Int32,
        hidden: Int64,
        hidden_stride: Int32,
        embed_weight: Int64,
        hidden_weight: Int64,
        out_embedded: Int64,
        out_hidden: Int64,
        rows: Int32,
        embed_eps: Float32,
        hidden_eps: Float32,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per row on `stream`: `embedded [rows, 2560]` and
        `hidden [rows, 10240]` BF16 with row strides in elements, the BF16
        weights, and contiguous BF16 outputs of the same shapes."""
        self.kernel(
            embedded,
            embedded_stride,
            hidden,
            hidden_stride,
            embed_weight,
            hidden_weight,
            out_embedded,
            out_hidden,
            embed_eps,
            hidden_eps,
        ).launch(
            grid=(rows, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        embedded: Int64,
        embedded_stride: Int32,
        hidden: Int64,
        hidden_stride: Int32,
        embed_weight: Int64,
        hidden_weight: Int64,
        out_embedded: Int64,
        out_hidden: Int64,
        embed_eps: Float32,
        hidden_eps: Float32,
    ):
        """One CTA: both norms of one token row."""
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        chunk_bytes = VECTOR * 2

        # The weights, before the grid-dependency wait (immutable).
        w_embed = list(load_weight_bf16x8(embed_weight + Int64(tid) * chunk_bytes))
        w_hidden = []
        for j in cutlass.range_constexpr(HIDDEN_CHUNKS):
            chunk = Int64(tid + THREADS * j)
            w_hidden.append(
                list(load_weight_bf16x8(hidden_weight + chunk * chunk_bytes))
            )

        allocator = cutlass.utils.SmemAllocator()
        # [norm][warp]: the warps' sums of squares (embedding, then hidden).
        sums = allocator.allocate_array(Float32, 2 * WARPS, byte_alignment=16).toint()

        cute.arch.griddepcontrol_wait()

        embed_row = embedded + Int64(row) * Int64(embedded_stride) * 2
        hidden_row = hidden + Int64(row) * Int64(hidden_stride) * 2
        x_embed = list(load_bf16x8(embed_row + Int64(tid) * chunk_bytes))
        x_hidden = []
        for j in cutlass.range_constexpr(HIDDEN_CHUNKS):
            chunk = Int64(tid + THREADS * j)
            x_hidden.append(list(load_bf16x8(hidden_row + chunk * chunk_bytes)))

        # Sums of squares: this thread's values in order, then its warp.
        square_embed = mul_rn(x_embed[0], x_embed[0])
        for i in cutlass.range_constexpr(1, VECTOR):
            square_embed = add_rn(square_embed, mul_rn(x_embed[i], x_embed[i]))
        square_hidden = mul_rn(x_hidden[0][0], x_hidden[0][0])
        for j in cutlass.range_constexpr(HIDDEN_CHUNKS):
            for i in cutlass.range_constexpr(VECTOR):
                if j + i > 0:
                    square_hidden = add_rn(
                        square_hidden, mul_rn(x_hidden[j][i], x_hidden[j][i])
                    )
        square_embed = _warp_sum(square_embed)
        square_hidden = _warp_sum(square_hidden)
        if lane == 0:
            store_shared_f32(sums + warp * 4, square_embed)
            store_shared_f32(sums + (WARPS + warp) * 4, square_hidden)
        cute.arch.sync_threads()

        # Every thread adds the warp sums in warp order.
        total_embed = load_shared_f32(sums)
        total_hidden = load_shared_f32(sums + WARPS * 4)
        for w in cutlass.range_constexpr(1, WARPS):
            total_embed = add_rn(total_embed, load_shared_f32(sums + w * 4))
            total_hidden = add_rn(total_hidden, load_shared_f32(sums + (WARPS + w) * 4))
        r_embed = rsqrt_approx(
            add_rn(mul_rn(total_embed, Float32(1.0 / EMBED_WIDTH)), embed_eps)
        )
        r_hidden = rsqrt_approx(
            add_rn(mul_rn(total_hidden, Float32(1.0 / HIDDEN_WIDTH)), hidden_eps)
        )

        out_embed_row = out_embedded + Int64(row) * (EMBED_WIDTH * 2)
        out_hidden_row = out_hidden + Int64(row) * (HIDDEN_WIDTH * 2)
        store_bf16x8(
            out_embed_row + Int64(tid) * chunk_bytes,
            tuple(_norm(x_embed, w_embed, r_embed)),
        )
        for j in cutlass.range_constexpr(HIDDEN_CHUNKS):
            chunk = Int64(tid + THREADS * j)
            store_bf16x8(
                out_hidden_row + chunk * chunk_bytes,
                tuple(_norm(x_hidden[j], w_hidden[j], r_hidden)),
            )


@cache
def _compiled(device_index: int):
    """`FrontNorm` compiled for a device (the row count is a launch
    argument)."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            FrontNorm(),
            Int64(0),
            Int32(EMBED_WIDTH),
            Int64(0),
            Int32(HIDDEN_WIDTH),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(1),
            Float32(1e-6),
            Float32(1e-6),
            make_fake_stream(),
        )


def _rows_ok(x: torch.Tensor, width: int) -> bool:
    """Whether `x` is a BF16 `[rows, width]` CUDA tensor whose rows are
    contiguous and 16-byte aligned."""
    return (
        x.dim() == 2
        and x.shape[1] == width
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.stride(1) == 1
        and x.stride(0) % VECTOR == 0
        and x.data_ptr() % 16 == 0
    )


def front_norm_supports(embedded: torch.Tensor, hidden: torch.Tensor) -> bool:
    """Whether `front_norm` takes these activations: BF16
    `[rows, 2560]` and `[rows, 10240]` with the same 1 to `MAX_ROWS`
    rows, contiguous 16-byte aligned rows."""
    rows = embedded.shape[0] if embedded.dim() == 2 else 0
    return (
        1 <= rows <= MAX_ROWS
        and hidden.dim() == 2
        and hidden.shape[0] == rows
        and _rows_ok(embedded, EMBED_WIDTH)
        and _rows_ok(hidden, HIDDEN_WIDTH)
        and embedded.device == hidden.device
    )


def front_norm(
    embedded: torch.Tensor,
    hidden: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    embed_eps: float,
    hidden_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both input norms of an MTP draft pass.

    Args:
        embedded: The new tokens' embeddings, BF16 `[rows, 2560]`.
        hidden: The previous pass's hyper-connection streams, BF16
            `[rows, 10240]`.
        embed_weight: `pre_fc_norm_embedding.weight`, BF16 `[2560]`
            (`w`; the kernel applies `1 + w`).
        hidden_weight: `pre_fc_norm_hidden.weight`, BF16 `[10240]`.
        embed_eps: The embedding norm's epsilon.
        hidden_eps: The hidden norm's epsilon.

    Returns:
        `GemmaRMSNorm` of each input with its weight, contiguous BF16
        `[rows, 2560]` and `[rows, 10240]`.

    """
    assert front_norm_supports(embedded, hidden), "unsupported activations"
    for weight, width in ((embed_weight, EMBED_WIDTH), (hidden_weight, HIDDEN_WIDTH)):
        assert weight.dtype == torch.bfloat16 and weight.shape == (width,)
        assert weight.is_contiguous() and weight.data_ptr() % 16 == 0
    rows = embedded.shape[0]
    out_embedded = torch.empty(
        rows, EMBED_WIDTH, dtype=torch.bfloat16, device=embedded.device
    )
    out_hidden = torch.empty(
        rows, HIDDEN_WIDTH, dtype=torch.bfloat16, device=embedded.device
    )
    _compiled(embedded.device.index)(
        Int64(embedded.data_ptr()),
        Int32(embedded.stride(0)),
        Int64(hidden.data_ptr()),
        Int32(hidden.stride(0)),
        Int64(embed_weight.data_ptr()),
        Int64(hidden_weight.data_ptr()),
        Int64(out_embedded.data_ptr()),
        Int64(out_hidden.data_ptr()),
        Int32(rows),
        Float32(embed_eps),
        Float32(hidden_eps),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out_embedded, out_hidden
