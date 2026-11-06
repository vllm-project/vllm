# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM commit of an FP32 KDA state: one CTA per (head, request, layer).

Folds the n accepted tokens of a verify window into the checkpoint as the stock
FP32 commit (``_commit_kda_state_kernel``) does, in its association:

    S_n = S_0 * D_n + sum_{t<n} (u_t * kn_t) * P_t,   P_t = prod_{t<s<n} d_s

(``u_t`` the verify's corrections, ``kn_t`` the normalized recorded keys, ``d_t`` the
per-key decays, ``D_n = P_{-1}``), and with prefix caching (``align``) also the
boundary state S_m of the window's first m tokens.

Thread c < 128 rebuilds key channel c's decays, normalized keys and suffix products
into shared memory; then all 8 warps fold the head's [128, 128] state from global
memory to global memory, thread 2 i + h taking keys 8 c + 4 h + [0, 4) of row i,
so each byte of the state is read once and written once (the stock FP32 kernel
splits a head over four CTAs that each rebuild the key operands)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .helpers import triton_exp, triton_sigmoid
from .host import architecture, dynamic_fake
from .primitives import (
    alloc,
    fma,
    launch,
    load_f32x4,
    load_words,
    prefetch_bytes,
    store_f32x4,
    warp_sum,
)

K = 128
THREADS = 256
CHUNKS = 16
PAGE_BYTES = K * K * 4


class KdaCommitF32:
    """Args as the stock CuTe commit's ``KdaCommit`` (base addresses, strides,
    a_log [L, H], dt_bias [L, H, K], source / commit length / destination /
    boundary slot / boundary length per request, lower bound, null slot, stream);
    the state is FP32 [H, V, K] per slot."""

    def __init__(
        self,
        heads: int,
        window: int,
        align: bool = False,
        ahead: int = 0,
        indexed: bool = False,
    ):
        assert 1 <= window <= 8
        self.heads, self.window, self.align = heads, window, align
        # ``indexed``: a trailing record_rows [B] operand gives each request's record
        # slot (deferred RecoverSSM keeps the records per request slot, not step row).
        self.indexed = indexed
        # Resident CTAs: L2 prefetch of the next wave's pages (0: off; measured
        # 372 -> 397 us at TP1 R128, so the commit runs without it).
        self.ahead = ahead

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads, args[8].shape[0], args[0].shape[0])
        launch(self.kernel, args, grid, THREADS)

    @cute.kernel
    def kernel(self, args):
        state_addrs, corr_addrs, kg_addrs = args[:3]
        state_strides, corr_strides, kg_strides, a_log, dt_bias = args[3:8]
        sources, commit_lens, destinations, boundaries, boundary_lens = args[8:13]
        lower_bound, null_slot = args[13:15]
        Q: cutlass.Constexpr = self.window
        ALIGN: cutlass.Constexpr = self.align
        tid, _, _ = cute.arch.thread_idx()
        head, request, layer = cute.arch.block_idx()
        warp, lane = tid // 32, tid % 32
        smem = cutlass.utils.SmemAllocator()
        keys = alloc(smem, Float32, (Q, K))  # kn_t
        suffix = alloc(smem, Float32, (Q + 1, K))  # P_t; row Q: D_n
        suffix_b = alloc(smem, Float32, (Q + 1, K))  # the same for S_m (align)
        norms = alloc(smem, Float32, (Q, 4))

        cute.arch.griddepcontrol_wait()
        n = commit_lens[request]
        source = Int64(sources[request])
        destination = Int64(destinations[request])
        live = (n > 0) & (source > Int64(null_slot)) & (destination > Int64(null_slot))
        m = Int32(0)
        boundary = Int64(null_slot)
        if cutlass.const_expr(ALIGN):
            m = boundary_lens[request]
            boundary = Int64(boundaries[request])
        boundary_live = (m > 0) & (boundary > Int64(null_slot))
        if cutlass.const_expr(self.ahead > 0):  # noqa: SIM102
            if tid == 0:
                _, requests, layers = cute.arch.grid_dim()
                item = head + self.heads * (request + requests * layer) + self.ahead
                if item < self.heads * requests * layers:
                    h = item % self.heads
                    r = (item // self.heads) % requests
                    lay = item // (self.heads * requests)
                    slot = Int64(sources[r])
                    if slot > Int64(null_slot):
                        address = Int64(state_addrs[lay])
                        address += slot * Int64(state_strides[lay]) * 4
                        address += Int64(h) * PAGE_BYTES
                        pointer = cute.make_ptr(
                            Float32, address, cute.AddressSpace.gmem, assumed_align=16
                        )
                        prefetch_bytes(pointer, PAGE_BYTES)
        if live:
            state_base = Int64(state_addrs[layer])
            stride = Int64(state_strides[layer]) * 4  # bytes per slot
            page = Int64(head) * PAGE_BYTES
            row, half = tid // 2, tid % 2

            def state_page(slot):
                address = state_base + slot * stride + page
                pointer = cute.make_ptr(
                    Float32, address, cute.AddressSpace.gmem, assumed_align=16
                )
                return cute.make_tensor(
                    pointer, cute.make_layout((K, K), stride=(K, 1))
                )

            src = state_page(source)
            corr_ptr = cute.make_ptr(
                Float32, corr_addrs[layer], cute.AddressSpace.gmem, assumed_align=16
            )
            kg_ptr = cute.make_ptr(
                cutlass.BFloat16,
                kg_addrs[layer],
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            record = Int64(request)
            if cutlass.const_expr(self.indexed):
                record = Int64(args[15][request])
            corr_slot = corr_ptr + record * Int64(corr_strides[layer])
            kg_slot = kg_ptr + record * Int64(kg_strides[layer])
            corr = cute.make_tensor(
                corr_slot + head * Q * K, cute.make_layout((Q, K), stride=(K, 1))
            )
            kg = cute.make_tensor(
                kg_slot + head * Q * 2 * K,
                cute.make_layout((Q, 2 * K), stride=(2 * K, 1)),
            )
            # This thread's row of corrections u_t (0 past n).
            u = cute.make_rmem_tensor(Q, Float32)
            for t in cutlass.range_constexpr(Q):
                u[t] = Float32(0)
                if t < n:
                    u[t] = corr[t, row]

            channel = tid % K
            kv = cute.make_rmem_tensor(Q, Float32)
            decays = cute.make_rmem_tensor(Q, Float32)
            if tid < K:
                bias = Float32(dt_bias[layer, head, channel])
                a = triton_exp(Float32(a_log[layer, head]))
                for t in cutlass.range_constexpr(Q):
                    kv[t], decays[t] = Float32(0), Float32(1)
                    if t < n:
                        kv[t] = Float32(kg[t, channel])
                        raw_g = Float32(kg[t, K + channel])
                        gate = lower_bound * triton_sigmoid(a * (raw_g + bias))
                        decays[t] = triton_exp(gate)
                for t in cutlass.range_constexpr(Q):
                    total = warp_sum(kv[t] * kv[t])
                    if lane == 0:
                        norms[t, warp] = total
            cute.arch.barrier()
            if tid < K:
                fd, fd_b = Float32(1), Float32(1)
                for r in cutlass.range_constexpr(Q):
                    t = Q - 1 - r  # reverse: the stock accumulation order
                    total = (norms[t, 0] + norms[t, 1]) + (norms[t, 2] + norms[t, 3])
                    keys[t, channel] = kv[t] * cute.math.rsqrt(total + Float32(1e-6))
                    suffix[t, channel] = fd
                    fd = fd * decays[t]  # 1 past n
                    if cutlass.const_expr(ALIGN):  # tokens past m add nothing
                        suffix_b[t, channel] = Float32(0)
                        if t < m:
                            suffix_b[t, channel] = fd_b
                            fd_b = fd_b * decays[t]
                suffix[Q, channel] = fd
                if cutlass.const_expr(ALIGN):
                    suffix_b[Q, channel] = fd_b
            cute.arch.barrier()

            dst = state_page(destination)
            if cutlass.const_expr(ALIGN):
                dst_b = state_page(boundary)
            load_shared = lambda tensor, t, column: load_words(
                tensor.iterator + t * K + column,
                count=4,
                shared=True,
                mutable=True,
                result_type=Float32,
                word_type="f32",
            )
            for c in cutlass.range_constexpr(CHUNKS):
                column = 8 * c + 4 * half
                initial = load_f32x4(src.iterator + row * K + column)
                final = [Float32(0) for _ in range(4)]
                final_b = [Float32(0) for _ in range(4)]
                for r in cutlass.range_constexpr(Q):
                    t = Q - 1 - r
                    kn = load_shared(keys, t, column)
                    p = load_shared(suffix, t, column)
                    if cutlass.const_expr(ALIGN):
                        p_b = load_shared(suffix_b, t, column)
                    for e in cutlass.range_constexpr(4):
                        update = u[t] * kn[e]  # 0 past n
                        final[e] = fma(update, p[e], final[e])
                        if cutlass.const_expr(ALIGN):
                            final_b[e] = fma(update, p_b[e], final_b[e])
                if cutlass.const_expr(ALIGN):  # first: the final state may share it
                    decay_m = load_shared(suffix_b, Q, column)
                    values_b = [
                        fma(initial[e], decay_m[e], final_b[e]) for e in range(4)
                    ]
                    store_f32x4(
                        dst_b.iterator + row * K + column, values_b, boundary_live
                    )
                decay_n = load_shared(suffix, Q, column)
                values = [fma(initial[e], decay_n[e], final[e]) for e in range(4)]
                store_f32x4(dst.iterator + row * K + column, values)
        cute.arch.griddepcontrol_launch_dependents()


@cache
def _compiled(
    heads: int, window: int, arch: str, param_dtype, align: bool, indexed=False
):
    from cutlass.cute.runtime import make_fake_stream

    i32, i64 = torch.int32, torch.int64
    fakes = (
        *(dynamic_fake(1, i64) for _ in range(6)),  # base addresses, strides [L]
        dynamic_fake(2, param_dtype),  # A_log [L, H]
        dynamic_fake(3, param_dtype),  # dt_bias [L, H, K]
        *(dynamic_fake(1, i32) for _ in range(5)),  # per-request plan [B]
    )
    rows = (dynamic_fake(1, i32),) if indexed else ()
    return cute.compile(
        KdaCommitF32(heads, window, align, indexed=indexed),
        *fakes,
        cutlass.Float32(0),
        cutlass.Int32(0),
        *rows,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch={arch} --enable-tvm-ffi",
    )


# (head, request, layer) items of a launch from which KdaCommitF32 beats the
# Triton commit (GB300, q5, us per layer Triton / ours over 69 layers: TP8 R1
# 0.89 / 0.66, R2 1.52 / 1.13; TP1 R1 4.99 / 3.34; over 8 layers: TP8 R1 1.63 / 1.88,
# R4 3.50 / 2.92, R16 10.1 / 7.6; over 2: TP1 R128 572 / 360). A Kimi K3.1 rank
# commits its 69 KDA layers in one launch, so this holds at every batch.
COMMIT_ITEMS = 384


def use_commit_f32(heads: int, batch: int, layers: int = 1) -> bool:
    return heads * batch * layers >= COMMIT_ITEMS


def supported(state: torch.Tensor, lower_bound: float | None) -> bool:
    """Whether ``commit_f32`` folds this checkpoint cache (FP32 [slot, H, 128, 128]
    pages, contiguous per slot, on SM10x) for a bounded gate (the stock softplus
    gate of ``gate_lower_bound=None`` is not implemented)."""
    return (
        lower_bound is not None
        and state.dtype == torch.float32
        and state.is_cuda
        and torch.cuda.get_device_capability(state.device)[0] == 10
        and state.shape[2:] == (K, K)
        and state.stride()[1:] == (K * K, K, 1)
    )


def compile_commit_f32(
    state: torch.Tensor, window: int, params, align: bool, indexed: bool = False
):
    """Compile before serving (the commit runs eagerly after sampling)."""
    arch = architecture(state.device)
    _compiled(state.shape[1], window, arch, params, align, indexed)


def commit_f32(
    context,
    state_indices: torch.Tensor,
    batch: int,
    align: bool,
    record_rows: torch.Tensor | None = None,
):
    """Fold a prepared commit plan of ``context`` (a ``KDARecoverSSMCommitContext``)
    into every layer's FP32 state in one launch on the current stream; the records
    of request b are in slot ``record_rows[b]`` (default: b)."""
    state = context.checkpoints[0]
    compiled = _compiled(
        state.shape[1],
        context.spec_query_len,
        architecture(state.device),
        context.A_log.dtype,
        align,
        record_rows is not None,
    )
    rows = () if record_rows is None else (record_rows[:batch],)
    compiled(
        context.state_base_addrs,
        context.correction_cache_base_addrs,
        context.kg_cache_base_addrs,
        context.state_block_strides,
        context.correction_cache_block_strides,
        context.kg_cache_block_strides,
        context.A_log,
        context.dt_bias,
        state_indices[:batch],
        context.commit_lens[:batch],
        context.final_state_indices[:batch],
        context.boundary_state_indices[:batch],
        context.boundary_recovery_lens[:batch],
        context.lower_bound,
        NULL_BLOCK_ID,
        *rows,
    )
