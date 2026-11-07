# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL direct DCP A2A LSE reduce (the DCP decode output combine) on SM100.

Drop-in for ``torch.ops._C.direct_dcp_a2a_lse_reduce``, which
``DirectDCPA2AWorkspace`` selects on SM100 for the shapes accepted by
``direct_dcp_a2a_cutedsl_supported``. It uses the same symmetric buffers and
the same epoch protocol (``epoch + 1`` selects the parity slot; source ``r``
publishes ``signal[parity][r] = epoch + 1`` with a system-scope release), and
its output is bit-identical, in two launches instead of four:

``push``: one CTA per (token, destination rank). It copies the token's head
    slice for that destination (``heads_per_rank x head_dim``) and its LSE into
    the destination's receive slot, all 16-byte loads in flight before the
    remote stores. Empty-shard rows send an LSE of ``-inf``; their payload is
    never read by the combine. The last CTA of each destination (a local ticket)
    publishes the signal after a system-scope fence.
``combine``: one warp per (token, local head). It waits for all sources,
    forms the base-2 LSE weights in source order exactly as the CUDA op does,
    loads all non-zero-weight partials at once and accumulates in FP32. The last
    CTA advances the epoch, which replaces the separate epoch launch.

No host synchronization and no host-side state: graph replays only advance the
device epoch.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.cute_utils import recast_val
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)

# Matches kSpinLimit of the CUDA implementation.
_SPIN_LIMIT = 100_000_000
# LSE element type, a runtime argument of the push (one compile serves all).
_LSE_KINDS = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_OUTPUT_DTYPES = (torch.bfloat16, torch.float16)
_PUSH_THREADS = 128
_COMBINE_WARPS = 4
# The benchmarked range: 2-8 ranks, up to 64 heads per rank (MLA has at most
# 128 heads and DCP ranks split TP ranks, so at least 2 TP ranks share them),
# and MLA's 512-dim latent output. The push sends one LSE per thread, so
# heads_per_rank must not exceed _PUSH_THREADS.
_MAX_WORLD_SIZE = 8
_MAX_HEADS_PER_RANK = 64
_HEAD_DIM = 512
# CUDART_L2E_F, so base-e LSEs scale exactly as in the CUDA op.
_LOG2_E = float(torch.tensor(1.4426950408889634, dtype=torch.float32))


def direct_dcp_a2a_cutedsl_supported(
    world_size: int, heads_per_rank: int, head_dim: int, dtype: torch.dtype
) -> bool:
    """Shapes the CuTe kernels cover; others keep the CUDA implementation."""
    return (
        dtype in _OUTPUT_DTYPES
        and 1 < world_size <= _MAX_WORLD_SIZE
        and 0 < heads_per_rank <= _MAX_HEADS_PER_RANK
        and head_dim == _HEAD_DIM
    )


def _asm(result_type, operands, asm: str, constraints: str, *, loc, ip):
    return llvm.inline_asm(
        result_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


def _asm_void(operands, asm: str, constraints: str, *, loc, ip) -> None:
    # A dummy result keeps the side-effecting asm in the DSL's value graph.
    _asm(
        T.i32(),
        operands,
        "mov.u32 $0, 0;\n\t" + asm,
        ",".join(("=r", constraints)) if constraints else "=r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None) -> list[Uint32]:
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _st_v4(addr: Int64, v0, v1, v2, v3, *, loc=None, ip=None) -> None:
    _asm_void(
        [Int64(addr), *(Uint32(v) for v in (v0, v1, v2, v3))],
        "st.global.v4.u32 [$1], {$2, $3, $4, $5};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [Int64(addr)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [Int64(addr)], "ld.global.u32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_i64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(T.i64(), [Int64(addr)], "ld.global.u64 $0, [$1];", "=l,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_bf16_as_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    bits = _asm(
        T.i32(),
        [Int64(addr)],
        "ld.global.u16 $0, [$1];\n\tshl.b32 $0, $0, 16;",
        "=r,l",
        loc=loc,
        ip=ip,
    )
    return recast_val(Uint32(bits), Float32)


@dsl_user_op
def _ld_f16_as_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr)],
            "{\n\t.reg .b16 h;\n\tld.global.b16 h, [$1];\n\tcvt.f32.f16 $0, h;\n\t}",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_f32(addr: Int64, value: Float32, *, loc=None, ip=None) -> None:
    _asm_void(
        [Int64(addr), Float32(value)], "st.global.f32 [$1], $2;", "l,f", loc=loc, ip=ip
    )


@dsl_user_op
def _st_zero(addr: Int64, *, loc=None, ip=None) -> None:
    _asm_void([Int64(addr)], "st.global.u32 [$1], 0;", "l", loc=loc, ip=ip)


@dsl_user_op
def _ld_acquire_sys(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_release_sys(addr: Int64, value: Int32, *, loc=None, ip=None) -> None:
    _asm_void(
        [Int64(addr), Int32(value)],
        "st.release.sys.global.u32 [$1], $2;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fetch_add_gpu(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Relaxed GPU-scope ticket: returns the previous value."""
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _trap(*, loc=None, ip=None) -> None:
    _asm_void([], "trap;", "", loc=loc, ip=ip)


@dsl_user_op
def _pack_x2(lo: Float32, hi: Float32, fp16: bool, *, loc=None, ip=None) -> Uint32:
    """Round two FP32 values to nearest-even bf16 or fp16, ``lo`` in bits 0-15."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(hi), Float32(lo)],
            f"cvt.rn.{'f16x2' if fp16 else 'bf16x2'}.f32 $0, $1, $2;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unpack_x2(word: Uint32, fp16: bool, *, loc=None, ip=None) -> list[Float32]:
    """Widen the two bf16 or fp16 halves of ``word`` (low first) to FP32."""
    if not fp16:
        return [
            recast_val(word << Uint32(16), Float32, loc=loc, ip=ip),
            recast_val(word & Uint32(0xFFFF0000), Float32, loc=loc, ip=ip),
        ]
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Uint32(word)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $2;\n\t"
        "cvt.f32.f16 $0, lo;\n\tcvt.f32.f16 $1, hi;\n\t}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(2)
    ]


def _epoch_tensor() -> Any:
    # The device epoch is the one tensor argument: TVM FFI takes the launch
    # device and stream from it. Everything else is a raw address.
    return cute.runtime.make_fake_tensor(Int64, (1,), stride=(1,), assumed_align=8)


class DirectDCPA2APushKernel(VllmCuTeDSLJitKernel["DirectDCPA2APushKernel.CompileKey"]):
    """Send each destination's head slice and LSE into its receive slot."""

    @dataclass(frozen=True)
    class CompileKey:
        world_size: int
        heads_per_rank: int
        head_dim: int

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        world_size = compile_key.world_size
        heads_per_rank = compile_key.heads_per_rank
        head_dim = compile_key.head_dim
        threads = _PUSH_THREADS
        # 16-byte vectors per (token, destination) row of 16-bit partials.
        row_vectors = heads_per_rank * head_dim // 8
        vectors_per_thread = (row_vectors + threads - 1) // threads
        source_head_bytes = heads_per_rank * head_dim * 2

        @cute.kernel
        def push_kernel(
            partial_output: Int64,
            output_token_stride_bytes: Int64,
            partial_lse: Int64,
            lse_token_stride: Int64,
            lse_head_stride: Int64,
            lse_kind: Int32,
            seq_lens: Int64,
            query_start_loc: Int64,
            num_seqs: Int32,
            peer_output_ptrs: Int64,
            peer_lse_ptrs: Int64,
            peer_signal_ptrs: Int64,
            tickets: Int64,
            epoch: cute.Tensor,
            rank: Int32,
            num_tokens: Int32,
            max_num_tokens: Int32,
        ):
            cta, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            token = cta // Int32(world_size)
            # Destinations interleave across CTAs, so every rank feeds all
            # links at once instead of all ranks hitting rank 0 first.
            destination = (rank + cta - token * Int32(world_size)) % Int32(world_size)
            next_epoch = epoch[0] + Int64(1)
            parity = next_epoch & Int64(1)
            item = (
                (parity * Int64(world_size) + Int64(rank)) * Int64(max_num_tokens)
                + Int64(token)
            ) * Int64(heads_per_rank)

            source = (
                partial_output
                + Int64(token) * output_token_stride_bytes
                + Int64(destination) * Int64(source_head_bytes)
            )
            target = _ld_i64(
                peer_output_ptrs + Int64(destination) * Int64(8)
            ) + item * Int64(head_dim * 2)
            payload = cute.make_rmem_tensor((vectors_per_thread * 4,), Uint32)
            for j in cutlass.range_constexpr(vectors_per_thread):
                vector_idx = tid + Int32(j * threads)
                if vector_idx < Int32(row_vectors):
                    words = _ld_v4(source + Int64(vector_idx) * Int64(16))
                    for k in cutlass.range_constexpr(4):
                        payload[j * 4 + k] = words[k]

            lse = Float32(-float("inf"))
            if tid < Int32(heads_per_rank):
                empty_kv = False
                if num_seqs > Int32(0):
                    left = Int32(0)
                    right = num_seqs
                    while left < right:
                        mid = (left + right) >> 1
                        if _ld_i32(query_start_loc + Int64(mid) * Int64(4)) <= token:
                            left = mid + Int32(1)
                        else:
                            right = mid
                    empty_kv = _ld_i32(
                        seq_lens + Int64(left - Int32(1)) * Int64(4)
                    ) == Int32(0)
                if not empty_kv:
                    lse_offset = (
                        Int64(token) * lse_token_stride
                        + Int64(destination * Int32(heads_per_rank) + tid)
                        * lse_head_stride
                    )
                    if lse_kind == Int32(0):
                        lse = _ld_f32(partial_lse + lse_offset * Int64(4))
                    elif lse_kind == Int32(1):
                        lse = _ld_bf16_as_f32(partial_lse + lse_offset * Int64(2))
                    else:
                        lse = _ld_f16_as_f32(partial_lse + lse_offset * Int64(2))

            for j in cutlass.range_constexpr(vectors_per_thread):
                vector_idx = tid + Int32(j * threads)
                if vector_idx < Int32(row_vectors):
                    _st_v4(
                        target + Int64(vector_idx) * Int64(16),
                        payload[j * 4],
                        payload[j * 4 + 1],
                        payload[j * 4 + 2],
                        payload[j * 4 + 3],
                    )
            if tid < Int32(heads_per_rank):
                _st_f32(
                    _ld_i64(peer_lse_ptrs + Int64(destination) * Int64(8))
                    + (item + Int64(tid)) * Int64(4),
                    lse,
                )

            # Publish after the last CTA of this destination: every CTA
            # releases its stores, the last ticket acquires them all.
            cute.arch.sync_threads()
            if tid == Int32(0):
                cute.arch.fence_acq_rel_sys()
                ticket_addr = tickets + Int64(destination) * Int64(4)
                if _fetch_add_gpu(ticket_addr) == num_tokens - Int32(1):
                    _st_zero(ticket_addr)
                    cute.arch.fence_acq_rel_sys()
                    _st_release_sys(
                        _ld_i64(peer_signal_ptrs + Int64(destination) * Int64(8))
                        + (parity * Int64(world_size) + Int64(rank)) * Int64(4),
                        Int32(next_epoch),
                    )

        @cute.jit
        def host_entrypoint(
            partial_output: Int64,
            output_token_stride_bytes: Int64,
            partial_lse: Int64,
            lse_token_stride: Int64,
            lse_head_stride: Int64,
            lse_kind: Int32,
            seq_lens: Int64,
            query_start_loc: Int64,
            num_seqs: Int32,
            peer_output_ptrs: Int64,
            peer_lse_ptrs: Int64,
            peer_signal_ptrs: Int64,
            tickets: Int64,
            epoch: cute.Tensor,
            rank: Int32,
            num_tokens: Int32,
            max_num_tokens: Int32,
            stream,
        ):
            push_kernel(
                partial_output,
                output_token_stride_bytes,
                partial_lse,
                lse_token_stride,
                lse_head_stride,
                lse_kind,
                seq_lens,
                query_start_loc,
                num_seqs,
                peer_output_ptrs,
                peer_lse_ptrs,
                peer_signal_ptrs,
                tickets,
                epoch,
                rank,
                num_tokens,
                max_num_tokens,
            ).launch(
                grid=(num_tokens * Int32(world_size), 1, 1),
                block=(threads, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        world_size: int,
        heads_per_rank: int,
        head_dim: int,
    ) -> CompileKey:
        return self.CompileKey(
            world_size=world_size,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
        )

    def get_warmup_keys(
        self,
        *,
        world_size: int,
        heads_per_rank: int,
        head_dim: int,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            world_size=world_size,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        return (
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int64(0),
            Int32(0),
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            _epoch_tensor(),
            Int32(0),
            Int32(1),
            Int32(1),
        )

    @kernel_launcher
    def __call__(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_seqs: int,
        peer_output_ptrs: torch.Tensor,
        peer_lse_ptrs: torch.Tensor,
        peer_signal_ptrs: torch.Tensor,
        tickets: torch.Tensor,
        epoch: torch.Tensor,
        *,
        world_size: int,
        rank: int,
        max_num_tokens: int,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        compile_key = self.dispatch(
            world_size=world_size,
            heads_per_rank=partial_output.shape[1] // world_size,
            head_dim=partial_output.shape[2],
        )
        return compile_key, (
            partial_output.data_ptr(),
            partial_output.stride(0) * partial_output.element_size(),
            partial_lse.data_ptr(),
            partial_lse.stride(0),
            partial_lse.stride(1),
            _LSE_KINDS[partial_lse.dtype],
            seq_lens.data_ptr(),
            query_start_loc.data_ptr(),
            num_seqs,
            peer_output_ptrs.data_ptr(),
            peer_lse_ptrs.data_ptr(),
            peer_signal_ptrs.data_ptr(),
            tickets.data_ptr(),
            epoch,
            rank,
            partial_output.shape[0],
            max_num_tokens,
        )


class DirectDCPA2ACombineKernel(
    VllmCuTeDSLJitKernel["DirectDCPA2ACombineKernel.CompileKey"]
):
    """Wait for every source, then LSE-combine this rank's heads in FP32."""

    @dataclass(frozen=True)
    class CompileKey:
        world_size: int
        heads_per_rank: int
        head_dim: int
        dtype: torch.dtype

    @staticmethod
    def kernel(compile_key: CompileKey) -> Any:
        world_size = compile_key.world_size
        heads_per_rank = compile_key.heads_per_rank
        head_dim = compile_key.head_dim
        fp16 = compile_key.dtype == torch.float16
        warps = _COMBINE_WARPS
        # Lane l owns dims [32 * 8 * j + 8 * l, +8): each warp-wide 16-byte
        # access covers 512 contiguous bytes.
        vectors_per_lane = head_dim // 256
        neg_inf = -float("inf")

        @cute.kernel
        def combine_kernel(
            received_output: Int64,
            received_lse: Int64,
            received_signal: Int64,
            epoch: cute.Tensor,
            done: Int64,
            combined_output: Int64,
            num_tokens: Int32,
            max_num_tokens: Int32,
            lse_scale: Float32,
        ):
            cta, _, _ = cute.arch.block_idx()
            num_ctas, _, _ = cute.arch.grid_dim()
            tid, _, _ = cute.arch.thread_idx()
            warp = cute.arch.warp_idx()
            lane = cute.arch.lane_idx()
            next_epoch = epoch[0] + Int64(1)
            parity = next_epoch & Int64(1)
            slot = parity * Int64(world_size)

            if warp == Int32(0) and lane < Int32(world_size):
                signal_addr = received_signal + (slot + Int64(lane)) * Int64(4)
                expected = Int32(next_epoch)
                spins = Int32(0)
                while _ld_acquire_sys(signal_addr) != expected:
                    spins += Int32(1)
                    if spins > Int32(_SPIN_LIMIT):
                        cute.printf(
                            "direct DCP A2A timeout source=%d epoch=%d\n",
                            lane,
                            expected,
                        )
                        _trap()
            cute.arch.sync_threads()

            row = cta * Int32(warps) + warp
            if row < num_tokens * Int32(heads_per_rank):
                token = row // Int32(heads_per_rank)
                head = row - token * Int32(heads_per_rank)
                # Same arithmetic, in the same source order, as the CUDA op.
                weights = cute.make_rmem_tensor((world_size,), Float32)
                lse_max = Float32(neg_inf)
                for source in cutlass.range_constexpr(world_size):
                    source_item = (
                        (slot + Int64(source)) * Int64(max_num_tokens) + Int64(token)
                    ) * Int64(heads_per_rank) + Int64(head)
                    value = _ld_f32(received_lse + source_item * Int64(4))
                    # NaN and +inf mark undefined rows (bit test: NaN-safe).
                    bits = recast_val(value, Int32)
                    magnitude = bits & Int32(0x7FFFFFFF)
                    if magnitude > Int32(0x7F800000) or bits == Int32(0x7F800000):
                        value = Float32(neg_inf)
                    value = value * lse_scale
                    weights[source] = value
                    lse_max = cute.arch.fmax(lse_max, value)
                if lse_max == Float32(neg_inf):
                    lse_max = Float32(0.0)
                lse_sum = Float32(0.0)
                for source in cutlass.range_constexpr(world_size):
                    lse_sum += cute.math.exp2(weights[source] - lse_max)
                inverse_lse_sum = Float32(0.0)
                if lse_sum > Float32(0.0):
                    inverse_lse_sum = Float32(1.0) / lse_sum
                for source in cutlass.range_constexpr(world_size):
                    weights[source] = (
                        cute.math.exp2(weights[source] - lse_max) * inverse_lse_sum
                    )

                # Issue every load first. Empty shards leave stale or undefined
                # payloads, so zero-weight sources are never read.
                payload = cute.make_rmem_tensor(
                    (world_size * vectors_per_lane * 4,), Uint32
                )
                for source in cutlass.range_constexpr(world_size):
                    source_row = (
                        (slot + Int64(source)) * Int64(max_num_tokens) + Int64(token)
                    ) * Int64(heads_per_rank) + Int64(head)
                    base = received_output + source_row * Int64(head_dim * 2)
                    for j in cutlass.range_constexpr(vectors_per_lane):
                        idx = (source * vectors_per_lane + j) * 4
                        for k in cutlass.range_constexpr(4):
                            payload[idx + k] = Uint32(0)
                        if weights[source] != Float32(0.0):
                            words = _ld_v4(base + Int64((j * 32 + lane) * 16))
                            for k in cutlass.range_constexpr(4):
                                payload[idx + k] = words[k]

                accumulator = cute.make_rmem_tensor((vectors_per_lane * 8,), Float32)
                for e in cutlass.range_constexpr(vectors_per_lane * 8):
                    accumulator[e] = Float32(0.0)
                for source in cutlass.range_constexpr(world_size):
                    weight = weights[source]
                    if weight != Float32(0.0):
                        for j in cutlass.range_constexpr(vectors_per_lane):
                            for k in cutlass.range_constexpr(4):
                                lo, hi = _unpack_x2(
                                    payload[(source * vectors_per_lane + j) * 4 + k],
                                    fp16,
                                )
                                e = j * 8 + k * 2
                                accumulator[e] += lo * weight
                                accumulator[e + 1] += hi * weight

                out_row = combined_output + Int64(row) * Int64(head_dim * 2)
                for j in cutlass.range_constexpr(vectors_per_lane):
                    e = j * 8
                    _st_v4(
                        out_row + Int64((j * 32 + lane) * 16),
                        _pack_x2(accumulator[e], accumulator[e + 1], fp16),
                        _pack_x2(accumulator[e + 2], accumulator[e + 3], fp16),
                        _pack_x2(accumulator[e + 4], accumulator[e + 5], fp16),
                        _pack_x2(accumulator[e + 6], accumulator[e + 7], fp16),
                    )

            # The last CTA advances the epoch once every CTA has read it.
            cute.arch.sync_threads()
            # Nested: the ticket must be taken by one thread only.
            if tid == Int32(0):  # noqa: SIM102
                if _fetch_add_gpu(done) == num_ctas - Int32(1):
                    _st_zero(done)
                    epoch[0] = next_epoch

        @cute.jit
        def host_entrypoint(
            received_output: Int64,
            received_lse: Int64,
            received_signal: Int64,
            epoch: cute.Tensor,
            done: Int64,
            combined_output: Int64,
            num_tokens: Int32,
            max_num_tokens: Int32,
            lse_scale: Float32,
            stream,
        ):
            rows = num_tokens * Int32(heads_per_rank)
            combine_kernel(
                received_output,
                received_lse,
                received_signal,
                epoch,
                done,
                combined_output,
                num_tokens,
                max_num_tokens,
                lse_scale,
            ).launch(
                grid=((rows + Int32(warps - 1)) // Int32(warps), 1, 1),
                block=(warps * 32, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def dispatch(  # type: ignore[override]
        self,
        *,
        world_size: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> CompileKey:
        return self.CompileKey(
            world_size=world_size,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
            dtype=dtype,
        )

    def get_warmup_keys(
        self,
        *,
        world_size: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(
            world_size=world_size,
            heads_per_rank=heads_per_rank,
            head_dim=head_dim,
            dtype=dtype,
        )

    def warmup_inputs(self, compile_key: CompileKey) -> tuple[Any, ...]:
        return (
            Int64(0),
            Int64(0),
            Int64(0),
            _epoch_tensor(),
            Int64(0),
            Int64(0),
            Int32(1),
            Int32(1),
            Float32(1.0),
        )

    @kernel_launcher
    def __call__(
        self,
        received_output: torch.Tensor,
        received_lse: torch.Tensor,
        received_signal: torch.Tensor,
        epoch: torch.Tensor,
        done: torch.Tensor,
        combined_output: torch.Tensor,
        *,
        world_size: int,
        max_num_tokens: int,
        lse_scale: float,
    ) -> CuTeDSLLaunchSpec[CompileKey]:
        compile_key = self.dispatch(
            world_size=world_size,
            heads_per_rank=combined_output.shape[1],
            head_dim=combined_output.shape[2],
            dtype=combined_output.dtype,
        )
        return compile_key, (
            received_output.data_ptr(),
            received_lse.data_ptr(),
            received_signal.data_ptr(),
            epoch,
            done.data_ptr(),
            combined_output.data_ptr(),
            combined_output.shape[0],
            max_num_tokens,
            lse_scale,
        )


DIRECT_DCP_A2A_PUSH_KERNEL = DirectDCPA2APushKernel()
DIRECT_DCP_A2A_COMBINE_KERNEL = DirectDCPA2ACombineKernel()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"CuTe DSL direct DCP A2A: {message}")


def _is_int32_vector(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 1 and tensor.stride(0) == 1 and tensor.dtype == torch.int32


def register_warmup(
    world_size: int, heads_per_rank: int, head_dim: int, dtype: torch.dtype
) -> None:
    """Compile both kernels during JIT warmup.

    These are the only compile keys a workspace can hit: the LSE dtype, LSE
    strides, token count and epoch are runtime arguments.
    """
    DIRECT_DCP_A2A_PUSH_KERNEL.register_warmup(
        world_size=world_size, heads_per_rank=heads_per_rank, head_dim=head_dim
    )
    DIRECT_DCP_A2A_COMBINE_KERNEL.register_warmup(
        world_size=world_size,
        heads_per_rank=heads_per_rank,
        head_dim=head_dim,
        dtype=dtype,
    )


def direct_dcp_a2a_lse_reduce_cutedsl(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    seq_lens: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    peer_output_ptrs: torch.Tensor,
    peer_lse_ptrs: torch.Tensor,
    peer_signal_ptrs: torch.Tensor,
    received_output: torch.Tensor,
    received_lse: torch.Tensor,
    received_signal: torch.Tensor,
    epoch: torch.Tensor,
    combined_output: torch.Tensor,
    world_size: int,
    rank: int,
    max_num_tokens: int,
    is_lse_base_on_e: bool,
    *,
    counters: torch.Tensor,
) -> None:
    """``torch.ops._C.direct_dcp_a2a_lse_reduce`` with the same arguments.

    ``counters`` is a zero-initialized int32 ``[world_size + 1]`` tensor (push
    tickets per destination, then the combine's CTA count) that the kernels
    leave at zero. The checks mirror the CUDA op's (metadata only, no sync): a
    shape mismatch would otherwise write out of bounds into peer memory.
    """
    _check(
        partial_output.dim() == 3 and partial_lse.dim() == 2,
        "expected output [T, H, D] and LSE [T, H]",
    )
    num_tokens, total_heads, head_dim = partial_output.shape
    _check(0 <= rank < world_size, "invalid rank")
    _check(total_heads % world_size == 0, "heads must divide across DCP ranks")
    heads_per_rank = total_heads // world_size
    _check(
        direct_dcp_a2a_cutedsl_supported(
            world_size, heads_per_rank, head_dim, partial_output.dtype
        ),
        "unsupported shape or dtype",
    )
    _check(0 < num_tokens <= max_num_tokens, "token count exceeds the workspace")
    _check(
        partial_output.stride(2) == 1
        and partial_output.stride(1) == head_dim
        and partial_output.stride(0) >= total_heads * head_dim
        and partial_output.stride(0) % 8 == 0,
        "partial output must have packed heads and an aligned token stride",
    )
    _check(
        partial_lse.shape == (num_tokens, total_heads)
        and partial_lse.dtype in _LSE_KINDS,
        "LSE must be FP32, FP16 or BF16 [T, H]",
    )
    _check(
        combined_output.shape == (num_tokens, heads_per_rank, head_dim)
        and combined_output.dtype == partial_output.dtype
        and combined_output.is_contiguous(),
        "combined output must be a contiguous [T, H / W, D] tensor",
    )
    _check(
        received_output.dtype == partial_output.dtype
        and received_lse.dtype == torch.float32
        and received_signal.dtype == torch.int32
        and peer_output_ptrs.dtype == torch.int64
        and peer_lse_ptrs.dtype == torch.int64
        and peer_signal_ptrs.dtype == torch.int64
        and counters.shape == (world_size + 1,)
        and counters.dtype == torch.int32,
        "invalid symmetric buffer or counter dtypes",
    )
    _check(
        (seq_lens is None) == (query_start_loc is None),
        "seq_lens and query_start_loc must be provided together",
    )
    num_seqs = 0
    if seq_lens is None or query_start_loc is None:
        seq_lens = query_start_loc = counters[:1]
    else:
        num_seqs = seq_lens.shape[0]
        _check(
            _is_int32_vector(seq_lens)
            and _is_int32_vector(query_start_loc)
            and num_seqs > 0
            and query_start_loc.shape[0] == num_seqs + 1,
            "seq_lens [N] and query_start_loc [N + 1] must be contiguous int32",
        )
    DIRECT_DCP_A2A_PUSH_KERNEL(
        partial_output,
        partial_lse,
        seq_lens,
        query_start_loc,
        num_seqs,
        peer_output_ptrs,
        peer_lse_ptrs,
        peer_signal_ptrs,
        counters,
        epoch,
        world_size=world_size,
        rank=rank,
        max_num_tokens=max_num_tokens,
    )
    DIRECT_DCP_A2A_COMBINE_KERNEL(
        received_output,
        received_lse,
        received_signal,
        epoch,
        counters[world_size:],
        combined_output,
        world_size=world_size,
        max_num_tokens=max_num_tokens,
        lse_scale=_LOG2_E if is_lse_base_on_e else 1.0,
    )
