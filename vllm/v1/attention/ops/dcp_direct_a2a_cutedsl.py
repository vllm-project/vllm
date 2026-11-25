# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL `torch.ops._C.direct_dcp_a2a_lse_reduce` for SM100 (BF16, 512 dims).

Same symmetric buffers, epoch and signal protocol, and bit-identical output as
the CUDA op, in two launches instead of four:

`push`: one CTA per (token, destination) copies the token's head slice and LSE
    to the destination's receive slot (tokens with an empty KV shard send LSE
    `-inf`). The last CTA per destination publishes the signal.
`combine`: one warp per (token, local head) waits for every source and
    combines in FP32 in the CUDA op's source order. The last CTA advances the
    epoch.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float16, Float32, Int32, Int64, Uint16, Uint32
from cutlass._mlir.dialects import llvm

from vllm.cute_utils import recast_val
from vllm.cute_utils.cvt import bf16x2_to_fp32x2, fp32x2_to_bf16x2
from vllm.cute_utils.ptx_mem import ld_global_v4, st_global_v4
from vllm.model_executor.warmup.jit_warmup import kernel_launcher
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
    CuTeDSLLaunchSpec,
    VllmCuTeDSLJitKernel,
)

# Matches kSpinLimit of the CUDA implementation.
_SPIN_LIMIT = 100_000_000
# LSE element type, a runtime argument of the push (one compile serves all).
_LSE_KINDS = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
_HEAD_DIM = 512
_PUSH_THREADS = 128
_COMBINE_WARPS = 4
# CUDART_L2E_F, so base-e LSEs scale exactly as in the CUDA op.
_LOG2_E = float(torch.tensor(1.4426950408889634, dtype=torch.float32))


@dataclass(frozen=True)
class _CompileKey:
    world_size: int
    heads_per_rank: int


def _ptr(dtype, addr: Int64) -> cute.Pointer:
    return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem)


def _ld(dtype, addr: Int64, **kwargs):
    return cute.arch.load(_ptr(dtype, addr), dtype, **kwargs)


def _st(addr: Int64, value, **kwargs) -> None:
    cute.arch.store(_ptr(type(value), addr), value, **kwargs)


def _take_ticket(addr: Int64) -> Int32:
    return cute.arch.atomic_add(_ptr(Int32, addr), Int32(1), sem="relaxed", scope="gpu")


def _epoch_tensor() -> Any:
    # The device epoch is the one tensor argument: TVM FFI takes the launch
    # device and stream from it. Everything else is a raw address.
    return cute.runtime.make_fake_tensor(Int64, (1,), stride=(1,), assumed_align=8)


class DirectDCPA2APushKernel(VllmCuTeDSLJitKernel[_CompileKey]):
    """Send each destination's head slice and LSE into its receive slot."""

    @staticmethod
    def kernel(compile_key: _CompileKey) -> Any:
        world_size = compile_key.world_size
        heads_per_rank = compile_key.heads_per_rank
        threads = _PUSH_THREADS
        # 16-byte vectors of one token's BF16 partials for one destination.
        token_vectors = heads_per_rank * _HEAD_DIM // 8
        vectors_per_thread = (token_vectors + threads - 1) // threads

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
                + Int64(destination) * Int64(heads_per_rank * _HEAD_DIM * 2)
            )
            target = _ld(
                Int64, peer_output_ptrs + Int64(destination) * Int64(8)
            ) + item * Int64(_HEAD_DIM * 2)
            # All loads in flight before the remote stores.
            payload = cute.make_rmem_tensor((vectors_per_thread * 4,), Uint32)
            for j in cutlass.range_constexpr(vectors_per_thread):
                vector_idx = tid + Int32(j * threads)
                if vector_idx < Int32(token_vectors):
                    words = ld_global_v4(source + Int64(vector_idx) * Int64(16))
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
                        if _ld(Int32, query_start_loc + Int64(mid) * Int64(4)) <= token:
                            left = mid + Int32(1)
                        else:
                            right = mid
                    empty_kv = _ld(
                        Int32, seq_lens + Int64(left - Int32(1)) * Int64(4)
                    ) == Int32(0)
                if not empty_kv:
                    lse_offset = (
                        Int64(token) * lse_token_stride
                        + Int64(destination * Int32(heads_per_rank) + tid)
                        * lse_head_stride
                    )
                    if lse_kind == Int32(0):
                        lse = _ld(Float32, partial_lse + lse_offset * Int64(4))
                    else:
                        bits = _ld(Uint16, partial_lse + lse_offset * Int64(2))
                        if lse_kind == Int32(1):
                            lse = recast_val(Uint32(bits) << Uint32(16), Float32)
                        else:
                            lse = recast_val(bits, Float16).to(Float32)

            for j in cutlass.range_constexpr(vectors_per_thread):
                vector_idx = tid + Int32(j * threads)
                if vector_idx < Int32(token_vectors):
                    st_global_v4(
                        target + Int64(vector_idx) * Int64(16),
                        payload[j * 4],
                        payload[j * 4 + 1],
                        payload[j * 4 + 2],
                        payload[j * 4 + 3],
                    )
            if tid < Int32(heads_per_rank):
                _st(
                    _ld(Int64, peer_lse_ptrs + Int64(destination) * Int64(8))
                    + (item + Int64(tid)) * Int64(4),
                    lse,
                )

            # Publish after the last CTA of this destination: every CTA
            # releases its stores, the last ticket acquires them all.
            cute.arch.sync_threads()
            if tid == Int32(0):
                cute.arch.fence_acq_rel_sys()
                ticket_addr = tickets + Int64(destination) * Int64(4)
                if _take_ticket(ticket_addr) == num_tokens - Int32(1):
                    _st(ticket_addr, Int32(0))
                    cute.arch.fence_acq_rel_sys()
                    _st(
                        _ld(Int64, peer_signal_ptrs + Int64(destination) * Int64(8))
                        + (parity * Int64(world_size) + Int64(rank)) * Int64(4),
                        Int32(next_epoch),
                        sem="release",
                        scope="sys",
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

    def get_warmup_keys(
        self, *, world_size: int, heads_per_rank: int
    ) -> list[_CompileKey]:
        return [_CompileKey(world_size, heads_per_rank)]

    def warmup_inputs(self, compile_key: _CompileKey) -> tuple[Any, ...]:
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
    ) -> CuTeDSLLaunchSpec[_CompileKey]:
        compile_key = _CompileKey(world_size, partial_output.shape[1] // world_size)
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


class DirectDCPA2ACombineKernel(VllmCuTeDSLJitKernel[_CompileKey]):
    """Wait for every source, then LSE-combine this rank's heads in FP32."""

    @staticmethod
    def kernel(compile_key: _CompileKey) -> Any:
        world_size = compile_key.world_size
        heads_per_rank = compile_key.heads_per_rank
        warps = _COMBINE_WARPS
        # Lane l owns dims [256 * j + 8 * l, +8): each warp-wide 16-byte
        # access covers 512 contiguous bytes.
        vectors_per_lane = _HEAD_DIM // 256
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
                while _ld(Int32, signal_addr, sem="acquire", scope="sys") != expected:
                    spins += Int32(1)
                    if spins > Int32(_SPIN_LIMIT):
                        cute.printf(
                            "direct DCP A2A timeout source=%d epoch=%d\n",
                            lane,
                            expected,
                        )
                        llvm.intr_trap()
            cute.arch.sync_threads()

            token_head = cta * Int32(warps) + warp
            if token_head < num_tokens * Int32(heads_per_rank):
                token = token_head // Int32(heads_per_rank)
                head = token_head - token * Int32(heads_per_rank)
                # Same arithmetic, in the same source order, as the CUDA op.
                weights = cute.make_rmem_tensor((world_size,), Float32)
                lse_max = Float32(neg_inf)
                for source in cutlass.range_constexpr(world_size):
                    source_item = (
                        (slot + Int64(source)) * Int64(max_num_tokens) + Int64(token)
                    ) * Int64(heads_per_rank) + Int64(head)
                    value = _ld(Float32, received_lse + source_item * Int64(4))
                    # NaN and +inf mark undefined partials (bit test: NaN-safe).
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
                    source_item = (
                        (slot + Int64(source)) * Int64(max_num_tokens) + Int64(token)
                    ) * Int64(heads_per_rank) + Int64(head)
                    base = received_output + source_item * Int64(_HEAD_DIM * 2)
                    for j in cutlass.range_constexpr(vectors_per_lane):
                        idx = (source * vectors_per_lane + j) * 4
                        for k in cutlass.range_constexpr(4):
                            payload[idx + k] = Uint32(0)
                        if weights[source] != Float32(0.0):
                            words = ld_global_v4(base + Int64((j * 32 + lane) * 16))
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
                                lo, hi = bf16x2_to_fp32x2(
                                    payload[(source * vectors_per_lane + j) * 4 + k]
                                )
                                e = j * 8 + k * 2
                                accumulator[e] += lo * weight
                                accumulator[e + 1] += hi * weight

                out_base = combined_output + Int64(token_head) * Int64(_HEAD_DIM * 2)
                for j in cutlass.range_constexpr(vectors_per_lane):
                    e = j * 8
                    st_global_v4(
                        out_base + Int64((j * 32 + lane) * 16),
                        fp32x2_to_bf16x2(accumulator[e], accumulator[e + 1]),
                        fp32x2_to_bf16x2(accumulator[e + 2], accumulator[e + 3]),
                        fp32x2_to_bf16x2(accumulator[e + 4], accumulator[e + 5]),
                        fp32x2_to_bf16x2(accumulator[e + 6], accumulator[e + 7]),
                    )

            # The last CTA advances the epoch once every CTA has read it.
            cute.arch.sync_threads()
            # Nested: the ticket must be taken by one thread only.
            if tid == Int32(0):  # noqa: SIM102
                if _take_ticket(done) == num_ctas - Int32(1):
                    _st(done, Int32(0))
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
            num_token_heads = num_tokens * Int32(heads_per_rank)
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
                grid=((num_token_heads + Int32(warps - 1)) // Int32(warps), 1, 1),
                block=(warps * 32, 1, 1),
                stream=stream,
            )

        return host_entrypoint

    def get_warmup_keys(
        self, *, world_size: int, heads_per_rank: int
    ) -> list[_CompileKey]:
        return [_CompileKey(world_size, heads_per_rank)]

    def warmup_inputs(self, compile_key: _CompileKey) -> tuple[Any, ...]:
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
    ) -> CuTeDSLLaunchSpec[_CompileKey]:
        compile_key = _CompileKey(world_size, combined_output.shape[1])
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


def register_warmup(world_size: int, heads_per_rank: int) -> None:
    """Compile both kernels during JIT warmup (their only compile keys)."""
    DIRECT_DCP_A2A_PUSH_KERNEL.register_warmup(
        world_size=world_size, heads_per_rank=heads_per_rank
    )
    DIRECT_DCP_A2A_COMBINE_KERNEL.register_warmup(
        world_size=world_size, heads_per_rank=heads_per_rank
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"CuTe DSL direct DCP A2A: {message}")


def _is_int32_vector(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 1 and tensor.stride(0) == 1 and tensor.dtype == torch.int32


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
    """`torch.ops._C.direct_dcp_a2a_lse_reduce` with the same arguments.

    `counters` is a zero-initialized int32 `[world_size + 1]` tensor (push
    tickets per destination, then the combine's CTA count) that the kernels
    leave at zero. The workspace owns every other buffer; the caller's inputs
    are checked (metadata only), as a bad shape would write into peer memory.
    """
    num_tokens, heads_per_rank, _ = combined_output.shape
    total_heads = world_size * heads_per_rank
    _check(
        0 < num_tokens <= max_num_tokens
        and partial_output.dtype == torch.bfloat16
        and partial_output.shape[1:] == (total_heads, _HEAD_DIM)
        and partial_output.stride()[1:] == (_HEAD_DIM, 1)
        and partial_output.stride(0) % 8 == 0,
        "partial output must be BF16 [T, H, 512] with packed heads and a 16-byte "
        "aligned token stride",
    )
    _check(
        partial_lse.shape == (num_tokens, total_heads)
        and partial_lse.dtype in _LSE_KINDS,
        "LSE must be FP32, FP16 or BF16 [T, H]",
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
            _is_int32_vector(seq_lens) and _is_int32_vector(query_start_loc),
            "seq_lens and query_start_loc must be contiguous int32",
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
