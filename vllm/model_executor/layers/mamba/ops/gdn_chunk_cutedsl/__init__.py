# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32, cute
from quack.compile_utils import make_fake_tensor

from vllm.triton_utils import triton

from .kernel_h import h_cutedsl
from .kernel_kkt_inv_uw import kkt_inv_uw_cutedsl
from .kernel_o import o_cutedsl


class PrepMetaKernel:
    def __init__(self, BT: int) -> None:
        self.BT = BT
        self.num_warps = 8

    @cute.jit
    def __call__(
        self,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        chunk_offsets: cute.Tensor,
        stream: CUstream,
    ):
        block = (self.num_warps * 32, 1, 1)
        self.kernel(
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
        ).launch(grid=(1, 1, 1), block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        cu_seqlens: cute.Tensor,
        chunk_indices: cute.Tensor,
        chunk_offsets: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        num_seqs = cu_seqlens.shape[0] - 1
        num_warps = self.num_warps
        tb_size = num_warps * 32

        if tid == 0:
            chunk_offsets[0] = 0

        coarsen = cute.ceil_div(num_seqs, tb_size)
        seq_start = tid * coarsen
        num_iters = cutlass.min(seq_start + coarsen, num_seqs) - seq_start

        # First pass: compute this thread's total chunk count.
        thread_sum = Int32(0)
        for i in range(num_iters):
            seq_id = seq_start + i
            seqlen = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]
            thread_sum += cute.ceil_div(seqlen, self.BT)

        # warp parallel scan
        cu_num_chunks = thread_sum
        for i in cutlass.range_constexpr(5):
            offset = cutlass.const_expr(1 << i)
            lower = cute.arch.shuffle_sync_up(
                cu_num_chunks, offset=offset, mask_and_clamp=0
            )
            if lane_id >= offset:
                cu_num_chunks += lower

        # cross-warp cumsum (CTA-wide)
        smem = cutlass.utils.SmemAllocator()
        warp_num_chunks = smem.allocate_array(Int32, num_warps)
        if lane_id == 31:
            warp_num_chunks[warp_id] = cu_num_chunks
        cute.arch.sync_threads()

        for i in cutlass.range_constexpr(1, num_warps):
            if warp_id >= i:
                cu_num_chunks += warp_num_chunks[i - 1]

        chunk_start = cu_num_chunks - thread_sum

        # Second pass: recompute per-sequence chunk counts and write results.
        for i in range(num_iters):
            seq_id = seq_start + i
            seqlen = cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]
            num_chunks = cute.ceil_div(seqlen, self.BT)
            chunk_end = chunk_start + num_chunks
            chunk_offsets[seq_id + 1] = chunk_end

            for chunk_id in range(num_chunks):
                chunk_indices[chunk_start + chunk_id, 0] = seq_id
                chunk_indices[chunk_start + chunk_id, 1] = chunk_id

            chunk_start = chunk_end

    @cache
    @staticmethod
    def compile(BT: int):
        cu_entries = cute.sym_int()
        upper_bound_chunks = cute.sym_int()

        cu_seqlens = make_fake_tensor(Int32, (cu_entries,), divisibility=1)
        chunk_indices = make_fake_tensor(Int32, (upper_bound_chunks, 2), divisibility=2)
        chunk_offsets = make_fake_tensor(Int32, (cu_entries,), divisibility=1)

        kernel = PrepMetaKernel(BT)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            stream,
            options="--enable-tvm-ffi",
        )


def _upper_bound_chunks(num_seqs: int, total_tokens: int, chunk_size: int) -> int:
    return (num_seqs - 1) + triton.cdiv(total_tokens - (num_seqs - 1), chunk_size)


def prepare_metadata_cutedsl(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_seqs = cu_seqlens.numel() - 1
    upper_bound_chunks = _upper_bound_chunks(num_seqs, total_tokens, chunk_size)
    chunk_offsets = cu_seqlens.new_empty(num_seqs + 1, dtype=torch.int32)
    chunk_indices = cu_seqlens.new_empty((upper_bound_chunks, 2), dtype=torch.int32)

    PrepMetaKernel.compile(chunk_size)(cu_seqlens, chunk_indices, chunk_offsets)
    return chunk_indices, chunk_offsets


def chunk_gated_delta_rule_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the GDN chunk CuteDSL prefill kernels.

    Args:
        q: Query tensor with shape ``[1, T, H, K]``.
        k: Key tensor with shape ``[1, T, H, K]``.
        v: Value tensor with shape ``[1, T, Hv, V]``.
        g: Log-space decay tensor with shape ``[1, T, Hv]``.
        beta: Delta-rule beta tensor with shape ``[1, T, Hv]``.
        initial_state: Recurrent state with shape ``[N, Hv, V, K]``.
        cu_seqlens: Cumulative sequence lengths with shape ``[N + 1]``.
        chunk_indices: Chunk index metadata with shape ``[NT, 2]``.
        chunk_offsets: Cumulative chunk offsets with shape ``[N + 1]``.
        core_attn_out: Optional output buffer with shape ``[T, Hv, V]``.

    Returns:
        A tuple ``(output, final_state)`` where ``output`` has shape
        ``[1, T, Hv, V]`` and ``final_state`` has shape ``[N, Hv, V, K]``.
        When ``core_attn_out`` is provided, ``output`` is an unsqueezed view of
        that buffer.
    """
    q = q.squeeze(0)
    k = k.squeeze(0)
    v = v.squeeze(0)
    g = g.squeeze(0)
    beta = beta.squeeze(0)

    _, _, K_dim = k.shape
    _, num_v_heads, V_dim = v.shape
    chunk_size = 64
    upper_bound_chunks = chunk_indices.shape[0]
    pad_t = upper_bound_chunks * chunk_size
    total_chunks_ptr = chunk_offsets[-1:]

    g_cu = torch.empty_like(g, dtype=torch.float32)
    u = q.new_empty(pad_t, num_v_heads, V_dim)
    w = q.new_empty(pad_t, num_v_heads, K_dim)

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    kkt_inv_uw_cutedsl(
        k,
        v,
        u,
        w,
        g,
        beta,
        g_cu,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        num_sms=num_sms,
    )

    h = k.new_empty(upper_bound_chunks, num_v_heads, V_dim, K_dim)
    v_new = q.new_empty(pad_t, num_v_heads, V_dim)
    final_state = torch.empty_like(initial_state)
    h_cutedsl(
        k,
        u,
        w,
        v_new,
        g_cu,
        h,
        initial_state,
        final_state,
        cu_seqlens,
        chunk_offsets,
    )

    output = core_attn_out if core_attn_out is not None else torch.empty_like(v)
    scale = K_dim**-0.5
    o_cutedsl(
        q,
        k,
        v_new,
        h,
        g_cu,
        output,
        cu_seqlens,
        chunk_indices,
        total_chunks_ptr,
        scale,
        num_sms=num_sms,
    )
    return output.unsqueeze(0), final_state


__all__ = [
    "chunk_gated_delta_rule_cutedsl",
    "prepare_metadata_cutedsl",
]
