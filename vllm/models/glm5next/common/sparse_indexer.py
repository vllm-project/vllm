# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the glm5next sparse attention indexer (kpool) layers."""

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _gather_kpool_prefill_warmup_inputs(
    *, tail_cache, head_dim: int, kpool: int, dtype: torch.dtype
):
    tail = tail_cache.kv_cache
    return {
        **{
            name: TritonWarmupTensor(dtype)
            for name in ("k_ptr", "score_ptr", "out_k_ptr", "out_score_ptr")
        },
        **{
            name: TritonWarmupTensor(torch.int64)
            for name in ("slots_ptr", "pos_ptr", "tail_slots_ptr")
        },
        "tail_ptr": TritonWarmupTensor(
            tail.dtype, ptr_range_bytes=tail.untyped_storage().nbytes()
        ),
        "write_mask_ptr": TritonWarmupTensor(torch.bool),
        "k_stride_0": head_dim,
        "k_stride_1": 1,
        "s_stride_0": head_dim,
        "s_stride_1": 1,
        **{f"t_stride_{i}": stride for i, stride in enumerate(tail.stride())},
        "POOL": kpool,
        "DIM": head_dim,
        "RING": tail.shape[2],
        "BLOCK_D": triton.next_power_of_2(head_dim),
        "num_warps": 2,
        "grid": (1,),
    }


@triton_kernel_dispatcher_with_warmup(
    warmup_inputs=_gather_kpool_prefill_warmup_inputs,
)
@triton.jit(do_not_specialize_on_alignment=["slots_ptr", "pos_ptr", "tail_slots_ptr"])
def _gather_kpool_prefill_kernel(
    k_ptr,
    score_ptr,
    slots_ptr,
    pos_ptr,
    tail_ptr,
    tail_slots_ptr,
    out_k_ptr,
    out_score_ptr,
    write_mask_ptr,
    k_stride_0: tl.constexpr,
    k_stride_1: tl.constexpr,
    s_stride_0: tl.constexpr,
    s_stride_1: tl.constexpr,
    t_stride_0: tl.constexpr,
    t_stride_1: tl.constexpr,
    t_stride_2: tl.constexpr,
    t_stride_3: tl.constexpr,
    POOL: tl.constexpr,
    DIM: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather [N, POOL, DIM] keys/scores for writable completions only.

    Batch and tail buffers are read-only; masked output rows stay untouched.
    """
    row = tl.program_id(0).to(tl.int64)
    tail_slot = tl.load(tail_slots_ptr + row).to(tl.int64)
    valid = (tl.load(slots_ptr + row) >= 0) & (tail_slot >= 0)
    tl.store(write_mask_ptr + row, valid)
    if valid:
        members = tl.arange(0, POOL)
        dims = tl.arange(0, BLOCK_D)
        source = row - (POOL - 1) + members
        position = tl.load(pos_ptr + row) - (POOL - 1) + members
        block = tail_slot // RING
        source_pos = tl.load(pos_ptr + source, source >= 0, other=-1)
        source_slot = tl.load(tail_slots_ptr + source, source >= 0, other=-1).to(
            tl.int64
        )
        from_batch = (
            (source >= 0)
            & (source_slot >= 0)
            & (source_pos == position)
            & (source_slot // RING == block)
        )
        k = tl.load(
            k_ptr + source[:, None] * k_stride_0 + dims[None, :] * k_stride_1,
            from_batch[:, None] & (dims[None, :] < DIM),
            other=0,
        )
        s = tl.load(
            score_ptr + source[:, None] * s_stride_0 + dims[None, :] * s_stride_1,
            from_batch[:, None] & (dims[None, :] < DIM),
            other=0,
        )
        tail_offset = (
            block * t_stride_0
            + (position % RING)[:, None] * t_stride_2
            + dims[None, :] * t_stride_3
        )
        tail_k = tl.load(
            tail_ptr + tail_offset,
            ~from_batch[:, None] & (dims[None, :] < DIM),
            other=0,
        )
        tail_s = tl.load(
            tail_ptr + tail_offset + t_stride_1,
            ~from_batch[:, None] & (dims[None, :] < DIM),
            other=0,
        )
        output = (row * POOL + members[:, None]) * DIM + dims[None, :]
        tl.store(
            out_k_ptr + output,
            tl.where(from_batch[:, None], k, tail_k),
            dims[None, :] < DIM,
        )
        tl.store(
            out_score_ptr + output,
            tl.where(from_batch[:, None], s, tail_s),
            dims[None, :] < DIM,
        )


def _gather_kpool_prefill(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    positions: torch.Tensor | None,
    tail_kv_cache: torch.Tensor | None,
    tail_slot_mapping: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble complete pools from this batch and each request's prior tail."""
    n = slot_mapping.shape[0]
    if (
        positions is not None
        and tail_kv_cache is not None
        and tail_slot_mapping is not None
    ):
        keys = torch.empty((n, kpool, k.shape[1]), dtype=k.dtype, device=k.device)
        scores = torch.empty_like(keys, dtype=gate_score.dtype)
        write_mask = torch.empty(n, dtype=torch.bool, device=k.device)
        _gather_kpool_prefill_kernel[(n,)](
            k,
            gate_score,
            slot_mapping,
            positions,
            tail_kv_cache,
            tail_slot_mapping,
            keys,
            scores,
            write_mask,
            *k.stride(),
            *gate_score.stride(),
            *tail_kv_cache.stride(),
            POOL=kpool,
            DIM=k.shape[1],
            RING=tail_kv_cache.shape[2],
            BLOCK_D=triton.next_power_of_2(k.shape[1]),
            num_warps=2,
        )
        return keys, scores, write_mask
    rows = torch.arange(n, device=k.device)
    offsets = torch.arange(kpool, device=k.device)
    indices = (rows[:, None] - (kpool - 1) + offsets).clamp(0, n - 1)
    write_mask = (slot_mapping >= 0) & (rows >= kpool - 1)
    return k[indices], gate_score[indices], write_mask


def _build_decode_scatter_indices(
    decode_lens: torch.Tensor,
    num_requests: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (request id, intra-request index) for a non-uniform decode
    batch, with ``n == decode_lens.sum()`` as a host int (avoids a
    device sync and keeps both repeat_interleaves sync-free).

    Shared by every ``_scatter_decode_tokens_by_request`` call in a step:
    building it per call would repeat the same repeat_interleave/cumsum chain
    up to 5x per layer on the eager decode break.
    """
    device = decode_lens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64),
        dl,
        output_size=n,
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    # Broadcast the per-request start offsets to per-token (length n ==
    # dl.sum()) so each token's intra-request index subtracts its own
    # request's start.
    starts = torch.repeat_interleave(req_starts, dl, output_size=n)
    intra = torch.arange(n, device=device, dtype=torch.int64) - starts
    return req_id, intra


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    pad_value,
    num_requests: int,
    lmax: int,
    scatter_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Group ``[N, ...]`` decode tokens into a padded ``[num_requests, lmax, ...]``
    layout: request ``r``'s tokens at row ``r`` in order; short requests padded.

    Unlike ``pack_seq_triton`` this is dtype-agnostic (needed for the int32
    slot/pos tensors) — it scatters with the shared per-step indices from
    ``_build_decode_scatter_indices``. Used only for the non-uniform
    (``requires_padding``) decode batch; uniform batches use a zero-copy
    reshape.
    """
    req_id, intra = scatter_indices
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    out[req_id, intra] = tokens
    return out


def _decode_topk_seq_lens(
    positions: torch.Tensor,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    requires_padding: bool,
) -> torch.Tensor:
    """Token-granular seq_len (pos + 1) per pool-topk row, layout-aware.

    ``pool_topk`` (and the logits it comes from) follow the padded
    ``[batch_size, next_n]`` grid whenever ``requires_padding`` is set, so row
    ``(b, t)`` corresponds to flat decode token ``offset_b + t`` -- NOT
    ``b * next_n + t``. Slicing flat ``positions[: batch_size * next_n]``
    (the uniform-layout shortcut) misaligns every row after the first
    non-uniform request and, past the decode region, reads prefill tokens'
    positions; ``expand_pools_and_append_tail`` then anchors the tail at
    another request's length, dropping the row's real tail tokens or emitting
    indices past its sequence (out-of-bounds block-table reads). Padded rows
    get 0 (empty tail); they are dropped by ``unpack_seq_triton`` anyway.
    """
    n = batch_size * next_n
    if not requires_padding:
        return positions[:n].to(torch.int32) + 1
    scatter_idx = _build_decode_scatter_indices(
        decode_lens, batch_size, num_decode_tokens
    )
    padded = _scatter_decode_tokens_by_request(
        positions[:num_decode_tokens].to(torch.int32),
        -1,
        batch_size,
        next_n,
        scatter_idx,
    )
    return padded.reshape(n) + 1  # pad rows: -1 + 1 = 0 -> empty tail


def _fill_causal_indices(rows: torch.Tensor, positions: torch.Tensor) -> None:
    causal_range = torch.arange(rows.shape[1], device=rows.device, dtype=torch.int32)
    positions = positions.to(torch.int32)
    rows[:] = causal_range[None, :]
    rows[causal_range[None, :] > positions[:, None]] = -1


def _fill_short_decode_causal_indices(
    topk_indices_buffer: torch.Tensor,
    positions: torch.Tensor | None,
    num_decode_tokens: int,
    max_seq_len: int,
    topk_tokens: int,
) -> bool:
    """Fill exact causal rows when sparse decode would select every token."""
    if positions is None or positions.numel() == 0 or max_seq_len > topk_tokens:
        return False
    _fill_causal_indices(
        topk_indices_buffer[:num_decode_tokens], positions[:num_decode_tokens]
    )
    return True


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)
