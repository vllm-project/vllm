# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the glm5next sparse attention indexer (kpool) layers."""

import torch

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _build_decode_gather_indices(
    decode_lens: torch.Tensor, lmax: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source indices and validity mask for a padded [requests, lmax] grid.

    ``n`` is the number of input token rows, including graph padding.
    Reused across token tensors in a step. Graph-padding input rows are ignored.
    """
    dl = decode_lens.to(torch.int64)
    starts = dl.cumsum(0) - dl
    columns = torch.arange(lmax, device=dl.device)
    indices = starts[:, None] + columns[None, :]
    valid = columns[None, :] < dl[:, None]
    # Invalid cells still need in-bounds indices before gathering.
    indices = indices.clamp(max=max(n - 1, 0))
    return indices, valid


def _gather_decode_tokens_by_request(
    tokens: torch.Tensor,
    pad_value: int | float,
    gather_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Group [N, ...] tokens by request, filling short or empty rows with padding."""
    indices, valid = gather_indices
    if tokens.shape[0] == 0:
        return tokens.new_full((*indices.shape, *tokens.shape[1:]), pad_value)
    mask = valid.reshape(*valid.shape, *([1] * (tokens.ndim - 1)))
    return torch.where(mask, tokens[indices], pad_value)


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
    gather_idx = _build_decode_gather_indices(decode_lens, next_n, num_decode_tokens)
    padded = _gather_decode_tokens_by_request(
        positions[:num_decode_tokens].to(torch.int32), -1, gather_idx
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
