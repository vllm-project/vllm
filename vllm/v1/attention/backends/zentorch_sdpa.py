# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder attention through the zentorch SDPA kernel on Zen CPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.v1.attention.backend import AttentionType

if TYPE_CHECKING:
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata

__all__ = ["should_use_zentorch_sdpa", "zentorch_encoder_sdpa"]


def should_use_zentorch_sdpa(
    attn_type: str,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int | None,
    dtype: torch.dtype,
) -> bool:
    """True when encoder attention should dispatch to zentorch_sdpa.

    Full, sliding-window, and ALiBi encoder layers all dispatch. A window is
    a single-head [1, 1, S, S] additive mask; ALiBi is a per-query-head
    [1, H, S, S] bias. Both are cached per sequence length in a call.

    dtype must match the op's ISA gate (bf16: AVX512-BF16, fp32: AVX512);
    fp16 needs AVX512-FP16, which torch exposes no query for, so it stays
    native.
    """
    if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return False
    if not has_zentorch_op(["zentorch_sdpa"]):
        return False
    if dtype == torch.bfloat16:
        return torch.cpu._is_avx512_bf16_supported()
    if dtype == torch.float32:
        return torch.cpu._is_avx512_supported()
    return False


def _sliding_window_attn_mask(
    seq_len: int, window: int, dtype: torch.dtype
) -> torch.Tensor:
    """Additive [1, 1, S, S] mask for bidirectional encoder sliding window.

    Token i attends [i-(W-1), i+(W-1)], matching native cpu_attn and the
    zentorch plugin (left = right = W-1). Entries inside the band are 0;
    outside they are -inf. Rank is 4 because zentorch_sdpa rejects 3D masks.
    """
    band = window - 1
    mask = torch.ones((seq_len, seq_len), dtype=dtype)
    mask = torch.tril(mask, diagonal=band)
    mask = torch.triu(mask, diagonal=-band)
    return torch.log(mask).view(1, 1, seq_len, seq_len)


def _alibi_attn_mask(
    seq_len: int, alibi_slopes: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Additive [1, H, S, S] ALiBi bias: slope * (k_pos - q_pos).

    Matches native cpu_attn apply_alibi_slopes. Encoder attention is
    bidirectional, so this does not apply a causal -inf triangle.
    """
    pos = torch.arange(seq_len, dtype=dtype)
    dist = pos[None, :] - pos[:, None]
    bias = dist[None, :, :] * alibi_slopes.to(dtype=dtype)[:, None, None]
    return bias.unsqueeze(0)


def _encoder_attn_mask(
    seq_len: int,
    dtype: torch.dtype,
    sliding_window: int,
    alibi_slopes: torch.Tensor | None,
) -> torch.Tensor | None:
    mask: torch.Tensor | None = None
    if alibi_slopes is not None:
        mask = _alibi_attn_mask(seq_len, alibi_slopes, dtype)
    if sliding_window not in (None, -1):
        window = _sliding_window_attn_mask(seq_len, sliding_window, dtype)
        mask = window if mask is None else mask + window
    return mask


def zentorch_encoder_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: CPUAttentionMetadata,
    scale: float,
    sliding_window: int = -1,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run encoder / encoder-only attention with zentorch_sdpa.

    Encoder attention is bidirectional and reads no KV cache, so the packed
    [num_tokens, num_heads, head_size] query/key/value are attended in place
    and written straight into `output`.

    Args:
        query: Query of shape [num_tokens, num_heads, head_size].
        key: Key of shape [num_tokens, num_kv_heads, head_size].
        value: Value of shape [num_tokens, num_kv_heads, head_size].
        output: Pre-allocated output, same shape as `query`.
        attn_metadata: Metadata carrying the per-sequence token offsets.
        scale: Softmax scale.
        sliding_window: Encoder window size, or -1 for full bidirectional
            attention. Applied as an additive mask into zentorch_sdpa.
        alibi_slopes: Per-query-head ALiBi slopes, or None. Applied as an
            additive per-head bias into zentorch_sdpa.

    Returns:
        `output`, filled in place.
    """
    # zentorch_sdpa takes no enable_gqa argument but derives the KV head count
    # from the key tensor and maps each query head onto its KV head, so GQA/MQA
    # needs no expansion here.
    query = query.movedim(0, query.dim() - 2)
    key = key.movedim(0, key.dim() - 2)
    value = value.movedim(0, value.dim() - 2)

    start_loc = attn_metadata.query_start_loc.numpy()
    seq_lens = start_loc[1:] - start_loc[:-1]
    mask_cache: dict[int, torch.Tensor | None] = {}

    def _attn_mask(seq_len: int) -> torch.Tensor | None:
        if seq_len not in mask_cache:
            mask_cache[seq_len] = _encoder_attn_mask(
                seq_len, query.dtype, sliding_window, alibi_slopes
            )
        return mask_cache[seq_len]

    # Encoder sequences are packed along the token dimension. When they all
    # have the same length, recover a dense BHSD batch and invoke zentorch_sdpa
    # once for the whole scheduler batch instead of once per sequence.
    if len(seq_lens) > 0 and (seq_lens == seq_lens[0]).all():
        batch_size = len(seq_lens)
        seq_len = int(seq_lens[0])

        def _packed_hsd_to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unflatten(1, (batch_size, seq_len)).permute(1, 0, 2, 3)

        torch.ops.zentorch.zentorch_sdpa.out(
            _packed_hsd_to_bhsd(query),
            _packed_hsd_to_bhsd(key),
            _packed_hsd_to_bhsd(value),
            dropout_p=0.0,
            is_causal=False,
            attn_mask=_attn_mask(seq_len),
            scale=scale,
            # The op writes [B, H, S, D]; hand it the matching view of the
            # packed buffer so it stores there instead of into a temporary.
            out=output.unflatten(0, (batch_size, seq_len)).permute(0, 2, 1, 3),
        )
        return output

    for start_q, end_q in zip(start_loc[:-1], start_loc[1:], strict=True):
        torch.ops.zentorch.zentorch_sdpa.out(
            query[None, :, start_q:end_q, :],
            key[None, :, start_q:end_q, :],
            value[None, :, start_q:end_q, :],
            dropout_p=0.0,
            is_causal=False,
            attn_mask=_attn_mask(int(end_q - start_q)),
            scale=scale,
            out=output[start_q:end_q, :, :].movedim(0, 1).unsqueeze(0),
        )
    return output
