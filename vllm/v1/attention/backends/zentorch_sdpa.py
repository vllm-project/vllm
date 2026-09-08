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

__all__ = ["should_use_zentorch_sdpa", "zentorch_sdpa_attn"]


def should_use_zentorch_sdpa(
    attn_type: str,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int | None,
    dtype: torch.dtype,
) -> bool:
    """True when encoder attention should dispatch to zentorch_sdpa.

    Full, sliding-window, and ALiBi encoder layers all dispatch. A window is
    a single-head [1, 1, S, S] additive mask; ALiBi is a per-query-head
    [1, H, S, S] bias. The zentorch_sdpa_attn op builds both.

    dtype must match the op's ISA gate (bf16: AVX512-BF16, fp32: AVX512);
    fp16 needs AVX512-FP16, which torch exposes no query for, so it stays
    native.
    """
    if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return False
    if not has_zentorch_op(["zentorch_sdpa_attn"]):
        return False
    if dtype == torch.bfloat16:
        return torch.cpu._is_avx512_bf16_supported()
    if dtype == torch.float32:
        return torch.cpu._is_avx512_supported()
    return False


def zentorch_sdpa_attn(
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

    Encoder attention reads no KV cache, so the packed
    [num_tokens, num_heads, head_size] query/key/value are attended in place
    and written straight into `output`.

    Args:
        query: Query of shape [num_tokens, num_heads, head_size].
        key: Key of shape [num_tokens, num_kv_heads, head_size].
        value: Value of shape [num_tokens, num_kv_heads, head_size].
        output: Pre-allocated output, same shape as `query`.
        attn_metadata: Metadata carrying the per-sequence token offsets and
            whether attention is causal.
        scale: Softmax scale.
        sliding_window: Encoder window size, or -1 for full bidirectional
            attention. Handed to the op, which masks token i to
            [i-(W-1), i+(W-1)].
        alibi_slopes: Per-query-head ALiBi slopes, or None. Handed to the op,
            which applies them as an additive per-head bias.

    Returns:
        `output`, filled in place.
    """
    # The op counts window tokens on each side of the diagonal; vLLM's window
    # includes the diagonal itself.
    window = -1 if sliding_window in (None, -1) else sliding_window - 1

    # zentorch_sdpa takes no enable_gqa argument but derives the KV head count
    # from the key tensor and maps each query head onto its KV head, so GQA/MQA
    # needs no expansion here.
    query = query.movedim(0, query.dim() - 2)
    key = key.movedim(0, key.dim() - 2)
    value = value.movedim(0, value.dim() - 2)

    start_loc = attn_metadata.query_start_loc.numpy()
    seq_lens = start_loc[1:] - start_loc[:-1]

    # Encoder sequences are packed along the token dimension. When they all
    # have the same length, recover a dense BHSD batch and invoke zentorch_sdpa
    # once for the whole scheduler batch instead of once per sequence.
    if len(seq_lens) > 0 and (seq_lens == seq_lens[0]).all():
        batch_size = len(seq_lens)
        seq_len = int(seq_lens[0])

        def _packed_hsd_to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unflatten(1, (batch_size, seq_len)).permute(1, 0, 2, 3)

        torch.ops.zentorch.zentorch_sdpa_attn(
            _packed_hsd_to_bhsd(query),
            _packed_hsd_to_bhsd(key),
            _packed_hsd_to_bhsd(value),
            # The op writes [B, H, S, D]; hand it the matching view of the
            # packed buffer so it stores there instead of into a temporary.
            output.unflatten(0, (batch_size, seq_len)).permute(0, 2, 1, 3),
            scale=scale,
            is_causal=attn_metadata.causal,
            alibi_slopes=alibi_slopes,
            left_window_size=window,
            right_window_size=window,
        )
        return output

    for start_q, end_q in zip(start_loc[:-1], start_loc[1:], strict=True):
        torch.ops.zentorch.zentorch_sdpa_attn(
            query[None, :, start_q:end_q, :],
            key[None, :, start_q:end_q, :],
            value[None, :, start_q:end_q, :],
            output[start_q:end_q, :, :].movedim(0, 1).unsqueeze(0),
            scale=scale,
            is_causal=attn_metadata.causal,
            alibi_slopes=alibi_slopes,
            left_window_size=window,
            right_window_size=window,
        )
    return output
