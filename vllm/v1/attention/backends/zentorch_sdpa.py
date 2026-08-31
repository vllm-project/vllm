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
    """Whether an attention layer should run on zentorch_sdpa.

    The op attends a whole sequence bidirectionally with no attention bias, so
    it only covers plain encoder attention: ALiBi and sliding-window layers
    would need a materialized mask per sequence to reproduce what the native
    kernel already does from its own arguments.

    The dtype check mirrors the ISA gate inside zentorch_sdpa, below which the
    op falls back to the ATen flash kernel the native backend would have run
    anyway, so dispatching to it would only add wrapper cost. float16 needs
    AVX512-FP16, which torch exposes no query for, so it stays native.

    Returns:
        True when the layer's attention should dispatch to zentorch_sdpa.
    """
    if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return False
    if alibi_slopes is not None or sliding_window not in (None, -1):
        return False
    if not has_zentorch_op(["zentorch_sdpa"]):
        return False
    if dtype == torch.bfloat16:
        return torch.cpu._is_avx512_bf16_supported()
    if dtype == torch.float32:
        return torch.cpu._is_avx512_supported()
    return False


def zentorch_encoder_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: CPUAttentionMetadata,
    scale: float,
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
            scale=scale,
            out=output[start_q:end_q, :, :].movedim(0, 1).unsqueeze(0),
        )
    return output
