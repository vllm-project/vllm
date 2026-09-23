# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder prefill attention through the zentorch SDPA kernel on Zen CPUs.

Decode (query_len <= reorder threshold) stays on cpu_attention_with_kv_cache.
Equal-length fresh prefills use zentorch_sdpa.out(is_causal=True). Ragged
fresh prefills and chunked extends use one padded call with a bottom-right
causal mask; mixed batches run that path for the prefill suffix only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.v1.attention.backend import AttentionType

if TYPE_CHECKING:
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata

__all__ = [
    "should_use_zentorch_prefill_sdpa",
    "zentorch_prefill_sdpa",
]


def should_use_zentorch_prefill_sdpa(
    attn_type: str,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    sinks: torch.Tensor | None,
    is_fp8_kv_cache: bool,
    kv_sharing_target_layer_name: str | None,
    dtype: torch.dtype,
) -> bool:
    """Whether a decoder layer's prefill should run on zentorch_sdpa.

    The op attends dense Q/K/V with no paged cache, softcap, or sinks. ALiBi
    and sliding-window would need a per-sequence mask that collides with the
    suffix causal mask used for extends. Cross-attention and encoder layers
    stay on the native kernel.

    The dtype check mirrors the ISA gate inside zentorch_sdpa, below which the
    op falls back to ATen flash attention and would only add wrapper cost.

    Returns:
        True when this layer should dispatch prefill to zentorch_sdpa.
    """
    if attn_type != AttentionType.DECODER:
        return False
    if alibi_slopes is not None or sliding_window not in (None, -1):
        return False
    if logits_soft_cap or sinks is not None:
        return False
    if is_fp8_kv_cache or kv_sharing_target_layer_name is not None:
        return False
    if not has_zentorch_op(["zentorch_sdpa"]):
        return False
    if dtype == torch.bfloat16:
        return torch.cpu._is_avx512_bf16_supported()
    if dtype == torch.float32:
        return torch.cpu._is_avx512_supported()
    return False


def _batched_suffix_causal_attn_mask(
    query_lens: list[int],
    seq_lens: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive [B, 1, max_q, max_k] mask; padded query rows attend key 0."""
    query_lens_t = torch.tensor(query_lens, device=device)
    seq_lens_t = torch.tensor(seq_lens, device=device)
    max_q = int(query_lens_t.max())
    max_k = int(seq_lens_t.max())
    q_idx = torch.arange(max_q, device=device).view(1, max_q, 1)
    k_idx = torch.arange(max_k, device=device).view(1, 1, max_k)
    query_len = query_lens_t.view(-1, 1, 1)
    seq_len = seq_lens_t.view(-1, 1, 1)
    keep = (
        (q_idx < query_len)
        & (k_idx < seq_len)
        & (k_idx <= seq_len - query_len + q_idx)
    )
    keep = torch.where(q_idx >= query_len, k_idx == 0, keep)
    return torch.log(keep.to(dtype)).unsqueeze(1)


def _copy_packed_to_padded_bhsd(
    packed: torch.Tensor, start_loc: list[int], max_len: int
) -> torch.Tensor:
    """[tokens, H, D] packed requests -> padded [B, H, max_len, D]."""
    batch = len(start_loc) - 1
    padded = packed.new_zeros(batch, packed.size(1), max_len, packed.size(2))
    for i, (start, end) in enumerate(
        zip(start_loc[:-1], start_loc[1:], strict=True)
    ):
        length = end - start
        if length:
            padded[i, :, :length] = packed[start:end].movedim(0, 1)
    return padded


def _copy_padded_bhsd_to_packed(
    padded: torch.Tensor, packed: torch.Tensor, start_loc: list[int]
) -> None:
    """Write the unpadded rows of [B, H, max_len, D] back into [tokens, H, D]."""
    for i, (start, end) in enumerate(
        zip(start_loc[:-1], start_loc[1:], strict=True)
    ):
        length = end - start
        if length:
            packed[start:end] = padded[i, :, :length].movedim(0, 1)


def _gather_paged_kv_batched(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Read cached sequences into dense [B, H, max_seq_len, D]."""
    block_size = cache.size(2)
    max_blocks = -(-max_seq_len // block_size)
    if block_table.size(1) < max_blocks:
        # Unused slots gather block 0; those positions are masked by the caller.
        pad = block_table.new_zeros(
            block_table.size(0), max_blocks - block_table.size(1)
        )
        block_ids = torch.cat([block_table, pad], dim=1)
    else:
        block_ids = block_table[:, :max_blocks]
    tokens = cache[block_ids].permute(0, 2, 1, 3, 4).flatten(2, 3)
    return tokens[:, :, :max_seq_len]


def _uniform_fresh_prefill_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    num_reqs: int,
    seq_len: int,
    scale: float,
) -> None:
    def _packed_to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unflatten(0, (num_reqs, seq_len)).permute(0, 2, 1, 3)

    torch.ops.zentorch.zentorch_sdpa.out(
        _packed_to_bhsd(query),
        _packed_to_bhsd(key),
        _packed_to_bhsd(value),
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
        out=_packed_to_bhsd(output),
    )


def _padded_prefill_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    start_loc: list[int],
    query_lens: list[int],
    seq_lens: list[int],
    num_extend_reqs: int,
    block_table: torch.Tensor,
    scale: float,
) -> None:
    max_q = max(query_lens)
    max_k = max(seq_lens)
    query_pad = _copy_packed_to_padded_bhsd(query, start_loc, max_q)
    output_pad = query_pad.new_empty(query_pad.shape)
    key_pad = key.new_zeros(len(query_lens), key.size(1), max_k, key.size(2))
    value_pad = value.new_zeros(
        len(query_lens), value.size(1), max_k, value.size(2)
    )

    if num_extend_reqs:
        # Extends attend cached K/V, not only the packed chunk from this step.
        extend_max_k = max(seq_lens[:num_extend_reqs])
        gathered_key = _gather_paged_kv_batched(
            key_cache, block_table[:num_extend_reqs], extend_max_k
        )
        gathered_value = _gather_paged_kv_batched(
            value_cache, block_table[:num_extend_reqs], extend_max_k
        )
        key_pad[:num_extend_reqs, :, :extend_max_k] = gathered_key
        value_pad[:num_extend_reqs, :, :extend_max_k] = gathered_value

    if num_extend_reqs < len(query_lens):
        fresh_token0 = start_loc[num_extend_reqs]
        fresh_loc = [
            offset - fresh_token0 for offset in start_loc[num_extend_reqs:]
        ]
        key_pad[num_extend_reqs:] = _copy_packed_to_padded_bhsd(
            key[fresh_token0:], fresh_loc, max_k
        )
        value_pad[num_extend_reqs:] = _copy_packed_to_padded_bhsd(
            value[fresh_token0:], fresh_loc, max_k
        )

    torch.ops.zentorch.zentorch_sdpa.out(
        query_pad,
        key_pad,
        value_pad,
        dropout_p=0.0,
        is_causal=False,
        attn_mask=_batched_suffix_causal_attn_mask(
            query_lens, seq_lens, query.dtype, query.device
        ),
        scale=scale,
        out=output_pad,
    )
    _copy_padded_bhsd_to_packed(output_pad, output, start_loc)


def zentorch_prefill_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: CPUAttentionMetadata,
    scale: float,
) -> None:
    """Run decoder prefill / extend attention with zentorch_sdpa.

    Tokens before ``num_native_tokens`` are decode (or spec-decode) and are
    left for cpu_attention_with_kv_cache. This fills the prefill suffix of
    ``output`` in place.

    Args:
        query: Query of shape [num_tokens, num_heads, head_size].
        key: Key of shape [num_tokens, num_kv_heads, head_size].
        value: Value of shape [num_tokens, num_kv_heads, head_size].
        output: Pre-allocated output, same shape as ``query``.
        key_cache: Paged key cache [num_blocks, H_kv, block_size, D].
        value_cache: Paged value cache, same shape as ``key_cache``.
        attn_metadata: Metadata with the prefill split recorded by the builder.
        scale: Softmax scale.
    """
    query_start_loc = attn_metadata.prefill_query_start_loc
    if query_start_loc is None or query_start_loc.numel() < 2:
        return

    start_loc = [int(offset) for offset in query_start_loc.tolist()]
    query_lens = [
        end - start
        for start, end in zip(start_loc[:-1], start_loc[1:], strict=True)
    ]
    if not query_lens:
        return

    num_extend_reqs = attn_metadata.num_extend_reqs
    first_req = attn_metadata.num_native_reqs
    first_token = attn_metadata.num_native_tokens
    end_token = first_token + start_loc[-1]
    prefill_query = query[first_token:end_token]
    prefill_key = key[first_token:end_token]
    prefill_value = value[first_token:end_token]
    prefill_output = output[first_token:end_token]
    num_prefill_reqs = len(query_lens)
    seq_lens = [
        int(length) for length in attn_metadata.seq_lens[first_req:].tolist()
    ][:num_prefill_reqs]

    all_fresh = num_extend_reqs == 0 and query_lens == seq_lens
    if all_fresh and len(set(query_lens)) == 1:
        _uniform_fresh_prefill_sdpa(
            prefill_query,
            prefill_key,
            prefill_value,
            prefill_output,
            num_prefill_reqs,
            query_lens[0],
            scale,
        )
        return

    _padded_prefill_sdpa(
        prefill_query,
        prefill_key,
        prefill_value,
        prefill_output,
        key_cache,
        value_cache,
        start_loc,
        query_lens,
        seq_lens,
        num_extend_reqs,
        attn_metadata.block_table[first_req : first_req + num_prefill_reqs],
        scale,
    )
