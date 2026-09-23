# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder attention and decoder prefill through zentorch SDPA on Zen CPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.linear.zentorch_utils import has_zentorch_op
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata

__all__ = [
    "should_use_zentorch_prefill_sdpa",
    "should_use_zentorch_sdpa",
    "zentorch_prefill_sdpa",
    "zentorch_sdpa_attn",
]

# The CPU cache writer stores an fp8 KV cache as uint8 holding `fp8(x / scale)`.
_FP8_KV_CACHE_DTYPES = {
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def _supports_dtype(dtype: torch.dtype) -> bool:
    """Whether the zentorch SDPA kernels run `dtype` on this CPU.

    dtype must match the op's ISA gate (bf16: AVX512-BF16, fp32: AVX512);
    fp16 needs AVX512-FP16, which torch exposes no query for, so it stays
    native.
    """
    if dtype == torch.bfloat16:
        return torch.cpu._is_avx512_bf16_supported()
    if dtype == torch.float32:
        return torch.cpu._is_avx512_supported()
    return False


def should_use_zentorch_sdpa(
    attn_type: str,
    dtype: torch.dtype,
) -> bool:
    """True when encoder attention should dispatch to zentorch_sdpa.

    Full, sliding-window, and ALiBi encoder layers all dispatch. A window is
    a single-head [1, 1, S, S] additive mask; ALiBi is a per-query-head
    [1, H, S, S] bias. The zentorch_sdpa_attn op builds both.
    """
    if attn_type not in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        return False
    if not has_zentorch_op(["zentorch_sdpa_attn"]):
        return False
    return _supports_dtype(dtype)


def should_use_zentorch_prefill_sdpa(
    attn_type: str,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None,
    logits_soft_cap: float | None,
    sinks: torch.Tensor | None,
    kv_sharing_target_layer_name: str | None,
) -> bool:
    """True when decoder prefill should dispatch to zentorch_sdpa.out.

    Only plain causal self-attention over an unquantized or per-tensor fp8 KV
    cache is routed; decode stays on the paged CPU kernel.
    """
    if (
        attn_type != AttentionType.DECODER
        or alibi_slopes is not None
        or logits_soft_cap
        or sinks is not None
        # KV-shared layers (Gemma) project Q only and receive no key/value.
        or kv_sharing_target_layer_name is not None
        or (
            is_quantized_kv_cache(kv_cache_dtype)
            and kv_cache_dtype not in _FP8_KV_CACHE_DTYPES
        )
    ):
        return False
    return (
        has_zentorch_op(["zentorch_sdpa"])
        and hasattr(torch.ops.zentorch.zentorch_sdpa, "out")
        and _supports_dtype(dtype)
    )


def _fp8_round_trip(
    tensor: torch.Tensor, fp8_dtype: torch.dtype, scale: float
) -> torch.Tensor:
    """`tensor` as the CPU cache writer stores it and the paged kernel reads it.

    The writer multiplies by the fp32 reciprocal of `scale`, rounds to nearest
    even and saturates at the fp8 maximum.
    """
    inv_scale = torch.tensor(scale, dtype=torch.float32).reciprocal()
    limit = torch.finfo(fp8_dtype).max
    scaled = (tensor.float() * inv_scale).clamp_(-limit, limit)
    return scaled.to(fp8_dtype).to(tensor.dtype).mul_(scale)


def _gather_cached_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_lens: list[int],
    seq_lens: list[int],
    window: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pack each request's cached keys/values, current chunk included.

    A request keeps only its last `window - 1` context tokens plus its query
    tokens when sliding-window attention is on. Returns the packed key, value
    and the `cu_seqlens_k` offsets.

    Keys are stored column-major, `[head_dim, block_size]` per head, by every
    non-AMX ISA; values are token-major, `[block_size, head_dim]`.
    """
    block_size = key_cache.size(2)
    keys, values, cu_seqlens_k = [], [], [0]
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens, strict=True)):
        begin = max(seq_len - query_len - window + 1, 0) if window else 0
        first_block = begin // block_size
        block_ids = block_table[i, first_block : (seq_len - 1) // block_size + 1]
        tokens = slice(
            begin - first_block * block_size, seq_len - first_block * block_size
        )

        key = key_cache.index_select(0, block_ids)
        num_blocks, num_heads, _, head_dim = key.shape
        key = key.reshape(num_blocks, num_heads, head_dim, block_size).transpose(2, 3)
        keys.append(key.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)[tokens])
        value = value_cache.index_select(0, block_ids)
        values.append(
            value.permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)[tokens]
        )
        cu_seqlens_k.append(cu_seqlens_k[-1] + seq_len - begin)
    return torch.cat(keys), torch.cat(values), cu_seqlens_k


def zentorch_prefill_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata: CPUAttentionMetadata,
    scale: float,
    sliding_window: int,
    isa: str,
    kv_cache_dtype: str,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> bool:
    """Run the prefill requests, which follow the decodes, with zentorch_sdpa.

    Fills the prefill rows of `output` and leaves the decode rows untouched.
    Fresh prompts attend the packed `key`/`value`; requests with cached
    context attend that context gathered from the KV cache, which already
    holds this step's tokens. With an fp8 cache, the fresh `key`/`value` get
    the cache writer's fp8 round trip and the gathered context is dequantized
    with `k_scale`/`v_scale`, so both match what the paged kernel attends.

    Returns:
        False when the batch must run on the native kernel instead: cached
        context under AMX, whose cache layout is tiled.

    """
    first_req = attn_metadata.num_decodes
    first_token = attn_metadata.num_decode_tokens
    cu_seqlens_q = [
        offset - first_token
        for offset in attn_metadata.query_start_loc[first_req:].tolist()
    ]
    seq_lens = attn_metadata.seq_lens[first_req:].tolist()
    query_lens = [end - start for start, end in zip(cu_seqlens_q, cu_seqlens_q[1:])]
    window = max(sliding_window, 0)

    fp8_dtype = _FP8_KV_CACHE_DTYPES.get(kv_cache_dtype)

    cu_seqlens_k = None
    tokens = slice(first_token, first_token + cu_seqlens_q[-1])
    if query_lens == seq_lens:
        key, value = key[tokens], value[tokens]
        if fp8_dtype is not None:
            key = _fp8_round_trip(key, fp8_dtype, k_scale)
            value = _fp8_round_trip(value, fp8_dtype, v_scale)
    elif isa == "amx":
        return False
    else:
        if fp8_dtype is not None:
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)
        key, value, cu_seqlens_k = _gather_cached_kv(
            key_cache,
            value_cache,
            attn_metadata.block_table[first_req:],
            query_lens,
            seq_lens,
            window,
        )
        if fp8_dtype is not None:
            key = key.to(query.dtype).mul_(k_scale)
            value = value.to(query.dtype).mul_(v_scale)
    query, output = query[tokens], output[tokens]

    if cu_seqlens_k is None and len(set(seq_lens)) == 1:
        # Equal-length fresh prompts: one dense batched call.
        def _to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.unflatten(0, (len(seq_lens), seq_lens[0])).permute(0, 2, 1, 3)

        query, key, value, output = map(_to_bhsd, (query, key, value, output))
        cu_seqlens_q = None

    torch.ops.zentorch.zentorch_sdpa.out(
        query,
        key,
        value,
        is_causal=True,
        scale=scale,
        sliding_window_size=window,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        out=output,
    )
    return True


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
    # zentorch_sdpa_attn builds the ALiBi mask in query.dtype. A bf16 per-head
    # mask is wrong when the query head count is not a power of two, so run
    # that path in fp32 and write bf16 back.
    if alibi_slopes is not None and query.dtype == torch.bfloat16:
        out32 = torch.empty_like(query, dtype=torch.float32)
        zentorch_sdpa_attn(
            query.float(),
            key.float(),
            value.float(),
            out32,
            attn_metadata,
            scale,
            sliding_window,
            alibi_slopes,
        )
        output.copy_(out32.to(dtype=output.dtype))
        return output

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
