from __future__ import annotations

import math
from copy import copy

import torch



def _packed_cache(rows: int, device: torch.device, block_size: int = 64) -> torch.Tensor:
    blocks = max(1, (int(rows) + block_size - 1) // block_size)
    block_stride = ((block_size * 584 + 575) // 576) * 576
    return torch.empty((blocks, block_stride), dtype=torch.uint8, device=device)


def _dequantize_packed_cache(
    cache: torch.Tensor, rows: int, *, block_size: int = 64
) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
    )

    output = torch.empty((1, rows, 512), dtype=torch.bfloat16, device=cache.device)
    if rows:
        blocks = (rows + block_size - 1) // block_size
        dequantize_and_gather_k_cache_triton(
            output,
            cache,
            torch.tensor([rows], dtype=torch.int32, device=cache.device),
            None,
            torch.arange(blocks, dtype=torch.int32, device=cache.device).unsqueeze(0),
            block_size,
            0,
        )
    return output.squeeze(0)


def quantized_main_k_visible(functional_k: torch.Tensor) -> torch.Tensor:
    from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache

    rows = functional_k.shape[0]
    if rows == 0:
        return functional_k
    cache = _packed_cache(rows, functional_k.device)
    slots = torch.arange(rows, dtype=torch.int64, device=functional_k.device)
    quantize_and_insert_k_cache(functional_k.detach().contiguous(), cache, slots, block_size=64)
    visible = _dequantize_packed_cache(cache, rows)
    return functional_k + (visible - functional_k).detach()


def official_compact_compressed_visible(
    functional_k: torch.Tensor,
    compact_score: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    compressed_group_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    operation,
    runtime_metadata,
    ratio: int,
    head_dim: int,
) -> torch.Tensor:
    """Use vLLM's official compressor value on MCore's compact CP groups.

    Compact groups are represented as one synthetic request per packed
    sequence segment.  Only compressor-boundary RoPE rows are observed, so map
    each synthetic group boundary to its original per-sequence compressed
    position.  The ordinary differentiable compact graph remains the sole VJP
    owner.
    """
    groups = compressed_group_ids.numel()
    if groups == 0:
        return functional_k
    # The official overlap compressor carries state within one request.  A
    # packed CP compact buffer is request-major and advertises each request
    # boundary by resetting its group id to zero.  Launch each segment as an
    # independent synthetic request so the official kernel performs its native
    # state reset; mutating scores at the boundary would also alter the final
    # valid group of the preceding request.
    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=compact_score.device),
            torch.nonzero(compressed_group_ids[1:] == 0, as_tuple=False)
            .flatten()
            .add(1),
        )
    )
    ends = torch.cat(
        (starts[1:], torch.tensor([groups], device=compact_score.device))
    )
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        dequantize_and_gather_k_cache_triton,
    )

    block_size = runtime_metadata.k_cache.shape[1]
    visible_parts = []
    for group_start, group_end in zip(starts.tolist(), ends.tolist(), strict=True):
        segment_groups = group_end - group_start
        segment_tokens = segment_groups * ratio
        segment = copy(runtime_metadata)
        segment.state_slot_mapping = runtime_metadata.state_slot_mapping[:segment_tokens]
        segment.token_to_req_indices = runtime_metadata.token_to_req_indices[:segment_tokens]
        segment.k_slot_mapping = runtime_metadata.k_slot_mapping[:segment_tokens]
        segment.state_cache.zero_()
        synthetic_positions = torch.arange(
            segment_tokens, dtype=torch.int64, device=compact_score.device
        )
        synthetic_starts = torch.arange(
            0, segment_tokens, ratio, dtype=torch.int64, device=compact_score.device
        )
        source_positions = (
            compressed_group_ids[group_start:group_end].clamp_min(0).long() * ratio
        )
        segment.cos_sin_cache.index_copy_(
            0,
            synthetic_starts,
            cos_sin_cache.index_select(0, source_positions),
        )
        operation(
            kv_score=compact_score[
                group_start * ratio : group_end * ratio
            ].detach(),
            positions=synthetic_positions,
            ape=ape.detach(),
            norm_weight=norm_weight.detach(),
            compress_ratio=ratio,
            head_dim=head_dim,
            metadata=segment,
        )
        output = torch.empty(
            (1, segment_groups, head_dim),
            dtype=torch.bfloat16,
            device=compact_score.device,
        )
        blocks = (segment_groups + block_size - 1) // block_size
        dequantize_and_gather_k_cache_triton(
            output,
            segment.k_cache,
            torch.tensor(
                [segment_groups], dtype=torch.int32, device=compact_score.device
            ),
            None,
            torch.arange(blocks, dtype=torch.int32, device=compact_score.device).unsqueeze(0),
            block_size,
            0,
        )
        visible_parts.append(output.squeeze(0))
    visible = torch.cat(visible_parts)
    return functional_k + (visible - functional_k).detach()


def official_local_qk_visible(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_insert,
    functional_rope,
    *,
    eps: float,
    rope_dim: int,
    padded_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = q.shape[0]
    cache = _packed_cache(rows, q.device)
    slots = torch.arange(rows, dtype=torch.int64, device=q.device)
    q_visible = kv_insert(
        q.detach(),
        kv.detach(),
        cache,
        slots,
        positions,
        cos_sin_cache,
        eps=eps,
        block_size=64,
        padded_heads=padded_heads,
    ).contiguous()
    k_visible = _dequantize_packed_cache(cache, rows)
    q_graph = functional_rope(
        q, positions, cos_sin_cache, rope_dim, eps, normalize=True
    )
    k_graph = functional_rope(
        kv, positions, cos_sin_cache, rope_dim, eps, normalize=False
    )
    return (
        q_graph + (q_visible - q_graph).detach(),
        k_graph + (k_visible - k_graph).detach(),
    )


def official_indexer_topk(
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    index_k_seq_major: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    *,
    global_start: int,
    ratio: int,
    topk: int,
) -> torch.Tensor:
    from vllm import _custom_ops as ops
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

    rows = index_q.shape[0]
    output = torch.full((rows, topk), -1, dtype=torch.int32, device=index_q.device)
    q_quant, weights = fused_indexer_q_rope_quant(
        positions,
        index_q.detach(),
        cos_sin_cache,
        index_weights.detach(),
        index_q.shape[-1] ** -0.5,
        index_q.shape[1] ** -0.5,
        use_fp4=False,
    )
    k_quant, k_scale = per_token_group_quant_fp8(
        index_k_seq_major.detach().contiguous(),
        group_size=index_k_seq_major.shape[-1],
        use_ue8m0=True,
    )
    global_rows = torch.arange(
        global_start, global_start + rows, dtype=torch.int32, device=index_q.device
    )
    seq_ids = torch.bucketize(
        global_rows,
        cu_seqlens[1:],
        out_int32=True,
        right=True,
    ).clamp_max(cu_seqlens.shape[0] - 2)
    row_starts = cu_seqlens_compressed[seq_ids]
    row_ends = row_starts + torch.div(positions + 1, ratio, rounding_mode="floor").to(
        row_starts.dtype
    )
    if index_k_seq_major.shape[0] == 0:
        return output
    logits = fp8_fp4_mqa_logits(
        (q_quant, None),
        (k_quant, k_scale.view(torch.float32).squeeze(-1)),
        weights,
        row_starts.contiguous(),
        row_ends.contiguous(),
        clean_logits=False,
    )
    ops.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        output,
        rows,
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    return output


def c128_all_visible_topk(
    positions: torch.Tensor,
    *,
    width: int,
    ratio: int,
) -> torch.Tensor:
    columns = torch.arange(width, dtype=torch.int32, device=positions.device)
    counts = torch.div(positions + 1, ratio, rounding_mode="floor").to(torch.int32)
    return torch.where(
        columns.unsqueeze(0) < counts.unsqueeze(1),
        columns.unsqueeze(0),
        torch.full((), -1, dtype=torch.int32, device=positions.device),
    )


def compressed_width(max_seqlen: int, ratio: int, index_topk: int) -> int:
    if ratio == 4:
        return int(index_topk)
    rows = max(1, max_seqlen // max(1, ratio))
    return max(128, math.ceil(rows / 128) * 128)
