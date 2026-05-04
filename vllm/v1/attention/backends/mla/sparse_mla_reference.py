# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference sparse MLA attention helpers.

The helpers in this module intentionally use PyTorch tensor operations. They
are the correctness-first contract for portable sparse MLA fallbacks and tests;
optimized Triton/CUDA kernels should preserve these semantics.
"""

import torch


def new_reference_attention_state(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.dim() == 4:
        q_bhd = q[:, 0, :, :].float()
    else:
        assert q.dim() == 3, f"Expected q shape [T, H, D], got {q.shape}"
        q_bhd = q.float()

    num_tokens = q_bhd.shape[0]
    num_heads = q_bhd.shape[1]
    head_dim = q_bhd.shape[2]
    max_score = torch.full(
        (num_tokens, num_heads),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    denom = torch.zeros_like(max_score)
    acc = torch.zeros(
        (num_tokens, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    return q_bhd, max_score, denom, acc


def accumulate_reference_attention_chunk(
    q_bhd: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_btd = kv.float()
    kv_btd = torch.where(
        valid_tokens[:, :, None],
        kv_btd,
        torch.zeros((), dtype=kv_btd.dtype, device=kv_btd.device),
    )
    scores = torch.einsum("bhd,btd->bht", q_bhd, kv_btd) * scale
    scores = scores.masked_fill(~valid_tokens[:, None, :], float("-inf"))

    chunk_max = scores.amax(dim=-1)
    next_max = torch.maximum(max_score, chunk_max)

    previous_scale = torch.exp(max_score - next_max)
    previous_scale = torch.nan_to_num(previous_scale)
    weights = torch.exp(scores - next_max[:, :, None])
    weights = torch.where(
        valid_tokens[:, None, :],
        weights,
        torch.zeros((), dtype=weights.dtype, device=weights.device),
    )
    weights = torch.nan_to_num(weights)

    acc = acc * previous_scale[:, :, None]
    denom = denom * previous_scale
    acc = acc + torch.einsum("bht,btd->bhd", weights, kv_btd)
    denom = denom + weights.sum(dim=-1)
    return next_max, denom, acc


def finish_reference_attention_no_sink(
    max_score: torch.Tensor,
    denom: torch.Tensor,
    acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = denom > 0
    safe_denom = torch.where(valid, denom, torch.ones_like(denom))
    subset_output = acc / safe_denom[:, :, None]
    subset_output = torch.where(
        valid[:, :, None],
        subset_output,
        torch.zeros((), dtype=subset_output.dtype, device=subset_output.device),
    )
    subset_lse = torch.where(
        valid,
        max_score + torch.log(safe_denom),
        torch.full_like(max_score, float("-inf")),
    )
    return subset_output, subset_lse


def reference_attention_no_sink(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_bhd, max_score, denom, acc = new_reference_attention_state(q)
    max_score, denom, acc = accumulate_reference_attention_chunk(
        q_bhd=q_bhd,
        kv=kv,
        valid_tokens=valid_tokens,
        max_score=max_score,
        denom=denom,
        acc=acc,
        scale=scale,
    )
    return finish_reference_attention_no_sink(max_score, denom, acc)


def merge_reference_attention_with_sink(
    subset_outputs: list[torch.Tensor],
    subset_lses: list[torch.Tensor],
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    assert subset_outputs, "At least one attention subset is required"
    assert len(subset_outputs) == len(subset_lses)

    sink = attn_sink[None, :].float()
    merge_max = sink
    for subset_lse in subset_lses:
        merge_max = torch.maximum(merge_max, subset_lse)

    safe_merge_max = torch.where(
        torch.isfinite(merge_max), merge_max, torch.zeros_like(merge_max)
    )
    merged_acc = torch.zeros_like(subset_outputs[0], dtype=torch.float32)
    sink_weight = torch.exp(sink - safe_merge_max)
    sink_weight = torch.nan_to_num(sink_weight)
    merged_denom = sink_weight
    for subset_output, subset_lse in zip(subset_outputs, subset_lses):
        subset_weight = torch.exp(subset_lse - safe_merge_max)
        subset_weight = torch.nan_to_num(subset_weight)
        merged_acc = merged_acc + subset_output.float() * subset_weight[:, :, None]
        merged_denom = merged_denom + subset_weight

    safe_denom = torch.where(
        merged_denom > 0, merged_denom, torch.ones_like(merged_denom)
    )
    reference_output = merged_acc / safe_denom[:, :, None]
    reference_output = torch.where(
        (merged_denom > 0)[:, :, None],
        reference_output,
        torch.zeros((), dtype=reference_output.dtype, device=reference_output.device),
    )
    output.copy_(reference_output.to(dtype=output.dtype))


def sink_aware_reference_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid_tokens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
) -> None:
    subset_output, subset_lse = reference_attention_no_sink(
        q=q,
        kv=kv,
        valid_tokens=valid_tokens,
        scale=scale,
    )
    merge_reference_attention_with_sink(
        subset_outputs=[subset_output],
        subset_lses=[subset_lse],
        attn_sink=attn_sink,
        output=output,
    )


def reference_sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    topk_chunk_size: int,
    query_chunk_size: int,
) -> None:
    kv_flat = kv.reshape(-1, q.shape[-1])
    topk_chunk_size = min(combined_indices.shape[-1], topk_chunk_size)
    query_chunk_size = min(q.shape[0], query_chunk_size)

    for token_start in range(0, q.shape[0], query_chunk_size):
        token_end = min(token_start + query_chunk_size, q.shape[0])
        q_chunk = q[token_start:token_end]
        lens_chunk = combined_lens[token_start:token_end]
        indices_chunk_full = combined_indices[token_start:token_end]
        q_bhd, max_score, denom, acc = new_reference_attention_state(q_chunk)

        for index_start in range(0, combined_indices.shape[-1], topk_chunk_size):
            index_end = min(
                index_start + topk_chunk_size,
                combined_indices.shape[-1],
            )
            indices_chunk = indices_chunk_full[:, index_start:index_end]
            index_offsets = torch.arange(
                index_start,
                index_end,
                device=q.device,
            )
            valid_tokens = (
                (index_offsets[None, :] < lens_chunk[:, None])
                & (indices_chunk >= 0)
            )
            safe_indices = torch.where(
                valid_tokens,
                indices_chunk,
                torch.zeros((), dtype=indices_chunk.dtype, device=q.device),
            ).long()
            gathered_kv = kv_flat[safe_indices]
            max_score, denom, acc = accumulate_reference_attention_chunk(
                q_bhd=q_bhd,
                kv=gathered_kv,
                valid_tokens=valid_tokens,
                max_score=max_score,
                denom=denom,
                acc=acc,
                scale=scale,
            )

        subset_output, subset_lse = finish_reference_attention_no_sink(
            max_score,
            denom,
            acc,
        )
        merge_reference_attention_with_sink(
            subset_outputs=[subset_output],
            subset_lses=[subset_lse],
            attn_sink=attn_sink,
            output=output[token_start:token_end],
        )
