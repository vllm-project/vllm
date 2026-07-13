# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility sparse-indexer logits implementations for SM120."""

import torch


def fp8_mqa_logits_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute non-paged MQA logits without unsupported DeepGEMM kernels."""
    if q.ndim != 3 or k.ndim != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"Unexpected q/k shapes: {q.shape=}, {k.shape=}")
    if weights.shape != q.shape[:2]:
        raise ValueError(f"Unexpected weights shape: {weights.shape}")

    q_bf16 = q.to(torch.bfloat16)
    k_bf16_t = k.to(torch.bfloat16).T.contiguous()
    logits = torch.zeros((q.shape[0], k.shape[0]), dtype=torch.float32, device=q.device)
    for head in range(q.shape[1]):
        head_logits = torch.mm(q_bf16[:, head], k_bf16_t).float()
        head_logits.relu_()
        head_logits.mul_(weights[:, head].float().unsqueeze(1))
        logits.add_(head_logits)
    logits.mul_(k_scale.float().reshape(1, -1))

    if clean_logits:
        positions = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
        valid = (positions >= cu_seqlen_ks.reshape(-1, 1)) & (
            positions < cu_seqlen_ke.reshape(-1, 1)
        )
        logits.masked_fill_(~valid, float("-inf"))
    return logits


def fp8_paged_mqa_logits_sm120(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute paged MQA logits for an FP8 indexer cache on SM120."""
    if q.ndim != 4:
        raise ValueError(f"Expected q to have shape [B, N, H, D], got {q.shape}")
    batch_size, next_n, num_heads, head_dim = q.shape
    block_size = kv_cache.shape[1]
    if kv_cache.shape[2:] != (1, head_dim + 4):
        raise ValueError(f"Unexpected indexer cache shape: {kv_cache.shape}")
    if weights.shape != (batch_size * next_n, num_heads):
        raise ValueError(f"Unexpected weights shape: {weights.shape}")

    max_pages = min(
        block_tables.shape[1], (max_model_len + block_size - 1) // block_size
    )
    padded_seq_len = min(max_pages * block_size, max_model_len)
    cache_rows = kv_cache.reshape(kv_cache.shape[0], block_size * (head_dim + 4))
    value_bytes = block_size * head_dim
    q_bf16 = q.to(torch.bfloat16)
    weights = weights.view(batch_size, next_n, num_heads).float()
    logits = torch.zeros(
        (batch_size, next_n, max_model_len),
        dtype=torch.float32,
        device=q.device,
    )

    # Process requests independently so the dequantized K workspace scales with
    # one sequence rather than batch_size * max_model_len.
    for batch in range(batch_size):
        page_ids = block_tables[batch, :max_pages].clamp_min(0).long()
        pages = cache_rows[page_ids]
        k = pages[:, :value_bytes].contiguous().view(q.dtype)
        k = k.view(max_pages * block_size, head_dim)[:padded_seq_len]
        k_scale = pages[:, value_bytes:].contiguous().view(torch.float32)
        k_scale = k_scale.view(-1)[:padded_seq_len]
        k_bf16_t = k.to(torch.bfloat16).T.contiguous()

        output = logits[batch, :, :padded_seq_len]
        for head in range(num_heads):
            head_logits = torch.mm(q_bf16[batch, :, head], k_bf16_t).float()
            head_logits.relu_()
            head_logits.mul_(weights[batch, :, head].unsqueeze(1))
            output.add_(head_logits)
        output.mul_(k_scale.float().unsqueeze(0))

    logits = logits.view(batch_size * next_n, max_model_len)
    if clean_logits:
        seq_lens = context_lens.reshape(batch_size, next_n)
        positions = torch.arange(max_model_len, device=q.device).unsqueeze(0)
        logits.masked_fill_(~(positions < seq_lens.reshape(-1, 1)), float("-inf"))
    return logits
