# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = k_fp8.to(torch.bfloat16)
    q = q.to(torch.bfloat16)
    device = q.device

    mask_lo = (
        torch.arange(0, seq_len_kv, device=device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    # ``score`` is [H, M, N]; ``scale`` is the per-KV-token scale, which
    # vLLM callers hand us as ``[N, 1]`` (a ``[N, 4]`` uint8 buffer cast
    # to fp32). PyTorch right-aligns dimensions for broadcasting, so a
    # naked ``score * scale`` would align ``scale``'s leading dim with
    # ``score``'s M dim and raise a shape mismatch. Flatten to ``[N]`` so
    # broadcasting lines up with the last dim of ``score``.
    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale.reshape(-1)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    if next_n == 1:
        # CUDA-graph safe: no .item() syncs and no per-batch Python loop.
        # Compute over the full padded shape and mask invalid positions
        # against ``context_lens`` instead of slicing per request.
        block_size = kv_cache.shape[1]
        if context_lens.dim() > 1:
            context_lens = context_lens.squeeze(-1)
        head_width = dim + 4  # ``dim`` fp8 bytes + 4-byte fp32 scale per token
        kv_cache_flat = kv_cache.view(-1, block_size * head_width)

        max_pages = block_tables.shape[1]
        padded_seq_len = max_pages * block_size

        # Gather all pages at once. Indices past each request's actual page
        # count point at valid (but irrelevant) blocks per vllm convention;
        # their contribution is zeroed by the position mask below.
        pages = block_tables[:batch_size, :max_pages]
        cache = kv_cache_flat[pages]  # [B, max_pages, block_size * head_width]
        scale_offset = block_size * dim

        cache_value_u8 = cache[..., :scale_offset].contiguous()
        cache_value = (
            cache_value_u8.view(fp8_dtype)
            .to(torch.float32)
            .reshape(batch_size, padded_seq_len, dim)
        )
        cache_scale_u8 = cache[..., scale_offset:].contiguous()
        cache_scale = cache_scale_u8.view(torch.float32).reshape(
            batch_size, padded_seq_len
        )

        q_fp32 = q[:, 0].to(torch.float32)  # [B, num_heads, dim]
        weights_b = weights[:batch_size]  # [B, num_heads]

        # score[b, t, h] = <cache_value[b, t], q[b, h]>
        score = torch.einsum("btd,bhd->bth", cache_value, q_fp32)
        score = F.relu(score)
        score = score * weights_b.unsqueeze(1)
        score = score.sum(dim=-1)  # [B, padded_seq_len]
        score = score * cache_scale

        pos = torch.arange(padded_seq_len, device=q.device, dtype=context_lens.dtype)
        valid = pos.unsqueeze(0) < context_lens.unsqueeze(1)
        score = torch.where(valid, score, torch.full_like(score, float("-inf")))

        logits = torch.full(
            [batch_size, max_model_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
        write_width = min(padded_seq_len, max_model_len)
        logits[:, :write_width] = score[:, :write_width]
        return logits

    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = context_lens[i]
        if context_len.ndim == 0:
            context_len_i = int(context_len.item())
            q_offsets = torch.arange(
                context_len_i - next_n, context_len_i, device=q.device
            )
            context_limit = torch.full(
                (next_n,), context_len_i, dtype=torch.int32, device=q.device
            )
        else:
            context_limit = context_len.to(device=q.device, dtype=torch.int32)
            q_offsets = context_limit - 1
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        max_context_len = int(context_limit.max().item())
        for block_rk in range(cdiv(max_context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
            )
            mask = (k_offsets[None, :] < context_limit[:, None]) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits
