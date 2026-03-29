# SPDX-License-Identifier: Apache-2.0
"""Triton kernel for fused TurboQuant decode attention score.

Computes attention scores directly from compressed KV cache without
full decompression. Key insight: precompute q_rot = Q @ Pi^T and
q_proj = Q @ S^T once per query, then per cached token the score
is just a centroid gather + QJL bit-unpack -- O(D) per token.

Score formula per cached token:
  score = vec_norm * (term1 + sqrt(pi/2)/D * res_norm * term2)
  term1 = sum_j q_rot[j] * centroids[idx[j]]    (gather)
  term2 = sum_j q_proj[j] * (2*sign[j] - 1)     (bit-unpack)
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tq_fused_score_kernel(
    # Query (already rotated/projected by caller)
    q_rot_ptr,         # [num_queries, num_q_heads, head_size] float32
    q_proj_ptr,        # [num_queries, num_q_heads, head_size] float32
    # Compressed KV cache
    key_cache_ptr,     # [num_blocks, block_size, num_kv_heads, packed_size] uint8
    # Block table
    block_table_ptr,   # [num_queries, max_num_blocks] int32
    # Sequence lengths
    seq_lens_ptr,      # [num_queries] int32
    # Centroids
    centroids_ptr,     # [n_centroids] float32
    # Output scores
    scores_ptr,        # [num_queries, num_q_heads, max_seq_len] float32
    # Dims
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    packed_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    max_seq_len: tl.constexpr,
    mse_bits: tl.constexpr,
    mse_bytes: tl.constexpr,
    qjl_bytes: tl.constexpr,
    n_centroids: tl.constexpr,
    # Constants
    correction_scale: tl.constexpr,  # sqrt(pi/2) / D
    attn_scale: tl.constexpr,        # 1/sqrt(head_size)
    kv_group_size: tl.constexpr,     # num_q_heads // num_kv_heads
):
    """One program per (query, q_head, cache_block)."""
    query_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    cache_block_tile = tl.program_id(2)

    # Which KV head does this Q head attend to?
    kv_head_idx = q_head_idx // kv_group_size

    # Check sequence length
    seq_len = tl.load(seq_lens_ptr + query_idx)
    start_pos = cache_block_tile * block_size
    if start_pos >= seq_len:
        return

    # Load block index from block table
    block_idx = tl.load(
        block_table_ptr + query_idx * max_num_blocks + cache_block_tile
    ).to(tl.int32)

    # Load q_rot and q_proj for this query+head
    d_offs = tl.arange(0, head_size)
    q_rot_base = query_idx * num_q_heads * head_size + q_head_idx * head_size
    q_rot = tl.load(q_rot_ptr + q_rot_base + d_offs)  # [D]

    q_proj_base = query_idx * num_q_heads * head_size + q_head_idx * head_size
    q_proj = tl.load(q_proj_ptr + q_proj_base + d_offs)  # [D]

    # Load centroids
    c_offs = tl.arange(0, n_centroids)
    centroids = tl.load(centroids_ptr + c_offs)  # [n_centroids]

    # Process each position in this cache block
    for pos_in_block in tl.static_range(block_size):
        abs_pos = start_pos + pos_in_block
        if abs_pos >= seq_len:
            break

        # Cache entry base address
        entry_base = (
            block_idx * block_size * num_kv_heads * packed_size
            + pos_in_block * num_kv_heads * packed_size
            + kv_head_idx * packed_size
        )

        # === Unpack MSE indices and compute Term 1 ===
        term1 = tl.zeros([], dtype=tl.float32)

        if mse_bits == 2:
            # 4 indices per byte
            for b in tl.static_range(mse_bytes):
                byte_val = tl.load(key_cache_ptr + entry_base + b).to(tl.int32)
                for k in tl.static_range(4):
                    j = b * 4 + k
                    if j < head_size:
                        idx_j = (byte_val >> (k * 2)) & 0x3
                        # Gather: q_rot[j] * centroids[idx_j]
                        c_val = tl.load(centroids_ptr + idx_j)
                        q_r_j = tl.sum(tl.where(d_offs == j, q_rot, 0.0))
                        term1 += q_r_j * c_val
        elif mse_bits == 3:
            # 3 bits per index, packed across byte boundaries
            for j in tl.static_range(head_size):
                bit_off = j * 3
                byte_idx = bit_off // 8
                bit_idx = bit_off % 8
                b0 = tl.load(key_cache_ptr + entry_base + byte_idx).to(tl.int32)
                idx_j = (b0 >> bit_idx) & 0x7
                if bit_idx > 5 and byte_idx + 1 < mse_bytes:
                    b1 = tl.load(key_cache_ptr + entry_base + byte_idx + 1).to(tl.int32)
                    idx_j |= (b1 << (8 - bit_idx)) & 0x7
                c_val = tl.load(centroids_ptr + (idx_j & 0x7))
                q_r_j = tl.sum(tl.where(d_offs == j, q_rot, 0.0))
                term1 += q_r_j * c_val

        # === Unpack QJL signs and compute Term 2 ===
        term2 = tl.zeros([], dtype=tl.float32)
        sign_base = entry_base + mse_bytes

        for b in tl.static_range(qjl_bytes):
            byte_val = tl.load(key_cache_ptr + sign_base + b).to(tl.int32)
            for k in tl.static_range(8):
                j = b * 8 + k
                if j < head_size:
                    sign_bit = (byte_val >> k) & 1
                    sign_val = tl.where(sign_bit == 1, 1.0, -1.0)
                    q_p_j = tl.sum(tl.where(d_offs == j, q_proj, 0.0))
                    term2 += q_p_j * sign_val

        # === Load norms ===
        norm_base = entry_base + mse_bytes + qjl_bytes
        # Read 2 bytes as uint16, bitcast to float16
        norm_lo = tl.load(key_cache_ptr + norm_base).to(tl.uint16)
        norm_hi = tl.load(key_cache_ptr + norm_base + 1).to(tl.uint16)
        norm_u16 = norm_lo | (norm_hi << 8)
        vec_norm = norm_u16.to(tl.float16, bitcast=True).to(tl.float32)

        gamma_lo = tl.load(key_cache_ptr + norm_base + 2).to(tl.uint16)
        gamma_hi = tl.load(key_cache_ptr + norm_base + 3).to(tl.uint16)
        gamma_u16 = gamma_lo | (gamma_hi << 8)
        res_norm = gamma_u16.to(tl.float16, bitcast=True).to(tl.float32)

        # === Combine ===
        score = vec_norm * (term1 + correction_scale * res_norm * term2)
        score = score * attn_scale

        # Store score
        score_idx = (
            query_idx * num_q_heads * max_seq_len
            + q_head_idx * max_seq_len
            + abs_pos
        )
        tl.store(scores_ptr + score_idx, score)


def triton_tq_fused_attention_score(
    query: torch.Tensor,      # [num_queries, num_q_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, packed_size]
    block_table: torch.Tensor,  # [num_queries, max_num_blocks]
    seq_lens: torch.Tensor,   # [num_queries]
    Pi: torch.Tensor,         # [head_size, head_size]
    S: torch.Tensor,          # [head_size, head_size]
    centroids: torch.Tensor,  # [n_centroids]
    mse_bits: int,
    attn_scale: float,
) -> torch.Tensor:
    """Compute fused TQ attention scores.

    Precomputes q_rot = Q @ Pi^T and q_proj = Q @ S^T,
    then launches Triton kernel for per-token score computation.

    Returns:
        scores: [num_queries, num_q_heads, max_seq_len] float32
    """
    num_queries, num_q_heads, head_size = query.shape
    block_size = key_cache.shape[1]
    num_kv_heads = key_cache.shape[2]
    packed_size = key_cache.shape[3]
    max_num_blocks = block_table.shape[1]
    max_seq_len = int(seq_lens.max().item())
    n_centroids = centroids.shape[0]
    kv_group_size = num_q_heads // num_kv_heads

    mse_bytes = math.ceil(head_size * mse_bits / 8)
    qjl_bytes = math.ceil(head_size / 8)
    correction_scale = math.sqrt(math.pi / 2) / head_size

    # Precompute rotated/projected queries (GEMM, very fast)
    q_float = query.float()
    q_rot = q_float @ Pi.T.contiguous()   # [N, Hq, D]
    q_proj = q_float @ S.T.contiguous()   # [N, Hq, D]

    # Allocate output
    scores = torch.full(
        (num_queries, num_q_heads, max_seq_len),
        float("-inf"),
        device=query.device,
        dtype=torch.float32,
    )

    # Number of cache blocks to process
    num_cache_blocks = (max_seq_len + block_size - 1) // block_size
    grid = (num_queries, num_q_heads, num_cache_blocks)

    _tq_fused_score_kernel[grid](
        q_rot.contiguous(),
        q_proj.contiguous(),
        key_cache,
        block_table,
        seq_lens,
        centroids.contiguous(),
        scores,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        packed_size=packed_size,
        max_num_blocks=max_num_blocks,
        max_seq_len=max_seq_len,
        mse_bits=mse_bits,
        mse_bytes=mse_bytes,
        qjl_bytes=qjl_bytes,
        n_centroids=n_centroids,
        correction_scale=correction_scale,
        attn_scale=attn_scale,
        kv_group_size=kv_group_size,
    )

    return scores
