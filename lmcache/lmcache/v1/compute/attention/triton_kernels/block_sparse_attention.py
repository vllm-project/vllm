# SPDX-License-Identifier: Apache-2.0
"""Triton block-sparse attention kernel for LMCache CacheBlend.

Replaces flashinfer's VariableBlockSparseAttentionWrapper for ROCm support.
Works on both CUDA and ROCm via Triton.

The kernel computes attention only over the KV blocks specified by the
sparse index structure (CSR format: block_indices + block_indptr), skipping
non-cached blocks. Returns both output and LSE for downstream blending.

Input layout: Q [M, H, D], K [N, H, D], V [N, H, D] (NHD / contiguous)
Sparse structure: block_indptr [num_q_blocks+1], block_indices [nnz_blocks]
Output: O [M, H, D], LSE [M, H] (log-sum-exp for blending)
"""

# Third Party
import torch
import triton
import triton.language as tl


@triton.jit
def _block_sparse_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    block_indices_ptr,  # [nnz_blocks] int32 — which KV blocks to attend
    block_indptr_ptr,  # [num_q_blocks+1] int32 — CSR row pointers
    sm_scale: tl.constexpr,
    seq_len_q,  # total query length
    seq_len_k,  # total KV length
    num_csr_rows,  # number of rows in CSR (may differ from grid dim 0)
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    stride_lm,
    stride_lh,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # query block size (typically 64 or 128)
    BLOCK_N: tl.constexpr,  # KV block size (typically 64 or 128)
):
    """Block-sparse attention forward kernel.

    Grid: (num_q_blocks, NUM_HEADS)
    Each program instance computes attention for one query block × one head,
    iterating only over the KV blocks specified by block_indices.
    Supports GQA: maps query head_idx to kv_head_idx.
    """
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA: map query head to KV head
    kv_head_idx = head_idx * NUM_KV_HEADS // NUM_HEADS

    # Query block range
    q_start = q_block_idx * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offs < seq_len_q

    # Dimension offsets (needed in early-return branch too)
    d_offs = tl.arange(0, HEAD_DIM)

    # Load the sparse KV block range for this query block
    # Guard against grid blocks exceeding CSR rows (partial block case)
    if q_block_idx >= num_csr_rows:
        # No CSR entry for this block — write zeros and -inf LSE
        o_ptrs = (
            O_ptr
            + q_offs[:, None] * stride_om
            + head_idx * stride_oh
            + d_offs[None, :] * stride_od
        )
        tl.store(
            o_ptrs,
            tl.zeros([BLOCK_M, HEAD_DIM], dtype=Q_ptr.dtype.element_ty),
            mask=q_mask[:, None],
        )
        lse_ptrs = LSE_ptr + q_offs * stride_lm + head_idx * stride_lh
        tl.store(
            lse_ptrs, tl.full([BLOCK_M], float("-inf"), dtype=tl.float32), mask=q_mask
        )
        return

    kv_block_start = tl.load(block_indptr_ptr + q_block_idx)
    kv_block_end = tl.load(block_indptr_ptr + q_block_idx + 1)
    num_kv_blocks = kv_block_end - kv_block_start

    # Load Q block: [BLOCK_M, HEAD_DIM]
    q_ptrs = (
        Q_ptr
        + q_offs[:, None] * stride_qm
        + head_idx * stride_qh
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Iterate over sparse KV blocks
    for kv_idx in range(0, num_kv_blocks):
        # Get the actual KV block index
        kv_block = tl.load(block_indices_ptr + kv_block_start + kv_idx)
        kv_start = kv_block * BLOCK_N
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < seq_len_k

        # Load K block: [BLOCK_N, HEAD_DIM] — use kv_head_idx for GQA
        k_ptrs = (
            K_ptr
            + kv_offs[:, None] * stride_km
            + kv_head_idx * stride_kh
            + d_offs[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        # Compute QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Mask invalid KV positions
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_new)
        # Compute softmax weights for current block
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update running sum
        l_new = alpha * l_i + l_ij

        # Rescale accumulator
        acc = acc * alpha[:, None]

        # Load V block: [BLOCK_N, HEAD_DIM] — use kv_head_idx for GQA
        v_ptrs = (
            V_ptr
            + kv_offs[:, None] * stride_vm
            + kv_head_idx * stride_vh
            + d_offs[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)

        # Accumulate: acc += P @ V
        acc += tl.dot(p.to(v.dtype), v)

        # Update softmax state
        m_i = m_new
        l_i = l_new

    # Final normalization — guard against zero-block case (l_i == 0)
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)

    # Compute LSE = m_i + log(l_i); -inf when no blocks attended
    lse = tl.where(l_i > 0, m_i + tl.log(l_i), float("-inf"))

    # Store output — use input dtype to support both fp16 and bf16
    o_ptrs = (
        O_ptr
        + q_offs[:, None] * stride_om
        + head_idx * stride_oh
        + d_offs[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(Q_ptr.dtype.element_ty), mask=q_mask[:, None])

    # Store LSE
    lse_ptrs = LSE_ptr + q_offs * stride_lm + head_idx * stride_lh
    tl.store(lse_ptrs, lse, mask=q_mask)


@triton.jit
def _causal_prefill_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    sm_scale: tl.constexpr,
    seq_len,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_km,
    stride_kh,
    stride_kd,
    stride_vm,
    stride_vh,
    stride_vd,
    stride_om,
    stride_oh,
    stride_od,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Standard causal prefill attention for the is_causal=True path.

    Grid: (cdiv(seq_len, BLOCK_M), NUM_HEADS)
    Supports GQA: maps query head_idx to kv_head_idx.
    """
    q_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # GQA: map query head to KV head
    kv_head_idx = head_idx * NUM_KV_HEADS // NUM_HEADS

    q_start = q_block_idx * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    q_mask = q_offs < seq_len
    d_offs = tl.arange(0, HEAD_DIM)

    # Load Q
    q_ptrs = (
        Q_ptr
        + q_offs[:, None] * stride_qm
        + head_idx * stride_qh
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal: only attend to kv_pos <= q_pos
    # Iterate over KV blocks up to the causal boundary
    num_kv_blocks = (q_start + BLOCK_M + BLOCK_N - 1) // BLOCK_N
    for kv_block_idx in range(num_kv_blocks):
        kv_start = kv_block_idx * BLOCK_N
        kv_offs = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_offs < seq_len

        k_ptrs = (
            K_ptr
            + kv_offs[:, None] * stride_km
            + kv_head_idx * stride_kh
            + d_offs[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale

        # Causal mask: attend only where kv_pos <= q_pos
        causal_mask = kv_offs[None, :] <= q_offs[:, None]
        qk = tl.where(causal_mask & kv_mask[None, :], qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = alpha * l_i + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = (
            V_ptr
            + kv_offs[:, None] * stride_vm
            + kv_head_idx * stride_vh
            + d_offs[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new
        l_i = l_new

    acc = acc / l_i[:, None]

    o_ptrs = (
        O_ptr
        + q_offs[:, None] * stride_om
        + head_idx * stride_oh
        + d_offs[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(Q_ptr.dtype.element_ty), mask=q_mask[:, None])


@triton.jit
def _merge_attention_kernel(
    O1_ptr,
    LSE1_ptr,
    O2_ptr,
    LSE2_ptr,
    Out_ptr,
    seq_len,
    stride_om,
    stride_oh,
    stride_od,
    stride_lm,
    stride_lh,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Merge two attention outputs using their LSE values.

    out = (o1 * exp(lse1) + o2 * exp(lse2)) / (exp(lse1) + exp(lse2))

    This is used to blend sparse-attention output (cached blocks)
    with recomputed output (missing blocks).

    Grid: (cdiv(seq_len, BLOCK_M), NUM_HEADS)
    """
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < seq_len
    d_offs = tl.arange(0, HEAD_DIM)

    # Load LSE values
    lse1 = tl.load(
        LSE1_ptr + offs * stride_lm + head_idx * stride_lh,
        mask=mask,
        other=float("-inf"),
    )
    lse2 = tl.load(
        LSE2_ptr + offs * stride_lm + head_idx * stride_lh,
        mask=mask,
        other=float("-inf"),
    )

    # Numerically stable blending — guard against both-inf case
    max_lse = tl.maximum(lse1, lse2)
    # When both are -inf, set max_lse to 0 to avoid NaN in exp(-inf - -inf)
    safe_max = tl.where(max_lse == float("-inf"), 0.0, max_lse)
    w1 = tl.exp(lse1 - safe_max)
    w2 = tl.exp(lse2 - safe_max)
    w_sum = w1 + w2
    # Guard against w_sum == 0 (both paths had no valid blocks)
    w_sum = tl.where(w_sum > 0, w_sum, 1.0)

    # Normalize weights
    w1 = w1 / w_sum
    w2 = w2 / w_sum

    # Load and blend outputs
    o1_ptrs = (
        O1_ptr
        + offs[:, None] * stride_om
        + head_idx * stride_oh
        + d_offs[None, :] * stride_od
    )
    o2_ptrs = (
        O2_ptr
        + offs[:, None] * stride_om
        + head_idx * stride_oh
        + d_offs[None, :] * stride_od
    )

    o1 = tl.load(o1_ptrs, mask=mask[:, None], other=0.0)
    o2 = tl.load(o2_ptrs, mask=mask[:, None], other=0.0)

    out = o1 * w1[:, None] + o2 * w2[:, None]

    out_ptrs = (
        Out_ptr
        + offs[:, None] * stride_om
        + head_idx * stride_oh
        + d_offs[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(O1_ptr.dtype.element_ty), mask=mask[:, None])


# ============================================================
# Python wrappers
# ============================================================


def block_sparse_attention(
    query: torch.Tensor,  # [M, H, D]
    key: torch.Tensor,  # [N, H, D]
    value: torch.Tensor,  # [N, H, D]
    block_indices: torch.Tensor,  # [nnz_blocks] int32
    block_indptr: torch.Tensor,  # [num_q_blocks+1] int32
    sm_scale: float,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute block-sparse attention.

    Args:
        query: [M, H, D] query tensor
        key: [N, H, D] key tensor
        value: [N, H, D] value tensor
        block_indices: [nnz_blocks] which KV blocks to attend to (CSR values)
        block_indptr: [num_q_blocks+1] CSR row pointers
        sm_scale: softmax scale (1/sqrt(head_dim))
        block_size: block size for both Q and KV (default 64)

    Returns:
        output: [M, H, D] attention output
        lse: [M, H] log-sum-exp for blending
    """
    M, H, D = query.shape
    N = key.shape[0]
    KV_H = key.shape[1]  # num_kv_heads (may differ from H for GQA)

    output = torch.empty_like(query)
    lse = torch.empty(M, H, dtype=torch.float32, device=query.device)

    num_q_blocks = (M + block_size - 1) // block_size
    # CSR rows may be fewer than grid blocks (partial trailing block)
    num_csr_rows = block_indptr.shape[0] - 1

    grid = (num_q_blocks, H)

    _block_sparse_attn_fwd_kernel[grid](
        query,
        key,
        value,
        output,
        lse,
        block_indices,
        block_indptr,
        sm_scale,
        M,
        N,
        num_csr_rows,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        NUM_HEADS=H,
        NUM_KV_HEADS=KV_H,
        HEAD_DIM=D,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
    )

    return output, lse


def causal_prefill_attention(
    query: torch.Tensor,  # [M, H, D]
    key: torch.Tensor,  # [M, H, D]
    value: torch.Tensor,  # [M, H, D]
    sm_scale: float,
    block_size: int = 64,
) -> torch.Tensor:
    """Standard causal prefill attention via Triton.

    Args:
        query: [M, H, D]
        key: [M, H, D]
        value: [M, H, D]
        sm_scale: 1/sqrt(head_dim)

    Returns:
        output: [M, H, D]
    """
    M, H, D = query.shape
    KV_H = key.shape[1]  # num_kv_heads (may differ from H for GQA)
    output = torch.empty_like(query)

    grid = ((M + block_size - 1) // block_size, H)

    _causal_prefill_attention_kernel[grid](
        query,
        key,
        value,
        output,
        sm_scale,
        M,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_HEADS=H,
        NUM_KV_HEADS=KV_H,
        HEAD_DIM=D,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
    )

    return output


def merge_attention_outputs(
    output1: torch.Tensor,  # [M, H, D]
    lse1: torch.Tensor,  # [M, H]
    output2: torch.Tensor,  # [M, H, D]
    lse2: torch.Tensor,  # [M, H]
    block_size: int = 128,
) -> torch.Tensor:
    """Merge two attention outputs using LSE-based blending.

    Used to combine sparse attention (cached blocks) with
    recomputed attention (missing blocks).

    Args:
        output1: [M, H, D] first attention output
        lse1: [M, H] first LSE
        output2: [M, H, D] second attention output
        lse2: [M, H] second LSE

    Returns:
        merged: [M, H, D] blended output
    """
    M, H, D = output1.shape
    merged = torch.empty_like(output1)

    grid = ((M + block_size - 1) // block_size, H)

    _merge_attention_kernel[grid](
        output1,
        lse1,
        output2,
        lse2,
        merged,
        M,
        output1.stride(0),
        output1.stride(1),
        output1.stride(2),
        lse1.stride(0),
        lse1.stride(1),
        NUM_HEADS=H,
        HEAD_DIM=D,
        BLOCK_M=block_size,
    )

    return merged
