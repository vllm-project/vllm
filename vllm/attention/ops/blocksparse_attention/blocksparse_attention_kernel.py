import torch
import triton
import triton.language as tl

@triton.jit
def _context_fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    Q,
    k,
    v,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAST_K_BLOCK: tl.constexpr,
    offs_m, 
    context_len,
    start_n,
    offs_n,
):
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) #[M,N]
    # q shape: BLOCK_M x BLOCK_D
    # k shape: BLOCK_D x BLOCK_N
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK 
    # apply causal mask
    if LAST_K_BLOCK:
        qk += tl.where(
            offs_m[:, None] + context_len >= (start_n + offs_n[None, :]),
            0,
            float("-inf"),
        )

    # flash-attn2
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    # update m_i
    m_i = m_ij
    l_i = l_i * alpha + l_ij
    p = p.to(Q.dtype.element_ty)
    acc += tl.dot(p, v)
    return acc, l_i, m_i


@triton.jit
def _context_fwd_kernel_batch_inference(
    Q,
    K,
    V,
    Out,
    sm_scale,
    # batch index for the corresponding query block
    query_batch_ids,
    # block ids for the corresponding query block
    query_block_ids,
    num_queries_per_kv,
    sparse_block_size: tl.constexpr, # 64
    kv_cache_block_size,
    x,
    context_lens_tensor,
    query_lens,
    query_start_loc,
    block_tables,
    # num_blocks, num_kv_heads, head_size // x, block_size, x 
    K_cache,
    # num_blocks, num_kv_heads, head_size, block_size  
    V_cache,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    BLOCK_D: tl.constexpr,
):
    # grid = (len(query_block_ids), n_heads, 1)
    # offset into the query block
    cur_query_block = tl.program_id(0)

    # offset into the head
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // num_queries_per_kv
    
    # The batch id of the current block
    # batch size == number of sequences
    batch_id_for_current_query_block = tl.load(query_batch_ids + cur_query_block).to(tl.int32)
    # The local block id for current block in the current sequence
    block_id_for_current_query_block = tl.load(query_block_ids + cur_query_block).to(tl.int32)

    # context length for current sequence
    context_len = tl.load(context_lens_tensor + batch_id_for_current_query_block).to(tl.int32)
    # query length for current sequence
    query_len = tl.load(query_lens + batch_id_for_current_query_block).to(tl.int32)

    query_start_block_id = context_len // sparse_block_size 
    is_align = context_len % sparse_block_size == 0
    context_end_block_id = query_start_block_id -1 if is_align else query_start_block_id

    cur_q_block_start_location = block_id_for_current_query_block * sparse_block_size
    next_q_block_start_location = (block_id_for_current_query_block + 1) * sparse_block_size
    query_offset_within_current_sequence = 0 if cur_q_block_start_location < context_len else cur_q_block_start_location - context_len

    # Along the query token dimension
    offs_m = query_offset_within_current_sequence + tl.arange(0, sparse_block_size)
    # Along the head hidden dimension which is 128
    offs_d = tl.arange(0, BLOCK_D)

    # global query start location for current sequence
    q_cu_start = tl.load(query_start_loc + batch_id_for_current_query_block).to(tl.int32)
    # Jump to the start location of current sequence
    Q += q_cu_start * stride_qbs + cur_head * stride_qh
    K += q_cu_start * stride_kbs + cur_kv_head * stride_kh
    V += q_cu_start * stride_vbs + cur_kv_head * stride_vh
    Out += q_cu_start * stride_obs + cur_head * stride_oh

    q = tl.load(
        Q + offs_m[:, None] * stride_qbs + offs_d[None, :] * stride_qd,
        # only load current query block, and also could not exceed query length 
        mask=(offs_m[:, None] < query_len) & (offs_m[:, None] + context_len < next_q_block_start_location),
        other=0.0
    )

    # layout_crow_ptr shape is [num_heads, num_blocks + 1], where
    # num_blocks = max_seqlen // block_size
    sparse_crow_ptr = (layout_crow_ptr + cur_head * layout_crow_stride_h +
                       block_id_for_current_query_block * layout_crow_stride_m)

    # Find the k block that current query block should attend to.
    # inclusive
    k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
    # exclusive
    k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

    # on-chip memory
    # running max value of attention score
    m_i = tl.zeros([sparse_block_size], dtype=tl.float32) - float("inf")
    # running sum of EXP of attention score
    l_i = tl.zeros([sparse_block_size], dtype=tl.float32)
    # The output, it won't be the final value until we process all the k blocks
    acc = tl.zeros([sparse_block_size, BLOCK_D], dtype=tl.float32)

    # initialize offsets
    # [N]; starts at 0
    offs_n = tl.arange(0, sparse_block_size)
    
    sm_scale *= (
        1.44269504  # 1/log2 as we use base2 for exponential and logarithm
    )
    # flash attention style tiling
    for k_block_col_idx in range(k_block_start, k_block_end):
        # the block id of the key matrix that the current query block should attend to
        k_block_id = tl.load(layout_col_ptr + cur_head * layout_col_stride_h +
                         k_block_col_idx * layout_col_stride_m).to(tl.int32)
        # attend to k_cache blocks
        if k_block_id <= context_end_block_id:
            # -- compute qk ----
            start_n = k_block_id * sparse_block_size
            # load the token block ids for the current key matrix
            bn = tl.load(block_tables + batch_id_for_current_query_block * stride_b_loc_b +
                         ((start_n + offs_n) // kv_cache_block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < context_len,
                         other=0)  # [N]
            # [D,N]
            # Index into the key matrix cache
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % kv_cache_block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            # [N,D]
            off_v = (
                    bn[:, None] * stride_v_cache_bs +
                    cur_kv_head * stride_v_cache_h +
                    offs_d[None, :] * stride_v_cache_d +
                    (start_n + offs_n[:, None]) % kv_cache_block_size * stride_v_cache_bl)
            k = tl.load(K_cache + off_k,
                             mask=(start_n + offs_n[None, :]) < context_len,
                             other=0.0)  # [D,N]
            v = tl.load(V_cache + off_v,
                             mask= (start_n + offs_n[:, None]) < context_len,
                             other=0.0)  # [N,D]
            acc, l_i, m_i = _context_fwd_kernel_inner(
                acc, l_i, m_i, q, Q, k, v, sm_scale, sparse_block_size, sparse_block_size, 
                # last key block in the key cache
                k_block_id == context_end_block_id,
                offs_m, 
                context_len,
                start_n,
                offs_n
            )

        # compute query against itself (with causal mask)
        if k_block_id >= query_start_block_id:
            start_n = k_block_id * sparse_block_size
            next_k_block_start_location = (k_block_id + 1) * sparse_block_size
            key_offset = 0 if k_block_id == query_start_block_id else start_n - context_len
            offs_n = key_offset + tl.arange(0, sparse_block_size)
            # -- compute qk ----
            k = tl.load(
                K + offs_n[None, :] * stride_kbs + offs_d[:, None] * stride_kd,
                # only load current query block, and also could not exceed query length 
                mask=(offs_n[None, :] < query_len) & (offs_n[None, :] + context_len < next_k_block_start_location),
                other=0.0
            )
            v = tl.load(
                V + offs_n[:, None] * stride_vbs + offs_d[None, :] * stride_vd,
                # only load current query block, and also could not exceed query length 
                mask=(offs_n[:, None] < query_len) & (offs_n[:, None] + context_len < next_k_block_start_location),
                other=0.0
            )
            acc, l_i, m_i = _context_fwd_kernel_inner(
                acc, l_i, m_i, q, Q, k, v, sm_scale, sparse_block_size, sparse_block_size, 
                # LAST_K_BLOCK
                k_block_id == block_id_for_current_query_block,
                offs_m, 
                context_len,
                context_len,
                offs_n
            )
    # flash-attn 2
    # m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # write output
    tl.store(Out + offs_m[:, None] * stride_obs + offs_d[None, :] * stride_od,
             acc,
             mask=(offs_m[:, None] < query_len) & (offs_m[:, None] + context_len < next_q_block_start_location))
    return

@torch.inference_mode()
def context_blocksparse_flash_attn_varlen_fwd(
        q,
        k,
        v, # [num_tokens, num_heads, head_size]
        k_cache,
        v_cache,
        block_tables, 
        query_start_loc,
        seq_lens_tensor,
        context_lens_tensor,
        sm_scale,
        sparse_layout,
        sparse_block_size):
    assert isinstance(sparse_layout, (list, tuple))
    _, n_heads, head_size = q.shape
    assert q.dim() == k.dim() == v.dim() == 3
    # num_tokens should be the same
    assert q.size(0) == k.size(0)
    # num of query head is multiple of num of key head
    assert q.size(1) % k.size(1) == 0
    # head dimension should be the same
    assert q.size(2) == k.size(2)
    # key and value should have the same shape
    assert k.shape == v.shape
    assert query_start_loc.dim() == 1
    assert seq_lens_tensor.dim() == 1
    assert context_lens_tensor.dim() == 1

    num_queries_per_kv = q.size(1) // k.size(1)
    query_lens = seq_lens_tensor - context_lens_tensor
    assert torch.all(query_lens != 0), "query_len should not be 0"

    # switch to use cpu to avoid too many kernel launches when iterated over
    # query start block id and end block id are inclusive 
    query_start_block_id = (context_lens_tensor // sparse_block_size).cpu() 
    query_end_block_id = ((seq_lens_tensor - 1) // sparse_block_size).cpu()
    query_block_ids = torch.cat(
        [torch.arange(s, e+1) for s, e in zip(query_start_block_id, query_end_block_id)]
    ).to(q.device)
    query_batch_ids = torch.cat(
        [torch.full((e - s + 1,), i) for i, (s, e) in enumerate(zip(query_start_block_id, query_end_block_id))]
    ).to(q.device)
    
    out = q.new_empty(q.shape)
    layout_crow_indices, layout_col_indices = sparse_layout
    block_d = triton.next_power_of_2(head_size)
    assert block_d == head_size

    grid = (len(query_block_ids), n_heads, 1)

    _context_fwd_kernel_batch_inference[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        query_batch_ids,
        query_block_ids,
        num_queries_per_kv,
        sparse_block_size,
        v_cache.shape[3],  # kv cache block size
        k_cache.shape[4],  # x
        context_lens_tensor,
        query_lens,
        query_start_loc,
        block_tables,
        k_cache,
        v_cache,
        block_tables.stride(0),
        block_tables.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),  #[num_blocks, num_kv_heads, head_size, block_size]
        layout_crow_indices,
        layout_col_indices,
        layout_crow_indices.stride(0),
        layout_crow_indices.stride(1),
        layout_col_indices.stride(0),
        layout_col_indices.stride(1),
        block_d,
        num_warps=4,
        num_stages=3)
    return out

def blocksparse_flash_attn_varlen_fwd(
        q,
        k,
        v,  # (#tokens, n_heads, head_size)
        cu_seqlens_k,
        cu_seqlens_q,
        sm_scale,
        sparse_layout,
        *,
        block_size=64,
        q_block_size=None,
        max_seqlen=None):
    # split q to blocks

    assert isinstance(sparse_layout, (list, tuple))

    _, n_heads, head_size = q.shape
    batch_size = cu_seqlens_k.size(0) - 1
    q_block_size = q_block_size or block_size

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    # TODO(linxihui): allow k, v to have different head_size
    assert k.shape == v.shape
    assert cu_seqlens_k.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    if cu_seqlens_q is None:
        if q.size(0) == batch_size:  # decoding only
            cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=cu_seqlens_k.dtype,
                device=cu_seqlens_k.device,
            )
        elif q.size(0) == k.size(0):
            cu_seqlens_q = cu_seqlens_k
        else:
            raise ValueError("cu_seqlens_q must be specified\
                    if it mix of prefilling and decoding.")
    else:
        assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)

    # switch to use cpu to avoid too many kernel launches when iterated over
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()

    assert torch.logical_or(q_lens == 1, k_lens == q_lens).all(), (
        "length of q should either be 1 (decoding) or same as k (prefilling).")

    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    n_blocks = (q_lens + q_block_size - 1) // q_block_size

    q_batch_ids = torch.tensor(
        [i for i, n in enumerate(n_blocks) for _ in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )
    q_start_sids = torch.tensor(
        [i * q_block_size for n in n_blocks for i in range(n)],
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )

    out = q.new_empty(q.shape)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    layout_crow_indices, layout_col_indices = sparse_layout
    block_d = triton.next_power_of_2(head_size)

    decoding_only = (q_lens == 1).all().item()
    grid = (len(q_start_sids), n_heads, 1)

    _fwd_kernel_batch_inference[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        cu_seqlens_q[:-1],
        cu_seqlens_q[1:],
        cu_seqlens_k[:-1],
        cu_seqlens_k[1:],
        q_batch_ids,
        q_start_sids,
        0,
        *q.stride(),
        0,
        *k.stride(),
        0,
        *v.stride(),
        0,
        *out.stride(),
        layout_crow_indices,
        layout_col_indices,
        *layout_crow_indices.stride(),
        *layout_col_indices.stride(),
        q_k_ratio,
        HAS_BATCH_DIM=False,
        D_HEAD=head_size,
        BLOCK_M=q_block_size,
        BLOCK_N=block_size,
        BLOCK_D=block_d,
        BLOCK_M_LOADING=(16 if decoding_only else
                         q_block_size),  # smaller for decoding
        EVEN_D=block_d == head_size,
        num_warps=1 if decoding_only else 4,
        num_stages=3)

    return out


@triton.jit
def _fwd_kernel_inner(
    acc,
    l_i,
    m_i,
    q,
    Q,
    k_block_col_idx,
    layout_col_ptr,
    layout_col_stride_h,
    layout_col_stride_m,
    k_ptrs,
    v_ptrs,
    off_h,
    offs_m,
    offs_n,
    offs_d,
    stride_kt,
    stride_vt,
    sm_scale,
    k_seqlen,
    past_len,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr,
):
    # the block id of the key matrix that the current query block should attend to
    k_block_id = tl.load(layout_col_ptr + off_h * layout_col_stride_h +
                         k_block_col_idx * layout_col_stride_m).to(tl.int32)
    start_n = k_block_id * BLOCK_N
    if LAST_K_BLOCK:
        if EVEN_D:
            k = tl.load(
                k_ptrs + start_n * stride_kt,
                mask=offs_n[None, :] + start_n < k_seqlen,
            )
        else:
            k = tl.load(
                k_ptrs + start_n * stride_kt,
                mask=(offs_n[None, :] + start_n < k_seqlen) &
                (offs_d[:, None] < D_HEAD),
            )
    else:
        if EVEN_D:
            k = tl.load(k_ptrs + start_n * stride_kt)
        else:
            k = tl.load(k_ptrs + start_n * stride_kt,
                        mask=offs_d[:, None] < D_HEAD)

    qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
    # q shape: BLOCK_M_LOADING x BLOCK_D
    # k shape: BLOCK_D x BLOCK_N
    qk += tl.dot(q, k)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK | M_LT_N:
        qk += tl.where(
            offs_m[:, None] + past_len >= (start_n + offs_n[None, :]),
            0,
            float("-inf"),
        )

    # flash-attn2
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    # update m_i
    m_i = m_ij
    l_i = l_i * alpha + l_ij

    p = p.to(Q.dtype.element_ty)
    # update acc
    if LAST_K_BLOCK:
        if EVEN_D:
            v = tl.load(
                v_ptrs + start_n * stride_vt,
                mask=offs_n[:, None] + start_n < k_seqlen,
            )
        else:
            v = tl.load(
                v_ptrs + start_n * stride_vt,
                mask=(offs_n[:, None] + start_n < k_seqlen) &
                (offs_d[None, :] < D_HEAD),
            )
    else:
        if EVEN_D:
            v = tl.load(v_ptrs + start_n * stride_vt)
        else:
            v = tl.load(v_ptrs + start_n * stride_vt,
                        mask=offs_d[None, :] < D_HEAD)

    acc += tl.dot(p, v)

    return acc, l_i, m_i


@triton.heuristics({
    "M_LT_N":
    lambda kwargs: kwargs["BLOCK_M"] < kwargs["BLOCK_N"],
})
@triton.jit
def _fwd_kernel_batch_inference(
    Q,
    K,
    V,
    Out,
    sm_scale,
    q_batch_starts,
    q_batch_ends,
    k_batch_starts,
    k_batch_ends,
    q_batch_ids,
    q_start_sids,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    q_k_ratio,
    HAS_BATCH_DIM: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr,
):
    """
    NOTATION:
    pid: position id
    sid: storage id
    sbid: storage block id
    pbid: position block id
    offs_m, offs_n: storage offsets of m-dim(q, row) and n-dim(k, col)

    TODO(linxihui):
    Optimize grouped-attn
    """
    off_zm = tl.program_id(0)
    off_h = tl.program_id(1)

    off_h_for_kv = off_h // q_k_ratio

    if HAS_BATCH_DIM:
        off_z = tl.program_id(2)
        Q += off_z * stride_qb
        K += off_z * stride_kb
        V += off_z * stride_vb
        Out += off_z * stride_ob
        start_m = off_zm
        q_start_sid = start_m * BLOCK_M  # always 0 for decoding
    else:
        off_z = tl.load(q_batch_ids + off_zm).to(tl.int32)  # [0, 0, 0, 1]
        q_start_sid = tl.load(q_start_sids + off_zm)
        start_m = q_start_sid // BLOCK_M  # q_sbid

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M_LOADING)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_cu_start = tl.load(q_batch_starts + off_z).to(tl.int32)
    q_seqlen = tl.load(q_batch_ends + off_z).to(tl.int32) - q_cu_start
    k_cu_start = tl.load(k_batch_starts + off_z).to(tl.int32)
    k_seqlen = tl.load(k_batch_ends + off_z).to(tl.int32) - k_cu_start
    past_len = k_seqlen - q_seqlen

    Q += q_cu_start * stride_qt + off_h * stride_qh
    K += k_cu_start * stride_kt + off_h_for_kv * stride_kh
    V += k_cu_start * stride_vt + off_h_for_kv * stride_vh
    Out += q_cu_start * stride_ot + off_h * stride_oh

    q_pbid = (past_len + q_start_sid) // BLOCK_M

    if EVEN_D:
        q = tl.load(
            Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < q_seqlen,
        )
    else:
        q = tl.load(
            Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
            other=0,
        )

    sparse_crow_ptr = (layout_crow_ptr + off_h * layout_crow_stride_h +
                       q_pbid * layout_crow_stride_m)

    # TODO(linxihui): load at once, with any Triton version
    # that supports `tl.split`, e.g., Triton 3.0
    k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
    k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

    m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_LOADING, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    sm_scale *= (
        1.44269504  # 1/log2 as we use base2 for exponential and logarithm
    )

    for k_block_col_idx in range(k_block_start, k_block_end - 1):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc,
            l_i,
            m_i,
            q,
            Q,
            k_block_col_idx,
            layout_col_ptr,
            layout_col_stride_h,
            layout_col_stride_m,
            k_ptrs,
            v_ptrs,
            off_h,
            offs_m,
            offs_n,
            offs_d,
            stride_kt,
            stride_vt,
            sm_scale,
            k_seqlen,
            past_len,
            False,
            BLOCK_M_LOADING,
            BLOCK_N,
            D_HEAD,
            EVEN_D,
            M_LT_N,
        )

    acc, l_i, m_i = _fwd_kernel_inner(
        acc,
        l_i,
        m_i,
        q,
        Q,
        k_block_end - 1,
        layout_col_ptr,
        layout_col_stride_h,
        layout_col_stride_m,
        k_ptrs,
        v_ptrs,
        off_h,
        offs_m,
        offs_n,
        offs_d,
        stride_kt,
        stride_vt,
        sm_scale,
        k_seqlen,
        past_len,
        True,
        BLOCK_M_LOADING,
        BLOCK_N,
        D_HEAD,
        EVEN_D,
        M_LT_N,
    )

    # flash-attn 2
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # write output
    if EVEN_D:
        tl.store(
            Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
            acc,
            mask=offs_m[:, None] < q_seqlen,
        )
    else:
        tl.store(
            Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od,
            acc,
            mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
        )