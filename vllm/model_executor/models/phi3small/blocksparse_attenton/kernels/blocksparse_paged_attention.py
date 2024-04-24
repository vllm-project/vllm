import triton
import triton.language as tl
import torch


def blocksparse_flash_attn_varlen_fwd_with_blocktable(
    q, # (#tokens, n_heads, head_size)
    k, # (num_blocks, n_heads, head_size / x, vllm_block_size, x)
    v, # (num_blocks, n_heads, head_size, vllm_block_size)
    block_tables,
    context_lens,
    sm_scale,
    sparse_layout,
    *,
    vllm_block_size=16,
    sparse_block_size=64,
    kv_scale=1,
    num_local_blocks=16,
    mode='split', # combine, split, local-only, remote-only
    max_seqlen=None,
):
    assert mode in ('split', 'combine', 'local-only', 'remote-only')
    split_local_remote = (mode == 'split')

    _, n_heads, head_size = q.shape
    batches = context_lens.size(0)
    assert batches == q.size(0)

    assert q.dim() == 3 and k.dim() == 5 and v.dim() == 4
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2) * k.size(4)
    assert k.size(1) == k.size(1)
    assert context_lens.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    # the following is only needed to determined q/k len
    context_lens = context_lens.contiguous()
    layout_crow_indices, layout_col_indices = sparse_layout

    assert sparse_block_size % vllm_block_size == 0
    block_d = triton.next_power_of_2(head_size)

    IS_FP8 = k.element_size() == 1
    X = 16 // k.element_size() # fixed in vllm

    if split_local_remote:
        start_local_head_idx = n_heads
        out = q.new_zeros((q.size(0), q.size(1) * 2, q.size(2)))
        m = q.new_zeros((q.size(0), q.size(1) * 2), dtype=torch.float32) - float('inf')

        grid = (batches, n_heads + n_heads // q_k_ratio, 1)
    else:
        m = q.new_empty((q.size(0), q.size(1)), dtype=torch.float32)
        out = q.new_zeros(q.shape)
        if mode == 'local-only':
            start_local_head_idx = 0
            grid = (batches, n_heads // q_k_ratio, 1)
        else:
            start_local_head_idx = n_heads + 1
            grid = (batches, n_heads, 1)

    _fwd_kernel_batch_inference_with_blocktable[grid](
    q, k, v, out, m,

    sm_scale,
    context_lens,

    *q.stride(),
    *k.stride(),
    *v.stride(),
    *out.stride(),
    *m.stride(),

    layout_crow_indices,
    layout_col_indices,
    *layout_crow_indices.stride(),
    *layout_col_indices.stride(),

    q_k_ratio,
    block_tables,
    *block_tables.stride(),

    kv_scale,

    start_local_head_idx,
    num_local_blocks,
    SPARSE_BLOCK_SIZE=sparse_block_size,

    X=X,
    IS_FP8=IS_FP8,

    D_HEAD = head_size,
    BLOCK_M = vllm_block_size,
    BLOCK_N = vllm_block_size,
    BLOCK_D = block_d,
    BLOCK_M_LOADING = 16,
    EVEN_D = block_d == head_size,
    num_warps = 1,
    num_stages = 3
    )

    if split_local_remote:
        m0 = m.view(m.size(0), 2, -1)
        m = torch.exp2(m0 - m0.max(1, keepdim=True)[0])

        m /= m.sum(1, keepdim=True)
        out = out.view(out.size(0), 2, -1, out.size(2))
        # out = out[:, 1] # local only
        out = (out * m.unsqueeze(-1).type_as(out)).sum(1)

    return out


@triton.jit
def _load_kv(
    k_block_id,
    K, V, Q,
    block_tables,
    stride_kb, stride_kd, stride_kt, stride_kx,
    stride_vb, stride_vd, stride_vt,
    stride_btb, stride_btt,
    off_z,
    kv_scale,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    X: tl.constexpr,
    IS_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    ):
    bt_id = tl.load(block_tables + off_z * stride_btb + k_block_id * stride_btt)

    k_ptrs = K + bt_id * stride_kb + tl.arange(0, BLOCK_N)[None, None] * stride_kt + \
        tl.arange(0, BLOCK_D // X)[:, None, None] * stride_kd + tl.arange(0, X)[None, :, None] * stride_kx
    k_ptrs = tl.reshape(k_ptrs, (BLOCK_D, BLOCK_N))

    v_ptrs = V + bt_id * stride_vb + tl.arange(0, BLOCK_N)[:, None] * stride_vt + \
        tl.arange(0, BLOCK_D)[None, :] * stride_vd

    ### for using vector-product
    # k_ptrs = K + bt_id * stride_kb + tl.arange(0, BLOCK_N)[:, None, None] * stride_kt + \
    #     tl.arange(0, BLOCK_D // X)[None, :, None] * stride_kd + tl.arange(0, X)[None, None, :] * stride_kx
    # k_ptrs = tl.reshape(k_ptrs, (BLOCK_N, BLOCK_D))
    # v_ptrs = V + bt_id * stride_vb + tl.arange(0, BLOCK_N)[None, :] * stride_vt + \
    #     tl.arange(0, BLOCK_D)[:, None] * stride_vd

    if EVEN_D:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_D)[:, None] < D_HEAD)
        v = tl.load(v_ptrs, mask=tl.arange(0, BLOCK_D)[None, :] < D_HEAD)

    if IS_FP8:
        k = k.to(tl.bfloat16) * kv_scale
        v = v.to(tl.bfloat16) * kv_scale

    return k, v


@triton.jit
def _fwd_kernel_inner_with_blocktable(
    acc, l_i, m_i,
    q, k, v, Q,
    k_block_id,
    offs_n,
    sm_scale,
    q_pid,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    start_n = k_block_id * BLOCK_N
    qk = tl.dot(q, k)
    # qk = tl.expand_dims(tl.sum(q * k.to(tl.float32), 1), 0)
    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK:
        qk += tl.where(start_n + offs_n[None, :] <= q_pid, 0, -float('inf'))

    ### flash-attn2
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
    acc = tl.dot(p, v, acc)
    # acc += tl.expand_dims(tl.sum(p* v.to(tl.float32), 1), 0)
    # acc += tl.expand_dims(tl.sum(p* tl.trans(v).to(tl.float32), 1), 0)

    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_batch_inference_with_blocktable(
    Q, K, V, Out, M,

    sm_scale,
    context_lens,

    stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_kt, stride_kx,
    stride_vb, stride_vh, stride_vd, stride_vt,
    stride_ot, stride_oh, stride_od,
    stride_mt, stride_mh,

    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,

    q_k_ratio,
    block_tables,
    stride_btb, stride_btt,

    kv_scale,

    start_local_head_idx: tl.constexpr,
    num_local_blocks: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,

    X: tl.constexpr, # X = 16/ #bytes of k dtype
    IS_FP8: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)

    is_local = off_h >= start_local_head_idx

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    k_seqlen = tl.load(context_lens + off_z).to(tl.int32)
    q_pid = k_seqlen - 1
    q_pbid = q_pid // BLOCK_M # to find column/k_block id


    if is_local:
        off_h_for_kv = off_h - start_local_head_idx
        q_mask_h = tl.arange(0, BLOCK_M_LOADING)
        q = tl.load(Q + off_z * stride_qt + (off_h_for_kv * q_k_ratio + q_mask_h[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
                    mask=q_mask_h[:, None] < q_k_ratio)
        # convert original sparsse block size to vllm qv-cache block size
        num_blocks_per_sparse_block = SPARSE_BLOCK_SIZE // BLOCK_M
        k_block_end = q_pbid + 1 # exclusive
        sparse_k_block_end = q_pid // SPARSE_BLOCK_SIZE + 1
        k_block_start = (sparse_k_block_end - num_local_blocks) * num_blocks_per_sparse_block
        if k_block_start < 0:
            k_block_start = 0

        m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M_LOADING, D_HEAD], dtype=tl.float32)

        K += off_h_for_kv * stride_kh
        V += off_h_for_kv * stride_vh

        if IS_FP8:
            q = q.to(tl.bfloat16)
        sm_scale *= 1.44269504 # 1/log2 as we use base2 for exponential and logorithm

        for k_block_id in range(k_block_start, k_block_end - 1):
            k, v = _load_kv(
                k_block_id,
                K, V, Q,
                block_tables,
                stride_kb, stride_kd, stride_kt, stride_kx,
                stride_vb, stride_vd, stride_vt,
                stride_btb, stride_btt,
                off_z,
                kv_scale,
                BLOCK_N,
                D_HEAD,
                X,
                IS_FP8,
                BLOCK_D,
                EVEN_D
                )

            acc, l_i, m_i = _fwd_kernel_inner_with_blocktable(
                    acc, l_i, m_i,
                    q, k, v, Q,
                    k_block_id,
                    offs_n,
                    sm_scale,
                    q_pid,
                    False,
                    BLOCK_M_LOADING,
                    BLOCK_N)

        k_block_id = k_block_end - 1
        k, v = _load_kv(
            k_block_id,
            K, V, Q,
            block_tables,
            stride_kb, stride_kd, stride_kt, stride_kx,
            stride_vb, stride_vd, stride_vt,
            stride_btb, stride_btt,
            off_z,
            kv_scale,
            BLOCK_N,
            D_HEAD,
            X,
            IS_FP8,
            BLOCK_D,
            EVEN_D
            )

        acc, l_i, m_i = _fwd_kernel_inner_with_blocktable(
                acc, l_i, m_i,
                q, k, v, Q,
                k_block_id,
                offs_n,
                sm_scale,
                q_pid,
                True,
                BLOCK_M_LOADING,
                BLOCK_N)

        # TODO(linxihui): split last block

        ### flash-attn 2
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]

        Out += off_z * stride_ot + (off_h_for_kv * q_k_ratio + start_local_head_idx) * stride_oh
        q_mask_h = tl.arange(0, BLOCK_M_LOADING)
        tl.store(Out + q_mask_h[:, None] * stride_oh + offs_d[None, :] * stride_od, acc,
                mask=q_mask_h[:, None] < q_k_ratio)

        M += off_z * stride_mt + (off_h_for_kv * q_k_ratio + start_local_head_idx) * stride_mh
        tl.store(M + q_mask_h * stride_mh, m_i, mask=q_mask_h < q_k_ratio)

    else:
        off_h_for_kv = off_h // q_k_ratio
        offs_m = tl.arange(0, BLOCK_M_LOADING)

        Q += off_z * stride_qt + off_h * stride_qh
        if EVEN_D:
            q = tl.load(Q + offs_d[None, :] * stride_qd)
        else:
            q = tl.load(Q + offs_d[None, :] * stride_qd, mask=(offs_d[None, :] < D_HEAD), other=0)

        q = tl.broadcast_to(q, (BLOCK_M_LOADING, D_HEAD))

        sparse_crow_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + q_pbid * layout_crow_stride_m

        # TODO(linxihui): load at once, supported in new Triton
        k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
        k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

        m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M_LOADING, D_HEAD], dtype=tl.float32)

        K += off_h_for_kv * stride_kh
        V += off_h_for_kv * stride_vh

        if IS_FP8:
            q = q.to(tl.bfloat16)
        sm_scale *= 1.44269504 # 1/log2 as we use base2 for exponential and logorithm

        for k_block_col_idx in range(k_block_start, k_block_end):
            k_block_id = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + k_block_col_idx * layout_col_stride_m).to(tl.int32)

            k, v = _load_kv(
                k_block_id,
                K, V, Q,
                block_tables,
                stride_kb, stride_kd, stride_kt, stride_kx,
                stride_vb, stride_vd, stride_vt,
                stride_btb, stride_btt,
                off_z,
                kv_scale,
                BLOCK_N,
                D_HEAD,
                X,
                IS_FP8,
                BLOCK_D,
                EVEN_D
                )

            acc, l_i, m_i = _fwd_kernel_inner_with_blocktable(
                    acc, l_i, m_i,
                    q, k, v, Q,
                    k_block_id,
                    offs_n,
                    sm_scale,
                    q_pid,
                    False, # can be safely set to False is it is remote only
                    BLOCK_M_LOADING,
                    BLOCK_N)

        ### flash-attn 2
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]

        # write output
        offs_m = tl.arange(0, BLOCK_M_LOADING)
        Out += off_z * stride_ot + off_h * stride_oh
        if EVEN_D:
            tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
                    mask=offs_m[:, None] < 1)
        else:
            tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
                    mask=(offs_m[:, None] < 1) & (offs_d[None, :] < D_HEAD))

        M += off_z * stride_mt + off_h * stride_mh
        tl.store(M + offs_m * stride_mh, m_i, mask=offs_m < 1)