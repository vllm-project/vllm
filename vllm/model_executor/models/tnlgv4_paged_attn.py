import triton
import triton.language as tl
import torch
import math
from vllm.model_executor.models.tnlgv4_utils import _get_sparse_attn_mask, dense_to_crow_col
# import ipdb
# _b = ipdb.set_trace

print(f'>> {triton.__version__=}, {triton.__file__=}')


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
    split_local_stride=False,
    max_seqlen=None,
):
    # print(f'> {q.shape=}, {k.shape=}, {v.shape=}, {block_tables.shape=}, {context_lens.shape=}')

    # split q to blocks
    _, n_heads, head_size = q.shape
    batches = context_lens.size(0)
    assert batches == q.size(0)

    assert q.dim() == 3 and k.dim() == 5 and v.dim() == 4
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2) * k.size(4)
    assert k.size(1) == k.size(1)

    assert context_lens.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    # q_batch_ids = torch.arange(batches, dtype=torch.int32, device=q.device)
    # q_start_sids = torch.zeros_like(q_batch_ids)

    # the following is only needed to determined q/k len
    context_lens = context_lens.contiguous()

    # k_batch_starts = torch.zeros_like(context_lens)
    # k_batch_ends = context_lens

    # q_batch_starts = torch.arange(0, batches, device=q.device, dtype=k_batch_starts.dtype)
    # q_batch_ends = q_batch_starts + 1

    layout_crow_indices, layout_col_indices = sparse_layout

    # assert block_size == vllm_block_size
    assert sparse_block_size % vllm_block_size == 0
    # if block_size != vllm_block_size:
    #     factor = block_size // vllm_block_size
    #     layout_crow_indices *= factor
    #     layout_col_indices *= factor
    #     layout_col_indices = layout_col_indices[:, None] + torch.arange(0, factor).type_as(layout_col_indices)[None, :, None]
    #     layout_col_indices = layout_col_indices.view(layout_col_indices.size(0), -1)

    #     print(layout_crow_indices)
    #     print(layout_col_indices)


        # import ipdb; ipdb.set_trace()
    block_d = triton.next_power_of_2(head_size)

    # if split_local_stride:
    #     grid = (batches, n_heads + n_heads // q_k_ratio, 1)
    # else:
    grid = (batches, n_heads, 1)
    # grid = (batches, n_heads // q_k_ratio, 1)
    # grid = (1, 1, 1)

    # print(f'>>> {q.dim()=}, {k.dim()=}, {v.dim()=}, {out.dim()=}, {m.dim()=}, {layout_crow_indices.dim()=}, {layout_col_indices.dim()=}')
    # qk = torch.zeros((vllm_block_size, vllm_block_size), dtype=torch.float32, device=q.device)
    # ko = torch.zeros((vllm_block_size, head_size)).type_as(k)

    IS_FP8 = k.element_size() == 1
    X = 16 // k.element_size()

    if split_local_stride:
        out = q.new_zeros((q.size(0), q.size(1) * 2, q.size(2)))
        m = q.new_zeros((q.size(0), q.size(1) * 2), dtype=torch.float32) - float('inf')

        is_locals = context_lens.new_zeros((n_heads + n_heads // q_k_ratio,))
        is_locals[n_heads:] = 1

        grid = (batches, n_heads + n_heads // q_k_ratio, 1)

    else:
        out = q.new_zeros(q.shape)
        m = q.new_zeros((1, 1), dtype=torch.float32)
        is_locals = context_lens.new_zeros((n_heads,))


    # q0 = q

    # _b()
    # print(f'>>> {grid=}, {n_heads=}, {num_local_blocks=}, {q.shape=}')
    _fwd_kernel_batch_inference_with_blocktable[grid](
    q, k, v, out, m,
    # qk,
    # ko,

    sm_scale,
    context_lens,

    *q.stride(),
    *k.stride(),
    *v.stride(),
    *out.stride(),
    *m.stride(),

    # *qk.stride(),
    # *ko.stride(
    layout_crow_indices,
    layout_col_indices,
    *layout_crow_indices.stride(),
    *layout_col_indices.stride(),

    q_k_ratio,
    block_tables,
    *block_tables.stride(),

    kv_scale,

    is_locals,
    n_heads,
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

    if split_local_stride:
        m0 = m.view(m.size(0), 2, -1)
        m = torch.exp2(m0 - m0.max(1, keepdim=True)[0])
        
        # print(f'>>> {m.shape=}')
        
        # # assert (q0 == q).all(), f'{q0=}\n{q=}'
        # if not (q0 == q).all():
        #     _b()
        # if torch.isnan(m).any():
        #     _b()
        #     print(torch.isnan(m0).any(), torch.isnan((m0 - m0.max(1)[0])).any())
 
        m /= m.sum(1, keepdim=True)
        # print(f'>>>> {m[:, 0]=}\n{m[:, 1]=}')
        # _b()

        out = out.view(out.size(0), 2, -1, out.size(2))
        # out = out[:, 1]
        out = (out * m.unsqueeze(-1).type_as(out)).sum(1)


    # print(f'... {qk=}')
    # print(f'...{ko=}')
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
    # -- compute qk ----
    # qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
    qk = tl.dot(q, k)

    # qk = tl.expand_dims(tl.sum(q * k.to(tl.float32), 1), 0)

    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK:
        qk += tl.where(start_n + offs_n[None, :] <= q_pid, 0, -float('inf'))

    # ## flash-attn1
    # # -- compute m_ij, p, l_ij
    # m_ij = tl.max(qk, 1)
    # p = tl.exp(qk - m_ij[:, None])
    # l_ij = tl.sum(p, 1)
    # # -- update m_i and l_i
    # m_i_new = tl.maximum(m_i, m_ij)
    # alpha = tl.exp(m_i - m_i_new)
    # beta = tl.exp(m_ij - m_i_new)
    # l_i_new = alpha * l_i + beta * l_ij
    # # -- update output accumulator --
    # # scale p
    # p_scale = beta / l_i_new
    # p = p * p_scale[:, None]
    # # scale acc
    # acc_scale = l_i / l_i_new * alpha
    # acc = acc * acc_scale[:, None]
    # # update m_i and l_i
    # l_i = l_i_new
    # m_i = m_i_new

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

    # # update m_i and l_i
    # l_i = l_i_new
    # m_i = m_i_new


    return acc, l_i, m_i


@triton.jit
def _fwd_kernel_batch_inference_with_blocktable(
    Q, K, V, Out, M,

    # QK,
    # KO,
    sm_scale,
    context_lens,

    stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_kt, stride_kx,
    stride_vb, stride_vh, stride_vd, stride_vt,
    stride_ot, stride_oh, stride_od,
    stride_mt, stride_mh,
    # stride_qk1, stride_qk2,
    # stride_ko1, stride_ko2,

    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,

    q_k_ratio,
    block_tables,
    stride_btb, stride_btt,

    kv_scale,

    is_locals,
    num_heads: tl.constexpr,
    num_local_blocks: tl.constexpr,
    SPARSE_BLOCK_SIZE: tl.constexpr,

    X: tl.constexpr,
    IS_FP8: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    '''
    NOTATION:
    pid: position id
    sid: storage id
    sbid: storage block id
    pbid: position block id
    offs_m, offs_n: storage offsets of m-dim(q, row) and n-dim(k, col)

    q and blocks in KV needs to be contiguous

    Arguments:
    kv_seq_lens: for compute past_len
    kv_storage_offsets: similar to block_tables in vllm, except it is dynamic.
        TODO: fix this

    TODO:
    Optimize grouped-attn

    CUDA graph support issue
        1. grid is dynamic: vllm set up multiple cuda graph in decoding phase, with diff max token size (16, 32, ...)
            since we mix prompt and decoing phase here, it can be more complex.
            need to set up diff cuda-graph for diff (off_zm, off_z)

            # indeed, q_batch_ids can be padded to maximum number of grid[0], i.e., assume all decoding
            therefore, cu_seqlens_q, kv_seq_lens

    '''
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)

    # is_local = tl.program_id(0) >= num_heads
    # tl.device_print('> is_local:', num_heads)

    is_local = tl.load(is_locals + off_h)
    # tl.device_print('> is_local:', is_local)

    # is_local = (off_h >= num_heads).to(tl.int32)
    # off_h_for_kv = off_h - H if is_local else off_h // q_k_ratio
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    k_seqlen = tl.load(context_lens + off_z).to(tl.int32)
    q_pid = k_seqlen - 1
    q_pbid = q_pid // BLOCK_M # to find column/k_block id


    # off_z = tl.load(q_batch_ids + off_zm).to(tl.int32)   # [0, 0, 0, 1]
    # q_start_sid = tl.load(q_start_sids + off_zm)
    # start_m = q_start_sid // BLOCK_M # q_sbid

    # tl.device_print('> off_h:', off_h)

    if is_local:
        off_h_for_kv = off_h - num_heads
        q_mask_h = tl.arange(0, BLOCK_M_LOADING)
        q = tl.load(Q + off_z * stride_qt + (off_h_for_kv * q_k_ratio + q_mask_h[:, None]) * stride_qh + offs_d[None, :] * stride_qd,
                    mask=q_mask_h[:, None] < q_k_ratio)
        # convert original sparsse block size to vllm qv-cache block size
        num_blocks_per_sparse_block: tl.constexpr = SPARSE_BLOCK_SIZE // BLOCK_M
        k_block_end = q_pid + 1 # exclusive
        sparse_k_block_end = q_pid // SPARSE_BLOCK_SIZE + 1
        k_block_start = (sparse_k_block_end - num_local_blocks) * num_blocks_per_sparse_block
        if k_block_start < 0:
            k_block_start = 0
        # tl.device_print('> k_block_start',  k_block_start)
        # tl.device_print('> off_h, k_seqlen, q_pbid:', off_h, k_seqlen, q_pbid)

        m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M_LOADING, D_HEAD], dtype=tl.float32)

        K += off_h_for_kv * stride_kh
        V += off_h_for_kv * stride_vh

        if IS_FP8:
            q = q.to(tl.bfloat16)
        sm_scale *= 1.44269504 # 1/log2 as we use base2 for exponential and logorithm

        # TABLE_BLOCK_SIZE = BLOCK_N
        for k_block_id in range(k_block_start, k_block_end):
            # tl.device_print('k_block_col_idx', k_block_col_idx)
            k, v = _load_kv(
                k_block_id,
                K, V, Q,
                block_tables,
                stride_kb, stride_kd, stride_kt, stride_kx,
                stride_vb, stride_vd, stride_vt,
                stride_btb, stride_btt,
                off_z,
                kv_scale,  # kv_scale
                BLOCK_N,
                D_HEAD,
                X, # X = 16/ #bytes of k dtype
                IS_FP8, # IS_FP8,
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

        # TODO: split last block

        ### flash-attn 2
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]

        Out += off_z * stride_ot + (off_h_for_kv * q_k_ratio + num_heads) * stride_oh
        q_mask_h = tl.arange(0, BLOCK_M_LOADING)
        tl.store(Out + q_mask_h[:, None] * stride_oh + offs_d[None, :] * stride_od, acc,
                mask=q_mask_h[:, None] < q_k_ratio)

        M += off_z * stride_mt + (off_h_for_kv * q_k_ratio + num_heads) * stride_mh
        tl.store(M + q_mask_h * stride_mh, m_i, mask=q_mask_h < q_k_ratio)

    else:
        off_h_for_kv = off_h // q_k_ratio
        offs_m = tl.arange(0, BLOCK_M_LOADING)

        Q += off_z * stride_qt + off_h * stride_qh
        if EVEN_D:
            q = tl.load(Q + offs_d[None, :] * stride_qd)
        else:
            q = tl.load(Q + offs_d[None, :] * stride_qd, mask=(offs_d[None, :] < D_HEAD), other=0)

        # q = q.to(tl.float32)
        q = tl.broadcast_to(q, (BLOCK_M_LOADING, D_HEAD))

        sparse_crow_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + q_pbid * layout_crow_stride_m

        # TODO: load at once, supported in new Triton
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
                kv_scale,  # kv_scale
                BLOCK_N,
                D_HEAD,
                X, # X = 16/ #bytes of k dtype
                IS_FP8, # IS_FP8,
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
                    True, # can be safely set to False is it is remote only
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


    # if is_local:
    #     k_block_id = k_block_end - 1
    # else:
    #     k_block_id = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + (k_block_end - 1) * layout_col_stride_m).to(tl.int32)
    # k, v = _load_kv(
    #     k_block_id,
    #     K, V, Q,
    #     block_tables,
    #     stride_kb, stride_kd, stride_kt, stride_kx,
    #     stride_vb, stride_vd, stride_vt,
    #     stride_btb, stride_btt,
    #     off_z,
    #     kv_scale,  # kv_scale
    #     BLOCK_N,
    #     D_HEAD,
    #     X, # X = 16/ #bytes of k dtype
    #     IS_FP8, # IS_FP8
    #     BLOCK_D,
    #     EVEN_D
    #     )

    # acc, l_i, m_i = _fwd_kernel_inner_with_blocktable(
    #         acc, l_i, m_i,
    #         q, k, v, Q,
    #         k_block_id,
    #         offs_n,
    #         sm_scale,
    #         q_pid,
    #         True,
    #         BLOCK_M_LOADING,
    #         BLOCK_N)

    # ### flash-attn 2
    # m_i += tl.math.log2(l_i)
    # acc = acc / l_i[:, None]


    # if is_local:
    #     Out += off_z * stride_ot + (off_h_for_kv * q_k_ratio) * stride_oh +  num_heads * stride_oh
    #     q_mask_h = tl.arange(0, BLOCK_M_LOADING)
    #     tl.store(Out + q_mask_h[:, None] * stride_oh + offs_d[None, :] * stride_od, acc,
    #             mask=q_mask_h[:, None] < q_k_ratio)
        
    #     M += off_z * stride_mt + (off_h_for_kv * q_k_ratio) * stride_mh + num_heads * stride_mh
    #     tl.store(M + q_mask_h * stride_mh, m_i, mask=q_mask_h < q_k_ratio)
    # else:
    #     # write output
    #     offs_m = tl.arange(0, BLOCK_M_LOADING)
    #     if EVEN_D:
    #         tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
    #                 mask=offs_m[:, None] < 1)
    #     else:
    #         tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
    #                 mask=(offs_m[:, None] < 1) & (offs_d[None, :] < D_HEAD))

    #     M += off_z * stride_mt + off_h * stride_mh
    #     tl.store(M + offs_m * stride_mh, m_i, mask=offs_m < 1)


# @triton.jit
# def _fwd_kernel_batch_inference_split_k_with_blocktable(
#     Q, K, V, Out,

#     # QK,
#     # KO,
#     sm_scale,
#     q_batch_starts,
#     q_batch_ends,
#     k_batch_starts,
#     k_batch_ends,
#     q_batch_ids,
#     q_start_sids,

#     stride_qt, stride_qh, stride_qd,
#     stride_kb, stride_kh, stride_kd, stride_kt, stride_kx,
#     stride_vb, stride_vh, stride_vd, stride_vt,
#     stride_ot, stride_oh, stride_od, stride_os,
#     # stride_qk1, stride_qk2,
#     # stride_ko1, stride_ko2,

#     layout_crow_ptr,
#     layout_col_ptr,
#     layout_crow_stride_h, layout_crow_stride_m,
#     layout_col_stride_h, layout_col_stride_m,

#     q_k_ratio,
#     block_tables,
#     stride_btb, stride_btt,

#     kv_scale,
#     X: tl.constexpr,
#     IS_FP8: tl.constexpr,
#     PER_K_SPLIT: tl.constexpr,
#     D_HEAD: tl.constexpr,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_D: tl.constexpr,
#     BLOCK_M_LOADING: tl.constexpr,
#     EVEN_D: tl.constexpr,
# ):
#     '''
#     NOTATION:
#     pid: position id
#     sid: storage id
#     sbid: storage block id
#     pbid: position block id
#     offs_m, offs_n: storage offsets of m-dim(q, row) and n-dim(k, col)

#     q and blocks in KV needs to be contiguous

#     Arguments:
#     kv_seq_lens: for compute past_len
#     kv_storage_offsets: similar to block_tables in vllm, except it is dynamic.
#         TODO: fix this

#     TODO:
#     Optimize grouped-attn

#     CUDA graph support issue
#         1. grid is dynamic: vllm set up multiple cuda graph in decoding phase, with diff max token size (16, 32, ...)
#             since we mix prompt and decoing phase here, it can be more complex.
#             need to set up diff cuda-graph for diff (off_zm, off_z)

#             # indeed, q_batch_ids can be padded to maximum number of grid[0], i.e., assume all decoding
#             therefore, cu_seqlens_q, kv_seq_lens

#     '''
#     off_zm = tl.program_id(0)
#     off_h = tl.program_id(1)
#     k_split_id = tl.program_id(2)

#     off_h_for_kv = off_h // q_k_ratio
#     off_z = tl.load(q_batch_ids + off_zm).to(tl.int32)   # [0, 0, 0, 1]
#     q_start_sid = tl.load(q_start_sids + off_zm)
#     start_m = q_start_sid // BLOCK_M # q_sbid

#     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M_LOADING)
#     offs_n = tl.arange(0, BLOCK_N)
#     offs_d = tl.arange(0, BLOCK_D)

#     q_cu_start = tl.load(q_batch_starts + off_z).to(tl.int32)
#     q_seqlen = tl.load(q_batch_ends + off_z).to(tl.int32) - q_cu_start

#     k_cu_start = tl.load(k_batch_starts + off_z).to(tl.int32)
#     k_seqlen = tl.load(k_batch_ends + off_z).to(tl.int32) - k_cu_start

#     past_len = k_seqlen - q_seqlen
#     Q += q_cu_start * stride_qt + off_h * stride_qh

#     Out += q_cu_start * stride_ot + off_h * stride_oh  + k_split_id * stride_os
#     q_pbid = (past_len + q_start_sid) // BLOCK_M

#     # if EVEN_D:
#     #     q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
#     #                 mask=offs_m[:, None] < q_seqlen)
#     # else:
#     #     q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
#     #                 mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
#     #                 other=0)
#     if EVEN_D:
#         q = tl.load(Q + start_m * BLOCK_M * stride_qt + offs_d[None, :] * stride_qd)
#         q = tl.broadcast_to(q, BLOCK_M_LOADING, D_HEAD)
#     else:
#         q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
#                     mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
#                     other=0)


#     # tl.device_print('> past_len, q_seqlen, k_seqlen, q_pbid:',  past_len, q_seqlen, k_seqlen, q_pbid)
#     sparse_crow_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + q_pbid * layout_crow_stride_m

#     # TODO: load at once, supported in new Triton
#     k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
#     k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

#     m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float('inf')
#     l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32)
#     acc = tl.zeros([BLOCK_M_LOADING, D_HEAD], dtype=tl.float32)

#     K += off_h_for_kv * stride_kh
#     V += off_h_for_kv * stride_vh

#     # TABLE_BLOCK_SIZE = BLOCK_N
#     start = k_block_start + k_split_id * PER_K_SPLIT
#     end = start + PER_K_SPLIT

#     if end > k_block_end:
#         end = k_block_end

#     if start < k_block_end:
#         # for k_block_col_idx in range(k_block_start, k_block_end - 1):
#         for k_block_col_idx in range(start, end -1):
#             k_block_id = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + k_block_col_idx * layout_col_stride_m).to(tl.int32)

#             k, v = _load_kv(
#                 k_block_id,
#                 K, V, Q,
#                 block_tables,
#                 stride_kb, stride_kd, stride_kt, stride_kx,
#                 stride_vb, stride_vd, stride_vt,
#                 stride_btb, stride_btt,
#                 off_z,
#                 kv_scale,  # kv_scale
#                 BLOCK_N,
#                 D_HEAD,
#                 X, # X = 16/ #bytes of k dtype
#                 IS_FP8, # IS_FP8,
#                 BLOCK_D,
#                 EVEN_D
#                 )

#             acc, l_i, m_i = _fwd_kernel_inner_with_blocktable(
#                     acc, l_i, m_i,
#                     q, k, v, Q,
#                     k_block_id,
#                     offs_m, offs_n,
#                     sm_scale,
#                     past_len,
#                     False,
#                     BLOCK_M_LOADING,
#                     BLOCK_N)


#         k_block_id = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + (k_block_end - 1) * layout_col_stride_m).to(tl.int32)
#         k, v = _load_kv(
#             k_block_id,
#             K, V, Q,
#             block_tables,
#             stride_kb, stride_kd, stride_kt, stride_kx,
#             stride_vb, stride_vd, stride_vt,
#             stride_btb, stride_btt,
#             off_z,
#             kv_scale,  # kv_scale
#             BLOCK_N,
#             D_HEAD,
#             X, # X = 16/ #bytes of k dtype
#             IS_FP8, # IS_FP8
#             BLOCK_D,
#             EVEN_D
#             )

#         acc, l_i, m_i = _fwd_kernel_inner_with_blocktable(
#                 acc, l_i, m_i,
#                 q, k, v, Q,
#                 k_block_id,
#                 offs_m, offs_n,
#                 sm_scale,
#                 past_len,
#                 True,
#                 BLOCK_M_LOADING,
#                 BLOCK_N)


#         # write output
#         if EVEN_D:
#             tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
#                     mask=offs_m[:, None] < q_seqlen)
#         else:
#             tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
#                     mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD))



class LocalStridedBlockSparseAttnInferenceBT(torch.nn.Module):
    '''
    Support both varlen or fixed-len (with left or right paddings
    The `forward` method will dispatch to `self.varlen_attn` and `self.fixedlen_attn`
        if k.dim() == 4: `self.fixedlen_attn`
        if k.dim() == 3: `self.varlen_attn`
    Please checkout the docstring of the method to use.

    NOTE:
    1. Currently does not support autograd

    '''
    def __init__(self, n_heads, max_seqlen,
                 local_blocks, vert_stride, block_size,
                device=None, dtype=torch.bfloat16, homo_head=False,
                active_head_range=None,
                vllm_block_size=None,
                split_local_stride=True):
        super().__init__()
        device = device or torch.cuda.current_device()
        self.max_seqlen = max_seqlen
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.vert_stride = vert_stride
        sparse_layout, sparse_pattern, _ = _get_sparse_attn_mask(n_heads, max_seqlen, max_seqlen, dtype, device,
                                                BLOCK=block_size,
                                                local_blocks=local_blocks, vert_stride=vert_stride,
                                                homo_head=homo_head, return_dense=False)
        self.split_local_stride = split_local_stride
        if split_local_stride:
            sparse_layout, sparse_pattern = self.get_strided_layout(n_heads, max_seqlen, dtype, device,
                                                                    block_size, local_blocks, vert_stride,
                                                                    homo_head=homo_head, return_dense=False)
        # import ipdb; ipdb.set_trace()

        if (not homo_head) and (active_head_range is not None):
            assert isinstance(active_head_range, tuple)
            assert len(active_head_range) == 2, '"active_head_range" should be a tuple of start/end index of the heads.'
            h_start, h_end = active_head_range
            sparse_layout = tuple(x[h_start:h_end] for x in sparse_layout)
            sparse_pattern = sparse_pattern[h_start:h_end]

        self.sparse_layout = sparse_layout
        self.sparse_pattern = sparse_pattern

        self.vllm_block_size = None
        if vllm_block_size:
            self.set_vllm_block_size(vllm_block_size)

    def set_vllm_block_size(self, vllm_block_size):
        if self.vllm_block_size is not None:
            raise ValueError('vllm_block_size has been set')

        self.vllm_block_size = vllm_block_size
        sparse_block_size = self.block_size
        kernel_block_size = vllm_block_size

        assert sparse_block_size % kernel_block_size == 0
        # self.block_size = self.vllm_block_size
        if sparse_block_size // kernel_block_size > 1:
            _mul = sparse_block_size // kernel_block_size
            # need to consider if block_m and block_n are different
            sparse_pattern = torch.kron(self.sparse_pattern, self.sparse_pattern.new_ones(_mul, _mul))
            num_sparse_blocks = sparse_pattern.size(-1)
            block_causal_mask = torch.arange(0, num_sparse_blocks)[:, None] >= torch.arange(0, num_sparse_blocks)[None]
            sparse_pattern *= block_causal_mask.type_as(sparse_pattern)
            sparse_layout = dense_to_crow_col(sparse_pattern)
            self.sparse_layout = sparse_layout
            self.sparse_pattern = self.sparse_pattern

    def get_strided_layout(self, n_heads, max_seqlen, dtype, device, block_size, local_blocks, vert_stride,
                            homo_head=False, return_dense=False):
        sparse_layout, sparse_pattern, _ = _get_sparse_attn_mask(n_heads, max_seqlen, max_seqlen, dtype, device,
                                                BLOCK=block_size,
                                                local_blocks=local_blocks, vert_stride=vert_stride,
                                                homo_head=homo_head, return_dense=False)
    
        _, sparse_pattern_local, _ = _get_sparse_attn_mask(n_heads, max_seqlen, max_seqlen, dtype, device,
                                                BLOCK=block_size,
                                                local_blocks=local_blocks, vert_stride=max_seqlen + 1,
                                                homo_head=homo_head, return_dense=return_dense)
        sparse_pattern_strides = sparse_pattern - sparse_pattern_local
    
        sparse_layout_strides = dense_to_crow_col(sparse_pattern_strides)
        return sparse_layout_strides, sparse_pattern_strides

        # sparse_pattern =

    def forward(self, q, k, v, block_tables, context_lens, sm_scale=None, kv_scale=1.0):
        '''
        q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
                Support grouped attention, with `q[:, i*r:(i*r + r)]`
                is correspondent to `k[:, i]`, where `r` is the q/k ratio.
        sm_scale: softmax scale, default to 1/sqrt(head_size).

        return: tensor of shape as q.
        '''
        if self.sparse_layout[0].size(0) != 1:
            assert q.size(1) == self.sparse_layout[0].size(0)

        sm_scale = sm_scale or 1. / math.sqrt(q.size(-1))

        if self.vllm_block_size is None:
            self.set_vllm_block_size(v.size(-1))

        # TODO: auto extend length to next_power_of_2
        assert block_tables.size(1) * self.vllm_block_size <= self.max_seqlen

        return blocksparse_flash_attn_varlen_fwd_with_blocktable(q, k, v,
                    block_tables,
                    context_lens,
                    sm_scale,
                    self.sparse_layout,
                    sparse_block_size=self.block_size,
                    vllm_block_size=self.vllm_block_size,
                    num_local_blocks=self.local_blocks,
                    split_local_stride=self.split_local_stride,
                    max_seqlen=self.max_seqlen,
                    kv_scale=kv_scale)

    def backward(self, *args):
        raise NotImplementedError('> backward is not supported.')


if __name__ == '__main__':
    d = torch.load('/tmp/vllm.pt')
    # torch.save({'q': query, 'k': key_cache, 'v': value_cache,
    #             'output': output,
    #             'block_tables': attn_metadata.block_tables,
    #             'context_lens': attn_metadata.context_lens,
    #             'num_kv_heads':  self.num_kv_heads,
    #             'scale': self.scale,
    #             }, '/tmp/vllm.pt')


    # q: [num_tokens, h, D]
    # k: [num_blocks, h, D/x, block_size, x]
    # v: [num_blocks, h, D, block_size]
    q, k, v, ref_output, block_tables, context_lens, scale = d['q'], d['k'], d['v'], d['output'], d['block_tables'], d['context_lens'], d['scale']

    # n_heads = q.size(1)
    paged_attn = LocalStridedBlockSparseAttnInferenceBT(32, 8192, 16, 8, 64, vllm_block_size=v.size(-1))

    output = paged_attn(q, k, v, block_tables, context_lens, sm_scale=scale)

    print(f'> {block_tables=}, {context_lens=}')
    print(f'> {output.shape=}, {output=}')
    print(f'> {ref_output.shape=}, {ref_output=}')


    # print(f'> {k[block_tables[0]].shape=}')
    # k2 = k[block_tables[0]].permute(0, 3, 1, 2, 4).contiguous().reshape(-1, 8, 128)[:context_lens[0]] # (n_tokens, 8, 128)
    # v2 = v[block_tables[0]].permute(0, 3, 1, 2).contiguous().reshape(-1, 8, 128)[:context_lens[0]]


    # k2 = torch.repeat_interleave(k2, 4, dim=1)
    # v2 = torch.repeat_interleave(v2, 4, dim=1)
    # print(f'> {k2.shape=}, {v2.shape=}')
    # qk = torch.einsum('qhd,khd->hqk', q, k2)
    # p = (qk.float() * scale).softmax(-1).type_as(q)
    # ref_output2 = torch.einsum('hqk,khd->qhd', p, v2)[0]
    # print(f'>> {ref_output2.shape=}, {ref_output2=}')
    # # print(f'>> {p[0][0]=}')
    # qk = torch.einsum('qhd,khd->hqk', q.float(), k2.float())
    # print(f'{qk[0, :, :32]=}\n{qk[0, :, 32:]=}')
    # print(f'{k2[:32, 0]=}')

    # print(f'{(output - ref_output).abs().max()=}')


    from vllm.model_executor.models.tnlgv4_flash_blocksparse_attn_batch_inference import LocalStridedBlockSparseAttnInference

    k2 = torch.cat([k[block_tables[i]].permute(0, 3, 1, 2, 4).contiguous().reshape(-1, 8, 128)[:context_lens[i]]
                        for i in range(context_lens.size(0))
                        ], dim=0)
                    # (n_tokens, 8, 128)
    v2 = torch.cat([v[block_tables[i]].permute(0, 3, 1, 2).contiguous().reshape(-1, 8, 128)[:context_lens[i]]
                        for i in range(context_lens.size(0))
                    ], dim=0)

    cu_seqlens_k = context_lens.new_zeros((len(context_lens) + 1,))
    cu_seqlens_k[1:] = context_lens.cumsum(0)

    bs_op = LocalStridedBlockSparseAttnInference(32, 8192, 16, 8, 64)

    ref_output_varlen = bs_op(q, k2, v2, cu_seqlens_k, sm_scale=scale)

    print(f'> {ref_output_varlen.shape=}, {ref_output_varlen=}')

    print(f'{(output - ref_output_varlen).abs().max()=}')

    # _b()