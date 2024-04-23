import triton
import triton.language as tl
import torch
import math


def blocksparse_flash_attn_varlen_fwd(
    q, k, v, # (#tokens, n_heads, head_size)
    cu_seqlens_k,
    cu_seqlens_q,
    sm_scale,
    sparse_layout,
    *,
    block_size=64,
    q_block_size=None,
    max_seqlen = None
):
    # split q to blocks

    assert isinstance(sparse_layout, (list, tuple))

    _, n_heads, head_size = q.shape
    batch_size = cu_seqlens_k.size(0) - 1
    q_block_size = q_block_size or block_size

    assert q.dim() == k.dim() == v.dim() == 3
    assert q.size(1) % k.size(1) == 0
    assert q.size(2) == k.size(2)
    assert k.shape == v.shape # TODO: allow diff head_size for k, v
    assert cu_seqlens_k.dim() == 1

    q_k_ratio = q.size(1) // k.size(1)

    if cu_seqlens_q is None:
        if q.size(0) == batch_size: # decoding only
            cu_seqlens_q = torch.arange(0, batch_size + 1,
                                        dtype=cu_seqlens_k.dtype,
                                        device=cu_seqlens_k.device)
        elif q.size(0) == k.size(0):
            cu_seqlens_q = cu_seqlens_k
        else:
            raise ValueError('cu_seqlens_q must be specified if it is mix of prefilling and decoding.')
    else:
        assert cu_seqlens_k.size(0) == cu_seqlens_q.size(0)

    # switch to use cpu to avoid too many kernel lauch when iterate over
    q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu()
    k_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).cpu()

    assert torch.logical_or(q_lens == 1, k_lens == q_lens).all(), \
        'length of q should either be 1 (decoding) or same as k (prefilling).'

    if max_seqlen:
        assert k_lens.max() <= max_seqlen

    n_blocks = (q_lens + q_block_size - 1) // q_block_size

    q_batch_ids = torch.tensor([i for i, n in enumerate(n_blocks) for _ in range(n)],
                                dtype=cu_seqlens_q.dtype,
                                device=cu_seqlens_q.device)
    q_start_sids = torch.tensor([i * q_block_size for n in n_blocks for i in range(n)],
                               dtype=cu_seqlens_q.dtype,
                               device=cu_seqlens_q.device)

    out = q.new_empty(q.shape)
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()

    layout_crow_indices, layout_col_indices = sparse_layout
    block_d = triton.next_power_of_2(head_size)

    decoding_only =  (q_lens == 1).all().item()
    grid = (len(q_start_sids), n_heads, 1)

    _fwd_kernel_batch_inference[grid](
    q, k, v, out,
    sm_scale,
    cu_seqlens_q[:-1],
    cu_seqlens_q[1:],
    cu_seqlens_k[:-1],
    cu_seqlens_k[1:],
    q_batch_ids,
    q_start_sids,

    0, *q.stride(),
    0, *k.stride(),
    0, *v.stride(),
    0, *out.stride(),

    layout_crow_indices,
    layout_col_indices,
    *layout_crow_indices.stride(),
    *layout_col_indices.stride(),

    q_k_ratio,
    HAS_BATCH_DIM = False,
    D_HEAD = head_size,
    BLOCK_M = q_block_size,
    BLOCK_N = block_size,
    BLOCK_D = block_d,
    BLOCK_M_LOADING = 16 if decoding_only else q_block_size, # smaller for decoding
    EVEN_D = block_d == head_size,
    num_warps = 1 if decoding_only else 4,
    num_stages = 3
    )

    return out


@triton.jit
def _fwd_kernel_inner(
    acc, l_i, m_i,
    q, Q,
    k_block_col_idx,
    layout_col_ptr,
    layout_col_stride_h, layout_col_stride_m,
    k_ptrs,
    v_ptrs,
    off_h, offs_m, offs_n, offs_d,
    stride_kt, stride_vt,
    sm_scale,
    k_seqlen,
    past_len,
    LAST_K_BLOCK: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr
):
    k_block_id = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + k_block_col_idx * layout_col_stride_m).to(tl.int32)
    start_n = k_block_id * BLOCK_N
    # -- compute qk ----
    if LAST_K_BLOCK:
        if EVEN_D:
            k = tl.load(k_ptrs + start_n * stride_kt,
                        mask=offs_n[None, :] + start_n < k_seqlen)
        else:
            # mask = mask & (offs_d[:, ])
            k = tl.load(k_ptrs + start_n * stride_kt,
                        mask=(offs_n[None, :] + start_n < k_seqlen) & (offs_d[:, None] < D_HEAD))
    else:
        if EVEN_D:
            k = tl.load(k_ptrs + start_n * stride_kt)
        else:
            k = tl.load(k_ptrs + start_n * stride_kt,
                        mask=offs_d[:, None] < D_HEAD)


    qk = tl.zeros([BLOCK_M_LOADING, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)

    qk *= sm_scale

    # the following is needed only when LAST_K_BLOCK or BLOCK_M < BLOCK_N
    if LAST_K_BLOCK | M_LT_N:
        qk += tl.where(offs_m[:, None] + past_len >= (start_n + offs_n[None, :]), 0, float('-inf'))

    # ### flash-attn 1
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
    if LAST_K_BLOCK:
        if EVEN_D:
            v = tl.load(v_ptrs + start_n * stride_vt,
                        mask=offs_n[:, None] + start_n < k_seqlen)
        else:
            v = tl.load(v_ptrs + start_n * stride_vt,
                        mask=(offs_n[:, None] + start_n < k_seqlen) & (offs_d[None, :] < D_HEAD))
    else:
        if EVEN_D:
            v = tl.load(v_ptrs + start_n * stride_vt)
        else:
            v = tl.load(v_ptrs + start_n * stride_vt,
                        mask=offs_d[None, :] < D_HEAD)

    acc += tl.dot(p, v)

    return acc, l_i, m_i


@triton.heuristics(
    {
        'M_LT_N': lambda kwargs: kwargs['BLOCK_M'] < kwargs['BLOCK_N'],
    }
)
@triton.jit
def _fwd_kernel_batch_inference(
    Q, K, V, Out,

    sm_scale,
    q_batch_starts,
    q_batch_ends,
    k_batch_starts,
    k_batch_ends,
    q_batch_ids,
    q_start_sids,

    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,

    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,

    q_k_ratio,

    HAS_BATCH_DIM: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M_LOADING: tl.constexpr,
    EVEN_D: tl.constexpr,
    M_LT_N: tl.constexpr
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
        q_start_sid = start_m * BLOCK_M # always 0 for decoding
    else:
        off_z = tl.load(q_batch_ids + off_zm).to(tl.int32)   # [0, 0, 0, 1]
        q_start_sid = tl.load(q_start_sids + off_zm)
        start_m = q_start_sid // BLOCK_M # q_sbid

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M_LOADING)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # q_cu_start = tl.load(cu_seqlens_q + off_z).to(tl.int32)
    # q_seqlen = tl.load(cu_seqlens_q + off_z + 1).to(tl.int32) - q_cu_start

    # k_cu_start = tl.load(cu_seqlens_k + off_z).to(tl.int32)
    # k_seqlen = tl.load(cu_seqlens_k + off_z + 1).to(tl.int32) - k_cu_start

    q_cu_start = tl.load(q_batch_starts + off_z).to(tl.int32)
    q_seqlen = tl.load(q_batch_ends + off_z).to(tl.int32) - q_cu_start

    k_cu_start = tl.load(k_batch_starts + off_z).to(tl.int32)
    k_seqlen = tl.load(k_batch_ends + off_z).to(tl.int32) - k_cu_start

    # k_cu_start = q_cu_start
    # k_cu_end = q_cu_end
    # k_seqlen = k_cu_end - k_cu_start
    past_len = k_seqlen - q_seqlen

    Q += q_cu_start * stride_qt + off_h * stride_qh
    K += k_cu_start * stride_kt + off_h_for_kv * stride_kh
    V += k_cu_start * stride_vt + off_h_for_kv * stride_vh
    Out += q_cu_start * stride_ot + off_h * stride_oh

    q_pbid = (past_len + q_start_sid) // BLOCK_M

    if EVEN_D:
        q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=offs_m[:, None] < q_seqlen)
    else:
        q = tl.load(Q + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
                    mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD),
                    other=0)

    # tl.device_print('> past_len, q_seqlen, k_seqlen, q_pbid:',  past_len, q_seqlen, k_seqlen, q_pbid)
    sparse_crow_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + q_pbid * layout_crow_stride_m

    # TODO: load at once, supported in new Triton
    k_block_start = tl.load(sparse_crow_ptr).to(tl.int32)
    k_block_end = tl.load(sparse_crow_ptr + 1).to(tl.int32)

    m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_LOADING, BLOCK_D], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd

    sm_scale *= 1.44269504 # 1/log2 as we use base2 for exponential and logorithm

    for k_block_col_idx in range(k_block_start, k_block_end - 1):
        acc, l_i, m_i = _fwd_kernel_inner(
            acc, l_i, m_i,
            q, Q,
            k_block_col_idx,
            layout_col_ptr,
            layout_col_stride_h, layout_col_stride_m,
            k_ptrs,
            v_ptrs,
            off_h, offs_m, offs_n, offs_d,
            stride_kt, stride_vt,
            sm_scale,
            k_seqlen,
            past_len,
            False,
            BLOCK_M_LOADING,
            BLOCK_N,
            D_HEAD,
            EVEN_D,
            M_LT_N
            )

    acc, l_i, m_i = _fwd_kernel_inner(
        acc, l_i, m_i,
        q, Q,
        k_block_end - 1,
        layout_col_ptr,
        layout_col_stride_h, layout_col_stride_m,
        k_ptrs,
        v_ptrs,
        off_h, offs_m, offs_n, offs_d,
        stride_kt, stride_vt,
        sm_scale,
        k_seqlen,
        past_len,
        True,
        BLOCK_M_LOADING,
        BLOCK_N,
        D_HEAD,
        EVEN_D,
        M_LT_N
        )

    ### flash-attn 2
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    # write output
    if EVEN_D:
        tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
                mask=offs_m[:, None] < q_seqlen)
    else:
        tl.store(Out + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od, acc,
                mask=(offs_m[:, None] < q_seqlen) & (offs_d[None, :] < D_HEAD))


# test
if __name__ == '__main__':
    import torch

    from flash_attn import flash_attn_varlen_func
    from torch.nn.functional import scaled_dot_product_attention
    from triton_flash_blocksparse_attn import get_local_strided_sparse_attention_op

    q_seqlens = torch.tensor([0, 61, 65, 7193, 118, 1371], dtype=torch.int32, device='cuda')  # first one always 0, not a sample
    # q_seqlens = torch.tensor([0, 67], dtype=torch.int32, device='cuda')

    BATCH, N_KV_HEADS, MAX_SEQ, D_HEAD = len(q_seqlens) - 1, 32, 8192, 96
    LOCAL, VERT, BLOCK_SIZE = 16, 8, 64
    sm_scale = 0.01

    q_k_ratio = 1
    N_HEADS = N_KV_HEADS * q_k_ratio

    PADDED_D_HEAD = triton.next_power_of_2(D_HEAD)

    torch.manual_seed(124)
    q = torch.empty((BATCH, MAX_SEQ, N_HEADS, PADDED_D_HEAD), dtype=torch.bfloat16, device='cuda').normal_(mean=0, std=.5) # .requires_grad_()
    k = torch.empty((BATCH, MAX_SEQ, N_KV_HEADS, PADDED_D_HEAD), dtype=torch.bfloat16, device='cuda').normal_(mean=0, std=.5) # .requires_grad_()
    v = torch.empty((BATCH, MAX_SEQ, N_KV_HEADS, PADDED_D_HEAD), dtype=torch.bfloat16, device='cuda').normal_(mean=0, std=.5) # .requires_grad_()

    q[..., D_HEAD:] = 0
    k[..., D_HEAD:] = 0
    v[..., D_HEAD:] = 0

    cu_seqlens = q_seqlens.cumsum(0).to(torch.int32)
    total_tokens = cu_seqlens[-1].item()

    # mask_csr, _, mask_dense = get_sparse_attn_mask(q, MAX_SEQ, BLOCK=sparse_block_size,
    #                         local_blocks=local_blocks, vert_stride=vert_stride, homo_head=homo_head, return_dense=True)

    mask_csr, _, mask_dense =_get_sparse_attn_mask(N_HEADS, MAX_SEQ, MAX_SEQ, q.dtype, q.device, BLOCK=BLOCK_SIZE,
                                                   local_blocks=LOCAL, vert_stride=VERT,
                                                   homo_head=False, return_dense=True)


    q_packed = q.new_empty((total_tokens, N_HEADS, D_HEAD))
    k_packed, v_packed = [q.new_empty((total_tokens, N_KV_HEADS, D_HEAD)) for _ in range(2)]

    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        q_packed[s:e] = q[i, :e - s, :, :D_HEAD]
        k_packed[s:e] = k[i, :e - s, :, :D_HEAD]
        v_packed[s:e] = v[i, :e - s, :, :D_HEAD]

    bs_attn = LocalStridedBlockSparseAttnInference(N_HEADS, MAX_SEQ, LOCAL, VERT, BLOCK_SIZE)

    bs_attn_spda = LocalStridedBlockSparseAttnInference(N_HEADS, MAX_SEQ, LOCAL, VERT, BLOCK_SIZE, use_spda=True)

    # non-contiguous passed
    # k_packed = k_packed.transpose(0, 1).contiguous().transpose(0, 1)
    # q_packed = q_packed.transpose(0, 1).contiguous().transpose(0, 1)

    out_packed = bs_attn(q_packed, k_packed, v_packed, cu_seqlens, sm_scale=sm_scale)

    out_packed_spda = bs_attn_spda(q_packed, k_packed, v_packed, cu_seqlens, sm_scale=sm_scale)
    # out_packed = blocksparse_flash_attn_varlen_fwd(q_packed, k_packed, v_packed,
    #                                             cu_seqlens, sm_scale, mask_csr, block_size=BLOCK_SIZE)


    sparse_attention_fn = get_local_strided_sparse_attention_op(N_HEADS, MAX_SEQ,
                                                            sparse_block_size=BLOCK_SIZE,
                                                            local_blocks=LOCAL,
                                                            vert_stride=VERT,
                                                            homo_head=False,
                                                            device=q.device,
                                                            dtype=q.dtype,
                                                            kernel_block_size=BLOCK_SIZE)


    max_q_len =  q_seqlens.max()
    # ref_out_flash = flash_attn_varlen_func(q_packed, k_packed, v_packed,
                                            # cu_seqlens, cu_seqlens, max_q_len, max_q_len,
                                            # softmax_scale=sm_scale, causal=True)

    k_expand, v_expand = [x.repeat_interleave(q_k_ratio, dim=2).transpose(1, 2) for x in [k, v]]
    ref_out = sparse_attention_fn(q.transpose(1, 2), k_expand, v_expand, sm_scale)
    ref_out = ref_out.transpose(1, 2).contiguous()

    print(f'>> {ref_out.shape=}, {out_packed.shape=}')
    ref_out_packed = torch.empty_like(out_packed)

    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        ref_out_packed[s:e] = ref_out[i, :e - s, :, :D_HEAD]


    # print(f'--->\n{out_packed=}\n')
    # print(f'--->\n{ref_out_packed=}\n')
    assert torch.allclose(out_packed, ref_out_packed, atol=1e-2, rtol=0)
    print('> prefilling phase test passed\n======\n')

    assert torch.allclose(out_packed, out_packed_spda, atol=1e-2, rtol=0)
    print('> prefilling phase using SPDA test passed\n======\n')

    exit()


    # ref_out_part = scaled_dot_product_attention(q[:, 64:128].transpose(1, 2).contiguous(),
    #                                             k[:, 64:128].transpose(1, 2).contiguous(),
    #                                             v[:, 64:128].transpose(1, 2).contiguous(),
    #                                             scale=sm_scale, is_causal=True)


    # assert torch.allclose(out_packed, ref_out_packed, atol=1e-2, rtol=0)


    #### test 2: decoding, past_len

    # cu_seqlens_k = cu_seqlens
    # cu_seqlens_q_dec = torch.cumsum(torch.tensor([0] + [1] * BATCH, dtype=cu_seqlens_k.dtype, device=q.device), dim=0)

    q_packed_dec = q.new_empty((BATCH, N_HEADS, D_HEAD))
    ref_out_packed_dec = torch.empty_like(q_packed_dec)

    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        q_packed_dec[i] = q[i, e - s - 1:e - s, :, :D_HEAD]
        ref_out_packed_dec[i] = ref_out[i, e - s - 1: e - s, :, :D_HEAD]

    out_packed_dec = bs_attn(q_packed_dec, k_packed, v_packed, cu_seqlens, sm_scale=sm_scale)

    # print(f'> {out_packed_dec=} \n {ref_out_packed_dec=}\n===')
    # print(f'> {out_packed_dec.shape=}, {ref_out_packed_dec.shape=}')
    assert torch.allclose(out_packed_dec, ref_out_packed_dec, atol=1e-2, rtol=0)

    print('> decoding phase test passed\n======\n')


    # test 3: right padding prefiling

    seqlens = q_seqlens[1:]

    out_right_padding = bs_attn(q, k, v, sm_scale=sm_scale, seqlens=seqlens)

    out_right_padding_packed = torch.empty_like(out_packed)
    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        out_right_padding_packed[s:e] = out_right_padding[i, :e - s, :, :D_HEAD]


    # print(f'> {out_right_padding_packed=}\n {ref_out_packed=}')

    assert torch.allclose(out_right_padding_packed, ref_out_packed, atol=1e-2, rtol=0), \
            f'{(out_right_padding_packed - ref_out_packed).abs().max()=}'

    print('> prefilling phase with right paddding test passed\n======\n')


    # test 4: left padding prefilling
    ql, kl, vl = [torch.zeros_like(x)[..., :D_HEAD] for x in (q, k, v)]
    left_paddings = []

    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        left_padding = k.size(1) - (e - s)
        ql[i, left_padding:] = q[i, :e - s, :, :D_HEAD]
        kl[i, left_padding:] = k[i, :e - s, :, :D_HEAD]
        vl[i, left_padding:] = v[i, :e - s, :, :D_HEAD]
        left_paddings.append(left_padding)

    left_paddings = torch.tensor(left_paddings).type_as(cu_seqlens)

    out_left_padding = bs_attn(ql, kl, vl, left_paddings=left_paddings, sm_scale=sm_scale)

    out_left_padding_packed = torch.empty_like(out_packed)
    for i, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        out_left_padding_packed[s:e] = out_left_padding[i, s - e:]

    assert torch.allclose(out_left_padding_packed, ref_out_packed, atol=1e-2, rtol=0), \
            f'{(out_left_padding_packed - ref_out_packed).abs().max()=}'

    print('> prefilling phase with left paddding test passed\n======\n')


    # test 5: elft padding decoding

    out_left_padding_dec =  bs_attn(ql[:, -1:], kl, vl, left_paddings=left_paddings, sm_scale=sm_scale)[:, 0]
    # print(f'> {out_left_padding_dec=} \n {ref_out_packed_dec=}\n===')
    # print(f'> {out_left_padding_dec.shape=}, {ref_out_packed_dec.shape=}')
    assert torch.allclose(out_left_padding_dec, ref_out_packed_dec, atol=1e-2, rtol=0)

    print('> decoding phase with left paddding test passed\n======\n')