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
    # print(f'> {q.shape=}, {k.shape=}, {v.shape=}, {block_tables.shape=}, {context_lens.shape=}')

    assert mode in ('split', 'combine', 'local-only', 'remote-only')
    split_local_remote = (mode == 'split')

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

    # the following is only needed to determined q/k len
    context_lens = context_lens.contiguous()
    layout_crow_indices, layout_col_indices = sparse_layout

    # assert block_size == vllm_block_size
    assert sparse_block_size % vllm_block_size == 0
    block_d = triton.next_power_of_2(head_size)

    # print(f'>>> {q.dim()=}, {k.dim()=}, {v.dim()=}, {out.dim()=}, {m.dim()=}, {layout_crow_indices.dim()=}, {layout_col_indices.dim()=}')
    # qk = torch.zeros((vllm_block_size, vllm_block_size), dtype=torch.float32, device=q.device)
    # ko = torch.zeros((vllm_block_size, head_size)).type_as(k)

    IS_FP8 = k.element_size() == 1
    X = 16 // k.element_size() # fixed in vllm

    if split_local_remote:
        start_local_head_idx = n_heads
        out = q.new_zeros((q.size(0), q.size(1) * 2, q.size(2)))
        m = q.new_zeros((q.size(0), q.size(1) * 2), dtype=torch.float32) - float('inf')

        # is_locals = context_lens.new_zeros((n_heads + n_heads // q_k_ratio,))
        # is_locals[n_heads:] = 1

        grid = (batches, n_heads + n_heads // q_k_ratio, 1)
    else:
        m = q.new_empty((q.size(0), q.size(1)), dtype=torch.float32)
        out = q.new_zeros(q.shape)
        if mode == 'local-only':
            start_local_head_idx = 0
            grid = (batches, n_heads // q_k_ratio, 1)
        else:
            start_local_head_idx = n_heads + 1 # is_local will always be false
            grid = (batches, n_heads, 1)


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

    # is_locals,
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

    start_local_head_idx: tl.constexpr,
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
    off_z = tl.program_id(0)
    off_h = tl.program_id(1)

    is_local = off_h >= start_local_head_idx

    # tl.device_print(f'>> is_local:', is_local)
    # is_local = tl.load(is_locals + off_h)

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
        # tl.device_print('> off_h, k_seqlen, q_pbid:', off_h, k_seqlen, q_pbid)

        m_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M_LOADING], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M_LOADING, D_HEAD], dtype=tl.float32)

        K += off_h_for_kv * stride_kh
        V += off_h_for_kv * stride_vh

        if IS_FP8:
            q = q.to(tl.bfloat16)
        sm_scale *= 1.44269504 # 1/log2 as we use base2 for exponential and logorithm

        # tl.device_print('> k_block_end',  k_block_end)

        for k_block_id in range(k_block_start, k_block_end - 1):
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


if __name__ == '__main__':
    import torch

    from flash_attn import flash_attn_varlen_func
    from torch.nn.functional import scaled_dot_product_attention
    from triton_flash_blocksparse_attn import get_local_strided_sparse_attention_op
    from vllm.model_executor.models.phi3small_flash_blocksparse_attn_batch_inference import LocalStridedBlockSparseAttnInference

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

    # non-contiguous passed
    # k_packed = k_packed.transpose(0, 1).contiguous().transpose(0, 1)
    # q_packed = q_packed.transpose(0, 1).contiguous().transpose(0, 1)

    out_packed = bs_attn(q_packed, k_packed, v_packed, cu_seqlens, sm_scale=sm_scale)
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
    
    ### test 3: block table

    NUM_BLOCKS = 12229
    
    # (num_blocks, n_heads, head_size / x, vllm_block_size, x)
    # (num_blocks, n_heads, head_size, vllm_block_size)


if __name__ == '__test__':
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


    from vllm.model_executor.models.phi3small_flash_blocksparse_attn_batch_inference import LocalStridedBlockSparseAttnInference

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