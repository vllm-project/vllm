import torch
import time
import itertools
import triton
import triton.language as tl
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def bench(func, iterations=10):
    """
    Benchmarks a function by running it a number of times and returning the average and standard deviation of execution time.

    :param func: The function to be benchmarked.
    :param iterations: Number of times the function will be executed. Default is 10.
    :return: An object with 'avg' and 'std' methods to get the average and standard deviation of execution times.
    """
    times = []

    for _ in range(iterations):
        start_time = time.time()
        func()
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)

    times = np.array(times)
    avg_time = np.mean(times)
    std_dev_time = np.std(times)

    class Result:
        def avg(self):
            return avg_time

        def std(self):
            return std_dev_time

    return Result()

def gc_torch():
    pass

if triton.__version__ >= "2.1.0":
    @triton.jit
    def _fwd_kernel(
        Q, K, V, K_cache, V_cache, B_Loc, sm_scale, B_Start_Loc, B_Seqlen, B_Ctxlen, block_size, x,
        Out,
        stride_b_loc_b, stride_b_loc_s,
        stride_qbs, stride_qh, stride_qd,
        stride_kbs, stride_kh, stride_kd,
        stride_vbs, stride_vh, stride_vd,
        stride_obs, stride_oh, stride_od,
        stride_k_cache_bs, stride_k_cache_h, stride_k_cache_d, stride_k_cache_bl, stride_k_cache_x,
        stride_v_cache_bs, stride_v_cache_h, stride_v_cache_d, stride_v_cache_bl,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd

        q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len, other=0.0)

        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b + ((start_n + offs_n) // block_size) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_ctx_len, other=0)
            off_k = bn[None, :] * stride_k_cache_bs + cur_head * stride_k_cache_h + (offs_d[:, None] // x) * stride_k_cache_d + ((start_n + offs_n[None,:]) % block_size) * stride_k_cache_bl + (offs_d[:, None] % x) * stride_k_cache_x
            off_v = bn[:, None] * stride_v_cache_bs + cur_head * stride_v_cache_h + offs_d[None,:] * stride_v_cache_d + (start_n + offs_n[:,None]) % block_size * stride_v_cache_bl
            k = tl.load(K_cache + off_k, mask=(start_n + offs_n[None,:]) < cur_batch_ctx_len, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf"))
            qk *= sm_scale

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(V_cache + off_v, mask=(start_n + offs_n[:, None]) < cur_batch_ctx_len, other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        
        off_k = offs_n[None, :] * stride_kbs + cur_head * stride_kh + offs_d[:, None] * stride_kd
        off_v = offs_n[:, None] * stride_vbs + cur_head * stride_vh + offs_d[None, :] * stride_vd
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=(start_n + offs_n[None, :]) < cur_batch_seq_len - cur_batch_ctx_len, other=0.0)
            # mask = tl.load(mask_ptrs + start_n, mask=start_n + offs_n < cur_batch_end_loc, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=(start_n + offs_n[:, None]) < cur_batch_seq_len - cur_batch_ctx_len, other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len)
        return

    @torch.inference_mode()
    def context_attention_fwd(q, k, v, o, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, b_ctx_len, max_input_len):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}

        sm_scale = 1.0 / (Lq**0.5)  # 计算scale系数
        batch, head = b_seq_len.shape[0], q.shape[1]

        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

        num_warps = 8 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q, k, v, k_cache, v_cache, b_loc, sm_scale, b_start_loc, b_seq_len, b_ctx_len, v_cache.shape[3], 8,
            o,
            b_loc.stride(0), b_loc.stride(1),
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3), k_cache.stride(4), #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3), #[num_blocks, num_kv_heads, head_size, block_size]
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    import random
    random.seed(0)
    torch.manual_seed(0)
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask
    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    subquery_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(subquery_lens, ctx_lens)]
    
    num_tokens = sum(subquery_lens)
    query = torch.empty(num_tokens,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens,
                    num_heads,
                    head_size,
                    dtype=dtype,
                    device='cuda')
                    
    
    kv = torch.empty(sum(seq_lens),
                      2,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    kv.uniform_(-1e-3, 1e-3)
    key,value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size, block_size, num_heads, head_size, dtype=dtype, device='cuda')
    v_cache = torch.zeros(cache_size, block_size, num_heads, head_size, dtype=dtype, device='cuda')
    k = torch.zeros(sum(subquery_lens), num_heads, head_size, dtype=dtype, device='cuda')
    v = torch.zeros(sum(subquery_lens), num_heads, head_size, dtype=dtype, device='cuda')
    values = torch.arange(0, cache_size, dtype=torch.long, device='cuda')
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(BS, max_block_per_request)
    b_loc = torch.zeros(BS, MAX_CTX_LEN, dtype=torch.long, device='cuda')  
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long, device='cuda')
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long, device='cuda')
    b_start_loc = torch.cumsum(torch.tensor([0] + subquery_lens[:-1], dtype=torch.long, device='cuda'), dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1], dtype=torch.long, device='cuda'), dim=0)
    for i in range(BS):
        for j in range(subquery_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i,block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_heads, head_size)[start_slot:end_slot].copy_(key[start_loc:end_loc])
            v_cache.view(-1, num_heads, head_size)[start_slot:end_slot].copy_(value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size] to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_heads, head_size//8, 8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size] to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_heads, head_size).permute(0, 2, 3, 1).contiguous()
    
    
    context_attention_fwd(query, k, v, output, 
                          k_cache, v_cache, block_table, 
                          b_start_loc, b_seq_len, 
                          b_ctx_len, max_input_len)
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(query, k, v, output, 
                          k_cache, v_cache, block_table, 
                          b_start_loc, b_seq_len, 
                          b_ctx_len, max_input_len)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    attn_op = xops.fmha.cutlass.FwOp()
    
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(subquery_lens, seq_lens)
    output_ref = xops.memory_efficient_attention_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = xops.memory_efficient_attention_forward(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
    output_ref = output_ref.squeeze(0)
    print(output_ref.shape)
    print("max ", torch.max(torch.abs(output_ref - output)))
    print("mean ", torch.mean(torch.abs(output_ref - output)))
    print(output[0,0,:10])
    print(output_ref[0,0,:10])
    assert torch.allclose(output_ref, output, atol=1e-6, rtol=0)

@torch.inference_mode()
def bench_contexted_kv_attention(
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    import random
    random.seed(0)
    torch.manual_seed(0)
    from xformers import ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask

    # seq_len = [16, 64, 128, 256, 512, 1024]
    # ctx_len = [16, 64, 128, 256, 512, 1024, 2048]
    seq_len = [16, 64, 128, 256, 512, 1024]
    ctx_len = [256, 512, 1024, 2048]
    BS = 20
    timings_triton = {}
    timings_xformer = {}
    for MAX_SEQ_LEN, MAX_CTX_LEN in itertools.product(seq_len, ctx_len):
        gc_torch()
        # MAX_SEQ_LEN = 1024
        # MAX_CTX_LEN = 2048
        outputs = [
        f"seq_len={MAX_SEQ_LEN}",
        f"ctx_len={MAX_CTX_LEN}",
        f"bs={BS}"
        ]
        cache_size = 40960
        block_size = 1
        max_block_per_request = 2048
        subquery_lens = [random.randint(MAX_SEQ_LEN, MAX_SEQ_LEN) for _ in range(BS)]
        ctx_lens = [random.randint(MAX_CTX_LEN, MAX_CTX_LEN) for _ in range(BS)]
        seq_lens = [a + b for a, b in zip(subquery_lens, ctx_lens)]
        
        num_tokens = sum(subquery_lens)
        query = torch.empty(num_tokens,
                        num_heads,
                        head_size,
                        dtype=dtype,
                        device='cuda')
        query.uniform_(-1e-3, 1e-3)
        output = torch.empty(num_tokens,
                        num_heads,
                        head_size,
                        dtype=dtype,
                        device='cuda')
                        
        
        kv = torch.empty(sum(seq_lens),
                        2,
                        num_heads,
                        head_size,
                        dtype=dtype,
                        device='cuda')
        kv.uniform_(-1e-3, 1e-3)
        key,value = kv.unbind(dim=1)

        k_cache = torch.zeros(cache_size, block_size, num_heads, head_size, dtype=dtype, device='cuda')
        v_cache = torch.zeros(cache_size, block_size, num_heads, head_size, dtype=dtype, device='cuda')
        k = torch.zeros(sum(subquery_lens), num_heads, head_size, dtype=dtype, device='cuda')
        v = torch.zeros(sum(subquery_lens), num_heads, head_size, dtype=dtype, device='cuda')
        values = torch.arange(0, cache_size, dtype=torch.long, device='cuda')
        values = values[torch.randperm(cache_size)]
        block_table = values[:BS * max_block_per_request].view(BS, max_block_per_request)
        b_loc = torch.zeros(BS, MAX_CTX_LEN, dtype=torch.long, device='cuda')  
        b_seq_len = torch.tensor(seq_lens, dtype=torch.long, device='cuda')
        b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long, device='cuda')
        b_start_loc = torch.cumsum(torch.tensor([0] + subquery_lens[:-1], dtype=torch.long, device='cuda'), dim=0)
        max_input_len = MAX_SEQ_LEN
        # copy kv to cache
        b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1], dtype=torch.long, device='cuda'), dim=0)
        for i in range(BS):
            for j in range(subquery_lens[i]):
                k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] + j])
                v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] + b_ctx_len[i] + j])
            cur_ctx = 0
            block_id = 0
            while cur_ctx < b_ctx_len[i]:
                start_loc = b_seq_start_loc[i] + cur_ctx
                if cur_ctx + block_size > b_ctx_len[i]:
                    end_loc = b_seq_start_loc[i] + b_ctx_len[i]
                else:
                    end_loc = start_loc + block_size
                start_slot = block_table[i,block_id] * block_size
                end_slot = start_slot + end_loc - start_loc
                k_cache.view(-1, num_heads, head_size)[start_slot:end_slot].copy_(key[start_loc:end_loc])
                v_cache.view(-1, num_heads, head_size)[start_slot:end_slot].copy_(value[start_loc:end_loc])
                cur_ctx += block_size
                block_id += 1
        # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size] to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
        k_cache = k_cache.view(-1, block_size, num_heads, head_size//8, 8).permute(0, 2, 3, 1, 4).contiguous()
        # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size] to V_cache[num_blocks, num_kv_heads, head_size, block_size]
        v_cache = v_cache.view(-1, block_size, num_heads, head_size).permute(0, 2, 3, 1).contiguous()
        
        
        context_attention_fwd(query, k, v, output, 
                            k_cache, v_cache, block_table, 
                            b_start_loc, b_seq_len, 
                            b_ctx_len, max_input_len)
        torch.cuda.synchronize()
        start_time = time.time()
        context_attention_fwd(query, k, v, output, 
                            k_cache, v_cache, block_table, 
                            b_start_loc, b_seq_len, 
                            b_ctx_len, max_input_len)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")

        scale = float(1.0 / (head_size**0.5))

        attn_op = xops.fmha.cutlass.FwOp()
        
        attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(subquery_lens, seq_lens)
        output_ref = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=scale,
            op=attn_op,
        )
        torch.cuda.synchronize()
        start_time = time.time()
        output_ref = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=scale,
            op=attn_op,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
        output_ref = output_ref.squeeze(0)
        # print(output_ref.shape)
        # print("max ", torch.max(torch.abs(output_ref - output)))
        # print("mean ", torch.mean(torch.abs(output_ref - output)))
        # print(output[0,0,:10])
        # print(output_ref[0,0,:10])
        assert torch.allclose(output_ref, output, atol=1e-6, rtol=0)

        result = bench(lambda: xops.memory_efficient_attention_forward(query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=scale,
            op=attn_op))
        outputs.append(f"\n xformer: {result.avg()*1e6:3.0f}us±{result.std()*1e6:3.0f}us")
        timings_xformer[(MAX_SEQ_LEN, MAX_CTX_LEN)] = result.avg()*1e6
        result2 = bench(lambda: context_attention_fwd(query, k, v, output, 
                            k_cache, v_cache, block_table, 
                            b_start_loc, b_seq_len, 
                            b_ctx_len, max_input_len))
        outputs.append(f"\n triton: {result2.avg()*1e6:3.0f}us±{result2.std()*1e6:3.0f}us")
        timings_triton[(MAX_SEQ_LEN, MAX_CTX_LEN)] = result2.avg()*1e6
        print(" | ".join(outputs))
    

# test_contexted_kv_attention(12, 128, torch.float16)
# bench_contexted_kv_attention(12, 128, torch.float16)