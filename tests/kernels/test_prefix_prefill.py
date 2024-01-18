import random
import pytest
import time

import torch
from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
    context_attention_fwd)
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask

NUM_HEADS = [12]
HEAD_SIZES = [128]
DTYPES = [torch.float16]


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
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
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_heads,
                          head_size,
                          dtype=dtype,
                          device='cuda')
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_heads,
                          head_size,
                          dtype=dtype,
                          device='cuda')
    k = torch.zeros(sum(subquery_lens),
                    num_heads,
                    head_size,
                    dtype=dtype,
                    device='cuda')
    v = torch.zeros(sum(subquery_lens),
                    num_heads,
                    head_size,
                    dtype=dtype,
                    device='cuda')
    values = torch.arange(0, cache_size, dtype=torch.long, device='cuda')
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long, device='cuda')
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long, device='cuda')
    b_start_loc = torch.cumsum(torch.tensor([0] + subquery_lens[:-1],
                                            dtype=torch.long,
                                            device='cuda'),
                               dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long,
                                                device='cuda'),
                                   dim=0)
    for i in range(BS):
        for j in range(subquery_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc])
            v_cache.view(-1, num_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()

    context_attention_fwd(query, k, v, output, k_cache, v_cache, block_table,
                          b_start_loc, b_seq_len, b_ctx_len, max_input_len)
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(query, k, v, output, k_cache, v_cache, block_table,
                          b_start_loc, b_seq_len, b_ctx_len, max_input_len)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    attn_op = xops.fmha.cutlass.FwOp()

    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        subquery_lens, seq_lens)
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
    assert torch.allclose(output_ref, output, atol=1e-6, rtol=0)
