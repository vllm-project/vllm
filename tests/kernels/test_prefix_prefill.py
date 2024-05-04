import random
import time

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask

from vllm.attention.ops.prefix_prefill import context_attention_fwd

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96]
DTYPES = [torch.float16]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
SLIDING_WINDOW = [0, 16, 64, 128, 256, 512, 2048]


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    device: str,
) -> None:
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(BS):
        for j in range(query_lens[i]):
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
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    context_attention_fwd(query,
                          k,
                          v,
                          output,
                          k_cache,
                          v_cache,
                          block_table,
                          b_start_loc,
                          b_seq_len,
                          b_ctx_len,
                          max_input_len,
                          sliding_window=sliding_window)
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(query,
                          k,
                          v,
                          output,
                          k_cache,
                          v_cache,
                          block_table,
                          b_start_loc,
                          b_seq_len,
                          b_ctx_len,
                          max_input_len,
                          sliding_window=sliding_window)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    attn_op = xops.fmha.cutlass.FwOp()

    if num_kv_heads != num_heads:
        # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
        # project the key and value tensors to the desired number of
        # heads.
        #
        # see also: vllm/model_executor/layers/attention.py
        query = query.view(query.shape[0], num_kv_heads, num_queries_per_kv,
                           query.shape[-1])
        key = key[:, :, None, :].expand(key.shape[0], num_kv_heads,
                                        num_queries_per_kv, key.shape[-1])
        value = value[:, :,
                      None, :].expand(value.shape[0], num_kv_heads,
                                      num_queries_per_kv, value.shape[-1])
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)

    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        query_lens, seq_lens)
    if sliding_window > 0:
        attn_bias = attn_bias.make_local_attention_from_bottomright(
            sliding_window)
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    start_time = time.time()
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
    output_ref = output_ref.reshape(output.shape)
    assert torch.allclose(output_ref, output, atol=1e-6, rtol=0)
