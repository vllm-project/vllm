import random

import pytest
import torch
from vllm_flash_attn import flash_attn_varlen_func
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask

NUM_HEADS = [8]
NUM_QUERIES_PER_KV = [1]
HEAD_SIZES = [128]
DTYPES = [torch.float16]


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flashinfer_append(num_heads: int, num_queries_per_kv: int,
                           head_size: int, dtype: torch.dtype):
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device("cuda")
    batch_size = 10
    cache_size = 640
    block_size = 16
    prefix_lens = [random.randint(100, 200) for _ in range(batch_size)]
    append_lens = [random.randint(1, 5) for _ in range(batch_size)]
    seq_lens = [a + b for a, b in zip(prefix_lens, append_lens)]
    num_tokens = sum(append_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-1, 1e-1)
    num_kv_heads = num_heads // num_queries_per_kv
    key_value = torch.empty(sum(seq_lens),
                            2,
                            num_kv_heads,
                            head_size,
                            dtype=dtype)
    key_value.uniform_(-1e-1, 1e-1)
    key, value = key_value.unbind(dim=1)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    max_block_per_request = int(cache_size / batch_size)
    block_table = values[:batch_size * max_block_per_request].view(
        batch_size, max_block_per_request)
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
    qo_indptr = torch.cumsum(torch.tensor([0] + append_lens),
                             dim=0,
                             dtype=torch.int32)
    seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens),
                                 dim=0,
                                 dtype=torch.int32)
    paged_kv_last_page_len = []
    paged_kv_indptr = [0]
    page_kv_indices = []
    total_block_num = 0
    for i in range(batch_size):
        # copy key, value to kv cache
        cur_prefix_id = 0
        block_id = 0
        while cur_prefix_id < seq_lens[i]:
            start_loc = seq_start_loc[i] + cur_prefix_id
            if cur_prefix_id + block_size > seq_lens[i]:
                end_loc = seq_start_loc[i] + seq_lens[i]
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
            cur_prefix_id += block_size
            block_id += 1
        paged_kv_last_page_len.append((seq_lens[i] - 1) % block_size + 1)
        cur_block_num = (seq_lens[i] - 1) // block_size + 1
        page_kv_indices.extend(block_table[i, :cur_block_num])
        total_block_num += cur_block_num
        paged_kv_indptr.append(total_block_num)
    output = flash_attn_varlen_func(
        query,
        k_cache,
        v_cache,
        cu_seqlens_q=qo_indptr,
        cu_seqlens_k=seq_start_loc,
        max_seqlen_q=max(append_lens),
        max_seqlen_k=max(seq_lens),
        causal=True,
        block_table=block_table,
    )
    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        append_lens, seq_lens)
    scale = float(1.0 / (head_size**0.5))
    attn_op = xops.fmha.cutlass.FwOp()
    output_ref = xops.memory_efficient_attention_forward(
        query,
        key,
        value,
        attn_bias=attn_bias,
        p=0.0,
        scale=scale,
        op=attn_op,
    ).squeeze(0)
    print((output - output_ref).abs().max())
    assert torch.allclose(output_ref, output, atol=1e-4, rtol=1e-2)
