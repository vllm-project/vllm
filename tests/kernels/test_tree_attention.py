#!/data2/kavioyu/micromamba/envs/cuda122/bin/ python3
import random
import time
from typing import List, Optional, Tuple

import pytest
import torch
from allclose_default import get_default_atol, get_default_rtol

from vllm.utils import get_max_shared_memory_bytes, is_hip
from vllm.attention.ops.tree_attn import tree_attention_fwd
from vllm.utils import create_kv_caches_with_random
from vllm import _custom_ops as ops

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 4321  # Arbitrary values for testing
# only test on half and bfloat16
DTYPES = [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
# HEAD_SIZES = [64, 80, 96, 112, 128, 256
#               ] if not is_hip() else [64, 80, 96, 112, 128]
HEAD_SIZES = [64]

BLOCK_SIZES = [16]
USE_ALIBI = [False]
KV_CACHE_DTYPE = ["auto"]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
TREEWIDTH = [8]

DTYPES = [torch.bfloat16]
NUM_HEADS = [(1, 1)]
BLOCK_SIZES = [16]
MAX_SEQ_LEN = 32

def create_tree_attention_mask(context_len, prompt_len, tree_width, num_kv_head, dtype):
    prompt_mask = torch.zeros((num_kv_head, tree_width, prompt_len), dtype=dtype)
    none_mask_value = torch.arange(context_len-prompt_len).repeat(tree_width, 1) - torch.arange(tree_width)[:, None]
    none_mask_value = none_mask_value % tree_width
    none_mask_value = none_mask_value == 0

    min_value = torch.finfo(dtype).min

    generate_mask = torch.full(none_mask_value.shape, min_value, dtype=dtype)
    generate_mask[none_mask_value] = 0
    generate_mask = generate_mask.unsqueeze(0).repeat(num_kv_head, 1, 1)
    return torch.concat([prompt_mask, generate_mask], dim=2)

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()

    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    prompt_lens: torch.Tensor,
    tree_width: int
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0] // tree_width
    query = query.reshape(num_seqs, tree_width, num_query_heads, head_size)
    output = output.reshape(query.shape)

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i]
        block_table = block_tables[i]
        context_len = int(context_lens[i])
        prompt_len = int(prompt_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        mask = create_tree_attention_mask(context_len, prompt_len, tree_width, num_query_heads, dtype=torch.float)
        
        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len).int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)
            mask += alibi_bias
        out = ref_masked_attention(q, keys, values, scale, mask)
        out = out.view(tree_width, num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    output.reshape(-1, num_kv_heads, head_size)


# @pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
# @pytest.mark.parametrize("num_heads", NUM_HEADS)
# @pytest.mark.parametrize("head_size", HEAD_SIZES)
# @pytest.mark.parametrize("use_alibi", USE_ALIBI)
# @pytest.mark.parametrize("block_size", BLOCK_SIZES)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# @pytest.mark.parametrize("tree_width", TREEWIDTH)
def test_paged_attention(
    kv_cache_factory,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    tree_width: int,
    device: str,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs*tree_width, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    prompt_lens = [random.randint(1, ctx_len) for ctx_len in context_lens]
    context_lens = torch.tensor(context_lens, dtype=torch.int)
    # prompt_lens = torch.tensor(prompt_lens, dtype=torch.int)
    prompt_lens = context_lens - tree_width

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                device)
    key_cache, value_cache = key_caches[0], value_caches[0]


    output = torch.empty_like(query)
    torch.cuda.synchronize()
    start_time = time.time()
    tree_attention_fwd(
        query,
        output,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        prompt_lens,
        tree_width,
        alibi_slopes
    )
    torch.cuda.synchronize()
    #print("tree attention duration:", time.time()-start_time)


    ref_output = torch.empty_like(query)
    ref_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
        prompt_lens,
        tree_width
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = 1e-4
    rtol = 2e-2
    
    def diff(a, b):
        print(((a-b).abs()/(b+1e-8)).mean())

    diff(output, ref_output)
    assert torch.allclose(output, ref_output, atol=atol, rtol=rtol)


test_paged_attention(create_kv_caches_with_random, 1, (1, 1), 64, False, 16, torch.bfloat16, 'auto', 0, 2, 'cuda')
#test_paged_attention(create_kv_caches_with_random, 1, (4, 4), 32, False, 16, torch.half, 'auto', 1, 2, 'cuda')