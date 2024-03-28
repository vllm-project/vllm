from xformers import ops as xops
import torch
import random
from vllm_flash_attn import flash_attn_varlen_func
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask
import pytest
import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

NUM_HEADS = [8]
NUM_QUERIES_PER_KV = [1]
HEAD_SIZES = [128]
DTYPES = [torch.float16]

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )
        
def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_flashatten_varlen(
    num_heads: int, num_queries_per_kv: int, head_size: int, dtype: torch.dtype
):

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device("cuda")
    batch_size = 10
    cache_size = 640
    block_size = 16

    prefix_lens = [random.randint(16, 128) for _ in range(batch_size)]
    append_lens = [random.randint(16, 128) for _ in range(batch_size)]
    seq_lens = [a + b for a, b in zip(prefix_lens, append_lens)]

    num_tokens = sum(append_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-1, 1e-1)

    num_kv_heads = num_heads // num_queries_per_kv
    key_value = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    key_value.uniform_(-1e-1, 1e-1)
    key, value = key_value.unbind(dim=1)

    # append_key = torch.zeros(sum(append_lens), num_kv_heads, head_size, dtype=dtype)
    # append_value = torch.zeros(sum(append_lens), num_kv_heads, head_size, dtype=dtype)

    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    max_block_per_request = int(cache_size / batch_size)
    block_table = values[: batch_size * max_block_per_request].view(
        batch_size, max_block_per_request
    )

    k_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=dtype)
    v_cache = torch.zeros(cache_size, block_size, num_kv_heads, head_size, dtype=dtype)

    qo_indptr = torch.cumsum(torch.tensor([0] + append_lens), dim=0, dtype=torch.int32)
    seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_lens), dim=0, dtype=torch.int32
    )
    cu_prefix_lens = torch.cumsum(torch.tensor([0] + prefix_lens), dim=0, dtype=torch.int32)

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
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_prefix_id += block_size
            block_id += 1


    scale = float(1.0 / (head_size**0.5))
    output = flash_attn_varlen_func(q=query, 
                           k=k_cache, 
                           v=v_cache,
                          cu_seqlens_q=qo_indptr,
                          cu_seqlens_k=seq_start_loc,
                          max_seqlen_q=max(append_lens),
                          max_seqlen_k=max(seq_lens),
                          softmax_scale=scale,
                          block_table=block_table)

    query = query.unsqueeze(0)
    key = key.unsqueeze(0)
    value = value.unsqueeze(0)
    attn_bias = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
        append_lens, seq_lens
    )
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

    output_ref, _ = attention_ref(query,key,value, causal=True)

    
    assert torch.allclose(output_ref, output, atol=1e-4, rtol=1e-5)
