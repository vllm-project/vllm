import random
from typing import List, Optional, Tuple

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.attention.ops.blocksparse_attention.interface import (
    LocalStridedBlockSparseAttn)
from vllm.platforms import current_platform
from vllm.utils import get_max_shared_memory_bytes

from .allclose_default import get_default_atol, get_default_rtol

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = get_max_shared_memory_bytes() // FLOAT32_BYTES - 512
# MAX_SEQ_LEN = 2771

# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 4321  # Arbitrary values for testing
PARTITION_SIZE = 512
DTYPES = [torch.half, torch.bfloat16]
NUM_GEN_SEQS = [3]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40)]  # Arbitrary values for testing

HEAD_SIZES = [64, 112]
BLOCK_SIZES = [16]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = ['cuda:0']
BLOCKSPARSE_LOCAL_BLOCKS = [16]
BLOCKSPARSE_VERT_STRIDES = [8]

BLOCKSPARSE_BLOCK_SIZES = [64]
BLOCKSPARSE_HEADS_SLIDINGS = [2, -1]
BLOCKSPARSE_HOMO_HEADS = [True, False]


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


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 1,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        if blocksparse_vert_stride >= 1:
            bsize = blocksparse_block_size
            hsliding = blocksparse_head_sliding_step
            vert = blocksparse_vert_stride
            locals = blocksparse_local_blocks
            qb = (seq_len - 1) // bsize
            attn_mask = q.new_zeros(
                (num_query_heads, 1, seq_len)).float() - torch.inf
            for h in range(num_query_heads):
                if hsliding >= 0:  # slide with q heads
                    bs_offset = (tp_rank * num_query_heads + h) * hsliding + 1
                else:  # slide with kv heads
                    bs_offset = (tp_rank * num_kv_heads +
                                 h // num_queries_per_kv) * (-hsliding) + 1
                for kb in range(qb + 1):
                    kj = kb * bsize
                    if (qb - kb) < locals or \
                        (kb + bs_offset) % vert == 0:
                        attn_mask[h, 0, kj:min(kj + bsize, seq_len)] = 0
            if alibi_bias is not None:
                attn_mask += alibi_bias
        else:
            attn_mask = alibi_bias

        out = ref_masked_attention(q, keys, values, scale, attn_mask=attn_mask)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("blocksparse_local_blocks", BLOCKSPARSE_LOCAL_BLOCKS)
@pytest.mark.parametrize("blocksparse_vert_stride", BLOCKSPARSE_VERT_STRIDES)
@pytest.mark.parametrize("blocksparse_block_size", BLOCKSPARSE_BLOCK_SIZES)
@pytest.mark.parametrize("blocksparse_head_sliding_step",
                         BLOCKSPARSE_HEADS_SLIDINGS)
def test_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
    blocksparse_local_blocks: int,
    blocksparse_vert_stride: int,
    blocksparse_block_size: int,
    blocksparse_head_sliding_step: int,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.rand(num_query_heads, dtype=torch.float)

    seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
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

    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    tp_rank = 0

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v1":
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank=tp_rank,
            blocksparse_local_blocks=blocksparse_local_blocks,
            blocksparse_vert_stride=blocksparse_vert_stride,
            blocksparse_block_size=blocksparse_block_size,
            blocksparse_head_sliding_step=blocksparse_head_sliding_step,
        )
    elif version == "v2":
        num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
            kv_cache_dtype,
            k_scale,
            v_scale,
            tp_rank=tp_rank,
            blocksparse_local_blocks=blocksparse_local_blocks,
            blocksparse_vert_stride=blocksparse_vert_stride,
            blocksparse_block_size=blocksparse_block_size,
            blocksparse_head_sliding_step=blocksparse_head_sliding_step,
        )
    else:
        raise AssertionError(f"Unknown version: {version}")

    # Run the reference implementation.
    if kv_cache_dtype == "fp8":
        # Convert cache data back to dtype.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x,
                           block_size, x)
        dequantized_key_cache = torch.empty(size=key_cache_shape,
                                            dtype=dtype,
                                            device=device)
        ops.convert_fp8(dequantized_key_cache, key_cache)
        key_cache = dequantized_key_cache

        value_cache_shape = value_cache.shape
        dequantized_value_cache = torch.empty(size=value_cache_shape,
                                              dtype=dtype,
                                              device=device)
        ops.convert_fp8(dequantized_value_cache, value_cache)
        value_cache = dequantized_value_cache

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        alibi_slopes,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    atol = get_default_atol(output) if current_platform.is_rocm() else 1e-3
    rtol = get_default_rtol(output) if current_platform.is_rocm() else 1e-5

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-3, 1e-5
    if kv_cache_dtype == "fp8":
        atol, rtol = 1e-2, 1e-5
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


def ref_multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype)

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


@pytest.mark.parametrize("num_seqs", NUM_PREFILL_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("blocksparse_local_blocks", BLOCKSPARSE_LOCAL_BLOCKS)
@pytest.mark.parametrize("blocksparse_vert_stride", BLOCKSPARSE_VERT_STRIDES)
@pytest.mark.parametrize("blocksparse_block_size", BLOCKSPARSE_BLOCK_SIZES)
@pytest.mark.parametrize("blocksparse_homo_heads", BLOCKSPARSE_HOMO_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_varlen_blocksparse_attention_prefill(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    blocksparse_local_blocks: int,
    blocksparse_vert_stride: int,
    blocksparse_block_size: int,
    blocksparse_homo_heads: bool,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    current_platform.seed_everything(seed)
    torch.set_default_device(device)
    # MAX_SEQ_LEN sometimes causes OOM in the reference implementation.
    # As the xformers library is already tested with its own tests, we can use
    # a smaller MAX_SEQ_LEN here.
    max_len = min(MAX_SEQ_LEN, 4096)
    seq_lens = random.sample(range(1, max_len), num_seqs)
    cu_seq_lens = torch.cumsum(torch.tensor([0] + seq_lens), dim=0)
    num_tokens = sum(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads

    qkv = torch.empty(num_tokens,
                      num_query_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype)
    qkv.uniform_(-scale, scale)
    query, key, value = qkv.split(
        [num_query_heads, num_kv_heads, num_kv_heads], dim=1)

    bs_attn_op = LocalStridedBlockSparseAttn(
        num_query_heads,
        max_len,
        local_blocks=blocksparse_local_blocks,
        vert_stride=blocksparse_vert_stride,
        block_size=blocksparse_block_size,
        device=device,
        dtype=dtype,
        homo_head=blocksparse_homo_heads)

    output = bs_attn_op(query,
                        key,
                        value,
                        cu_seq_lens.to(device),
                        sm_scale=scale)

    if num_queries_per_kv > 1:
        # Handle MQA and GQA
        key = torch.repeat_interleave(key, num_queries_per_kv, dim=1)
        value = torch.repeat_interleave(value, num_queries_per_kv, dim=1)

    ref_output = ref_multi_query_kv_attention(
        cu_seq_lens.tolist(),
        query,
        key,
        value,
        scale,
        dtype,
    )
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
