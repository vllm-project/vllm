# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.prefix_prefill import context_attention_fwd

pytestmark = pytest.mark.skip_global_cleanup

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 64]
HEAD_SIZES = [24, 128]
DTYPES = [torch.float16]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]
SLIDING_WINDOW = [0, 16, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]

OPS = [chunked_prefill_paged_decode, context_attention_fwd]


def create_causal_attention_mask_for_sdpa(
    query_lens: list[int],
    seq_lens: list[int],
    sliding_window: int = 0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    total_queries = sum(query_lens)
    total_keys = sum(seq_lens)

    # Create a mask filled with -inf
    mask = torch.full(
        (total_queries, total_keys), float("-inf"), device=device, dtype=dtype
    )

    query_start = 0
    key_start = 0

    for query_len, seq_len in zip(query_lens, seq_lens):
        query_end = query_start + query_len
        key_end = key_start + seq_len
        q_indices = torch.arange(query_len, device=device)
        k_indices = torch.arange(seq_len, device=device)
        q_pos_in_seq = seq_len - query_len + q_indices

        valid_mask = k_indices[None, :] <= q_pos_in_seq[:, None]

        if sliding_window > 0:
            valid_mask &= k_indices[None, :] >= (
                q_pos_in_seq[:, None] - sliding_window + 1
            )

        mask[query_start:query_end, key_start:key_end][valid_mask] = 0.0

        query_start = query_end
        key_start = key_end

    return mask


def create_alibi_causal_mask(
    query_len: int,
    seq_len: int,
    alibi_slopes: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    query_pos = torch.arange(
        seq_len - query_len, seq_len, device=device, dtype=torch.float32
    )
    key_pos = torch.arange(seq_len, device=device, dtype=torch.float32)

    rel_pos = key_pos[None, :] - query_pos[:, None]

    # Apply ALiBi slopes: [num_heads, query_len, seq_len]
    alibi_bias = alibi_slopes[:, None, None] * rel_pos[None, :, :]
    alibi_bias = alibi_bias.to(dtype)

    # Apply causal mask: prevent attending to future positions
    # causal_mask[i, j] = True if key_pos[j] <= query_pos[i]
    causal_mask = key_pos[None, :] <= query_pos[:, None]
    alibi_bias = alibi_bias.masked_fill(~causal_mask[None, :, :], float("-inf"))

    # Add batch dimension: [1, num_heads, query_len, seq_len]
    # SDPA expects batch dimension even for single sequences
    return alibi_bias.unsqueeze(0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
    block_size: int = 32,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    if (
        current_platform.is_rocm()
        and op is chunked_prefill_paged_decode
        and kv_cache_dtype == "fp8_e5m2"
    ):
        pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")

    set_random_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.accelerator.set_device_index(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1

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

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    for i in range(BS):
        for j in range(query_lens[i]):
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
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=sliding_window,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")

    scale = float(1.0 / (head_size**0.5))

    # Reshape for SDPA: (seq_len, num_heads, head_size) ->
    # (1, num_heads, seq_len, head_size)
    query_sdpa = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, num_tokens, head_size
    )

    # Expand key and value for GQA/MQA to match query heads
    key_sdpa = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    value_sdpa = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        query_lens, seq_lens, sliding_window, device=device, dtype=dtype
    )

    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"PyTorch SDPA Time: {(end_time - start_time) * 1000:.2f} ms")

    # Reshape output back to (num_tokens, num_heads, head_size)
    output_ref = output_ref.view(num_heads, num_tokens, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend")
@pytest.mark.parametrize("sliding_window", [None, 1, 4])
@pytest.mark.parametrize("query_len", [1, 3])
@torch.inference_mode()
def test_rocm_backend_sliding_window_includes_boundary_token(
    sliding_window: int | None, query_len: int
) -> None:
    """A window of W must include the current token and W - 1 preceding tokens."""
    from vllm.v1.attention.backends.rocm_attn import RocmAttentionImpl
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    device, dtype = "cuda:0", torch.float16
    num_heads, head_size, seq_len = 4, 128, 8
    query = torch.zeros(query_len, num_heads, head_size, dtype=dtype, device=device)
    key = torch.zeros(seq_len, 1, head_size, dtype=dtype, device=device)
    values = torch.arange(1, seq_len + 1, dtype=dtype, device=device)
    value = values[:, None, None].expand_as(key)
    kv_cache = torch.zeros(2, 1, 16, head_size, dtype=dtype, device=device)
    _, value_cache = PagedAttention.split_kv_cache(kv_cache, 1, head_size)
    value_cache[..., :seq_len] = value.permute(1, 2, 0)
    scale = torch.tensor(1.0, device=device)
    metadata = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=query_len,
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        max_query_len=query_len,
        max_seq_len=seq_len,
        block_table=torch.zeros(1, 1, dtype=torch.int32, device=device),
        causal=True,
        mm_prefix_range_tensor=None,
    )
    impl = RocmAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
    )
    output = torch.empty_like(query)
    impl.forward(
        SimpleNamespace(_k_scale=scale, _v_scale=scale),
        query,
        key[-query_len:],
        value[-query_len:],
        kv_cache.transpose(0, 1),
        metadata,
        output,
    )
    mask = create_causal_attention_mask_for_sdpa(
        [query_len], [seq_len], sliding_window or 0, device=device, dtype=dtype
    )
    expected = (mask.float().softmax(dim=-1) @ values.float())[:, None, None]
    torch.testing.assert_close(output, expected.to(dtype).expand_as(output))


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    block_size: int = 32,
) -> None:
    # Exercises the KV_FROM_CACHE path of context_attention_fwd: the current
    # chunk K/V are not passed as dense tensors (k=v=None); they are read back
    # from the paged KV cache, as done by layers that re-attend an already
    # cached sequence with the query only (e.g. IQuest LoopCoder's
    # attn(q, None, None)). The whole sequence therefore lives in the cache and
    # the result must still match a dense causal SDPA reference.
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    # ensure one sequence in batch is a decode
    query_lens[-1] = 1
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

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    # Unlike the dense test, write the WHOLE sequence (context + current chunk)
    # into the paged cache, since the current chunk is read back from the cache.
    for i in range(BS):
        cur = 0
        block_id = 0
        while cur < seq_lens[i]:
            start_loc = b_seq_start_loc[i] + cur
            if cur + block_size > seq_lens[i]:
                end_loc = b_seq_start_loc[i] + seq_lens[i]
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
            cur += block_size
            block_id += 1
    # transpose to the paged cache layouts the kernel expects:
    #   K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    #   V_cache[num_blocks, num_kv_heads, head_size, block_size]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Cached-K/V path: current-chunk k and v are None.
    context_attention_fwd(
        query,
        None,
        None,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        sliding_window=0,
    )
    torch.accelerator.synchronize()

    scale = float(1.0 / (head_size**0.5))

    query_sdpa = query.view(num_tokens, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, num_tokens, head_size
    )
    key_sdpa = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )
    value_sdpa = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, sum(seq_lens), head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        query_lens, seq_lens, 0, device=device, dtype=dtype
    )
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    output_ref = output_ref.view(num_heads, num_tokens, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-4
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("use_mm_prefix", [False, True])
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv_block_table_boundary(
    device: str, use_mm_prefix: bool
) -> None:
    """Partial context and query tiles must not read past an exact block table."""
    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    dtype = torch.float16
    kv_cache_dtype = "auto"
    num_heads = 4
    num_queries_per_kv = 1
    num_kv_heads = num_heads // num_queries_per_kv
    head_size = 32
    block_size = 16 if use_mm_prefix else 32
    num_blocks = 1 if use_mm_prefix else 4
    seq_len = num_blocks * block_size
    query_len = 3 if use_mm_prefix else 99

    query = torch.empty(query_len, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(query_len, num_heads, head_size, dtype=dtype)

    kv = torch.empty(seq_len, 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
    # Exact-sized block table: row length == num_blocks (identity mapping).
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    b_seq_len = torch.tensor([seq_len], dtype=torch.int32)
    b_start_loc = torch.tensor([0, query_len], dtype=torch.int32)

    # Write the whole sequence (context + current chunk) into the paged cache.
    for cur in range(0, seq_len, block_size):
        block_id = cur // block_size
        end = min(cur + block_size, seq_len)
        start_slot = block_table[0, block_id] * block_size
        end_slot = start_slot + (end - cur)
        k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            key[cur:end]
        )
        v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
            value[cur:end]
        )

    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Cached-K/V path: current-chunk k and v are None.
    context_attention_fwd(
        query,
        None,
        None,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        seq_len,
        query_len,
        k_scale,
        v_scale,
        sliding_window=0,
        mm_prefix_range=(
            torch.tensor([[[0, seq_len - 1]]], dtype=torch.int32, device=device)
            if use_mm_prefix
            else None
        ),
    )
    torch.accelerator.synchronize()

    scale = float(1.0 / (head_size**0.5))
    query_sdpa = query.view(query_len, num_kv_heads, num_queries_per_kv, head_size)
    query_sdpa = query_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, query_len, head_size
    )
    key_sdpa = key[:, :, None, :].expand(
        seq_len, num_kv_heads, num_queries_per_kv, head_size
    )
    key_sdpa = key_sdpa.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size)
    value_sdpa = value[:, :, None, :].expand(
        seq_len, num_kv_heads, num_queries_per_kv, head_size
    )
    value_sdpa = value_sdpa.permute(1, 2, 0, 3).reshape(
        1, num_heads, seq_len, head_size
    )

    attn_mask = create_causal_attention_mask_for_sdpa(
        [query_len], [seq_len], 0, device=device, dtype=dtype
    )
    if use_mm_prefix:
        attn_mask.zero_()
    output_ref = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=scale,
    )
    output_ref = output_ref.view(num_heads, query_len, head_size)
    output_ref = output_ref.permute(1, 0, 2).contiguous()
    torch.testing.assert_close(output, output_ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_cached_kv_alibi_unsupported(device: str) -> None:
    # The cached-K/V (k=None) path is not supported together with ALiBi; the
    # entry point must reject it up-front with a clear NotImplementedError
    # rather than launching the kernel with an unsupported combination.
    set_random_seed(0)
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)

    num_heads = 4
    num_kv_heads = 4
    head_size = 16
    x = 8
    block_size = 16
    num_blocks = 4
    query_len = 8

    query = torch.empty(query_len, num_heads, head_size, dtype=torch.float16)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size // x, block_size, x, dtype=torch.float16
    )
    v_cache = torch.zeros(
        num_blocks, num_kv_heads, head_size, block_size, dtype=torch.float16
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks)
    b_seq_len = torch.tensor([query_len], dtype=torch.int32)
    b_start_loc = torch.tensor([0, query_len], dtype=torch.int32)
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    alibi_slopes = torch.ones(num_heads, dtype=torch.float32, device=device)

    with pytest.raises(NotImplementedError):
        context_attention_fwd(
            query,
            None,
            None,
            output,
            "auto",
            k_cache,
            v_cache,
            block_table,
            b_start_loc,
            b_seq_len,
            query_len,
            query_len,
            k_scale,
            v_scale,
            alibi_slopes=alibi_slopes,
        )


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
    block_size: int = 32,
) -> None:
    if "fp8" in kv_cache_dtype and not current_platform.has_device_capability(89):
        pytest.skip(
            "Triton limitation: fp8e4nv data type is not supported on CUDA arch < 89"
        )

    if (
        current_platform.is_rocm()
        and op is chunked_prefill_paged_decode
        and kv_cache_dtype == "fp8_e5m2"
    ):
        pytest.skip("ROCm custom paged attention does not support fp8_e5m2 KV cache")

    set_random_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.accelerator.set_device_index(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        # Fork from: vllm/vllm/model_executor/models/bloom.py#L44
        closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
        base = torch.tensor(
            2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(
                2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32,
            )
            num_remaining_heads = min(
                closest_power_of_2, total_num_heads - closest_power_of_2
            )
            extra_powers = torch.arange(
                start=1, end=1 + 2 * num_remaining_heads, step=2, dtype=torch.int32
            )
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes

    alibi_slopes = _get_alibi_slopes(num_heads).to(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
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
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    v_cache = torch.zeros(
        cache_size, block_size, num_kv_heads, head_size, dtype=cache_dtype
    )
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.int32)
    values = values[torch.randperm(cache_size)]
    block_table = values[: BS * max_block_per_request].view(BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.int32)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens), dim=0).to(torch.int32)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1]), dim=0).to(
        torch.int32
    )
    for i in range(BS):
        for j in range(query_lens[i]):
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
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                key[start_loc:end_loc]
            )
            v_cache.view(-1, num_kv_heads, head_size)[start_slot:end_slot].copy_(
                value[start_loc:end_loc]
            )
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = (
        k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = (
        v_cache.view(-1, block_size, num_kv_heads, head_size)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Warm up the Triton kernel by calling it once before actually measuring
    # generation time
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.accelerator.synchronize()
    start_time = time.time()
    op(
        query,
        k,
        v,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        MAX_CTX_LEN,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
    )
    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time) * 1000:.2f} ms")
    scale = float(1.0 / (head_size**0.5))

    # Prepare query, key, value for SDPA
    # Expand key and value for GQA/MQA to match query heads
    key_expanded = key[:, :, None, :].expand(
        key.shape[0], num_kv_heads, num_queries_per_kv, key.shape[-1]
    )
    value_expanded = value[:, :, None, :].expand(
        value.shape[0], num_kv_heads, num_queries_per_kv, value.shape[-1]
    )

    output_ref = torch.empty_like(output)

    torch.accelerator.synchronize()
    start_time = time.time()

    query_start = 0
    key_start = 0
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        query_end = query_start + query_len
        key_end = key_start + seq_len

        # Get query, key, value for this sequence
        q = query[query_start:query_end]  # [query_len, num_heads, head_size]
        k = key_expanded[
            key_start:key_end
        ]  # [seq_len, num_kv_heads, num_queries_per_kv, head_size]
        v = value_expanded[
            key_start:key_end
        ]  # [seq_len, num_kv_heads, num_queries_per_kv, head_size]

        # Reshape for SDPA: (batch=1, num_heads, seq_len, head_size)
        q_sdpa = q.view(query_len, num_kv_heads, num_queries_per_kv, head_size)
        q_sdpa = (
            q_sdpa.permute(1, 2, 0, 3)
            .reshape(1, num_heads, query_len, head_size)
            .contiguous()
        )

        k_sdpa = (
            k.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size).contiguous()
        )
        v_sdpa = (
            v.permute(1, 2, 0, 3).reshape(1, num_heads, seq_len, head_size).contiguous()
        )

        # Create ALiBi causal mask for this sequence using utility function
        alibi_mask = create_alibi_causal_mask(
            query_len, seq_len, alibi_slopes, device, dtype
        )

        # Compute attention
        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=alibi_mask,
            dropout_p=0.0,
            scale=scale,
        )

        # Reshape output back to [query_len, num_heads, head_size]
        out = out.view(num_heads, query_len, head_size).permute(1, 0, 2)
        output_ref[query_start:query_end].copy_(out)

        query_start = query_end
        key_start = key_end

    torch.accelerator.synchronize()
    end_time = time.time()
    print(f"PyTorch SDPA Time: {(end_time - start_time) * 1000:.2f} ms")
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-6
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


# These tests are optional to only run when explicitly invoked
#
# pytest -v -s --optional \
# tests/kernels/test_prefix_prefill.py::test_contexted_kv_attention_f32
#
# These tests are useful to test model dtype float32 on Turing devices.
# We skip them to not increase the time when running tests on CI
@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention(
        num_heads,
        num_queries_per_kv,
        head_size,
        sliding_window,
        dtype,
        kv_cache_dtype,
        device,
        op,
    )


@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_contexted_kv_attention_alibi_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
    op: Callable,
) -> None:
    test_contexted_kv_attention_alibi(
        num_heads, num_queries_per_kv, head_size, dtype, kv_cache_dtype, device, op
    )


# Hybrid mamba + full-attention models get a non-power-of-2 attention page.
NONSTANDARD_BLOCK_SIZE_SHAPES = [
    (64, 1, 128, 544),
    (8, 4, 256, 1040),
    (8, 4, 256, 1056),
]


@pytest.mark.parametrize(
    "num_heads,num_queries_per_kv,head_size,block_size", NONSTANDARD_BLOCK_SIZE_SHAPES
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("op", OPS)
@torch.inference_mode()
def test_qwen3_nonstandard_block_size(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    device: str,
    op: Callable,
) -> None:
    """Non-power-of-2 pages must match, even when a tile straddles a page."""
    if not current_platform.is_rocm():
        pytest.skip("Non-power-of-2 block sizes are only exercised on ROCm CI.")

    test_contexted_kv_attention(
        num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        block_size=block_size,
        sliding_window=0,
        dtype=dtype,
        kv_cache_dtype="auto",
        device=device,
        op=op,
    )


@pytest.mark.parametrize(
    "dtype,sliding_window,clamp,cached_kv,block_size,head_size,kv_cache_dtype",
    [
        pytest.param(torch.float16, None, False, False, 16, 128, "auto", id="dense"),
        pytest.param(torch.bfloat16, None, False, True, 16, 256, "auto", id="cached"),
        pytest.param(
            torch.bfloat16, 16, False, False, 544, 256, "auto", id="window-dense"
        ),
        pytest.param(
            torch.float16, 16, False, True, 544, 128, "auto", id="window-cached"
        ),
        pytest.param(
            torch.bfloat16, 16, True, False, 16, 256, "fp8", id="clamped-fp8-dense"
        ),
        pytest.param(
            torch.bfloat16, 16, True, True, 544, 256, "fp8", id="clamped-fp8-cached"
        ),
    ],
)
@torch.inference_mode()
def test_prefix_lm_matches_dense_reference(
    dtype: torch.dtype,
    sliding_window: int | None,
    clamp: bool,
    cached_kv: bool,
    block_size: int,
    head_size: int,
    kv_cache_dtype: str,
) -> None:
    """Image spans cross query tiles and cached context without changing text masks."""
    from vllm.v1.attention.backends.utils import compute_mm_prefix_range_tensor
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    if (
        kv_cache_dtype == "fp8"
        and not current_platform.is_rocm()
        and not current_platform.has_device_capability(89)
    ):
        pytest.skip("FP8 requires CUDA compute capability >= 8.9")

    device = "cuda:0"
    set_random_seed(0)
    query_lens, context_lens = [193, 37, 1], [0, 89, 80]
    seq_lens = [q + c for q, c in zip(query_lens, context_lens)]
    num_heads, num_kv_heads = 4, 2
    ranges = {0: [(5, 170), (180, 189)], 1: [(70, 110)]}
    blocks_per_req = math.ceil(max(seq_lens) / block_size)
    block_table = (
        torch.randperm(3 * blocks_per_req, device=device).reshape(3, -1).to(torch.int32)
    )
    cache_dtype = current_platform.fp8_dtype() if kv_cache_dtype == "fp8" else dtype
    k_scale, v_scale = 0.5, 1.75
    cache = torch.empty(
        2,
        3 * blocks_per_req,
        block_size,
        num_kv_heads * head_size,
        device=device,
        dtype=cache_dtype,
    )
    k_cache, v_cache = PagedAttention.split_kv_cache(cache, num_kv_heads, head_size)
    # Unwritten page tails must never contribute to attention.
    k_cache.fill_(float("nan"))
    v_cache.fill_(float("nan"))
    query = torch.randn(
        sum(query_lens), num_heads, head_size, device=device, dtype=dtype
    )
    dense_keys, dense_values, references = [], [], []
    offset = 0
    for req_idx, (q_len, ctx_len, seq_len) in enumerate(
        zip(query_lens, context_lens, seq_lens)
    ):
        key = torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        value = torch.randn_like(key)
        if kv_cache_dtype == "fp8":
            cached_key = (key / k_scale).to(cache_dtype)
            cached_value = (value / v_scale).to(cache_dtype)
        else:
            cached_key, cached_value = key, value
        positions = torch.arange(seq_len, device=device)
        pages = block_table[req_idx, positions // block_size]
        key_storage = cached_key.reshape(
            seq_len, num_kv_heads, head_size // k_cache.shape[-1], k_cache.shape[-1]
        )
        # Indexed writes to FP8 storage need byte views on ROCm.
        if kv_cache_dtype == "fp8":
            k_cache.view(torch.uint8)[pages, :, :, positions % block_size, :] = (
                key_storage.view(torch.uint8)
            )
            v_cache.view(torch.uint8)[pages, :, :, positions % block_size] = (
                cached_value.view(torch.uint8)
            )
        else:
            k_cache[pages, :, :, positions % block_size, :] = key_storage
            v_cache[pages, :, :, positions % block_size] = cached_value
        dense_keys.append(key[ctx_len:])
        dense_values.append(value[ctx_len:])
        key, value = key.clone(), value.clone()
        cache_end = seq_len if cached_kv else ctx_len
        if kv_cache_dtype == "fp8":
            key[:cache_end] = (cached_key[:cache_end].float() * k_scale).to(dtype)
            value[:cache_end] = (cached_value[:cache_end].float() * v_scale).to(dtype)
        q_pos = torch.arange(ctx_len, seq_len, device=device)
        k_pos = torch.arange(seq_len, device=device)
        keep = k_pos[None, :] <= q_pos[:, None]
        window = q_pos[:, None] - k_pos[None, :] < (sliding_window or seq_len + 1)
        keep &= window
        for start, end in ranges.get(req_idx, []):
            image = (
                (q_pos[:, None] >= start)
                & (q_pos[:, None] <= end)
                & (k_pos[None, :] >= start)
                & (k_pos[None, :] <= end)
            )
            keep |= (image & window) if clamp else image
        repeated_key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
        repeated_value = value.repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = (
            torch.einsum(
                "qhd,khd->hqk",
                query[offset : offset + q_len].float(),
                repeated_key.float(),
            )
            * head_size**-0.5
        )
        references.append(
            torch.einsum(
                "hqk,khd->qhd",
                scores.masked_fill(~keep, -torch.inf).softmax(-1),
                repeated_value.float(),
            )
        )
        offset += q_len
    k_scale_tensor = torch.tensor(k_scale, device=device)
    v_scale_tensor = torch.tensor(v_scale, device=device)
    output = torch.empty_like(query)
    query_start = torch.tensor([0, 193, 230, 231], device=device, dtype=torch.int32)
    seq_lens_tensor = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    mm_ranges = compute_mm_prefix_range_tensor(ranges, 3, torch.device(device))
    key = None if cached_kv else torch.cat(dense_keys)
    value = None if cached_kv else torch.cat(dense_values)

    chunked_prefill_paged_decode(
        query,
        key,
        value,
        output,
        kv_cache_dtype,
        k_cache,
        v_cache,
        block_table,
        query_start,
        seq_lens_tensor,
        max(seq_lens),
        max(query_lens),
        k_scale_tensor,
        v_scale_tensor,
        sliding_window=sliding_window,
        mm_prefix_range=mm_ranges,
        mm_prefix_clamp_sliding_window=clamp,
    )
    torch.testing.assert_close(
        output.float(), torch.cat(references), atol=2e-2, rtol=2e-2
    )


@pytest.mark.parametrize("clamp", [False, True])
@torch.inference_mode()
def test_prefix_lm_ignores_evicted_window_pages(clamp: bool) -> None:
    """Zero-probability keys in released pages must not poison the output."""
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    device, dtype = "cuda:0", torch.bfloat16
    seq_len, query_len, head_size, block_size = 129, 33, 128, 16
    cache = torch.zeros(2, 10, block_size, head_size, device=device, dtype=dtype)
    k_cache, v_cache = PagedAttention.split_kv_cache(cache, 1, head_size)
    k_cache[0].fill_(float("nan"))
    v_cache.fill_(1)
    v_cache[0].fill_(float("nan"))
    table = torch.arange(1, 10, dtype=torch.int32, device=device)[None, :]
    table[:, :4] = 0
    query = torch.zeros(query_len, 1, head_size, device=device, dtype=dtype)
    key = torch.zeros_like(query)
    value = torch.ones_like(query)
    output = torch.empty_like(query)
    scale = torch.tensor(1.0, device=device)
    context_attention_fwd(
        query,
        key,
        value,
        output,
        "auto",
        k_cache,
        v_cache,
        table,
        torch.tensor([0, query_len], dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        seq_len,
        query_len,
        scale,
        scale,
        sliding_window=32,
        mm_prefix_range=torch.tensor([[[88, 110]]], dtype=torch.int32, device=device),
        mm_prefix_clamp_sliding_window=clamp,
    )
    torch.testing.assert_close(output, value)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm backend")
@pytest.mark.parametrize(
    "sliding_window,clamp,query_len,num_heads",
    [
        pytest.param(None, False, 1, 4, id="single-token-hip"),
        pytest.param(32, False, 1, 1, id="single-token-triton"),
        pytest.param(None, False, 4, 1, id="multi-token-global"),
        pytest.param(32, False, 4, 1, id="multi-token-window"),
        pytest.param(32, True, 4, 1, id="multi-token-clamped"),
    ],
)
@torch.inference_mode()
def test_rocm_prefix_prompt_extension_replays_decode_graph(
    sliding_window: int | None,
    clamp: bool,
    query_len: int,
    num_heads: int,
) -> None:
    """Image prompt extensions must replay the causal graph with updated inputs."""
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.rocm_attn import (
        RocmAttentionImpl,
        RocmAttentionMetadataBuilder,
    )
    from vllm.v1.attention.ops.paged_attn import PagedAttention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    device, dtype = "cuda:0", torch.bfloat16
    head_size, block_size = 128, 16
    torch.accelerator.set_device_index(device)
    layer = SimpleNamespace(
        sliding_window=sliding_window,
        use_mm_prefix=True,
        mm_prefix_clamp_sliding_window=clamp,
        _k_scale=torch.tensor(1.0, device=device),
        _v_scale=torch.tensor(1.0, device=device),
    )
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            is_mm_prefix_lm=True,
            get_sliding_window=lambda: sliding_window,
            get_num_attention_heads=lambda _: num_heads,
            get_num_kv_heads=lambda _: 1,
            get_head_size=lambda: head_size,
        ),
        parallel_config=None,
        compilation_config=SimpleNamespace(static_forward_context={"attn": layer}),
    )
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=head_size, dtype=dtype
    )
    builder = RocmAttentionMetadataBuilder(spec, ["attn"], config, device)
    cache = torch.zeros(2, 5, block_size, head_size, device=device, dtype=dtype)
    _, v_cache = PagedAttention.split_kv_cache(cache, 1, head_size)
    values = torch.arange(80, device=device, dtype=dtype)
    v_cache.copy_(values.reshape(5, 1, 1, block_size).expand_as(v_cache))
    query = torch.zeros(query_len, num_heads, head_size, device=device, dtype=dtype)
    output = torch.empty_like(query)
    common = CommonAttentionMetadata(
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=80,
        query_start_loc=torch.tensor(
            [0, query_len, query_len], dtype=torch.int32, device=device
        ),
        query_start_loc_cpu=torch.tensor(
            [0, query_len, query_len], dtype=torch.int32, device="cpu"
        ),
        seq_lens_cpu_upper_bound=torch.tensor([1, 0], dtype=torch.int32, device="cpu"),
        seq_lens=torch.tensor([1, 0], dtype=torch.int32, device=device),
        block_table_tensor=torch.arange(5, dtype=torch.int32, device=device).repeat(
            2, 1
        ),
        slot_mapping=torch.zeros(query_len, dtype=torch.int64, device=device),
        causal=True,
        num_reqs=2,
        is_prefilling=torch.tensor([False, False], device="cpu"),
        mm_req_doc_ranges=None,
    )
    metadata = builder.build_for_cudagraph_capture(common)
    impl = RocmAttentionImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
    )

    def run() -> None:
        impl.forward(layer, query, None, None, cache.transpose(0, 1), metadata, output)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    assert common.seq_lens_cpu_upper_bound is not None
    assert common.is_prefilling is not None
    for seq_len, active_query_len in ((65, query_len), (73, 1)):
        starts = torch.tensor(
            [0, active_query_len, active_query_len], dtype=torch.int32, device="cpu"
        )
        common.query_start_loc.copy_(starts)
        common.query_start_loc_cpu.copy_(starts)
        common.seq_lens[0] = seq_len
        common.seq_lens_cpu_upper_bound[0] = seq_len
        common.is_prefilling[0] = True
        first_query = seq_len - active_query_len
        # The real prompt token is the image's last token; later queries are drafts.
        start = (
            first_query - sliding_window + 1
            if sliding_window is not None and not clamp
            else 0
        )
        common.mm_req_doc_ranges = {0: [(start, first_query)]}
        assert builder.build(0, common).mm_prefix_range_tensor is None
        output.fill_(float("nan"))
        graph.replay()
        expected = (
            torch.stack(
                [
                    values[max(0, pos + 1 - (sliding_window or seq_len)) : pos + 1]
                    .float()
                    .mean()
                    for pos in range(first_query, seq_len)
                ]
            )
            .to(dtype)[:, None, None]
            .expand(active_query_len, num_heads, head_size)
        )
        torch.testing.assert_close(output[:active_query_len], expected)
        torch.testing.assert_close(common.query_start_loc.cpu(), starts)
        assert common.seq_lens[0].item() == seq_len

    if query_len > 1:
        # Rejection may leave different device query lengths within one graph.
        common.query_start_loc.copy_(torch.tensor([0, 1, query_len], device=device))
        common.seq_lens.copy_(torch.tensor([73, 61], device=device))
        output.fill_(float("nan"))
        graph.replay()
        expected = (
            torch.stack(
                [
                    values[max(0, pos + 1 - (sliding_window or 80)) : pos + 1]
                    .float()
                    .mean()
                    for pos in [72, *range(61 - (query_len - 1), 61)]
                ]
            )
            .to(dtype)[:, None, None]
            .expand_as(output)
        )
        torch.testing.assert_close(output, expected)
