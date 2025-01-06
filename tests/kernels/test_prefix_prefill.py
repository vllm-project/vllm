import math
import random
import time

import pytest
import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalFromBottomRightMask

from vllm.attention.backends.xformers import _make_alibi_bias
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

NUM_HEADS = [64]
NUM_QUERIES_PER_KV = [1, 8, 64]
HEAD_SIZES = [128, 96, 24]
DTYPES = [torch.float16]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]
SLIDING_WINDOW = [0, 16, 64, 128, 256, 512, 2048]
KV_CACHE_DTYPES = ["auto", "fp8", "fp8_e5m2"]


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("sliding_window", SLIDING_WINDOW)
@torch.inference_mode()
def test_contexted_kv_attention(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:

    if 'fp8' in kv_cache_dtype and not current_platform.has_device_capability(
            89):
        pytest.skip(
            'Triton limitation: fp8e4nv data type is not supported on CUDA'
            ' arch < 89')

    current_platform.seed_everything(0)
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

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
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
                          kv_cache_dtype,
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
                          kv_cache_dtype,
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
    atol = 1e-3 if "fp8" in kv_cache_dtype else 1e-6
    torch.testing.assert_close(output, output_ref, atol=atol, rtol=0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_alibi(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:

    if 'fp8' in kv_cache_dtype and not current_platform.has_device_capability(
            89):
        pytest.skip(
            'Triton limitation: fp8e4nv data type is not supported on CUDA'
            ' arch < 89')

    current_platform.seed_everything(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
        # Fork from: vllm/vllm/model_executor/models/bloom.py#L44
        closest_power_of_2 = 2**math.floor(math.log2(total_num_heads))
        base = torch.tensor(
            2**(-(2**-(math.log2(closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != total_num_heads:
            extra_base = torch.tensor(
                2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                dtype=torch.float32,
            )
            num_remaining_heads = min(closest_power_of_2,
                                      total_num_heads - closest_power_of_2)
            extra_powers = torch.arange(start=1,
                                        end=1 + 2 * num_remaining_heads,
                                        step=2,
                                        dtype=torch.int32)
            slopes = torch.cat(
                [slopes, torch.pow(extra_base, extra_powers)], dim=0)
        return slopes

    alibi_slopes = _get_alibi_slopes(num_heads).to(device)

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
    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
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
                          kv_cache_dtype,
                          k_cache,
                          v_cache,
                          block_table,
                          b_start_loc,
                          b_seq_len,
                          b_ctx_len,
                          max_input_len,
                          alibi_slopes=alibi_slopes)
    torch.cuda.synchronize()
    start_time = time.time()
    context_attention_fwd(query,
                          k,
                          v,
                          output,
                          kv_cache_dtype,
                          k_cache,
                          v_cache,
                          block_table,
                          b_start_loc,
                          b_seq_len,
                          b_ctx_len,
                          max_input_len,
                          alibi_slopes=alibi_slopes)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"triton Time: {(end_time - start_time)*1000:.2f} ms")
    scale = float(1.0 / (head_size**0.5))

    # NOTE(DefTruth): In order to reuse _make_alibi_bias function,
    # we have to pad query tensor before MQA/GQA expanding.
    if query.shape[0] != key.shape[0]:
        query_pad = torch.empty(sum(seq_lens),
                                num_heads,
                                head_size,
                                dtype=dtype)
        query_pad.uniform_(-1e-3, 1e-3)
        seq_start = 0
        query_start = 0
        for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
            seq_end = seq_start + seq_len
            query_end = query_start + query_len
            query_pad[seq_start:seq_end, ...] = torch.cat([
                torch.zeros(
                    seq_len - query_len, num_heads, head_size, dtype=dtype),
                query[query_start:query_end, ...]
            ],
                                                          dim=0)
            seq_start += seq_len
            query_start += query_len
        query = query_pad

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

    attn_bias = _make_alibi_bias(alibi_slopes, num_kv_heads, dtype, seq_lens)
    output_ref = torch.empty_like(output)
    seq_start = 0
    query_start = 0
    start_time = time.time()
    # Attention with alibi slopes.
    # FIXME(DefTruth): Because xformers does not support dynamic sequence
    # lengths with custom attention bias, we process each prompt one by
    # one. This is inefficient, especially when we have many short prompts.
    # modified from: vllm/attention/backends/xformers.py#L343
    for i, (query_len, seq_len) in enumerate(zip(query_lens, seq_lens)):
        seq_end = seq_start + seq_len
        query_end = query_start + query_len
        out = xops.memory_efficient_attention_forward(query[:,
                                                            seq_start:seq_end],
                                                      key[:,
                                                          seq_start:seq_end],
                                                      value[:,
                                                            seq_start:seq_end],
                                                      attn_bias=attn_bias[i],
                                                      p=0.0,
                                                      scale=scale)
        out = out.view_as(query[:, seq_start:seq_end]).view(
            seq_len, num_heads, head_size)
        output_ref[query_start:query_end, ...].copy_(out[seq_len - query_len:,
                                                         ...])
        seq_start += seq_len
        query_start += query_len
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"xformers Time: {(end_time - start_time)*1000:.2f} ms")
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
@torch.inference_mode()
def test_contexted_kv_attention_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:
    test_contexted_kv_attention(num_heads, num_queries_per_kv, head_size,
                                sliding_window, dtype, kv_cache_dtype, device)


@pytest.mark.optional
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_queries_per_kv", NUM_QUERIES_PER_KV)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_contexted_kv_attention_alibi_f32(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> None:
    test_contexted_kv_attention_alibi(num_heads, num_queries_per_kv, head_size,
                                      dtype, kv_cache_dtype, device)
