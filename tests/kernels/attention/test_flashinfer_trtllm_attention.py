# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import flashinfer
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability(100):
    pytest.skip("This TRTLLM kernel requires NVIDIA Blackwell.",
                allow_module_level=True)

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8

# KV Cache Layout for TRT-LLM
# kv_cache_shape = (num_blocks, 2, num_kv_heads, page_size, head_dim)

MAX_Q_LEN = 1024
MAX_KV_LEN = 4096
BATCH_SIZES = [4, 12]
NUM_HEADS = [(64, 8), (16, 16), (40, 8), (32, 8)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16, 32]
KV_LAYOUTS = ["HND"]
DTYPES = [torch.float16, torch.bfloat16]
KV_CACHE_DTYPES = [None, current_platform.fp8_dtype()]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
SOFT_CAPS = [None, 50.0]


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("kv_layout", KV_LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@torch.inference_mode
def test_flashinfer_trtllm_decode_with_baseline(
    batch_size: int,
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    kv_layout: str,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[torch.dtype],
    soft_cap: Optional[float],
) -> None:
    kv_cache_dtype = dtype if kv_cache_dtype is None else kv_cache_dtype

    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    kv_lens = torch.randint(1, MAX_KV_LEN, (batch_size, ), dtype=torch.int32)
    kv_lens[-1] = MAX_KV_LEN
    max_kv_len = torch.max(kv_lens).item()
    num_seqs = len(kv_lens)

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    scale = head_size**-0.5

    query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

    kv_cache_shape = None
    if kv_layout == "NHD":
        kv_cache_shape = (NUM_BLOCKS, 2, block_size, num_kv_heads, head_size)
    elif kv_layout == "HND":
        kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, block_size, head_size)
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")
    key_value_cache = torch.randn(kv_cache_shape, dtype=dtype)
    kv_scale = 1.0
    if kv_cache_dtype is current_platform.fp8_dtype():
        key_value_cache, kv_scale = to_float8(key_value_cache,
                                              current_platform.fp8_dtype())

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    k_scale = v_scale = kv_scale
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = kv_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_tensor_cores=((num_query_heads // num_kv_heads) > 4))
    wrapper.plan(kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_query_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 "NONE",
                 sm_scale=scale,
                 q_data_type=dtype,
                 kv_data_type=kv_cache_dtype,
                 logits_soft_cap=soft_cap)

    output = torch.empty(query.shape, dtype=dtype)
    wrapper.run(query,
                key_value_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                out=output)

    # TRTLLM Decode
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    output_trtllm = torch.empty(query.shape, dtype=dtype)
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=query.contiguous(),
        kv_cache=key_value_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=kv_lens_tensor,
        max_seq_len=max_kv_len,
        bmm1_scale=k_scale * scale,
        bmm2_scale=v_scale,
        out=output_trtllm,
    )

    torch.testing.assert_close(output, output_trtllm, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - output_trtllm))}"


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("kv_layout", KV_LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@torch.inference_mode
def test_flashinfer_trtllm_prefill_with_baseline(
    batch_size: int,
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    kv_layout: str,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[torch.dtype],
    soft_cap: Optional[float],
) -> None:
    kv_cache_dtype = dtype if kv_cache_dtype is None else kv_cache_dtype
    if dtype != kv_cache_dtype:
        pytest.skip(f"Not supported dtype({dtype}) with "
                    "kv_cache_dtype({kv_cache_dtype})")

    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q_lens = torch.randint(1, MAX_Q_LEN, (batch_size, ), dtype=torch.int32)
    q_lens[-1] = MAX_Q_LEN
    max_q_len = torch.max(q_lens).item()
    q_indptr = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.cumsum(q_lens, dim=0, dtype=torch.int32),
    ])

    kv_lens = torch.randint(0, MAX_KV_LEN, (batch_size, ), dtype=torch.int32)
    kv_lens[-1] = MAX_KV_LEN

    seq_lens = kv_lens + q_lens
    max_seq_len = torch.max(seq_lens).item()
    num_seqs = len(seq_lens)

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    scale = head_size**-0.5

    query = torch.randn(torch.sum(q_lens).item(),
                        num_query_heads,
                        head_size,
                        dtype=dtype)

    kv_cache_shape = None
    if kv_layout == "NHD":
        kv_cache_shape = (NUM_BLOCKS, 2, block_size, num_kv_heads, head_size)
    elif kv_layout == "HND":
        kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, block_size, head_size)
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")
    key_value_cache = torch.randn(kv_cache_shape, dtype=dtype)
    kv_scale = 1.0
    if kv_cache_dtype is current_platform.fp8_dtype():
        key_value_cache, kv_scale = to_float8(key_value_cache,
                                              current_platform.fp8_dtype())

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    k_scale = v_scale = kv_scale
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(num_seqs):
        seq_len = seq_lens[i]
        assert seq_len > 0
        num_blocks = (seq_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)
        kv_last_page_len = seq_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout)
    wrapper.plan(q_indptr,
                 kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_query_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 causal=True,
                 sm_scale=scale,
                 q_data_type=dtype,
                 kv_data_type=kv_cache_dtype,
                 logits_soft_cap=soft_cap)

    output = torch.empty(query.shape, dtype=dtype)
    wrapper.run(query,
                key_value_cache,
                k_scale=k_scale,
                v_scale=v_scale,
                out=output)

    # TRTLLM Decode
    output_trtllm = torch.empty(query.shape, dtype=dtype)
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=query.contiguous(),
        kv_cache=key_value_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_seq_len,
        bmm1_scale=k_scale * scale,
        bmm2_scale=v_scale,
        batch_size=num_seqs,
        cum_seq_lens_q=q_indptr,
        cum_seq_lens_kv=kv_indptr,
        out=output_trtllm,
    )

    torch.testing.assert_close(output, output_trtllm, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - output_trtllm))}"
