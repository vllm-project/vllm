# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import flashinfer
import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (FLOAT4_E2M1_MAX,
                                                    FLOAT8_E4M3_MAX,
                                                    dequantize_nvfp4_to_dtype)
from vllm.platforms import current_platform
from vllm.utils import round_up

if not current_platform.is_device_capability(100):
    pytest.skip("This TRTLLM kernel requires NVIDIA Blackwell.",
                allow_module_level=True)

FLOAT32_BYTES = torch.finfo(torch.float).bits // 8
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


DTYPE = [torch.bfloat16]
QUANT_DTYPES = [
    # (q_quant_dtype, kv_quant_dtype, o_quant_dtype)
    (None, None, None),
    (None, FP8_DTYPE, None),
    (FP8_DTYPE, FP8_DTYPE, None),
    (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
    (FP8_DTYPE, FP8_DTYPE, FP4_DTYPE),
]
BATCH_SIZE = [4, 12]
MAX_SEQ_LENS = [(1024, 4096)]
NUM_HEADS = [(64, 8), (40, 8)]
HEAD_SIZE = [128]
KV_LAYOUT = ["HND"]  # currently only HND is supported
BLOCK_SIZE = [16]
WINDOW_LEFT = [-1, 127]
SOFT_CAP = [None, 50.0]

NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.


@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("quant_dtypes", QUANT_DTYPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("max_seq_lens", MAX_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZE)
@pytest.mark.parametrize("kv_layout", KV_LAYOUT)
@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("window_left", WINDOW_LEFT)
@pytest.mark.parametrize("soft_cap", SOFT_CAP)
@torch.inference_mode
def test_flashinfer_trtllm_decode_with_baseline(
    dtype: torch.dtype,
    quant_dtypes: tuple[Optional[torch.dtype], Optional[torch.dtype],
                        Optional[torch.dtype]],
    batch_size: int,
    max_seq_lens: tuple[int, int],
    num_heads: tuple[int, int],
    head_size: int,
    kv_layout: str,
    block_size: int,
    window_left: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtypes
    q_quant_dtype = q_quant_dtype or dtype
    kv_quant_dtype = kv_quant_dtype or dtype
    o_quant_dtype = o_quant_dtype or dtype

    _, max_kv_len = max_seq_lens

    num_qo_heads, num_kv_heads = num_heads
    assert num_qo_heads % num_kv_heads == 0

    sm_scale = float(1.0 / (head_size**0.5))

    kv_cache_shape = None
    if kv_layout == "NHD":
        kv_cache_shape = (NUM_BLOCKS, 2, block_size, num_kv_heads, head_size)
    elif kv_layout == "HND":
        kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, block_size, head_size)
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")

    query = torch.randn(batch_size, num_qo_heads, head_size, dtype=dtype)
    if q_quant_dtype == FP8_DTYPE:
        query, q_scale = to_float8(query)
        ref_query = query.to(dtype) * q_scale
    else:
        q_scale = 1.0
        ref_query = query

    kv_lens = torch.randint(1, max_kv_len, (batch_size, ), dtype=torch.int32)
    kv_lens[-1] = max_kv_len

    seq_lens = kv_lens
    max_seq_len = torch.max(seq_lens).item()

    kv_cache = torch.randn(kv_cache_shape, dtype=dtype)
    if kv_quant_dtype == FP8_DTYPE:
        kv_cache, kv_scale = to_float8(kv_cache)
        ref_kv_cache = kv_cache.to(dtype) * kv_scale
    else:
        kv_scale = 1.0
        ref_kv_cache = kv_cache
    k_scale = v_scale = kv_scale

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (batch_size, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(batch_size):
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
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.int8)

    # Baseline Decode
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True)
    wrapper.plan(kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_qo_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 "NONE",
                 sm_scale=sm_scale,
                 q_data_type=dtype,
                 kv_data_type=dtype,
                 window_left=window_left,
                 logits_soft_cap=soft_cap)

    output = torch.empty(ref_query.shape, dtype=dtype)
    wrapper.run(ref_query, ref_kv_cache, out=output)
    o_scale = 1.0
    o_sf_scale = None
    if o_quant_dtype == FP8_DTYPE:
        _, o_scale = to_float8(output)
    elif o_quant_dtype == FP4_DTYPE:
        o_sf_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                      torch.amax(output.flatten(), dim=-1)).to(torch.float32)

    # TRTLLM Decode
    if o_quant_dtype == FP4_DTYPE:
        output_trtllm = flashinfer.utils.FP4Tensor(
            torch.empty(query.shape[:-1] + (query.shape[-1] // 2, ),
                        dtype=torch.uint8),
            torch.empty((round_up(query.shape[0], 128),
                         round_up(query.shape[1] * query.shape[2] // 16, 4)),
                        dtype=torch.float8_e4m3fn),
        )
    else:
        output_trtllm = torch.empty(query.shape, dtype=o_quant_dtype)

    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        bmm1_scale=q_scale * k_scale * sm_scale,
        bmm2_scale=v_scale / o_scale,
        window_left=window_left,
        o_sf_scale=o_sf_scale,
        out=output_trtllm,
    )
    if o_quant_dtype == FP8_DTYPE:
        output_trtllm = output_trtllm.to(dtype) * o_scale
    elif o_quant_dtype == FP4_DTYPE:
        output_trtllm.data = output_trtllm.data.reshape(
            -1, query.shape[1] * query.shape[2] // 2)
        output_trtllm = dequantize_nvfp4_to_dtype(output_trtllm.data,
                                                  output_trtllm.scale,
                                                  o_sf_scale, dtype,
                                                  query.device)
        output_trtllm = output_trtllm.reshape(-1, query.shape[1],
                                              query.shape[2])

    if q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP4_DTYPE:
        rtol, atol = 3e-1, 1e0
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP8_DTYPE:
        rtol, atol = 5e-2, 7e-2
    else:
        rtol, atol = 1e-2, 2e-2

    torch.testing.assert_close(output, output_trtllm, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - output_trtllm))}"


@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("quant_dtypes", QUANT_DTYPES)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("max_seq_lens", MAX_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZE)
@pytest.mark.parametrize("kv_layout", KV_LAYOUT)
@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("window_left", WINDOW_LEFT)
@pytest.mark.parametrize("soft_cap", [None])
@torch.inference_mode
def test_flashinfer_trtllm_prefill_with_baseline(
    dtype: torch.dtype,
    quant_dtypes: tuple[Optional[torch.dtype], Optional[torch.dtype],
                        Optional[torch.dtype]],
    batch_size: int,
    max_seq_lens: tuple[int, int],
    num_heads: tuple[int, int],
    head_size: int,
    kv_layout: str,
    block_size: int,
    window_left: int,
    soft_cap: Optional[float],
) -> None:
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)

    q_quant_dtype, kv_quant_dtype, o_quant_dtype = quant_dtypes
    q_quant_dtype = q_quant_dtype or dtype
    kv_quant_dtype = kv_quant_dtype or dtype
    o_quant_dtype = o_quant_dtype or dtype

    if q_quant_dtype != kv_quant_dtype:
        pytest.skip("Skipped mixed QKV dtypes for prefill")

    max_q_len, max_kv_len = max_seq_lens

    num_qo_heads, num_kv_heads = num_heads
    assert num_qo_heads % num_kv_heads == 0

    sm_scale = float(1.0 / (head_size**0.5))

    kv_cache_shape = None
    if kv_layout == "NHD":
        kv_cache_shape = (NUM_BLOCKS, 2, block_size, num_kv_heads, head_size)
    elif kv_layout == "HND":
        kv_cache_shape = (NUM_BLOCKS, 2, num_kv_heads, block_size, head_size)
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")

    q_lens = torch.randint(1, max_q_len, (batch_size, ), dtype=torch.int32)
    q_lens[-1] = max_q_len
    q_indptr = torch.cat([
        torch.tensor([0], dtype=torch.int32),
        torch.cumsum(q_lens, dim=0, dtype=torch.int32),
    ])

    query = torch.randn(torch.sum(q_lens).item(),
                        num_qo_heads,
                        head_size,
                        dtype=dtype)
    if q_quant_dtype == FP8_DTYPE:
        query, q_scale = to_float8(query)
        ref_query = query.to(dtype) * q_scale
    else:
        q_scale = 1.0
        ref_query = query

    kv_lens = torch.randint(0, max_kv_len, (batch_size, ), dtype=torch.int32)
    kv_lens[-1] = max_kv_len

    seq_lens = kv_lens + q_lens
    max_seq_len = torch.max(seq_lens).item()

    kv_cache = torch.randn(kv_cache_shape, dtype=dtype)
    if kv_quant_dtype == FP8_DTYPE:
        kv_cache, kv_scale = to_float8(kv_cache)
        ref_kv_cache = kv_cache.to(dtype) * kv_scale
    else:
        kv_scale = 1.0
        ref_kv_cache = kv_cache
    k_scale = v_scale = kv_scale

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (batch_size, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(batch_size):
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
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.int8)

    # Baseline Prefill
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout)
    wrapper.plan(q_indptr,
                 kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_qo_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 causal=True,
                 sm_scale=sm_scale,
                 q_data_type=dtype,
                 kv_data_type=dtype,
                 window_left=window_left,
                 logits_soft_cap=soft_cap)

    output = torch.empty(ref_query.shape, dtype=dtype)
    wrapper.run(ref_query, ref_kv_cache, out=output)
    o_scale = 1.0
    o_sf_scale = None
    if o_quant_dtype == FP8_DTYPE:
        _, o_scale = to_float8(output)
    elif o_quant_dtype == FP4_DTYPE:
        o_sf_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                      torch.amax(output.flatten(), dim=-1)).to(torch.float32)

    # TRTLLM Prefill
    if o_quant_dtype == FP4_DTYPE:
        output_trtllm = flashinfer.utils.FP4Tensor(
            torch.empty(query.shape[:-1] + (query.shape[-1] // 2, ),
                        dtype=torch.uint8),
            torch.empty((round_up(query.shape[0], 128),
                         round_up(query.shape[1] * query.shape[2] // 16, 4)),
                        dtype=torch.float8_e4m3fn),
        )
    else:
        output_trtllm = torch.empty(query.shape, dtype=o_quant_dtype)

    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_seq_len,
        bmm1_scale=q_scale * k_scale * sm_scale,
        bmm2_scale=v_scale / o_scale,
        batch_size=batch_size,
        cum_seq_lens_q=q_indptr,
        cum_seq_lens_kv=kv_indptr,
        window_left=window_left,
        o_sf_scale=o_sf_scale,
        out=output_trtllm,
    )
    if o_quant_dtype == FP8_DTYPE:
        output_trtllm = output_trtllm.to(dtype) * o_scale
    elif o_quant_dtype == FP4_DTYPE:
        output_trtllm.data = output_trtllm.data.reshape(
            -1, query.shape[1] * query.shape[2] // 2)
        output_trtllm = dequantize_nvfp4_to_dtype(output_trtllm.data,
                                                  output_trtllm.scale,
                                                  o_sf_scale, dtype,
                                                  query.device)
        output_trtllm = output_trtllm.reshape(-1, query.shape[1],
                                              query.shape[2])

    if q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP4_DTYPE:
        rtol, atol = 4e-1, 1e0
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP8_DTYPE:
        rtol, atol = 5e-2, 7e-2
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == dtype:
        rtol, atol = 4e-2, 6e-2
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(output, output_trtllm, atol=atol, rtol=rtol), \
        f"{torch.max(torch.abs(output - output_trtllm))}"
