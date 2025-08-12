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

NUM_HEADS = [(64, 8), (16, 16), (40, 8), (32, 8)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16, 32]
DTYPES = [torch.float16, torch.bfloat16]
NUM_BLOCKS = 32768  # Large enough to test overflow in index calculation.
SOFT_CAPS = [None, 30.0, 50.0]


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.parametrize("kv_lens", [[1328, 18, 463], [1, 54, 293, 70]])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", SOFT_CAPS)
@torch.inference_mode
def test_flashinfer_trtllm_decode_with_baseline(
    kv_lens: list[int],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    kv_layout: str,
) -> None:
    torch.set_default_device("cuda")
    current_platform.seed_everything(0)
    num_seqs = len(kv_lens)
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]

    assert num_query_heads % num_kv_heads == 0
    max_kv_len = max(kv_lens)
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

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 NUM_BLOCKS,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    k_scale = v_scale = 1.0
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
    wrapper = flashinfer.\
        BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout,
                use_tensor_cores=(
                    (num_query_heads//num_kv_heads) > 4)
                )
    wrapper.plan(kv_indptr,
                 kv_indices,
                 kv_last_page_lens,
                 num_query_heads,
                 num_kv_heads,
                 head_size,
                 block_size,
                 "NONE",
                 q_data_type=dtype,
                 kv_data_type=dtype,
                 logits_soft_cap=soft_cap)

    output = wrapper.run(query, key_value_cache, scale)

    # TRTLLM Decode
    max_kv_len = max(kv_lens)
    kv_lens_tensor = torch.tensor(kv_lens,
                                  dtype=torch.int,
                                  device=query.device)
    output_trtllm = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query.contiguous(),
        key_value_cache,
        workspace_buffer,
        num_query_heads,
        num_kv_heads,
        scale,
        block_tables,
        kv_lens_tensor,
        block_size,
        max_kv_len,
        "auto",
        k_scale,
        v_scale,
    )

    torch.testing.assert_close(output, output_trtllm, atol=1e-2, rtol=1e-2), \
        f"{torch.max(torch.abs(output - output_trtllm))}"
