# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx1x, on_gfx1151
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops import chunked_prefill_paged_decode as decode_ops
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    kernel_paged_attention_2d,
    paged_attention_2d_splitkv_decode,
    paged_attention_rocm_splitkv_decode,
)

DEVICE_TYPE = current_platform.device_type


@pytest.mark.skipif(not on_gfx1151(), reason="gfx1151 native split-KV dispatch test")
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="native split-KV test requires a GPU"
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8_e4m3"])
@torch.inference_mode()
def test_gfx1151_native_splitkv_dispatch(
    kv_cache_dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Large LBHNC pages must reach the new native operator, including FP8."""
    set_random_seed(0)
    torch.set_default_device(DEVICE_TYPE)

    batch_size = 3
    num_query_heads = 24
    num_kv_heads = 4
    head_size = 256
    page_size = 1584
    seq_lens = torch.tensor([1001, 1023, 997], dtype=torch.int32)
    query = torch.randn(batch_size, num_query_heads, head_size, dtype=torch.bfloat16)
    current_key = torch.empty(batch_size, num_kv_heads, head_size, dtype=query.dtype)
    k_scale = torch.tensor(0.02, dtype=torch.float32)
    v_scale = torch.tensor(0.025, dtype=torch.float32)
    dense_key = torch.randn(
        batch_size,
        page_size,
        num_kv_heads,
        head_size,
        dtype=query.dtype,
    )
    dense_value = torch.randn_like(dense_key)
    if kv_cache_dtype == "fp8_e4m3":
        dense_key = (dense_key / k_scale).to(current_platform.fp8_dtype())
        dense_value = (dense_value / v_scale).to(current_platform.fp8_dtype())
    key_cache, value_cache = _to_vllm_kv_cache(dense_key, dense_value)
    block_tables = torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1)
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32)
    output = torch.empty_like(query)

    called = 0
    native_decode = decode_ops.paged_attention_rocm_splitkv_decode

    def record_native_decode(*args, **kwargs):
        nonlocal called
        called += 1
        return native_decode(*args, **kwargs)

    monkeypatch.setattr(
        decode_ops, "paged_attention_rocm_splitkv_decode", record_native_decode
    )
    decode_ops.chunked_prefill_paged_decode(
        query=query,
        key=current_key,
        value=current_key,
        output=output,
        kv_cache_dtype=kv_cache_dtype,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max().item()),
        max_query_len=1,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    assert called == 1

    num_splits = 16 if kv_cache_dtype == "fp8_e4m3" else 4
    expected = native_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=head_size**-0.5,
        actual_max_splits=num_splits,
        max_seq_len=int(seq_lens.max().item()),
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)


def _to_vllm_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    x = 16 // key_cache.element_size()
    key_cache = (
        key_cache.view(num_blocks, block_size, num_kv_heads, head_size // x, x)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    value_cache = value_cache.permute(0, 2, 3, 1).contiguous()
    return key_cache, value_cache


# Use the non-split triton kernel as the reference
# implementation since we are replacing this kernel
def _ref_paged_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    query_start_loc: torch.Tensor | None = None,
    filter_by_query_len: bool = False,
    k_scale: torch.Tensor | float = 1.0,
    v_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Reference decode using the non-split kernel_paged_attention_2d."""
    output = torch.empty_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = query.shape[2]
    physical_block_size = key_cache.shape[3]
    num_seqs = seq_lens.shape[0]

    is_pow2 = physical_block_size > 0 and (
        physical_block_size & (physical_block_size - 1) == 0
    )
    triton_block_size = min(physical_block_size, 128) if is_pow2 else 32

    kernel_paged_attention_2d[(num_seqs, num_kv_heads)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        sink_ptr=None,
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
        out_scale_inv=1.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(num_queries_per_kv), 16),
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        BLOCK_SIZE=triton_block_size,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        USE_SINKS=False,
        USE_FP8=False,
    )
    return output


@pytest.mark.skipif(
    not on_gfx1x(), reason="split-KV decode kernel is only activated on gfx1x"
)
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="split-KV decode test requires a GPU"
)
@pytest.mark.parametrize("num_query_heads,num_kv_heads", [(4, 4), (16, 4)])
@pytest.mark.parametrize("block_size", [16, 32, 528, 1584])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [257, 513, 1025, 2048, 4096],
        [2049, 4097, 8193, 8192, 16384],
        [8193, 12289, 16385, 16384, 16384],
    ],
)
@torch.inference_mode()
def test_paged_attention_2d_splitkv_decode(
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    batch_size: int,
    seq_lens: list[int],
) -> None:
    set_random_seed(0)
    torch.set_default_device(DEVICE_TYPE)

    head_size = 256
    dtype = torch.bfloat16
    seq_lens_tensor = torch.tensor(seq_lens[:batch_size], dtype=torch.int32)
    max_seq_len = int(seq_lens_tensor.max().item())
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_seq
    scale = head_size**-0.5

    query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
    dense_key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    dense_value_cache = torch.randn_like(dense_key_cache)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks_per_seq
    )
    key_cache, value_cache = _to_vllm_kv_cache(dense_key_cache, dense_value_cache)

    output = paged_attention_2d_splitkv_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        actual_max_splits=4,
        max_seq_len=max_seq_len,
    )
    ref_output = _ref_paged_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
    )

    # NOTE: Due to the kernel-level split-KV reduction and bfloat16 precision,
    # there is a small numerical difference. Use dtype-aware default tolerances.
    atol = get_default_atol(output)
    rtol = get_default_rtol(output)
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.skipif(
    not on_gfx1x(), reason="native ROCm split-KV decode is only built for gfx1x"
)
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="native split-KV test requires a GPU"
)
@pytest.mark.parametrize("block_size", [32, 528, 1584])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
@pytest.mark.parametrize("num_query_heads", [16, 24])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [1001, 1023, 997, 1019, 991],
        [4001, 4095, 3983, 4077, 3967],
        [16001, 16383, 15983, 16277, 15871],
    ],
)
@torch.inference_mode()
def test_paged_attention_rocm_splitkv_decode(
    block_size: int,
    batch_size: int,
    num_query_heads: int,
    dtype: torch.dtype,
    seq_lens: list[int],
) -> None:
    set_random_seed(0)
    torch.set_default_device(DEVICE_TYPE)

    head_size = 256
    num_kv_heads = 4
    seq_lens_tensor = torch.tensor(seq_lens[:batch_size], dtype=torch.int32)
    max_seq_len = int(seq_lens_tensor.max().item())
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_seq
    scale = head_size**-0.5

    query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
    dense_key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    dense_value_cache = torch.randn_like(dense_key_cache)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks_per_seq
    )
    key_cache, value_cache = _to_vllm_kv_cache(dense_key_cache, dense_value_cache)

    triton_output = paged_attention_2d_splitkv_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        actual_max_splits=4,
        max_seq_len=max_seq_len,
    )
    native_output = paged_attention_rocm_splitkv_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        actual_max_splits=4,
        max_seq_len=max_seq_len,
    )

    torch.testing.assert_close(
        native_output,
        triton_output,
        atol=get_default_atol(native_output),
        rtol=get_default_rtol(native_output),
    )


@pytest.mark.skipif(
    not on_gfx1x(), reason="native ROCm split-KV decode is only built for gfx1x"
)
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="native split-KV test requires a GPU"
)
@pytest.mark.parametrize("block_size", [32, 528, 1584])
@pytest.mark.parametrize("batch_size", [1, 3, 5])
@pytest.mark.parametrize("num_query_heads", [16, 24])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [1001, 1023, 997, 1019, 991],
        [4001, 4095, 3983, 4077, 3967],
        [16001, 16383, 15983, 16277, 15871],
    ],
)
@torch.inference_mode()
def test_paged_attention_rocm_splitkv_decode_fp8_e4m3(
    block_size: int,
    batch_size: int,
    num_query_heads: int,
    seq_lens: list[int],
) -> None:
    set_random_seed(0)
    torch.set_default_device(DEVICE_TYPE)

    head_size = 256
    num_kv_heads = 4
    dtype = torch.bfloat16
    fp8_dtype = current_platform.fp8_dtype()
    seq_lens_tensor = torch.tensor(seq_lens[:batch_size], dtype=torch.int32)
    max_seq_len = int(seq_lens_tensor.max().item())
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_seq
    scale = head_size**-0.5
    k_scale = torch.tensor(0.02, dtype=torch.float32)
    v_scale = torch.tensor(0.025, dtype=torch.float32)

    query = torch.randn(batch_size, num_query_heads, head_size, dtype=dtype)
    dense_key_cache = (
        torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
        / k_scale
    ).to(fp8_dtype)
    dense_value_cache = (
        torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=dtype)
        / v_scale
    ).to(fp8_dtype)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks_per_seq
    )
    key_cache, value_cache = _to_vllm_kv_cache(dense_key_cache, dense_value_cache)

    ref_output = _ref_paged_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    native_output = paged_attention_rocm_splitkv_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        actual_max_splits=4,
        max_seq_len=max_seq_len,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch.testing.assert_close(native_output, ref_output, atol=0.02, rtol=0.02)


@pytest.mark.skipif(
    not on_gfx1x(), reason="native ROCm split-KV decode is only built for gfx1x"
)
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="native split-KV test requires a GPU"
)
@torch.inference_mode()
def test_paged_attention_rocm_splitkv_reduction_state() -> None:
    """Check the reduction result directly from stage-1 max/sum/output."""
    set_random_seed(0)
    torch.set_default_device(DEVICE_TYPE)

    page_size = 1584
    seq_len = 4001
    num_query_heads = 24
    num_kv_heads = 4
    head_size = 256
    num_splits = 4
    num_pages = (seq_len + page_size - 1) // page_size
    query = torch.randn(1, num_query_heads, head_size, dtype=torch.bfloat16)
    dense_key_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_size, dtype=query.dtype
    )
    dense_value_cache = torch.randn_like(dense_key_cache)
    key_cache, value_cache = _to_vllm_kv_cache(dense_key_cache, dense_value_cache)
    block_tables = torch.arange(num_pages, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    partial_out = torch.empty(
        1, num_query_heads, num_splits, head_size, dtype=torch.float32
    )
    partial_max = torch.empty(1, num_query_heads, num_splits, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)

    output = paged_attention_rocm_splitkv_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        head_size**-0.5,
        actual_max_splits=num_splits,
        max_seq_len=seq_len,
        partial_out=partial_out,
        partial_max=partial_max,
        partial_sum=partial_sum,
    )
    global_max = partial_max.max(dim=2, keepdim=True).values
    split_weight = torch.exp(partial_max - global_max)
    expected = (partial_out * split_weight.unsqueeze(-1)).sum(dim=2) / (
        partial_sum * split_weight
    ).sum(dim=2, keepdim=True)

    torch.testing.assert_close(
        output,
        expected.to(output.dtype),
        atol=get_default_atol(output),
        rtol=get_default_rtol(output),
    )


@pytest.mark.skipif(
    not on_gfx1x(), reason="split-KV decode kernel is only activated on gfx1x"
)
@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="split-KV decode test requires a GPU"
)
@pytest.mark.parametrize("block_size", [16, 32, 528, 1584])
@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ([1, 3, 1], [257, 513, 1025]),
        ([1, 1, 1], [2049, 4097, 8193]),
        ([1, 3, 1], [8193, 12289, 16385]),
        ([1], [4097]),
        ([1, 1], [8193, 16385]),
    ],
)
@torch.inference_mode()
def test_paged_attention_2d_splitkv_decode_filters_prefill_queries(
    block_size: int,
    query_lens: list[int],
    seq_lens: list[int],
) -> None:
    set_random_seed(0)
    torch.set_default_device(DEVICE_TYPE)

    query_start_loc = torch.tensor(
        [0] + [sum(query_lens[: i + 1]) for i in range(len(query_lens))],
        dtype=torch.int32,
    )
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    batch_size = len(query_lens)
    num_query_heads = 4
    num_kv_heads = 4
    head_size = 256
    dtype = torch.bfloat16
    max_seq_len = int(seq_lens_tensor.max().item())
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_seq
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    dense_key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    dense_value_cache = torch.randn_like(dense_key_cache)
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(
        batch_size, num_blocks_per_seq
    )
    key_cache, value_cache = _to_vllm_kv_cache(dense_key_cache, dense_value_cache)
    output = torch.full_like(query, 7.0)

    paged_attention_2d_splitkv_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        output=output,
        actual_max_splits=4,
        max_seq_len=max_seq_len,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )
    ref_output = _ref_paged_decode(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        scale=scale,
        query_start_loc=query_start_loc,
        filter_by_query_len=True,
    )

    # NOTE: Due to the kernel-level split-KV reduction and bfloat16 precision,
    # there is a small numerical difference. Use dtype-aware default tolerances.
    atol = get_default_atol(output)
    rtol = get_default_rtol(output)
    # Decode queries (query_len==1) should match the reference.
    # Prefill queries (query_len>1) should be untouched (still 7.0).
    prefill_start = 0
    for i, ql in enumerate(query_lens):
        if ql > 1:
            torch.testing.assert_close(
                output[prefill_start : prefill_start + ql],
                torch.full_like(output[prefill_start : prefill_start + ql], 7.0),
            )
        else:
            torch.testing.assert_close(
                output[prefill_start],
                ref_output[prefill_start],
                atol=atol,
                rtol=rtol,
            )
        prefill_start += ql
