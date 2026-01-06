# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (
    dequantize_nvfp4_to_dtype,
    get_nvfp4_global_scale,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed

if not current_platform.is_device_capability_family(100):
    pytest.skip(
        "This TRTLLM kernel requires NVIDIA Blackwell.", allow_module_level=True
    )
else:
    import flashinfer

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
HAS_SINKS = [True, False]

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
@pytest.mark.parametrize("has_sinks", HAS_SINKS)
@torch.inference_mode
def test_flashinfer_trtllm_decode_with_baseline(
    dtype: torch.dtype,
    quant_dtypes: tuple[torch.dtype | None, torch.dtype | None, torch.dtype | None],
    batch_size: int,
    max_seq_lens: tuple[int, int],
    num_heads: tuple[int, int],
    head_size: int,
    kv_layout: str,
    block_size: int,
    window_left: int,
    soft_cap: float | None,
    has_sinks: bool,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(42)

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

    # max_q_len = 1
    q_lens = torch.ones((batch_size,), dtype=torch.int32)
    q_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ]
    )

    query = torch.randn(torch.sum(q_lens).item(), num_qo_heads, head_size, dtype=dtype)
    if q_quant_dtype == FP8_DTYPE:
        query, q_scale = to_float8(query)
        ref_query = query.to(dtype) * q_scale
    else:
        q_scale = 1.0
        ref_query = query

    kv_lens = torch.randint(1, max_kv_len, (batch_size,), dtype=torch.int32)
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
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
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
    if has_sinks:
        sinks = torch.rand(num_qo_heads, dtype=torch.float32) * 5
        wrapper = flashinfer.BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace_buffer, kv_layout=kv_layout, backend="fa2"
        )
    else:
        sinks = None
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=workspace_buffer, kv_layout=kv_layout, backend="fa2"
        )

    wrapper.plan(
        qo_indptr=q_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_lens,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_size,
        page_size=block_size,
        causal=True,
        sm_scale=sm_scale,
        window_left=window_left,
        logits_soft_cap=soft_cap,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    output = torch.empty(ref_query.shape, dtype=dtype)
    wrapper.run(ref_query, ref_kv_cache, sinks, sm_scale, out=output)

    o_scale = 1.0
    o_sf_scale_float = None
    if o_quant_dtype == FP8_DTYPE:
        _, o_scale = to_float8(output)
    elif o_quant_dtype == FP4_DTYPE:
        o_sf_scale = get_nvfp4_global_scale(output)
        o_sf_scale_float = o_sf_scale.item()

    # TRTLLM Decode
    if o_quant_dtype == FP4_DTYPE:
        output_trtllm = flashinfer.utils.FP4Tensor(
            torch.empty(query.shape[:-1] + (query.shape[-1] // 2,), dtype=torch.uint8),
            torch.empty(
                (
                    round_up(query.shape[0], 128),
                    round_up(query.shape[1] * query.shape[2] // 16, 4),
                ),
                dtype=torch.float8_e4m3fn,
            ),
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
        sinks=sinks,
        o_sf_scale=o_sf_scale_float,
        out=output_trtllm,
    )
    if o_quant_dtype == FP8_DTYPE:
        output_trtllm = output_trtllm.to(dtype) * o_scale
    elif o_quant_dtype == FP4_DTYPE:
        output_trtllm.data = output_trtllm.data.reshape(
            -1, query.shape[1] * query.shape[2] // 2
        )
        output_trtllm = dequantize_nvfp4_to_dtype(
            output_trtllm.data, output_trtllm.scale, o_sf_scale, dtype, query.device
        )
        output_trtllm = output_trtllm.reshape(-1, query.shape[1], query.shape[2])

    if q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP4_DTYPE:
        rtol, atol = 7e-2, 9e-2
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP8_DTYPE:
        rtol, atol = 3e-2, 4e-2
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == dtype:
        rtol, atol = 2e-2, 2e-2
    elif kv_quant_dtype == FP8_DTYPE:
        rtol, atol = 4e-2, 6e-2
    else:
        rtol, atol = 1e-2, 1e-2

    (
        torch.testing.assert_close(output, output_trtllm, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - output_trtllm))}",
    )


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
@pytest.mark.parametrize("has_sinks", HAS_SINKS)
@torch.inference_mode
def test_flashinfer_trtllm_prefill_with_baseline(
    dtype: torch.dtype,
    quant_dtypes: tuple[torch.dtype | None, torch.dtype | None, torch.dtype | None],
    batch_size: int,
    max_seq_lens: tuple[int, int],
    num_heads: tuple[int, int],
    head_size: int,
    kv_layout: str,
    block_size: int,
    window_left: int,
    soft_cap: float | None,
    has_sinks: bool,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(42)

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

    q_lens = torch.randint(1, max_q_len, (batch_size,), dtype=torch.int32)
    q_lens[-1] = max_q_len
    q_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            torch.cumsum(q_lens, dim=0, dtype=torch.int32),
        ]
    )

    query = torch.randn(torch.sum(q_lens).item(), num_qo_heads, head_size, dtype=dtype)
    if q_quant_dtype == FP8_DTYPE:
        query, q_scale = to_float8(query)
        ref_query = query.to(dtype) * q_scale
    else:
        q_scale = 1.0
        ref_query = query

    kv_lens = torch.randint(1, max_kv_len, (batch_size,), dtype=torch.int32)
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
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
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
    if has_sinks:
        sinks = torch.rand(num_qo_heads, dtype=torch.float32) * 5
        wrapper = flashinfer.BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace_buffer, kv_layout=kv_layout, backend="fa2"
        )
    else:
        sinks = None
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_workspace_buffer=workspace_buffer, kv_layout=kv_layout, backend="fa2"
        )

    wrapper.plan(
        qo_indptr=q_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_lens,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_size,
        page_size=block_size,
        causal=True,
        sm_scale=sm_scale,
        window_left=window_left,
        logits_soft_cap=soft_cap,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    output = torch.empty(ref_query.shape, dtype=dtype)
    wrapper.run(ref_query, ref_kv_cache, sinks, sm_scale, out=output)

    o_scale = 1.0
    o_sf_scale_float = None
    if o_quant_dtype == FP8_DTYPE:
        _, o_scale = to_float8(output)
    elif o_quant_dtype == FP4_DTYPE:
        o_sf_scale = get_nvfp4_global_scale(output)
        o_sf_scale_float = o_sf_scale.item()

    # TRTLLM Prefill
    if o_quant_dtype == FP4_DTYPE:
        output_trtllm = flashinfer.utils.FP4Tensor(
            torch.empty(query.shape[:-1] + (query.shape[-1] // 2,), dtype=torch.uint8),
            torch.empty(
                (
                    round_up(query.shape[0], 128),
                    round_up(query.shape[1] * query.shape[2] // 16, 4),
                ),
                dtype=torch.float8_e4m3fn,
            ),
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
        sinks=sinks,
        o_sf_scale=o_sf_scale_float,
        out=output_trtllm,
    )
    if o_quant_dtype == FP8_DTYPE:
        output_trtllm = output_trtllm.to(dtype) * o_scale
    elif o_quant_dtype == FP4_DTYPE:
        output_trtllm.data = output_trtllm.data.reshape(
            -1, query.shape[1] * query.shape[2] // 2
        )
        output_trtllm = dequantize_nvfp4_to_dtype(
            output_trtllm.data, output_trtllm.scale, o_sf_scale, dtype, query.device
        )
        output_trtllm = output_trtllm.reshape(-1, query.shape[1], query.shape[2])

    if q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP4_DTYPE:
        rtol, atol = 3e-1, 4e-1
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP8_DTYPE:
        rtol, atol = 4e-2, 6e-2
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == dtype:
        rtol, atol = 2e-2, 3e-2
    else:
        rtol, atol = 1e-2, 1e-2

    (
        torch.testing.assert_close(output, output_trtllm, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - output_trtllm))}",
    )


def test_trtllm_attention_rejects_num_kv_heads_1() -> None:
    """Test that TRTLLM attention correctly rejects num_kv_heads=1.

    When num_kv_heads=1 (MQA), the KV cache strides become degenerate
    (stride_heads == stride_batch), which causes CUDA's cuTensorMapEncodeTiled
    to fail because TMA descriptors cannot handle degenerate 4D tensors with
    singleton dimensions.

    This test verifies that can_use_trtllm_attention returns False for
    num_kv_heads=1 configurations.
    """
    from vllm.utils.flashinfer import can_use_trtllm_attention

    # num_kv_heads=1 should be rejected
    assert not can_use_trtllm_attention(num_qo_heads=64, num_kv_heads=1), (
        "can_use_trtllm_attention should return False for num_kv_heads=1"
    )
    assert not can_use_trtllm_attention(num_qo_heads=32, num_kv_heads=1), (
        "can_use_trtllm_attention should return False for num_kv_heads=1"
    )

    # num_kv_heads > 1 should be accepted (if platform supports it)
    # Note: This may return False on non-Blackwell platforms, which is fine
    result_kv8 = can_use_trtllm_attention(num_qo_heads=64, num_kv_heads=8)
    result_kv1 = can_use_trtllm_attention(num_qo_heads=64, num_kv_heads=1)

    # Even if platform doesn't support TRTLLM, num_kv_heads=1 should never
    # return True when num_kv_heads > 1 returns True
    if result_kv8:
        assert not result_kv1, (
            "If TRTLLM is supported for num_kv_heads=8, "
            "it must be rejected for num_kv_heads=1"
        )
