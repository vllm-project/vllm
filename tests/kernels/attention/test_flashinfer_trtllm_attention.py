# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.quantization.nvfp4_utils import (
    dequant_nvfp4_kv_cache,
    dequantize_nvfp4_to_dtype,
    get_nvfp4_global_scale,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import (
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
    set_random_seed,
)

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


def build_paged_kv_metadata(
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged-KV indptr/indices/last_page_lens from seq_lens + block_tables."""
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []
    for i in range(len(seq_lens)):
        sl = int(seq_lens[i])
        assert sl > 0
        nb = (sl + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :nb].tolist())
        kv_indptr.append(kv_indptr[-1] + nb)
        kv_last_page_lens.append(sl % block_size or block_size)
    return (
        torch.tensor(kv_indptr, dtype=torch.int32),
        torch.tensor(kv_indices, dtype=torch.int32),
        torch.tensor(kv_last_page_lens, dtype=torch.int32),
    )


def make_nvfp4_kv_cache(
    kv_bf16_hnd: torch.Tensor, block_size: int, head_size: int
) -> tuple:
    """Quantize bf16 KV cache to nvfp4 via reshape_and_cache_flash.

    Returns (k_data, v_data), (k_scales, v_scales), kv_scale, ref_kv_bf16.
    """
    num_blocks, _, num_kv_heads, _, _ = kv_bf16_hnd.shape
    kv_scale_val = (kv_bf16_hnd.abs().amax() / 448.0).item()
    kv_scale_tensor = torch.tensor(
        kv_scale_val, dtype=torch.float32, device=kv_bf16_hnd.device
    )

    # Allocate in HND physical order, permute to NHD logical order.
    # hnd_order swaps dims 2↔3; it is its own inverse.
    full_dim = nvfp4_kv_cache_full_dim(head_size)
    hnd_order = (0, 1, 3, 2, 4)
    kv_cache = torch.zeros(
        (num_blocks, 2, num_kv_heads, block_size, full_dim),
        dtype=torch.uint8,
        device=kv_bf16_hnd.device,
    ).permute(*hnd_order)

    # Flatten NHD [N, T, H, D] → token tensors [N*T, H, D] for the kernel.
    num_tokens = num_blocks * block_size
    k_tokens = (
        kv_bf16_hnd[:, 0]
        .permute(0, 2, 1, 3)
        .reshape(num_tokens, num_kv_heads, head_size)
    )
    v_tokens = (
        kv_bf16_hnd[:, 1]
        .permute(0, 2, 1, 3)
        .reshape(num_tokens, num_kv_heads, head_size)
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=kv_bf16_hnd.device)

    # reshape_and_cache_flash: kernel receives kv_cache[:, 0] and [:, 1]
    # (full K/V buffers containing both data and scale).
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        k_tokens,
        v_tokens,
        kv_cache[:, 0],
        kv_cache[:, 1],
        slot_mapping,
        "nvfp4",
        kv_scale_tensor,
        kv_scale_tensor,
    )

    # Split in HND order for trtllm kernel (expects HND numTokensPerPage).
    kv_cache_hnd = kv_cache.permute(*hnd_order)
    (k_data, v_data), (k_scales, v_scales) = nvfp4_kv_cache_split_views(kv_cache_hnd)

    # Dequantize for the FA2 reference baseline.
    ref_k = dequant_nvfp4_kv_cache(
        k_data, k_scales, kv_scale_val, head_size, block_size
    ).to(torch.bfloat16)
    ref_v = dequant_nvfp4_kv_cache(
        v_data, v_scales, kv_scale_val, head_size, block_size
    ).to(torch.bfloat16)
    ref_kv_bf16 = torch.stack([ref_k, ref_v], dim=1)  # [N, 2, H, T, D]

    return (k_data, v_data), (k_scales, v_scales), kv_scale_val, ref_kv_bf16


def make_quantized_kv_cache(
    kv_cache: torch.Tensor,
    kv_quant_dtype: torch.dtype,
    block_size: int,
    head_size: int,
) -> tuple:
    """Quantize kv_cache based on dtype. Returns (kv_cache, kv_cache_sf,
    kv_scale, ref_kv_cache, is_nvfp4_kv)."""
    is_nvfp4_kv = kv_quant_dtype == FP4_DTYPE
    if is_nvfp4_kv:
        data, scales, kv_scale, ref = make_nvfp4_kv_cache(
            kv_cache, block_size, head_size
        )
        return data, scales, kv_scale, ref, True
    elif kv_quant_dtype == FP8_DTYPE:
        kv_fp8, kv_scale = to_float8(kv_cache)
        ref = kv_fp8.to(kv_cache.dtype) * kv_scale
        return kv_fp8, None, kv_scale, ref, False
    else:
        return kv_cache, None, 1.0, kv_cache, False


DTYPE = [torch.bfloat16]
QUANT_DTYPES = [
    # (q_quant_dtype, kv_quant_dtype, o_quant_dtype)
    (None, None, None),
    (None, FP8_DTYPE, None),
    (FP8_DTYPE, FP8_DTYPE, None),
    (FP8_DTYPE, FP8_DTYPE, FP8_DTYPE),
    (FP8_DTYPE, FP8_DTYPE, FP4_DTYPE),
    (FP8_DTYPE, FP4_DTYPE, FP8_DTYPE),  # nvfp4 KV cache
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
    kv_cache, kv_cache_sf, kv_scale, ref_kv_cache, is_nvfp4_kv = (
        make_quantized_kv_cache(kv_cache, kv_quant_dtype, block_size, head_size)
    )

    k_scale = v_scale = kv_scale

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    kv_indptr, kv_indices, kv_last_page_lens = build_paged_kv_metadata(
        seq_lens, block_tables, block_size
    )
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
        kv_cache_sf=kv_cache_sf,
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

    if is_nvfp4_kv:
        rtol, atol = 1.0, 1.0  # nvfp4 has higher quantization error
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP4_DTYPE:
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

    # FP8 Q + nvfp4 KV is the required combination for the nvfp4 KV path.
    # All other mixed Q/KV dtype combinations are unsupported.
    is_nvfp4_kv = kv_quant_dtype == FP4_DTYPE
    if q_quant_dtype != kv_quant_dtype and not (
        q_quant_dtype == FP8_DTYPE and is_nvfp4_kv
    ):
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
    kv_cache, kv_cache_sf, kv_scale, ref_kv_cache, is_nvfp4_kv = (
        make_quantized_kv_cache(kv_cache, kv_quant_dtype, block_size, head_size)
    )

    k_scale = v_scale = kv_scale

    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (batch_size, max_num_blocks_per_seq), dtype=torch.int32
    )
    kv_indptr, kv_indices, kv_last_page_lens = build_paged_kv_metadata(
        seq_lens, block_tables, block_size
    )
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
        kv_cache_sf=kv_cache_sf,
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

    if is_nvfp4_kv:
        rtol, atol = 1.0, 1.5  # nvfp4 has higher quantization error
    elif q_quant_dtype == FP8_DTYPE and o_quant_dtype == FP4_DTYPE:
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
