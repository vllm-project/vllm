# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

NUM_HEADS = [(4, 4), (8, 2), (40, 8)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16]
QDTYPES = (
    [None, torch.float8_e4m3fn]
    if not current_platform.is_rocm()
    else [None, torch.float8_e4m3fnuz]
)
FP8_DTYPE = (
    torch.float8_e4m3fn if not current_platform.is_rocm() else torch.float8_e4m3fnuz
)
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]

# 0: use 2D kernel for decode
# 8: use 3D kernel for decode
SEQ_THRESHOLD_3D_VALUES = [0, 8]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype | None,
    seq_threshold_3D: int,
) -> None:
    torch.set_default_device("cuda")

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = None  # Not yet supported
        k_descale = torch.rand(scale_shape, dtype=torch.float32)
        v_descale = torch.rand(scale_shape, dtype=torch.float32)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1)] * 533,
        [(533, 533)] * 533,
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_fp16_input_fp8_output(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test with fp16 input and fp8 output using output_scale."""
    torch.set_default_device("cuda")

    set_random_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    dtype = torch.float16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty(sum(query_lens), num_query_heads, head_size, dtype=FP8_DTYPE)

    output_scale = torch.tensor(0.5, dtype=torch.float32)

    num_par_softmax_segments = 16
    head_size_padded = next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_scale=output_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    output_fp16 = output.to(torch.float32) * output_scale.item()
    output_fp16 = output_fp16.to(torch.float16)

    atol, rtol = 2e-1, 2e-1
    (
        torch.testing.assert_close(output_fp16, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output_fp16 - ref_output))}",
    )
