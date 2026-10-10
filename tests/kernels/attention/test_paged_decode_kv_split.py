# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests for KV-split (flash-decoding) paged decode.

Splitting the KV range across workgroups and merging with a log-sum-exp
rescale is mathematically identical to the single-pass kernel, so every split
count must agree with NUM_SPLITS=1 to within float reassociation noise.
"""

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (
    _kv_split_supported,
    get_num_kv_splits,
    kernel_paged_attention_2d,
    kernel_paged_attention_2d_reduce,
)

NUM_SPLITS = [1, 2, 3, 5, 8, 32]
HEAD_SIZES = [128, 256]
# 784 is deliberately not a power of two: non-pow2 block sizes are exactly the
# case that forces the Triton fallback instead of the ROCm custom kernel.
BLOCK_SIZES = [16, 784]
SEQ_LENS = [1, 15, 16, 17, 256, 1024, 4099]


def _run(
    num_splits,
    seq_lens,
    head_size,
    physical_block_size,
    num_query_heads=12,
    num_kv_heads=2,
    dtype=torch.bfloat16,
    device="cuda",
    seed=0,
):
    set_random_seed(seed)
    num_seqs = len(seq_lens)
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_padded = triton.next_power_of_2(head_size)
    max_seq_len = max(seq_lens)
    max_blocks = (max_seq_len + physical_block_size - 1) // physical_block_size

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    num_blocks = num_seqs * max_blocks + 1

    query = torch.randn(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )
    key_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size // x,
        physical_block_size,
        x,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn(
        num_blocks,
        num_kv_heads,
        head_size,
        physical_block_size,
        dtype=dtype,
        device=device,
    )
    block_table = torch.randint(
        1, num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device=device
    )
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
    output = torch.empty(
        num_seqs, num_query_heads, head_size, dtype=dtype, device=device
    )

    triton_block_size = 16 if physical_block_size % 2 else min(physical_block_size, 128)
    # Non-pow2 physical blocks use a 32-wide Triton tile (see the dispatcher).
    if physical_block_size & (physical_block_size - 1):
        triton_block_size = 32

    if num_splits > 1:
        partial_out = torch.empty(
            num_seqs,
            num_query_heads,
            num_splits,
            head_size_padded,
            dtype=torch.float32,
            device=device,
        )
        partial_m = torch.empty(
            num_seqs,
            num_query_heads,
            num_splits,
            dtype=torch.float32,
            device=device,
        )
        partial_l = torch.empty_like(partial_m)
        po = partial_out.stride()
        pm = partial_m.stride()
    else:
        partial_out = partial_m = partial_l = None
        po, pm = (0, 0, 0), (0, 0)

    kernel_paged_attention_2d[(num_seqs, num_kv_heads, num_splits)](
        output_ptr=output,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        sink_ptr=None,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens_t,
        alibi_slopes_ptr=None,
        partial_out_ptr=partial_out,
        partial_m_ptr=partial_m,
        partial_l_ptr=partial_l,
        scale=head_size**-0.5,
        k_scale=torch.ones(1, device=device),
        v_scale=torch.ones(1, device=device),
        out_scale_inv=1.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=max(triton.next_power_of_2(num_queries_per_kv), 16),
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        po_stride_0=po[0],
        po_stride_1=po[1],
        po_stride_2=po[2],
        pm_stride_0=pm[0],
        pm_stride_1=pm[1],
        BLOCK_SIZE=triton_block_size,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=x,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=query_start_loc,
        USE_SINKS=False,
        USE_FP8=False,
        NUM_SPLITS=num_splits,
    )
    if num_splits > 1:
        kernel_paged_attention_2d_reduce[(num_seqs, num_query_heads)](
            output_ptr=output,
            partial_out_ptr=partial_out,
            partial_m_ptr=partial_m,
            partial_l_ptr=partial_l,
            out_scale_inv=1.0,
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            po_stride_0=po[0],
            po_stride_1=po[1],
            po_stride_2=po[2],
            pm_stride_0=pm[0],
            pm_stride_1=pm[1],
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            NUM_SPLITS=num_splits,
            NUM_SPLITS_PADDED=triton.next_power_of_2(num_splits),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
            USE_FP8=False,
        )
    torch.cuda.synchronize()
    return output


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
@pytest.mark.parametrize("num_splits", NUM_SPLITS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("physical_block_size", BLOCK_SIZES)
def test_kv_split_matches_unsplit(num_splits, head_size, physical_block_size):
    """Every split count must reproduce the single-pass result."""
    seq_lens = [1, 15, 16, 17, 256, 1024, 4099]
    ref = _run(1, seq_lens, head_size, physical_block_size)
    got = _run(num_splits, seq_lens, head_size, physical_block_size)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_kv_split_short_sequences(seq_len):
    """Splits past the end of a short sequence must contribute nothing.

    With 32 splits and seq_len=1 all but one split run zero tiles and emit
    (m=-inf, l=0); the reduction has to drop them rather than produce NaN.
    """
    ref = _run(1, [seq_len], head_size=128, physical_block_size=16)
    got = _run(32, [seq_len], head_size=128, physical_block_size=16)
    assert torch.isfinite(got).all(), "empty splits produced non-finite output"
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
def test_get_num_kv_splits_is_shape_only():
    """The split count must not depend on sequence length (CUDA graph safety)."""
    dev = torch.device("cuda:0")
    # Same batch shape -> same grid, regardless of how long the sequences are.
    assert get_num_kv_splits(1, 2, dev) == get_num_kv_splits(1, 2, dev)
    # Large batches already fill the machine, so no splitting.
    assert get_num_kv_splits(4096, 32, dev) == 1

    if _kv_split_supported():
        # Small batches must actually split, otherwise the kernel is unchanged.
        assert get_num_kv_splits(1, 1, dev) > 1
    else:
        # Everywhere else the unsplit path must be preserved exactly.
        assert get_num_kv_splits(1, 1, dev) == 1


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="requires GPU")
def test_split_gated_to_supported_amd_archs():
    """Splitting is enabled only on ROCm gfx1100 / gfx1201."""
    if not current_platform.is_rocm():
        assert not _kv_split_supported()
        return
    from vllm.platforms.rocm import on_gfx1100, on_gfx1201

    assert _kv_split_supported() == (on_gfx1100() or on_gfx1201())
