# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    make_local_attention_virtual_batches,
    max_local_attention_virtual_batches,
)


@dataclass
class LocalAttentionTestData:
    # Input parameters
    batch_spec: BatchSpec
    attn_chunk_size: int
    block_size: int
    # Expected return values
    expected_q_seqlens: list[int]
    expected_k_seqlens: list[int]
    expected_local_block_table: list[list[int]]


DEVICE_TYPE = current_platform.device_type

test_data_list = [
    # Same as example in docstring of make_local_attention_virtual_batches
    # except block table has 9 columns instead of 10
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[4, 10, 5],
            seq_lens=[6, 17, 9],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[2, 2, 1, 4, 4, 1, 4, 1],
        expected_k_seqlens=[4, 2, 4, 4, 4, 1, 4, 1],
        # 2 pages per local branch
        # (chunk size 4 // block size 2)
        expected_local_block_table=[
            [0, 1],  # local-batch 0, (batch 0, starting from k[0])
            [2, 3],  # local-batch 1, (batch 0, starting from k[4])
            [11, 12],  # local-batch 2, (batch 1, starting from k[4])
            [13, 14],  # local-batch 3, (batch 1, starting from k[8])
            [15, 16],  # local-batch 4, (batch 1, starting from k[12])
            [17, 17],  # local-batch 5, (batch 1, starting from k[16])
            [20, 21],  # local-batch 6, (batch 2, starting from k[4])
            [22, 23],  # local-batch 7, (batch 2, starting from k[8])
        ],
    ),
    # Case where block indices are not clipped to block table ncols-1
    # because tokens_in_last_block == attn_chunk_size
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[8],
            seq_lens=[12],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[4, 4],
        expected_k_seqlens=[4, 4],
        expected_local_block_table=[
            [2, 3],
            [4, 5],
        ],
    ),
    # Case where all kv_seq positions are involved in attn
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[7],
            # 10 - 7 = 3 previously computed tokens
            seq_lens=[10],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[1, 4, 2],
        expected_k_seqlens=[4, 4, 2],
        expected_local_block_table=[
            [0, 1],
            [2, 3],
            [4, 4],
        ],
    ),
    # Case where attn_chunk_size > kv_seq_len
    # so no extra mini virtual batches are created
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[4],
            seq_lens=[6],
        ),
        # Larger than kv_seq_len
        attn_chunk_size=10,
        block_size=2,
        # No change to q_seqlens and k_seqlens
        expected_q_seqlens=[4],
        expected_k_seqlens=[6],
        # In this case, we only need a block-table like:
        #  block_table = [ [0, 1, 2] ] # 1 batch, 3 pages
        # But we need to pad it to 5 pages per local batch
        # because currently the pages_per_local_batch
        # is calculated as (attn_chunk_size // block_size)
        expected_local_block_table=[
            [0, 1, 2, 2, 2],
        ],
    ),
    # Block size equal to chunk size
    # Expect single page per batch in local batch table
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[6, 6],
            seq_lens=[8, 8],
        ),
        attn_chunk_size=4,
        block_size=4,
        expected_q_seqlens=[2, 4, 2, 4],
        expected_k_seqlens=[4, 4, 4, 4],
        # Initial block table = [
        #    [0, 1], < batch 0
        #    [2, 3], < batch 1
        # ]
        expected_local_block_table=[
            [0],  # local-batch 0, (batch 0, starting from k[0])
            [1],  # local-batch 1, (batch 0, starting from k[4])
            [2],  # local-batch 1, (batch 0, starting from k[0])
            [3],  # local-batch 1, (batch 0, starting from k[4])
        ],
    ),
    # Case where query falls in the second attention chunk
    #  k_toks >   0 1 2 3 4
    #  q_toks v  _____________
    #         0 | 1
    #         1 | 1 1
    #         2 | 1 1 1
    #         3 | 1 1 1 1
    #         4 |         1
    #  where tokens 0,1,2,3 have been pre-computed
    LocalAttentionTestData(
        batch_spec=BatchSpec(
            query_lens=[1],
            seq_lens=[5],
        ),
        attn_chunk_size=4,
        block_size=2,
        expected_q_seqlens=[1],
        expected_k_seqlens=[1],
        expected_local_block_table=[
            [2, 2],
        ],
    ),
]


@pytest.mark.parametrize("test_data", test_data_list)
def test_local_attention_virtual_batches(test_data: LocalAttentionTestData):
    device = torch.device(f"{DEVICE_TYPE}:0")
    batch_spec = test_data.batch_spec
    attn_chunk_size = test_data.attn_chunk_size
    block_size = test_data.block_size
    expected_q_seqlens = test_data.expected_q_seqlens
    expected_k_seqlens = test_data.expected_k_seqlens
    expected_local_block_table = test_data.expected_local_block_table

    # Create common attention metadata
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size,
        device,
        # Use torch.arange instead of torch.randint so we can assert on
        # block table tensor values. The block table will have shape
        # (num_batches, cdiv(max_seq_len, block_size)) and the values will be
        # arranged from 0 to cdiv(max_seq_len, block_size)-1
        arange_block_indices=True,
    )

    # Call the function
    result, _ = make_local_attention_virtual_batches(
        attn_chunk_size, common_attn_metadata, block_size
    )

    # Convert to numpy for easier comparison
    actual_q_seqlens = np.diff(result.query_start_loc_cpu.numpy())
    actual_k_seqlens = result.seq_lens_cpu.numpy()

    # Check that all query lengths are less than or equal to attn_chunk_size
    assert all(q_len <= attn_chunk_size for q_len in actual_q_seqlens)
    # Check that all key lengths are less than or equal to attn_chunk_size
    assert all(k_len <= attn_chunk_size for k_len in actual_k_seqlens)
    # Check that the total number of query tokens is preserved
    assert sum(actual_q_seqlens) == sum(batch_spec.query_lens)

    # Verify results
    np.testing.assert_array_equal(actual_q_seqlens, expected_q_seqlens)
    np.testing.assert_array_equal(actual_k_seqlens, expected_k_seqlens)

    expected_block_table_tensor = torch.tensor(
        expected_local_block_table, dtype=torch.int32, device=device
    )

    print(f"Expected block table:\n{expected_block_table_tensor}")
    print(f"Actual block table:\n{result.block_table_tensor}")

    torch.testing.assert_close(result.block_table_tensor, expected_block_table_tensor)


@pytest.mark.parametrize(
    "query_lens,seq_lens,attn_chunk_size,max_num_seqs,expected_num_reqs",
    [
        # A single prefill longer than the chunk size, with max_num_seqs=1:
        # the condition that overflowed FlashInfer's per-request buffers in
        # https://github.com/vllm-project/vllm/issues/49980.
        ([1000], [1000], 256, 1, 4),
        # Partially computed context, so the first local block is partial.
        ([1000], [1255], 256, 1, 5),
        # Several requests, each spanning several chunks.
        ([300, 700, 90], [300, 1400, 90], 128, 4, 10),
        # Chunk size larger than every sequence: no extra virtual batches.
        ([64, 64], [64, 64], 256, 2, 2),
        # Decodes: one virtual batch each, so the token-count cap binds exactly.
        ([1, 1, 1, 1], [16, 32, 48, 64], 16, 4, 4),
    ],
)
def test_max_local_attention_virtual_batches_bounds_num_reqs(
    query_lens: list[int],
    seq_lens: list[int],
    attn_chunk_size: int,
    max_num_seqs: int,
    expected_num_reqs: int,
):
    """The virtual batch count must never exceed the advertised upper bound.

    Attention backends preallocate per-request buffers from this bound, so an
    underestimate overflows or silently truncates them. `expected_num_reqs`
    pins the split itself, so a bound that collapsed back to `max_num_seqs`
    fails here rather than passing vacuously.
    """
    block_size = 16
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(query_lens=query_lens, seq_lens=seq_lens),
        block_size,
        torch.device("cpu"),
    )
    result, _ = make_local_attention_virtual_batches(
        attn_chunk_size, common_attn_metadata, block_size
    )
    assert result.num_reqs == expected_num_reqs

    bound = max_local_attention_virtual_batches(
        attn_chunk_size, max_num_seqs, sum(query_lens)
    )
    assert result.num_reqs <= bound

    # Total pages must also fit `bound * pages_per_virtual_batch`, which is how
    # the FlashInfer builder sizes `paged_kv_indices`.
    pages_per_virtual_batch = cdiv(attn_chunk_size, block_size)
    num_pages = sum(cdiv(int(k), block_size) for k in result.seq_lens)
    assert num_pages <= bound * pages_per_virtual_batch
