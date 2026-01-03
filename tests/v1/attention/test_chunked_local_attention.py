# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.attention.layers.chunked_local_attention import (
    create_chunked_local_attention_backend,
)
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)


def create_mock_underlying_backend(device: torch.device):
    """Create a mock underlying attention backend for testing."""

    class MockMetadata:
        """Minimal metadata that captures what was passed to build()."""

        pass

    class MockUnderlyingBuilder(AttentionMetadataBuilder[MockMetadata]):
        def __init__(
            self,
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
        ):
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device
            # Capture what was passed to build for verification
            self.last_common_attn_metadata = None

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> MockMetadata:
            # Capture the metadata for test verification
            self.last_common_attn_metadata = common_attn_metadata
            return MockMetadata()

        def update_block_table(self, metadata, blk_table, slot_mapping):
            return metadata

    class MockUnderlyingBackend:
        @classmethod
        def get_builder_cls(cls):
            return MockUnderlyingBuilder

    return MockUnderlyingBackend


def build_chunked_local_attention(
    batch_spec: BatchSpec,
    attn_chunk_size: int,
    block_size: int,
    device: torch.device,
    arange_block_indices: bool = True,
) -> CommonAttentionMetadata:
    """Build chunked local attention metadata using the real builder."""
    # Create the backend
    mock_backend = create_mock_underlying_backend(device)
    chunked_backend = create_chunked_local_attention_backend(
        mock_backend, attn_chunk_size, block_size
    )

    # Create mock kv_cache_spec
    mock_kv_cache_spec = MagicMock()
    mock_kv_cache_spec.block_size = block_size

    # Create vllm_config with enough capacity
    vllm_config = create_vllm_config(
        max_num_seqs=len(batch_spec.query_lens),
        max_num_batched_tokens=max(
            sum(batch_spec.query_lens), len(batch_spec.query_lens)
        ),
        block_size=block_size,
    )

    # Create the builder
    builder_cls = chunked_backend.get_builder_cls()
    builder = builder_cls(
        kv_cache_spec=mock_kv_cache_spec,
        layer_names=["layer0"],
        vllm_config=vllm_config,
        device=device,
    )

    # Create common attention metadata
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size,
        device,
        arange_block_indices=arange_block_indices,
    )

    # Build and return the result
    builder.build(0, common_attn_metadata)

    # The underlying builder's last_common_attn_metadata has the virtual batches
    return builder.last_common_attn_metadata


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
    device = torch.device("cuda:0")
    batch_spec = test_data.batch_spec
    attn_chunk_size = test_data.attn_chunk_size
    block_size = test_data.block_size
    expected_q_seqlens = test_data.expected_q_seqlens
    expected_k_seqlens = test_data.expected_k_seqlens
    expected_local_block_table = test_data.expected_local_block_table

    # Call the builder
    result = build_chunked_local_attention(
        batch_spec,
        attn_chunk_size,
        block_size,
        device,
        arange_block_indices=True,
    )

    # Get actual count (trim padding - find first zero in k_seqlens)
    actual_count = len(expected_k_seqlens)

    # Convert to numpy for comparison (use GPU tensors, then transfer to CPU)
    actual_q_seqlens = np.diff(result.query_start_loc.cpu().numpy())[:actual_count]
    actual_k_seqlens = result.seq_lens.cpu().numpy()[:actual_count]

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
    print(f"Actual block table:\n{result.block_table_tensor[:actual_count]}")

    torch.testing.assert_close(
        result.block_table_tensor[:actual_count], expected_block_table_tensor
    )
