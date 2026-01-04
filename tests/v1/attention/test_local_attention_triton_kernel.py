# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Triton kernel implementation of chunked local attention.

Tests focus on:
1. Edge cases (single batch, single token, large batches)
2. Various chunk sizes and block sizes
3. Consistency between seqlens_q, seqlens_k, and cu_seqlens
4. Equivalence with the original numpy implementation
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.attention.layers.chunked_local_attention import (
    create_chunked_local_attention_backend,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)


def _make_local_attention_virtual_batches_reference(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Reference implementation using numpy (the original algorithm).
    Returns: (cu_seqlens_q_local, seqlens_q_local, seqlens_k_local, block_table_local)
    """
    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # q_tokens_in_first_block
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Batched arange
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1

    # seqlens_q_local
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # cu_seqlens_q_local
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # seqlens_k_local
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    # Block table
    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    block_starts = k_seqstarts_absolute // block_size
    pages_per_local_batch = attn_chunk_size // block_size

    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)
    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
        virtual_batches, -1
    )

    return cu_seqlens_q_local, seqlens_q_local, seqlens_k_local, block_table_local


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


def create_test_metadata(
    query_lens: list[int],
    seq_lens: list[int],
    block_size: int,
    device: torch.device,
    arange_block_indices: bool = True,
) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata for testing."""
    batch_size = len(query_lens)
    max_seq_len = max(seq_lens)
    max_blocks = cdiv(max_seq_len, block_size)

    # Create cumulative query_start_loc
    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, q in enumerate(query_lens):
        query_start_loc[i + 1] = query_start_loc[i] + q

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    # Create block table
    if arange_block_indices:
        block_table = torch.arange(
            batch_size * max_blocks, dtype=torch.int32, device=device
        ).view(batch_size, max_blocks)
    else:
        block_table = torch.randint(
            0, 1000, (batch_size, max_blocks), dtype=torch.int32, device=device
        )

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens_t,
        num_reqs=batch_size,
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table,
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64, device=device),
    )


def build_virtual_batches(
    query_lens: list[int],
    seq_lens: list[int],
    attn_chunk_size: int,
    block_size: int,
    device: torch.device,
    arange_block_indices: bool = True,
) -> CommonAttentionMetadata:
    """Build chunked local attention metadata using the real builder."""
    meta = create_test_metadata(
        query_lens=query_lens,
        seq_lens=seq_lens,
        block_size=block_size,
        device=device,
        arange_block_indices=arange_block_indices,
    )

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
        max_num_seqs=len(query_lens),
        max_num_batched_tokens=max(sum(query_lens), len(query_lens)),
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

    # Build and return the result
    builder.build(0, meta)

    return builder.last_common_attn_metadata, meta


def get_actual_seqlens(result: CommonAttentionMetadata) -> tuple[list[int], list[int]]:
    """Extract actual (non-padded) seqlens from result."""
    q_seqlens = (result.query_start_loc[1:] - result.query_start_loc[:-1]).cpu()
    k_seqlens = result.seq_lens.cpu()
    # Find actual count (first zero or end)
    nonzero_mask = k_seqlens > 0
    if nonzero_mask.all():
        actual_count = len(k_seqlens)
    else:
        actual_count = int(nonzero_mask.int().argmin())
        if actual_count == 0 and k_seqlens[0] > 0:
            actual_count = len(k_seqlens)
    return q_seqlens[:actual_count].tolist(), k_seqlens[:actual_count].tolist()


class TestLocalAttentionKernelBasic:
    """Basic correctness tests."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def test_single_batch_single_chunk(self, device):
        """Single batch that fits in one chunk."""
        result, _ = build_virtual_batches(
            query_lens=[3], seq_lens=[3], attn_chunk_size=4, block_size=2, device=device
        )

        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == [3]
        assert k_seqlens == [3]

    def test_single_batch_multiple_chunks(self, device):
        """Single batch spanning multiple chunks."""
        result, _ = build_virtual_batches(
            query_lens=[10],
            seq_lens=[10],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        # 10 tokens with chunk_size=4: chunks at [0-4), [4-8), [8-10)
        # -> 3 virtual batches
        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == [4, 4, 2]
        assert k_seqlens == [4, 4, 2]

    def test_multiple_batches_uniform(self, device):
        """Multiple batches with uniform sizes."""
        result, _ = build_virtual_batches(
            query_lens=[4, 4, 4],
            seq_lens=[4, 4, 4],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        # Each batch produces 1 virtual batch
        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == [4, 4, 4]
        assert k_seqlens == [4, 4, 4]

    def test_docstring_example(self, device):
        """Test the example from the docstring."""
        result, _ = build_virtual_batches(
            query_lens=[4, 10, 5],
            seq_lens=[6, 17, 9],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        expected_q_seqlens = [2, 2, 1, 4, 4, 1, 4, 1]
        expected_k_seqlens = [4, 2, 4, 4, 4, 1, 4, 1]

        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == expected_q_seqlens
        assert k_seqlens == expected_k_seqlens


class TestLocalAttentionKernelEdgeCases:
    """Edge case tests."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def test_single_token_per_batch(self, device):
        """Each batch has only one token."""
        result, _ = build_virtual_batches(
            query_lens=[1, 1, 1],
            seq_lens=[1, 1, 1],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == [1, 1, 1]
        assert k_seqlens == [1, 1, 1]

    def test_chunk_size_larger_than_seq(self, device):
        """Chunk size larger than sequence length."""
        result, _ = build_virtual_batches(
            query_lens=[3],
            seq_lens=[5],
            attn_chunk_size=10,
            block_size=2,
            device=device,
        )

        # Everything fits in one chunk
        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert k_seqlens == [5]

    def test_chunk_equals_block_size(self, device):
        """Chunk size equals block size."""
        result, _ = build_virtual_batches(
            query_lens=[8], seq_lens=[8], attn_chunk_size=4, block_size=4, device=device
        )

        # 8 tokens with chunk=block=4: 2 virtual batches
        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == [4, 4]

    def test_prefill_with_context(self, device):
        """Query starts in the middle of a chunk (has context)."""
        # seq_lens=5, query_lens=1 -> 4 context tokens
        # With chunk_size=4, query starts in second chunk
        result, _ = build_virtual_batches(
            query_lens=[1], seq_lens=[5], attn_chunk_size=4, block_size=2, device=device
        )

        # Query is in second chunk [4-5), so 1 virtual batch
        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert q_seqlens == [1]
        assert k_seqlens == [1]


class TestLocalAttentionKernelLargeBatches:
    """Tests with larger batch sizes to stress binary search."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def test_large_batch_count(self, device):
        """Many small batches."""
        batch_size = 100
        result, _ = build_virtual_batches(
            query_lens=[4] * batch_size,
            seq_lens=[4] * batch_size,
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        # Each batch produces 1 virtual batch
        q_seqlens, k_seqlens = get_actual_seqlens(result)
        assert len(q_seqlens) == batch_size

    def test_large_batch_varying_sizes(self, device):
        """Many batches with varying sizes."""
        batch_size = 50
        query_lens = [(i % 10) + 1 for i in range(batch_size)]
        seq_lens = [(i % 10) + 5 for i in range(batch_size)]

        result, _ = build_virtual_batches(
            query_lens=query_lens,
            seq_lens=seq_lens,
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        # Verify total query tokens preserved
        q_seqlens, _ = get_actual_seqlens(result)
        assert sum(q_seqlens) == sum(query_lens)


class TestLocalAttentionKernelBlockTable:
    """Tests for block table correctness."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def test_block_table_values(self, device):
        """Verify block table values are correct."""
        result, _ = build_virtual_batches(
            query_lens=[4, 10, 5],
            seq_lens=[6, 17, 9],
            attn_chunk_size=4,
            block_size=2,
            device=device,
            arange_block_indices=True,
        )

        # Expected block table from docstring
        expected = [
            [0, 1],  # batch 0, k[0-4)
            [2, 3],  # batch 0, k[4-6)
            [11, 12],  # batch 1, k[4-8)
            [13, 14],  # batch 1, k[8-12)
            [15, 16],  # batch 1, k[12-16)
            [17, 17],  # batch 1, k[16-17) - clipped
            [20, 21],  # batch 2, k[4-8)
            [22, 23],  # batch 2, k[8-9)
        ]
        expected_tensor = torch.tensor(expected, dtype=torch.int32, device=device)
        # Compare only actual (non-padded) entries
        _, k_seqlens = get_actual_seqlens(result)
        actual_count = len(k_seqlens)
        torch.testing.assert_close(
            result.block_table_tensor[:actual_count], expected_tensor
        )

    def test_block_table_shape(self, device):
        """Verify block table has correct shape for actual entries."""
        result, _ = build_virtual_batches(
            query_lens=[8],
            seq_lens=[12],
            attn_chunk_size=4,
            block_size=4,
            device=device,
        )

        # pages_per_local_batch = 4 / 4 = 1
        # 2 actual virtual batches (may have padding)
        _, k_seqlens = get_actual_seqlens(result)
        actual_count = len(k_seqlens)
        assert actual_count == 2
        assert result.block_table_tensor.shape[1] == 1  # pages_per_local_batch


class TestLocalAttentionKernelInvariants:
    """Tests for mathematical invariants."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    @pytest.mark.parametrize("chunk_size", [4, 8, 16])
    @pytest.mark.parametrize("block_size", [2, 4])
    def test_seqlens_invariants(self, device, chunk_size, block_size):
        """Verify seqlen invariants hold."""
        if chunk_size % block_size != 0:
            pytest.skip("chunk_size must be divisible by block_size")

        result, _ = build_virtual_batches(
            query_lens=[7, 15, 3],
            seq_lens=[10, 20, 8],
            attn_chunk_size=chunk_size,
            block_size=block_size,
            device=device,
        )

        q_seqlens, k_seqlens = get_actual_seqlens(result)

        # All q_seqlens <= chunk_size
        assert all(q <= chunk_size for q in q_seqlens)

        # All k_seqlens <= chunk_size
        assert all(k <= chunk_size for k in k_seqlens)

        # Total q tokens preserved
        assert sum(q_seqlens) == 7 + 15 + 3

    def test_cumsum_consistency(self, device):
        """Verify cu_seqlens is consistent with seqlens."""
        result, _ = build_virtual_batches(
            query_lens=[5, 12, 7],
            seq_lens=[8, 15, 10],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

        q_seqlens_full = result.query_start_loc[1:] - result.query_start_loc[:-1]

        # Recompute cumsum and verify
        expected_cu = torch.zeros(
            result.num_reqs + 1, dtype=torch.int32, device=result.query_start_loc.device
        )
        torch.cumsum(q_seqlens_full, dim=0, out=expected_cu[1:])

        torch.testing.assert_close(result.query_start_loc, expected_cu)


class TestLocalAttentionVsReference:
    """Tests comparing Triton implementation against original numpy reference."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda:0")

    def _run_comparison(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        attn_chunk_size: int,
        block_size: int,
        device: torch.device,
    ):
        """Run both implementations and compare results."""
        result, meta = build_virtual_batches(
            query_lens=query_lens,
            seq_lens=seq_lens,
            attn_chunk_size=attn_chunk_size,
            block_size=block_size,
            device=device,
            arange_block_indices=True,
        )

        # Run reference implementation
        query_start_loc_np = meta.query_start_loc.cpu().numpy()
        seq_lens_np = meta.seq_lens.cpu().numpy()
        ref_cu_seqlens_q, ref_seqlens_q, ref_seqlens_k, ref_block_table = (
            _make_local_attention_virtual_batches_reference(
                attn_chunk_size,
                query_start_loc_np,
                seq_lens_np,
                meta.block_table_tensor,
                block_size,
            )
        )

        # Get actual count (non-padded entries)
        actual_count = len(ref_seqlens_q)

        # Compare results (trim padding)
        actual_seqlens_q = (
            (result.query_start_loc[1:] - result.query_start_loc[:-1])
            .cpu()
            .numpy()[:actual_count]
        )
        actual_seqlens_k = result.seq_lens.cpu().numpy()[:actual_count]
        actual_cu_seqlens_q = result.query_start_loc.cpu().numpy()[: actual_count + 1]
        actual_block_table = result.block_table_tensor[:actual_count]

        np.testing.assert_array_equal(
            actual_seqlens_q, ref_seqlens_q, err_msg="seqlens_q mismatch"
        )
        np.testing.assert_array_equal(
            actual_seqlens_k, ref_seqlens_k, err_msg="seqlens_k mismatch"
        )
        np.testing.assert_array_equal(
            actual_cu_seqlens_q, ref_cu_seqlens_q, err_msg="cu_seqlens_q mismatch"
        )
        torch.testing.assert_close(
            actual_block_table, ref_block_table, msg="block_table mismatch"
        )

    def test_docstring_example_vs_reference(self, device):
        """Test the docstring example against reference."""
        self._run_comparison(
            query_lens=[4, 10, 5],
            seq_lens=[6, 17, 9],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

    def test_single_batch_vs_reference(self, device):
        """Single batch comparison."""
        self._run_comparison(
            query_lens=[8],
            seq_lens=[12],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

    def test_many_batches_vs_reference(self, device):
        """Many batches comparison."""
        self._run_comparison(
            query_lens=[3, 7, 2, 9, 5, 1, 8, 4],
            seq_lens=[5, 10, 6, 15, 8, 3, 12, 7],
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

    def test_large_chunk_size_vs_reference(self, device):
        """Large chunk size comparison."""
        self._run_comparison(
            query_lens=[5, 12, 8],
            seq_lens=[10, 20, 15],
            attn_chunk_size=16,
            block_size=4,
            device=device,
        )

    def test_chunk_equals_block_vs_reference(self, device):
        """Chunk size equals block size comparison."""
        self._run_comparison(
            query_lens=[8, 12],
            seq_lens=[8, 12],
            attn_chunk_size=4,
            block_size=4,
            device=device,
        )

    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_random_batches_vs_reference(self, device, batch_size):
        """Random batch configurations comparison."""
        np.random.seed(42 + batch_size)
        query_lens = np.random.randint(1, 20, size=batch_size).tolist()
        # seq_lens >= query_lens
        seq_lens = [q + np.random.randint(0, 10) for q in query_lens]

        self._run_comparison(
            query_lens=query_lens,
            seq_lens=seq_lens,
            attn_chunk_size=4,
            block_size=2,
            device=device,
        )

    @pytest.mark.parametrize(
        "chunk_size,block_size", [(4, 2), (8, 2), (8, 4), (16, 4), (16, 8), (32, 8)]
    )
    def test_various_sizes_vs_reference(self, device, chunk_size, block_size):
        """Various chunk and block size combinations."""
        self._run_comparison(
            query_lens=[7, 15, 3, 22, 9],
            seq_lens=[10, 25, 8, 30, 12],
            attn_chunk_size=chunk_size,
            block_size=block_size,
            device=device,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
