# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PCP Virtual Request Manager with DualChunkSwap."""

import numpy as np
import pytest
import torch

from vllm.v1.worker.pcp_virtual_request import PCPVirtualRequestManager


class TestPCPVirtualRequestPartition:
    """Tests for PCPVirtualRequestManager.partition() with DualChunkSwap."""

    def setup_method(self):
        self.device = torch.device("cpu")

    def test_even_token_distribution_dualchunkswap(self):
        """Test DualChunkSwap with 8 tokens (plan example).

        For 8 tokens with PCP=2:
          - 4 chunks of size 2 each: [0,1], [2,3], [4,5], [6,7]
          - Rank 0: chunk 0 [0,1] + chunk 3 [6,7] = positions [0,1,6,7]
          - Rank 1: chunk 1 [2,3] + chunk 2 [4,5] = positions [2,3,4,5]
        """
        manager_r0 = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=0,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )
        manager_r1 = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=1,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )

        num_scheduled = np.array([8], dtype=np.int32)
        num_computed = np.array([0], dtype=np.int32)

        v_sched_r0, v_req_idx_r0, v_pos_r0, cu_r0 = manager_r0.partition(
            num_scheduled, num_computed
        )
        v_sched_r1, v_req_idx_r1, v_pos_r1, cu_r1 = manager_r1.partition(
            num_scheduled, num_computed
        )

        # Each rank gets 4 tokens
        assert list(v_sched_r0) == [4]
        assert list(v_sched_r1) == [4]

        # Verify DualChunkSwap positions
        assert sorted(int(p) for p in v_pos_r0) == [0, 1, 6, 7]
        assert sorted(int(p) for p in v_pos_r1) == [2, 3, 4, 5]

        # All positions covered
        all_positions = list(v_pos_r0) + list(v_pos_r1)
        assert sorted(int(p) for p in all_positions) == list(range(8))

    def test_ten_tokens_distribution(self):
        """Test DualChunkSwap with 10 tokens.

        For 10 tokens with PCP=2:
          - 4 chunks with sizes [3, 3, 2, 2] (first 2 chunks get +1)
          - Rank 0: chunk 0 [0,1,2] + chunk 3 [8,9] = 5 tokens
          - Rank 1: chunk 1 [3,4,5] + chunk 2 [6,7] = 5 tokens
        """
        managers = [
            PCPVirtualRequestManager(
                pcp_world_size=2,
                pcp_rank=r,
                max_num_reqs=10,
                max_num_batched_tokens=1000,
                device=self.device,
            )
            for r in range(2)
        ]

        num_scheduled = np.array([10], dtype=np.int32)
        num_computed = np.array([0], dtype=np.int32)

        results = [m.partition(num_scheduled, num_computed) for m in managers]

        # Verify token counts
        assert results[0][0][0] == 5  # rank 0
        assert results[1][0][0] == 5  # rank 1

        # Verify positions
        assert sorted(int(p) for p in results[0][2]) == [0, 1, 2, 8, 9]
        assert sorted(int(p) for p in results[1][2]) == [3, 4, 5, 6, 7]

        # All positions covered
        all_positions = list(results[0][2]) + list(results[1][2])
        assert sorted(int(p) for p in all_positions) == list(range(10))

    def test_odd_token_distribution(self):
        """Test DualChunkSwap with 7 tokens.

        For 7 tokens with PCP=2:
          - 4 chunks with sizes [2, 2, 2, 1] (first 3 chunks get +1)
          - Rank 0: chunk 0 [0,1] + chunk 3 [6] = 3 tokens
          - Rank 1: chunk 1 [2,3] + chunk 2 [4,5] = 4 tokens
        """
        managers = [
            PCPVirtualRequestManager(
                pcp_world_size=2,
                pcp_rank=r,
                max_num_reqs=10,
                max_num_batched_tokens=1000,
                device=self.device,
            )
            for r in range(2)
        ]

        num_scheduled = np.array([7], dtype=np.int32)
        num_computed = np.array([0], dtype=np.int32)

        results = [m.partition(num_scheduled, num_computed) for m in managers]

        # Verify totals add up
        total = sum(r[0][0] for r in results)
        assert total == 7

        # All positions covered
        all_positions = list(results[0][2]) + list(results[1][2])
        assert sorted(int(p) for p in all_positions) == list(range(7))

    def test_single_token_decode(self):
        """Test decode scenario with 1 token per request."""
        manager_r0 = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=0,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )
        manager_r1 = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=1,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )

        # 1 token per request -> rank0 gets all (chunk 0 = 1 token, chunk 3 = 0)
        num_scheduled = np.array([1, 1], dtype=np.int32)
        num_computed = np.array([100, 50], dtype=np.int32)

        v_sched_r0, _, v_pos_r0, _ = manager_r0.partition(num_scheduled, num_computed)
        v_sched_r1, _, v_pos_r1, _ = manager_r1.partition(num_scheduled, num_computed)

        # Rank 0 gets all single tokens
        assert list(v_sched_r0) == [1, 1]
        assert list(v_sched_r1) == [0, 0]

        # Positions should account for num_computed
        assert list(int(p) for p in v_pos_r0) == [100, 50]
        assert len(v_pos_r1) == 0

    def test_with_num_computed_tokens(self):
        """Test positions account for already computed tokens."""
        managers = [
            PCPVirtualRequestManager(
                pcp_world_size=2,
                pcp_rank=r,
                max_num_reqs=10,
                max_num_batched_tokens=1000,
                device=self.device,
            )
            for r in range(2)
        ]

        num_scheduled = np.array([8], dtype=np.int32)
        num_computed = np.array([100], dtype=np.int32)

        results = [m.partition(num_scheduled, num_computed) for m in managers]

        # Positions should be offset by num_computed
        # Rank 0: [100, 101, 106, 107]
        # Rank 1: [102, 103, 104, 105]
        assert sorted(int(p) for p in results[0][2]) == [100, 101, 106, 107]
        assert sorted(int(p) for p in results[1][2]) == [102, 103, 104, 105]

    def test_pcp_world_size_4(self):
        """Test DualChunkSwap with PCP world size of 4."""
        managers = [
            PCPVirtualRequestManager(
                pcp_world_size=4,
                pcp_rank=r,
                max_num_reqs=10,
                max_num_batched_tokens=1000,
                device=self.device,
            )
            for r in range(4)
        ]

        # 16 tokens with PCP=4:
        # 8 chunks of size 2 each
        # Rank 0: chunk 0 + chunk 7 = [0,1,14,15]
        # Rank 1: chunk 1 + chunk 6 = [2,3,12,13]
        # Rank 2: chunk 2 + chunk 5 = [4,5,10,11]
        # Rank 3: chunk 3 + chunk 4 = [6,7,8,9]
        num_scheduled = np.array([16], dtype=np.int32)
        num_computed = np.array([0], dtype=np.int32)

        results = [m.partition(num_scheduled, num_computed) for m in managers]

        # Each rank gets 4 tokens
        for rank_idx in range(4):
            assert results[rank_idx][0][0] == 4

        # Verify totals
        total = sum(res[0][0] for res in results)
        assert total == 16

        # Verify positions cover [0, 16) without overlap
        all_positions: list[int] = []
        for res in results:
            all_positions.extend(list(int(p) for p in res[2]))
        assert sorted(all_positions) == list(range(16))


class TestPCPRestoreIndices:
    """Tests for PCP restore index computation."""

    def setup_method(self):
        self.device = torch.device("cpu")

    def test_restore_indices_structure(self):
        """Test that restore indices are computed correctly."""
        manager = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=0,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )

        num_scheduled = np.array([4, 2], dtype=np.int32)
        num_computed = np.array([0, 0], dtype=np.int32)

        manager.partition(num_scheduled, num_computed)

        # After partition, restore indices should be set
        assert manager._allgather_restore_idx is not None
        assert manager._global_num_tokens == 6  # 4 + 2 total

        # Restore indices should have global_num_tokens elements
        assert len(manager._allgather_restore_idx) == 6

    def test_kv_restore_indices(self):
        """Test that KV restore indices are computed correctly."""
        manager = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=0,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )

        num_scheduled = np.array([8], dtype=np.int32)
        num_computed = np.array([0], dtype=np.int32)

        manager.partition(num_scheduled, num_computed)

        # KV restore indices should be set
        kv_restore_idx = manager.get_kv_restore_idx()
        assert kv_restore_idx is not None
        assert len(kv_restore_idx) == 8  # total tokens

        # All indices should be valid (0 to 7)
        assert kv_restore_idx.min() >= 0
        assert kv_restore_idx.max() < 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
