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

    def test_eight_tokens_dualchunkswap(self):
        """Test DualChunkSwap with 8 tokens (plan example).

        For 8 tokens with PCP=2, physical request becomes 2 virtual requests:
          Rank 0:
            - vreq 0 (head): num_scheduled=2, num_computed=0
            - vreq 1 (tail): num_scheduled=2, num_computed=6
          Rank 1:
            - vreq 0 (head): num_scheduled=2, num_computed=2
            - vreq 1 (tail): num_scheduled=2, num_computed=4
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

        v_sched_r0, v_computed_r0, v_to_phys_r0 = manager_r0.partition(
            num_scheduled, num_computed
        )
        v_sched_r1, v_computed_r1, v_to_phys_r1 = manager_r1.partition(
            num_scheduled, num_computed
        )

        # 1 physical request -> 2 virtual requests per rank
        assert len(v_sched_r0) == 2
        assert len(v_sched_r1) == 2

        # Rank 0: head=2 tokens at pos 0, tail=2 tokens at pos 6
        assert list(v_sched_r0) == [2, 2]
        assert list(v_computed_r0) == [0, 6]
        assert list(v_to_phys_r0) == [0, 0]

        # Rank 1: head=2 tokens at pos 2, tail=2 tokens at pos 4
        assert list(v_sched_r1) == [2, 2]
        assert list(v_computed_r1) == [2, 4]
        assert list(v_to_phys_r1) == [0, 0]

        # Verify positions cover [0, 8) when using standard formula
        # positions = num_computed + arange
        all_positions = set()
        for num_sched, num_comp in zip(v_sched_r0, v_computed_r0):
            for i in range(num_sched):
                all_positions.add(num_comp + i)
        for num_sched, num_comp in zip(v_sched_r1, v_computed_r1):
            for i in range(num_sched):
                all_positions.add(num_comp + i)
        assert sorted(all_positions) == list(range(8))

    def test_ten_tokens_distribution(self):
        """Test DualChunkSwap with 10 tokens.

        For 10 tokens with PCP=2:
          - 4 chunks with sizes [3, 3, 2, 2]
          - Rank 0: chunk 0 (pos 0-2) + chunk 3 (pos 8-9) = 5 tokens
          - Rank 1: chunk 1 (pos 3-5) + chunk 2 (pos 6-7) = 5 tokens
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

        # Verify total token counts
        total_tokens = sum(r[0].sum() for r in results)
        assert total_tokens == 10

        # Rank 0: head=3 (pos 0), tail=2 (pos 8)
        assert list(results[0][0]) == [3, 2]
        assert list(results[0][1]) == [0, 8]

        # Rank 1: head=3 (pos 3), tail=2 (pos 6)
        assert list(results[1][0]) == [3, 2]
        assert list(results[1][1]) == [3, 6]

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

        # 1 token per request -> rank 0 gets all (chunk 0 = 1, chunk 3 = 0)
        num_scheduled = np.array([1, 1], dtype=np.int32)
        num_computed = np.array([100, 50], dtype=np.int32)

        v_sched_r0, v_computed_r0, _ = manager_r0.partition(num_scheduled, num_computed)
        v_sched_r1, v_computed_r1, _ = manager_r1.partition(num_scheduled, num_computed)

        # 2 physical reqs -> 4 virtual reqs per rank
        assert len(v_sched_r0) == 4
        assert len(v_sched_r1) == 4

        # Rank 0 gets all tokens (head=1, tail=0 for each)
        assert list(v_sched_r0) == [1, 0, 1, 0]
        # Head positions: 100 and 50
        # Tail positions don't matter since num_scheduled=0, but they're computed
        assert v_computed_r0[0] == 100  # head of req 0
        assert v_computed_r0[2] == 50  # head of req 1

        # Rank 1 gets nothing
        assert list(v_sched_r1) == [0, 0, 0, 0]

        # Verify total tokens
        assert v_sched_r0.sum() + v_sched_r1.sum() == 2

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

        # num_computed should be offset by the base (100)
        # Rank 0: head at 100, tail at 106
        assert list(results[0][1]) == [100, 106]
        # Rank 1: head at 102, tail at 104
        assert list(results[1][1]) == [102, 104]

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

        # Each rank gets 4 tokens total (2 virtual reqs x 2 tokens each)
        for r in range(4):
            assert results[r][0].sum() == 4

        # Verify totals
        total = sum(r[0].sum() for r in results)
        assert total == 16

        # Verify positions cover [0, 16) without overlap
        all_positions: set[int] = set()
        for v_sched, v_computed, _ in results:
            for num_sched, num_comp in zip(v_sched, v_computed):
                for i in range(num_sched):
                    all_positions.add(int(num_comp + i))
        assert sorted(all_positions) == list(range(16))

    def test_virtual_to_physical_mapping(self):
        """Test that virtual_to_physical correctly maps to physical requests."""
        manager = PCPVirtualRequestManager(
            pcp_world_size=2,
            pcp_rank=0,
            max_num_reqs=10,
            max_num_batched_tokens=1000,
            device=self.device,
        )

        # 3 physical requests
        num_scheduled = np.array([8, 4, 6], dtype=np.int32)
        num_computed = np.array([0, 0, 0], dtype=np.int32)

        v_sched, v_computed, v_to_phys = manager.partition(num_scheduled, num_computed)

        # 3 physical reqs -> 6 virtual reqs
        assert len(v_to_phys) == 6

        # Each physical req maps to 2 consecutive virtual reqs
        assert list(v_to_phys) == [0, 0, 1, 1, 2, 2]


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
