# SPDX-License-Identifier: Apache-2.0
"""Standalone tests for InputBatch.swap_states / apply_permutation logic.

These functions live in vllm/v1/worker/gpu_input_batch.py but depend on
torch / CUDA.  We extract the core state-management logic into a pure-numpy
FakeBatch so the tests run on any machine with only numpy + pytest.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pytest


# -----------------------------------------------------------------------
# FakeBatch: mirrors the fields that swap_states / apply_permutation touch
# -----------------------------------------------------------------------

class FakeBatch:
    """Lightweight replica of InputBatch's reorder-relevant state."""

    def __init__(self, req_ids: list[str], max_seq_len: int = 32):
        n = len(req_ids)
        self.num_reqs = n
        # lists
        self._req_ids: list[str | None] = list(req_ids) + [None] * 4
        self.req_output_token_ids: list[list[int]] = [
            [i * 10 + j for j in range(3)] for i in range(n)
        ] + [[] for _ in range(4)]
        self.spec_token_ids: list[list[int]] = [
            [i * 100] for i in range(n)
        ] + [[] for _ in range(4)]
        # dict
        self.req_id_to_index: dict[str, int] = {
            rid: idx for idx, rid in enumerate(req_ids)
        }
        # 1-D numpy arrays
        self.num_tokens = np.arange(100, 100 + n, dtype=np.int32)
        self.num_tokens_no_spec = np.arange(200, 200 + n, dtype=np.int32)
        self.num_prompt_tokens = np.arange(300, 300 + n, dtype=np.int32)
        self.num_computed_tokens_cpu = np.arange(400, 400 + n, dtype=np.int32)
        self.temperature_cpu = np.linspace(0.1, 1.0, n, dtype=np.float32)
        # 2-D numpy array (token_ids_cpu analogue)
        self.token_ids_cpu = np.arange(n * max_seq_len, dtype=np.int64).reshape(
            n, max_seq_len
        )
        # prompt embeds (sparse dict[int, np.ndarray])
        self.req_prompt_embeds: dict[int, np.ndarray] = {
            0: np.array([1.0, 2.0]),
        }

    # -- properties used by reorder_batch_to_split_cp_and_normal ----------
    @property
    def req_ids(self) -> list[str | None]:
        return self._req_ids

    # -- swap_states (mirrors gpu_input_batch.py:509-614) -----------------
    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = (
            self._req_ids[i2],
            self._req_ids[i1],
        )
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        self.spec_token_ids[i1], self.spec_token_ids[i2] = (
            self.spec_token_ids[i2],
            self.spec_token_ids[i1],
        )
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        self.num_tokens[i1], self.num_tokens[i2] = (
            self.num_tokens[i2],
            self.num_tokens[i1],
        )
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (
            self.num_computed_tokens_cpu[i2],
            self.num_computed_tokens_cpu[i1],
        )
        self.temperature_cpu[i1], self.temperature_cpu[i2] = (
            self.temperature_cpu[i2],
            self.temperature_cpu[i1],
        )
        # 2-D row swap (same pattern as the real code: copy to avoid aliasing)
        tmp = self.token_ids_cpu[i1].copy()
        self.token_ids_cpu[i1] = self.token_ids_cpu[i2]
        self.token_ids_cpu[i2] = tmp
        # prompt embeds
        e1 = self.req_prompt_embeds.get(i1)
        e2 = self.req_prompt_embeds.get(i2)
        if e1 is not None:
            self.req_prompt_embeds[i2] = e1
        else:
            self.req_prompt_embeds.pop(i2, None)
        if e2 is not None:
            self.req_prompt_embeds[i1] = e2
        else:
            self.req_prompt_embeds.pop(i1, None)

    # -- apply_permutation (mirrors gpu_input_batch.py:641-740) -----------
    def apply_permutation(self, permutation: list[int] | np.ndarray) -> bool:
        n = self.num_reqs
        perm_np = np.asarray(permutation, dtype=np.int32)
        if perm_np.shape != (n,):
            raise ValueError(f"length mismatch: expected {n}, got {perm_np.shape}")
        if not np.array_equal(np.sort(perm_np), np.arange(n, dtype=np.int32)):
            raise ValueError(f"invalid permutation: {perm_np.tolist()}")
        if np.array_equal(perm_np, np.arange(n, dtype=np.int32)):
            return False

        perm_list = perm_np.tolist()

        old_ids = self._req_ids[:n].copy()
        self._req_ids[:n] = [old_ids[j] for j in perm_list]

        old_out = self.req_output_token_ids[:n].copy()
        self.req_output_token_ids[:n] = [old_out[j] for j in perm_list]

        old_spec = self.spec_token_ids[:n].copy()
        self.spec_token_ids[:n] = [old_spec[j] for j in perm_list]

        self.req_id_to_index = {
            rid: idx
            for idx, rid in enumerate(self._req_ids[:n])
            if rid is not None
        }

        self.num_tokens[:n] = self.num_tokens[perm_np].copy()
        self.num_tokens_no_spec[:n] = self.num_tokens_no_spec[perm_np].copy()
        self.num_prompt_tokens[:n] = self.num_prompt_tokens[perm_np].copy()
        self.num_computed_tokens_cpu[:n] = self.num_computed_tokens_cpu[perm_np].copy()
        self.temperature_cpu[:n] = self.temperature_cpu[perm_np].copy()
        self.token_ids_cpu[:n] = self.token_ids_cpu[perm_np].copy()

        self.req_prompt_embeds = {
            new_idx: self.req_prompt_embeds[old_idx]
            for new_idx, old_idx in enumerate(perm_list)
            if old_idx in self.req_prompt_embeds
        }
        return True


# -----------------------------------------------------------------------
# Snapshot helper: capture state for comparison
# -----------------------------------------------------------------------

@dataclass
class BatchSnapshot:
    req_ids: list
    req_id_to_index: dict
    num_tokens: np.ndarray
    num_computed_tokens_cpu: np.ndarray
    token_ids_row0: np.ndarray
    req_output_token_ids: list
    req_prompt_embeds_keys: set


def _snap(b: FakeBatch) -> BatchSnapshot:
    n = b.num_reqs
    return BatchSnapshot(
        req_ids=list(b._req_ids[:n]),
        req_id_to_index=dict(b.req_id_to_index),
        num_tokens=b.num_tokens[:n].copy(),
        num_computed_tokens_cpu=b.num_computed_tokens_cpu[:n].copy(),
        token_ids_row0=b.token_ids_cpu[0].copy(),
        req_output_token_ids=[list(x) for x in b.req_output_token_ids[:n]],
        req_prompt_embeds_keys=set(b.req_prompt_embeds.keys()),
    )


def _assert_index_consistent(b: FakeBatch):
    """req_id_to_index must be the inverse of _req_ids."""
    for idx in range(b.num_reqs):
        rid = b._req_ids[idx]
        assert rid is not None
        assert b.req_id_to_index[rid] == idx, (
            f"req_id_to_index[{rid}] = {b.req_id_to_index[rid]}, expected {idx}"
        )
    assert len(b.req_id_to_index) == b.num_reqs


# =======================================================================
# swap_states tests
# =======================================================================

class TestSwapStates:

    def test_swap_two(self):
        b = FakeBatch(["a", "b", "c"])
        before = _snap(b)
        b.swap_states(0, 2)
        assert b._req_ids[:3] == ["c", "b", "a"]
        assert b.num_tokens[0] == before.num_tokens[2]
        assert b.num_tokens[2] == before.num_tokens[0]
        _assert_index_consistent(b)

    def test_swap_same_index_is_noop(self):
        b = FakeBatch(["a", "b"])
        before = _snap(b)
        b.swap_states(0, 0)
        after = _snap(b)
        assert before.req_ids == after.req_ids
        np.testing.assert_array_equal(before.num_tokens, after.num_tokens)
        _assert_index_consistent(b)

    def test_swap_is_own_inverse(self):
        b = FakeBatch(["a", "b", "c", "d"])
        before = _snap(b)
        b.swap_states(1, 3)
        b.swap_states(1, 3)
        after = _snap(b)
        assert before.req_ids == after.req_ids
        np.testing.assert_array_equal(before.num_tokens, after.num_tokens)
        np.testing.assert_array_equal(
            before.num_computed_tokens_cpu, after.num_computed_tokens_cpu
        )
        _assert_index_consistent(b)

    def test_swap_preserves_2d_rows(self):
        b = FakeBatch(["a", "b", "c"])
        row0_before = b.token_ids_cpu[0].copy()
        row2_before = b.token_ids_cpu[2].copy()
        b.swap_states(0, 2)
        np.testing.assert_array_equal(b.token_ids_cpu[0], row2_before)
        np.testing.assert_array_equal(b.token_ids_cpu[2], row0_before)

    def test_swap_prompt_embeds(self):
        b = FakeBatch(["a", "b", "c"])
        # Initially only index 0 has embeds
        assert 0 in b.req_prompt_embeds
        assert 1 not in b.req_prompt_embeds
        b.swap_states(0, 1)
        # Now index 1 should have the embeds, index 0 should not
        assert 1 in b.req_prompt_embeds
        assert 0 not in b.req_prompt_embeds

    def test_swap_output_token_ids(self):
        b = FakeBatch(["a", "b", "c"])
        out0 = list(b.req_output_token_ids[0])
        out2 = list(b.req_output_token_ids[2])
        b.swap_states(0, 2)
        assert b.req_output_token_ids[0] == out2
        assert b.req_output_token_ids[2] == out0


# =======================================================================
# apply_permutation tests
# =======================================================================

class TestApplyPermutation:

    def test_identity_returns_false(self):
        b = FakeBatch(["a", "b", "c"])
        assert b.apply_permutation([0, 1, 2]) is False

    def test_single_element_returns_false(self):
        b = FakeBatch(["a"])
        assert b.apply_permutation([0]) is False

    def test_reverse(self):
        b = FakeBatch(["a", "b", "c"])
        before = _snap(b)
        b.apply_permutation([2, 1, 0])
        assert b._req_ids[:3] == ["c", "b", "a"]
        assert b.num_tokens[0] == before.num_tokens[2]
        assert b.num_tokens[2] == before.num_tokens[0]
        _assert_index_consistent(b)

    def test_cycle(self):
        b = FakeBatch(["a", "b", "c", "d"])
        # rotate left: [b, c, d, a]
        b.apply_permutation([1, 2, 3, 0])
        assert b._req_ids[:4] == ["b", "c", "d", "a"]
        _assert_index_consistent(b)

    def test_preserves_2d_rows(self):
        b = FakeBatch(["a", "b", "c"])
        rows_before = b.token_ids_cpu[:3].copy()
        b.apply_permutation([2, 0, 1])
        np.testing.assert_array_equal(b.token_ids_cpu[0], rows_before[2])
        np.testing.assert_array_equal(b.token_ids_cpu[1], rows_before[0])
        np.testing.assert_array_equal(b.token_ids_cpu[2], rows_before[1])

    def test_prompt_embeds_reindexed(self):
        b = FakeBatch(["a", "b", "c"])
        assert 0 in b.req_prompt_embeds
        b.apply_permutation([1, 2, 0])
        # old index 0 is now at new index 2
        assert 2 in b.req_prompt_embeds
        assert 0 not in b.req_prompt_embeds

    def test_invalid_permutation_raises(self):
        b = FakeBatch(["a", "b", "c"])
        with pytest.raises(ValueError, match="invalid permutation"):
            b.apply_permutation([0, 0, 1])

    def test_length_mismatch_raises(self):
        b = FakeBatch(["a", "b", "c"])
        with pytest.raises(ValueError, match="length mismatch"):
            b.apply_permutation([0, 1])


# =======================================================================
# swap_states and apply_permutation must agree
# =======================================================================

class TestSwapPermutationEquivalence:
    """Apply the same logical reordering via both paths, compare results."""

    @pytest.mark.parametrize(
        "perm",
        [
            [1, 0, 2, 3],      # single swap
            [3, 2, 1, 0],      # full reverse
            [1, 2, 3, 0],      # cycle
            [0, 3, 1, 2],      # complex
            [2, 0, 3, 1],      # complex
        ],
        ids=["swap01", "reverse", "cycle", "complex1", "complex2"],
    )
    def test_equivalence(self, perm: list[int]):
        ids = ["a", "b", "c", "d"]

        # Path 1: apply_permutation
        b_perm = FakeBatch(ids)
        b_perm.apply_permutation(perm)

        # Path 2: swap_states via the same algorithm as
        # reorder_batch_to_split_cp_and_normal
        b_swap = FakeBatch(ids)
        n = len(perm)
        items_at_pos = list(range(n))
        pos_of_item = list(range(n))
        for dst_pos, target_item in enumerate(perm):
            src_pos = pos_of_item[target_item]
            if src_pos == dst_pos:
                continue
            b_swap.swap_states(dst_pos, src_pos)
            displaced_item = items_at_pos[dst_pos]
            items_at_pos[dst_pos], items_at_pos[src_pos] = (
                items_at_pos[src_pos],
                items_at_pos[dst_pos],
            )
            pos_of_item[target_item] = dst_pos
            pos_of_item[displaced_item] = src_pos

        # Compare all fields
        assert b_perm._req_ids[:n] == b_swap._req_ids[:n]
        assert b_perm.req_id_to_index == b_swap.req_id_to_index
        np.testing.assert_array_equal(
            b_perm.num_tokens[:n], b_swap.num_tokens[:n]
        )
        np.testing.assert_array_equal(
            b_perm.num_computed_tokens_cpu[:n],
            b_swap.num_computed_tokens_cpu[:n],
        )
        np.testing.assert_array_equal(
            b_perm.token_ids_cpu[:n], b_swap.token_ids_cpu[:n]
        )
        assert b_perm.req_output_token_ids[:n] == b_swap.req_output_token_ids[:n]
        assert set(b_perm.req_prompt_embeds.keys()) == set(
            b_swap.req_prompt_embeds.keys()
        )
        _assert_index_consistent(b_perm)
        _assert_index_consistent(b_swap)


# =======================================================================
# End-to-end: reorder_batch_to_split_cp_and_normal with FakeBatch
# =======================================================================

def reorder_batch_to_split_cp_and_normal(input_batch, scheduler_output):
    """Inline copy from vllm/v1/attention/backends/utils.py."""
    req_ids = input_batch.req_ids
    num_reqs = len([r for r in req_ids if r is not None])
    active_ids = req_ids[:num_reqs]
    cp_indices = [
        idx for idx, rid in enumerate(active_ids)
        if scheduler_output.cp_rank_scheduled_tokens[rid] > 1
    ]
    ncp_indices = [
        idx for idx, rid in enumerate(active_ids)
        if scheduler_output.cp_rank_scheduled_tokens[rid] <= 1
    ]
    target_order = cp_indices + ncp_indices
    if target_order == list(range(num_reqs)):
        return False
    if hasattr(input_batch, "apply_permutation"):
        return input_batch.apply_permutation(target_order)
    items_at_pos = list(range(num_reqs))
    pos_of_item = list(range(num_reqs))
    for dst_pos, target_item in enumerate(target_order):
        src_pos = pos_of_item[target_item]
        if src_pos == dst_pos:
            continue
        input_batch.swap_states(dst_pos, src_pos)
        displaced_item = items_at_pos[dst_pos]
        items_at_pos[dst_pos], items_at_pos[src_pos] = (
            items_at_pos[src_pos], items_at_pos[dst_pos],
        )
        pos_of_item[target_item] = dst_pos
        pos_of_item[displaced_item] = src_pos
    return True


class MockSchedulerOutput:
    def __init__(self, cp_map):
        self.cp_rank_scheduled_tokens = cp_map


class TestEndToEndWithFakeBatch:
    """Drive reorder through FakeBatch to exercise real swap/permutation."""

    def _make_batch(self, ids, use_permutation):
        """Create a FakeBatch; hide apply_permutation for swap path."""
        if use_permutation:
            return FakeBatch(ids)

        class SwapOnlyBatch(FakeBatch):
            # Override to make hasattr(..., "apply_permutation") return False
            @property
            def apply_permutation(self):
                raise AttributeError

        # delete the property so hasattr returns False
        del SwapOnlyBatch.apply_permutation
        return SwapOnlyBatch(ids)

    @pytest.mark.parametrize("use_permutation", [False, True],
                             ids=["swap_path", "permutation_path"])
    def test_interleaved(self, use_permutation: bool):
        # [ncp, cp, ncp, cp] -> [cp, cp, ncp, ncp]
        ids = ["a", "b", "c", "d"]
        cp_map = {"a": 1, "b": 4, "c": 1, "d": 4}

        b = self._make_batch(ids, use_permutation)
        sched = MockSchedulerOutput(cp_map)
        modified = reorder_batch_to_split_cp_and_normal(b, sched)

        assert modified is True
        assert b._req_ids[:4] == ["b", "d", "a", "c"]
        _assert_index_consistent(b)

    @pytest.mark.parametrize("use_permutation", [False, True],
                             ids=["swap_path", "permutation_path"])
    def test_already_ordered(self, use_permutation: bool):
        ids = ["x", "y", "z"]
        cp_map = {"x": 8, "y": 1, "z": 1}

        b = self._make_batch(ids, use_permutation)
        before = _snap(b)
        sched = MockSchedulerOutput(cp_map)
        modified = reorder_batch_to_split_cp_and_normal(b, sched)

        assert modified is False
        assert _snap(b).req_ids == before.req_ids

    @pytest.mark.parametrize("use_permutation", [False, True],
                             ids=["swap_path", "permutation_path"])
    def test_complex_preserves_data(self, use_permutation: bool):
        """After reorder, each req's num_tokens must follow its req_id."""
        ids = ["r0", "r1", "r2", "r3", "r4"]
        cp_map = {"r0": 1, "r1": 4, "r2": 1, "r3": 4, "r4": 1}

        b = self._make_batch(ids, use_permutation)
        # Record original per-req num_tokens
        original = {rid: int(b.num_tokens[i]) for i, rid in enumerate(ids)}

        sched = MockSchedulerOutput(cp_map)
        reorder_batch_to_split_cp_and_normal(b, sched)

        # Verify data followed its request
        for i in range(len(ids)):
            rid = b._req_ids[i]
            assert b.num_tokens[i] == original[rid], (
                f"pos {i}: req {rid} has num_tokens={b.num_tokens[i]}, "
                f"expected {original[rid]}"
            )
        _assert_index_consistent(b)
