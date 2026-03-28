# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SteeringManager.

The SteeringManager maintains a steering table with the following row layout:
    row 0  — always zeros (no-steering sentinel)
    row 1  — global-only steering vector
    rows 2+ — global + per-request combined vectors

Tests are grouped into three classes covering the registration/release
lifecycle, the row-lookup semantics, and the table-population logic.
"""

import torch
import torch.nn as nn

from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN_SIZE = 8
MAX_CONFIGS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSteerableLayer(nn.Module):
    """Minimal module that exposes a ``steering_table`` buffer."""

    def __init__(self, num_rows: int, hidden_size: int):
        super().__init__()
        self.register_buffer(
            "steering_table",
            torch.zeros(num_rows, hidden_size),
        )


def _make_manager(max_configs: int = MAX_CONFIGS) -> SteeringManager:
    return SteeringManager(max_steering_configs=max_configs)


def _make_layers(
    manager: SteeringManager,
    layer_indices: list[int] | None = None,
    hidden_size: int = HIDDEN_SIZE,
) -> dict[int, nn.Module]:
    """Build a dict of FakeSteerableLayer keyed by layer index."""
    if layer_indices is None:
        layer_indices = [0, 1]
    num_rows = manager.max_steering_configs + 2
    return {
        idx: FakeSteerableLayer(num_rows, hidden_size)
        for idx in layer_indices
    }


# ---------------------------------------------------------------------------
# TestRegisterRelease
# ---------------------------------------------------------------------------


class TestRegisterRelease:
    """Registration and release lifecycle for per-request configs."""

    def test_register_returns_row_gte_2(self):
        """Registered configs must occupy rows >= 2 (0=zeros, 1=global)."""
        mgr = _make_manager()
        vectors = {0: [1.0] * HIDDEN_SIZE}
        row = mgr.register_config(config_hash=42, vectors=vectors)
        assert row >= 2

    def test_duplicate_hash_returns_same_row(self):
        """Re-registering the same hash bumps the refcount, same row."""
        mgr = _make_manager()
        vectors = {0: [1.0] * HIDDEN_SIZE}
        row1 = mgr.register_config(config_hash=42, vectors=vectors)
        row2 = mgr.register_config(config_hash=42, vectors=vectors)
        assert row1 == row2

    def test_different_hashes_get_different_rows(self):
        """Distinct config hashes must not alias to the same row."""
        mgr = _make_manager()
        vectors_a = {0: [1.0] * HIDDEN_SIZE}
        vectors_b = {0: [2.0] * HIDDEN_SIZE}
        row_a = mgr.register_config(config_hash=100, vectors=vectors_a)
        row_b = mgr.register_config(config_hash=200, vectors=vectors_b)
        assert row_a != row_b

    def test_release_decrements_refcount_still_active(self):
        """Releasing once with refcount > 1 keeps the config active."""
        mgr = _make_manager()
        vectors = {0: [1.0] * HIDDEN_SIZE}
        mgr.register_config(config_hash=42, vectors=vectors)
        mgr.register_config(config_hash=42, vectors=vectors)  # refcount = 2
        mgr.release_config(config_hash=42)  # refcount -> 1
        # Config should still be active and resolvable.
        row = mgr.get_row_for_config(config_hash=42)
        assert row >= 2

    def test_release_frees_row_at_refcount_zero(self):
        """Fully releasing a config makes its row reclaimable."""
        mgr = _make_manager()
        vectors = {0: [1.0] * HIDDEN_SIZE}
        mgr.register_config(config_hash=42, vectors=vectors)
        assert mgr.num_active_configs == 1
        mgr.release_config(config_hash=42)
        assert mgr.num_active_configs == 0

    def test_freed_row_is_reusable(self):
        """After full release, the row can be assigned to a new config."""
        mgr = _make_manager(max_configs=1)
        vectors = {0: [1.0] * HIDDEN_SIZE}
        row_old = mgr.register_config(config_hash=42, vectors=vectors)
        mgr.release_config(config_hash=42)

        # A completely new config should be able to use the freed slot.
        vectors_new = {0: [9.0] * HIDDEN_SIZE}
        row_new = mgr.register_config(config_hash=99, vectors=vectors_new)
        assert row_new == row_old  # same slot recycled

    def test_capacity_exhaustion_raises(self):
        """Exceeding max_steering_configs raises RuntimeError."""
        mgr = _make_manager(max_configs=2)
        mgr.register_config(config_hash=1, vectors={0: [1.0] * HIDDEN_SIZE})
        mgr.register_config(config_hash=2, vectors={0: [2.0] * HIDDEN_SIZE})
        try:
            mgr.register_config(
                config_hash=3, vectors={0: [3.0] * HIDDEN_SIZE}
            )
            assert False, "Expected RuntimeError for capacity exhaustion"
        except RuntimeError:
            pass

    def test_release_nonexistent_hash_is_noop(self):
        """Releasing a hash that was never registered must not raise."""
        mgr = _make_manager()
        mgr.release_config(config_hash=999)  # should not raise


# ---------------------------------------------------------------------------
# TestGetRow
# ---------------------------------------------------------------------------


class TestGetRow:
    """Row-lookup semantics for get_row_for_config."""

    def test_hash_zero_returns_global_row(self):
        """Hash 0 is the sentinel for 'use global only' -> row 1."""
        mgr = _make_manager()
        assert mgr.get_row_for_config(config_hash=0) == 1

    def test_registered_hash_returns_assigned_row(self):
        """A registered config returns the row assigned at registration."""
        mgr = _make_manager()
        vectors = {0: [1.0] * HIDDEN_SIZE}
        row = mgr.register_config(config_hash=42, vectors=vectors)
        assert mgr.get_row_for_config(config_hash=42) == row

    def test_unregistered_nonzero_hash_returns_global_row(self):
        """An unknown nonzero hash falls back to global row 1."""
        mgr = _make_manager()
        assert mgr.get_row_for_config(config_hash=12345) == 1


# ---------------------------------------------------------------------------
# TestPopulateTables
# ---------------------------------------------------------------------------


class TestPopulateTables:
    """Table-population logic exercised via populate_steering_tables."""

    def test_row_zero_always_zeros(self):
        """Row 0 must be all zeros even if the buffer was dirtied."""
        mgr = _make_manager()
        layers = _make_layers(mgr, layer_indices=[0])
        # Dirty row 0 on purpose.
        layers[0].steering_table[0].fill_(99.0)
        mgr.populate_steering_tables(layers)
        assert torch.allclose(
            layers[0].steering_table[0],
            torch.zeros(HIDDEN_SIZE),
        )

    def test_row_one_gets_global_vector(self):
        """Row 1 should contain the global steering vector for each layer."""
        mgr = _make_manager()
        global_vec = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(layer_idx=0, vector=global_vec)
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        assert torch.allclose(
            layers[0].steering_table[1],
            global_vec,
        )

    def test_row_one_zero_without_global(self):
        """Without a global vector set, row 1 must remain zeros."""
        mgr = _make_manager()
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        assert torch.allclose(
            layers[0].steering_table[1],
            torch.zeros(HIDDEN_SIZE),
        )

    def test_per_request_row_is_additive(self):
        """Per-request rows should contain global + per_request vectors."""
        mgr = _make_manager()
        global_vec = torch.ones(HIDDEN_SIZE) * 2.0
        per_req_vec = torch.ones(HIDDEN_SIZE) * 5.0
        mgr.update_global_vectors(layer_idx=0, vector=global_vec)
        vectors = {0: per_req_vec.tolist()}
        row = mgr.register_config(config_hash=42, vectors=vectors)

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        expected = global_vec + per_req_vec
        assert torch.allclose(
            layers[0].steering_table[row],
            expected,
        ), (
            f"Row {row} should be global+per_request.\n"
            f"Expected: {expected}\n"
            f"Got: {layers[0].steering_table[row]}"
        )

    def test_per_request_only_no_global(self):
        """Per-request config without global should just have per_request."""
        mgr = _make_manager()
        per_req_vec = torch.ones(HIDDEN_SIZE) * 7.0
        vectors = {0: per_req_vec.tolist()}
        row = mgr.register_config(config_hash=42, vectors=vectors)

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        assert torch.allclose(
            layers[0].steering_table[row],
            per_req_vec,
        )

    def test_layer_without_per_request_gets_global_only(self):
        """A layer not covered by per-request vectors gets global in row."""
        mgr = _make_manager()
        global_vec_0 = torch.ones(HIDDEN_SIZE) * 2.0
        global_vec_1 = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(layer_idx=0, vector=global_vec_0)
        mgr.update_global_vectors(layer_idx=1, vector=global_vec_1)

        # Per-request config only targets layer 0, not layer 1.
        vectors = {0: [5.0] * HIDDEN_SIZE}
        row = mgr.register_config(config_hash=42, vectors=vectors)

        layers = _make_layers(mgr, layer_indices=[0, 1])
        mgr.populate_steering_tables(layers)

        # Layer 0: global + per_request
        expected_layer0 = global_vec_0 + torch.tensor([5.0] * HIDDEN_SIZE)
        assert torch.allclose(
            layers[0].steering_table[row],
            expected_layer0,
        )

        # Layer 1: only global (per-request didn't target it)
        assert torch.allclose(
            layers[1].steering_table[row],
            global_vec_1,
        )

    def test_clear_global_vectors_zeros_row_one(self):
        """After clear_global_vectors, row 1 must return to zeros."""
        mgr = _make_manager()
        global_vec = torch.ones(HIDDEN_SIZE) * 4.0
        mgr.update_global_vectors(layer_idx=0, vector=global_vec)
        mgr.clear_global_vectors()

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        assert torch.allclose(
            layers[0].steering_table[1],
            torch.zeros(HIDDEN_SIZE),
        )
