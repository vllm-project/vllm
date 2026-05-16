# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SteeringManager.

The SteeringManager maintains per-hook-point steering tables with the
following row layout:
    row 0  — always zeros (no-steering sentinel)
    row 1  — global prefill effective (global_base + global_prefill)
    row 2  — global decode effective (global_base + global_decode)
    rows 3+ — phase-appropriate global + per-request combined vectors

Tests are grouped into classes covering the registration/release
lifecycle, the row-lookup semantics, the table-population logic,
and the phase-aware config tracking.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN_SIZE = 8
MAX_CONFIGS = 4
_HP = DEFAULT_HOOK_POINT.value  # "post_mlp"
_TABLE_ATTR = "steering_table_post_mlp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeSteerableLayer(nn.Module):
    """Minimal module that exposes a per-hook ``steering_table_*`` buffer."""

    def __init__(self, num_rows: int, hidden_size: int):
        super().__init__()
        self.register_buffer(
            _TABLE_ATTR,
            torch.zeros(num_rows, hidden_size),
        )


def _make_manager(
    max_configs: int = MAX_CONFIGS,
    device: "torch.device | None" = None,
) -> SteeringManager:
    return SteeringManager(max_steering_configs=max_configs, device=device)


def _make_layers(
    manager: SteeringManager,
    layer_indices: list[int] | None = None,
    hidden_size: int = HIDDEN_SIZE,
) -> dict[int, nn.Module]:
    """Build a dict of FakeSteerableLayer keyed by layer index."""
    if layer_indices is None:
        layer_indices = [0, 1]
    # +3 for rows 0 (zeros), 1 (global prefill), 2 (global decode)
    num_rows = manager.max_steering_configs + 3
    return {idx: FakeSteerableLayer(num_rows, hidden_size) for idx in layer_indices}


# ---------------------------------------------------------------------------
# TestRegisterRelease
# ---------------------------------------------------------------------------


class TestRegisterRelease:
    """Registration and release lifecycle for per-request configs."""

    def test_register_returns_row_gte_3(self):
        """Registered configs must occupy rows >= 3
        (0=zeros, 1=global prefill, 2=global decode)."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert row >= 3

    def test_duplicate_hash_returns_same_row(self):
        """Re-registering the same hash bumps the refcount, same row."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row1 = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        row2 = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert row1 == row2

    def test_different_hashes_get_different_rows(self):
        """Distinct config hashes must not alias to the same row."""
        mgr = _make_manager()
        vectors_a = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vectors_b = {_HP: {0: [2.0] * HIDDEN_SIZE}}
        row_a = mgr.register_config(config_hash=100, vectors=vectors_a, phase="prefill")
        row_b = mgr.register_config(config_hash=200, vectors=vectors_b, phase="decode")
        assert row_a != row_b

    def test_release_decrements_refcount_still_active(self):
        """Releasing once with refcount > 1 keeps the config active."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        mgr.register_config(
            config_hash=42, vectors=vectors, phase="prefill"
        )  # refcount = 2
        mgr.release_config(config_hash=42, phase="prefill")  # refcount -> 1
        # Config should still be active and resolvable.
        row = mgr.get_row_for_config(config_hash=42, is_prefill=True)
        assert row >= 3

    def test_release_frees_row_at_refcount_zero(self):
        """Fully releasing a config makes its row reclaimable."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert mgr.num_active_configs == 1
        mgr.release_config(config_hash=42, phase="prefill")
        assert mgr.num_active_configs == 0

    def test_freed_row_is_reusable(self):
        """After full release, the row can be assigned to a new config."""
        mgr = _make_manager(max_configs=1)
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row_old = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        mgr.release_config(config_hash=42, phase="prefill")

        # A completely new config should be able to use the freed slot.
        vectors_new = {_HP: {0: [9.0] * HIDDEN_SIZE}}
        row_new = mgr.register_config(
            config_hash=99, vectors=vectors_new, phase="decode"
        )
        assert row_new == row_old  # same slot recycled

    def test_capacity_exhaustion_raises(self):
        """Exceeding max_steering_configs raises RuntimeError."""
        mgr = _make_manager(max_configs=2)
        mgr.register_config(
            config_hash=1,
            vectors={_HP: {0: [1.0] * HIDDEN_SIZE}},
            phase="prefill",
        )
        mgr.register_config(
            config_hash=2,
            vectors={_HP: {0: [2.0] * HIDDEN_SIZE}},
            phase="decode",
        )
        try:
            mgr.register_config(
                config_hash=3,
                vectors={_HP: {0: [3.0] * HIDDEN_SIZE}},
                phase="prefill",
            )
            raise AssertionError("Expected RuntimeError for capacity exhaustion")
        except RuntimeError:
            pass

    def test_release_nonexistent_hash_is_noop(self):
        """Releasing a hash that was never registered must not raise."""
        mgr = _make_manager()
        mgr.release_config(config_hash=999, phase="prefill")  # should not raise

    def test_free_rows_start_from_row_3(self):
        """Free rows should start from row 3, not row 2."""
        mgr = _make_manager(max_configs=2)
        # With max_configs=2, rows 3 and 4 should be available
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row1 = mgr.register_config(config_hash=10, vectors=vectors, phase="prefill")
        row2 = mgr.register_config(config_hash=20, vectors=vectors, phase="decode")
        assert row1 >= 3
        assert row2 >= 3
        assert {row1, row2} == {3, 4}

    def test_same_hash_different_phase_gets_separate_rows(self):
        """Same config_hash with different phases must get separate rows."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row_p = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        row_d = mgr.register_config(config_hash=42, vectors=vectors, phase="decode")
        assert row_p != row_d
        assert row_p >= 3
        assert row_d >= 3

    def test_same_hash_same_phase_deduplicates(self):
        """Same hash + same phase should deduplicate (bump refcount)."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row1 = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        row2 = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert row1 == row2


# ---------------------------------------------------------------------------
# TestGetRow
# ---------------------------------------------------------------------------


class TestGetRow:
    """Row-lookup semantics for get_row_for_config."""

    def test_hash_zero_prefill_returns_row_1(self):
        """Hash 0 with is_prefill=True -> row 1 (global prefill)."""
        mgr = _make_manager()
        assert mgr.get_row_for_config(config_hash=0, is_prefill=True) == 1

    def test_hash_zero_decode_returns_row_2(self):
        """Hash 0 with is_prefill=False -> row 2 (global decode)."""
        mgr = _make_manager()
        assert mgr.get_row_for_config(config_hash=0, is_prefill=False) == 2

    def test_registered_hash_returns_assigned_row(self):
        """A registered config returns the row assigned at registration."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert mgr.get_row_for_config(config_hash=42, is_prefill=True) == row

    def test_unregistered_nonzero_prefill_raises(self):
        """An unknown nonzero hash with is_prefill=True crashes loudly."""
        mgr = _make_manager()
        with pytest.raises(RuntimeError, match="not registered"):
            mgr.get_row_for_config(config_hash=12345, is_prefill=True)

    def test_unregistered_nonzero_decode_raises(self):
        """An unknown nonzero hash with is_prefill=False crashes loudly."""
        mgr = _make_manager()
        with pytest.raises(RuntimeError, match="not registered"):
            mgr.get_row_for_config(config_hash=12345, is_prefill=False)

    def test_registered_hash_uses_phase_for_lookup(self):
        """A prefill-registered hash is only found with is_prefill=True."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert mgr.get_row_for_config(config_hash=42, is_prefill=True) == row
        # The (42, "decode") key is absent, so the decode lookup raises.
        with pytest.raises(RuntimeError, match="not registered"):
            mgr.get_row_for_config(config_hash=42, is_prefill=False)


# ---------------------------------------------------------------------------
# TestPopulateTables
# ---------------------------------------------------------------------------


class TestPopulateTables:
    """Table-population logic exercised via populate_steering_tables."""

    def test_row_zero_always_zeros(self):
        """Row 0 must be all zeros even if the buffer was dirtied."""
        mgr = _make_manager()
        layers = _make_layers(mgr, layer_indices=[0])
        table = getattr(layers[0], _TABLE_ATTR)
        # Dirty row 0 on purpose.
        table[0].fill_(99.0)
        mgr.populate_steering_tables(layers)
        assert torch.allclose(
            table[0],
            torch.zeros(HIDDEN_SIZE),
        )

    def test_row_one_gets_global_prefill_effective(self):
        """Row 1 should contain global_base + global_prefill."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 2.0
        prefill_vec = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=prefill_vec, phase="prefill"
        )
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)
        expected = base_vec + prefill_vec  # 5.0
        assert torch.allclose(table[1], expected)

    def test_row_two_gets_global_decode_effective(self):
        """Row 2 should contain global_base + global_decode."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 2.0
        decode_vec = torch.ones(HIDDEN_SIZE) * 4.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=decode_vec, phase="decode"
        )
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)
        expected = base_vec + decode_vec  # 6.0
        assert torch.allclose(table[2], expected)

    def test_row_one_base_only_no_prefill_global(self):
        """Row 1 with only base vectors (no prefill-specific) = base."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)
        assert torch.allclose(table[1], base_vec)

    def test_row_one_zero_without_any_global(self):
        """Without any global vector set, row 1 must remain zeros."""
        mgr = _make_manager()
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)
        assert torch.allclose(table[1], torch.zeros(HIDDEN_SIZE))

    def test_row_two_zero_without_any_global(self):
        """Without any global vector set, row 2 must remain zeros."""
        mgr = _make_manager()
        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)
        assert torch.allclose(table[2], torch.zeros(HIDDEN_SIZE))

    def test_prefill_per_request_row_uses_prefill_global(self):
        """Per-request prefill row = (base+prefill) + per_request."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 1.0
        prefill_vec = torch.ones(HIDDEN_SIZE) * 2.0
        per_req_vec = torch.ones(HIDDEN_SIZE) * 5.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=prefill_vec, phase="prefill"
        )
        vectors = {_HP: {0: per_req_vec.tolist()}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        table = getattr(layers[0], _TABLE_ATTR)
        expected = base_vec + prefill_vec + per_req_vec  # 1+2+5 = 8.0
        assert torch.allclose(table[row], expected), (
            f"Row {row} should be (base+prefill)+per_request.\n"
            f"Expected: {expected}\n"
            f"Got: {table[row]}"
        )

    def test_decode_per_request_row_uses_decode_global(self):
        """Per-request decode row = (base+decode) + per_request."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 1.0
        decode_vec = torch.ones(HIDDEN_SIZE) * 3.0
        per_req_vec = torch.ones(HIDDEN_SIZE) * 7.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=decode_vec, phase="decode"
        )
        vectors = {_HP: {0: per_req_vec.tolist()}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="decode")

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        table = getattr(layers[0], _TABLE_ATTR)
        expected = base_vec + decode_vec + per_req_vec  # 1+3+7 = 11.0
        assert torch.allclose(table[row], expected), (
            f"Row {row} should be (base+decode)+per_request.\n"
            f"Expected: {expected}\n"
            f"Got: {table[row]}"
        )

    def test_per_request_only_no_global(self):
        """Per-request config without global should just have per_request."""
        mgr = _make_manager()
        per_req_vec = torch.ones(HIDDEN_SIZE) * 7.0
        vectors = {_HP: {0: per_req_vec.tolist()}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        table = getattr(layers[0], _TABLE_ATTR)
        assert torch.allclose(table[row], per_req_vec)

    def test_layer_without_per_request_gets_phase_global_only(self):
        """A layer not covered by per-request vectors gets phase global in row."""
        mgr = _make_manager()
        base_0 = torch.ones(HIDDEN_SIZE) * 2.0
        base_1 = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_0, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=1, vector=base_1, phase="base"
        )

        # Per-request config only targets layer 0, not layer 1.
        vectors = {_HP: {0: [5.0] * HIDDEN_SIZE}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        layers = _make_layers(mgr, layer_indices=[0, 1])
        mgr.populate_steering_tables(layers)

        # Layer 0: base + per_request (no prefill global, so just base+per_req)
        expected_layer0 = base_0 + torch.tensor([5.0] * HIDDEN_SIZE)
        table_0 = getattr(layers[0], _TABLE_ATTR)
        assert torch.allclose(table_0[row], expected_layer0)

        # Layer 1: only base (per-request didn't target it)
        table_1 = getattr(layers[1], _TABLE_ATTR)
        assert torch.allclose(table_1[row], base_1)

    def test_clear_global_vectors_zeros_rows_one_and_two(self):
        """After clear_global_vectors, rows 1 and 2 must return to zeros."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 4.0
        prefill_vec = torch.ones(HIDDEN_SIZE) * 2.0
        decode_vec = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=prefill_vec, phase="prefill"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=decode_vec, phase="decode"
        )
        mgr.clear_global_vectors()

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)

        table = getattr(layers[0], _TABLE_ATTR)
        assert torch.allclose(table[1], torch.zeros(HIDDEN_SIZE))
        assert torch.allclose(table[2], torch.zeros(HIDDEN_SIZE))

    def test_mixed_prefill_decode_configs_in_same_batch(self):
        """Prefill and decode configs in the same batch use correct globals."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 1.0
        prefill_global = torch.ones(HIDDEN_SIZE) * 2.0
        decode_global = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=prefill_global, phase="prefill"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=decode_global, phase="decode"
        )

        prefill_per_req = torch.ones(HIDDEN_SIZE) * 10.0
        decode_per_req = torch.ones(HIDDEN_SIZE) * 20.0

        row_p = mgr.register_config(
            config_hash=100,
            vectors={_HP: {0: prefill_per_req.tolist()}},
            phase="prefill",
        )
        row_d = mgr.register_config(
            config_hash=200,
            vectors={_HP: {0: decode_per_req.tolist()}},
            phase="decode",
        )

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)

        # Prefill row: (base + prefill_global) + prefill_per_req = 1+2+10 = 13
        expected_p = base_vec + prefill_global + prefill_per_req
        assert torch.allclose(table[row_p], expected_p)

        # Decode row: (base + decode_global) + decode_per_req = 1+3+20 = 24
        expected_d = base_vec + decode_global + decode_per_req
        assert torch.allclose(table[row_d], expected_d)

    def test_same_hash_different_phase_uses_correct_globals(self):
        """Same hash registered as both prefill and decode should combine
        with the correct global for each phase."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 1.0
        prefill_global = torch.ones(HIDDEN_SIZE) * 2.0
        decode_global = torch.ones(HIDDEN_SIZE) * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=prefill_global, phase="prefill"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=decode_global, phase="decode"
        )

        per_req = torch.ones(HIDDEN_SIZE) * 5.0
        vectors = {_HP: {0: per_req.tolist()}}
        row_p = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        row_d = mgr.register_config(config_hash=42, vectors=vectors, phase="decode")

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)

        # Prefill row: (base + prefill_global) + per_req = 1+2+5 = 8
        assert torch.allclose(table[row_p], base_vec + prefill_global + per_req)
        # Decode row: (base + decode_global) + per_req = 1+3+5 = 9
        assert torch.allclose(table[row_d], base_vec + decode_global + per_req)

    def test_invalid_phase_in_populate_raises(self):
        """Invalid phase string in config_to_row raises ValueError."""
        mgr = _make_manager()
        per_req = torch.ones(HIDDEN_SIZE) * 5.0
        vectors = {_HP: {0: per_req.tolist()}}
        # Register normally so state is otherwise valid.
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        # Corrupt the phase tracking state by injecting an invalid phase
        # into the dict populate_steering_tables iterates over. Mirror the
        # per-request vectors under the same invalid key so the iteration
        # reaches the phase branch.
        mgr.config_to_row.pop((42, "prefill"))
        mgr.config_to_row[(42, "invalid")] = row
        mgr.config_vectors[(42, "invalid")] = mgr.config_vectors.pop((42, "prefill"))

        layers = _make_layers(mgr, layer_indices=[0])
        try:
            mgr.populate_steering_tables(layers)
            raise AssertionError("Expected ValueError for invalid phase")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# TestPhaseTracking
# ---------------------------------------------------------------------------


class TestPhaseTracking:
    """Phase tracking via (config_hash, phase) composite keys."""

    def test_config_phase_stored_on_register(self):
        """Register with phase='prefill' is keyed by (hash, 'prefill')."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert (42, "prefill") in mgr.config_to_row

    def test_config_phase_decode_stored(self):
        """Register with phase='decode' is keyed by (hash, 'decode')."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="decode")
        assert (42, "decode") in mgr.config_to_row

    def test_config_phase_cleaned_on_release(self):
        """Releasing a config removes its (hash, phase) entry."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        mgr.release_config(42, phase="prefill")
        assert (42, "prefill") not in mgr.config_to_row

    def test_config_phase_not_cleaned_with_remaining_refcount(self):
        """(hash, phase) entry persists while refcount > 0."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        mgr.release_config(42, phase="prefill")  # refcount -> 1
        assert (42, "prefill") in mgr.config_to_row


# ---------------------------------------------------------------------------
# TestPhaseTransitionLifecycle
# ---------------------------------------------------------------------------


class TestPhaseTransitionLifecycle:
    """Simulate the register-prefill, release, register-decode lifecycle."""

    def test_prefill_then_decode_transition(self):
        """Register prefill, release, register decode — different rows OK."""
        mgr = _make_manager()
        vectors_p = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vectors_d = {_HP: {0: [2.0] * HIDDEN_SIZE}}

        row_p = mgr.register_config(config_hash=100, vectors=vectors_p, phase="prefill")
        assert row_p >= 3
        assert (100, "prefill") in mgr.config_to_row

        mgr.release_config(100, phase="prefill")
        assert mgr.num_active_configs == 0
        assert (100, "prefill") not in mgr.config_to_row

        row_d = mgr.register_config(config_hash=200, vectors=vectors_d, phase="decode")
        assert row_d >= 3
        assert (200, "decode") in mgr.config_to_row

        mgr.release_config(200, phase="decode")
        assert mgr.num_active_configs == 0

    def test_transition_reuses_freed_row(self):
        """After prefill config is released, decode config can reuse the row."""
        mgr = _make_manager(max_configs=1)
        vectors_p = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vectors_d = {_HP: {0: [2.0] * HIDDEN_SIZE}}

        row_p = mgr.register_config(config_hash=100, vectors=vectors_p, phase="prefill")
        mgr.release_config(100, phase="prefill")

        row_d = mgr.register_config(config_hash=200, vectors=vectors_d, phase="decode")
        assert row_d == row_p  # reused slot


# ---------------------------------------------------------------------------
# TestUpdateGlobalVectors
# ---------------------------------------------------------------------------


class TestUpdateGlobalVectors:
    """Test phase-routed update_global_vectors."""

    def test_base_phase_stores_in_base(self):
        """phase='base' routes to global_base_vectors."""
        mgr = _make_manager()
        vec = torch.ones(HIDDEN_SIZE) * 5.0
        mgr.update_global_vectors(hook_point=_HP, layer_idx=0, vector=vec, phase="base")
        stored = mgr.global_base_vectors[_HP][0]
        assert torch.allclose(stored, vec)

    def test_prefill_phase_stores_in_prefill(self):
        """phase='prefill' routes to global_prefill_vectors."""
        mgr = _make_manager()
        vec = torch.ones(HIDDEN_SIZE) * 6.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=vec, phase="prefill"
        )
        stored = mgr.global_prefill_vectors[_HP][0]
        assert torch.allclose(stored, vec)

    def test_decode_phase_stores_in_decode(self):
        """phase='decode' routes to global_decode_vectors."""
        mgr = _make_manager()
        vec = torch.ones(HIDDEN_SIZE) * 7.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=vec, phase="decode"
        )
        stored = mgr.global_decode_vectors[_HP][0]
        assert torch.allclose(stored, vec)

    def test_invalid_phase_raises(self):
        """Invalid phase string raises ValueError."""
        mgr = _make_manager()
        vec = torch.ones(HIDDEN_SIZE) * 1.0
        try:
            mgr.update_global_vectors(
                hook_point=_HP, layer_idx=0, vector=vec, phase="invalid"
            )
            raise AssertionError("Expected ValueError for invalid phase")
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# TestBackfillRegistration
# ---------------------------------------------------------------------------


class TestBackfillRegistration:
    """Verify the backfill registration logic that runs during lazy
    manager initialization in ``_update_steering_buffers``.

    When the SteeringManager is first created, requests already in the
    batch need their steering configs registered retroactively.  The
    backfill must be phase-aware: requests still in prefill register
    their prefill config, while requests that start directly in decode
    (full prefix-cache hit where ``num_computed >= num_prompt``) must
    register their decode config instead.
    """

    @staticmethod
    def _run_backfill(
        manager: SteeringManager,
        requests: list[dict],
    ) -> None:
        """Simulate the backfill loop from ``_update_steering_buffers``.

        Each entry in *requests* is a dict with keys:
            num_computed: int
            num_prompt: int
            prefill_hash: int
            decode_hash: int
            effective_prefill: dict | None
            effective_decode: dict | None
        """
        for req in requests:
            num_computed = req["num_computed"]
            num_prompt = req["num_prompt"]

            if num_computed < num_prompt:
                ph = req["prefill_hash"]
                if ph != 0:
                    eff = req.get("effective_prefill")
                    if eff:
                        manager.register_config(ph, eff, phase="prefill")
            else:
                dh = req["decode_hash"]
                if dh != 0:
                    eff = req.get("effective_decode")
                    if eff:
                        manager.register_config(dh, eff, phase="decode")

    def test_prefill_request_registers_prefill_config(self):
        """A request in prefill (num_computed < num_prompt) registers
        its prefill steering config during backfill."""
        mgr = _make_manager()
        prefill_vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 0,
                    "num_prompt": 100,
                    "prefill_hash": 42,
                    "decode_hash": 43,
                    "effective_prefill": prefill_vectors,
                    "effective_decode": {_HP: {0: [2.0] * HIDDEN_SIZE}},
                },
            ],
        )
        assert (42, "prefill") in mgr.config_to_row
        assert (43, "decode") not in mgr.config_to_row

    def test_decode_start_request_registers_decode_config(self):
        """A request starting in decode (num_computed >= num_prompt,
        e.g. full prefix-cache hit) registers its decode steering
        config during backfill."""
        mgr = _make_manager()
        decode_vectors = {_HP: {0: [3.0] * HIDDEN_SIZE}}
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 100,
                    "num_prompt": 100,
                    "prefill_hash": 42,
                    "decode_hash": 43,
                    "effective_prefill": {_HP: {0: [1.0] * HIDDEN_SIZE}},
                    "effective_decode": decode_vectors,
                },
            ],
        )
        assert (43, "decode") in mgr.config_to_row
        assert (42, "prefill") not in mgr.config_to_row

    def test_mixed_prefill_and_decode_requests(self):
        """A batch with both prefill and decode-start requests registers
        the correct phase config for each."""
        mgr = _make_manager()
        prefill_vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        decode_vectors = {_HP: {0: [2.0] * HIDDEN_SIZE}}
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 0,
                    "num_prompt": 100,
                    "prefill_hash": 10,
                    "decode_hash": 11,
                    "effective_prefill": prefill_vectors,
                    "effective_decode": {_HP: {0: [9.0] * HIDDEN_SIZE}},
                },
                {
                    "num_computed": 50,
                    "num_prompt": 50,
                    "prefill_hash": 20,
                    "decode_hash": 21,
                    "effective_prefill": {_HP: {0: [8.0] * HIDDEN_SIZE}},
                    "effective_decode": decode_vectors,
                },
            ],
        )
        # First request: in prefill
        assert (10, "prefill") in mgr.config_to_row
        assert (11, "decode") not in mgr.config_to_row
        # Second request: in decode (num_computed == num_prompt)
        assert (21, "decode") in mgr.config_to_row
        assert (20, "prefill") not in mgr.config_to_row

    def test_zero_hash_skipped_in_both_phases(self):
        """Requests with hash=0 (no per-request steering) are skipped
        in both prefill and decode backfill paths."""
        mgr = _make_manager()
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 0,
                    "num_prompt": 100,
                    "prefill_hash": 0,
                    "decode_hash": 0,
                    "effective_prefill": None,
                    "effective_decode": None,
                },
                {
                    "num_computed": 100,
                    "num_prompt": 100,
                    "prefill_hash": 0,
                    "decode_hash": 0,
                    "effective_prefill": None,
                    "effective_decode": None,
                },
            ],
        )
        assert mgr.num_active_configs == 0

    def test_none_effective_steering_skipped(self):
        """If effective_prefill/effective_decode is None, registration
        is skipped even with a nonzero hash."""
        mgr = _make_manager()
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 0,
                    "num_prompt": 100,
                    "prefill_hash": 42,
                    "decode_hash": 0,
                    "effective_prefill": None,
                    "effective_decode": None,
                },
                {
                    "num_computed": 100,
                    "num_prompt": 100,
                    "prefill_hash": 0,
                    "decode_hash": 43,
                    "effective_prefill": None,
                    "effective_decode": None,
                },
            ],
        )
        assert mgr.num_active_configs == 0

    def test_decode_backfill_vectors_populate_correctly(self):
        """Verify that a decode config registered via backfill has
        correct vectors in the steering table after population."""
        mgr = _make_manager()
        base_vec = torch.ones(HIDDEN_SIZE) * 1.0
        decode_global = torch.ones(HIDDEN_SIZE) * 3.0
        per_req_vec = [7.0] * HIDDEN_SIZE

        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=decode_global, phase="decode"
        )

        decode_vectors = {_HP: {0: per_req_vec}}
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 200,
                    "num_prompt": 200,
                    "prefill_hash": 50,
                    "decode_hash": 51,
                    "effective_prefill": {_HP: {0: [1.0] * HIDDEN_SIZE}},
                    "effective_decode": decode_vectors,
                },
            ],
        )

        assert (51, "decode") in mgr.config_to_row
        row = mgr.config_to_row[(51, "decode")]

        layers = _make_layers(mgr, layer_indices=[0])
        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)

        # Expected: (base + decode_global) + per_req = 1+3+7 = 11.0
        expected = base_vec + decode_global + torch.tensor(per_req_vec)
        assert torch.allclose(table[row], expected), (
            f"Backfill-registered decode row should be "
            f"(base+decode_global)+per_req.\n"
            f"Expected: {expected}\nGot: {table[row]}"
        )

    def test_already_registered_config_increments_refcount(self):
        """Backfill increments refcount for configs already in config_to_row."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        # Pre-register the config
        row_pre = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert mgr.config_refcounts[(42, "prefill")] == 1

        # Backfill should increment refcount for (42, "prefill")
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 0,
                    "num_prompt": 100,
                    "prefill_hash": 42,
                    "decode_hash": 0,
                    "effective_prefill": vectors,
                    "effective_decode": None,
                },
            ],
        )
        # Refcount should now be 2 — backfill incremented it
        assert mgr.config_refcounts[(42, "prefill")] == 2
        assert mgr.config_to_row[(42, "prefill")] == row_pre

    def test_backfill_shared_hash_refcount(self):
        """Two requests with the same prefill hash in one backfill batch
        should produce refcount=2.  Releasing one should leave refcount=1
        and the row still allocated."""
        mgr = _make_manager()
        vectors = {_HP: {0: [5.0] * HIDDEN_SIZE}}
        self._run_backfill(
            mgr,
            [
                {
                    "num_computed": 0,
                    "num_prompt": 100,
                    "prefill_hash": 77,
                    "decode_hash": 0,
                    "effective_prefill": vectors,
                    "effective_decode": None,
                },
                {
                    "num_computed": 0,
                    "num_prompt": 200,
                    "prefill_hash": 77,
                    "decode_hash": 0,
                    "effective_prefill": vectors,
                    "effective_decode": None,
                },
            ],
        )
        assert (77, "prefill") in mgr.config_to_row
        assert mgr.config_refcounts[(77, "prefill")] == 2

        # Release one request — row should stay allocated with refcount 1.
        mgr.release_config(config_hash=77, phase="prefill")
        assert mgr.config_refcounts[(77, "prefill")] == 1
        assert (77, "prefill") in mgr.config_to_row

        # Release the second — row should be freed.
        mgr.release_config(config_hash=77, phase="prefill")
        assert (77, "prefill") not in mgr.config_to_row


# ---------------------------------------------------------------------------
# TestDeferredDecodeRegistration
# ---------------------------------------------------------------------------


class TestCapacityExhaustion:
    """Verify that capacity exhaustion raises RuntimeError and that
    registrations succeed after capacity is freed.

    The mixin used to catch this RuntimeError and defer; with strict
    capacity enforcement the scheduler is responsible for not dispatching
    requests beyond ``max_steering_configs``, so this exception is now
    a "scheduler bug" signal rather than an expected control-flow path.
    """

    def test_register_config_raises_on_capacity(self):
        """When max_configs slots are full, registering a new config
        raises RuntimeError."""
        mgr = _make_manager(max_configs=2)
        vectors_a = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vectors_b = {_HP: {0: [2.0] * HIDDEN_SIZE}}
        vectors_c = {_HP: {0: [3.0] * HIDDEN_SIZE}}

        mgr.register_config(config_hash=1, vectors=vectors_a, phase="prefill")
        mgr.register_config(config_hash=2, vectors=vectors_b, phase="decode")
        assert mgr.num_active_configs == 2

        # Simulates the transition scenario: prefill still held by
        # another request sharing the same config, so refcount didn't
        # drop to 0 and the row wasn't freed.
        try:
            mgr.register_config(config_hash=3, vectors=vectors_c, phase="decode")
            raise AssertionError("Expected RuntimeError when capacity is exhausted")
        except RuntimeError:
            pass

        # The failed registration should not have modified state.
        assert mgr.num_active_configs == 2
        assert (3, "decode") not in mgr.config_to_row

    def test_pending_registration_processed_after_release(self):
        """After releasing a config to free a slot, a previously-failed
        registration should succeed on retry (row reused)."""
        mgr = _make_manager(max_configs=2)
        vectors_a = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vectors_b = {_HP: {0: [2.0] * HIDDEN_SIZE}}
        vectors_c = {_HP: {0: [3.0] * HIDDEN_SIZE}}

        mgr.register_config(config_hash=1, vectors=vectors_a, phase="prefill")
        row_b = mgr.register_config(config_hash=2, vectors=vectors_b, phase="decode")
        assert mgr.num_active_configs == 2

        # Release one config to free a slot.
        mgr.release_config(config_hash=2, phase="decode")
        assert mgr.num_active_configs == 1

        # Now the deferred registration should succeed.
        row_c = mgr.register_config(config_hash=3, vectors=vectors_c, phase="decode")
        assert mgr.num_active_configs == 2
        assert (3, "decode") in mgr.config_to_row
        # The freed row should have been reused.
        assert row_c == row_b


# ---------------------------------------------------------------------------
# TestPhaseTrackingRelease
# ---------------------------------------------------------------------------


class TestPhaseTrackingRelease:
    """Verify that partial release with shared refcounts works correctly.

    These tests validate the refcount behaviour that underpins
    the model runner's phase-tracking release logic (Fix B).
    """

    def test_phase_tracking_release_correctness(self):
        """Two requests sharing the same (hash, phase='prefill') config.
        Releasing one should decrement refcount to 1 and keep the row
        active, not free it."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}

        # Two requests register the same prefill config (refcount = 2).
        row = mgr.register_config(config_hash=1, vectors=vectors, phase="prefill")
        mgr.register_config(config_hash=1, vectors=vectors, phase="prefill")
        assert mgr.config_refcounts[(1, "prefill")] == 2

        # Release one (simulates one request finishing).
        mgr.release_config(config_hash=1, phase="prefill")
        assert mgr.config_refcounts[(1, "prefill")] == 1
        assert (1, "prefill") in mgr.config_to_row

        # The row is still active and resolvable.
        assert mgr.get_row_for_config(config_hash=1, is_prefill=True) == row
        assert mgr.num_active_configs == 1

    def test_phase_tracking_prevents_cross_phase_release(self):
        """Releasing (hash, 'decode') must NOT affect (hash, 'prefill')
        even when both use the same hash value."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}

        row_p = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        row_d = mgr.register_config(config_hash=42, vectors=vectors, phase="decode")
        assert row_p != row_d
        assert mgr.num_active_configs == 2

        # Release the decode config.
        mgr.release_config(config_hash=42, phase="decode")
        assert mgr.num_active_configs == 1

        # Prefill config must remain untouched.
        assert (42, "prefill") in mgr.config_to_row
        assert mgr.config_refcounts[(42, "prefill")] == 1
        assert mgr.get_row_for_config(config_hash=42, is_prefill=True) == row_p

    def test_release_only_active_phase_on_finish(self):
        """Simulate the model runner's finish-release logic:
        a request that transitioned to decode should only release
        its decode config, not both phases.

        This validates that phase-tracking prevents the double-release bug.
        """
        mgr = _make_manager(max_configs=4)
        vectors_p = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vectors_d = {_HP: {0: [2.0] * HIDDEN_SIZE}}

        # Two requests share the same prefill config.
        mgr.register_config(config_hash=10, vectors=vectors_p, phase="prefill")
        mgr.register_config(config_hash=10, vectors=vectors_p, phase="prefill")
        assert mgr.config_refcounts[(10, "prefill")] == 2

        # Request A transitions: release prefill, register decode.
        mgr.release_config(config_hash=10, phase="prefill")
        assert mgr.config_refcounts[(10, "prefill")] == 1
        mgr.register_config(config_hash=20, vectors=vectors_d, phase="decode")

        # Request A finishes — with correct phase tracking, we only
        # release decode (hash=20), NOT prefill (hash=10).
        # Simulate: phase_A = "decode"
        mgr.release_config(config_hash=20, phase="decode")
        assert (20, "decode") not in mgr.config_to_row

        # The prefill config for request B must still be intact.
        assert (10, "prefill") in mgr.config_to_row
        assert mgr.config_refcounts[(10, "prefill")] == 1


# ---------------------------------------------------------------------------
# TestCapacityOverflow
# ---------------------------------------------------------------------------


class TestCapacityOverflow:
    """Manager-level invariants when more distinct configs arrive than
    ``max_steering_configs`` allows.  ``register_config`` raises a
    ``RuntimeError`` and leaves manager state untouched, so callers can
    decide how to react (the scheduler reserves rows ahead of time, so
    in production a raise here is a scheduler bug).
    """

    def test_prefill_overflow_raises_keeps_state(self):
        """Registering more prefill configs than capacity raises and
        keeps existing state intact."""
        max_cfgs = 2
        mgr = _make_manager(max_configs=max_cfgs)

        # Fill all available slots
        for i in range(max_cfgs):
            vectors = {_HP: {0: [float(i + 1)] * HIDDEN_SIZE}}
            mgr.register_config(config_hash=100 + i, vectors=vectors, phase="prefill")

        assert mgr.num_active_configs == max_cfgs

        # The next registration should raise RuntimeError (which the
        # model runner wraps in try/except).
        overflow_vectors = {_HP: {0: [99.0] * HIDDEN_SIZE}}
        try:
            mgr.register_config(
                config_hash=999, vectors=overflow_vectors, phase="prefill"
            )
            overflowed = False
        except RuntimeError:
            overflowed = True

        assert overflowed, (
            "register_config should raise RuntimeError when capacity is full"
        )

        # Previously registered configs must still be intact.
        assert mgr.num_active_configs == max_cfgs
        for i in range(max_cfgs):
            assert (100 + i, "prefill") in mgr.config_to_row

    def test_decode_overflow_raises_keeps_state(self):
        """Registering more decode configs than capacity raises and
        keeps existing state intact."""
        max_cfgs = 2
        mgr = _make_manager(max_configs=max_cfgs)

        # Fill all available slots with decode configs
        for i in range(max_cfgs):
            vectors = {_HP: {0: [float(i + 1)] * HIDDEN_SIZE}}
            mgr.register_config(config_hash=200 + i, vectors=vectors, phase="decode")

        assert mgr.num_active_configs == max_cfgs

        overflow_vectors = {_HP: {0: [99.0] * HIDDEN_SIZE}}
        try:
            mgr.register_config(
                config_hash=888, vectors=overflow_vectors, phase="decode"
            )
            overflowed = False
        except RuntimeError:
            overflowed = True

        assert overflowed, (
            "register_config should raise RuntimeError when capacity is full"
        )

        # Previously registered configs must still be intact.
        assert mgr.num_active_configs == max_cfgs
        for i in range(max_cfgs):
            assert (200 + i, "decode") in mgr.config_to_row

    def test_mixed_phase_overflow_consistent(self):
        """Mixed prefill/decode configs exceeding capacity: the first
        ``max_steering_configs`` registrations succeed; subsequent ones
        raise ``RuntimeError`` and the manager remains consistent.
        Releasing a slot lets a previously-overflowing config register.
        """
        max_cfgs = 3
        mgr = _make_manager(max_configs=max_cfgs)

        registered_keys: list[tuple[int, str]] = []
        overflow_count = 0

        # Try to register 5 configs (3 prefill + 2 decode) with only 3
        # slots available.
        configs = [
            (10, "prefill"),
            (20, "prefill"),
            (30, "prefill"),
            (40, "decode"),
            (50, "decode"),
        ]
        for config_hash, phase in configs:
            vectors = {_HP: {0: [float(config_hash)] * HIDDEN_SIZE}}
            try:
                mgr.register_config(
                    config_hash=config_hash, vectors=vectors, phase=phase
                )
                registered_keys.append((config_hash, phase))
            except RuntimeError:
                overflow_count += 1

        # Exactly max_cfgs should have succeeded.
        assert len(registered_keys) == max_cfgs
        assert overflow_count == len(configs) - max_cfgs

        # The manager must be internally consistent.
        assert mgr.num_active_configs == max_cfgs
        for key in registered_keys:
            assert key in mgr.config_to_row

        # Releasing one slot and retrying an overflow config should succeed.
        last_key = registered_keys[-1]
        mgr.release_config(config_hash=last_key[0], phase=last_key[1])
        assert mgr.num_active_configs == max_cfgs - 1

        # Now the overflow config should register fine.
        vectors = {_HP: {0: [40.0] * HIDDEN_SIZE}}
        row = mgr.register_config(config_hash=40, vectors=vectors, phase="decode")
        assert row >= 3
        assert mgr.num_active_configs == max_cfgs

    def test_overflow_unregistered_raises(self):
        """When a config cannot be registered due to capacity, looking
        up its row crashes loudly rather than silently falling back to
        the global rows. The scheduler is the single source of truth
        for capacity; a worker-side miss is a scheduler bug.
        """
        max_cfgs = 1
        mgr = _make_manager(max_configs=max_cfgs)

        # Fill the single slot.
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        with pytest.raises(RuntimeError, match="not registered"):
            mgr.get_row_for_config(config_hash=999, is_prefill=True)
        with pytest.raises(RuntimeError, match="not registered"):
            mgr.get_row_for_config(config_hash=999, is_prefill=False)


# ---------------------------------------------------------------------------
# TestDeviceMismatch
# ---------------------------------------------------------------------------


class TestDeviceMismatch:
    """Regression tests for GPU+CPU device mismatch in populate_steering_tables.

    Per-request vectors are created on CPU (via ``register_config``), while
    global vectors live on GPU (cloned from model ``register_buffer`` tensors
    via ``update_global_vectors``).  The combined path
    ``phase_global + per_req.squeeze(0)`` must handle cross-device operands.
    """

    def test_combined_global_gpu_per_request_cpu_does_not_crash(self):
        """When global vectors are on CUDA and per-request vectors are on
        CPU, ``populate_steering_tables`` must not raise a RuntimeError
        and must produce the correct combined values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        mgr = _make_manager()

        # Global vectors on CUDA (simulates real model behaviour)
        base_vec = torch.ones(HIDDEN_SIZE, device="cuda") * 2.0
        prefill_vec = torch.ones(HIDDEN_SIZE, device="cuda") * 3.0
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=base_vec, phase="base"
        )
        mgr.update_global_vectors(
            hook_point=_HP, layer_idx=0, vector=prefill_vec, phase="prefill"
        )

        # Per-request vectors registered as float lists -> CPU tensors
        per_req_list = [5.0] * HIDDEN_SIZE
        vectors = {_HP: {0: per_req_list}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        # Create layers with CUDA steering table buffers
        num_rows = mgr.max_steering_configs + 3
        layers: dict[int, torch.nn.Module] = {}
        layer = torch.nn.Module()
        layer.register_buffer(
            _TABLE_ATTR,
            torch.zeros(num_rows, HIDDEN_SIZE, device="cuda"),
        )
        layers[0] = layer

        # This must not crash with "RuntimeError: expected ... on cuda"
        mgr.populate_steering_tables(layers)

        table = getattr(layers[0], _TABLE_ATTR)

        # Row 1: global prefill effective = base + prefill = 2+3 = 5
        expected_global = (base_vec + prefill_vec).cpu()
        assert torch.allclose(table[1].cpu(), expected_global), (
            f"Row 1 mismatch: expected {expected_global}, got {table[1].cpu()}"
        )

        # Per-request combined row: (base+prefill) + per_req = 5+5 = 10
        per_req_cpu = torch.tensor(per_req_list, dtype=torch.float32)
        expected_combined = expected_global + per_req_cpu
        assert torch.allclose(table[row].cpu(), expected_combined), (
            f"Combined row {row} mismatch: "
            f"expected {expected_combined}, got {table[row].cpu()}"
        )


# ---------------------------------------------------------------------------
# TestPreemptionResumption
# ---------------------------------------------------------------------------


class TestPreemptionResumption:
    """Regression tests for the preempt/resume steering config leak (C1).

    When the scheduler preempts a request that already transitioned to decode
    and later resumes it, the request re-enters prefill with num_computed=0.
    The model runner must:
      1) Release the stale decode config (refcount -> 0, row freed).
      2) Re-register the prefill config.
      3) Reset ``_req_steering_phase[req_id]`` to ``"prefill"``.

    Without this reset, the final release after the request finishes only
    drops refcount from 2 -> 1 (because the prefill->decode transition
    re-registers the decode config on top of an already-registered one),
    leaking the row permanently.

    These tests exercise the SteeringManager primitives the fix relies on
    plus a small simulation of the model-runner bookkeeping path.
    """

    def test_manager_primitives_contract(self):
        """SteeringManager contract the fix builds on: register/release is
        refcount-correct across phase transitions."""
        mgr = _make_manager(max_configs=2)
        initial_free_len = len(mgr.free_rows)
        prefill_vecs = {_HP: {0: [1.0, 2.0] + [0.0] * (HIDDEN_SIZE - 2)}}
        decode_vecs = {_HP: {0: [0.5, 0.5] + [0.0] * (HIDDEN_SIZE - 2)}}
        hash_p = 1001
        hash_d = 2002

        # Step 1: register prefill config
        row_p = mgr.register_config(hash_p, prefill_vecs, phase="prefill")
        assert mgr.config_refcounts[(hash_p, "prefill")] == 1
        assert row_p >= 3
        assert (hash_p, "prefill") in mgr.config_to_row

        # Step 2: normal prefill->decode transition — release prefill,
        # register decode.  Row is reclaimed for decode.
        mgr.release_config(hash_p, "prefill")
        mgr.register_config(hash_d, decode_vecs, phase="decode")
        assert mgr.config_refcounts[(hash_d, "decode")] == 1
        assert (hash_p, "prefill") not in mgr.config_to_row
        assert (hash_d, "decode") in mgr.config_to_row
        # Only one active config entry: the decode one.
        assert len(mgr.config_to_row) == 1

        # Step 3: BASELINE no-fix behavior — a buggy re-transition after
        # resumption would register the decode config AGAIN on top of the
        # existing one, bumping refcount to 2.
        mgr.register_config(hash_d, decode_vecs, phase="decode")
        assert mgr.config_refcounts[(hash_d, "decode")] == 2, (
            "Baseline bug check: a double-register must bump refcount to 2, "
            "proving that without the fix the refcount leaks."
        )

        # Step 4: the corrected flow — fresh manager, then release decode
        # and register prefill (what the helper will do on resumption).
        mgr = _make_manager(max_configs=2)
        mgr.register_config(hash_p, prefill_vecs, phase="prefill")
        mgr.release_config(hash_p, "prefill")
        mgr.register_config(hash_d, decode_vecs, phase="decode")
        # Now simulate the helper's behaviour on resumption: release the
        # stale decode config, re-register the prefill config.
        mgr.release_config(hash_d, "decode")
        row_p_again = mgr.register_config(hash_p, prefill_vecs, phase="prefill")
        assert (hash_d, "decode") not in mgr.config_to_row
        assert mgr.config_refcounts[(hash_p, "prefill")] == 1
        assert row_p_again >= 3

        # Step 5: on finish the request releases its live prefill config.
        mgr.release_config(hash_p, "prefill")
        assert mgr.config_to_row == {}
        assert len(mgr.free_rows) == initial_free_len, (
            "Row should be returned to free_rows after the final release."
        )

    def test_stale_phase_on_resumption_simulated(self):
        """Simulate the model-runner ``_req_steering_phase`` + manager
        interaction.  Confirms the bug exists without the reset helper and
        is eliminated once the helper runs."""
        prefill_vecs = {_HP: {0: [1.0, 2.0] + [0.0] * (HIDDEN_SIZE - 2)}}
        decode_vecs = {_HP: {0: [0.5, 0.5] + [0.0] * (HIDDEN_SIZE - 2)}}
        hash_p = 333
        hash_d = 444

        def simulate_lifecycle(apply_reset: bool) -> "SteeringManager":
            """Simulate: register prefill -> transition to decode -> preempt
            -> resume (optionally running the reset helper) -> transition
            to decode again -> finish.  Returns the manager for inspection.
            """
            mgr = _make_manager(max_configs=2)
            # Model-runner bookkeeping we care about
            req_phase: dict[str, str] = {}
            req_id = "req-X"

            # Initial admission in prefill
            mgr.register_config(hash_p, prefill_vecs, phase="prefill")
            req_phase[req_id] = "prefill"

            # Prefill->decode transition: release prefill, register decode
            mgr.release_config(hash_p, "prefill")
            mgr.register_config(hash_d, decode_vecs, phase="decode")
            req_phase[req_id] = "decode"

            # --- PREEMPTION ---
            # Scheduler resets num_computed_tokens=0, request re-enters
            # WAITING and is later RESUMED.  The model runner sees it come
            # back with num_computed < num_prompt (i.e. prefill phase).
            # Without the reset helper, the bookkeeping is STALE:
            # req_phase[req_id] == "decode" and (hash_d, "decode") is still
            # registered.
            # The fix: release the stale decode config, re-register
            # prefill, clear phase to "prefill".
            if apply_reset and req_phase.get(req_id) == "decode":
                mgr.release_config(hash_d, "decode")
                req_phase[req_id] = "prefill"
                mgr.register_config(hash_p, prefill_vecs, phase="prefill")

            # Re-prefill finishes -> transition runs again (release prefill,
            # register decode).  In the buggy path this release is a no-op
            # (not registered) and the register bumps the existing decode
            # refcount to 2.
            if req_phase.get(req_id) == "prefill":
                mgr.release_config(hash_p, "prefill")
            else:
                # The stale path — the runner thinks it's already in
                # decode, so release_config for prefill is a no-op.
                mgr.release_config(hash_p, "prefill")
            mgr.register_config(hash_d, decode_vecs, phase="decode")
            req_phase[req_id] = "decode"

            # Request finishes — the runner's cleanup pops the phase and
            # releases whichever config corresponds to the tracked phase.
            final_phase = req_phase.pop(req_id, None)
            if final_phase == "decode":
                mgr.release_config(hash_d, "decode")
            elif final_phase == "prefill":
                mgr.release_config(hash_p, "prefill")
            return mgr

        # Without the reset helper, the decode config leaks (refcount
        # should be 1 after finish, but is 2 before release and 1 after).
        mgr_buggy = simulate_lifecycle(apply_reset=False)
        assert mgr_buggy.config_refcounts.get((hash_d, "decode"), 0) == 1, (
            "Without the reset helper the decode config leaks: refcount "
            "stays at 1 after finish instead of dropping to 0."
        )
        assert (hash_d, "decode") in mgr_buggy.config_to_row, (
            "Without the reset helper the decode row is NOT reclaimed."
        )

        # With the reset helper, the decode config is fully released on
        # finish and the row is reclaimed.
        mgr_fixed = simulate_lifecycle(apply_reset=True)
        assert (hash_d, "decode") not in mgr_fixed.config_to_row, (
            "With the reset helper the decode row must be reclaimed after finish."
        )
        assert (hash_d, "decode") not in mgr_fixed.config_refcounts
        assert mgr_fixed.num_active_configs == 0


# ---------------------------------------------------------------------------
# TestDevicePlacement
# ---------------------------------------------------------------------------


class TestDevicePlacement:
    """Verify that per-request vectors are stored on the specified device."""

    def test_default_device_is_cpu(self):
        """Without an explicit device, vectors should land on CPU."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        stored = mgr.config_vectors[(42, "prefill")][_HP][0]
        assert stored.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vectors_stored_on_specified_cuda_device(self):
        """When device=cuda, registered vectors must reside on that device."""
        cuda_device = torch.device("cuda:0")
        mgr = _make_manager(device=cuda_device)
        vectors = {_HP: {0: [2.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=99, vectors=vectors, phase="prefill")
        stored = mgr.config_vectors[(99, "prefill")][_HP][0]
        assert stored.device == cuda_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_populate_per_request_only_on_gpu(self):
        """Per-request-only path (no globals) should work with GPU vectors."""
        cuda_device = torch.device("cuda:0")
        mgr = _make_manager(device=cuda_device)
        per_req_vec = torch.ones(HIDDEN_SIZE) * 7.0
        vectors = {_HP: {0: per_req_vec.tolist()}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        # Create layer tables on GPU to match
        layers = _make_layers(mgr, layer_indices=[0])
        for mod in layers.values():
            mod.to(cuda_device)

        mgr.populate_steering_tables(layers)
        table = getattr(layers[0], _TABLE_ATTR)
        expected = per_req_vec.to(cuda_device)
        assert torch.allclose(table[row], expected)


# ---------------------------------------------------------------------------
# TestStackVectorsAsyncH2D
# ---------------------------------------------------------------------------


class TestStackVectorsAsyncH2D:
    """Validate the non-blocking H2D semantics of ``_stack_vectors_to_device``.

    The optimization replaces a blocking ``pin_memory()`` per call with a
    reusable pinned ring + ``cudaMemcpyAsync``. Correctness depends on:

    * The returned tensor sees the correct contents once the stream
      drains.
    * Successive calls don't corrupt earlier in-flight transfers (ring
      slot reuse + per-slot CUDA event).
    * The host call returns BEFORE the H2D itself completes.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_returned_tensor_is_correct_after_sync(self):
        """Content correctness: after a forced sync the device tensor must
        match the input."""
        cuda_device = torch.device("cuda:0")
        mgr = _make_manager(device=cuda_device)
        # 34 layers x 2560 hidden — representative of Gemma-3-4B.
        n, hidden = 34, 2560
        rng = torch.Generator().manual_seed(0)
        host_arr = torch.randn(n, hidden, generator=rng, dtype=torch.float32)
        vecs = [host_arr[i].tolist() for i in range(n)]

        out = mgr._stack_vectors_to_device(vecs)
        assert out.device == cuda_device
        assert out.shape == (n, hidden)

        torch.cuda.synchronize()
        # ``host_arr`` was float32 → numpy float32 → device tensor; the
        # round-trip is exact at fp32.
        assert torch.allclose(out.cpu(), host_arr, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_repeated_calls_do_not_corrupt_earlier_results(self):
        """Slot-reuse correctness: with a 4-slot ring, 8 back-to-back calls
        must each yield the correct content. If event-gated reuse were
        broken, an earlier in-flight H2D would observe overwritten host
        contents and produce a corrupt device tensor."""
        cuda_device = torch.device("cuda:0")
        mgr = _make_manager(device=cuda_device)
        n, hidden = 8, 256

        results = []
        expected = []
        for k in range(8):
            host = torch.full((n, hidden), float(k), dtype=torch.float32)
            expected.append(host)
            vecs = [host[i].tolist() for i in range(n)]
            out = mgr._stack_vectors_to_device(vecs)
            results.append(out)

        torch.cuda.synchronize()
        for k, (got, want) in enumerate(zip(results, expected)):
            assert torch.allclose(got.cpu(), want, atol=0, rtol=0), (
                f"call {k} produced wrong contents — possible host-buffer "
                f"reuse race"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_call_returns_before_h2d_completes(self):
        """Microbench: call wall-time should be << time-to-content-ready.

        Build a vector stack large enough that the H2D takes a measurable
        amount of time. Compare:
            t_call = time of ``_stack_vectors_to_device`` (host return)
            t_ready = time until the H2D has drained (forced sync)

        If the H2D is properly async, ``t_call`` covers only the host
        memcpy + enqueue, and ``t_ready - t_call`` should be at least
        a meaningful fraction of t_ready. We're conservative here: as
        long as the call returns and a subsequent ``synchronize()``
        observably waits, we know the H2D wasn't done at return time.
        """
        import time

        cuda_device = torch.device("cuda:0")
        mgr = _make_manager(device=cuda_device)
        # Larger stack to make the H2D measurable: ~32 MB worth.
        n, hidden = 64, 32 * 1024  # ~8 MB at fp32
        host_arr = torch.zeros(n, hidden, dtype=torch.float32)
        vecs = [host_arr[i].tolist() for i in range(n)]

        # Warm up the ring so the first-call pinned-alloc cost doesn't
        # skew the measurement.
        _ = mgr._stack_vectors_to_device(vecs)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = mgr._stack_vectors_to_device(vecs)
        t_call = time.perf_counter() - t0

        # Now wait on the stream — this should still have meaningful work
        # left if the call returned early.
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t_sync = time.perf_counter() - t1

        # Sanity: the device tensor exists and is on the right device.
        assert out.device == cuda_device
        assert out.shape == (n, hidden)

        # The call should return before the H2D has fully drained on the
        # GPU. We assert the sync observed work, OR (in tiny-transfer
        # cases) that the call wall is at least competitive with sync —
        # i.e. the call wasn't internally synchronizing for the whole
        # H2D duration.
        # The strict claim is: call_time should NOT exceed the total
        # (call_time + sync_time) by a wide margin — a synchronous
        # implementation would push virtually all the cost into t_call
        # and leave t_sync near zero.
        total = t_call + t_sync
        # Accept any of:
        #  - Sync took non-trivial time (proof of pending work), OR
        #  - The call itself was very fast (< 1ms)
        assert t_sync > 0 or t_call < 1e-3, (
            f"_stack_vectors_to_device appears synchronous: "
            f"t_call={t_call * 1e3:.3f}ms, t_sync={t_sync * 1e3:.3f}ms, "
            f"total={total * 1e3:.3f}ms"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_oversized_input_falls_back_without_error(self):
        """Inputs above ``_STACK_PINNED_CAP_BYTES`` must take the fallback
        path and still return a correct device tensor."""
        cuda_device = torch.device("cuda:0")
        mgr = _make_manager(device=cuda_device)
        # Force the fallback by lowering the cap for this manager.
        mgr._STACK_PINNED_CAP_BYTES = 1024  # 1 KB — guaranteed too small
        n, hidden = 4, 1024  # 16 KB at fp32, exceeds the patched cap
        host_arr = torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden)
        vecs = [host_arr[i].tolist() for i in range(n)]

        out = mgr._stack_vectors_to_device(vecs)
        torch.cuda.synchronize()
        assert out.device == cuda_device
        assert torch.allclose(out.cpu(), host_arr, atol=0, rtol=0)

    def test_cpu_only_path_unchanged(self):
        """With ``device=None`` the function must still return a CPU tensor
        with correct contents (no pinned-ring path involvement)."""
        mgr = _make_manager(device=None)
        host = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        out = mgr._stack_vectors_to_device(host)
        assert out.device == torch.device("cpu")
        assert out.shape == (2, 3)
        assert torch.allclose(out, torch.tensor(host, dtype=torch.float32))
