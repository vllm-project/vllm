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

import torch
import torch.nn as nn

from vllm.model_executor.layers.steering import DEFAULT_HOOK_POINT
from vllm.v1.worker.steering_manager import SteeringManager

HIDDEN_SIZE = 8
MAX_CONFIGS = 4
_HP = DEFAULT_HOOK_POINT.value  # "post_mlp_pre_ln"
_TABLE_ATTR = "steering_table_post_mlp_pre_ln"


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

    def test_unregistered_nonzero_prefill_returns_row_1(self):
        """An unknown nonzero hash with is_prefill=True falls back to row 1."""
        mgr = _make_manager()
        assert mgr.get_row_for_config(config_hash=12345, is_prefill=True) == 1

    def test_unregistered_nonzero_decode_returns_row_2(self):
        """An unknown nonzero hash with is_prefill=False falls back to row 2."""
        mgr = _make_manager()
        assert mgr.get_row_for_config(config_hash=12345, is_prefill=False) == 2

    def test_registered_hash_uses_phase_for_lookup(self):
        """A prefill-registered hash is only found with is_prefill=True."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        row = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert mgr.get_row_for_config(config_hash=42, is_prefill=True) == row
        # With is_prefill=False the (42, "decode") key is absent, falls back
        assert mgr.get_row_for_config(config_hash=42, is_prefill=False) == 2


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
                if ph != 0 and (ph, "prefill") not in manager.config_to_row:
                    eff = req.get("effective_prefill")
                    if eff:
                        manager.register_config(ph, eff, phase="prefill")
            else:
                dh = req["decode_hash"]
                if dh != 0 and (dh, "decode") not in manager.config_to_row:
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

    def test_already_registered_config_not_duplicated(self):
        """Backfill skips configs already present in config_to_row."""
        mgr = _make_manager()
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        # Pre-register the config
        row_pre = mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")
        assert mgr.config_refcounts[(42, "prefill")] == 1

        # Backfill should skip because (42, "prefill") already registered
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
        # Refcount should still be 1 — backfill did not re-register
        assert mgr.config_refcounts[(42, "prefill")] == 1
        assert mgr.config_to_row[(42, "prefill")] == row_pre


# ---------------------------------------------------------------------------
# TestDeferredDecodeRegistration
# ---------------------------------------------------------------------------


class TestDeferredDecodeRegistration:
    """Verify that capacity exhaustion raises RuntimeError and that
    registrations succeed after capacity is freed.

    These tests validate the SteeringManager behaviour that underpins
    the model runner's deferred-registration logic (Fix A).
    """

    def test_deferred_decode_registration_on_capacity(self):
        """When max_configs slots are full, registering a new config
        raises RuntimeError — the model runner catches this to defer."""
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
# TestLazyInitCapacityExhaustion
# ---------------------------------------------------------------------------


class TestLazyInitCapacityExhaustion:
    """Simulate the lazy-init registration path where more distinct
    configs arrive in the first batch than ``max_steering_configs``
    allows.  Failed registrations are deferred into
    ``_pending_steering_registrations`` and retried with priority on the
    next step, matching the ``_handle_steering_transition`` pattern.

    This tests the SteeringManager's underlying behaviour (RuntimeError
    on capacity) and the global-row fallback used during the deferral
    period.
    """

    def test_prefill_overflow_does_not_crash(self):
        """Registering more prefill configs than capacity should not raise."""
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

    def test_decode_overflow_does_not_crash(self):
        """Registering more decode configs than capacity should not raise."""
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

    def test_mixed_phase_overflow_graceful(self):
        """Simulate a lazy-init batch with mixed prefill/decode configs
        exceeding capacity.  The first ``max_steering_configs`` registrations
        succeed; subsequent ones raise RuntimeError (caught by the
        model runner).  The manager remains consistent throughout.
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

    def test_overflow_falls_back_to_global_row(self):
        """When a config cannot be registered due to capacity, the
        ``get_row_for_config`` call should return the global row
        (1 for prefill, 2 for decode) rather than crash.
        """
        max_cfgs = 1
        mgr = _make_manager(max_configs=max_cfgs)

        # Fill the single slot.
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=42, vectors=vectors, phase="prefill")

        # Overflow config is NOT registered — get_row_for_config
        # must fall back to the global row.
        prefill_row = mgr.get_row_for_config(config_hash=999, is_prefill=True)
        decode_row = mgr.get_row_for_config(config_hash=999, is_prefill=False)

        assert prefill_row == 1, (
            "Unregistered prefill hash should fall back to global prefill row"
        )
        assert decode_row == 2, (
            "Unregistered decode hash should fall back to global decode row"
        )


# ---------------------------------------------------------------------------
# TestPendingSteering Registrations
# ---------------------------------------------------------------------------

_PendingEntry = tuple[str, int, dict[str, dict[int, list[float]]], str]


class TestPendingSteeringRegistrations:
    """Validate the unified deferred-registration retry loop that runs
    in ``_update_steering_buffers`` before table population.

    The loop must:
    - Drop entries for requests that have finished (prevents row leaks).
    - Drop entries whose request changed phase (e.g. prefill deferred
      but request already transitioned to decode).
    - Register entries whose requests are still alive and in the
      matching phase.
    - Keep entries that still can't register (capacity still full).
    """

    @staticmethod
    def _run_retry_loop(
        mgr: "SteeringManager",
        pending: list[_PendingEntry],
        live_requests: set[str],
        req_phases: dict[str, str],
    ) -> list[_PendingEntry]:
        """Simulate the retry loop from _update_steering_buffers."""
        still_pending: list[_PendingEntry] = []
        for d_req_id, d_hash, d_vecs, d_phase in pending:
            if d_req_id not in live_requests:
                continue
            if req_phases.get(d_req_id) != d_phase:
                continue
            try:
                mgr.register_config(d_hash, d_vecs, phase=d_phase)
            except RuntimeError:
                still_pending.append((d_req_id, d_hash, d_vecs, d_phase))
        return still_pending

    def test_finished_request_entry_dropped(self):
        """Deferred entry for a finished request must be silently dropped."""
        mgr = _make_manager(max_configs=2)
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}

        pending: list[_PendingEntry] = [
            ("req-gone", 42, vectors, "decode"),
        ]
        # req-gone is NOT in live_requests
        result = self._run_retry_loop(mgr, pending, live_requests=set(), req_phases={})
        assert result == []
        # Must NOT have been registered
        assert mgr.num_active_configs == 0

    def test_phase_changed_entry_dropped(self):
        """Deferred prefill entry dropped when request transitioned
        to decode."""
        mgr = _make_manager(max_configs=2)
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}

        pending: list[_PendingEntry] = [
            ("req-1", 42, vectors, "prefill"),
        ]
        # req-1 is alive but now in decode phase
        result = self._run_retry_loop(
            mgr,
            pending,
            live_requests={"req-1"},
            req_phases={"req-1": "decode"},
        )
        assert result == []
        assert mgr.num_active_configs == 0

    def test_active_request_registered_on_retry(self):
        """Deferred entry for an active, same-phase request registers
        successfully when capacity is available."""
        mgr = _make_manager(max_configs=2)
        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}

        pending: list[_PendingEntry] = [
            ("req-1", 42, vectors, "decode"),
        ]
        result = self._run_retry_loop(
            mgr,
            pending,
            live_requests={"req-1"},
            req_phases={"req-1": "decode"},
        )
        assert result == []
        assert mgr.num_active_configs == 1
        assert (42, "decode") in mgr.config_to_row

    def test_still_at_capacity_keeps_active_entries(self):
        """Active entries that still can't register stay in the pending
        list."""
        mgr = _make_manager(max_configs=1)
        blocker = {_HP: {0: [99.0] * HIDDEN_SIZE}}
        mgr.register_config(config_hash=1, vectors=blocker, phase="prefill")

        vectors = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        pending: list[_PendingEntry] = [
            ("req-1", 42, vectors, "decode"),
        ]
        result = self._run_retry_loop(
            mgr,
            pending,
            live_requests={"req-1"},
            req_phases={"req-1": "decode"},
        )
        assert len(result) == 1
        assert result[0] == ("req-1", 42, vectors, "decode")
        assert mgr.num_active_configs == 1  # only the blocker

    def test_mixed_active_and_stale_in_pending(self):
        """Only active, matching-phase entries survive the retry loop."""
        mgr = _make_manager(max_configs=2)
        vecs_a = {_HP: {0: [1.0] * HIDDEN_SIZE}}
        vecs_b = {_HP: {0: [2.0] * HIDDEN_SIZE}}
        vecs_c = {_HP: {0: [3.0] * HIDDEN_SIZE}}

        pending: list[_PendingEntry] = [
            ("req-finished", 10, vecs_a, "decode"),  # request gone
            ("req-transitioned", 20, vecs_b, "prefill"),  # now in decode
            ("req-active", 30, vecs_c, "decode"),  # still alive
        ]
        result = self._run_retry_loop(
            mgr,
            pending,
            live_requests={"req-transitioned", "req-active"},
            req_phases={
                "req-transitioned": "decode",
                "req-active": "decode",
            },
        )
        # Only req-active should have been processed
        assert result == []
        assert mgr.num_active_configs == 1
        assert (30, "decode") in mgr.config_to_row
        # The other two must NOT be registered
        assert (10, "decode") not in mgr.config_to_row
        assert (20, "prefill") not in mgr.config_to_row
