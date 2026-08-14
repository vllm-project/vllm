# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.benchmarks.weight_offload import planner


def make_manifest(rank: int = 0, **overrides) -> dict:
    """Six positions, of which 1, 3 and 5 are offloadable.

    The recorded run uses 2/1/1, so it selects those three. Unit indices
    deliberately differ from module indices, which is where a hand conversion
    goes wrong.
    """
    manifest = {
        "schema_version": 1,
        "rank": rank,
        "module_count": 6,
        "config": {"group_size": 2, "num_in_group": 1, "prefetch_step": 1},
        "positions": [
            {"module_index": 0, "offloadable": False, "logical_parameter_bytes": 0},
            {"module_index": 1, "offloadable": True, "logical_parameter_bytes": 100},
            {"module_index": 2, "offloadable": False, "logical_parameter_bytes": 0},
            {"module_index": 3, "offloadable": True, "logical_parameter_bytes": 100},
            {"module_index": 4, "offloadable": False, "logical_parameter_bytes": 0},
            {"module_index": 5, "offloadable": True, "logical_parameter_bytes": 100},
        ],
        "units": [
            {"unit_idx": 0, "module_index": 1, "direct_runtime_buffer_bytes": 0},
            {"unit_idx": 1, "module_index": 3, "direct_runtime_buffer_bytes": 0},
            {"unit_idx": 2, "module_index": 5, "direct_runtime_buffer_bytes": 0},
        ],
        "pooled_buffer_layouts": [
            {"layout_id": "slab", "bytes_per_slot": 40, "unit_indices": [0, 1, 2]},
        ],
    }
    manifest.update(overrides)
    return manifest


@pytest.fixture
def profile() -> planner.RunProfile:
    return planner.profile_from_manifest(make_manifest())


def test_profile_maps_buffers_from_unit_to_module_index():
    manifest = make_manifest()
    manifest["pooled_buffer_layouts"] = [
        {"layout_id": "a", "bytes_per_slot": 40, "unit_indices": [0]},
        {"layout_id": "b", "bytes_per_slot": 70, "unit_indices": [1, 2]},
    ]
    profile = planner.profile_from_manifest(manifest)
    by_module = {item.module_index: item for item in profile.positions}
    # Unit 0 is module 1 and unit 1 is module 3, not modules 0 and 1.
    assert by_module[1].pooled_bytes_per_slot == 40
    assert by_module[3].pooled_bytes_per_slot == 70
    assert by_module[0].pooled_bytes_per_slot is None


def test_selection_skips_positions_the_selector_misses(profile):
    # 1/1/x would take every position, but only three are offloadable.
    assert planner.selected_positions(profile, planner.Schedule(1, 1, 1)) == (1, 3, 5)
    assert planner.selected_positions(profile, planner.Schedule(2, 1, 1)) == (1, 3, 5)
    assert planner.selected_positions(profile, planner.Schedule(3, 1, 1)) == (5,)


def test_the_recorded_schedule_is_its_own_reference_point(profile):
    """Scoring the run against itself has to come out at zero."""
    candidate = planner.evaluate(profile, profile.schedule)
    assert candidate.resident_delta_bytes == 0
    assert candidate.offloaded_bytes == 300
    assert candidate.runtime_buffer_bytes == 40


def test_a_wider_window_holds_more_buffers_and_frees_less(profile):
    candidate = planner.evaluate(profile, planner.Schedule(2, 1, 3))
    # Same three units, but three slots of the one layout instead of one.
    assert candidate.offloaded_bytes == 300
    assert candidate.runtime_buffer_bytes == 120
    assert candidate.resident_delta_bytes == 80


def test_units_alone_in_a_slot_cost_no_steady_state_transfer(profile):
    """Three units in three slots stay resident after the first prefetch."""
    resident = planner.evaluate(profile, planner.Schedule(2, 1, 3))
    assert resident.h2d_bytes_per_forward == 0
    shared = planner.evaluate(profile, planner.Schedule(2, 1, 1))
    assert shared.h2d_bytes_per_forward == 300


def test_slot_reuse_matches_the_runtime_for_a_non_divisible_window():
    assert planner.prefetch_after_units(20, 3)[:4] == (3, 4, 5, 6)
    assert planner.prefetch_after_units(20, 3)[17:] == (2, 0, 1)
    assert planner.prefetch_after_units(3, 3) == (None, None, None)


def test_enumeration_reports_each_distinct_outcome_once(profile):
    """Schedules the tool cannot tell apart are one row, with a count."""
    candidates = planner.enumerate_candidates(profile, max_prefetch=3)
    outcomes = [
        (item.resident_delta_bytes, item.h2d_bytes_per_forward) for item in candidates
    ]
    assert outcomes and len(outcomes) == len(set(outcomes))
    # Collapsing must not lose schedules, only fold them into a count.
    assert sum(item.equivalent_schedules for item in candidates) > len(candidates)


def _heterogeneous_manifest() -> dict:
    """A run that offloads 2, 5 and 8 using two different layouts.

    Positions 0 and 1 are offloadable but were never selected, so nothing in
    the manifest says which layout they would land in.
    """
    return {
        "schema_version": 1,
        "rank": 0,
        "module_count": 9,
        "config": {"group_size": 3, "num_in_group": 1, "prefetch_step": 1},
        "positions": [
            {
                "module_index": index,
                "offloadable": index in {0, 1, 2, 5, 8},
                "logical_parameter_bytes": 100 if index in {0, 1, 2, 5, 8} else 0,
            }
            for index in range(9)
        ],
        "units": [
            {"unit_idx": 0, "module_index": 2, "direct_runtime_buffer_bytes": 0},
            {"unit_idx": 1, "module_index": 5, "direct_runtime_buffer_bytes": 0},
            {"unit_idx": 2, "module_index": 8, "direct_runtime_buffer_bytes": 0},
        ],
        "pooled_buffer_layouts": [
            {"layout_id": "a", "bytes_per_slot": 40, "unit_indices": [0]},
            {"layout_id": "b", "bytes_per_slot": 70, "unit_indices": [1, 2]},
        ],
    }


def test_refuses_to_guess_buffers_for_an_unrecorded_position():
    """With two layouts in play, an unseen position could be either."""
    profile = planner.profile_from_manifest(_heterogeneous_manifest())
    # The recorded schedule stays answerable.
    assert planner.evaluate(profile, profile.schedule).runtime_buffer_bytes == 110
    # 1/1/1 also pulls in positions 0 and 1, which the run never offloaded.
    with pytest.raises(ValueError, match="not offloaded in the recorded run"):
        planner.evaluate(profile, planner.Schedule(1, 1, 1))


def test_reads_a_manifest_out_of_server_log_output():
    payload = json.dumps(make_manifest(), separators=(",", ":"))
    log = (
        "INFO 08-10 00:00:00 [prefetch.py:1] starting\n"
        f"INFO 08-10 00:00:01 [x.py:1] [PrefetchOffloader] manifest_json={payload}\n"
        "INFO 08-10 00:00:02 [prefetch.py:2] ready\n"
    )
    manifests = planner.iter_manifests(log)
    assert len(manifests) == 1
    assert manifests[0]["module_count"] == 6


def test_ranks_that_disagree_require_an_explicit_choice():
    """Schedules are rank-local, so silently taking the first would mislead."""
    other = make_manifest(rank=1)
    other["positions"][1]["logical_parameter_bytes"] = 999
    with pytest.raises(ValueError, match="--rank"):
        planner.select_manifest([make_manifest(), other], None)
    assert planner.select_manifest([make_manifest(), other], 1)["rank"] == 1
