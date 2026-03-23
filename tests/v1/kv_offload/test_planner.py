# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.planner import HybridOffloadPlanner


def test_fixed_chunk_marks_large_groups_as_partial():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 65536, 65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.offload_unit_sizes == (16384, 16384, 16384, 1056)
    assert planner.requires_partial_group_offload == (True, True, True, False)
    assert planner.group_hash_factors == (1024, 1024, 1024, 66)


def test_fixed_chunk_rejects_non_positive_size():
    with pytest.raises(ValueError, match="must be positive"):
        HybridOffloadPlanner(
            hash_block_size=16,
            gpu_block_sizes=(65536, 1056),
            fixed_chunk_size=0,
        )


def test_fixed_chunk_rejects_smaller_than_hash_block_size():
    with pytest.raises(
        ValueError, match="greater than or equal to hash_block_size"
    ):
        HybridOffloadPlanner(
            hash_block_size=1056,
            gpu_block_sizes=(65536, 1056),
            fixed_chunk_size=1024,
        )


def test_fixed_chunk_leaves_indivisible_large_groups_unsplit():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 50000, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.offload_unit_sizes == (16384, 50000, 1056)
    assert planner.requires_partial_group_offload == (True, False, False)


def test_storeable_prefix_uses_common_fully_covered_units():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 65536, 65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.storeable_prefix_tokens(10_000) == 0
    assert planner.storeable_prefix_tokens(16_384) == 15_840
    assert planner.storeable_prefix_tokens(20_000) == 16_384
    assert planner.storeable_prefix_tokens(33_000) == 32_736


def test_loadable_prefix_reconciles_existing_group_coverage():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 65536, 65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.loadable_prefix_tokens((16384, 16384, 16384, 15840)) == 15840
    assert planner.loadable_prefix_tokens((32768, 32768, 32768, 32736)) == 32736
    assert planner.loadable_prefix_tokens((16384, 0, 16384, 15840)) == 0


def test_planner_reports_partial_group_requirement():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.requires_partial_group_offload_any is True


def test_planner_allows_engine_hash_size_to_differ_from_hybrid_chunk():
    planner = HybridOffloadPlanner(
        hash_block_size=1056,
        gpu_block_sizes=(65536, 65536, 65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.offload_unit_sizes == (16384, 16384, 16384, 1056)
    assert planner.group_hash_factors == (None, None, None, 1)
    assert planner.chunk_prefix_tokens(1) == 15840


def test_chunk_prefix_tokens_uses_common_covered_prefix():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 65536, 65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.chunk_prefix_tokens(0) == 0
    assert planner.chunk_prefix_tokens(1) == 15840
    assert planner.chunk_prefix_tokens(2) == 32736
    assert planner.chunk_prefix_tokens(4) == 65472


def test_chunk_count_for_tokens_inverts_common_prefix_boundaries():
    planner = HybridOffloadPlanner(
        hash_block_size=16,
        gpu_block_sizes=(65536, 65536, 65536, 1056),
        fixed_chunk_size=16384,
    )

    assert planner.chunk_count_for_tokens(0) == 0
    assert planner.chunk_count_for_tokens(15839) == 0
    assert planner.chunk_count_for_tokens(15840) == 1
    assert planner.chunk_count_for_tokens(32735) == 1
    assert planner.chunk_count_for_tokens(32736) == 2
