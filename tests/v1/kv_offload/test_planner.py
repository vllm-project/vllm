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


def test_fixed_chunk_rejects_non_hash_aligned_size():
    with pytest.raises(ValueError, match="must be divisible by hash_block_size"):
        HybridOffloadPlanner(
            hash_block_size=16,
            gpu_block_sizes=(65536, 1056),
            fixed_chunk_size=10001,
        )


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
