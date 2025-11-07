# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ABOUTME: Tests union mask behavior for EPS block filtering.
# ABOUTME: Ensures union-scoped EPS mask rewrites block tables correctly.

import numpy as np
import torch

from vllm.config.eps import EpsConfig
from vllm.v1.eps.block_filter import apply_union_mask
from vllm.v1.eps.config import to_runtime_config
from vllm.v1.eps.telemetry import EpsStepCounters, blocks_to_groups
from vllm.v1.worker.block_table import BlockTable


def _make_block_table(
    max_reqs: int, max_blocks: int, block_size: int = 16
) -> BlockTable:
    return BlockTable(
        block_size=block_size,
        max_num_reqs=max_reqs,
        max_num_blocks_per_req=max_blocks,
        max_num_batched_tokens=block_size * max_blocks,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=block_size,
    )


def test_apply_union_mask_drops_requested_groups():
    table = _make_block_table(max_reqs=1, max_blocks=4)
    table.add_row([10, 11, 20, 21], row_idx=0)

    before, after = apply_union_mask(
        block_table=table,
        row_idx=0,
        visit_groups={0},
        group_blocks=2,
        sentinel=-1,
    )

    assert before == 4
    assert after == 2
    assert table.num_blocks_per_row[0] == 2
    np.testing.assert_array_equal(table.block_table.np[0, :2], np.array([10, 11]))
    np.testing.assert_array_equal(table.block_table.np[0, 2:4], np.array([-1, -1]))


def test_apply_union_mask_keeps_all_groups_when_requested():
    table = _make_block_table(max_reqs=1, max_blocks=4)
    table.add_row([1, 2, 3, 4], row_idx=0)

    before, after = apply_union_mask(
        block_table=table,
        row_idx=0,
        visit_groups={0, 1},
        group_blocks=2,
    )

    assert before == after == 4
    np.testing.assert_array_equal(table.block_table.np[0, :4], np.array([1, 2, 3, 4]))


def test_apply_union_mask_handles_partial_groups():
    table = _make_block_table(max_reqs=1, max_blocks=5)
    table.add_row([100, 101, 200, 201, 202], row_idx=0)

    before, after = apply_union_mask(
        block_table=table,
        row_idx=0,
        visit_groups={0},
        group_blocks=3,
        sentinel=0,
    )

    assert before == 5
    assert after == 3
    np.testing.assert_array_equal(
        table.block_table.np[0, :3], np.array([100, 101, 200])
    )
    np.testing.assert_array_equal(table.block_table.np[0, 3:5], np.array([0, 0]))


def test_apply_union_mask_operates_per_row():
    table = _make_block_table(max_reqs=2, max_blocks=4)
    table.add_row([1, 2, 3, 4], row_idx=0)
    table.add_row([10, 11, 12, 13], row_idx=1)

    apply_union_mask(table, row_idx=0, visit_groups={0}, group_blocks=2)
    apply_union_mask(table, row_idx=1, visit_groups={0, 1}, group_blocks=2)

    assert table.num_blocks_per_row[0] == 2
    assert table.num_blocks_per_row[1] == 4
    np.testing.assert_array_equal(table.block_table.np[0, :2], np.array([1, 2]))
    np.testing.assert_array_equal(
        table.block_table.np[1, :4], np.array([10, 11, 12, 13])
    )


def test_blocks_to_groups_rounds_up():
    assert blocks_to_groups(0, 3) == 0
    assert blocks_to_groups(1, 3) == 1
    assert blocks_to_groups(3, 3) == 1
    assert blocks_to_groups(4, 3) == 2


def test_eps_step_counters_aggregates_metrics():
    counters = EpsStepCounters()
    counters.blocks_total += 4
    counters.blocks_kept += 2
    counters.groups_total += 2
    counters.groups_kept += 1
    counters.unique_blocks_total += 3
    counters.unique_blocks_kept += 2
    counters.kv_bytes_total += 192
    counters.kv_bytes_kept += 128

    assert counters.blocks_total == 4
    assert counters.blocks_kept == 2
    assert counters.blocks_dropped == 2
    assert counters.groups_total == 2
    assert counters.groups_kept == 1
    assert counters.groups_dropped == 1
    assert counters.unique_blocks_total == 3
    assert counters.unique_blocks_kept == 2
    assert counters.unique_blocks_dropped == 1
    assert counters.kv_bytes_total == 192
    assert counters.kv_bytes_kept == 128
    assert counters.kv_bytes_saved == 64


def test_to_runtime_config_passes_through_defaults():
    cfg = EpsConfig()
    runtime = to_runtime_config(cfg)
    assert runtime.enabled is False
    assert runtime.scope == "union"
    assert runtime.group_blocks == cfg.group_blocks
    assert runtime.last_n == cfg.last_n
    assert runtime.alpha == cfg.alpha
    assert runtime.metrics_path is None
