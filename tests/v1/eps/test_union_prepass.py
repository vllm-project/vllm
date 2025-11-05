# ABOUTME: Smoke tests for the EPS union pre-pass on CPU tables.
# ABOUTME: Ensures counters and masking behave as expected with JL energy.

import pytest
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch required for these tests
    pytest.skip("torch not available", allow_module_level=True)

from types import SimpleNamespace

from vllm.config.eps import EpsConfig
from vllm.v1.eps.config import to_runtime_config
from vllm.v1.eps.runtime import build_eps_forward_context
from vllm.v1.eps.state import EpsJLState
from vllm.v1.eps.union_pass import run_union_prepass
from vllm.v1.worker.block_table import BlockTable


def _make_table(block_size: int, blocks: list[list[int]]) -> BlockTable:
    max_blocks = max(len(row) for row in blocks)
    table = BlockTable(
        block_size=block_size,
        max_num_reqs=len(blocks),
        max_num_blocks_per_req=max_blocks,
        max_num_batched_tokens=block_size * max_blocks,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=block_size,
    )
    for idx, row in enumerate(blocks):
        table.add_row(row, idx)
    return table


def _make_ctx(cfg: EpsConfig, block_ids, frob2_values):
    runtime_cfg = to_runtime_config(cfg)
    state = EpsJLState.init(
        num_layers=1,
        num_heads=1,
        head_dim=2,
        num_groups=len(frob2_values),
        sketch_dim=2,
        device=torch.device("cpu"),
    )
    for i, value in enumerate(frob2_values):
        state.frob2[0, 0, i] = torch.tensor(float(value))
    return runtime_cfg, build_eps_forward_context(
        cfg=runtime_cfg,
        layer_lookup={"layer0": (0, 0)},
        group_block_sizes=[cfg.group_blocks],
        request_ids=["req0"],
        request_states=[[state]],
        request_block_ids=[(block_ids,)],
        token_request_indices=None,
        cudagraph_capture=False,
    )


def test_union_prepass_drops_low_energy_groups():
    cfg = EpsConfig(enabled=True, method="jl", group_blocks=2, last_n=0, top_pages=1)
    runtime_cfg, ctx = _make_ctx(cfg, block_ids=[0, 1, 2, 3], frob2_values=[100.0, 1.0])
    table = _make_table(2, [[0, 1, 2, 3]])

    counters = run_union_prepass(
        [table],
        [SimpleNamespace(page_size_bytes=64)],
        [SimpleNamespace(layer_names=["layer0"])],
        runtime_cfg,
        num_reqs=1,
        eps_ctx=ctx,
    )

    assert counters.groups_total == 2
    assert counters.groups_kept == 1
    assert counters.pages_total == 2
    assert counters.pages_skipped == 1
    assert table.num_blocks_per_row[0] == 2
    row = table.block_table.np[0]
    assert list(row[:2]) == [0, 1]
    assert list(row[2:]) == [runtime_cfg.sentinel, runtime_cfg.sentinel]


def test_union_prepass_noop_when_last_n_covers_all():
    cfg = EpsConfig(enabled=True, method="jl", group_blocks=4, last_n=4)
    runtime_cfg, ctx = _make_ctx(cfg, block_ids=[10, 11, 12, 13], frob2_values=[5.0])
    table = _make_table(4, [[10, 11, 12, 13]])

    counters = run_union_prepass(
        [table],
        [SimpleNamespace(page_size_bytes=32)],
        [SimpleNamespace(layer_names=["layer0"])],
        runtime_cfg,
        num_reqs=1,
        eps_ctx=ctx,
    )

    assert counters.groups_dropped == 0
    assert counters.pages_skipped == 0
    assert table.num_blocks_per_row[0] == 4
