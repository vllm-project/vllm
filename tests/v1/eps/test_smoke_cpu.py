# ABOUTME: Smoke tests verifying EPS pre-pass invariants on CPU block tables.
# ABOUTME: Covers disabled gating and last_n>=groups behaviour.

import pytest
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch required for these tests
    pytest.skip("torch not available", allow_module_level=True)

from types import SimpleNamespace

from vllm.v1.eps.config import EpsRuntimeConfig
from vllm.v1.eps.runtime import build_eps_forward_context
from vllm.v1.eps.state import EpsJLState
from vllm.v1.eps.union_pass import run_union_prepass
from vllm.v1.worker.block_table import BlockTable


def _make_block_table(num_blocks: int, block_size: int = 2) -> BlockTable:
    table = BlockTable(
        block_size=block_size,
        max_num_reqs=1,
        max_num_blocks_per_req=num_blocks,
        max_num_batched_tokens=num_blocks * block_size,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=block_size,
    )
    table.add_row(list(range(num_blocks)), 0)
    return table


def _make_context(cfg: EpsRuntimeConfig, block_ids, frob_values) -> object:
    state = EpsJLState.init(
        num_layers=1,
        num_heads=1,
        head_dim=2,
        num_groups=len(frob_values),
        sketch_dim=2,
        device=torch.device("cpu"),
    )
    for idx, value in enumerate(frob_values):
        state.frob2[0, 0, idx] = torch.tensor(float(value))

    return build_eps_forward_context(
        cfg=cfg,
        layer_lookup={"layer0": (0, 0)},
        group_block_sizes=[cfg.group_blocks],
        request_ids=["req0"],
        request_states=[[state]],
        request_block_ids=[(block_ids,)],
        token_request_indices=None,
        cudagraph_capture=False,
    )


def _dummy_specs():
    return [SimpleNamespace(page_size_bytes=128)], [SimpleNamespace(layer_names=["layer0"])]


def test_eps_disabled_leaves_block_table_unchanged():
    table = _make_block_table(num_blocks=4)
    cfg = EpsRuntimeConfig(
        enabled=False,
        scope="union",
        method="off",
        head_scope="all",
        group_blocks=2,
        last_n=0,
        alpha=1.0,
        dim=2,
    )

    kv_specs, kv_groups = _dummy_specs()
    counters = run_union_prepass([table], kv_specs, kv_groups, cfg, num_reqs=1, eps_ctx=None)

    assert table.num_blocks_per_row[0] == 4
    assert counters.blocks_total == 0
    assert counters.blocks_kept == 0
    assert counters.pages_skipped == 0


def test_last_n_covers_all_groups_skips_no_blocks():
    table = _make_block_table(num_blocks=4)
    cfg = EpsRuntimeConfig(
        enabled=True,
        scope="union",
        method="jl",
        head_scope="all",
        group_blocks=2,
        last_n=2,  # >= total groups
        alpha=1.0,
        dim=2,
    )
    ctx = _make_context(cfg, block_ids=[0, 1, 2, 3], frob_values=[10.0, 1.0])

    kv_specs, kv_groups = _dummy_specs()
    counters = run_union_prepass([table], kv_specs, kv_groups, cfg, num_reqs=1, eps_ctx=ctx)

    assert counters.blocks_kept == counters.blocks_total
    assert counters.pages_skipped == 0


def test_top_pages_limits_visited_groups():
    table = _make_block_table(num_blocks=4)
    cfg = EpsRuntimeConfig(
        enabled=True,
        scope="union",
        method="jl",
        head_scope="all",
        group_blocks=2,
        last_n=0,
        alpha=1.0,
        dim=2,
        top_pages=1,
    )
    ctx = _make_context(cfg, block_ids=[0, 1, 2, 3], frob_values=[100.0, 0.1])

    kv_specs, kv_groups = _dummy_specs()
    counters = run_union_prepass([table], kv_specs, kv_groups, cfg, num_reqs=1, eps_ctx=ctx)

    assert table.num_blocks_per_row[0] == 2
    assert list(table.block_table.np[0, :2]) == [0, 1]
    assert counters.pages_skipped == 1
    assert counters.pages_unique_kept == 1
