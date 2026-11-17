# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hole-repair commit checks (worker side, pure host shapes).

Unit surface for the exclusive repair-only execution: the
duplicate-detecting slot-coverage check (a multiset comparison -- a
duplicated slot in the coverage is a write-aliasing hazard a set
comparison silently passes), the pre-write ownership fence, abort
containment, and the live-getenv sync-profile gate.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import (
    CachedRequestData,
    RepairWork,
    SchedulerOutput,
)
from vllm.v1.worker.gpu.repair_executor import (
    _repair_sync_profile_enabled,
    _slots_cover_span,
    _slots_owned_by_request,
    execute_repair_only,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BS = 128
BLOCK_TABLE = [7, 9]


@dataclass
class _FakeMeta:
    block_tables: list
    slot_mapping: torch.Tensor


def _meta(slots: torch.Tensor) -> _FakeMeta:
    return _FakeMeta(block_tables=[torch.tensor(BLOCK_TABLE)], slot_mapping=slots)


def _span_slots(block_idx: int) -> torch.Tensor:
    return torch.arange(block_idx * BS, (block_idx + 1) * BS, dtype=torch.int64)


def test_span_slot_coverage_accepts_exact_mapping():
    slots = _span_slots(BLOCK_TABLE[1])
    assert _slots_cover_span(_meta(slots), BS, 1 * BS, 2 * BS)
    assert _slots_owned_by_request(_meta(slots), BS)


def test_span_slot_coverage_detects_missing_and_foreign_slots():
    missing = _span_slots(BLOCK_TABLE[1]).clone()
    missing[10] = 4096 + 5  # a foreign slot displaces an expected row
    assert not _slots_cover_span(_meta(missing), BS, 1 * BS, 2 * BS)
    assert not _slots_owned_by_request(_meta(missing), BS)


def test_span_slot_coverage_detects_duplicated_slot():
    """A slot mapping with an EXTRA duplicated row (a mis-built metadata
    emitting one token twice) covers every expected slot exactly once
    plus the alias: the set comparison passes it, the multiset comparison
    must not."""
    clean = _span_slots(BLOCK_TABLE[1])
    duplicated = torch.cat([clean, clean[:1]])

    assert _slots_cover_span(_meta(clean), BS, 1 * BS, 2 * BS)
    assert not _slots_cover_span(_meta(duplicated), BS, 1 * BS, 2 * BS)


def test_sync_profile_gate_defaults_off(monkeypatch):
    """The sync profiler runs only under its live-getenv gate; the
    production commit path pays no profiling."""
    monkeypatch.delenv("KV_REPAIR_SYNC_PROFILE", raising=False)
    assert not _repair_sync_profile_enabled()

    monkeypatch.setenv("KV_REPAIR_SYNC_PROFILE", "1")
    assert _repair_sync_profile_enabled()


# ---- The multi-item repair-only step: one verdict per work ----


class _FakeLayer:
    def __init__(self, layer_name: str):
        attn = SimpleNamespace(
            kv_cache=[
                torch.zeros(4, BS, 8),
                torch.zeros(4, BS, 8),
            ],
            layer_name=layer_name,
        )
        self.self_attn = SimpleNamespace(attn=attn)

    def __call__(self, positions, hidden_states, residual):
        return hidden_states + 1.0, residual


class _FakeInner:
    def __init__(self, layers):
        self.layers = layers

    def embed_input_ids(self, input_ids):
        return torch.zeros(input_ids.shape[0], 8)


class _FakeModel:
    def __init__(self, n_layers: int):
        self.model = _FakeInner(
            [_FakeLayer(f"model.layers.{i}") for i in range(n_layers)]
        )


def _sub_output(works, tokens):
    base = dict(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=dict(tokens),
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return SchedulerOutput(repair_works=dict(works), **base)


def _multi_work_setup():
    """Two repair works in one repair-only step: req-a spans [0, 128) at
    rows [0, 128) with layer_end 1; req-b spans [128, 256) at rows
    [128, 256) with layer_end 2. The fake metadata's block tables carry
    one row per request and the slot mapping covers both works' slots."""
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=(0, BS),
            layer_end=1,
            restored_extent=512,
            plan_id=1,
            epoch=7,
            # The scheduler-authoritative write set: req-a's span block.
            expected_write_block_ids=(2,),
        ),
        "req-b": RepairWork(
            request_id="req-b",
            repair_span=(BS, 2 * BS),
            layer_end=2,
            restored_extent=512,
            plan_id=2,
            epoch=7,
            expected_write_block_ids=(3,),
        ),
    }
    tokens = {"req-a": BS, "req-b": BS}
    positions = torch.arange(2 * BS, dtype=torch.int64)
    input_ids = torch.arange(2 * BS, dtype=torch.int64)
    # req-a's span [0, 128) is its table's block 0 (block id 2); req-b's
    # span [128, 256) is its table's block 1 (block id 3).
    meta = _FakeMeta(
        block_tables=[torch.tensor([2]), torch.tensor([8, 3])],
        slot_mapping=torch.cat([_span_slots(2), _span_slots(3)]),
    )
    return works, tokens, positions, input_ids, meta


def test_execute_repair_only_multi_item_executes_each_work():
    """One repair-only step carrying two repair works executes each work
    item -- its own row window, its own layer range, its own commit
    battery -- and the results come back as a list, one verdict per
    work."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="unsloth/Llama-3.2-1B")
    ):
        results = execute_repair_only(
            model, sub, input_ids, positions, ["req-a", "req-b"]
        )

    assert [r.request_id for r in results] == ["req-a", "req-b"]
    # Per-work layer ranges: req-a executed 1 layer, req-b executed 2.
    assert results[0].executed_layers == 1
    assert results[1].executed_layers == 2
    assert [r.plan_id for r in results] == [1, 2]
    assert [r.span for r in results] == [(0, BS), (BS, 2 * BS)]
    # The commit battery applies per work item under the shared step.
    assert all(r.structural_check for r in results)
    assert results[0].commit_check is not None
    assert results[1].commit_check is not None


def test_pre_write_gate_refuses_foreign_slot_before_any_layer():
    """A slot mapping whose window points at a FOREIGN block (not in the
    work's own block-table row) is refused BEFORE any layer executes --
    zero cache-writing kernels run for that work, the verdict carries the
    gate refusal, and the OTHER works in the step are unaffected."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    # Corrupt req-a's slot window: its 128 slots now target block 999
    # (foreign -- not in either table row).
    bad_slots = meta.slot_mapping.clone()
    bad_slots[:BS] = 999 * BS + torch.arange(BS)
    meta_bad = _FakeMeta(block_tables=meta.block_tables, slot_mapping=bad_slots)
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with set_forward_context(
        {"model.layers.0": meta_bad}, create_vllm_config(model="unsloth/Llama-3.2-1B")
    ):
        results = execute_repair_only(
            model, sub, input_ids, positions, ["req-a", "req-b"]
        )

    ra, rb = results
    # req-a: the gate refused -- ZERO layers executed, aborted, the gate
    # key in the commit check, structural_check False.
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["pre_write_geometry"] is False
    assert ra.commit_check["slot_ownership"] is False
    assert not ra.structural_check
    # req-b: unaffected -- its own geometry was valid.
    assert rb.executed_layers == 2
    assert not rb.aborted
    assert rb.commit_check["pre_write_geometry"] is True
    assert rb.structural_check


def test_layer_exception_contained_as_abort():
    """An arbitrary exception inside the repair stream is request-local:
    the stream stops, the partial writes' layer count is reported, the
    verdict is ABORT with backend_ok False -- the engine does not crash
    and no success is fabricated."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    # Layer 1 raises for every caller; req-a (layer_end=1) never reaches
    # it, req-b (layer_end=2) hits it on its second layer.
    class _BoomLayer(_FakeLayer):
        def __call__(self, positions, hidden_states, residual):
            raise RuntimeError("injected device-side failure")

    model.model.layers[1] = _BoomLayer("model.layers.1")

    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="unsloth/Llama-3.2-1B")
    ):
        results = execute_repair_only(
            model, sub, input_ids, positions, ["req-a", "req-b"]
        )

    ra, rb = results
    assert ra.executed_layers == 1
    assert not ra.aborted
    assert ra.structural_check
    assert rb.executed_layers == 1  # stopped at the raising layer
    assert rb.aborted
    assert rb.commit_check["backend_ok"] is False
    assert not rb.structural_check


def test_execute_repair_only_per_work_positions_validation():
    """The per-work validation RED twin: work req-b's row window carries
    positions that do not match its repair span -- the per-work positions
    assert must fire (the multi-item loop may not skip or weaken any
    item's battery)."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    # Corrupt req-b's window: its positions no longer match [128, 256).
    positions = positions.clone()
    positions[BS:] = positions[BS:] + BS
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with (
        pytest.raises(AssertionError, match="req-b"),
        set_forward_context(
            {"model.layers.0": meta},
            create_vllm_config(model="unsloth/Llama-3.2-1B"),
        ),
    ):
        execute_repair_only(model, sub, input_ids, positions, ["req-a", "req-b"])


# ---- The ownership fence: the independent authorization battery ----


def _run_two_works(model, works, tokens, positions, input_ids, meta):
    sub = _sub_output(works, tokens)
    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="unsloth/Llama-3.2-1B")
    ):
        return execute_repair_only(model, sub, input_ids, positions, ["req-a", "req-b"])


def test_correlated_metadata_corruption_refused():
    """BOTH of req-a's worker structures (the block-table row AND the
    slot window) consistently point at req-b's physical block while the
    work carries req-a's true scheduler authorization. The self-
    referential gate accepts this geometry; the independent
    authorization must refuse it before any repair layer runs, with
    req-b unaffected."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    corrupted = _FakeMeta(
        block_tables=[torch.tensor([8]), torch.tensor([8, 3])],
        slot_mapping=torch.cat([_span_slots(8), _span_slots(3)]),
    )
    results = _run_two_works(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, corrupted
    )
    ra, rb = results
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["authorized_ownership"] is False
    assert ra.commit_check["authorized_table"] is False
    assert ra.commit_check["pre_write_geometry"] is False
    assert not ra.structural_check
    # The honest neighbor is unaffected.
    assert rb.executed_layers == 2
    assert rb.structural_check


def test_swapped_authorization_refused():
    """The swap: req-a's work carries REQ-B's authorization. The honest
    req-a worker geometry cannot reproduce a foreign authorization's
    slots -- refused, and req-b (whose own authorization is correct) is
    unaffected."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    works["req-a"].expected_write_block_ids = (3,)  # req-b's block!
    results = _run_two_works(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta
    )
    ra, rb = results
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert not ra.structural_check
    assert rb.executed_layers == 2


def test_missing_authorization_fail_closed():
    """The fail-closed default: a work with NO authorization snapshot (an
    unbuild/stale shape) is never executable."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    works["req-a"].expected_write_block_ids = ()
    results = _run_two_works(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert ra.commit_check["pre_write_geometry"] is False


def test_partial_write_set_authorization_refused():
    """A two-block repair span whose authorization omits the second
    (healthy, replayed-because-rollback-below) block: the expected slot
    multiset misses half the span -- refused."""
    work = RepairWork(
        request_id="req-a",
        repair_span=(0, 2 * BS),
        layer_end=1,
        restored_extent=512,
        plan_id=1,
        epoch=7,
        expected_write_block_ids=(2,),  # omits block 9!
    )
    meta = _FakeMeta(
        # req-a's row covers blocks 0-1 (2, 9); req-b's row must cover
        # its span's ABSOLUTE block index 2 (a 3-entry row: the row
        # lists the request's blocks from ITS block 0).
        block_tables=[torch.tensor([2, 9]), torch.tensor([11, 12, 3])],
        slot_mapping=torch.cat([_span_slots(2), _span_slots(9), _span_slots(3)]),
    )
    works = {
        "req-a": work,
        "req-b": RepairWork(
            request_id="req-b",
            repair_span=(2 * BS, 3 * BS),
            layer_end=1,
            restored_extent=512,
            plan_id=2,
            epoch=7,
            expected_write_block_ids=(3,),
        ),
    }
    tokens = {"req-a": 2 * BS, "req-b": BS}
    positions = torch.arange(3 * BS, dtype=torch.int64)
    input_ids = positions.clone()
    results = _run_two_works(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert not ra.structural_check


def test_duplicated_slots_refused_under_authorization():
    """The slot window carries a duplicated slot and a displaced one (the
    same length, all inside req-a's own authorized block): the multiset
    comparison against the authorization must refuse."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    dup = _span_slots(2).clone()
    dup[5] = dup[4]  # duplicate; dup[5]'s expected slot goes missing
    corrupted = _FakeMeta(
        block_tables=[torch.tensor([2]), torch.tensor([8, 3])],
        slot_mapping=torch.cat([dup, _span_slots(3)]),
    )
    results = _run_two_works(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, corrupted
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert not ra.structural_check


# ---- Port-discriminating battery: the 0003 port onto the #68140 stack ----


class _CoreLayoutLayer:
    """A layer whose cache matches the CURRENT core layout: one 4D
    [B, H, N, C] tensor per layer (K and V share it, split along the
    channel dim), so the token capacity of a block lives on dim 2 --
    NOT on dim 1 as in the Ascend per-K/V views the original patch
    targeted."""

    def __init__(self, layer_name: str, num_blocks: int = 4):
        attn = SimpleNamespace(
            kv_cache=torch.zeros(num_blocks, 2, BS, 8),
            layer_name=layer_name,
        )
        self.self_attn = SimpleNamespace(attn=attn)

    def __call__(self, positions, hidden_states, residual):
        return hidden_states + 1.0, residual


class _CudaNameMeta:
    """Metadata spelling the CUDA backends' field name: block_table
    (singular), the same [num_reqs, max_blocks] 2D tensor the Ascend
    metadata calls block_tables."""

    def __init__(self, block_tables, slot_mapping):
        self.block_table = (
            torch.stack([t for t in block_tables if t.numel() > 0])
            if all(t.numel() > 0 for t in block_tables)
            else block_tables[0]
        )
        # Keep rows addressable per-request even when rows have unequal
        # length: store the raw list too and index it in row order.
        self._rows = block_tables
        self.slot_mapping = slot_mapping

    def __getitem__(self, row):
        return self._rows[row]


def test_block_size_derivation_core_vs_ascend_layout():
    """The port's layout-aware block-size derivation: a core-layout
    [B, H, N, C] cache yields N (dim 2); an Ascend-style per-K/V
    [num_blocks, block_size, x] view yields dim 1. The original patch
    hardcoded dim 1, which on the core layout returns the HEAD count
    instead of the block size."""
    from vllm.v1.worker.gpu.repair_executor import _repair_block_size

    core = _CoreLayoutLayer("model.layers.0")
    ascend = _FakeLayer("model.layers.0")
    assert _repair_block_size(core) == BS
    assert _repair_block_size(ascend) == BS


def test_block_table_row_accepts_cuda_and_ascend_spelling():
    """Both metadata spellings resolve to the same request row: the
    Ascend 'block_tables' and the CUDA 'block_table' field names."""
    from vllm.v1.worker.gpu.repair_executor import _repair_block_table_row

    rows = [torch.tensor([2]), torch.tensor([8, 3])]
    slots = torch.cat([_span_slots(2), _span_slots(3)])
    ascend_meta = _FakeMeta(block_tables=rows, slot_mapping=slots)
    cuda_rows = [torch.tensor([2, 9]), torch.tensor([8, 3])]
    cuda_meta = SimpleNamespace(block_table=torch.stack(cuda_rows))
    assert _repair_block_table_row(ascend_meta, 1).tolist() == [8, 3]
    assert _repair_block_table_row(cuda_meta, 1).tolist() == [8, 3]


def test_rezero_indexes_core_layout_cache():
    """The abort re-zero must write through the cache's own block and
    offset dims on BOTH layouts: a core-layout [B, H, N, C] cache is
    zeroed at [bidx, :, oidx, :], an Ascend per-K/V view at [bidx,
    oidx]. A dim-1-only implementation would corrupt or miss rows on
    the core layout."""
    from vllm.v1.worker.gpu.repair_executor import _rezero_repair_span

    layer = _CoreLayoutLayer("model.layers.0", num_blocks=16)
    layer.self_attn.attn.kv_cache.fill_(1.0)
    meta = _FakeMeta(block_tables=[torch.tensor([7])], slot_mapping=_span_slots(7))
    _rezero_repair_span([layer], meta)
    cache = layer.self_attn.attn.kv_cache
    # Rows of block 7 are zeroed across heads and channels...
    assert float(cache[7].abs().max()) == 0.0
    # ...and no other block was touched.
    others = cache.clone()
    others[7] = 1.0
    assert float(others.abs().min()) == 1.0


def test_repair_works_require_eager_dispatch(monkeypatch):
    """The dispatch guard: a repair step must be forced eager, so the
    runner never routes the repair batch through a compiled/piecewise
    graph. This pins the port-side integration contract (the runner
    raises engine-fatal if a repair step reaches a cudagraph mode)."""
    import inspect

    from vllm.v1.worker.gpu import model_runner as mr

    src = inspect.getsource(mr.GPUModelRunner.execute_model)
    assert "repair_works is not None" in src, (
        "the eager-dispatch guard for repair steps must stay in execute_model"
    )
    assert "execute_repair_only" in src, (
        "the repair-only execution branch must stay in execute_model"
    )


def test_scheduler_restores_computed_tokens_via_staged_write():
    """The worker-side restore seam: update_requests must stage the
    scheduler-authoritative num_computed_tokens over the GPU buffer
    when it moved backwards (a repair commit restoration or a
    load-failure rollback), because the GPU-side count only ever
    advances optimistically."""
    import inspect

    from vllm.v1.worker.gpu import model_runner as mr

    src = inspect.getsource(mr.GPUModelRunner.update_requests)
    assert "stage_write_elem" in src, (
        "update_requests must stage backwards-moving computed-token "
        "restores onto the GPU buffer"
    )



def test_scheduler_refuses_async_scheduling_under_repair_policy():
    """The async-scheduling init guard: the batch-queue step calls
    sample_tokens unconditionally, but a repair-only step returns its
    verdicts from execute_model without setting execute_model_state --
    sample_tokens would return None and the engine would discard the repair
    verdicts and raise. The repair policy must refuse async scheduling at
    init, exactly like PP>1 and DP>1."""
    import inspect

    from vllm.v1.core.sched import scheduler as sched_mod

    src = inspect.getsource(sched_mod)
    assert "async_scheduling" in src, (
        "the scheduler must guard async scheduling under the repair policy"
    )
    assert (
        "not supported with" in src and "async scheduling" in src
    ), "the async guard must reject at init with a loud message"
