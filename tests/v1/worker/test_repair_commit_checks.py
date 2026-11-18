# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hole-repair commit checks (worker side, pure host shapes).

Unit surface for the exclusive repair-only execution: the
duplicate-detecting slot-coverage check (a multiset comparison -- a
duplicated slot in the coverage is a write-aliasing hazard a set
comparison silently passes), the pre-write ownership fence, abort
containment, and the live-getenv sync-profile gate.
"""

from dataclasses import replace as dataclasses_replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.v1.core.test_final_audit_adversarial import full_fault_evidence
from tests.v1.core.test_final_validation_fixes import deliver
from tests.v1.core.test_scheduler_repair_budget import (
    admit_sync_external_hit,
    drive_repair_result,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import create_request, create_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import (
    CachedRequestData,
    RepairWork,
    SchedulerOutput,
)
from vllm.v1.outputs import RepairResultReport
from vllm.v1.worker.gpu.repair_executor import (
    _repair_cache_views,
    _repair_sync_profile_enabled,
    _slots_cover_span,
    _slots_owned_by_request,
    execute_repair_only,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BS = 128
BLOCK_TABLE = [7, 9]


def _cpu_meta(block_rows: list[torch.Tensor], slot_mapping: torch.Tensor) -> Any:
    """Real CPUAttentionMetadata with a padded 2D block table: the
    production constructor for attention metadata, so the executor runs
    against the real field layout (the per-request row is
    `block_table[row]`)."""
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadata

    width = max(len(r) for r in block_rows)
    table = torch.zeros(len(block_rows), width, dtype=torch.int32)
    for i, row in enumerate(block_rows):
        table[i, : len(row)] = row
    num_reqs = len(block_rows)
    return CPUAttentionMetadata(
        num_actual_tokens=slot_mapping.shape[0],
        max_query_len=slot_mapping.shape[0],
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32),
        max_seq_len=slot_mapping.shape[0],
        seq_lens=torch.full((num_reqs,), slot_mapping.shape[0], dtype=torch.int32),
        block_table=table,
        slot_mapping=slot_mapping,
        scheduler_metadata=None,
    )


def _meta(slots: torch.Tensor) -> Any:
    return _cpu_meta([torch.tensor(BLOCK_TABLE)], slots)


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
    """A slot mapping with an EXTRA duplicated row (a malformed metadata
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


# ---- The repair-only step shape and the pre-write ownership fence ----


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
    meta = _cpu_meta(
        block_rows=[torch.tensor([2]), torch.tensor([8, 3])],
        slot_mapping=torch.cat([_span_slots(2), _span_slots(3)]),
    )
    return works, tokens, positions, input_ids, meta


def test_repair_step_refuses_multiple_works():
    """A v1 repair-only step carries exactly ONE work: the scheduler
    admits one repair per step, and per-work attention-metadata slicing
    is unsupported (the whole-step metadata describes a single span).
    A multi-work step is refused fail-closed: every work still gets its
    failed verdict and its consensus collective, but zero layers
    execute."""
    works, tokens, positions, input_ids, meta = _multi_work_setup()
    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta
    )

    assert [r.request_id for r in results] == ["req-a", "req-b"]
    for r in results:
        assert r.executed_layers == 0
        assert r.aborted
        assert not r.structural_check
        assert "exactly one work" in (r.error_detail or "")


def _single_work_setup():
    """One repair work at rows [0, BS) with span [0, BS) and its own
    block-table row (block 2) -- the only shape a v1 repair step
    carries."""
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=(0, BS),
            layer_end=1,
            restored_extent=512,
            plan_id=1,
            epoch=7,
            expected_write_block_ids=(2,),
        )
    }
    tokens = {"req-a": BS}
    positions = torch.arange(BS, dtype=torch.int64)
    meta = _cpu_meta([torch.tensor([2])], _span_slots(2))
    return works, tokens, positions, positions.clone(), meta


def _run_repair(model, works, tokens, positions, input_ids, meta):
    sub = _sub_output(works, tokens)
    # Every replayed layer's name carries an entry, as in production
    # (build_attn_metadata maps every layer of the group to the one
    # metadata object the fence validates).
    n = len(model.model.layers)
    ctx = {f"model.layers.{i}": meta for i in range(n)}
    with set_forward_context(ctx, create_vllm_config(model="facebook/opt-125m")):
        return execute_repair_only(model, sub, input_ids, positions, list(works))


def test_pre_write_gate_refuses_foreign_slot_before_any_layer():
    """A slot mapping whose window points at a FOREIGN block (not in the
    work's own block-table row) is refused BEFORE any layer executes --
    zero cache-writing kernels run, and the verdict carries the gate
    refusal."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    bad_slots = meta.slot_mapping.clone()
    bad_slots[:] = 999 * BS + torch.arange(BS)
    meta_bad = _cpu_meta([torch.tensor([2])], bad_slots)
    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta_bad
    )

    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["pre_write_geometry"] is False
    assert ra.commit_check["slot_ownership"] is False
    assert not ra.structural_check


def test_layer_exception_contained_as_abort():
    """An arbitrary exception inside the repair stream is request-local:
    the stream stops, the partial writes' layer count is reported, the
    verdict is ABORT with backend_ok False -- the engine does not crash
    and no success is fabricated."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    # Layer 0 runs clean; layer 1 raises on its first call.
    works["req-a"].layer_end = 2
    model = _FakeModel(n_layers=4)

    class _BoomLayer(_FakeLayer):
        def __call__(self, positions, hidden_states, residual):
            raise RuntimeError("injected device-side failure")

    model.model.layers[1] = _BoomLayer("model.layers.1")

    results = _run_repair(model, works, tokens, positions, input_ids, meta)

    ra = results[0]
    assert ra.executed_layers == 1  # stopped at the raising layer
    assert ra.aborted
    assert ra.commit_check["backend_ok"] is False
    assert not ra.structural_check


# ---- The ownership fence: the independent authorization battery ----


def test_correlated_metadata_corruption_refused():
    """BOTH worker structures (the block-table row AND the slot window)
    consistently point at a foreign physical block while the work
    carries the true scheduler authorization. The self-referential gate
    accepts this geometry; the independent authorization must refuse it
    before any repair layer runs."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    corrupted = _cpu_meta(
        block_rows=[torch.tensor([8])],
        slot_mapping=_span_slots(8),
    )
    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, corrupted
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["authorized_ownership"] is False
    assert ra.commit_check["authorized_table"] is False
    assert ra.commit_check["pre_write_geometry"] is False
    assert not ra.structural_check


def test_swapped_authorization_refused():
    """The swap: the work carries a FOREIGN authorization. The honest
    worker geometry cannot reproduce a foreign authorization's slots --
    refused."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].expected_write_block_ids = (3,)  # the foreign block
    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert not ra.structural_check


def test_missing_authorization_fail_closed():
    """The fail-closed default: a work with NO authorization snapshot (an
    unbuild/stale shape) is never executable."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].expected_write_block_ids = ()
    results = _run_repair(
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
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=(0, 2 * BS),
            layer_end=1,
            restored_extent=512,
            plan_id=1,
            epoch=7,
            expected_write_block_ids=(2,),  # omits block 9!
        )
    }
    tokens = {"req-a": 2 * BS}
    positions = torch.arange(2 * BS, dtype=torch.int64)
    meta = _cpu_meta(
        # The row covers the span's blocks 0-1 (2, 9).
        block_rows=[torch.tensor([2, 9])],
        slot_mapping=torch.cat([_span_slots(2), _span_slots(9)]),
    )
    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, positions.clone(), meta
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert not ra.structural_check


def test_duplicated_slots_refused_under_authorization():
    """The slot window carries a duplicated slot and a displaced one (the
    same length, all inside the work's own authorized block): the
    multiset comparison against the authorization must refuse."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    dup = _span_slots(2).clone()
    dup[5] = dup[4]  # duplicate; dup[5]'s expected slot goes missing
    corrupted = _cpu_meta([torch.tensor([2])], dup)
    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, corrupted
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.commit_check["authorized_ownership"] is False
    assert not ra.structural_check


# ---- Replay-compatibility canary: the model's forward must match ----


class _CanaryLayer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual):
        return hidden_states, residual


class _PlainInner(torch.nn.Module):
    """A faithful native shape: embed_input_ids feeding
    layer(positions, hidden_states, residual)."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_CanaryLayer()])

    def embed_input_ids(self, input_ids):
        return torch.ones(input_ids.shape[0], 8)

    def forward(self, input_ids, positions, intermediate_tensors=None):
        hidden_states = self.embed_input_ids(input_ids)
        for layer in self.layers:
            hidden_states, _residual = layer(positions, hidden_states, None)
        return hidden_states


def test_replay_canary_accepts_plain_native_shape():
    """The canary passes exactly the shape the replay assumes:
    embed_input_ids feeding a 3-positional-arg layer with no extras."""
    from vllm.v1.worker.gpu.repair_executor import verify_replay_compatibility

    verify_replay_compatibility(SimpleNamespace(model=_PlainInner()))


def test_replay_canary_refuses_post_embed_transform():
    """A model that scales hidden states AFTER embed_input_ids (the muP
    input-scaling family: HyperClovAX embedding_multiplier, AFMoE
    mup_enabled) would replay with silently-wrong KV -- the canary must
    refuse it at init."""
    from vllm.v1.worker.gpu.repair_executor import verify_replay_compatibility

    class _ScalingInner(_PlainInner):
        def forward(self, input_ids, positions, intermediate_tensors=None):
            hidden_states = self.embed_input_ids(input_ids) * 3.0
            for layer in self.layers:
                hidden_states, _residual = layer(positions, hidden_states, None)
            return hidden_states

    with pytest.raises(RuntimeError, match="transforms the embedding"):
        verify_replay_compatibility(SimpleNamespace(model=_ScalingInner()))


def test_replay_canary_refuses_per_layer_extras():
    """A model whose loop supplies extra per-layer inputs (the Gemma4
    per-layer-input family) would replay with silently-wrong KV -- the
    canary must refuse it at init."""
    from vllm.v1.worker.gpu.repair_executor import verify_replay_compatibility

    class _ExtraArgInner(_PlainInner):
        def forward(self, input_ids, positions, intermediate_tensors=None):
            hidden_states = self.embed_input_ids(input_ids)
            for layer in self.layers:
                hidden_states, _residual = layer(
                    positions, hidden_states, None, per_layer_input=torch.zeros(1)
                )
            return hidden_states

    with pytest.raises(RuntimeError, match="per-layer inputs"):
        verify_replay_compatibility(SimpleNamespace(model=_ExtraArgInner()))


def test_replay_canary_refuses_unprobeable_forward():
    """A model whose forward cannot be driven to layer 0 (signature
    drift, required extras) is unverifiable: refuse rather than assume."""
    from vllm.v1.worker.gpu.repair_executor import verify_replay_compatibility

    class _BrokenInner(_PlainInner):
        def forward(self, input_ids, positions, intermediate_tensors=None):
            raise ValueError("boom before layer 0")

    with pytest.raises(RuntimeError, match="cannot verify"):
        verify_replay_compatibility(SimpleNamespace(model=_BrokenInner()))


# ---- Authorization-shape battery: aliasing and grid-range geometry ----


def test_aliased_authorization_refused():
    """An authorization naming the same block twice -- with the worker's
    block table and slot window aliased to match -- passes every multiset
    the fence derives from it: two span tokens would write one cache row
    and half the span's KV would be silently lost. The authorization must
    name distinct blocks."""
    slots = torch.cat([_span_slots(5), _span_slots(5)])
    meta = _cpu_meta([torch.tensor([5, 5])], slots)
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=(0, 2 * BS),
            layer_end=1,
            restored_extent=512,
            plan_id=1,
            epoch=7,
            expected_write_block_ids=(5, 5),
        )
    }
    tokens = {"req-a": 2 * BS}
    positions = torch.arange(2 * BS)
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="facebook/opt-125m")
    ):
        results = execute_repair_only(
            model, sub, positions, positions.clone(), ["req-a"]
        )

    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["authorized_distinct"] is False
    assert not ra.structural_check


def test_unaligned_span_grid_range_authorization_accepted():
    """The worker's block-count formula must match the scheduler's grid
    range (req_block_ids[rollback // bs : ceil(corrupt_end / bs)]): a span
    starting mid-block covers one block more than its rounded-up length.
    With the scheduler-correct grid-range authorization the fence accepts
    the unaligned geometry."""
    span = (10, 138)
    slots = torch.cat(
        [
            torch.arange(2 * BS + 10, 2 * BS + BS),
            torch.arange(3 * BS, 3 * BS + 10),
        ]
    )
    meta = _cpu_meta([torch.tensor([2, 3])], slots)
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=span,
            layer_end=1,
            restored_extent=512,
            plan_id=1,
            epoch=7,
            expected_write_block_ids=(2, 3),
        )
    }
    tokens = {"req-a": span[1] - span[0]}
    positions = torch.arange(span[0], span[1])
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="facebook/opt-125m")
    ):
        results = execute_repair_only(
            model, sub, positions, positions.clone(), ["req-a"]
        )

    ra = results[0]
    assert ra.executed_layers == 1
    assert not ra.aborted
    assert ra.commit_check["authorized_table"] is True
    assert ra.structural_check


def test_ambiguous_three_dim_cache_refused():
    """A 3-D cache with leading dim 2 and equal trailing dims is ambiguous
    between the core [num_blocks, block_size, C] layout (2 blocks) and
    the Ascend [2, num_blocks, block_size] K/V stack: both readings derive
    the same block size, so the fence cannot separate them. Refuse loudly
    instead of guessing (a wrong guess re-zeros the wrong cells)."""

    def _layer(cache):
        return SimpleNamespace(
            self_attn=SimpleNamespace(
                attn=SimpleNamespace(kv_cache=cache, layer_name="model.layers.0")
            )
        )

    with pytest.raises(ValueError, match="ambiguous"):
        _repair_cache_views(_layer(torch.zeros(2, 8, 8)))
    # The unambiguous neighbors keep their readings.
    views = _repair_cache_views(_layer(torch.zeros(3, 8, 8)))
    assert len(views) == 1
    views = _repair_cache_views(_layer(torch.zeros(2, 8, 6)))
    assert len(views) == 2


# ---- Envelope containment for a step whose FIRST work fails early ----


def _single_work(layer_end: int = 1):
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=(0, BS),
            layer_end=layer_end,
            restored_extent=512,
            plan_id=1,
            epoch=7,
            expected_write_block_ids=(2,),
        )
    }
    return works, {"req-a": BS}, torch.arange(BS)


def test_single_work_positions_failure_returns_failed_verdict():
    """An early per-work failure in a step whose FIRST (and only) work
    fails must converge to that work's failed verdict instead of
    escaping the envelope -- an escape raises out of execute_repair_only,
    the step's works get no verdict and no consensus collective, and
    the TP peers block inside theirs."""
    works, tokens, positions = _single_work()
    mismatched = positions + 7  # span says [0, 128)
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with set_forward_context(
        {"model.layers.0": _meta(_span_slots(2))},
        create_vllm_config(model="facebook/opt-125m"),
    ):
        results = execute_repair_only(
            model, sub, mismatched, mismatched.clone(), ["req-a"]
        )

    assert len(results) == 1
    ra = results[0]
    assert ra.aborted
    assert not ra.structural_check
    assert "positions" in (ra.error_detail or "")
    assert ra.commit_check["authorized_distinct"] is False


def test_zero_width_window_refused():
    """A zero-width repair window (an all-zero num_scheduled_tokens
    entry) must be refused as malformed: indexing positions[hi - 1] with
    hi == 0 reads the LAST row of the whole batch (Python negative
    indexing), which can spuriously satisfy the span check against an
    unrelated request's position."""
    works = {
        "req-a": RepairWork(
            request_id="req-a",
            repair_span=(10, 20),
            layer_end=1,
            restored_extent=64,
            plan_id=1,
            epoch=7,
            expected_write_block_ids=(2,),
        )
    }
    tokens = {"req-a": 0}
    # The batch ends at position 19: positions[-1] == 19 == span_end - 1,
    # so the unguarded span check would pass spuriously.
    positions = torch.arange(10, 20)
    sub = _sub_output(works, tokens)
    model = _FakeModel(n_layers=4)

    with set_forward_context(
        {"model.layers.0": _meta(_span_slots(2))},
        create_vllm_config(model="facebook/opt-125m"),
    ):
        results = execute_repair_only(
            model, sub, positions, positions.clone(), ["req-a"]
        )

    ra = results[0]
    assert ra.aborted
    assert not ra.structural_check
    assert "zero-width" in (ra.error_detail or "")


def test_single_work_ambiguous_cache_contained():
    """A cache geometry the block-size derivation cannot interpret (the
    ambiguous 3-D form) in a step whose FIRST work fails must be
    contained as that work's failed verdict, not escape the envelope."""
    inner = SimpleNamespace(
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    attn=SimpleNamespace(
                        kv_cache=torch.zeros(2, BS, BS),
                        layer_name="model.layers.0",
                    )
                )
            )
        ],
        embed_input_ids=lambda ids: torch.zeros(ids.shape[0], 8),
    )
    works, tokens, positions = _single_work()
    sub = _sub_output(works, tokens)

    with set_forward_context(
        {"model.layers.0": _meta(_span_slots(2))},
        create_vllm_config(model="facebook/opt-125m"),
    ):
        results = execute_repair_only(
            SimpleNamespace(model=inner), sub, positions, positions.clone(), ["req-a"]
        )

    assert len(results) == 1
    ra = results[0]
    assert ra.aborted
    assert not ra.structural_check
    assert "ambiguous" in (ra.error_detail or "")
    assert ra.commit_check["authorized_distinct"] is False


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
    ascend_meta = _cpu_meta(block_rows=rows, slot_mapping=slots)
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
    meta = _cpu_meta(block_rows=[torch.tensor([7])], slot_mapping=_span_slots(7))
    _rezero_repair_span([layer], meta)
    cache = layer.self_attn.attn.kv_cache
    # Rows of block 7 are zeroed across heads and channels...
    assert float(cache[7].abs().max()) == 0.0
    # ...and no other block was touched.
    others = cache.clone()
    others[7] = 1.0
    assert float(others.abs().min()) == 1.0


def test_rewound_flag_rides_repair_moves_and_drains():
    """The scheduler-signaled staging contract: the worker may restage a
    request's GPU-side num_computed_tokens only when the scheduler says
    so, because under async scheduling with speculative decoding the
    echoed count is an optimistic upper bound and the on-device
    rejection correction is the truth. Every repair-policy count move
    (the plan-synthesis rollback, the commit restoration) flags the
    request, the flag rides the very SchedulerOutput that schedules it,
    and the ordinary advance never flags."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]

    # The plan synthesis rolled the count back to the rollback boundary:
    # the repair step that schedules the request must carry the flag so
    # the worker stages the rolled-back value over its GPU count.
    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    assert r1.request_id in out2.rewound_req_ids

    # The commit restores the logical extent: the next step that
    # schedules the request carries the flag again.
    drive_repair_result(
        scheduler, r1, out2, structural_check=True, plan=plan, span=plan.repair_span
    )
    assert r1.num_computed_tokens == plan.restored_extent
    out3 = scheduler.schedule()
    assert r1.request_id in out3.num_scheduled_tokens
    assert r1.request_id in out3.rewound_req_ids

    # The ordinary advance never flags: the flag drained with out3 and
    # no later output carries any rewound row.
    out4 = scheduler.schedule()
    assert out4.rewound_req_ids == set()


def test_speculative_decoding_coexists_with_repair_steps():
    """Speculative decoding is not refused under the repair policy, but a
    repair-only step must never carry drafts: the span arithmetic yields
    none for a rewound request, the repair-work construction pins it, and
    the repair still commits while the worker's drafter skips the step."""
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        max_num_batched_tokens=4096,
        kv_load_failure_policy="repair",
        num_speculative_tokens=3,
    )
    # The fixture default resolves async_scheduling on, which the repair
    # policy refuses at init (same pin as make_scheduler).
    vllm_config.scheduler_config.async_scheduling = False
    scheduler = create_scheduler(vllm_config)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]

    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    # The repair step carries no drafts for the rewound request.
    assert r1.request_id not in out2.scheduled_spec_decode_tokens
    drive_repair_result(
        scheduler, r1, out2, structural_check=True, plan=plan, span=plan.repair_span
    )
    out3 = scheduler.schedule()
    assert r1.request_id in out3.num_scheduled_tokens
    assert r1.request_id in out3.rewound_req_ids
    out4 = scheduler.schedule()
    assert out4.rewound_req_ids == set()


def test_intra_span_permutation_refused():
    """The fence validates ORDER, not just membership: a metadata bug
    that permutes the span's slot rows or positions inside the
    authorized block passes every multiset and boundary check while
    writing each token's KV into a neighboring row of the same request.
    The ordered comparisons must refuse both shapes."""
    works, tokens, positions, input_ids, meta = _single_work_setup()

    # (a) permuted slot rows: the multiset is intact, the order is not.
    half = BS // 2
    swapped = meta.slot_mapping.clone()
    swapped[:half] = meta.slot_mapping[half:]
    swapped[half:] = meta.slot_mapping[:half]
    meta_a = _cpu_meta([torch.tensor([2])], swapped)
    ra = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta_a
    )[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert not ra.structural_check

    # (b) permuted positions: the boundaries are intact, the order is not.
    permuted = positions.clone()
    permuted[[10, 117]] = permuted[[117, 10]]
    rb = _run_repair(
        _FakeModel(n_layers=4),
        works,
        tokens,
        permuted,
        input_ids,
        _meta(_span_slots(2)),
    )[0]
    assert rb.executed_layers == 0
    assert rb.aborted
    assert not rb.structural_check


def test_forged_understated_layer_count_never_commits():
    """The commit arm treats the verdict as untrusted input: a green
    structural_check with an executed count below the plan's layer range
    (impossible from the worker, forgeable in transit) must fall back to
    recompute, not restore the extent."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]

    out2 = scheduler.schedule()
    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=True,
        plan=plan,
        span=plan.repair_span,
        executed_layers=0,
    )
    # Not committed: the malformed gate sends the forged verdict to the
    # recompute fallback (truncated to the span start).
    assert r1.num_computed_tokens == plan.repair_span[0]


def test_repair_step_output_tokens_suppressed():
    """A repair-only step produces no tokens by contract. If a runner
    ignored repair_works and returned sampled tokens for the repair
    request anyway (possible only for an out-of-tree platform runner),
    the scheduler must suppress them: the request's real output
    continues on the next ordinary step, after the verdict settled."""
    from tests.v1.kv_connector.unit.utils import create_model_runner_output

    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]

    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    # A non-conforming runner: the repair step returns sampled tokens
    # and no repair verdicts.
    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = None
    scheduler.update_from_output(out2, output)
    # The plan is consumed by the missing-verdict synthesis (abort) ...
    assert scheduler.recovery_plans.get(r1.request_id) is None
    assert r1.num_computed_tokens == plan.repair_span[0]
    # ... and the spurious token never reaches the request's output.
    assert r1.num_output_tokens == 0


def test_rewound_flag_rides_repair_abort():
    """The abort truncation (a failed verdict rolls the request back to
    the repair span start for the recompute fallback) flags the request
    exactly like the commit does, and the flag drains with the step that
    stages it."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)

    out2 = scheduler.schedule()
    assert r1.request_id in out2.rewound_req_ids
    plan = scheduler.recovery_plans[r1.request_id]
    drive_repair_result(
        scheduler, r1, out2, structural_check=False, plan=plan, span=plan.repair_span
    )
    out3 = scheduler.schedule()
    assert r1.request_id in out3.num_scheduled_tokens
    assert r1.request_id in out3.rewound_req_ids
    out4 = scheduler.schedule()
    assert r1.request_id not in out4.rewound_req_ids


def test_repair_dispatch_stays_eager_and_unsplit():
    """The dispatch contract for a repair-only step, behaviorally:
    need_eager forces CUDAGraphMode.NONE without consulting the
    cudagraph manager, and the dp_size==1 all-reduce-overlap re-check
    cannot split the batch once allow_ubatching is False -- while the
    same shape with ubatching allowed DOES split, proving the gate
    blocks a live dispatch path rather than an unreachable one."""
    from vllm.config import ParallelConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

    calls = []

    class _GraphPickingManager:
        def dispatch(self, *args, **kwargs):
            calls.append(1)
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=64,
                num_reqs=2,
                num_active_loras=0,
            )

    cfg = ParallelConfig(
        tensor_parallel_size=2,
        enable_dbo=True,
        dbo_schedule="all-reduce-overlap",
        dbo_prefill_token_threshold=8,
    )

    # The repair shape: eager, no split, manager never consulted.
    desc, _sync = dispatch_cg_and_sync_dp(
        _GraphPickingManager(),
        2,
        64,
        None,
        1,
        0,
        need_eager=True,
        parallel_config=cfg,
        allow_ubatching=False,
    )
    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_ubatches == 1
    assert calls == []

    # The control: identical shape with ubatching allowed splits, so the
    # gate above blocks a real re-check, not a dead branch.
    desc, _sync = dispatch_cg_and_sync_dp(
        _GraphPickingManager(),
        2,
        64,
        None,
        1,
        0,
        need_eager=True,
        parallel_config=cfg,
        allow_ubatching=True,
    )
    assert desc.num_ubatches == 2


def test_scheduler_refuses_async_scheduling_under_repair_policy():
    """kv_load_failure_policy='repair' + async_scheduling must fail at init:
    the batch-queue step calls sample_tokens unconditionally, but a
    repair-only step returns verdicts without execute_model_state, so
    sample_tokens returns None and the engine raises at the batch-queue
    None check."""
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        kv_load_failure_policy="repair",
    )
    vllm_config.scheduler_config.async_scheduling = True
    with pytest.raises(RuntimeError, match="not supported with async scheduling"):
        create_scheduler(vllm_config)


def test_repair_policy_allows_sync_scheduling():
    """The async guard is repair-specific: the same config with synchronous
    scheduling constructs a scheduler that carries the repair policy."""
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        kv_load_failure_policy="repair",
    )
    vllm_config.scheduler_config.async_scheduling = False
    assert create_scheduler(vllm_config).repair_kv_load_failures


def test_recompute_policy_allows_async_scheduling():
    """The async guard does not over-reject: the recompute policy under
    async scheduling constructs normally (its fallback rides the ordinary
    step shape)."""
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        kv_load_failure_policy="recompute",
    )
    vllm_config.scheduler_config.async_scheduling = True
    scheduler = create_scheduler(vllm_config)
    assert scheduler.recompute_kv_load_failures
    assert not scheduler.repair_kv_load_failures


def test_single_rank_consensus_is_local_success():
    """The single-rank consensus short-circuit returns the local-success
    predicate (uniform with the multi-rank semantics): a failed or aborted
    local run never reports agreement."""
    from vllm.v1.worker.gpu.repair_executor import _ranks_repair_agree

    green = dict(
        aborted=False,
        executed_layers=4,
        pre_write_ok=True,
        auth_ownership_ok=True,
        auth_table_ok=True,
        slot_ownership=True,
        write_coverage=True,
        layer_end=4,
        backend_ok=True,
    )
    assert _ranks_repair_agree(**green)
    assert not _ranks_repair_agree(**{**green, "aborted": True, "executed_layers": 2})
    assert not _ranks_repair_agree(**{**green, "backend_ok": False})
    assert not _ranks_repair_agree(**{**green, "pre_write_ok": False})
    assert not _ranks_repair_agree(**{**green, "executed_layers": 3})


def test_short_block_table_row_is_refused_not_raised():
    """A block-table row too short for the span (the span needs 2 blocks,
    the row lists 1) is a malformed geometry: the executor must return a
    REFUSED verdict (aborted, no layers executed, error_detail recorded),
    never raise out of execute_repair_only."""
    work = RepairWork(
        request_id="req-r",
        repair_span=(0, 2 * BS),
        layer_end=1,
        restored_extent=512,
        plan_id=1,
        epoch=7,
        expected_write_block_ids=(2, 3),
    )
    meta = _cpu_meta(
        block_rows=[torch.tensor([2])],  # one entry: the span needs two
        slot_mapping=torch.arange(2 * BS),
    )
    sub = _sub_output({"req-r": work}, {"req-r": 2 * BS})
    model = _FakeModel(n_layers=4)
    positions = torch.arange(2 * BS, dtype=torch.int64)

    with set_forward_context(
        {"model.layers.0": meta}, create_vllm_config(model="facebook/opt-125m")
    ):
        results = execute_repair_only(
            model, sub, positions.clone(), positions, ["req-r"]
        )

    (result,) = results
    assert result.executed_layers == 0
    assert result.aborted
    assert not result.structural_check
    assert result.error_detail is not None


class _PartialWriteBoomLayer(_FakeLayer):
    """A layer that writes SOME KV rows and then raises -- the realistic
    partial-write shape (a kernel that partially completed). The write and
    the raise are scoped to req-a's row window (its positions start below
    BS)."""

    def __call__(self, positions, hidden_states, residual):
        if int(positions.min()) < BS:
            cache = self.self_attn.attn.kv_cache[0]
            cache[2, :64] = 7.5  # block 2 (req-a's span block), rows 0..63
            raise RuntimeError("injected partial-write layer failure")
        return hidden_states + 1.0, residual


def test_abort_rezero_covers_the_raising_layer_partial_writes():
    """The raising layer may have written some KV rows before raising; the
    abort re-zero must cover its partial writes too (defense in depth --
    the scheduler-side eviction remains the authoritative invalidation),
    leaving no partially written state readable."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    # The work spans [0, 128) over block 2 with layer_end=2: layer 0
    # runs clean, layer 1 partially writes then raises.
    works["req-a"].layer_end = 2
    model = _FakeModel(n_layers=4)
    model.model.layers[1] = _PartialWriteBoomLayer("model.layers.1")
    # Pre-fill every layer's caches with a sentinel (AFTER the layer
    # replacement, so the boom layer's fresh caches are covered too) so
    # the test can see exactly which rows the repair touched/re-zeroed.
    for layer in model.model.layers:
        for cache in layer.self_attn.attn.kv_cache:
            cache.fill_(3.25)

    results = _run_repair(model, works, tokens, positions, input_ids, meta)
    ra = results[0]
    assert ra.aborted
    assert ra.executed_layers == 1
    assert not ra.structural_check
    assert "partial-write layer failure" in (ra.error_detail or "")

    layer0, layer1, layer2, layer3 = model.model.layers
    # Layer 0 (fully executed): its span rows were re-zeroed.
    assert layer0.self_attn.attn.kv_cache[0][2].abs().max().item() == 0
    # Layer 1 (the RAISING layer): its partial writes (rows 0..63 of
    # block 2) must be re-zeroed too.
    assert layer1.self_attn.attn.kv_cache[0][2].abs().max().item() == 0
    # Layers beyond the work's layer_end never execute: untouched.
    assert layer2.self_attn.attn.kv_cache[0][2, 0, 0].item() == 3.25
    assert layer3.self_attn.attn.kv_cache[0][2, 0, 0].item() == 3.25
    # A foreign block (3, named by no table row): untouched sentinels.
    for layer in model.model.layers:
        assert layer.self_attn.attn.kv_cache[0][3, 0, 0].item() == 3.25


# ---- Configuration guards and the multimodal admission exclusion ----


def test_scheduler_refuses_context_parallelism_under_repair_policy():
    """kv_load_failure_policy='repair' + context parallelism must fail at
    init: CP-local positions cannot satisfy the executor's absolute-span
    validation."""
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        kv_load_failure_policy="repair",
    )
    vllm_config.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(RuntimeError, match="not supported with context"):
        create_scheduler(vllm_config)


def test_repair_works_never_dispatched_to_ubatch_runner():
    """The DBO guard: a repair-only step must never be split into
    microbatches -- the dp_size == 1 fast path re-checks the
    all-reduce-overlap split after the eager descriptor and would route the
    repair batch through the ordinary full-depth ubatch forward before the
    repair branch is evaluated. The dispatch call must disable ubatching
    when repair work is scheduled."""
    import inspect

    from vllm.v1.worker.gpu import model_runner as mr

    src = inspect.getsource(mr.GPUModelRunner.execute_model)
    assert "repair_works is None" in src, (
        "the ubatching dispatch must be disabled for repair-only steps"
    )


def test_mm_request_repairs_fall_back_to_recompute():
    """A multimodal request's load-failure reports must NOT authorize a
    repair plan: the repair replay embeds the raw placeholder token ids
    (not the merged vision features), so the replayed KV would be silently
    wrong while the geometry-only commit battery passes. The admission
    pass routes the request to the recompute fallback instead: the
    request lands in the failed/failed_kv_load set and no recovery plan
    is installed."""
    from tests.v1.kv_connector.unit.utils import (
        create_request,
        create_scheduler,
        create_vllm_config,
    )
    from vllm.v1.outputs import KVLoadFailureReport
    from vllm.v1.request import RequestStatus

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        kv_load_failure_policy="repair",
    )
    vllm_config.scheduler_config.async_scheduling = False
    scheduler = create_scheduler(vllm_config)
    request = create_request()
    # Make the request multimodal: a non-empty mm_features list is the
    # scheduler-side marker the admission pass checks.
    request.mm_features = ["mm"]  # type: ignore[list-item]
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.requests[request.request_id] = request

    report = KVLoadFailureReport(
        request_id=request.request_id,
        source_rank=0,
        token_span=(0, 128),
        rollback_boundary=0,
        failed_block_ids=[0],
        restored_extent=128,
        healthy_above_span=True,
        observed_failed_layers=[1],
        transfer_generation=1,
    )

    failed = scheduler._handle_repair_reports([report])

    assert request.request_id in failed, (
        "the MM request must be failed closed to the recompute fallback"
    )
    assert scheduler.recovery_plans == {}, (
        "no repair plan may be installed for a multimodal request"
    )


# ---- The pre-execution preflight consensus (A3: TP safety before native
# ---- execution) ----


def test_preflight_refusal_keeps_commit_check_contract():
    """A refused preflight (here: a foreign slot mapping the fence
    refuses) reports every fence flag False AND the two new preflight
    keys (preflight_agree, capability_ok) with exact boolean values,
    zero layers executed, and an error naming the preflight -- the
    single-rank contract the multi-rank divergence tests build on."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    bad_slots = meta.slot_mapping.clone()
    bad_slots[:] = 999 * BS + torch.arange(BS)
    meta_bad = _cpu_meta([torch.tensor([2])], bad_slots)

    results = _run_repair(
        _FakeModel(n_layers=4), works, tokens, positions, input_ids, meta_bad
    )

    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["preflight_agree"] is False
    assert ra.commit_check["capability_ok"] is True
    assert ra.commit_check["pre_write_geometry"] is False
    assert not ra.structural_check
    assert "preflight" in (ra.error_detail or "").lower()


def test_preflight_divergence_detail_names_peer_reasons():
    """A green local preflight that a PEER refuses reports the peer's
    reason: the local rank executed nothing and the error names both the
    local and the peer codes (single-rank reduction of the divergence
    path; the multi-rank behavior is test_repair_tp_consensus's)."""
    from vllm.v1.worker.gpu.repair_executor import (
        _REPAIR_PF_FENCE,
        _preflight_divergence_detail,
        _ranks_repair_preflight_agree,
        _repair_preflight,
    )

    pf = _repair_preflight(
        _REPAIR_PF_FENCE,
        plan_id=3,
        epoch=9,
        span=(0, BS),
        layer_end=2,
        auth_fingerprint=1234,
    )
    assert not _ranks_repair_preflight_agree(pf)
    detail = _preflight_divergence_detail(pf)
    assert "local_reason=4" in detail
    assert "peer_reasons" in detail
    assert str(_REPAIR_PF_FENCE) in detail


def test_preflight_ok_summary_roundtrips_identity():
    """The all-gatherable summary tensor carries the full operation
    identity: an OK preflight's summary must equal the same fields
    repacked, so consensus compares identity, not a boolean."""
    from vllm.v1.worker.gpu.repair_executor import (
        _REPAIR_PF_OK,
        _RepairPreflight,
    )

    pf = _RepairPreflight(
        authorized=True,
        reason=_REPAIR_PF_OK,
        plan_id=7,
        epoch=12,
        span=(128, 256),
        layer_end=4,
        auth_fingerprint=987654321,
        capability=0,
    )
    summary = pf.summary()
    assert summary.tolist() == [1, 0, 7, 12, 128, 256, 4, 987654321, 0]
    # Any single identity field diverging produces a different tensor.
    for mutate in ("plan_id", "epoch", "layer_end", "auth_fingerprint", "capability"):
        other = dataclasses_replace(pf, **{mutate: getattr(pf, mutate) + 1})
        assert not torch.equal(other.summary(), summary), mutate


def test_preflight_refused_work_reports_fence_reason_in_error():
    """A refused fence inside the preflight surfaces the ordered fence
    failure in the verdict's error detail (not just a bare abort): the
    diagnosis must distinguish a fence refusal from a consensus
    divergence."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    # Layer stack shorter than the work's layer_end: the geometry checks
    # refuse inside the preflight.
    results = _run_repair(
        _FakeModel(n_layers=1),  # work.layer_end == 1 is still fine; use 2
        works,
        tokens,
        positions,
        input_ids,
        meta,
    )
    works["req-a"].layer_end = 2
    results = _run_repair(
        _FakeModel(n_layers=1), works, tokens, positions, input_ids, meta
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert "layer_end" in (ra.error_detail or "")
    assert ra.commit_check["preflight_agree"] is False


# ---- A2: per-layer attention-metadata identity over the replayed layers ----


class _NamedLayer(_FakeLayer):
    """A layer whose attention wrapper carries a layer_name, like the
    production Attention (llama: layer.self_attn.attn.layer_name)."""

    def __init__(self, layer_name: str):
        super().__init__(layer_name)
        self.self_attn.attn.layer_name = layer_name


def _named_model(n_layers: int) -> _FakeModel:
    model = _FakeModel(n_layers)
    model.model.layers = [_NamedLayer(f"model.layers.{i}") for i in range(n_layers)]
    return model


def test_replayed_layers_must_share_the_fenced_metadata_object():
    """Every replayed layer resolves its OWN metadata from the forward
    context (attn_metadata[layer_name]); the fence validates ONE object.
    A layer whose entry is a DIFFERENT metadata object would write
    through an unfenced geometry -- the preflight must refuse before any
    layer executes."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].layer_end = 2
    model = _named_model(4)
    # Layer 1 resolves a DIFFERENT metadata object: a foreign slot
    # window pointing at block 999 (outside every authorized region).
    foreign_meta = _cpu_meta(
        [torch.tensor([2])],
        torch.arange(999 * BS, 999 * BS + BS),
    )
    results = _run_repair_ctx(
        model,
        works,
        tokens,
        positions,
        input_ids,
        {"model.layers.0": meta, "model.layers.1": foreign_meta},
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert ra.commit_check["preflight_agree"] is False
    assert "DIFFERENT attention metadata" in (ra.error_detail or "")


def test_replayed_layer_missing_metadata_entry_refused():
    """A replayed layer with NO metadata entry would raise inside its
    metadata lookup mid-execution -- the preflight must refuse it up
    front."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].layer_end = 2
    model = _named_model(4)
    results = _run_repair_ctx(
        model,
        works,
        tokens,
        positions,
        input_ids,
        {"model.layers.0": meta},  # layer 1 has no entry
    )
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert "no attention metadata entry" in (ra.error_detail or "")


def test_replayed_layer_missing_slot_mapping_refused():
    """A replayed layer whose per-layer slot_mapping entry is absent
    would have its KV cache update silently SKIPPED (the unified op
    guards on layer_slot_mapping is None) -- a partial repair under a
    green verdict. The preflight must refuse it."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].layer_end = 2
    model = _named_model(4)
    with set_forward_context(
        {"model.layers.0": meta, "model.layers.1": meta},
        create_vllm_config(model="facebook/opt-125m"),
        slot_mapping={"model.layers.0": meta.slot_mapping},  # layer 1 absent
    ):
        sub = _sub_output(works, tokens)
        results = execute_repair_only(model, sub, input_ids, positions, list(works))
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert "no slot_mapping entry" in (ra.error_detail or "")


def test_named_layers_with_shared_metadata_still_commit():
    """The control: named layers sharing the fenced metadata object (the
    single-group construction invariant) still execute and commit."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].layer_end = 2
    model = _named_model(4)
    with set_forward_context(
        {"model.layers.0": meta, "model.layers.1": meta},
        create_vllm_config(model="facebook/opt-125m"),
        slot_mapping={
            "model.layers.0": meta.slot_mapping,
            "model.layers.1": meta.slot_mapping,
        },
    ):
        sub = _sub_output(works, tokens)
        results = execute_repair_only(model, sub, input_ids, positions, list(works))
    ra = results[0]
    assert ra.executed_layers == 2
    assert not ra.aborted
    assert ra.structural_check


def _run_repair_ctx(model, works, tokens, positions, input_ids, attn_meta):
    """_run_repair with an explicit attn_metadata dict (no per-layer
    slot_mapping entries: the identity check only requires presence when
    the forward context carries a dict)."""
    sub = _sub_output(works, tokens)
    with set_forward_context(attn_meta, create_vllm_config(model="facebook/opt-125m")):
        return execute_repair_only(model, sub, input_ids, positions, list(works))


def test_replayed_layer_divergent_cache_layout_refused():
    """A replayed layer whose cache geometry implies a DIFFERENT block
    size than the fenced layer 0 (an unexpected attention layout): the
    same slot numbers would split into different block/offset pairs and
    write outside the authorized rows. The preflight must refuse before
    any layer executes."""
    works, tokens, positions, input_ids, meta = _single_work_setup()
    works["req-a"].layer_end = 2
    model = _named_model(4)
    # Layer 1's cache carries a doubled block size: dim-1 = 2*BS.
    model.model.layers[1].self_attn.attn.kv_cache = [
        torch.zeros(4, 2 * BS, 8),
        torch.zeros(4, 2 * BS, 8),
    ]
    results = _run_repair(model, works, tokens, positions, input_ids, meta)
    ra = results[0]
    assert ra.executed_layers == 0
    assert ra.aborted
    assert "layout divergence" in (ra.error_detail or "")
    assert ra.commit_check["preflight_agree"] is False


def test_single_kv_group_guard_refuses_multi_group_at_init():
    """The construction invariant the per-layer metadata identity relies
    on: with more than one KV cache group, different replayed layers
    could legitimately carry DIFFERENT slot mappings and block sizes, so
    the single-metadata fence could not cover the write set. The
    scheduler must refuse the repair policy at init for multi-group
    configurations (making per-layer slot-mapping divergence unreachable
    by construction)."""
    from tests.v1.core.utils import new_kv_cache_group
    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        kv_load_failure_policy="repair",
    )
    vllm_config.scheduler_config.async_scheduling = False
    block_size = vllm_config.cache_config.block_size
    spec = AttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    two_groups = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            new_kv_cache_group(["layers.0-7"], spec),
            new_kv_cache_group(["layers.8-15"], spec),
        ],
    )
    with pytest.raises(RuntimeError, match="single KV cache group"):
        create_scheduler(vllm_config, kv_cache_config=two_groups)


# ---- A5: error handling and cleanup beyond the mapped battery ----


def test_verdict_collective_failure_after_clean_run_fails_closed(monkeypatch):
    """A failure of the VERDICT collective itself after a clean
    full-depth replay (process-group error or backend timeout) must
    converge to an aborted verdict with ranks_agree False: a successful
    transaction is never fabricated, and the scheduler's eviction
    remains the authoritative invalidation (the clean run's rows are
    not re-zeroed -- documented best-effort)."""
    from vllm.v1.worker.gpu import repair_executor

    works, tokens, positions, input_ids, meta = _single_work_setup()
    model = _FakeModel(n_layers=4)

    def _raise(*args, **kwargs):
        raise RuntimeError("pg timeout inside the verdict collective")

    monkeypatch.setattr(repair_executor, "_ranks_repair_agree", _raise)
    results = _run_repair(model, works, tokens, positions, input_ids, meta)

    ra = results[0]
    # The clean run executed every layer, but the transaction failed.
    assert ra.executed_layers == 1
    assert ra.aborted
    assert not ra.structural_check
    assert ra.commit_check["ranks_agree"] is False
    assert "pg timeout" in (ra.error_detail or "")


def test_request_gone_before_verdict_settles_without_touching():
    """A request that finished (or was otherwise removed) before its
    repair verdict arrived must settle the plan without crashing: the
    disposition is derived from the verdict alone, no restoration or
    truncation touches a request object that no longer exists."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]

    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    # The request disappears before the verdict arrives.
    scheduler.requests.pop(r1.request_id, None)
    disposition = scheduler._consume_repair_result(
        RepairResultReport(
            request_id=r1.request_id,
            plan_id=plan.plan_id,
            epoch=plan.admitted_step,
            structural_check=True,
            executed_layers=plan.layer_end,
            span=plan.repair_span,
            aborted=False,
            commit_check={},
            error_detail=None,
        )
    )
    assert disposition == "committed"
    assert scheduler.recovery_plans.get(r1.request_id) is None

    # The aborted twin: same shape, failed verdict, still no crash.
    # A fresh scheduler: the first scenario's dangling per-request state
    # (the hand-popped request's step bookkeeping) is orthogonal to the
    # verdict path under test here.
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r2, 3072)
    out3 = scheduler.schedule()
    reports2, attestations2 = full_fault_evidence(scheduler, r2)
    deliver(scheduler, out3, [r2], reports=reports2, attestations=attestations2)
    plan2 = scheduler.recovery_plans[r2.request_id]
    out4 = scheduler.schedule()
    assert out4.repair_works is not None
    scheduler.requests.pop(r2.request_id, None)
    disposition2 = scheduler._consume_repair_result(
        RepairResultReport(
            request_id=r2.request_id,
            plan_id=plan2.plan_id,
            epoch=plan2.admitted_step,
            structural_check=False,
            executed_layers=0,
            span=plan2.repair_span,
            aborted=True,
            commit_check={},
            error_detail="boom",
        )
    )
    assert disposition2 == "aborted"
    assert scheduler.recovery_plans.get(r2.request_id) is None
