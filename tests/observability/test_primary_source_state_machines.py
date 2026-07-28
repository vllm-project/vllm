from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]


def _load_primary_usdt():
    path = ROOT / "vllm" / "primary_usdt.py"
    spec = importlib.util.spec_from_file_location("g2_primary_usdt_state", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_primary_usdt()


def _source(relative: str) -> str:
    text = (ROOT / relative).read_text()
    ast.parse(text)
    return text


def test_a07_block_generation_allocate_share_reuse_and_overflow_contract():
    generation = 0
    generation = M.next_block_generation(generation)
    assert generation == 1
    shared_touch_generation = generation
    cache_idle_generation = shared_touch_generation
    assert cache_idle_generation == 1
    generation = M.next_block_generation(cache_idle_generation)
    assert generation == 2
    with pytest.raises(OverflowError):
        M.next_block_generation(M.UINT64_MAX - 1)


def test_a09_owner_ledger_preserves_sharing_and_rejects_duplicate_edges():
    ledger = M.PrimaryOwnerLedger()
    key = (17, 3)
    first = ledger.add(key, "request-A")
    second = ledger.add(key, "request-B")
    assert (first.count_before, first.count_after) == (0, 1)
    assert (second.count_before, second.count_after) == (1, 2)
    assert ledger.owners(key) == frozenset({"request-A", "request-B"})
    with pytest.raises(M.PrimaryUSDTError, match="duplicate owner add"):
        ledger.add(key, "request-A")
    removed = ledger.remove(key, "request-A")
    assert (removed.count_before, removed.count_after) == (2, 1)
    assert ledger.owners(key) == frozenset({"request-B"})
    ledger.remove(key, "request-B")
    assert not ledger.owners(key)
    with pytest.raises(M.PrimaryUSDTError, match="missing owner remove"):
        ledger.remove(key, "request-B")


def test_a08_refcount_transitions_accept_only_actual_plus_or_minus_one():
    M.validate_refcount_transition(0, 1, 1)
    M.validate_refcount_transition(1, 2, 2)
    M.validate_refcount_transition(2, 1, 3)
    with pytest.raises(M.PrimaryUSDTError):
        M.validate_refcount_transition(0, 2, 1)
    with pytest.raises(M.PrimaryUSDTError):
        M.validate_refcount_transition(0, 0, 3)


def test_a12_a13_mapping_completion_is_count_and_failure_closed():
    assert M.mapping_is_complete(1025, 1025, 0)
    assert not M.mapping_is_complete(1025, 1024, 0)
    assert not M.mapping_is_complete(1025, 1025, 1)
    assert not M.mapping_is_complete(1, 1, 2)


def test_a06_scheduler_has_independent_checked_identities_and_explicit_events():
    source = _source("vllm/v1/core/sched/scheduler.py")
    assert "_primary_scheduler_output_counter = SourceOwnedCounter()" in source
    assert "_primary_scheduler_batch_counter = SourceOwnedCounter()" in source
    assert "scheduler_step_begin_v2" in source
    assert "scheduler_step_end_v2" in source
    assert "scheduler_output_member_v2" in source
    assert "scheduler_queue_member_v2" in source
    assert "scheduler_state_transition_v3" in source
    assert "_primary_queue_snapshot_counter = SourceOwnedCounter()" in source
    assert "queue_snapshot_id=queue_snapshot_id" in source
    assert "queue_position=queue_position" in source
    assert "membership_complete=int(emitted_member_count == len(members))" in source
    assert "scheduler_output_id = scheduler_step_id" not in source


def test_a08_a09_source_refcount_owner_and_generation_are_actual_state():
    block = _source("vllm/v1/core/block_pool.py")
    manager = _source("vllm/v1/core/single_type_kv_cache_manager.py")
    assert "before = block.ref_cnt" in block
    assert "block.primary_generation = next_block_generation" in block
    assert "add_primary_owner" in block and "remove_primary_owner" in block
    assert "duplicate_owner_add" in block and "missing_owner_remove" in block
    assert "add_primary_owner(request_id" in manager
    assert "remove_primary_owner(request_id" in manager


def test_a10_engine_release_and_frontend_closure_are_semantically_separate():
    scheduler = _source("vllm/v1/core/sched/scheduler.py")
    output = _source("vllm/v1/engine/output_processor.py")
    assert "request_terminal_v2" in output
    assert "request_cleanup_v3" in output
    assert "request_cleanup_v2" not in scheduler
    assert "request_cleanup_v3" not in scheduler
    assert "self.request_states.pop(req_id, None)" in output
    assert "_emit_primary_frontend_closure" in output
    assert "mark_terminal_once" in output


def test_a11_a12_worker_uses_actual_complete_mapping_and_generation_handoff():
    worker = _source("vllm/v1/worker/gpu_model_runner.py")
    output = _source("vllm/v1/core/sched/output.py")
    assert "_emit_primary_slot_mapping" in worker
    assert "worker_slot_mapping_begin_v2" in worker
    assert "worker_slot_mapping_entry_v2" in worker
    assert "worker_slot_mapping_end_v2" in worker
    assert "primary_block_generations" in worker and "primary_block_generations" in output
    assert "mapping_is_complete" in worker


def test_a16_false_scheduler_slot_fallback_is_removed():
    manager = _source("vllm/v1/core/kv_cache_manager.py")
    assert "kv_cache_manager_allocate_slots_fallback" not in manager


def test_a17_cpu_only_suite_has_no_collector_gpu_or_auxiliary_backfill_calls():
    primary = _source("vllm/primary_usdt.py")
    forbidden = (
        "nvidia-smi",
        "bpftool",
        "bpftrace",
        "CUDA_VISIBLE_DEVICES=",
        "python_jsonl",
        "sidecar_backfill",
        "time_nearest",
    )
    assert all(token not in primary.lower() for token in forbidden)
