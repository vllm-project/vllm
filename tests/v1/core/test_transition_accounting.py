# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Observe the real allocator; wrappers return original values unchanged."""

import json

import pytest

from .test_transition_runtime_reference import (
    _complete_hybrid_step,
    _create_hybrid_transition_scheduler,
)


@pytest.mark.parametrize("usable_blocks", [148, 163, 164])
@pytest.mark.parametrize("reject_drafts", [False, True])
def test_transition_accounting(tmp_path, monkeypatch, usable_blocks, reject_drafts):
    scheduler, requests = _create_hybrid_transition_scheduler(
        tmp_path, usable_blocks=usable_blocks
    )
    cache = scheduler.kv_cache_manager
    pool = cache.block_pool
    events = []
    context = {"step": -1, "request": None}
    allocate = cache.allocate_slots

    def observed_allocate(request, *args, **kwargs):
        previous = context["request"]
        context["request"] = request.request_id
        before = pool.get_num_free_blocks()
        result = allocate(request, *args, **kwargs)
        events.append(
            dict(
                context,
                event="allocate_slots",
                before=before,
                after=pool.get_num_free_blocks(),
                accepted=result is not None,
                reserved=kwargs.get("reserved_blocks", 0),
                transition=kwargs.get("transition_blocks", 0),
            )
        )
        context["request"] = previous
        return result

    monkeypatch.setattr(cache, "allocate_slots", observed_allocate)
    new_blocks = pool.get_new_blocks

    def observed_new(n, *args, **kwargs):
        before = pool.get_num_free_blocks()
        result = new_blocks(n, *args, **kwargs)
        after = pool.get_num_free_blocks()
        assert len(result) == n
        assert before - after == n
        events.append(
            dict(
                context,
                event="physical_allocation",
                count=n,
                before=before,
                after=after,
                block_ids=[b.block_id for b in result],
            )
        )
        return result

    monkeypatch.setattr(pool, "get_new_blocks", observed_new)
    for index, manager in enumerate(cache.coordinator.single_type_managers):
        original = manager.get_num_blocks_to_allocate

        def observe(
            *args, _original=original, _manager=manager, _index=index, **kwargs
        ):
            value = _original(*args, **kwargs)
            rid = args[0] if args else kwargs.get("request_id")
            events.append(
                dict(
                    context,
                    event="group_prediction",
                    request=rid,
                    group=_index,
                    manager=type(_manager).__name__,
                    blocks=value,
                    free=pool.get_num_free_blocks(),
                    partial_hit=rid in getattr(_manager, "_partial_hit_reqs", {}),
                    checkpoint_blocks=getattr(
                        _manager.kv_cache_spec, "num_prefill_checkpoint_blocks", 0
                    ),
                    admission_cap=kwargs.get("apply_admission_cap", False),
                )
            )
            return value

        monkeypatch.setattr(manager, "get_num_blocks_to_allocate", observe)
    for step in range(12):
        context["step"] = step
        before = pool.get_num_free_blocks()
        output = scheduler.schedule()
        events.append(
            dict(
                context,
                event="schedule",
                before=before,
                after=pool.get_num_free_blocks(),
                preempted=sorted(output.preempted_req_ids or []),
                scheduled=dict(output.num_scheduled_tokens),
                computed={r.request_id: r.num_computed_tokens for r in requests},
            )
        )
        _complete_hybrid_step(scheduler, output, reject_drafts)
        events.append(
            dict(context, event="output_complete", free=pool.get_num_free_blocks())
        )
        if output.preempted_req_ids or not scheduler.running:
            break
    out = tmp_path / "block-accounting.json"
    out.write_text(json.dumps(events, indent=2))


@pytest.mark.parametrize("usable_blocks", [148, 163, 164])
@pytest.mark.parametrize("reject_drafts", [False, True])
def test_capacity_matrix_completes_without_preemption(
    tmp_path, usable_blocks, reject_drafts
):
    scheduler, requests = _create_hybrid_transition_scheduler(
        tmp_path, usable_blocks=usable_blocks
    )
    for _ in range(6000):
        if not scheduler.running:
            break
        output = scheduler.schedule()
        assert not output.preempted_req_ids
        assert output.total_num_scheduled_tokens > 0
        _complete_hybrid_step(scheduler, output, reject_drafts)
    assert not scheduler.running and not scheduler.waiting
    assert [r.num_output_tokens for r in requests] == [1024] * 7
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == usable_blocks
