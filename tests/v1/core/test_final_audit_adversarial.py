#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final-audit adversarial matrix (v1.3, Phase 3): the shapes from the
task brief that the H10-H14 suites left untested."""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    _structured_report,
    _terminal_attestation,
    admit_sync_external_hit,
    drive_repair_result,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
)
from vllm.v1.outputs import KVConnectorOutput, RepairResultReport
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def deliver(scheduler, out, reqs, reports=(), attestations=(), finished_recving=None):
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        finished_recving=finished_recving,
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    scheduler.update_from_output(out, output)
    return output


def full_fault_evidence(scheduler, request):
    report = _structured_report(scheduler, request)
    gen = scheduler._req_transfer_generation[request.request_id]
    failed = len(report.failed_block_ids)
    return [report], [
        _terminal_attestation(
            request.request_id,
            rank=0,
            generation=gen,
            loaded=192 - failed,
            failed=failed,
            certified=192 - failed,
        )
    ]


def test_aborted_repair_block_reuse_recomputes_locally():
    """32: an ABORTED repair must not leave the span published."""
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
    drive_repair_result(scheduler, r1, out2, False, plan=plan, aborted=True)

    pool = scheduler.kv_cache_manager.block_pool
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    for bid in block_ids[96:]:
        assert pool.blocks[bid].block_hash is None
    assert r1.request_id not in scheduler.recovery_plans
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens.get(r1.request_id, 0) > 0


def test_committed_repair_prefix_reused_by_two_peers():
    """34: a COMMITTED repair publishes the verified extent; later
    same-prompt requests match it as ordinary prefix cache."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.healthy_above_span

    out2 = scheduler.schedule()
    drive_repair_result(scheduler, r1, out2, True, plan=plan, span=plan.repair_span)

    pool = scheduler.kv_cache_manager.block_pool
    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    assert all(pool.blocks[b].block_hash is not None for b in r1_blocks[:192])

    r2 = create_request(num_tokens=3328, common_prefix_len=3072)
    r3 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r2)
    scheduler.add_request(r3)
    # Drain until both peers are admitted (r1's decode owns the budget
    # for a few steps); neither peer ever consults the external source
    # (a fresh local admission, no new attempt) and neither is poisoned.
    for _ in range(8):
        out4 = scheduler.schedule()
        if (
            out4.num_scheduled_tokens.get(r2.request_id, 0) > 0
            and out4.num_scheduled_tokens.get(r3.request_id, 0) > 0
        ):
            break
        scheduler.update_from_output(
            out4,
            create_model_runner_output(
                reqs=[
                    rq
                    for rq in (r1, r2, r3)
                    if rq.request_id in out4.num_scheduled_tokens
                ],
                use_eos=False,
            ),
        )
    assert r2.request_id not in scheduler._req_transfer_generation
    assert r2.request_id not in scheduler._untrusted_external_reqs
    assert r3.request_id not in scheduler._untrusted_external_reqs


def test_missing_finished_recving_parks_without_freezing():
    """37: no finished_recving ever. The report defers, the plan never
    installs, the freeze never engages; peers keep scheduling."""
    from tests.v1.core.test_scheduler_repair_budget import (
        admit_async_external_hit,
    )

    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=512)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r2.request_id in out1.num_scheduled_tokens

    report = _structured_report(scheduler, r1)
    gen = scheduler._req_transfer_generation[r1.request_id]
    failed = len(report.failed_block_ids)
    deliver(
        scheduler,
        out1,
        [r1, r2],
        reports=[report],
        attestations=[
            _terminal_attestation(
                r1.request_id,
                generation=gen,
                loaded=192 - failed,
                failed=failed,
                certified=192 - failed,
            )
        ],
    )
    assert r1.request_id not in scheduler.recovery_plans
    assert scheduler._req_pending_async_reports.get(r1.request_id)
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    for _ in range(4):
        out = scheduler.schedule()
        assert r2.request_id in out.num_scheduled_tokens
        scheduler.update_from_output(
            out,
            create_model_runner_output(reqs=[r2], use_eos=False),
        )
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.recovery_plans == {}


def test_duplicate_verdict_consumes_plan_once():
    """45: a repeated repair result for the same plan is stale."""
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
    verdict = RepairResultReport(
        request_id=r1.request_id,
        plan_id=plan.plan_id,
        epoch=plan.admitted_step,
        structural_check=True,
        executed_layers=plan.layer_end,
        span=plan.repair_span,
    )
    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(repair_results=[verdict, verdict])
    scheduler.update_from_output(out2, output)
    assert r1.request_id not in scheduler.recovery_plans
    # The commit restored the external extent; the step's own sampled
    # token advances the count by one.
    assert r1.num_computed_tokens >= plan.restored_extent

    out3 = scheduler.schedule()
    output2 = create_model_runner_output(reqs=[r1], use_eos=False)
    output2.kv_connector_output = KVConnectorOutput(repair_results=[verdict])
    scheduler.update_from_output(out3, output2)
    assert r1.request_id not in scheduler.recovery_plans
    assert r1.num_computed_tokens >= plan.restored_extent


def test_two_requests_independent_failures_isolated():
    """46: two requests, each with its own fault: isolated fail-closed."""
    from unittest.mock import Mock

    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    scheduler.connector = Mock()
    scheduler.connector.get_external_cache_hit_sources.return_value = None
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = ()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        3072,
        False,
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    out1 = scheduler.schedule()

    scheduler.update_from_output(
        out1,
        create_model_runner_output(reqs=[r1, r2], use_eos=False),
    )
    for r in (r1, r2):
        assert r.request_id in scheduler._untrusted_external_reqs
        assert r.num_computed_tokens == 0
        assert scheduler.recovery_plans == {}
        pool = scheduler.kv_cache_manager.block_pool
        (blocks,) = scheduler.kv_cache_manager.get_block_ids(r.request_id)
        assert all(pool.blocks[b].block_hash is None for b in blocks)

    # Both requests are scheduled across the next steps (chunked under
    # the 4096-token budget); neither is stuck.
    scheduled = {}
    for _ in range(6):
        out2 = scheduler.schedule()
        for rid, n in out2.num_scheduled_tokens.items():
            scheduled[rid] = scheduled.get(rid, 0) + n
        if (
            scheduled.get(r1.request_id, 0) >= 3328
            and scheduled.get(r2.request_id, 0) >= 3328
        ):
            break
        scheduler.update_from_output(
            out2,
            create_model_runner_output(
                reqs=[
                    rq for rq in (r1, r2) if rq.request_id in out2.num_scheduled_tokens
                ],
                use_eos=False,
            ),
        )
    assert scheduled.get(r1.request_id, 0) >= 3328
    assert scheduled.get(r2.request_id, 0) >= 3328
