# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded admission for recovery plans (hardening H4).

A DETECTED plan is the drain-first admission of its steps: every OTHER
request is frozen while the plan waits for a budget it can fit, so an
unbounded deferral is an engine-wide stall, not a per-request one. The
pre-hardening code incremented plan.num_deferrals and never read it.

The admission condition reduces to input_budget - draft_slots >=
max(E, n_eff) once the long-prefill threshold is exempted (the repair
step is exclusive, so the threshold's starvation protection has no one
to protect, and the span cannot be chunked). These tests pin:

- the static never-admissible gate (a plan no step could ever fund --
  running arm: E + n_eff must fit the input budget beside the draft
  slots; waiting arm: max(E, n_eff) -- dissolves at its first budget
  call, not after freezing the engine);
- the long-prefill threshold exemption (the pre-fix engine froze
  forever when the threshold fell below the span);
- the deferral bound (a plan deferred REPAIR_MAX_ADMISSION_DEFERRALS
  consecutive steps dissolves to the recompute fallback);
- cancellation/completion/preemption cleaning up pending plans;
- one repair admission per step with competing plans;
- ordinary traffic resuming in the very step a plan dissolves.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    N_EFF,
    E,
    _deliver_report,
    _structured_report,
    admit_async_external_hit,
    admit_sync_external_hit,
    install_plan,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import (
    KVConnectorOutput,  # noqa: F401  (re-exported fixture helper)
)
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _plan_installed(scheduler, request, e=E, span=(96 * BS, 96 * BS + N_EFF)):
    return install_plan(scheduler, request, e, span=span)


def test_full_depth_repair_shares_step_with_healthy_decode(tmp_path):
    """A full-depth repair must not freeze an already decoding request."""
    from transformers import OPTConfig

    OPTConfig().save_pretrained(tmp_path)
    scheduler = make_scheduler(
        model=str(tmp_path),
        kv_load_failure_policy="repair",
        max_num_batched_tokens=8192,
    )
    damaged = create_request(num_tokens=3328)
    healthy = create_request(num_tokens=256)
    scheduler.add_request(damaged)
    scheduler.add_request(healthy)
    admit_sync_external_hit(scheduler, damaged, 3072)
    first = scheduler.schedule()
    scheduler.update_from_output(
        first, create_model_runner_output(reqs=[damaged, healthy], use_eos=False)
    )
    plan = install_plan(scheduler, damaged, E, layer_end=16)
    damaged.num_computed_tokens = plan.repair_span[0]

    mixed = scheduler.schedule()
    assert mixed.num_scheduled_tokens[damaged.request_id] == N_EFF
    assert mixed.num_scheduled_tokens.get(healthy.request_id) == 1
    assert mixed.repair_full_forward
    assert set(mixed.repair_works) == {damaged.request_id}


def test_busy_batch_defers_full_depth_repair_without_dissolving(tmp_path):
    """A decode token can defer a boundary-sized plan without rejecting it."""
    from transformers import OPTConfig

    OPTConfig().save_pretrained(tmp_path)
    scheduler = make_scheduler(
        model=str(tmp_path),
        kv_load_failure_policy="repair",
        max_num_batched_tokens=8192,
    )
    healthy = create_request(num_tokens=256)
    damaged = create_request(num_tokens=3328)
    scheduler.add_request(healthy)
    scheduler.add_request(damaged)
    admit_sync_external_hit(scheduler, damaged, 3072)
    first = scheduler.schedule()
    scheduler.update_from_output(
        first, create_model_runner_output(reqs=[healthy, damaged], use_eos=False)
    )
    plan = install_plan(scheduler, damaged, 8192, layer_end=16)
    damaged.num_computed_tokens = plan.repair_span[0]

    busy = scheduler.schedule()
    assert busy.num_scheduled_tokens[healthy.request_id] > 0
    assert damaged.request_id not in busy.num_scheduled_tokens
    assert scheduler.recovery_plans[damaged.request_id] is plan
    assert plan.state == "DETECTED"


def test_admission_arithmetic_of_the_abort_fixture():
    """The abort fixture's arithmetic (max_num_batched_tokens=8192, E=2623,
    n_eff=1536): the static gate passes (max(E, n_eff) = 2623 <= 8192)
    and the admission debits exactly the surcharge beside the span --
    the formula the fixture's comment documents."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]

    out = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out.num_scheduled_tokens[r1.request_id] == N_EFF
    assert out.repair_works is not None


def test_static_never_admissible_plan_dissolves_immediately():
    """A span that can never fit the input budget (n_eff=4992 with
    max_num_batched_tokens=4096 but max_num_scheduled_tokens=8192: the
    pre-hardening gate only checked E against the batched budget and
    n_eff against the scheduled budget, so this plan deferred forever
    while freezing the engine) dissolves at its first budget call."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=4096
    )
    scheduler.max_num_scheduled_tokens = 8192
    r1 = create_request(num_tokens=5888)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 4992)
    out1 = scheduler.schedule()
    assert r1.request_id in out1.num_scheduled_tokens

    # n_eff = 4992 > input_budget 4096 but <= max_num_scheduled_tokens.
    plan = _plan_installed(scheduler, r1, e=1500, span=(96 * BS, 96 * BS + 4992))
    r1.num_computed_tokens = plan.repair_span[0]
    out2 = scheduler.schedule()
    # Dissolved: no repair work, and the owner skips THIS step (the
    # dissolution returns a zero schedule so the threshold exemption
    # the plan carried cannot leak an uncapped chunk -- the review's
    # N5); the next step schedules ordinarily, threshold applied.
    assert scheduler.recovery_plans == {}
    assert out2.repair_works is None
    assert r1.request_id not in out2.num_scheduled_tokens
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] > 0


def test_long_prefill_threshold_never_freezes_the_engine():
    """The threshold deadlock (pre-fix): with another request present the
    long-prefill cap applied to the damaged request, and a threshold
    below the repair span deferred the plan forever -- while the drain-
    first freeze stalled the whole engine. The exemption admits the
    repair; the competitor rides the very next step."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    scheduler.scheduler_config.long_prefill_token_threshold = 256
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=256)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r2.request_id in out1.num_scheduled_tokens
    # Process out1's output so r2 carries a sampled token (without it,
    # the fully-prefilled r2 has no work and would be skipped for
    # reasons unrelated to the freeze under test).
    output1 = create_model_runner_output(reqs=[r1, r2], use_eos=False)
    scheduler.update_from_output(out1, output1)

    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]
    # The threshold (256) is far below the span (1536): without the
    # exemption this admission would cap at 256 and defer forever.
    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out2.repair_works is not None
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
    # The repair step is exclusive: r2 waits exactly one step, then
    # resumes.
    assert r2.request_id not in out2.num_scheduled_tokens
    out3 = scheduler.schedule()
    assert r2.request_id in out3.num_scheduled_tokens


def test_budget_exactly_at_threshold_admits():
    """E + n_eff == input_budget: the running arm's boundary is inclusive
    -- the full-charge reserve leaves exactly n_eff, which suffices."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=E + N_EFF
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    plan = _plan_installed(scheduler, r1)
    r1.num_computed_tokens = plan.repair_span[0]

    out = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out.num_scheduled_tokens[r1.request_id] == N_EFF


def test_deferral_bound_dissolves_blocked_owner(monkeypatch):
    """The review's no-tick shape under the H14 deferral semantics: a
    contract-violating connector reports a failed async load but never
    delivers finished_recving, so the owner stays parked in
    WAITING_FOR_REMOTE_KV. The report of an attempt still in flight is
    DEFERRED (its completeness cannot be judged until the attempt is
    terminal), so NO plan installs and the drain-first freeze never
    engages: healthy traffic (r2) runs uninterrupted from the first
    step -- strictly better than the pre-H14 bounded freeze, which held
    the engine for REPAIR_MAX_ADMISSION_DEFERRALS steps before
    dissolving. The owner stays parked until the connector delivers
    finished_recving (the deferred report is then judged with the
    attempt's complete evidence) or the request is aborted -- the
    documented connector-liveness risk of a hung transfer."""
    monkeypatch.setattr(Scheduler, "REPAIR_MAX_ADMISSION_DEFERRALS", 3)
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

    # One update: r2's decode output plus the failure report of the
    # async load that never completes (no finished_recving).
    _deliver_report(
        scheduler, r1, out1, _structured_report(scheduler, r1), reqs=[r1, r2]
    )
    assert r1.request_id not in scheduler.recovery_plans
    assert scheduler._req_pending_async_reports.get(r1.request_id)

    # No freeze ever engages: r2 (a healthy decode request) schedules on
    # every step while the owner stays parked.
    for _ in range(3):
        out = scheduler.schedule()
        assert r2.request_id in out.num_scheduled_tokens
        scheduler.update_from_output(
            out, create_model_runner_output(reqs=[r2], use_eos=False)
        )
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Still deferred, still no plan, still no freeze.
    out = scheduler.schedule()
    assert r2.request_id in out.num_scheduled_tokens
    assert out.repair_works is None
    assert scheduler.recovery_plans == {}


def test_deferral_below_bound_keeps_waiting(monkeypatch):
    """The mirror of the bound: a deferral below the bound keeps the
    plan pending (the freeze holds, the tick counts) -- the bound must
    not dissolve plans eagerly. H14: the report of an async attempt
    still in flight no longer installs a plan (it is deferred until
    the attempt is terminal), so the parked-owner tick is constructed
    directly -- the shape still arises whenever a plan's owner is
    skipped before its budget call (blocked remote-KV waits, paused
    sessions, DP-balancing and PP-cadence skips)."""
    monkeypatch.setattr(Scheduler, "REPAIR_MAX_ADMISSION_DEFERRALS", 32)
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    # The owner is parked in WAITING_FOR_REMOTE_KV; a pending plan for
    # it cannot be admitted this step (the owner never reaches its
    # budget call), so the tick counts.
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    plan = install_plan(scheduler, r1, E, failed_block_ids=[block_ids[96]])

    scheduler.schedule()
    assert scheduler.recovery_plans == {r1.request_id: plan}
    assert plan.num_deferrals == 1


def test_cancellation_during_deferral_purges_plan():
    """A client disconnect (finish_requests) while the plan waits: the
    plan is purged with the request -- no leaked plan, no leaked
    freeze."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    _plan_installed(scheduler, r1)

    scheduler.finish_requests(
        [r1.request_id], finished_status=RequestStatus.FINISHED_ABORTED
    )
    assert scheduler.recovery_plans == {}
    # The engine schedules normally afterwards.
    out = scheduler.schedule()
    assert r1.request_id not in out.num_scheduled_tokens


def test_completion_during_deferral_purges_plan():
    """The stopped-completion variant of the cancellation path."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
    _plan_installed(scheduler, r1)

    scheduler.finish_requests(
        [r1.request_id], finished_status=RequestStatus.FINISHED_STOPPED
    )
    assert scheduler.recovery_plans == {}


def test_competing_plans_admit_one_per_step():
    """Two damaged requests: the repair step is exclusive, so the plans
    admit one per step (the second waits -- deferred, not dissolved)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    admit_sync_external_hit(scheduler, r2, 3072)
    scheduler.schedule()

    plan1 = _plan_installed(scheduler, r1)
    plan2 = _plan_installed(scheduler, r2)
    r1.num_computed_tokens = plan1.repair_span[0]
    r2.num_computed_tokens = plan2.repair_span[0]

    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    assert len(out2.repair_works) == 1
    first = next(iter(out2.repair_works))
    assert {plan1.state, plan2.state} == {"ADMITTED", "DETECTED"}

    out3 = scheduler.schedule()
    assert out3.repair_works is not None
    assert len(out3.repair_works) == 1
    assert next(iter(out3.repair_works)) != first


def test_dissolving_step_resumes_ordinary_traffic_immediately():
    """When a plan dissolves mid-step, the drain-first freeze lifts for
    the rest of THAT step: an ordinary request scheduled after the
    damaged one is not made to wait an extra step."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=4096
    )
    scheduler.max_num_scheduled_tokens = 8192
    r1 = create_request(num_tokens=5888)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 4992)
    out1 = scheduler.schedule()
    # r1's external-hit chunk plus r2's ordinary chunk exhaust out1's
    # input budget, leaving r2 with work in hand for the next step.
    assert r1.request_id in out1.num_scheduled_tokens
    assert r2.request_id in out1.num_scheduled_tokens
    assert r2.num_tokens - r2.num_computed_tokens > 0

    plan = _plan_installed(scheduler, r1, e=1500, span=(96 * BS, 96 * BS + 4992))
    r1.num_computed_tokens = plan.repair_span[0]
    plan.num_deferrals = Scheduler.REPAIR_MAX_ADMISSION_DEFERRALS - 1
    # The next step: r1's budget call dissolves the never-admissible
    # plan, and r2 (behind r1 in the running queue) schedules in the
    # SAME step -- the freeze does not outlive the plan.
    out2 = scheduler.schedule()
    assert scheduler.recovery_plans == {}
    assert out2.repair_works is None
    # The freeze does not outlive the plan: r2 (examined after r1 in
    # the running queue) schedules in the SAME step. r1 itself skips
    # this one step (the dissolution's zero schedule) and resumes the
    # next.
    assert r2.request_id in out2.num_scheduled_tokens
    out3 = scheduler.schedule()
    assert r1.request_id in out3.num_scheduled_tokens


def test_waiting_queue_plan_admits_repair():
    """The waiting-queue admission arm: an async external claim that
    fails leaves the request WAITING with a plan; the repair admits
    from the waiting queue with the same exclusivity. (H14: the flat
    channel no longer produces this shape -- it fails the whole claim
    closed -- so the truncated-parking shape is set up directly; in
    production the structured report path installs the plan.)"""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    scheduler.schedule()

    # The failed delivery shape: truncated to the failed boundary, still
    # parked (the flat channel's pre-H14 truncation, simulated directly).
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    r1.num_computed_tokens = 96 * BS
    scheduler.finished_recving_kv_req_ids.add(r1.request_id)
    plan = install_plan(scheduler, r1, E, failed_block_ids=[block_ids[96]])
    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out2.repair_works is not None
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
