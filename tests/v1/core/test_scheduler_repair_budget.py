# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hole-repair budget accounting through the real Scheduler.schedule().

Each test drives the REAL v0.31.0 Scheduler with real Request objects
through the existing kv-load-failure fixtures: an external-KV hit fails at
one block, the failure is reported, a pending RecoveryPlan is installed
(the "repair" policy's report arm), and the NEXT schedule() is the repair
admission.

The repair step is EXCLUSIVE (plan-first): the layer-limited repair
execution cannot share a forward with ordinary work, so a pending plan is
the first and only admission of its step. The equivalent-token charge (E)
still debits the step's input budget beside n_eff (the draft_slots
precedent).
"""

from unittest.mock import Mock

import pytest

from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm.v1.core.sched.scheduler import RecoveryPlan, Scheduler
from vllm.v1.outputs import (
    KVConnectorOutput,
    KVLoadAttestation,
    KVLoadFailureReport,
    RepairResultReport,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BS = 16  # fixture block_size
BUDGET = 4096
N_EFF = 1536  # the physical repair span [1536, 3072)
E = 2623  # a plan charge the budget tests reserve beside the span
# (chosen so E + n_eff straddles interesting budget boundaries; the
# real estimator derives it from layer counts at admission)
LAYER_END = 4
RESTORED = 3072  # the pre-failure external computed extent


def make_scheduler(**kwargs) -> Scheduler:
    vllm_config = create_vllm_config(
        model=kwargs.pop("model", "facebook/opt-125m"),
        # Same geometry as the original fixture model (16 hidden
        # layers, long context) so the validation battery's bounds
        # and the prompt-length fixtures are unchanged.
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", BUDGET),
        kv_load_failure_policy=kwargs.pop("kv_load_failure_policy", "recompute"),
    )
    # Pin the plain Scheduler (the traced class); the fixture default
    # resolves async_scheduling to the AsyncScheduler on this platform.
    vllm_config.scheduler_config.async_scheduling = False
    if not kwargs.pop("reserve_full_isl", True):
        vllm_config.scheduler_config.scheduler_reserve_full_isl = False
    if not kwargs.pop("enable_prefix_caching", True):
        # The v1 shared-block fence: with APC off, blocks are never
        # published, so a failed block is private and repairable.
        vllm_config.cache_config.enable_prefix_caching = False
    return create_scheduler(vllm_config, **kwargs)


def admit_async_external_hit(
    scheduler: Scheduler, request: Request, num_tokens: int
) -> None:
    """One connector Mock answering a full-prompt async external hit."""
    scheduler.connector = Mock()
    scheduler.connector.get_external_cache_hit_sources.return_value = None
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = ()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (num_tokens, True) if req is request else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()


def admit_sync_external_hit(
    scheduler: Scheduler, request: Request, num_tokens: int
) -> None:
    """One connector Mock answering a sync external hit: the request is
    admitted RUNNING with the external tokens committed as computed."""
    scheduler.connector = Mock()
    scheduler.connector.get_external_cache_hit_sources.return_value = None
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = ()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (num_tokens, False) if req is request else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()


def fail_and_truncate(
    scheduler: Scheduler, request: Request, out, failed_block_idx: int, executed=()
) -> None:
    """Report the async load of block `failed_block_idx` as failed."""
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    output = create_model_runner_output(
        reqs=list(executed),
        finished_recving={request.request_id},
        invalid_block_ids={block_ids[failed_block_idx]},
        use_eos=False,
    )
    scheduler.update_from_output(out, output)
    assert request.num_computed_tokens == failed_block_idx * BS


def install_plan(
    scheduler: Scheduler,
    request: Request,
    e: int,
    layer_end: int = LAYER_END,
    restored_extent: int = RESTORED,
    failed_block_ids: list[int] | None = None,
    span: tuple[int, int] | None = None,
    healthy_above_span: bool = True,
) -> RecoveryPlan:
    plan = RecoveryPlan(
        request_id=request.request_id,
        repair_span=span if span is not None else (96 * BS, 192 * BS),
        est_equivalent_tokens=e,
        layer_end=layer_end,
        restored_extent=restored_extent,
        failed_block_ids=failed_block_ids or [],
        plan_id=1,
        healthy_above_span=healthy_above_span,
    )
    scheduler.recovery_plans[request.request_id] = plan
    return plan


def drive_repair_result(
    scheduler: Scheduler,
    request: Request,
    out,
    structural_check: bool,
    plan: RecoveryPlan | None = None,
    *,
    plan_id: int | None = None,
    epoch: int | None = None,
    span: tuple[int, int] | None = None,
    aborted: bool = False,
    executed_layers: int = LAYER_END,
) -> None:
    """update_from_output of the repair step carrying the worker verdict.
    The repair step publishes no tokens (the kv-conn-only output shape).
    Defaults echo the pending plan's identity; the kwargs force the stale
    shapes."""
    if plan is not None and plan_id is None:
        plan_id = plan.plan_id
    if plan is not None and epoch is None:
        epoch = plan.admitted_step
    output = create_model_runner_output(reqs=[request], use_eos=False)
    output.sampled_token_ids = None
    output.kv_connector_output = KVConnectorOutput(
        repair_results=[
            RepairResultReport(
                request_id=request.request_id,
                plan_id=plan_id if plan_id is not None else 1,
                epoch=epoch if epoch is not None else -1,
                structural_check=structural_check,
                executed_layers=executed_layers,
                span=span if span is not None else (96 * BS, 192 * BS),
                aborted=aborted,
            )
        ]
    )
    scheduler.update_from_output(out, output)


def test_repair_step_is_exclusive_and_admits_span():
    """The repair admission schedules exactly the span, exclusively: no
    other request shares the step (the layer-limited execution's v1
    constraint), the competitor is admitted the very next step, and the
    commit restores the logical extent."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    plan = install_plan(scheduler, r1, E)
    r2 = create_request(num_tokens=3000)
    scheduler.add_request(r2)

    out2 = scheduler.schedule()

    # The worker geometry stays physical: the repair step schedules n_eff.
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
    # Exclusivity: the competitor does NOT share the repair step.
    assert r2.request_id not in out2.num_scheduled_tokens
    assert out2.total_num_scheduled_tokens == N_EFF
    assert out2.repair_works is not None
    assert out2.repair_works[r1.request_id].layer_end == LAYER_END
    assert out2.repair_works[r1.request_id].restored_extent == RESTORED
    # The optimistic advance covers exactly the repair span: B -> corrupt_end.
    assert r1.num_computed_tokens == 96 * BS + N_EFF
    assert plan.state == "ADMITTED"

    # The commit: the logical extent is restored.
    drive_repair_result(scheduler, r1, out2, structural_check=True, plan=plan)
    assert r1.num_computed_tokens == RESTORED
    assert scheduler.recovery_plans == {}

    # The competitor resumes at the next step with the full budget.
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r2.request_id] == min(3000, BUDGET)
    assert r1.request_id not in out3.num_scheduled_tokens


def test_repair_abort_falls_back_to_recompute_boundary():
    """A failed structural check truncates to the rollback boundary (the
    ordinary rollback-recompute reschedule owns the request)."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)

    install_plan(scheduler, r1, E)

    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
    assert r1.num_computed_tokens == 96 * BS + N_EFF
    plan = scheduler.recovery_plans[r1.request_id]

    drive_repair_result(scheduler, r1, out2, structural_check=False, plan=plan)
    assert r1.num_computed_tokens == 96 * BS
    assert scheduler.recovery_plans == {}

    # The ordinary reschedule owns the request from the boundary.
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] == N_EFF


def test_report_arm_installs_plan_and_skips_detection_output():
    """The structured report arm (the sync shape): truncation to the
    report's rollback boundary, the plan installed with the estimator's E,
    and the detection step's garbage output skipped for the affected
    request. APC OFF: the failed block is private (the v1 fence)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens[r1.request_id] == 256
    assert r1.status == RequestStatus.RUNNING
    # The optimistic advance: 3072 external + 256 scheduled.
    assert r1.num_computed_tokens == 3328

    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    report = KVLoadFailureReport(
        request_id=r1.request_id,
        failed_block_ids=[block_ids[96]],
        token_span=(96 * BS, 97 * BS),
        rollback_boundary=96 * BS,
        observed_failed_layers=[3],
        healthy_above_span=True,
        restored_extent=3072,
        transfer_generation=scheduler._req_transfer_generation[r1.request_id],
    )
    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=[report],
        # The conforming positive half of the evidence: the reporting
        # rank's terminal, fully certified attestation (TP=1 fixture),
        # stamped with the attempt's identity.
        load_attestations=[
            _terminal_attestation(
                r1.request_id,
                generation=scheduler._req_transfer_generation.get(r1.request_id),
            )
        ],
    )
    n_output_tokens_before = len(r1.output_token_ids)
    scheduler.update_from_output(out1, output)

    assert r1.num_computed_tokens == 96 * BS
    # The garbage sampled token was skipped, not appended.
    assert len(r1.output_token_ids) == n_output_tokens_before
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.repair_span == (96 * BS, 97 * BS)
    assert plan.layer_end == 4
    assert plan.restored_extent == 3072
    assert plan.est_equivalent_tokens > 0

    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == plan.n_eff
    assert out2.repair_works is not None
    assert out2.repair_works[r1.request_id].layer_end == 4


def test_no_plan_baseline_undercharges():
    """Control: without a plan the competitor shares the step and gets
    budget - n_eff (2560). The plan-first exclusivity (not the charge) is
    what separates this from the repair-step shape."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)

    r2 = create_request(num_tokens=3000)
    scheduler.add_request(r2)

    out2 = scheduler.schedule()

    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
    assert out2.num_scheduled_tokens[r2.request_id] == BUDGET - N_EFF


def test_over_budget_plan_falls_back_to_ordinary_reschedule():
    """E > the whole per-step budget: the plan dissolves (no step can fund
    it -- deferral would starve), the exclusivity constraint lifts within
    the step, and the ordinary rollback-recompute reschedule owns the
    request at ordinary budget. The owner skips the dissolution step
    itself (the zero schedule keeps the plan's threshold exemption from
    leaking an uncapped chunk) and resumes the next."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)

    install_plan(scheduler, r1, E + BUDGET)  # 6719 > 4096
    r2 = create_request(num_tokens=3000)
    scheduler.add_request(r2)

    out2 = scheduler.schedule()

    assert scheduler.recovery_plans == {}
    # The healthy competitor rides the dissolution step (the freeze does
    # not outlive the plan); the damaged request resumes next step.
    assert out2.num_scheduled_tokens[r2.request_id] > 0
    assert r1.request_id not in out2.num_scheduled_tokens
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] > 0


def test_plan_first_repairs_before_healthy_running_work():
    """A healthy RUNNING request does not share the repair step: it is
    held one step and resumes immediately after."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    r0 = create_request(num_tokens=6000)
    scheduler.add_request(r0)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens[r0.request_id] == BUDGET
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96, executed=[r0])

    plan = install_plan(scheduler, r1, E)
    r2 = create_request(num_tokens=3000)
    scheduler.add_request(r2)

    # Step 2 (the repair step): only the plan's request is admitted.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == N_EFF
    assert r0.request_id not in out2.num_scheduled_tokens
    assert r2.request_id not in out2.num_scheduled_tokens
    assert r0.num_computed_tokens == BUDGET  # held, not advanced
    assert plan.state == "ADMITTED"

    # Step 3: the healthy work resumes (R0's 1905-token continuation).
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r0.request_id] == 1905


def test_preemption_voids_plan():
    """Preemption while the plan is pending voids it: the preempted
    request recomputes its whole prompt, so the repair is pointless."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)

    install_plan(scheduler, r1, E)
    assert r1.request_id in scheduler.recovery_plans

    r1.status = RequestStatus.RUNNING
    scheduler.running.append(r1)
    scheduler._preempt_request(r1, timestamp=0.0)

    assert scheduler.recovery_plans == {}
    assert r1.status == RequestStatus.PREEMPTED
    assert r1.num_computed_tokens == 0
    assert scheduler.waiting.peek_request() is r1


def _structured_report(scheduler, r1, failed_block_idx=96, **overrides):
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    fields = dict(
        request_id=r1.request_id,
        failed_block_ids=[block_ids[failed_block_idx]],
        token_span=(96 * BS, 97 * BS),
        rollback_boundary=96 * BS,
        observed_failed_layers=[3],
        healthy_above_span=True,
        restored_extent=3072,
        # The conforming shape: the aggregator-stamped rank (the
        # production evidence always carries it) and the attempt's
        # identity (the scheduler assigned it at load scheduling).
        source_rank=0,
        transfer_generation=scheduler._req_transfer_generation.get(r1.request_id),
    )
    fields.update(overrides)
    return KVLoadFailureReport(**fields)


def _terminal_attestation(
    request_id,
    *,
    rank=0,
    expected=192,
    loaded=192,
    failed=0,
    certification="sha256",
    certified=192,
    cert_failures=0,
    terminal=True,
    generation=None,
    **overrides,
):
    """A conforming per-rank attestation: terminal, accounting for the
    complete expected object set, with full SHA certification and its
    attempt's transfer identity. 192 objects = the fixture's
    3072-token external claim at BS=16. The generation defaults to
    None; callers with scheduler access stamp it (as _deliver_report
    does) or pass it explicitly."""
    fields = dict(
        request_id=request_id,
        expected_objects=expected,
        loaded_objects=loaded,
        failed_objects=failed,
        certification=certification,
        certified_objects=certified,
        certification_failures=cert_failures,
        terminal=terminal,
        transfer_generation=generation,
        source_rank=rank,
    )
    fields.update(overrides)
    return KVLoadAttestation(**fields)


def _deliver_report(scheduler, r1, out, report, reqs=None, attestations=None):
    """Deliver a report through update_from_output with the CONFORMING
    evidence shape: a terminal attestation accompanies the report (the
    fixture scheduler is TP=1), both stamped with the request's latest
    scheduled attempt identity and RECONCILED with the report (the
    attestation's failed count equals the report's failed regions;
    everything that loaded is certified), so the completeness, identity,
    and reconciliation gates see consistent evidence. Tests exercising
    the incomplete- or stale-evidence shapes pass their own (or no)
    attestations and override the report's transfer_generation."""
    output = create_model_runner_output(reqs=reqs or [r1], use_eos=False)
    if attestations is None:
        failed = len(report.failed_block_ids)
        attestations = [
            _terminal_attestation(
                r1.request_id,
                loaded=192 - failed,
                failed=failed,
                certified=192 - failed,
                generation=scheduler._req_transfer_generation.get(r1.request_id),
            )
        ]
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=[report],
        load_attestations=attestations,
    )
    scheduler.update_from_output(out, output)


def _sync_claim_scenario_helper():
    """A repair-policy scheduler with one sync external claim, scheduled:
    the (scheduler, request, scheduler_output) triple a report delivery
    needs."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    return scheduler, r1, out1


def test_delayed_publication_keeps_sync_claim_private_until_verdict():
    """The flip of the v1 published-block fence: under the repair
    policy + APC ON the sync external-claim blocks are NOT hashed into
    the prefix cache at admission (the content-blind publication is
    delayed until the step's delivery verdict), so the report's write
    set is private and unpublished -- the fence passes and the plan
    installs instead of rejecting. The failed span never becomes
    matchable."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool
    # At admission nothing is published: the claim span (and the
    # optimistic incoming span) stay unmatchable until the update.
    for bid in block_ids:
        assert block_pool.blocks[bid].block_hash is None
    assert block_pool.blocks[block_ids[96]].ref_cnt == 1

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))

    # The write set is private + unpublished: the plan installs.
    assert r1.request_id in scheduler.recovery_plans
    assert r1.num_computed_tokens == 96 * BS
    # The failed span stays unmatchable (never published).
    for bid in block_ids[96:]:
        assert block_pool.blocks[bid].block_hash is None

    # The repair admission owns the request from the boundary.
    out2 = scheduler.schedule()
    assert out2.repair_works is not None


def test_delayed_publication_no_same_pass_sharing_of_unverified_claim():
    """Two same-prompt requests, the first carrying a sync external
    claim. Under the old admission-time publication the second matched
    the first's published claim blocks in the SAME schedule() pass
    (ref_cnt 2 -- poisoned content physically reachable); the delayed
    publication keeps the claim unmatchable, so the same-batch sharer
    cannot exist and the failed content is unreachable by construction."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328, common_prefix_len=1600)
    scheduler.add_request(r1)
    r2 = create_request(num_tokens=3328, common_prefix_len=1600)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()

    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    (r2_blocks,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)
    # No shared physical blocks: r2 could not match r1's unverified claim.
    assert set(r1_blocks).isdisjoint(r2_blocks)
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.blocks[r1_blocks[96]].ref_cnt == 1

    _deliver_report(
        scheduler, r1, out1, _structured_report(scheduler, r1), reqs=[r1, r2]
    )

    # The write set is private + unpublished: the plan installs, and r2
    # is untouched by r1's failure channel.
    assert r1.request_id in scheduler.recovery_plans
    assert r1.num_computed_tokens == 96 * BS
    for bid in r1_blocks[96:]:
        assert block_pool.blocks[bid].block_hash is None


def test_delayed_publication_publishes_verified_claim_at_update():
    """A healthy sync external claim with COMPLETE, CONTENT-CERTIFIED
    evidence: nothing is matchable at admission; the step's update
    publishes the verified extent, and a later same-prompt request
    matches it as ordinary prefix-cached blocks.

    H14-01/H14-02: publication is no longer the default outcome of a
    step with no failure channel -- it requires the positive evidence
    (a terminal, sha-certified attestation from every TP rank of the
    attempt). The no-evidence shape (the pre-H14 behavior this test
    used to pin) fails closed instead and is covered by the H14 module
    (test_sync_claim_no_evidence_never_publishes)."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool
    for bid in r1_blocks:
        assert block_pool.blocks[bid].block_hash is None

    # The step completed with the conforming positive evidence for the
    # attempt (TP=1: one terminal, fully sha-certified attestation).
    generation = scheduler._req_transfer_generation[r1.request_id]
    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        load_attestations=[_terminal_attestation(r1.request_id, generation=generation)]
    )
    scheduler.update_from_output(out1, output)
    assert not scheduler._delayed_sync_claim_pubs
    for bid in r1_blocks:
        assert block_pool.blocks[bid].block_hash is not None
    assert r1.num_computed_tokens == 3328
    assert r1.request_id not in scheduler._untrusted_external_reqs

    # A later same-prompt request now matches the published (verified)
    # claim as ordinary prefix-cached blocks.
    r2 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r2)
    scheduler.schedule()
    (r2_blocks,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)
    assert set(r1_blocks) & set(r2_blocks)


def test_delayed_publication_skips_aborted_request():
    """A request aborted between its delayed-publication admission and
    the update is not published: its blocks were freed, and publishing
    would map the prefix hashes onto freed blocks."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool

    scheduler.finish_requests(r1.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert not scheduler._delayed_sync_claim_pubs
    for bid in block_ids:
        assert block_pool.blocks[bid].block_hash is None


def test_delayed_publication_default_policy_preserves_admission_publication():
    """Off the repair policy the upstream behavior is untouched: a sync
    external claim is published at admission (the old semantics -- the
    v1 fence's published-block trigger), and no delayed set is kept."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.blocks[block_ids[96]].block_hash is not None

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert block_pool.blocks[block_ids[96]].block_hash is not None


def test_report_for_claimless_request_dropped_not_fenced():
    """The identity gate precedes the fences: a connector reporting a
    load failure for a request that never had an external load attempt
    (no transfer identity) is contract-bogus. The report is dropped --
    loudly -- and the request is untouched: no plan, no truncation, and
    its locally published blocks stay published (there is no external
    data to distrust). (The pre-identity version of this test drove
    the same synthetic report into the write-set fence; under the
    transfer-identity contract the drop is the correct, earlier,
    fail-safe.)"""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r1)
    # No external hit: an ordinary local prefill; APC ON publishes at
    # admission.
    admit_sync_external_hit(scheduler, r1, 0)

    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool
    assert block_pool.blocks[block_ids[96]].block_hash is not None
    assert r1.request_id not in scheduler._req_transfer_generation

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))

    assert scheduler.recovery_plans == {}
    # Untouched: the request keeps its computed state and its
    # publications.
    assert r1.num_computed_tokens == 3328
    assert block_pool.blocks[block_ids[96]].block_hash is not None


def test_fence_rejects_repair_into_published_span_block():
    """The write-set fence stays fail-closed for genuinely published
    blocks: a request with a REAL external claim (a transfer identity)
    whose reported span block has been published through any path --
    here the ordinary publication of the request's verified prefix
    (kv_cache_manager.cache_blocks, the same call the delayed-claim
    publisher uses) -- refuses the repair loudly. The delayed
    publication narrows the fence's trigger surface for the request's
    OWN claim; it does not remove the fence."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool
    assert r1.request_id in scheduler._req_transfer_generation
    # The claim itself is unpublished (the delayed publication).
    assert block_pool.blocks[block_ids[96]].block_hash is None

    # Any other publication path (a verified prefix of the same request,
    # another request's cached blocks) publishes the span block.
    scheduler.kv_cache_manager.cache_blocks(r1, 97 * BS)
    assert block_pool.blocks[block_ids[96]].block_hash is not None

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS
    # Everything from the boundary downstream is evicted: no future
    # request matches the untrusted span.
    for bid in block_ids[96:]:
        assert block_pool.blocks[bid].block_hash is None


def make_result(
    request: Request,
    structural_check: bool,
    plan: RecoveryPlan,
    *,
    plan_id: int | None = None,
    epoch: int | None = None,
    span: tuple[int, int] | None = None,
    aborted: bool = False,
    executed_layers: int = LAYER_END,
) -> RepairResultReport:
    return RepairResultReport(
        request_id=request.request_id,
        plan_id=plan_id if plan_id is not None else plan.plan_id,
        epoch=epoch if epoch is not None else plan.admitted_step,
        structural_check=structural_check,
        executed_layers=executed_layers,
        span=span if span is not None else plan.repair_span,
        aborted=aborted,
    )


def test_stale_repair_result_rejected_on_identity():
    """A repair result that does not echo the pending plan's identity
    (plan_id / epoch / span) is a stale report: rejected loudly, the plan
    untouched, the request's count not corrupted."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    install_plan(scheduler, r1, E)
    out2 = scheduler.schedule()
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.state == "ADMITTED"
    assert plan.admitted_step is not None

    for stale in (
        make_result(r1, True, plan, plan_id=999),
        make_result(r1, True, plan, epoch=plan.admitted_step - 1),
        make_result(r1, True, plan, span=(0, BS)),
    ):
        scheduler._consume_repair_result(stale)
        assert scheduler.recovery_plans[r1.request_id] is plan
        assert r1.num_computed_tokens == 96 * BS + N_EFF

    # The matching identity still commits after the stale rejections
    # (through the full update_from_output path).
    drive_repair_result(scheduler, r1, out2, structural_check=True, plan=plan)
    assert r1.num_computed_tokens == RESTORED
    assert scheduler.recovery_plans == {}


def test_duplicate_repair_result_rejected_after_commit():
    """The plan is consumed exactly once: after the commit the plan is
    purged, so a duplicate delivery of the same result is a stale report
    (no plan) and must not move the request's count a second time."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    install_plan(scheduler, r1, E)
    scheduler.schedule()
    plan = scheduler.recovery_plans[r1.request_id]

    scheduler._consume_repair_result(make_result(r1, True, plan))
    assert r1.num_computed_tokens == RESTORED
    assert scheduler.recovery_plans == {}

    scheduler._consume_repair_result(make_result(r1, True, plan))
    assert r1.num_computed_tokens == RESTORED
    assert scheduler.recovery_plans == {}


def test_duplicate_verdict_list_consumed_once():
    """The TP-aggregation shape: every rank reports the same verdict, so
    the result list repeats the request. The first verdict commits and
    the duplicates are rejected as stale (the plan is already consumed)
    -- no double restore."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    install_plan(scheduler, r1, E)
    out2 = scheduler.schedule()
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.state == "ADMITTED"

    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.sampled_token_ids = None
    output.kv_connector_output = KVConnectorOutput(
        repair_results=[make_result(r1, True, plan) for _ in range(2)]
    )
    scheduler.update_from_output(out2, output)

    assert r1.num_computed_tokens == RESTORED
    assert scheduler.recovery_plans == {}


def test_duplicate_report_absorbed_divergent_falls_back():
    """The report arm's duplicate handling: an identical report for an
    existing plan is absorbed (idempotent, no second plan); a divergent
    report names a different region and is REFUSED as evidence -- the
    plan dissolves and the request falls back to the ordinary
    rollback-recompute path without killing the engine (the hardening
    contract: broken evidence invalidates the recovery, not the
    request)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    assert scheduler.recovery_plans[r1.request_id] is plan
    assert r1.num_computed_tokens == 96 * BS

    divergent = _structured_report(
        scheduler,
        r1,
        failed_block_idx=97,
        token_span=(97 * BS, 98 * BS),
        rollback_boundary=97 * BS,
    )
    # No raise: the divergent evidence dissolves the plan and truncates to
    # the minimum certified rollback (the plan's 96 blocks, below the
    # divergent report's own 97).
    _deliver_report(scheduler, r1, out1, divergent)
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS

    # A divergent report that is ALSO malformed (its span/block claim does
    # not correspond) is uninterpretable down to the rollback boundary:
    # the full-recompute floor applies.
    scheduler2, r2, out2 = _sync_claim_scenario_helper()
    _deliver_report(scheduler2, r2, out2, _structured_report(scheduler2, r2))
    malformed_divergent = _structured_report(
        scheduler2, r2, token_span=(97 * BS, 98 * BS), rollback_boundary=97 * BS
    )
    _deliver_report(scheduler2, r2, out2, malformed_divergent)
    assert scheduler2.recovery_plans == {}
    assert r2.num_computed_tokens == 0

    # The engine keeps scheduling: the ordinary recompute path owns the
    # request from the rollback boundary.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS
    assert out2.repair_works is None


def test_abort_evicts_failed_span_from_prefix_cache():
    """The ABORT terminal state invalidates the partial writes: the failed
    block and everything downstream are evicted from the prefix cache (no
    future request can match untrustworthy content), the request truncates
    to the rollback boundary, and the ordinary reschedule owns it."""
    # The RUNNING-arm admission (this test's sync claim leaves the request
    # RUNNING, unlike the waiting-queue arms) reserves the plan's full
    # equivalent-token charge before the span cap, so the step budget must
    # satisfy max_num_batched_tokens >= E + n_eff: the default 4096 leaves
    # 4096 - 2623 = 1473 < n_eff = 1536 and the admission would defer
    # forever, never reaching the abort this test exists to exercise.
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    # Published blocks are rejected by the shared-block fence; install the
    # plan directly with the failed block to drive the abort consumption.
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    plan = install_plan(scheduler, r1, E, failed_block_ids=[block_ids[96]])
    block_pool = scheduler.kv_cache_manager.block_pool

    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out2.repair_works is not None
    work = out2.repair_works[r1.request_id]
    assert work.repair_span == plan.repair_span
    assert work.expected_write_block_ids

    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=False,
        plan=plan,
        aborted=True,
        executed_layers=2,
    )
    assert r1.num_computed_tokens == 96 * BS
    assert scheduler.recovery_plans == {}
    for bid in block_ids[96:]:
        assert block_pool.blocks[bid].block_hash is None

    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS


def test_write_set_fence_rejects_shared_span_block_outside_failed_region():
    """The full-write-set fence: the repair stream writes the span's
    blocks at EVERY executed layer, so a shared block INSIDE the repair
    span but OUTSIDE the failed region must refuse the repair too. A
    failed-blocks-only fence would pass this report and install a plan
    over a block another request's table reads."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool
    # The mixed local-adopt + external-claim shape: the rollback boundary
    # sits two blocks below the failed region, and one span block
    # (94 -- between the rollback and the failed block 96) is held by a
    # second table (ref_cnt 2). The failed block itself stays private and
    # unpublished, so the failed-blocks-only fence passes it.
    block_pool.blocks[block_ids[94]].ref_cnt += 1
    assert block_pool.blocks[block_ids[96]].ref_cnt == 1
    assert block_pool.blocks[block_ids[96]].block_hash is None

    _deliver_report(
        scheduler,
        r1,
        out1,
        _structured_report(scheduler, r1, rollback_boundary=94 * BS),
    )

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 94 * BS

    # The ordinary recompute reschedule owns the request from the boundary.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == 3328 - 94 * BS
    assert out2.repair_works is None


def test_unhealthy_above_span_caps_restoration_at_span_end():
    """The conditional extent restoration: when the report carries
    healthy_above_span=False the above-span external ranges are NOT
    certified byte-exact, so the COMMIT must NOT restore the original
    cached extent C -- it restores to the repair span's end (the largest
    commit-checked boundary) and the resume recomputes the untrusted
    [span_end, C) extent through the ordinary path."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    _deliver_report(
        scheduler, r1, out1, _structured_report(scheduler, r1, healthy_above_span=False)
    )
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.healthy_above_span is False

    out2 = scheduler.schedule()
    assert out2.repair_works is not None

    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=True,
        plan=plan,
        span=plan.repair_span,
    )

    # NOT restored_extent (3072): the restoration is capped at the span end.
    assert r1.num_computed_tokens == 97 * BS
    assert scheduler.recovery_plans == {}

    # The resume recomputes the untrusted above-span extent.
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] == 3328 - 97 * BS
    assert out3.repair_works is None


def test_report_validation_battery_rejects_malformed_geometry():
    """The admission validation battery: a report whose geometry does not
    check out is a connector contract violation -- refused loudly (no
    plan ever installed): block ownership, span block-alignment, nonempty
    in-range failed layers, and the token-span/block-ID correspondence."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    (foreign_blocks,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)

    cases = {
        "ownership": (
            _structured_report(scheduler, r1, failed_block_ids=[foreign_blocks[5]]),
            "block ownership violation",
        ),
        "rollback_alignment": (
            _structured_report(scheduler, r1, rollback_boundary=96 * BS - 3),
            "not block-aligned",
        ),
        "span_end_alignment": (
            _structured_report(scheduler, r1, token_span=(96 * BS, 97 * BS + 1)),
            "not block-aligned",
        ),
        "layers_empty": (
            _structured_report(scheduler, r1, observed_failed_layers=[]),
            "invalid failed-layer information",
        ),
        "layers_out_of_range": (
            _structured_report(scheduler, r1, observed_failed_layers=[999]),
            "invalid failed-layer information",
        ),
        "correspondence": (
            _structured_report(
                scheduler,
                r1,
                token_span=(95 * BS, 96 * BS),
                rollback_boundary=95 * BS,
            ),
            "correspondence violation",
        ),
    }
    for name, (report, _message) in cases.items():
        # No raise (the hardening contract): a malformed report is a
        # request-scoped fail-closed fallback -- logged at ERROR,
        # recovery refused, full recompute -- never an engine-wide
        # crash. The scheduler keeps serving afterwards.
        _deliver_report(scheduler, r1, out1, report, reqs=[r1, r2])
        assert scheduler.recovery_plans == {}, f"plan leaked after {name}"
        assert r1.num_computed_tokens == 0, f"not reset to recompute after {name}"

    # The engine keeps scheduling after every malformed report.
    out2 = scheduler.schedule()
    assert r1.request_id in out2.num_scheduled_tokens


def _admitted_repair_plan():
    """A fresh scenario driven to the ADMITTED repair plan."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    install_plan(scheduler, r1, E)
    out2 = scheduler.schedule()
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.state == "ADMITTED"
    return scheduler, r1, out2, plan


def test_malformed_committed_result_diverts_to_recompute_fallback():
    """The result is untrusted input: an aborted verdict that claims a
    passing structural check, or a negative executed layer count, must
    never advance the frontier -- the fail-safe recompute fallback owns
    the request."""
    for malformed_kwargs in (
        {"aborted": True},
        {"executed_layers": -1},
    ):
        scheduler, r1, out2, plan = _admitted_repair_plan()
        scheduler._consume_repair_result(
            make_result(r1, True, plan, **malformed_kwargs)
        )
        assert r1.num_computed_tokens == 96 * BS
        assert scheduler.recovery_plans == {}
        # The ordinary reschedule owns the request from the boundary.
        out3 = scheduler.schedule()
        assert out3.repair_works is None
        assert out3.num_scheduled_tokens[r1.request_id] == N_EFF


def test_missing_repair_result_fails_safe():
    """A repair step whose update carries NO verdict for an admitted work
    must not leave the plan ADMITTED: the synthesized failed verdict
    drives the abort fallback (the rollback floor), never a silent retry
    loop."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3328)
    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)
    install_plan(scheduler, r1, E)
    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.sampled_token_ids = None
    output.kv_connector_output = KVConnectorOutput()
    scheduler.update_from_output(out2, output)
    assert r1.num_computed_tokens == 96 * BS
    assert scheduler.recovery_plans == {}
    out3 = scheduler.schedule()
    assert out3.repair_works is None
    assert out3.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS


def _drive_repair_txn_step():
    """Drive to the exclusive repair step's SchedulerOutput (the sync
    external-claim shape: the plan installed from a structured report,
    the repair admitted with its blocks left unpublished)."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None

    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    assert r1.request_id in scheduler._delayed_repair_txn_pubs
    return scheduler, r1, out2, plan


def test_repair_txn_publishes_only_on_commit():
    """A repair-step admission publishes its blocks ONLY on the accepted
    verdict: before the update nothing is matchable; a committed verdict
    publishes the verified (committed) extent."""
    scheduler, r1, out2, plan = _drive_repair_txn_step()
    block_pool = scheduler.kv_cache_manager.block_pool
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    # Before the verdict: the repair-step admission left everything
    # unmatchable (the transactional-publication coverage).
    for bid in block_ids:
        assert block_pool.blocks[bid].block_hash is None

    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=True,
        plan=plan,
        span=plan.repair_span,
    )

    assert scheduler._delayed_repair_txn_pubs == set()
    assert r1.num_computed_tokens == RESTORED
    assert scheduler.recovery_plans == {}
    # Committed: the verified extent is published and matchable again...
    for bid in block_ids[: RESTORED // BS]:
        assert block_pool.blocks[bid].block_hash is not None
    # ...while the locally-computed tail above the committed extent
    # stays unpublished until its own ordinary publication.
    for bid in block_ids[RESTORED // BS :]:
        assert block_pool.blocks[bid].block_hash is None


def test_repair_txn_aborted_never_publishes():
    """An aborted verdict leaves the transaction's blocks unpublished
    (the abort's eviction unpublishes the span and above; the recompute
    fallback republishes through the ordinary path after its rewrite)."""
    scheduler, r1, out2, plan = _drive_repair_txn_step()
    block_pool = scheduler.kv_cache_manager.block_pool
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    for bid in block_ids:
        assert block_pool.blocks[bid].block_hash is None

    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=False,
        plan=plan,
        span=plan.repair_span,
        aborted=True,
        executed_layers=2,
    )

    assert scheduler._delayed_repair_txn_pubs == set()
    assert r1.num_computed_tokens == 96 * BS
    assert scheduler.recovery_plans == {}
    # Aborted: nothing ever publishes (the span and above are evicted;
    # the un-failed prefix was never hashed).
    for bid in block_ids:
        assert block_pool.blocks[bid].block_hash is None


def test_pending_plan_suppresses_step_outputs():
    """While a recovery plan is pending, an arriving step output for the
    damaged request is suppressed: under async scheduling a schedule-ahead
    step's sample is computed over the untrusted state and must never be
    appended (the verdict consumption restores the request)."""
    scheduler = make_scheduler()
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)

    out1 = scheduler.schedule()
    fail_and_truncate(scheduler, r1, out1, failed_block_idx=96)

    plan = install_plan(scheduler, r1, E)

    speculative = create_model_runner_output(reqs=[r1], use_eos=False)
    speculative.sampled_token_ids = [[7]]
    scheduler.update_from_output(out1, speculative)

    assert r1.num_output_tokens == 0
    assert scheduler.recovery_plans[r1.request_id] is plan


def test_stale_only_verdict_set_fails_safe():
    """The final audit's FA-1: a verdict set that is ENTIRELY stale (no
    echo matches the ADMITTED plan -- the shape H6's work-lookup
    placeholder verdict produces, or a worker executing a stale
    schedule) must not leave the plan ADMITTED indefinitely. The
    missing-verdict synthesis fires for stale dispositions too: the plan
    is consumed by the abort fallback and the request reschedules."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", max_num_batched_tokens=8192
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    _deliver_report(scheduler, r1, out1, _structured_report(scheduler, r1))
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    plan = install_plan(scheduler, r1, E, failed_block_ids=[block_ids[96]])
    out2 = scheduler.schedule()
    assert plan.state == "ADMITTED"
    assert out2.repair_works is not None

    # A stale-only verdict set: the echo matches NOTHING admitted (the
    # placeholder shape: plan_id -1, foreign span).
    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.sampled_token_ids = None
    output.kv_connector_output = KVConnectorOutput(
        repair_results=[
            RepairResultReport(
                request_id=r1.request_id,
                plan_id=-1,
                epoch=-1,
                structural_check=True,
                executed_layers=4,
                span=(-1, -1),
                aborted=False,
            )
        ]
    )
    scheduler.update_from_output(out2, output)

    # The plan is CONSUMED by the synthesized failed verdict (not left
    # ADMITTED forever), the request truncates to the rollback boundary,
    # and the ordinary reschedule owns it.
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r1.request_id] == 3328 - 96 * BS
