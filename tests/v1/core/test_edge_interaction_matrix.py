# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Agent F -- the edge-case INTERACTION matrix (scheduler side).

Isolated-function tests exist for each mechanism in
tests/v1/core/test_scheduler_{transfer_generation,transfer_evidence,
h14_publication,repair_budget,repair_invalidation,report_consistency}.py;
this module drives the COMBINATIONS where those mechanisms meet inside
one Scheduler instance -- the shapes the mission brief lists as the
places bugs hide. Every test names the wrong implementation it would
catch (non-vacuity contract).

Shapes covered here (producer-side shapes live in
tests/v1/kv_connector/unit/test_evidence_edge_interactions.py):

S1  retry + delayed old attestation + request-ID reuse        -> matrix_01
S2  preemption + partial receive + physical block reuse       -> matrix_02
S3  multiple TP ranks + one rank exception + async completion -> matrix_03
S4  multiple failed blocks + valid-looking per-region reports -> matrix_04
S5  evidence arriving after an attempt is superseded          -> matrix_05
S6  duplicate terminal notifications                          -> matrix_06*
S7  repeated abort/commit cycles                              -> matrix_07
S8  cleanup during a pending async transfer                   -> matrix_08
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    _structured_report,
    _terminal_attestation,
    admit_async_external_hit,
    admit_sync_external_hit,
    drive_repair_result,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def deliver(scheduler, out, reqs, reports=(), attestations=(), **kv_extra):
    """update_from_output carrying one connector output built from the
    given structured-evidence channels (plus optional async channels)."""
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    fields = dict(
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    fields.update(kv_extra)
    output.kv_connector_output = KVConnectorOutput(**fields)
    scheduler.update_from_output(out, output)


def att_reported(scheduler, request, rank=0, failed=1, generation=None):
    """A conforming report-bearing rank attestation (failed cells
    reconciled with the single-block report, loaded set certified)."""
    return _terminal_attestation(
        request.request_id,
        rank=rank,
        loaded=192 - failed,
        failed=failed,
        certified=192 - failed,
        generation=generation
        if generation is not None
        else scheduler._req_transfer_generation.get(request.request_id),
    )


def gen_of(scheduler, request):
    return scheduler._req_transfer_generation[request.request_id]


def preempt(scheduler, request):
    """Take a RUNNING request out of the running queue and preempt it (the
    test_preemption_voids_plan fixture pattern)."""
    scheduler.running.remove(request)
    scheduler._preempt_request(request, timestamp=0.0)
    assert request.status == RequestStatus.PREEMPTED


def async_scenario(tp_size=1, apc=False):
    """A repair-policy scheduler holding ONE async external claim that is
    scheduled and in flight (WAITING_FOR_REMOTE_KS)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=apc
    )
    scheduler._evidence_tp_size = tp_size
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    return scheduler, r1, out1


def all_unpublished(scheduler, request) -> bool:
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    pool = scheduler.kv_cache_manager.block_pool
    return all(pool.blocks[b].block_hash is None for b in block_ids)


# ---------------------------------------------------------------------------
# S1: retry + delayed old attestation + request-ID reuse.
# ---------------------------------------------------------------------------


def test_matrix_01_retry_with_delayed_old_attestation_and_id_reuse():
    """An async attempt fails flatly; the promotion releases the claim and
    the request retries LOCALLY, finishes, and a NEW request REUSES its id
    with a fresh external attempt (a new generation). The FIRST request's
    terminal attestation -- the delayed straggler -- arrives during the
    second attempt's flight together with attempt 2's own rank-1
    attestation, then attempt 2's rank-0 attestation completes the
    delivery.

    Catches: an implementation that folds ANY arriving attestation into
    the request's accumulated state (a fold keyed on request_id only, no
    generation partition): the stale rank-0 attestation of attempt 1
    would satisfy the completeness gate of attempt 2, so a rank that
    loaded nothing for attempt 2 vouches for it and the unverified retry
    publishes at promotion. With the partition, the stale attestation is
    dropped and the promotion only succeeds once attempt 2's OWN rank-0
    attestation arrives."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    scheduler._evidence_tp_size = 2
    r1 = create_request(num_tokens=3328, request_id=1)
    r1.request_id = "reused"
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    gen1 = gen_of(scheduler, r1)

    # Attempt 1 ends flatly: the promotion (inside the next schedule)
    # releases the claim and marks the source untrusted; the retry is a
    # full local recompute.
    deliver(
        scheduler,
        out1,
        [],
        finished_recving={r1.request_id},
        failed_recving={r1.request_id},
    )
    scheduler.schedule()
    assert r1.status == RequestStatus.RUNNING
    assert r1.request_id in scheduler._untrusted_external_reqs

    # The request finishes; a NEW request reuses the id and schedules a
    # fresh external attempt (the untrusted fence died with the old
    # request: _free_request discards it).
    scheduler.finish_requests(r1.request_id, RequestStatus.FINISHED_STOPPED)
    r2 = create_request(num_tokens=3328, request_id=1)
    r2.request_id = "reused"
    assert r2.request_id == r1.request_id
    scheduler.add_request(r2)
    admit_async_external_hit(scheduler, r2, 3072)
    out2 = scheduler.schedule()
    gen2 = gen_of(scheduler, r2)
    assert gen2 > gen1

    # The delayed OLD attestation (attempt 1's identity, rank 0) arrives
    # while attempt 2 is in flight, together with attempt 2's own
    # rank-1 attestation. The stale one must not accumulate.
    deliver(
        scheduler,
        out2,
        [],
        attestations=[
            _terminal_attestation(r2.request_id, rank=0, generation=gen1),
            _terminal_attestation(r2.request_id, rank=1, generation=gen2),
        ],
    )
    accumulated = scheduler._req_transfer_attestations.get(r2.request_id, [])
    assert [a.source_rank for a in accumulated] == [1], (
        "the stale attempt-1 attestation must not accumulate"
    )

    # Attempt 2's own rank-0 attestation completes the delivery: the
    # promotion publishes (both ranks terminal, certified, current).
    deliver(
        scheduler,
        out2,
        [],
        attestations=[_terminal_attestation(r2.request_id, rank=0, generation=gen2)],
        finished_recving={r2.request_id},
    )
    scheduler.schedule()
    assert r2.status == RequestStatus.RUNNING
    assert r2.request_id not in scheduler._untrusted_external_reqs
    assert scheduler.recovery_plans == {}


# ---------------------------------------------------------------------------
# S2: preemption + partial receive + physical block reuse.
# ---------------------------------------------------------------------------


def test_matrix_02_preempted_partial_receive_blocks_not_matched_on_reuse():
    """A sync external claim partially loads (one failed block reported
    through the structured channel; a RecoveryPlan is admitted), the
    request is then PREEMPTED (the plan dissolves, its evidence is
    remembered as consumed, the blocks free), and a fresh same-prompt
    request is admitted afterwards: the freed physical blocks -- which
    still carry the partial external writes -- must never be
    prefix-matched, and a late re-delivery of the dissolved plan's report
    must not re-admit a plan.

    Catches: a preemption path that dissolves the plan but leaves the
    delayed publication or the hash entries alive (the freed blocks
    re-enter the pool and the same-prompt request matches the partially
    loaded content), and a consumed-evidence record that fails to cover
    the dissolved plan (its late re-delivery would install a second plan
    over freed blocks)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=True
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.RUNNING

    # Partial receive: one failed block -> report + reconciled
    # attestation -> the plan is admitted (APC on; the delayed
    # publication keeps the write set private).
    report = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att_reported(scheduler, r1)],
    )
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None
    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)

    # Preempt: the plan dissolves, its evidence is remembered consumed,
    # the blocks free.
    preempt(scheduler, r1)
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    pool = scheduler.kv_cache_manager.block_pool
    for bid in r1_blocks:
        assert pool.blocks[bid].block_hash is None, (
            "a freed block of the preempted partial receive kept a hash entry"
        )

    # A late re-delivery of the dissolved plan's report: consumed
    # (absorbed or harmlessly rejected) -- never a second plan, never an
    # untrusted poisoning of the preempted request's retry.
    deliver(scheduler, out1, [r1], reports=[report])
    assert scheduler.recovery_plans == {}
    assert r1.request_id not in scheduler._untrusted_external_reqs

    # The damaged request leaves; its blocks are in the free pool.
    scheduler.finish_requests(r1.request_id, RequestStatus.FINISHED_STOPPED)

    # A same-prompt request is admitted AFTER the reuse: nothing of the
    # failed transfer may be matchable -- its whole prompt is scheduled.
    r2 = create_request(num_tokens=3328)
    r2.request_id = "clone-" + r2.request_id
    r2.prompt_token_ids = list(r1.prompt_token_ids)
    scheduler.add_request(r2)
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens.get(r2.request_id) == 3328, (
        "prefix-cache hit on the preempted partial receive's content"
    )
    assert r2.request_id not in scheduler._untrusted_external_reqs


# ---------------------------------------------------------------------------
# S3: multiple TP ranks + one rank exception + async completion.
# ---------------------------------------------------------------------------


def test_matrix_03_tp2_one_rank_exception_async_completion_fails_closed():
    """TP=2 async delivery: rank 0 completes healthy and attests; rank 1's
    receive path crashed before attesting (its evidence never reaches the
    aggregator), and the delivery completes anyway (finished_recving).
    The completeness gate must fail the whole claim closed at promotion:
    rank 1's silence about its own shard is exactly the silent-rank gap,
    even though rank 0's attestation is terminal, complete, and
    content-certified.

    Catches: a promotion path that treats finished_recving as the
    transfer's success signal and publishes on it (the pre-H13 shape),
    or a completeness check that counts attestations instead of the
    exact rank set (rank 0's attestation counted twice would satisfy a
    count-based TP=2 gate): the unverified rank-1 shard would be
    published and shareable."""
    scheduler, r1, out1 = async_scenario(tp_size=2)

    # Rank 0 attests healthy (attempt in flight): folded, no verdict.
    deliver(
        scheduler,
        out1,
        [],
        attestations=[
            _terminal_attestation(
                r1.request_id, rank=0, generation=gen_of(scheduler, r1)
            )
        ],
    )
    assert r1.request_id not in scheduler._untrusted_external_reqs
    accumulated = scheduler._req_transfer_attestations.get(r1.request_id, [])
    assert [a.source_rank for a in accumulated] == [0]

    # The delivery completes with rank 1 silent (its exception consumed
    # the evidence before the aggregator): the promotion gate fails the
    # claim closed.
    deliver(scheduler, out1, [], finished_recving={r1.request_id})
    scheduler.schedule()
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert scheduler.recovery_plans == {}
    # The full local recompute is scheduled (nothing of the claim kept).
    assert r1.num_computed_tokens == 3328


# ---------------------------------------------------------------------------
# S4: multiple failed blocks + valid-looking per-region reports.
# ---------------------------------------------------------------------------


def test_matrix_04_multi_block_failure_with_valid_looking_report_pair():
    """Two blocks of one attempt genuinely fail; the connector emits a
    report for EACH failed block separately (each report individually
    well-formed, geometry valid, same generation, counts reconciled with
    the attestation). Divergent evidence -- two different regions of one
    attempt -- must never authorize a recovery: no plan, the minimum
    rollback floor at the rejection, and the promotion fails the whole
    claim closed.

    Catches: an admission path that folds per-region reports into ONE
    plan using only the first region (or the deepest rollback): one
    repaired region would leave the other failed block's corruption in
    the cache. Also catches a rejection that keeps a region's rollback
    boundary without checking the reports' mutual consistency (here the
    min rollback 1536 correctly excludes block 96, the deeper failure)."""
    scheduler, r1, out1 = async_scenario()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    gen = gen_of(scheduler, r1)

    region_a = _structured_report(
        scheduler, r1, failed_block_idx=96, transfer_generation=gen
    )
    region_b = _structured_report(
        scheduler,
        r1,
        failed_block_idx=100,
        token_span=(100 * BS, 101 * BS),
        rollback_boundary=100 * BS,
        transfer_generation=gen,
    )
    # The attestation accounts for BOTH failed blocks (reconciled with
    # the two rank-0 reports: 1 block x 1 layer each).
    attestation = _terminal_attestation(
        r1.request_id, loaded=190, failed=2, certified=190, generation=gen
    )
    deliver(
        scheduler,
        out1,
        [],
        reports=[region_a, region_b],
        attestations=[attestation],
        finished_recving={r1.request_id},
    )

    # Divergent: no plan, and the truncation floor is the MINIMUM
    # rollback of the well-formed reports (block 96's 1536 -- the deeper
    # failure's boundary), never region B's deeper claim of 1600.
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS

    # The promotion fails the whole claim closed: full local recompute
    # (nothing of the claim kept), the source untrusted, the engine alive.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == 3328
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert scheduler.recovery_plans == {}


# ---------------------------------------------------------------------------
# S5: evidence arriving after an attempt is superseded.
# ---------------------------------------------------------------------------


def test_matrix_05_superseded_attempt_evidence_cannot_publish_or_repair():
    """A sync claim's failure report admits a RecoveryPlan; the request is
    then PREEMPTED (the plan dissolves; a later re-admission with an
    external hit gets a NEW generation -- attempt 2). The FIRST attempt's
    terminal evidence (its report and attestation, carrying the old
    generation) is re-delivered at attempt 2's delivery update: it must
    neither authorize a plan for the superseded fault nor satisfy
    attempt 2's publication gate.

    Catches: an implementation whose evidence partition attributes by
    request alone (ignoring the generation): the stale attestation would
    satisfy attempt 2's completeness gate and PUBLISH attempt 2's
    unverified claim, and the stale report (naming attempt 1's freed
    block ids) would be validated against attempt 2's block table and
    either authorize or fail the request on the wrong grounds."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=True
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    gen1 = gen_of(scheduler, r1)

    # Attempt 1: one failed block -> plan admitted.
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[_structured_report(scheduler, r1)],
        attestations=[att_reported(scheduler, r1)],
    )
    assert r1.request_id in scheduler.recovery_plans
    (attempt1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)

    # Preemption supersedes the attempt: the plan dissolves, the blocks
    # free, and the re-admission starts attempt 2 (a new generation).
    preempt(scheduler, r1)
    assert scheduler.recovery_plans == {}
    out2 = scheduler.schedule()
    gen2 = gen_of(scheduler, r1)
    assert gen2 > gen1
    assert r1.status == RequestStatus.RUNNING
    assert r1.num_computed_tokens == 3328  # attempt 2's optimistic advance

    # The superseded attempt's delayed evidence arrives at attempt 2's
    # delivery update: both halves carry the old generation.
    stale_report = _structured_report(
        scheduler,
        r1,
        transfer_generation=gen1,
        failed_block_ids=[attempt1_blocks[96]],
    )
    stale_att = _terminal_attestation(
        r1.request_id, loaded=191, failed=1, certified=191, generation=gen1
    )
    deliver(scheduler, out2, [r1], reports=[stale_report], attestations=[stale_att])

    # No plan for the stale fault; attempt 2's delivery fails CLOSED (the
    # stale attestation cannot substitute for attempt 2's missing
    # current evidence): untrusted, full recompute, nothing published.
    assert scheduler.recovery_plans == {}
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert r1.num_computed_tokens == 0
    assert all_unpublished(scheduler, r1)

    # The engine keeps scheduling ordinary traffic afterwards.
    r2 = create_request(num_tokens=300)
    scheduler.add_request(r2)
    out3 = scheduler.schedule()
    assert out3.num_scheduled_tokens[r2.request_id] > 0


# ---------------------------------------------------------------------------
# S6: duplicate terminal notifications.
# ---------------------------------------------------------------------------


def test_matrix_06a_duplicate_repair_verdict_on_repair_step_is_stale():
    """The repair step's verdict is delivered, the plan COMMITS, and the
    SAME verdict is re-delivered in a second update of the SAME repair
    step (the connector re-emitting it; the repair step's output still
    carries repair_works, so the missing-verdict fail-safe also runs).
    The duplicate is rejected as stale (the plan is consumed), the
    fail-safe finds no ADMITTED plan and does nothing, and the request's
    restored extent is unchanged.

    Catches: a _consume_repair_result that re-consumes a consumed plan
    (double restore: num_computed_tokens jumping a second time), and a
    fail-safe that synthesizes an abort for a request whose plan was
    already legitimately committed (the duplicate stale disposition
    would roll the request back to the rollback boundary after a
    successful repair)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    deliver(
        scheduler,
        out1,
        [r1],
        reports=[_structured_report(scheduler, r1)],
        attestations=[att_reported(scheduler, r1)],
    )
    plan = scheduler.recovery_plans[r1.request_id]
    out_repair = scheduler.schedule()
    assert out_repair.repair_works is not None

    # The commit (identity echoed exactly).
    drive_repair_result(
        scheduler,
        r1,
        out_repair,
        structural_check=True,
        plan=plan,
        span=plan.repair_span,
    )
    assert r1.num_computed_tokens == 3072
    assert scheduler.recovery_plans == {}

    # The DUPLICATE verdict, re-delivered on the repair step's output:
    # stale (no plan), the fail-safe inert, nothing changes.
    drive_repair_result(
        scheduler,
        r1,
        out_repair,
        structural_check=True,
        plan=plan,
        span=plan.repair_span,
    )
    assert r1.num_computed_tokens == 3072
    assert scheduler.recovery_plans == {}
    assert r1.request_id not in scheduler._untrusted_external_reqs


def test_matrix_07_repeated_abort_commit_cycles_across_requests():
    """Two repair cycles back to back through the full report->plan->
    repair-step->verdict pipeline: request r1's plan ABORTS (evicts its
    span, recompute fallback), then a fresh request r2's plan COMMITS.
    The abort's eviction and the consumed-evidence bookkeeping must not
    poison the later commit: r2's admission still works, its repair
    commits, the restored extent is honored, and a late re-delivery of
    r1's consumed report changes nothing.

    Catches: abort handling that leaves the block pool or the
    consumed-evidence record in a state the next commit misreads -- e.g.
    a consumed-evidence key so broad that r2's fresh evidence is
    absorbed as a duplicate of r1's consumed attempt (silently refusing
    r2's plan), or a global plan-id/epoch collision after the abort."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )

    # --- Cycle 1: abort. ---
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    report1 = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report1],
        attestations=[att_reported(scheduler, r1)],
    )
    plan1 = scheduler.recovery_plans[r1.request_id]
    out_repair1 = scheduler.schedule()
    drive_repair_result(
        scheduler,
        r1,
        out_repair1,
        structural_check=False,
        plan=plan1,
        span=plan1.repair_span,
        aborted=True,
        executed_layers=2,
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 96 * BS
    scheduler.finish_requests(r1.request_id, RequestStatus.FINISHED_STOPPED)

    # --- Cycle 2: commit on a fresh request (same block pool). ---
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r2, 3072)
    out2 = scheduler.schedule()
    report2 = _structured_report(scheduler, r2)
    deliver(
        scheduler,
        out2,
        [r2],
        reports=[report2],
        attestations=[att_reported(scheduler, r2)],
    )
    plan2 = scheduler.recovery_plans[r2.request_id]
    assert plan2 is not None, (
        "the abort cycle poisoned the admission: r2's fresh evidence was not authorized"
    )
    out_repair2 = scheduler.schedule()
    drive_repair_result(
        scheduler,
        r2,
        out_repair2,
        structural_check=True,
        plan=plan2,
        span=plan2.repair_span,
    )
    assert r2.num_computed_tokens == 3072  # restored extent honored
    assert scheduler.recovery_plans == {}

    # A late re-delivery of r1's consumed report (its attempt reached a
    # terminal disposition) changes nothing.
    deliver(scheduler, out2, [r2], reports=[report1])
    assert scheduler.recovery_plans == {}
    assert r2.num_computed_tokens == 3072


# ---------------------------------------------------------------------------
# S8: cleanup during a pending async transfer.
# ---------------------------------------------------------------------------


def test_matrix_08_free_request_during_pending_async_transfer():
    """A request is FINISHED (client disconnect / stop) while its async
    transfer is still in flight. The cleanup must free every per-request
    evidence structure -- and the attempt's evidence, arriving later in
    the same step's update, must be dropped as unattributable rather
    than crashing (the request object is gone) or resurrecting state.

    Catches: a free path that leaves _req_transfer_generation behind
    (the later evidence would be attributed to a finished request and
    _reject_uninterpretable_reports would KeyError on requests.get), and
    an evidence handler that assumes every reported request is live."""
    scheduler, r1, out1 = async_scenario()
    gen = gen_of(scheduler, r1)

    # The request finishes mid-flight (stop token delivered by the local
    # step): cleanup runs while the transfer is pending.
    scheduler.finish_requests(r1.request_id, RequestStatus.FINISHED_STOPPED)
    assert r1.request_id not in scheduler._req_transfer_generation
    assert r1.request_id not in scheduler._req_transfer_attestations
    assert r1.request_id not in scheduler._req_pending_async_reports

    # The attempt's evidence arrives anyway (the receive thread finished
    # after the free): dropped, no crash, no state resurrection.
    late_att = _terminal_attestation(
        r1.request_id, loaded=192, failed=0, certified=192, generation=gen
    )
    deliver(
        scheduler,
        out1,
        [],
        attestations=[late_att],
        finished_recving={r1.request_id},
    )
    assert not scheduler._req_transfer_attestations.get(r1.request_id)
    assert scheduler.recovery_plans == {}

    # The engine keeps scheduling ordinary traffic.
    r2 = create_request(num_tokens=300)
    scheduler.add_request(r2)
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r2.request_id] > 0


def test_matrix_09_flat_failed_recving_conflicts_with_structured_plan():
    """The flat request-level failure channel and the structured evidence
    can name the same request in one update (the aggregator releases
    failed_recving at the same terminal pass the reports ride). The plan
    the structured arm admits is void: the flat recompute path owns the
    request, and no repair step is ever dispatched over the truncated
    geometry (pre-fix: the plan survived with a detached span and the
    next schedule admitted a repair step over an empty write-set
    fence)."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    gen = scheduler._req_transfer_generation[r1.request_id]
    report = _structured_report(scheduler, r1, failed_block_idx=96)
    attestation = _terminal_attestation(
        r1.request_id, generation=gen, loaded=191, failed=1, certified=191
    )
    deliver(
        scheduler,
        out1,
        [],
        reports=[report],
        attestations=[attestation],
        finished_recving={r1.request_id},
        failed_recving={r1.request_id},
    )
    assert scheduler.recovery_plans.get(r1.request_id) is None
    assert r1.num_computed_tokens == 0
    # The connector does not re-offer after the failure: the local
    # recompute owns the request, with no repair step dispatched.
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        0,
        False,
    )
    out2 = scheduler.schedule()
    assert out2.repair_works is None
    assert r1.num_computed_tokens > 0


def test_matrix_10_flat_invalid_blocks_conflict_with_structured_plan():
    """The flat per-block channel can name a block at or above the
    structured report's rollback boundary (the structured arm runs
    earlier in the same update and rewinds the frontier to it), which
    the truncation walk's visibility window cannot see. The conflict
    rule must void the plan anyway: the flat recompute path owns the
    request, and no repair step may commit a frontier covering blocks
    the flat channel declared invalid (pre-fix: the plan survived and
    the commit restored a frontier over the corrupt region)."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    gen = scheduler._req_transfer_generation[r1.request_id]
    report = _structured_report(scheduler, r1, failed_block_idx=96)
    attestation = _terminal_attestation(
        r1.request_id, generation=gen, loaded=191, failed=1, certified=191
    )
    # The flat channel names a DIFFERENT block (index 150) above the
    # report's span [1536, 1552): tokens [2400, 2416).
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    deliver(
        scheduler,
        out1,
        [],
        reports=[report],
        attestations=[attestation],
        invalid_block_ids={block_ids[150]},
    )
    # The plan is voided by the flat conflict, not merely truncated.
    assert scheduler.recovery_plans.get(r1.request_id) is None
    # The flat recompute path owns the request: no repair step is
    # dispatched, and the next step is an ordinary LOCAL recompute from
    # the rollback floor that covers the flat-declared-corrupt region
    # (pre-fix: the plan survived, the repair step committed a frontier
    # to 3072, and [2400, 2416) stayed corrupt while consumed).
    out2 = scheduler.schedule()
    assert out2.repair_works is None
    assert out2.num_scheduled_tokens[r1.request_id] == 1792
