# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer generations: late reports vs genuine repeated failures (H11).

The ambiguity this closes: a late duplicate can arrive after its
original recovery plan has been consumed, and a genuine new failure
can produce evidence identical to that late duplicate -- correlation
on region metadata alone cannot tell them apart. The transfer identity
is assigned at the moment the load attempt is SCHEDULED (early enough
to accompany the actual transfer and everything it reports), it is
globally monotonic and never reused (request-ID reuse included), and
every report and attestation echoes it. The scheduler then holds three
distinct opinions:

- evidence of the LATEST attempt is current and processed;
- evidence of an older attempt is a late artifact: dropped, unable to
  authorize OR dissolve anything;
- the same (attempt, evidence) pair that already reached a terminal
  disposition is a duplicate of a known outcome: absorbed.

A genuine repeated failure is a NEW scheduled attempt with a NEW
generation: never dropped, never absorbed.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
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
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def scenario(num_tokens=3328):
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=num_tokens)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    return scheduler, r1, out1


def deliver(scheduler, out, reqs, reports=(), attestations=()):
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    scheduler.update_from_output(out, output)


def current_gen(scheduler, request):
    return scheduler._req_transfer_generation[request.request_id]


def att_reported(scheduler, request, failed=1):
    """The CONFORMING attestation for a report-bearing delivery: the
    failed count reconciled with the report's region, everything that
    loaded certified (the W2-ON fault shape)."""
    return _terminal_attestation(
        request.request_id,
        loaded=192 - failed,
        failed=failed,
        certified=192 - failed,
        generation=scheduler._req_transfer_generation.get(request.request_id),
    )


# ---------------------------------------------------------------------------
# Identity assignment and propagation.
# ---------------------------------------------------------------------------


def test_generation_assigned_at_scheduling_and_echoed_in_plan():
    """The identity is assigned when the external load attempt is
    scheduled -- before any evidence can exist -- and the admitted plan
    carries it."""
    scheduler, r1, out1 = scenario()
    gen = current_gen(scheduler, r1)
    assert gen >= 1

    report = _structured_report(scheduler, r1)
    attestation = att_reported(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=[report], attestations=[attestation])

    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.transfer_generation == gen


def test_multiple_requests_get_distinct_generations():
    """Two requests scheduled with external claims in one step carry
    distinct, ordered identities: evidence of one request can never be
    mistaken for the other's attempt."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    # ONE Mock admitting both claims (a second single-request admission
    # would replace the first's side_effect and strip r1's identity).
    from tests.v1.core.test_scheduler_report_consistency import (
        _admit_sync_external_hits,
    )

    _admit_sync_external_hits(scheduler, [r1, r2], 3072)
    out1 = scheduler.schedule()

    g1, g2 = current_gen(scheduler, r1), current_gen(scheduler, r2)
    assert g1 != g2
    assert out1.num_scheduled_tokens[r1.request_id] > 0
    assert out1.num_scheduled_tokens[r2.request_id] > 0


# ---------------------------------------------------------------------------
# Stale evidence: late artifacts of consumed attempts.
# ---------------------------------------------------------------------------


def test_request_id_reuse_distinguishes_late_report_from_new_failure():
    """The scenario that motivated generations: request A (id X) had a
    failed attempt; A finishes; a NEW request reuses id X and schedules
    a fresh attempt. A late report of A's old fault arrives carrying
    the old identity: it is stale against the new request's attempt --
    dropped, not treated as the new request's failure. The new
    request's OWN report (new generation) authorizes its plan."""
    scheduler = None
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    # First request: id "reused", external claim, then finished.
    first = create_request(num_tokens=3328, request_id=1)
    first.request_id = "reused"
    scheduler.add_request(first)
    admit_sync_external_hit(scheduler, first, 3072)
    scheduler.schedule()
    old_gen = current_gen(scheduler, first)
    # Finish it (the identity bookkeeping dies with the request).
    # finish_requests sets the status itself: pre-setting it would make
    # the request look already-finished and skip the free. (The fixture
    # prefixes ids, so always address requests by request.request_id.)
    scheduler.finish_requests(first.request_id, RequestStatus.FINISHED_STOPPED)
    assert first.request_id not in scheduler._req_transfer_generation

    # A new request reuses the id and schedules a fresh attempt.
    second = create_request(num_tokens=3328, request_id=1)
    second.request_id = "reused"
    scheduler.add_request(second)
    admit_sync_external_hit(scheduler, second, 3072)
    out1 = scheduler.schedule()
    new_gen = current_gen(scheduler, second)
    assert new_gen > old_gen

    # The late report of the FIRST request's fault (old identity, old
    # evidence) arrives -- with the old attempt's OWN attestation
    # (also old identity): the entire delivery is stale and must be
    # absorbed without authorizing (or truncating) anything.
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(second.request_id)
    late = _structured_report(
        scheduler,
        second,
        transfer_generation=old_gen,
        failed_block_ids=[block_ids[96]],
    )
    stale_attestation = _terminal_attestation(
        second.request_id,
        loaded=191,
        failed=1,
        certified=191,
        generation=old_gen,
    )
    # The new attempt's own failure evidence arrives TOGETHER with
    # the stale pair in the same step's delivery (the realistic
    # shape): the fresh report + reconciled attestation authorize;
    # the stale report and attestation are dropped by the identity
    # partition.
    fresh = _structured_report(scheduler, second)
    assert fresh.transfer_generation == new_gen
    deliver(
        scheduler,
        out1,
        [second],
        reports=[late, fresh],
        attestations=[stale_attestation, att_reported(scheduler, second)],
    )
    plan = scheduler.recovery_plans[second.request_id]
    assert plan is not None
    assert plan.transfer_generation == new_gen
    assert second.num_computed_tokens == 96 * BS

    # A later re-delivery of the stale report alone changes nothing: no
    # dissolution, no truncation.
    deliver(scheduler, out1, [second], reports=[late])
    assert scheduler.recovery_plans[second.request_id] is plan
    assert second.num_computed_tokens == 96 * BS


def test_stale_report_cannot_dissolve_pending_plan():
    """A pending plan is built on the CURRENT attempt's evidence. A
    late, divergent-looking report of an OLDER attempt is absorbed as
    stale -- it must not dissolve a plan built on newer evidence."""
    scheduler, r1, out1 = scenario()
    gen = current_gen(scheduler, r1)
    report = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att_reported(scheduler, r1)],
    )
    plan = scheduler.recovery_plans[r1.request_id]

    # A late report of an older attempt, with evidence DIVERGENT from
    # the plan's: stale by identity, absorbed, never a dissolution.
    # (Delivered as a second update of the SAME step -- the established
    # multi-update fixture pattern -- so the plan is still pending.)
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    stale_divergent = _structured_report(
        scheduler,
        r1,
        failed_block_idx=97,
        token_span=(97 * BS, 98 * BS),
        rollback_boundary=97 * BS,
        transfer_generation=gen - 1,
    )
    deliver(scheduler, out1, [r1], reports=[stale_divergent])
    assert scheduler.recovery_plans[r1.request_id] is plan
    assert r1.num_computed_tokens == 96 * BS


def test_late_duplicate_of_consumed_attempt_is_absorbed():
    """After the plan's verdict is consumed, a late re-delivery of the
    SAME (attempt, evidence) is a duplicate of a known outcome: no new
    plan, and the request is untouched (its outputs of the duplicate's
    step are ordinary, so nothing is skipped either)."""
    scheduler, r1, out1 = scenario()
    current_gen(scheduler, r1)
    report = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att_reported(scheduler, r1)],
    )
    plan = scheduler.recovery_plans[r1.request_id]

    out2 = scheduler.schedule()
    drive_repair_result(scheduler, r1, out2, structural_check=True, plan=plan)
    assert scheduler.recovery_plans == {}
    committed_extent = r1.num_computed_tokens

    # The late duplicate: same identity, same evidence, later step.
    out3 = scheduler.schedule()
    # The step's schedule advanced the computed frontier (the ordinary
    # continuation); the duplicate must not TRUNCATE it back.
    extent_before_duplicate = r1.num_computed_tokens
    duplicate = _structured_report(scheduler, r1)
    deliver(scheduler, out3, [r1], reports=[duplicate])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == extent_before_duplicate
    assert extent_before_duplicate >= committed_extent


# ---------------------------------------------------------------------------
# The genuine repeated failure: a NEW attempt is never discarded.
# ---------------------------------------------------------------------------


def test_flat_failure_retry_does_not_reschedule_external_claim():
    """A corollary worth pinning: after a flat invalid-blocks failure
    the recovery path reschedules the request for LOCAL recompute and
    does NOT consult the connector for a fresh external claim of the
    same corrupt source -- no new attempt identity is assigned. (The
    genuine repeated-failure disambiguation -- a NEW attempt's report
    honored while the OLD attempt's identical evidence stays stale --
    is driven end-to-end by the request-ID-reuse test above; the
    identity is assigned at the scheduling site, which is
    request-agnostic.)

    H14: under the repair policy the flat channel no longer truncates
    to a 'valid prefix' -- that prefix is unverified external content,
    and the ordinary path would hash-publish it at the next allocation.
    The flat failure now fails the whole claim closed (rollback 0,
    source marked untrusted) exactly like missing structured evidence;
    the no-re-attempt invariant is unchanged."""
    from tests.v1.core.test_scheduler_repair_budget import (
        admit_async_external_hit,
    )

    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    gen1 = current_gen(scheduler, r1)

    # The attempt fails through the flat path: the whole claim is
    # untrusted (rollback 0), and the source is never re-consulted.
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    fail_output = create_model_runner_output(
        reqs=[],
        finished_recving={r1.request_id},
        invalid_block_ids={block_ids[96]},
        use_eos=False,
    )
    scheduler.update_from_output(out1, fail_output)
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs

    # The reschedule is local: no second external attempt, no second
    # identity.
    out2 = scheduler.schedule()
    assert out2.repair_works is None
    assert scheduler._req_transfer_generation.get(r1.request_id) == gen1


# ---------------------------------------------------------------------------
# Identity violations: fail closed.
# ---------------------------------------------------------------------------


def test_report_without_identity_fails_closed():
    """Under the repair policy the identity contract is mandatory: a
    report with no transfer identity cannot be attributed to any
    attempt and fails the request closed (rollback 0)."""
    scheduler, r1, out1 = scenario()
    current_gen(scheduler, r1)
    report = _structured_report(scheduler, r1, transfer_generation=None)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att_reported(scheduler, r1)],
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_report_with_issued_only_identity_fails_closed():
    """A report naming a generation the scheduler never issued is
    unattributable and hostile: fail closed."""
    scheduler, r1, out1 = scenario()
    gen = current_gen(scheduler, r1)
    report = _structured_report(scheduler, r1, transfer_generation=gen + 100)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att_reported(scheduler, r1)],
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_unattributable_report_for_claimless_request_is_dropped():
    """A report for a request that never had an external load attempt
    (no identity): the request holds no external data to distrust, so
    the report is dropped -- loudly -- and the request is untouched."""
    scheduler, r1, out1 = scenario()
    # A second request with a purely local schedule: no external claim,
    # no identity.
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r2)
    out2 = scheduler.schedule()
    assert r2.request_id not in scheduler._req_transfer_generation

    bogus = _structured_report(scheduler, r2)
    deliver(scheduler, out2, [r2], reports=[bogus])
    assert scheduler.recovery_plans == {}
    assert r2.num_computed_tokens == 3328


def test_stale_attestation_does_not_count_toward_completeness():
    """Cross-rank identity divergence (TP=2): rank 1's attestation
    carries an older attempt's identity. It does not count toward the
    completeness gate -- the evidence is incomplete and the recovery is
    refused even though every rank physically attested."""
    from tests.v1.core.test_scheduler_transfer_evidence import make_tp_scheduler

    scheduler = make_tp_scheduler(2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    gen = current_gen(scheduler, r1)

    report = _structured_report(scheduler, r1)
    report.source_rank = 0
    rank0 = att_reported(scheduler, r1)
    rank1 = _terminal_attestation(
        r1.request_id,
        rank=1,
        loaded=191,
        failed=1,
        certified=191,
        generation=gen - 1,
    )
    deliver(scheduler, out1, [r1], reports=[report], attestations=[rank0, rank1])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
