# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H14: the distributed-evidence publication and ownership contract.

The publication invariants this module pins (every test asserts CACHE
STATE -- block hashes, physical block ownership, computed-token
accounting -- not merely the absence of a RecoveryPlan):

- No missing evidence can result in verified publication: a sync claim
  whose step produced NO evidence (no connector output at all, or empty
  evidence lists) fails closed -- nothing is hash-published, the request
  recomputes from zero, and the source is never re-consulted (H14-01).
- The same for the async delivery path: a request promoted from
  WAITING_FOR_REMOTE_KV without complete evidence is released, never
  published (H14-01).
- Publication requires CONTENT certification, not just transfer
  completion: "size"/"none" certification levels are the connector
  reporting it could NOT verify content, and fail closed (H14-02);
  "sha256" and "content" (the documented store-trust mode) publish.
- A connector must never be able to introduce a transfer identity: a
  future-generation attestation is a contract violation that fails the
  request closed, while a late OLD-generation attestation is merely a
  stale artifact (H14-05).
- Split arrivals are legitimate: evidence arriving across several
  passes completes at the delivery's decision point (the fold), and a
  report of an async attempt still in flight is deferred until the
  attempt is terminal instead of being judged incomplete (H14-07).
- Preemption and request cleanup drop pending publication and evidence
  state; nothing leaks across request-ID reuse (H14-07).
- A rejected claim cannot poison an unrelated request or a sharing
  peer: unverified content never acquires hash entries of its own
  (H14-09), and the ordinary recompute/publication flow keeps working
  (invariants 11 and 12).
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    _structured_report,
    _terminal_attestation,
    admit_async_external_hit,
    admit_sync_external_hit,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def make_tp_scheduler(tp_size: int = 1, apc: bool = True):
    """A repair-policy scheduler expecting attestations from tp_size TP
    ranks (APC on: the publication assertions need the hash table)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=apc
    )
    scheduler._evidence_tp_size = tp_size
    return scheduler


def block_hashes(scheduler, request):
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    pool = scheduler.kv_cache_manager.block_pool
    return [pool.blocks[b].block_hash for b in block_ids], block_ids


def all_unpublished(scheduler, request) -> bool:
    hashes, _ = block_hashes(scheduler, request)
    return all(h is None for h in hashes)


def all_published(scheduler, request) -> bool:
    hashes, _ = block_hashes(scheduler, request)
    return all(h is not None for h in hashes)


def gen_of(scheduler, request):
    return scheduler._req_transfer_generation[request.request_id]


def att(scheduler, request, rank=0, **overrides):
    """A conforming terminal, sha-certified attestation for the request's
    latest scheduled attempt (the generation override is honored for the
    stale/future shapes)."""
    overrides.setdefault("generation", gen_of(scheduler, request))
    return _terminal_attestation(
        request.request_id,
        rank=rank,
        **overrides,
    )


def deliver(
    scheduler,
    out,
    reqs,
    reports=(),
    attestations=(),
    finished_recving=None,
    failed_recving=None,
):
    """update_from_output with structured evidence (and optionally the
    async completion channels) on one connector output."""
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        finished_recving=finished_recving,
        failed_recving=failed_recving or set(),
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    scheduler.update_from_output(out, output)
    return output


def sync_claim(scheduler, request, num_tokens=3072):
    admit_sync_external_hit(scheduler, request, num_tokens)
    return scheduler.schedule()


# ---------------------------------------------------------------------------
# H14-01: missing evidence must never authorize publication (sync).
# ---------------------------------------------------------------------------


def test_sync_claim_no_evidence_never_publishes():
    """The core H14-01 shape: a sync external claim whose step produced NO
    connector output at all. Under the pre-H14 code the publication gate
    only ran when some evidence arrived, so this step published all 208
    blocks of the unverified claim. Now: nothing is published, the
    request truncates to zero, the source is marked untrusted, and the
    request keeps making progress through the local recompute."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )

    assert all_unpublished(scheduler, r1)
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert not scheduler._delayed_sync_claim_pubs
    assert scheduler.recovery_plans == {}

    # Not stuck: the reschedule is a full local recompute that does not
    # consult the (untrusted) connector again.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == 3328
    assert gen_of(scheduler, r1) is not None  # no new attempt was scheduled


def test_sync_claim_empty_evidence_output_never_publishes():
    """The same hole with an explicit-but-empty KVConnectorOutput (a
    worker whose connector produced no reports and no attestations)."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput()
    scheduler.update_from_output(out1, output)

    assert all_unpublished(scheduler, r1)
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_sync_claim_tp2_silent_rank_never_publishes():
    """TP=2 with only rank 0 attesting (rank 1 silent -- the silent-rank
    gap): the whole claim fails closed, nothing is published."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(scheduler, out1, [r1], attestations=[att(scheduler, r1, rank=0)])

    assert all_unpublished(scheduler, r1)
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_sync_claim_tp2_one_rank_reports_success_one_silent():
    """Rank 0 attests a fully certified healthy load; rank 1 says nothing
    at all. Silence is detected, not assumed healthy: no publication."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[
            att(scheduler, r1, rank=0, expected=192, loaded=192, certified=192)
        ],
    )

    assert all_unpublished(scheduler, r1)
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_sync_claim_tp4_three_of_four_ranks_never_publishes():
    """TP=4 with one silent rank: the exact-set completeness gate fails
    closed regardless of group size."""
    scheduler = make_tp_scheduler(tp_size=4)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[att(scheduler, r1, rank=r) for r in range(3)],
    )

    assert all_unpublished(scheduler, r1)
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_repeated_attempts_after_evidence_rejection_stay_local():
    """After an evidence rejection the connector is not re-consulted: the
    recompute completes locally and a re-offered external hit cannot loop
    the rejection cycle (invariants 10 and 12)."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)
    gen1 = gen_of(scheduler, r1)

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert r1.request_id in scheduler._untrusted_external_reqs

    # The connector (still mocking a full external hit) is NOT consulted.
    out2 = scheduler.schedule()
    assert out2.num_scheduled_tokens[r1.request_id] == 3328
    assert gen_of(scheduler, r1) == gen1
    scheduler.update_from_output(
        out2, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert r1.num_computed_tokens == 3328
    # The locally recomputed blocks are ordinary published cache.
    assert all_published(scheduler, r1)


def test_unrelated_request_unaffected_by_evidence_rejection():
    """Request-scoping: r1's claim is rejected for missing evidence; r2
    (a plain local request, no external claim) is untouched -- scheduled,
    computed and published normally."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=512)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert out1.num_scheduled_tokens.get(r2.request_id) == 512

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1, r2], use_eos=False)
    )

    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert r2.num_computed_tokens == 512
    assert r2.request_id not in scheduler._untrusted_external_reqs
    assert all_published(scheduler, r2)
    assert all_unpublished(scheduler, r1)


# ---------------------------------------------------------------------------
# H14-01: the async delivery path (finished_recving / promotion).
# ---------------------------------------------------------------------------


def test_async_no_evidence_never_publishes():
    """The async form of H14-01: finished_recving arrives with no evidence;
    the promotion releases the claim instead of publishing it, and the
    request recomputes locally from zero."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    output = create_model_runner_output(
        reqs=[], finished_recving={r1.request_id}, use_eos=False
    )
    scheduler.update_from_output(out1, output)
    assert r1.request_id in scheduler.finished_recving_kv_req_ids

    # Promotion: the gate fires before any caching happens.
    scheduler.schedule()
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert r1.num_computed_tokens == 3328  # full local recompute scheduled
    assert r1.status == RequestStatus.RUNNING
    assert scheduler.recovery_plans == {}


def test_async_partial_then_complete_evidence_publishes():
    """Split arrivals are legitimate (the fold): rank 0 attests in the
    pass before the delivery completes, rank 1 attests with
    finished_recving. The promotion sees the accumulated complete,
    certified evidence and publishes -- the per-arrival judgment of the
    pre-H14 code would have failed this request closed at rank 0's
    arrival."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    # Pass 1: rank 0's attestation arrives; the attempt is still in
    # flight (no finished_recving) -- no rejection.
    deliver(scheduler, out1, [], attestations=[att(scheduler, r1, rank=0)])
    assert r1.request_id not in scheduler._untrusted_external_reqs
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS

    # Pass 2: rank 1 attests and the delivery completes.
    deliver(
        scheduler,
        out1,
        [],
        attestations=[att(scheduler, r1, rank=1)],
        finished_recving={r1.request_id},
    )

    scheduler.schedule()
    assert r1.status == RequestStatus.RUNNING
    assert r1.request_id not in scheduler._untrusted_external_reqs
    assert r1.num_computed_tokens == 3328
    # The verified claim is published and consumable as prefix cache.
    assert all_published(scheduler, r1)
    r2 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r2)
    scheduler.schedule()
    (r2_blocks,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)
    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    assert set(r1_blocks) & set(r2_blocks)


def test_async_report_before_terminal_pass_is_deferred_not_rejected():
    """A failure report arriving while the async attempt is still in
    flight (before finished_recving) is DEFERRED, not judged incomplete:
    the other ranks' attestations may arrive in a later pass. At the
    terminal pass the stored report is re-examined and authorizes the
    recovery -- the pre-H14 code rejected this delivery at the report's
    pass (a false-positive rejection of a legitimate repair)."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    # Pass 1: rank 0's failure report + reconciled attestation; the
    # attempt is in flight -> deferred, request NOT failed.
    report = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [],
        reports=[report],
        attestations=[att(scheduler, r1, rank=0, loaded=191, failed=1, certified=191)],
    )
    assert r1.request_id not in scheduler._untrusted_external_reqs
    assert scheduler.recovery_plans == {}
    assert scheduler._req_pending_async_reports.get(r1.request_id)

    # Pass 2: rank 1's healthy attestation + the delivery completes ->
    # the deferred report is re-examined and the plan is admitted.
    deliver(
        scheduler,
        out1,
        [],
        attestations=[att(scheduler, r1, rank=1)],
        finished_recving={r1.request_id},
    )

    assert r1.request_id in scheduler.recovery_plans
    assert not scheduler._req_pending_async_reports.get(r1.request_id)
    assert r1.request_id not in scheduler._untrusted_external_reqs


def test_async_flat_failed_recving_releases_claim():
    """The flat failed_recving channel under the repair policy: the claim
    is truncated to zero and the promotion releases the blocks -- the
    pre-H13 'success arm publishes the failed claim' shape stays closed,
    and the recompute owns the request."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    deliver(
        scheduler,
        out1,
        [],
        finished_recving={r1.request_id},
        failed_recving={r1.request_id},
    )
    assert r1.num_computed_tokens == 0

    out2 = scheduler.schedule()
    assert r1.status == RequestStatus.RUNNING
    # The full local recompute (no external re-admission: the source is
    # untrusted, and the schedule shows the whole prompt locally).
    assert out2.num_scheduled_tokens[r1.request_id] == 3328
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert scheduler.recovery_plans == {}


# ---------------------------------------------------------------------------
# H14-02: transfer completion is not content verification.
# ---------------------------------------------------------------------------


def test_all_ranks_sha_certified_publishes():
    """The conforming all-healthy delivery: every rank terminal, complete,
    sha-certified -> published and reusable."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[att(scheduler, r1, rank=r) for r in range(2)],
    )
    assert all_published(scheduler, r1)
    assert r1.request_id not in scheduler._untrusted_external_reqs


def test_all_ranks_content_addressed_publishes():
    """The documented store-trust mode: content-addressed immutable keys
    ("content") with every loaded object certified publish -- GET success
    implies correct content iff the store's key->data binding is
    trustworthy, which is the mode's stated assumption."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[
            att(
                scheduler,
                r1,
                rank=r,
                certification="content",
                certified=192,
            )
            for r in range(2)
        ],
    )
    assert all_published(scheduler, r1)


def test_size_only_certification_fails_closed():
    """Size-only certification: the connector reports it checked only
    lengths. Same-size stale content passes such a check, so publication
    is refused and the delivery recomputes."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[
            att(
                scheduler,
                r1,
                rank=r,
                certification="size",
                certified=0,
            )
            for r in range(2)
        ],
    )
    assert all_unpublished(scheduler, r1)
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_no_certification_fails_closed():
    """No certification on every rank: GET success was the only
    'verification'. Publication requires proof, not optimism."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[
            att(
                scheduler,
                r1,
                rank=r,
                certification="none",
                certified=0,
            )
            for r in range(2)
        ],
    )
    assert all_unpublished(scheduler, r1)
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_mixed_certification_levels_fail_closed():
    """TP0 certifies with sha256; TP1 only size-checks. The claim as a
    whole is only as verified as its least-certified shard."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[
            att(scheduler, r1, rank=0),
            att(
                scheduler,
                r1,
                rank=1,
                certification="size",
                certified=0,
            ),
        ],
    )
    assert all_unpublished(scheduler, r1)
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_successful_get_without_certifying_loaded_content_fails_closed():
    """A rank whose certification level is real ("sha256") but whose
    certified count does not cover its loaded set: the uncertified
    objects are exactly the unverifiable ones."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[
            att(scheduler, r1, rank=0),
            att(scheduler, r1, rank=1, certified=100),
        ],
    )
    assert all_unpublished(scheduler, r1)
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_uncertified_evidence_demotes_rollback_to_full_recompute():
    """F1 (adversarial review): the rollback prefix is external content
    too. An async delivery whose conforming failure report carries a
    healthy rollback boundary, but whose rank attests only SIZE-level
    certification, must NOT publish or repair over the prefix: the
    claim demotes to the full-recompute fallback exactly like an
    uncertified all-healthy delivery (no plan, rollback 0, source
    untrusted, nothing hash-published). With the pre-F1 code the
    promotion published all 96 rollback-prefix blocks of size-checked
    (unverified) content."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    report = _structured_report(scheduler, r1)  # rollback 96*BS
    gen_of_old = gen_of(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[
            att(
                scheduler,
                r1,
                loaded=191,
                failed=1,
                certified=0,
                certification="size",
            )
        ],
        finished_recving={r1.request_id},
    )

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs
    # Nothing of the unverified claim was ever hash-published (the
    # delayed publication held it out and the rejection evicted).
    pool = scheduler.kv_cache_manager.block_pool
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    assert all(pool.blocks[b].block_hash is None for b in block_ids)
    # The promotion releases the claim and the reschedule is a full
    # LOCAL recompute of the whole prompt (the untrusted source is not
    # re-consulted, so no second external attempt: the generation map
    # still names the OLD attempt and no NEW one was assigned; whatever
    # the local step publishes afterwards is ordinary locally-written
    # content).
    out2 = scheduler.schedule()
    assert r1.status == RequestStatus.RUNNING
    assert out2.num_scheduled_tokens[r1.request_id] == 3328
    assert scheduler._req_transfer_generation.get(r1.request_id) == gen_of_old


def test_certified_evidence_keeps_report_rollback_boundary():
    """The mirror of F1: with the rank content-certified, the report
    rollback boundary is honored -- the prefix is verified external
    content and the repair span starts at the boundary."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    report = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att(scheduler, r1, loaded=191, failed=1, certified=191)],
        finished_recving={r1.request_id},
    )

    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.repair_span[0] == 96 * BS
    assert r1.num_computed_tokens == 96 * BS


def test_uncertified_failed_region_with_certified_loaded_set_repairs():
    """The separation the recovery path depends on (H12/H14-02): a rank
    with a failed region still certifies_content() for its LOADED set --
    the failed region is the repair's write set -- so the recovery is
    authorized with the above-span trust intact."""
    scheduler = make_tp_scheduler(tp_size=2)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    reports = []
    attestations = []
    for r in range(2):
        rank_report = _structured_report(scheduler, r1)
        rank_report.source_rank = r
        reports.append(rank_report)
        attestations.append(
            att(scheduler, r1, rank=r, loaded=191, failed=1, certified=191)
        )
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)

    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.healthy_above_span is True
    assert r1.request_id not in scheduler._untrusted_external_reqs


# ---------------------------------------------------------------------------
# H14-05: future-generation evidence fails closed.
# ---------------------------------------------------------------------------


def test_future_generation_attestation_fails_closed():
    """A connector naming a generation the scheduler never issued for the
    request: an explicit contract violation (the identity is
    scheduler-authoritative), not an ignorable artifact. The async
    in-flight shape makes this DISCRIMINATING: with no pending sync
    publication, only the identity violation itself (not the delivery
    gate) can fail the request -- with H14-05 reverted the future
    attestation would be silently dropped and the request left alone."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    future = _terminal_attestation(r1.request_id, generation=gen_of(scheduler, r1) + 5)
    deliver(scheduler, out1, [], attestations=[future])

    assert r1.request_id in scheduler._untrusted_external_reqs
    assert r1.num_computed_tokens == 0
    assert not scheduler._req_transfer_attestations.get(r1.request_id)


def test_old_generation_attestation_is_stale_not_violating():
    """A late attestation of an older attempt is dropped as stale: it can
    neither authorize nor dissolve anything, and -- unlike a future
    identity -- it is not a contract violation. The request's outcome
    here comes solely from the publication gate (the stale attestation
    does not count as current evidence); the stale evidence itself
    contributed nothing to the accumulated state."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)
    gen1 = gen_of(scheduler, r1)

    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[att(scheduler, r1, generation=gen1 - 1)],
    )
    # Stale (older than the current attempt): dropped, not folded.
    assert not scheduler._req_transfer_attestations.get(r1.request_id)
    # The delivery still fails closed through the publication gate
    # (missing CURRENT evidence), but not because of a violation.
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_future_attestation_alongside_valid_current_report():
    """A valid current-generation report + attestation accompanied by a
    future-generation attestation: the unissued identity fails the whole
    request closed -- divergent evidence never authorizes."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    report = _structured_report(scheduler, r1)
    future = _terminal_attestation(r1.request_id, generation=gen_of(scheduler, r1) + 1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[
            att(scheduler, r1, loaded=191, failed=1, certified=191),
            future,
        ],
    )
    assert scheduler.recovery_plans == {}
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_future_attestation_dissolves_pending_plan():
    """A future-generation attestation while a RecoveryPlan is pending:
    the violation dissolves the plan (recording its evidence as
    consumed) and fails the request to the recompute fallback."""
    from vllm.v1.core.sched.scheduler import RecoveryPlan

    scheduler = make_tp_scheduler(apc=False)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)
    gen1 = gen_of(scheduler, r1)

    plan = RecoveryPlan(
        request_id=r1.request_id,
        repair_span=(96 * BS, 192 * BS),
        est_equivalent_tokens=100,
        layer_end=4,
        restored_extent=3072,
        failed_block_ids=[1],
        plan_id=1,
        healthy_above_span=True,
        report_evidence=("k",),
        transfer_generation=gen1,
    )
    plan.state = "DETECTED"
    scheduler.recovery_plans[r1.request_id] = plan

    future = _terminal_attestation(r1.request_id, generation=gen1 + 3)
    deliver(scheduler, out1, [r1], attestations=[future])

    assert scheduler.recovery_plans == {}
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert r1.num_computed_tokens == 0


def test_future_attestation_is_request_scoped():
    """A future-generation attestation for request A must not consume or
    dissolve anything belonging to request B."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    r2 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    # One connector answering external hits for BOTH requests (each
    # admit_* helper replaces the connector mock wholesale).
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        3072,
        False,
    )
    out1 = scheduler.schedule()

    future_for_r1 = _terminal_attestation(
        r1.request_id, generation=gen_of(scheduler, r1) + 9
    )
    deliver(
        scheduler,
        out1,
        [r1, r2],
        attestations=[future_for_r1, att(scheduler, r2)],
    )

    assert r1.request_id in scheduler._untrusted_external_reqs
    assert r2.request_id not in scheduler._untrusted_external_reqs
    assert all_published(scheduler, r2)
    assert all_unpublished(scheduler, r1)


def test_late_duplicate_attestation_does_not_dissolve_admitted_plan():
    """A conforming async delivery admits the repair plan; a connector
    re-delivering the same attempt's attestation in a LATER step must
    not be read as 'failed objects without a report' (there is no
    report in that step) and dissolve the legitimately admitted plan --
    the false-positive rejection this guards against is safe (the
    recompute fallback) but wasteful, and the plan's own verdict
    machinery owns the request from admission."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    report = _structured_report(scheduler, r1)
    deliver(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[att(scheduler, r1, loaded=191, failed=1, certified=191)],
        finished_recving={r1.request_id},
    )
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None

    # The duplicate attestation arrives alone in a later step.
    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[att(scheduler, r1, loaded=191, failed=1, certified=191)],
    )
    assert scheduler.recovery_plans.get(r1.request_id) is plan
    assert r1.request_id not in scheduler._untrusted_external_reqs


# ---------------------------------------------------------------------------
# H14-07: lifecycle hygiene (preemption, cleanup, reuse).
# ---------------------------------------------------------------------------


def test_preempted_request_pending_publication_discarded():
    """A request preempted between its sync-claim admission and the
    update: the pending publication is discarded (its blocks were
    freed; publishing would map hashes onto freed blocks), and the
    evidence state dies with the attempt."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    out1 = sync_claim(scheduler, r1)

    scheduler.running.remove(r1)
    scheduler._preempt_request(r1, 0.0)
    assert r1.status == RequestStatus.PREEMPTED

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert not scheduler._delayed_sync_claim_pubs
    assert scheduler._req_transfer_attestations.get(r1.request_id) is None


def test_request_free_clears_evidence_state():
    """_free_request drops every per-request evidence structure; a later
    request reusing the ID cannot inherit state. (The in-flight async
    shape: the attempt is not terminal, so nothing has been published or
    rejected and the accumulated evidence is genuinely live.)"""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    deliver(scheduler, out1, [r1], attestations=[att(scheduler, r1)])
    assert scheduler._req_transfer_attestations.get(r1.request_id)

    scheduler.finish_requests(r1.request_id, RequestStatus.FINISHED_STOPPED)
    assert r1.request_id not in scheduler._req_transfer_attestations
    assert r1.request_id not in scheduler._req_transfer_generation
    assert r1.request_id not in scheduler._req_pending_async_reports


# ---------------------------------------------------------------------------
# H14-09: physical ownership and the sharing peer.
# ---------------------------------------------------------------------------


def test_rejected_claim_never_poisons_sharing_peer():
    """Two same-prefix requests; r1 carries the sync claim whose evidence
    is missing. r1's unverified blocks never acquire hash entries; the
    peer r2 is scheduled independently and never matches r1's claim
    blocks (the sharing happens only through r1's later LOCAL recompute
    publication, whose content r1 itself wrote)."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r1)
    (r1_blocks,) = ([],)
    out1 = sync_claim(scheduler, r1)
    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert all_unpublished(scheduler, r1)

    # The peer arrives AFTER the rejection and is admitted in the same
    # pass as r1's local recompute: the queue processes r2's lookup
    # BEFORE r1's recompute blocks are optimistically hashed, so r2
    # finds nothing to match and computes fully locally. Either way the
    # invariant holds: no hash entry for the claim's unverified content
    # ever existed (the delayed publication held it out and the rejection
    # evicted everything), so the peer cannot consume it.
    r2 = create_request(num_tokens=3328, common_prefix_len=3072)
    scheduler.add_request(r2)
    scheduler.schedule()
    (r2_blocks,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)
    assert r2.request_id not in scheduler._untrusted_external_reqs
    # The peer never consulted the connector (a plain local request).
    assert r2.request_id not in scheduler._req_transfer_generation
    assert r2.num_computed_tokens == 3328  # full local compute, no match


def test_shared_local_prefix_survives_peers_flat_failure():
    """A published LOCAL shared prefix owned by two requests: when r1's
    later external claim fails through the flat channel and r1's blocks
    are evicted, the peer r2 keeps physical ownership of the shared
    blocks (same block ids, still allocated) -- eviction unpublishes
    hashes, it never deallocates another request's blocks."""
    scheduler = make_tp_scheduler()
    r1 = create_request(num_tokens=3328, common_prefix_len=1600)
    scheduler.add_request(r1)
    out0 = scheduler.schedule()  # local compute of the shared prefix
    scheduler.update_from_output(
        out0, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    (r1_blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    pool = scheduler.kv_cache_manager.block_pool

    r2 = create_request(num_tokens=3328, common_prefix_len=1600)
    scheduler.add_request(r2)
    out1 = scheduler.schedule()  # r2 matches the published local prefix
    (r2_blocks,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)
    shared = set(r1_blocks) & set(r2_blocks)
    assert shared  # the local prefix is genuinely shared
    shared_ids = sorted(shared)

    # r1 now takes an external claim for its remaining prompt and fails
    # through the flat channel.
    admit_sync_external_hit(scheduler, r1, 3072 - 1600)
    output = create_model_runner_output(reqs=[r1, r2], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        invalid_block_ids={sorted(r1_blocks)[100]}
    )
    scheduler.update_from_output(out1, output)

    # r2's ownership of the shared blocks is untouched.
    (r2_blocks_after,) = scheduler.kv_cache_manager.get_block_ids(r2.request_id)
    assert set(shared_ids) <= set(r2_blocks_after)
    for bid in shared_ids:
        assert pool.blocks[bid].ref_cnt >= 1
    assert r2.num_computed_tokens >= 1600


# ---------------------------------------------------------------------------
# Compatibility: the recompute policy and the default are unchanged.
# ---------------------------------------------------------------------------


def test_recompute_policy_flat_failure_keeps_upstream_truncation():
    """Off the repair policy the flat channel's semantics are untouched:
    truncation to the failed block (not the fail-closed rollback 0),
    exactly the pre-H14 behavior (Part IV compatibility)."""
    scheduler = make_scheduler(
        kv_load_failure_policy="recompute", enable_prefix_caching=False
    )
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)

    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(invalid_block_ids={block_ids[96]})
    scheduler.update_from_output(out1, output)
    assert r1.num_computed_tokens == 96 * BS
    assert r1.request_id not in scheduler._untrusted_external_reqs


def test_default_policy_publication_and_no_evidence_required():
    """No failure policy: the sync claim publishes at admission (the old
    semantics) and an evidence-free step is not a violation -- the
    evidence contract exists only under the repair policy."""
    scheduler = make_scheduler()  # default policy, APC on
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()

    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.blocks[block_ids[96]].block_hash is not None

    scheduler.update_from_output(
        out1, create_model_runner_output(reqs=[r1], use_eos=False)
    )
    assert r1.num_computed_tokens == 3328
    assert r1.request_id not in scheduler._untrusted_external_reqs
