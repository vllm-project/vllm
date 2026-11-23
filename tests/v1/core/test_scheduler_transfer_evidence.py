# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Positive transfer evidence: the silent-rank completeness contract (H10).

The report channel's fundamental gap, closed here: failure reports are
NEGATIVE evidence. A rank whose GET 'succeeded' against same-size stale
content under a reused key -- or one that failed before reporting --
supplies no report at all, so reports alone cannot distinguish 'every
rank verified its portion' from 'one rank verified and the rest said
nothing'. The attestation contract supplies the positive half: every
rank that participated in a load attempt attests, terminally, its
expected object set, its completion accounting, and the integrity
certification it actually applied.

These tests drive the real Scheduler (TP=1 and TP=2 fixtures) through
update_from_output with aggregated evidence -- the exact shape the TP
output aggregator produces -- and pin the contract:

- a recovery authorization requires terminal attestations from EVERY
  rank of the TP group; a silent rank demotes the whole external claim
  to a full recompute (rollback 0: TP attention reads every rank's
  shard, so a boundary certified by one rank's report is not certified
  on a silent rank's shard);
- a terminal attestation that accounts for less than its expected
  object set, or one that observed failures/certification failures
  without the matching report, is a contract violation that fails the
  request closed even when the reports look clean;
- identical duplicate attestations from one rank are absorbed;
  divergent ones fail closed;
- the above-span trust (healthy_above_span) is a GLOBAL claim requiring
  every rank's independent content certification: a rank with only
  size-level (W2-OFF positional keys) or no certification forces the
  above-span extent to be recomputed;
- all-healthy evidence (attestations only, no reports) changes nothing:
  no plan, no rejection, the request proceeds.
"""

import copy

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    _structured_report,
    _terminal_attestation,
    admit_sync_external_hit,
    make_scheduler,
)
from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm.v1.outputs import KVConnectorOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def att(scheduler, request, **overrides):
    """A conforming attestation for the request's latest scheduled
    attempt (the generation stamped automatically)."""
    return _terminal_attestation(
        request.request_id,
        generation=scheduler._req_transfer_generation.get(request.request_id),
        **overrides,
    )


def make_tp_scheduler(tp_size: int):
    """A repair-policy scheduler whose evidence gate expects
    attestations from tp_size TP ranks, built through the REAL config
    path (APC off: the private-block fixture shape)."""
    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        # Same geometry as the original fixture model (16 hidden
        # layers, long context) so the validation battery's bounds
        # and the prompt-length fixtures are unchanged.
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        max_num_batched_tokens=4096,
        kv_load_failure_policy="repair",
    )
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.cache_config.enable_prefix_caching = False
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    scheduler = create_scheduler(vllm_config)
    assert scheduler._evidence_tp_size == tp_size
    return scheduler


def sync_claim_scenario(tp_size: int = 1):
    scheduler = make_tp_scheduler(tp_size)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    return scheduler, r1, out1


def deliver_evidence(scheduler, out, reqs, reports=(), attestations=()):
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    scheduler.update_from_output(out, output)


def full_evidence(scheduler, request, *, report=None, tp_size=1, **att_overrides):
    """The conforming shape: the production duplicate-fold -- every TP
    rank reports the identical fault (the reports differ only in the
    aggregator-stamped source_rank; the evidence-key fold absorbs
    them) and every rank attests terminally, reconciled with its own
    report (the failed region excluded from its loaded set; everything
    that loaded is sha-certified)."""
    if report is None:
        report = _structured_report(scheduler, request)
    generation = scheduler._req_transfer_generation.get(request.request_id)
    failed = len(report.failed_block_ids)
    fields = dict(
        loaded=192 - failed,
        failed=failed,
        certified=192 - failed,
    )
    fields.update(att_overrides)
    reports = []
    attestations = []
    for r in range(tp_size):
        rank_report = copy.deepcopy(report)
        rank_report.source_rank = r
        reports.append(rank_report)
        attestations.append(
            _terminal_attestation(
                request.request_id, rank=r, generation=generation, **fields
            )
        )
    return reports, attestations


# ---------------------------------------------------------------------------
# The completeness gate: silence is detected, not assumed.
# ---------------------------------------------------------------------------


def test_complete_evidence_installs_plan_with_healthy_above_span():
    """The canonical shape: every rank attests terminal, complete,
    sha-certified completion; the failure report authorizes the plan and
    the report's above-span claim is honored globally."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=2)
    reports, attestations = full_evidence(scheduler, r1, tp_size=2)
    deliver_evidence(scheduler, out1, [r1], reports=reports, attestations=attestations)

    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None
    assert plan.healthy_above_span is True
    assert r1.num_computed_tokens == 96 * BS


def test_silent_rank_rejects_recovery_with_full_recompute_floor():
    """THE silent-rank shape (TP=2): rank 0 reports the failure and
    attests, rank 1 says NOTHING. The evidence is incomplete: no plan,
    and the rollback floor is ZERO -- not rank 0's certified boundary --
    because TP attention reads every rank's shard, so a prefix certified
    only by the reporting rank is not certified on the silent rank."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=2)
    report = _structured_report(scheduler, r1)
    rank0 = att(scheduler, r1, rank=0, loaded=191, failed=1, certified=191)
    deliver_evidence(scheduler, out1, [r1], reports=[report], attestations=[rank0])

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    # The engine keeps scheduling: the ordinary recompute owns the
    # request from zero.
    out2 = scheduler.schedule()
    assert out2.repair_works is None
    assert out2.num_scheduled_tokens[r1.request_id] == 3328


def test_no_attestations_at_all_rejects_recovery():
    """The legacy-connector bootstrap shape: a well-formed report with
    no positive evidence at all. Under the repair policy the attestation
    contract is mandatory: fail closed, loudly."""
    scheduler, r1, out1 = sync_claim_scenario()
    deliver_evidence(scheduler, out1, [r1], reports=[_structured_report(scheduler, r1)])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_nonterminal_attestation_counts_as_missing():
    """A non-terminal attestation is an in-progress statement: it cannot
    complete the evidence any more than silence can."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=2)
    report = _structured_report(scheduler, r1)
    # Rank 0 terminal, rank 1 in flight.
    attestations = [
        att(scheduler, r1, rank=0, loaded=191, failed=1, certified=191),
        att(scheduler, r1, rank=1, terminal=False),
    ]
    deliver_evidence(scheduler, out1, [r1], reports=[report], attestations=attestations)
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_tp1_requires_exactly_one_attestation():
    """The gate counts distinct attesting ranks against the TP size: at
    TP=1 the single rank's own attestation completes the evidence."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=1)
    reports, attestations = full_evidence(scheduler, r1, tp_size=1)
    deliver_evidence(scheduler, out1, [r1], reports=reports, attestations=attestations)
    assert r1.request_id in scheduler.recovery_plans


# ---------------------------------------------------------------------------
# Contract violations: corruption detected through the positive channel.
# ---------------------------------------------------------------------------


def test_incomplete_object_set_fails_closed():
    """'Success' with a hole: the rank attests terminal completion but
    accounts for fewer objects than its expected set (and no failure
    report covers the difference). The structural battery rejects it;
    the whole external claim demotes to recompute-from-zero."""
    scheduler, r1, out1 = sync_claim_scenario()
    attestation = att(scheduler, r1, expected=192, loaded=191, certified=191)
    deliver_evidence(scheduler, out1, [r1], attestations=[attestation])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_divergent_manifest_fails_closed():
    """'Success' with corrupt content: the rank's sha certification
    caught a digest mismatch (a same-size stale object under a reused
    key is exactly this shape) but no failure report accompanies it.
    Silent corruption is detected, not trusted."""
    scheduler, r1, out1 = sync_claim_scenario()
    attestation = att(scheduler, r1, certified=191, cert_failures=1)
    deliver_evidence(scheduler, out1, [r1], attestations=[attestation])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_valid_failure_with_silent_corruption_on_peer_fails_closed():
    """The P0 mission scenario (TP=2): rank 0 reports a genuine,
    well-formed failure and attests cleanly; rank 1's GET 'succeeded'
    everywhere, its attestation shows the certification caught one
    divergent cell, but rank 1 reports NOTHING. The union of a valid
    report and a silently-corrupt peer never authorizes a recovery:
    rank 1's corruption may extend outside rank 0's reported span."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=2)
    report = _structured_report(scheduler, r1)
    # The report is rank 0's evidence (aggregator-stamped).
    report.source_rank = 0
    rank0 = att(scheduler, r1, rank=0, loaded=191, failed=1, certified=191)
    rank1 = att(scheduler, r1, rank=1, certified=191, cert_failures=1)
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[rank0, rank1]
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_failed_objects_beyond_reported_regions_fail_closed():
    """The count reconciliation: a rank whose attestation counts MORE
    failed objects than its reports describe leaves failures outside
    the reported span -- holes no plan would cover. Fail closed."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    report.source_rank = 0
    # The report names ONE region; the attestation counts TWO failed
    # objects on the same rank.
    attestation = att(scheduler, r1, loaded=190, failed=2, certified=190)
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[attestation]
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


def test_certified_loaded_set_honors_above_span_with_in_span_fault():
    """The W2-ON fault shape: the failed object sits INSIDE the repair
    span (covered by the report) and every object that loaded is
    sha-certified. The above-span claim is provable -- the predicate
    certifies the LOADED set, not the expected set (the span's failed
    object is recomputed, not trusted)."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    report.source_rank = 0
    attestation = att(scheduler, r1, loaded=191, failed=1, certified=191)
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[attestation]
    )
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None
    assert plan.healthy_above_span is True


def test_failed_objects_with_matching_report_is_consistent():
    """The reconciled shape: the attestation's failed objects are
    covered by the same step's structured report (the connector's
    negative and positive channels agree). No violation; the plan
    installs."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    report.source_rank = 0
    attestation = att(scheduler, r1, loaded=191, failed=1, certified=191)
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[attestation]
    )
    assert r1.request_id in scheduler.recovery_plans


def test_certification_failure_with_matching_report_fails_closed():
    """A rank whose certification caught a divergent cell AND reported
    the failure: the report covers the corruption, so the evidence is
    consistent -- but H14-02/F1 semantics: the rank's own loaded set is
    NOT fully certified, so nothing of the claim is publishable and the
    whole delivery fails closed to the full-recompute fallback (the
    above-span trust question never arises)."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    report.source_rank = 0
    attestation = att(
        scheduler, r1, loaded=191, failed=1, certified=190, cert_failures=1
    )
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[attestation]
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_duplicate_identical_attestations_are_absorbed():
    """Same-step duplicate delivery (the aggregator may repeat a rank's
    attestation across gathered outputs): identical duplicates from one
    rank are absorbed, not treated as divergence."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    attestation = att(scheduler, r1, loaded=191, failed=1, certified=191)
    deliver_evidence(
        scheduler,
        out1,
        [r1],
        reports=[report],
        attestations=[attestation, attestation],
    )
    assert r1.request_id in scheduler.recovery_plans


def test_divergent_attestations_from_one_rank_fail_closed():
    """Two attestations from one rank for one request with different
    accounting: unattributable evidence. Fail closed."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    a = att(scheduler, r1)
    b = att(scheduler, r1, certified=191)
    deliver_evidence(scheduler, out1, [r1], reports=[report], attestations=[a, b])
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0


# ---------------------------------------------------------------------------
# The global above-span trust gate (successful transfer != verified data).
# ---------------------------------------------------------------------------


def test_healthy_above_span_requires_every_ranks_certification():
    """The report claims the above-span ranges landed byte-exact, and
    rank 0's attestation certifies its full set -- but rank 1 (TP=2)
    applied NO certification (W2-OFF with positional keys). The claim is
    a global decision, and H14-02/F1 tightened it one step further: the
    missing certification does not merely cap the restoration -- the
    whole delivery (including the rollback prefix) is unverified
    external content and fails closed to the full-recompute fallback."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=2)
    report = _structured_report(scheduler, r1)
    rank0 = att(scheduler, r1, rank=0, loaded=191, failed=1, certified=191)
    rank1 = att(scheduler, r1, rank=1, certification="none", certified=0)
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[rank0, rank1]
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_size_only_certification_is_not_content_verification():
    """The same-size-stale-object shape: the connector checked only
    lengths (certification='size'), which a reused key with stale
    same-size content passes. 'Successful GET' is not 'verified data':
    H14-02/F1 -- nothing of the claim (prefix or above-span) is
    publishable, and the delivery fails closed to the full-recompute
    fallback."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    attestation = att(
        scheduler,
        r1,
        certification="size",
        certified=0,
        loaded=191,
        failed=1,
    )
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[attestation]
    )
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_content_addressed_keys_honor_the_claim_under_store_trust():
    """Certification='content' (content-addressed immutable keys: GET
    success implies correct content iff the store's key->data binding
    is trustworthy) honors the above-span claim: the W2-OFF conditional
    safety assumption, made explicit and enforceable."""
    scheduler, r1, out1 = sync_claim_scenario()
    report = _structured_report(scheduler, r1)
    attestation = att(
        scheduler,
        r1,
        certification="content",
        loaded=191,
        failed=1,
        certified=191,
    )
    deliver_evidence(
        scheduler, out1, [r1], reports=[report], attestations=[attestation]
    )
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan is not None
    assert plan.healthy_above_span is True


# ---------------------------------------------------------------------------
# The all-healthy path: positive evidence with no fault anywhere.
# ---------------------------------------------------------------------------


def test_all_ranks_healthy_no_reports_changes_nothing():
    """Every rank attests terminal, complete, certified success -- no
    failure reports at all (nothing failed). No plan, no rejection, no
    truncation: the request keeps its scheduled state and the ordinary
    flow owns it."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=2)
    computed_before = r1.num_computed_tokens
    attestations = [att(scheduler, r1, rank=r) for r in range(2)]
    deliver_evidence(scheduler, out1, [r1], attestations=attestations)
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == computed_before
    assert not r1.is_finished()


def test_rejected_claim_keeps_blocks_unpublished():
    """The publication gate: a request whose evidence failed closed
    (silent rank) never publishes its sync external claim -- unmatchable
    is the safe state for unverified content (APC ON)."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    block_pool = scheduler.kv_cache_manager.block_pool

    report = _structured_report(scheduler, r1)
    rank0 = att(scheduler, r1, rank=0, loaded=191, failed=1, certified=191)
    # TP=2, rank 1 silent.
    scheduler._evidence_tp_size = 2
    deliver_evidence(scheduler, out1, [r1], reports=[report], attestations=[rank0])
    assert scheduler._evidence_tp_size == 2

    assert scheduler.recovery_plans == {}
    # Nothing published: the whole claim stays unmatchable.
    for bid in block_ids:
        assert block_pool.blocks[bid].block_hash is None
