#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the final-validation audit fixes (agents A/B).

Each test pins the FIXED behavior and would fail against the pre-fix
code (verified during development: every scenario here was first
reproduced as a defect by the corresponding agent repro script under
evidence/agent_a/ or evidence/agent_b/).

- A-1 (D4) structured transfer evidence under a NON-repair policy is
      a request-scoped refusal (was: engine-wide RuntimeError).
- A-2 preemption pops the transfer identity: a late same-generation
      report of the freed attempt is unattributable, never a plan
      authorization (was: plan installed against the recycled block
      table).
- A-3 a divergent re-attestation folded while the plan was pending
      caps the commit's restoration at the repair span end (was: the
      certification-failed above-span extent restored as verified).
- A-4 a consumer that matched a published claim is rewound when the
      same attempt's failure report is finally judged (was: r2 kept
      attending over the corrupt block, ref_cnt blocking eviction).
- A-5 async transfer evidence deferred beyond the liveness bound fails
      the request closed and releases the claim (was: unbounded
      WAITING_FOR_REMOTE_KVS holding blocks).
- B-1 a stale receive-thread crash drops only the crashed attempt's
      buffer entry (was: identity-blind drop destroyed the current
      attempt's terminal evidence).
- B-4 mixed local+external hits: the report battery bounds the span
      and restoration by the attempt's ABSOLUTE extent, not the delta
      claim (was: honest reports rejected as malformed -> full
      recompute).
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    _structured_report,
    _terminal_attestation,
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


def deliver(scheduler, out, reqs, reports=(), attestations=(), finished_recving=None):
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        finished_recving=finished_recving,
        load_failure_reports=list(reports) or None,
        load_attestations=list(attestations) or None,
    )
    scheduler.update_from_output(out, output)
    return output


def test_a1_evidence_under_nonrepair_policy_is_request_scoped():
    """A-1 (D4): a connector emitting structured evidence under a
    non-repair policy fails the affected request closed; the engine and
    every other request survive."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    gen = 1
    report = _structured_report(scheduler, r1)
    report.transfer_generation = gen
    # Pre-fix: RuntimeError("the repair policy is required to consume
    # transfer evidence") killed the engine.
    deliver(scheduler, out1, [r1], reports=[report])
    # The request degrades to the recompute floor; the engine lives on.
    assert r1.num_computed_tokens == 0
    out2 = scheduler.schedule()
    assert out2 is not None


def test_a2_preemption_pops_transfer_identity():
    """A-2: preemption drops the attempt identity, so a late
    same-generation report of the freed attempt cannot authorize a
    recovery plan against the re-admitted request."""
    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    gen = scheduler._req_transfer_generation[r1.request_id]
    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[_terminal_attestation(r1.request_id, generation=gen)],
    )
    # Preempt.
    scheduler.running.remove(r1)
    scheduler._preempt_request(r1, 0.0)
    assert scheduler._req_transfer_generation.get(r1.request_id) is None
    assert scheduler._req_transfer_claim.get(r1.request_id) is None
    # Re-admit with NO external hit; the local recompute completes.
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        0,
        False,
    )
    out2 = scheduler.schedule()
    assert r1.num_computed_tokens == 3329
    # The freed attempt's late report: must NOT install a plan.
    report = _structured_report(scheduler, r1)
    report.transfer_generation = gen
    attestation = _terminal_attestation(
        r1.request_id, generation=gen, loaded=191, failed=1, certified=191
    )
    deliver(scheduler, out2, [r1], reports=[report], attestations=[attestation])
    assert scheduler.recovery_plans.get(r1.request_id) is None
    assert r1.num_computed_tokens == 3329  # untouched by stale evidence


def test_a3_divergent_reattestation_caps_commit_restoration():
    """A-3: a non-certifying re-attestation folded while the plan was
    pending caps the commit's restoration at the repair span end."""
    from tests.v1.core.test_final_audit_adversarial import full_fault_evidence
    from tests.v1.core.test_scheduler_repair_budget import drive_repair_result

    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]
    assert plan.healthy_above_span
    # Divergent re-attestation while the plan is pending: same
    # generation and counts, but certification "none".
    gen = scheduler._req_transfer_generation[r1.request_id]
    divergent = _terminal_attestation(
        r1.request_id,
        generation=gen,
        loaded=191,
        failed=1,
        certified=0,
        certification="none",
    )
    out2 = scheduler.schedule()
    assert out2.repair_works is not None
    deliver(scheduler, out2, [r1], attestations=[divergent])
    drive_repair_result(scheduler, r1, out2, True, plan=plan)
    # The divergent (non-certifying) attestation must not let the commit
    # restore the full pre-failure extent 3072: the restoration is
    # capped at the repair span end at most (the commit-time re-check
    # fires before the verdict; the fail-safe abort arm may cap further
    # to the rollback boundary -- both are the safe bound).
    assert r1.num_computed_tokens <= plan.repair_span[1]


def test_a4_consumer_of_published_claim_rewound_on_late_failure():
    """A-4: when the same attempt's failure report is judged after the
    healthy-attestation publication, every request that matched the
    published prefix is rewound below the failed blocks."""
    scheduler = make_scheduler(
        kv_load_failure_policy="repair", enable_prefix_caching=True
    )
    r1 = create_request(request_id=1, num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    gen = scheduler._req_transfer_generation[r1.request_id]
    deliver(
        scheduler,
        out1,
        [r1],
        attestations=[_terminal_attestation(r1.request_id, generation=gen)],
    )
    (bids1,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    corrupt = bids1[96]
    assert scheduler.kv_cache_manager.block_pool.blocks[corrupt].block_hash is not None
    # r2 matches the published prefix (including the corrupt block).
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        0,
        False,
    )
    r2 = create_request(request_id=1, num_tokens=3328)
    r2.request_id = "id-2"
    scheduler.add_request(r2)
    out2 = scheduler.schedule()
    assert r2.num_computed_tokens == 3328
    # The late failure report for r1's attempt.
    report = _structured_report(scheduler, r1)
    attestation = _terminal_attestation(
        r1.request_id, generation=gen, loaded=191, failed=1, certified=191
    )
    deliver(scheduler, out2, [r2], reports=[report], attestations=[attestation])
    # Pre-fix: r2 kept computed=3328 over the corrupt region.
    assert r2.num_computed_tokens <= 1536
    assert scheduler.kv_cache_manager.block_pool.blocks[corrupt].block_hash is None


def test_a5_async_deferral_bound_fails_closed():
    """A-5: an async attempt whose report stays deferred beyond the
    liveness bound fails the request closed and releases the claim."""
    from tests.v1.core.test_scheduler_repair_budget import (
        admit_async_external_hit,
    )

    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_async_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    gen = scheduler._req_transfer_generation[r1.request_id]
    report = _structured_report(scheduler, r1)
    attestation = _terminal_attestation(
        r1.request_id, generation=gen, loaded=191, failed=1, certified=191
    )
    deliver(scheduler, out1, [], reports=[report], attestations=[attestation])
    assert scheduler._req_pending_async_reports.get(r1.request_id) is not None
    # Beyond the bound (32): silent steps with no connector output.
    bound = scheduler.ASYNC_REPORT_MAX_DEFERRAL_STEPS
    for _ in range(bound + 1):
        outn = scheduler.schedule()
        deliver(scheduler, outn, [])
    # Pre-fix: still WAITING_FOR_REMOTE_KS holding blocks forever.
    assert scheduler._req_pending_async_reports.get(r1.request_id) is None
    assert r1.num_computed_tokens == 0
    (bids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    assert len(bids) == 0
    # Post-review regression: expiry must also RESCHEDULE the request
    # (the attempt can never deliver finished_recving -- that is the
    # shape the bound exists for), release the inflight-KV-load slot,
    # and let the local recompute finish it; pre-fix the request stayed
    # WAITING_FOR_REMOTE_KS forever and leaked the fetch slot.
    assert r1.status == RequestStatus.WAITING
    assert sum(scheduler._kv_fetch_counts.values()) == 0
    for _ in range(20):
        outn = scheduler.schedule()
        deliver(scheduler, outn, [r1])
        if r1.is_finished():
            break
    assert r1.is_finished()


def test_b4_mixed_hit_report_accepted_against_absolute_extent():
    """B-4: mixed local+external hit -- the report battery bounds the
    span and restoration by the ABSOLUTE extent, not the delta claim."""
    from tests.v1.core.test_final_audit_adversarial import full_fault_evidence

    scheduler = make_scheduler(kv_load_failure_policy="repair")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    # The mixed-hit geometry: a local APC hit of 1024 means the DELTA
    # claim recorded by the scheduler is 2048 while the producer's
    # geometry (and restored_extent) is request-absolute at 3072.
    scheduler._req_transfer_claim[r1.request_id] = 2048
    scheduler._req_transfer_claim_extent[r1.request_id] = 3072
    reports, attestations = full_fault_evidence(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=reports, attestations=attestations)
    # Pre-fix: restored_extent=3072 > the delta claim 2048 rejected the
    # honest report ("exceeds the attempt's external claim") -- a full
    # recompute for a representable one-block repair.
    assert scheduler.recovery_plans.get(r1.request_id) is not None
    assert scheduler.recovery_plans[r1.request_id].restored_extent == 3072


def test_e1_divergent_uncertified_reports_rollback_collapses_to_zero():
    """E-1 (agent E, final review): two divergent failure reports for one
    attempt whose terminal attestation does NOT certify content must fail
    closed to the FULL-recompute floor -- the divergent path must not keep
    the reports' rollback prefix (unverified external content) any more
    than the single-report path does (H14-02)."""
    from unittest.mock import Mock

    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )
    from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        # Same geometry as the original fixture model (16 hidden
        # layers, long context) so the validation battery's bounds
        # and the prompt-length fixtures are unchanged.
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        max_num_batched_tokens=4096,
        kv_load_failure_policy="repair",
    )
    vllm_config.cache_config.enable_prefix_caching = False
    # Pin the plain Scheduler: the fixture default resolves
    # async_scheduling on this platform, which the repair policy
    # refuses at init; the folding subject below is
    # scheduling-mode-orthogonal (same pin as make_scheduler).
    vllm_config.scheduler_config.async_scheduling = False
    scheduler = create_scheduler(vllm_config)
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.connector = Mock()
    scheduler.connector.get_external_cache_hit_sources.return_value = None
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = ()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (3072, False) if req is r1 else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    out = scheduler.schedule()
    gen = scheduler._req_transfer_generation[r1.request_id]
    (blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)

    def report(idx):
        return KVLoadFailureReport(
            request_id=r1.request_id,
            failed_block_ids=[blocks[idx]],
            token_span=(idx * 16, (idx + 1) * 16),
            rollback_boundary=idx * 16,
            observed_failed_layers=[3],
            healthy_above_span=False,
            restored_extent=3072,
            source_rank=0,
            transfer_generation=gen,
        )

    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=[report(96), report(97)],
        load_attestations=[
            KVLoadAttestation(
                request_id=r1.request_id,
                expected_objects=192,
                loaded_objects=190,
                failed_objects=2,
                certification="none",
                certified_objects=0,
                certification_failures=0,
                terminal=True,
                transfer_generation=gen,
                source_rank=0,
            )
        ],
    )
    scheduler.update_from_output(out, output)
    # Pre-fix: computed stayed 1536 (the reports' min rollback kept as
    # consumed UNVERIFIED external content).
    assert r1.num_computed_tokens == 0
    assert scheduler.recovery_plans.get(r1.request_id) is None


def test_e2_divergent_incomplete_attestations_rollback_collapses_to_zero():
    """E-2 (agent E, final review): divergent cross-rank reports with a
    SILENT rank (incomplete attestations) must fail closed to the
    full-recompute floor -- the kept prefix would otherwise rest on a
    shard no rank certified for the group."""
    from unittest.mock import Mock

    from tests.v1.kv_connector.unit.utils import (
        create_scheduler,
        create_vllm_config,
    )
    from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        # Same geometry as the original fixture model (16 hidden
        # layers, long context) so the validation battery's bounds
        # and the prompt-length fixtures are unchanged.
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        max_num_batched_tokens=4096,
        kv_load_failure_policy="repair",
    )
    vllm_config.cache_config.enable_prefix_caching = False
    # Pin the plain Scheduler: the fixture default resolves
    # async_scheduling on this platform, which the repair policy
    # refuses at init; the folding subject below is
    # scheduling-mode-orthogonal (same pin as make_scheduler).
    vllm_config.scheduler_config.async_scheduling = False
    scheduler = create_scheduler(vllm_config)
    scheduler._evidence_tp_size = 2
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    scheduler.connector = Mock()
    scheduler.connector.get_external_cache_hit_sources.return_value = None
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = ()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (3072, False) if req is r1 else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    out = scheduler.schedule()
    gen = scheduler._req_transfer_generation[r1.request_id]
    (blocks,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)

    def report(idx, rank):
        return KVLoadFailureReport(
            request_id=r1.request_id,
            failed_block_ids=[blocks[idx]],
            token_span=(idx * 16, (idx + 1) * 16),
            rollback_boundary=idx * 16,
            observed_failed_layers=[3],
            healthy_above_span=False,
            restored_extent=3072,
            source_rank=rank,
            transfer_generation=gen,
        )

    output = create_model_runner_output(reqs=[r1], use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        load_failure_reports=[report(96, 0), report(80, 1)],
        load_attestations=[
            KVLoadAttestation(
                request_id=r1.request_id,
                expected_objects=192,
                loaded_objects=191,
                failed_objects=1,
                certification="sha256",
                certified_objects=191,
                certification_failures=0,
                terminal=True,
                transfer_generation=gen,
                source_rank=0,
            )
        ],
    )
    scheduler.update_from_output(out, output)
    # Pre-fix: computed stayed 1280 (min rollback) with rank 1 silent.
    assert r1.num_computed_tokens == 0
    assert scheduler.recovery_plans.get(r1.request_id) is None
