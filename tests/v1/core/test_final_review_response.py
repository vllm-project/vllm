# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final independent-review response: the evidence-battery bounds (F-1..F-6).

Every test pins a fail-closed check added in response to the final
fresh-context adversarial review; each shape was first REPRODUCED against
the pre-fix scheduler by the reviewer (reports/final-independent-review.md)
and fails there:

* F-1 (HIGH): restored_extent is bounded by the attempt's external claim --
  an inflated restoration can no longer push the commit's frontier (and
  with it publication) past the claimed content, wedging the request.
* F-3 (MEDIUM): the failed span must lie INSIDE the attempt's external
  claim; a span beyond it is a misreporting connector.
* F-2 (MEDIUM): every terminal rank's expected-object set must COVER the
  claim (>= one object per claimed block, unit-agnostic) -- an
  under-reporting attestation can no longer certify a full claim.
* F-6 (LOW): an over-stated executed_layers verdict is malformed -> abort.
* F-4 (MEDIUM): data_parallel_size>1 is rejected at init (the repair
  consensus is TP-group-scoped), mirroring the PP>1 gate.
"""

import pytest

from tests.v1.core.test_scheduler_repair_budget import (
    BS,
    _structured_report,
    _terminal_attestation,
    drive_repair_result,
)
from tests.v1.core.test_scheduler_transfer_evidence import (
    deliver_evidence,
    full_evidence,
    sync_claim_scenario,
)
from tests.v1.kv_connector.unit.utils import (
    create_scheduler,
    create_vllm_config,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_f1_inflated_restored_extent_fails_closed():
    """The HIGH finding, closed: restored_extent beyond the attempt's
    external claim is a malformed report -> request-scoped recompute,
    nothing published, the source marked untrusted."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=1)
    report = _structured_report(scheduler, r1, restored_extent=1_000_000)
    reports, attestations = full_evidence(scheduler, r1, report=report)
    deliver_evidence(scheduler, out1, [r1], reports=reports, attestations=attestations)

    # The malformed disposition (the pre-existing battery's semantics):
    # no plan, full-recompute floor, everything evicted, the source NOT
    # yet burned (a retry may honestly succeed); the engine stays alive.
    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    out2 = scheduler.schedule()
    assert out2.repair_works is None
    assert out2.num_scheduled_tokens[r1.request_id] > 0


def test_f3_failed_span_beyond_claim_fails_closed():
    """F-3: the failed span must lie inside the attempt's external claim
    (the failure is a failed EXTERNAL object). A span over the
    request's own incoming region is a misreporting connector -- the
    pre-fix scheduler accepted it and advanced the trusted frontier
    onto unverified rows."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=1)
    (block_ids,) = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    report = _structured_report(
        scheduler,
        r1,
        token_span=(200 * BS, 201 * BS),
        rollback_boundary=200 * BS,
        restored_extent=201 * BS,
        failed_block_ids=[block_ids[200]],
    )
    reports, attestations = full_evidence(scheduler, r1, report=report)
    deliver_evidence(scheduler, out1, [r1], reports=reports, attestations=attestations)

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    out2 = scheduler.schedule()
    assert out2.repair_works is None


def test_f2_under_covering_attestation_fails_closed():
    """F-2: a terminal, well-formed, reconciled, certifying attestation
    whose expected set does not COVER the claim (2 objects for a
    192-block claim) can no longer authorize a plan over the whole
    claim."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=1)
    report = _structured_report(scheduler, r1)
    tiny = _terminal_attestation(
        r1.request_id,
        expected=2,
        loaded=1,
        failed=1,
        certified=1,
        generation=scheduler._req_transfer_generation.get(r1.request_id),
    )
    deliver_evidence(scheduler, out1, [r1], reports=[report], attestations=[tiny])

    assert scheduler.recovery_plans == {}
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_f2_covering_evidence_still_admits():
    """The conforming shape still passes the new coverage tie: the
    fixture's 192-object expected set covers the 192-block claim."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=1)
    reports, attestations = full_evidence(
        scheduler, r1, report=_structured_report(scheduler, r1)
    )
    deliver_evidence(scheduler, out1, [r1], reports=reports, attestations=attestations)
    assert r1.request_id in scheduler.recovery_plans


def test_f6_overstated_executed_layers_is_malformed_abort():
    """F-6: structural_check=True with executed_layers beyond the plan's
    own layer range is malformed -> the abort fallback, never a clean
    commit of an over-stated diagnostic."""
    scheduler, r1, out1 = sync_claim_scenario(tp_size=1)
    reports, attestations = full_evidence(
        scheduler, r1, report=_structured_report(scheduler, r1)
    )
    deliver_evidence(scheduler, out1, [r1], reports=reports, attestations=attestations)
    plan = scheduler.recovery_plans[r1.request_id]
    out2 = scheduler.schedule()
    assert r1.request_id in out2.repair_works
    drive_repair_result(
        scheduler,
        r1,
        out2,
        structural_check=True,
        executed_layers=99999,
        plan=plan,
    )
    assert r1.num_computed_tokens == 96 * BS
    assert scheduler.recovery_plans == {}


def test_f4_data_parallel_rejected_at_init():
    """F-4: the repair verdict consensus is TP-group-scoped (the worker
    all_gather has no TP-group handle); DP>1 must be rejected at init
    exactly like PP>1, not dead-lock the world group mid-flight."""
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
    vllm_config.parallel_config.data_parallel_size = 2
    with pytest.raises(RuntimeError, match=r"data_parallel_size=1"):
        create_scheduler(vllm_config)
