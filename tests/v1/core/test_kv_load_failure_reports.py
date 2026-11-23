# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side consumption of the structured load-failure reporting
channel under the EXISTING policies.

The reporting channel (KVConnectorOutput.load_failure_reports) is
transport-only until the repair policy interprets it. These tests pin
its fail-safe consumption under the existing fail / recompute
policies:

- a reported failure degrades the affected request to the policy's
  conservative treatment (the full-recompute floor: no prefix of an
  external claim whose failure was reported can be trusted), never
  the engine (regression pin for the A-1/D4
  fix: the pre-fix behavior was an engine-wide RuntimeError);
- reports for unknown or finished requests are dropped loudly;
- malformed reports take the same request-scoped refusal;
- consumers of an already-matched claim are rewound when the failure
  is reported late (the consumer-invalidation walk);
- a request whose evidence failed closed is never re-consulted for
  an external hit (the unbounded-re-attempt guard);
- a healthy step is byte-identical with the channel idle.
"""

from unittest.mock import Mock

import pytest

from tests.v1.kv_connector.unit.utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput, KVLoadFailureReport
from vllm.v1.request import RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BS = 16  # the fixture's block_size


def make_scheduler(**kwargs) -> Scheduler:
    vllm_config = create_vllm_config(
        model="facebook/opt-125m",
        # Same geometry as the original fixture model (16 hidden
        # layers, long context) so the validation battery's bounds and
        # the prompt-length fixtures are unchanged.
        hf_overrides={"num_hidden_layers": 16, "max_position_embeddings": 8192},
        max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", 4096),
        kv_load_failure_policy=kwargs.pop("kv_load_failure_policy", "recompute"),
    )
    # Pin the plain Scheduler (the fixture default resolves
    # async_scheduling to the AsyncScheduler on this platform).
    vllm_config.scheduler_config.async_scheduling = False
    if not kwargs.pop("enable_prefix_caching", True):
        vllm_config.cache_config.enable_prefix_caching = False
    return create_scheduler(vllm_config, **kwargs)


def admit_sync_external_hit(scheduler: Scheduler, request, num_tokens: int) -> None:
    """One connector Mock answering a sync external hit: the request is
    admitted RUNNING with the external tokens committed as computed."""
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (num_tokens, False) if req is request else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduler.connector.get_external_cache_hit_sources.return_value = None


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
        # The conforming shape: the rank tag the TP output aggregator
        # (or the TP=1 executor stamp) authoritatively applies.
        source_rank=0,
    )
    fields.update(overrides)
    return KVLoadFailureReport(**fields)


def deliver(scheduler, out, reqs, reports=(), finished_recving=None):
    output = create_model_runner_output(reqs=list(reqs), use_eos=False)
    output.kv_connector_output = KVConnectorOutput(
        finished_recving=finished_recving,
        load_failure_reports=list(reports) or None,
    )
    scheduler.update_from_output(out, output)
    return output


def test_reports_under_recompute_policy_fail_request_closed():
    """A connector emitting structured reports under a
    non-repair policy fails the affected request closed; the engine and
    every other request survive."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    report = _structured_report(scheduler, r1)
    # Pre-fix behavior: RuntimeError("the repair policy is required to
    # consume transfer evidence") killed the engine.
    deliver(scheduler, out1, [r1], reports=[report])
    # The request degrades to the recompute floor; the engine lives on.
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs
    out2 = scheduler.schedule()
    assert out2 is not None


def test_reports_under_fail_policy_error_the_request():
    """Under kv_load_failure_policy='fail' the affected request is
    errored through the existing failure union, exactly like the flat
    channels."""
    scheduler = make_scheduler(kv_load_failure_policy="fail")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    report = _structured_report(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=[report])
    assert r1.is_finished()
    assert r1.status == RequestStatus.FINISHED_ERROR


def test_reports_for_unknown_request_are_dropped():
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    report = _structured_report(scheduler, r1)
    report.request_id = "no-such-request"
    deliver(scheduler, out1, [r1], reports=[report])
    # The real request's step is untouched by a bogus foreign report.
    assert not r1.is_finished()
    assert r1.request_id not in scheduler._untrusted_external_reqs
    assert scheduler.schedule() is not None


def test_malformed_reports_fail_the_request_closed():
    """A report that fails the geometry battery is uninterpretable: the
    affected request still takes the request-scoped refusal (never an
    engine crash, never a silent ignore)."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    report = _structured_report(scheduler, r1, rollback_boundary=96 * BS + 8)
    deliver(scheduler, out1, [r1], reports=[report])
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs


def test_consumer_of_reported_claim_rewound():
    """When a failure report is judged after the external claim was
    published, every request that matched the published prefix is
    rewound below the failed blocks."""
    scheduler = make_scheduler(
        kv_load_failure_policy="recompute", enable_prefix_caching=True
    )
    r1 = create_request(request_id=1, num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    scheduler.schedule()
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
    # The late failure report for r1's claim.
    report = _structured_report(scheduler, r1)
    deliver(scheduler, out2, [r2], reports=[report])
    # Pre-fix shape: r2 kept computed=3328 over the corrupt region.
    assert r2.num_computed_tokens <= 1536
    assert scheduler.kv_cache_manager.block_pool.blocks[corrupt].block_hash is None


def test_refused_request_external_source_not_reconsulted():
    """After the evidence failed closed once, a connector re-offering
    the same external hit cannot start another external attempt: the
    retry is a local recompute."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3328)
    scheduler.add_request(r1)
    admit_sync_external_hit(scheduler, r1, 3072)
    out1 = scheduler.schedule()
    report = _structured_report(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=[report])
    assert r1.num_computed_tokens == 0
    # The connector still offers the full external hit on re-offer...
    out2 = scheduler.schedule()
    # ...but the request recomputes LOCALLY: the whole prompt is
    # scheduled, no external tokens are credited.
    assert out2.num_scheduled_tokens.get(r1.request_id) == 3328


def test_healthy_step_unaffected_by_idle_channel():
    """With no reports the reporting channel is inert: an ordinary step
    behaves identically (no untrusted marks, normal progress)."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    deliver(scheduler, out1, [r1])
    assert not r1.is_finished()
    assert r1.num_computed_tokens > 0
    assert not scheduler._untrusted_external_reqs
    assert scheduler.schedule() is not None


def test_report_for_async_transfer_takes_the_failure_promotion():
    """A report arriving while the request waits for an async load must
    arm the promotion's FAILURE arm (the flat channels' treatment):
    the claim's blocks are freed and the failed-load zeroing bookkeeping
    applies -- the success arm must not publish the untrusted claim."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    scheduler.connector = Mock()
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (3072, True) if req is r1 else (0, False)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduler.connector.get_external_cache_hit_sources.return_value = None
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = ()
    out1 = scheduler.schedule()
    assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    report = _structured_report(scheduler, r1)
    deliver(scheduler, out1, [], reports=[report])
    assert r1.request_id in scheduler.failed_recving_kv_req_ids
    # The transfer terminates (as it does for a failed load): the
    # promotion then takes the failure arm, frees the claim, and the
    # request recomputes locally with no external tokens credited.
    scheduler.connector.get_num_new_matched_tokens.side_effect = lambda req, _: (
        (0, False)
    )
    out1b = scheduler.schedule()
    deliver(scheduler, out1b, [], finished_recving={r1.request_id})
    out2 = scheduler.schedule()
    assert r1.status == RequestStatus.RUNNING
    assert out2.num_scheduled_tokens.get(r1.request_id) == 3072


def test_reports_on_multi_group_layouts_fail_closed_without_crash():
    """Hybrid (multi-group) layouts are outside the single-group
    reporting contract: the report is refused loudly per request
    (mirroring the flat per-block channel's guard) and the request fails
    closed -- never an unpack crash on the scheduler path."""
    import torch

    from tests.v1.kv_connector.unit.utils import make_kv_cache_group
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

    vllm_config = create_vllm_config(
        kv_load_failure_policy="recompute",
        max_num_batched_tokens=4096,
    )
    vllm_config.scheduler_config.async_scheduling = False
    spec = AttentionSpec(
        block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    scheduler = create_scheduler(
        vllm_config,
        kv_cache_config=KVCacheConfig(
            num_blocks=10000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                make_kv_cache_group(["layer_a"], spec),
                make_kv_cache_group(["layer_b"], spec),
            ],
        ),
    )
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    groups = scheduler.kv_cache_manager.get_block_ids(r1.request_id)
    assert len(groups) == 2
    report = KVLoadFailureReport(
        request_id=r1.request_id,
        failed_block_ids=[groups[0][1]],
        token_span=(16, 32),
        rollback_boundary=16,
        observed_failed_layers=[3],
        healthy_above_span=True,
        restored_extent=3072,
    )
    deliver(scheduler, out1, [r1], reports=[report])
    assert r1.num_computed_tokens == 0
    assert r1.request_id in scheduler._untrusted_external_reqs
    assert scheduler.schedule() is not None


def test_repeated_report_delivery_is_idempotent():
    """Re-delivering the same report in a later update re-refuses
    harmlessly: the request stays at the recompute floor and the engine
    keeps scheduling."""
    scheduler = make_scheduler(kv_load_failure_policy="recompute")
    r1 = create_request(num_tokens=3072)
    scheduler.add_request(r1)
    out1 = scheduler.schedule()
    report = _structured_report(scheduler, r1)
    deliver(scheduler, out1, [r1], reports=[report])
    assert r1.num_computed_tokens == 0
    out2 = scheduler.schedule()
    deliver(scheduler, out2, [r1], reports=[report])
    assert r1.num_computed_tokens == 0
    assert not r1.is_finished()
    assert scheduler.schedule() is not None
