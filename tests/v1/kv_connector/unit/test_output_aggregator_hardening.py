# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregator rank attribution for load-failure reports .

The scheduler's report handling needs to name the
conflicting workers in its diagnostics; the only component that knows
which output came from which rank is the TP output aggregator. These
tests pin the stamping contract: every report the scheduler receives
carries the index of the worker that produced it, connectors never set
the tag themselves, and the repair-verdict channel is untouched. The
same contract covers the positive attestation channel, whose stamp is
the identity the completeness gate COUNTS against the TP size.
"""

import pytest

from tests.v1.kv_connector.unit.test_output_aggregator import (
    DummyModelRunnerOutput,
)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.outputs import KVLoadFailureReport

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _report(req_id="req-1", **overrides):
    fields = dict(
        request_id=req_id,
        failed_block_ids=[7],
        token_span=(16, 32),
        rollback_boundary=16,
        observed_failed_layers=[3],
        healthy_above_span=True,
        restored_extent=64,
    )
    fields.update(overrides)
    return KVLoadFailureReport(**fields)


def test_aggregator_stamps_source_rank_per_output_index():
    """Each rank's reports are tagged with that rank's index in the
    gathered outputs -- the diagnostic the divergent-report fallback
    names the conflicting workers by."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(load_failure_reports=[_report()]),
        DummyModelRunnerOutput(load_failure_reports=[_report()]),
    ]
    aggregated = aggregator.aggregate(outputs)

    reports = aggregated.kv_connector_output.load_failure_reports
    assert [r.source_rank for r in reports] == [0, 1]


def test_aggregator_overwrites_connector_supplied_source_rank():
    """The tag is AUTHORITATIVELY aggregator-owned: a connector-supplied
    value (a contract violation -- or an attempt to impersonate another
    rank in the divergence diagnostics) is always overwritten with the
    true originating rank (the final audit's FA-2)."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(load_failure_reports=[_report(source_rank=1)]),
        DummyModelRunnerOutput(load_failure_reports=[_report(source_rank=0)]),
    ]
    aggregated = aggregator.aggregate(outputs)
    reports = aggregated.kv_connector_output.load_failure_reports
    # Both reports came from their actual output indices, regardless of
    # what the connector claimed.
    assert [r.source_rank for r in reports] == [0, 1]


def test_aggregator_stamps_rank_with_multiple_reports_and_requests():
    """Per-report stamping across a heterogeneous step: rank 0 reports
    two requests' faults, rank 1 reports one; every report is tagged
    with its own rank."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(
            load_failure_reports=[_report("req-1"), _report("req-2")]
        ),
        DummyModelRunnerOutput(load_failure_reports=[_report("req-1")]),
    ]
    aggregated = aggregator.aggregate(outputs)
    reports = aggregated.kv_connector_output.load_failure_reports
    assert [(r.request_id, r.source_rank) for r in reports] == [
        ("req-1", 0),
        ("req-2", 0),
        ("req-1", 1),
    ]


def test_single_rank_reports_stamped_zero():
    aggregator = KVOutputAggregator(expected_finished_count=1)
    outputs = [DummyModelRunnerOutput(load_failure_reports=[_report()])]
    aggregated = aggregator.aggregate(outputs)
    (report,) = aggregated.kv_connector_output.load_failure_reports
    assert report.source_rank == 0


def test_no_reports_leaves_channel_empty():
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(),
        DummyModelRunnerOutput(load_failure_reports=[]),
    ]
    aggregated = aggregator.aggregate(outputs)
    assert aggregated.kv_connector_output.load_failure_reports is None


def test_salvaged_output_evidence_stamped_rank_none():
    """Outputs salvaged from a failed worker replay into a later
    aggregate; their evidence carries no authoritative rank (the
    originating worker is untracked across the replay) and is stamped
    None rather than trusted as any particular rank's statement."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    salvaged = DummyModelRunnerOutput(load_failure_reports=[_report()])
    aggregator.ft_defer_kv_connector_outputs([salvaged.kv_connector_output, None])
    live = DummyModelRunnerOutput(load_failure_reports=[_report()])
    aggregated = aggregator.aggregate([live, DummyModelRunnerOutput()])
    reports = aggregated.kv_connector_output.load_failure_reports
    assert [r.source_rank for r in reports] == [None, 0]
