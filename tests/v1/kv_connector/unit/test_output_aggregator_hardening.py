# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregator rank attribution for load-failure reports (hardening H1)
and transfer attestations (H10).

The scheduler's cross-rank report consensus needs to name the
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
from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

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


# ---------------------------------------------------------------------------
# Positive transfer attestations (H10): the completeness channel.
# ---------------------------------------------------------------------------


def _attestation(req_id="req-1", **overrides):
    fields = dict(
        request_id=req_id,
        expected_objects=4,
        loaded_objects=4,
        failed_objects=0,
        certification="sha256",
        certified_objects=4,
        certification_failures=0,
        terminal=True,
    )
    fields.update(overrides)
    return KVLoadAttestation(**fields)


def test_aggregator_stamps_and_merges_attestations():
    """Every rank's attestation is stamped with its originating index
    and the per-rank lists are concatenated: the scheduler's
    completeness gate counts distinct stamped ranks against the TP
    size, so the stamp IS the identity the gate counts."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(load_attestations=[_attestation()]),
        DummyModelRunnerOutput(load_attestations=[_attestation()]),
    ]
    aggregated = aggregator.aggregate(outputs)
    attestations = aggregated.kv_connector_output.load_attestations
    assert [a.source_rank for a in attestations] == [0, 1]


def test_aggregator_overwrites_connector_supplied_attestation_rank():
    """The attestation stamp is AUTHORITATIVELY aggregator-owned, like
    the reports': a connector-supplied source_rank (an attempt to
    impersonate another rank in the completeness gate -- e.g. a
    contract-violating connector attesting twice under different rank
    identities to fake completeness) is always overwritten with the
    true originating rank."""
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(load_attestations=[_attestation(source_rank=1)]),
        DummyModelRunnerOutput(load_attestations=[]),
    ]
    aggregated = aggregator.aggregate(outputs)
    (attestation,) = aggregated.kv_connector_output.load_attestations
    assert attestation.source_rank == 0


def test_no_attestations_leaves_channel_empty():
    aggregator = KVOutputAggregator(expected_finished_count=2)
    outputs = [
        DummyModelRunnerOutput(),
        DummyModelRunnerOutput(load_attestations=[]),
    ]
    aggregated = aggregator.aggregate(outputs)
    assert aggregated.kv_connector_output.load_attestations is None


def test_reports_and_attestations_coexist_in_one_output():
    """A rank produces its (possibly empty) report list AND its
    attestation in the same output; both channels survive aggregation
    with their own stamps."""
    aggregator = KVOutputAggregator(expected_finished_count=1)
    outputs = [
        DummyModelRunnerOutput(
            load_failure_reports=[_report()],
            load_attestations=[_attestation()],
        ),
    ]
    aggregated = aggregator.aggregate(outputs)
    (report,) = aggregated.kv_connector_output.load_failure_reports
    (attestation,) = aggregated.kv_connector_output.load_attestations
    assert report.source_rank == 0
    assert attestation.source_rank == 0
