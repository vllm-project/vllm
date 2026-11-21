# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H14-03: MultiConnector preserves the transfer-evidence contract.

The wrapper delegates external loading to child connectors, so it must
also delegate the contract's two halves: the scheduler->worker half
(note_transfer_attempt reaching the child actually chosen for the
request -- and only it) and the worker->scheduler half (structured
failure reports and positive attestations aggregated from the children
without dropping or merging). Before H14 the wrapper inherited the
base no-op implementations: the chosen child never learned its
attempt's generation (every report it produced was identity-less and
the scheduler rejected it -- the repair policy silently degraded to
permanent recompute), and the children's evidence was discarded
entirely.
"""

from unittest.mock import Mock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.outputs import KVLoadAttestation, KVLoadFailureReport

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def report_for(req_id, generation):
    return KVLoadFailureReport(
        request_id=req_id,
        failed_block_ids=[7],
        token_span=(112, 128),
        rollback_boundary=112,
        observed_failed_layers=[2],
        healthy_above_span=True,
        restored_extent=128,
        source_rank=None,
        transfer_generation=generation,
    )


def attestation_for(req_id, generation, rank=0):
    return KVLoadAttestation(
        request_id=req_id,
        expected_objects=8,
        loaded_objects=8,
        failed_objects=0,
        certification="sha256",
        certified_objects=8,
        certification_failures=0,
        terminal=True,
        transfer_generation=generation,
        source_rank=rank,
    )


class FakeChild:
    """A child connector with independently configurable evidence and
    recording of the transfer-attempt notifications it received."""

    def __init__(
        self,
        name="child",
        reports=(),
        attestations=(),
        raise_reports=False,
        raise_attestations=False,
    ):
        self.name = name
        self._reports = list(reports)
        self._attestations = list(attestations)
        self._raise_reports = raise_reports
        self._raise_attestations = raise_attestations
        self.notified: list[tuple[str, int]] = []

    def note_transfer_attempt(self, request, generation):
        self.notified.append((request.request_id, generation))

    def get_load_failure_reports(self):
        if self._raise_reports:
            raise RuntimeError(self.name + " report getter exploded")
        return list(self._reports)

    def get_load_attestations(self):
        if self._raise_attestations:
            raise RuntimeError(self.name + " attestation getter exploded")
        return list(self._attestations)


def make_multi(children, selection=None):
    """A MultiConnector with the given children and an explicit
    request->connector-index selection map (no vllm config, no worker)."""
    multi = object.__new__(MultiConnector)
    multi._connectors = list(children)
    multi._requests_to_connector = dict(selection or {})
    multi._broken_evidence_children = set()
    return multi


def request(req_id):
    r = Mock()
    r.request_id = req_id
    return r


# ---------------------------------------------------------------------------
# note_transfer_attempt: chosen child only.
# ---------------------------------------------------------------------------


def test_attempt_noted_by_first_connector_only():
    children = [FakeChild("c0"), FakeChild("c1")]
    multi = make_multi(children, selection={"r1": 0})
    multi.note_transfer_attempt(request("r1"), 42)
    assert children[0].notified == [("r1", 42)]
    assert children[1].notified == []


def test_attempt_noted_by_second_connector_only():
    children = [FakeChild("c0"), FakeChild("c1")]
    multi = make_multi(children, selection={"r1": 1})
    multi.note_transfer_attempt(request("r1"), 7)
    assert children[0].notified == []
    assert children[1].notified == [("r1", 7)]


def test_attempt_with_no_selection_notifies_no_one():
    """No child was chosen for the request: there is no load attempt to
    notify anyone about (notifying all children as though they were
    loading would let them emit evidence for an attempt they never
    executed)."""
    children = [FakeChild("c0"), FakeChild("c1")]
    multi = make_multi(children)
    multi.note_transfer_attempt(request("r1"), 42)
    assert children[0].notified == []
    assert children[1].notified == []


def test_generation_forwarded_unchanged():
    child = FakeChild()
    multi = make_multi([child], selection={"r1": 0})
    multi.note_transfer_attempt(request("r1"), 12345)
    assert child.notified == [("r1", 12345)]


def test_two_requests_two_selections():
    children = [FakeChild("c0"), FakeChild("c1")]
    multi = make_multi(children, selection={"r1": 0, "r2": 1})
    multi.note_transfer_attempt(request("r1"), 1)
    multi.note_transfer_attempt(request("r2"), 2)
    assert children[0].notified == [("r1", 1)]
    assert children[1].notified == [("r2", 2)]


# ---------------------------------------------------------------------------
# Evidence aggregation: nothing dropped, nothing merged.
# ---------------------------------------------------------------------------


def test_failure_reports_propagated_from_loading_child():
    report = report_for("r1", 42)
    children = [FakeChild("c0"), FakeChild("c1", reports=[report])]
    multi = make_multi(children)
    out = multi.get_load_failure_reports()
    assert out == [report]


def test_attestations_propagated_with_fields_preserved():
    attestation = attestation_for("r1", 42, rank=3)
    children = [FakeChild("c0", attestations=[attestation]), FakeChild("c1")]
    multi = make_multi(children)
    out = multi.get_load_attestations()
    assert out == [attestation]
    assert out[0].transfer_generation == 42
    assert out[0].source_rank == 3
    assert out[0].certification == "sha256"


def test_empty_child_evidence_yields_empty_lists():
    multi = make_multi([FakeChild(), FakeChild()])
    assert multi.get_load_failure_reports() == []
    assert multi.get_load_attestations() == []


def test_conflicting_child_evidence_both_preserved():
    """Two children producing evidence for the same request (one of them
    unselected or misbehaving): both statements survive -- attribution
    and identity are the scheduler's judgment, not the wrapper's."""
    a = report_for("r1", 42)
    b = report_for("r1", 42)
    b.rollback_boundary = 96  # divergent metadata
    children = [FakeChild("c0", reports=[a]), FakeChild("c1", reports=[b])]
    multi = make_multi(children)
    out = multi.get_load_failure_reports()
    assert len(out) == 2
    assert a in out and b in out


def test_reports_and_attestations_from_different_children_combined():
    report = report_for("r1", 42)
    attestation = attestation_for("r1", 42)
    children = [
        FakeChild("c0", reports=[report]),
        FakeChild("c1", attestations=[attestation]),
    ]
    multi = make_multi(children)
    assert multi.get_load_failure_reports() == [report]
    assert multi.get_load_attestations() == [attestation]


def test_child_exception_cannot_become_apparent_success():
    """A child raising in an evidence getter does not silently complete
    the step as healthy. F3 semantics: a child whose report channel
    raised is EXCLUDED from the attestation drain too (its healthy-
    looking positive statements cannot vouch for a step whose negative
    channel failed), and a child whose attestation getter raised
    contributes no attestations. Both children broken -> BOTH evidence
    lists come back empty: the scheduler's completeness gates see the
    missing evidence and fail affected requests closed (the recompute
    fallback), never a silent pass."""
    report = report_for("r1", 42)
    attestation = attestation_for("r1", 42)
    children = [
        FakeChild("c0", reports=[report], raise_attestations=True),
        FakeChild("c1", attestations=[attestation], raise_reports=True),
    ]
    multi = make_multi(children)
    # The drain order the model runner uses: reports first (memo reset),
    # then attestations.
    assert multi.get_load_failure_reports() == [report]  # c1 raised -> memo
    assert multi.get_load_attestations() == []  # both children excluded


def test_all_children_raising_yields_empty_evidence():
    children = [
        FakeChild("c0", raise_reports=True, raise_attestations=True),
        FakeChild("c1", raise_reports=True, raise_attestations=True),
    ]
    multi = make_multi(children)
    assert multi.get_load_failure_reports() == []
    assert multi.get_load_attestations() == []


def test_broken_report_child_healthy_attestations_excluded():
    """F3 (adversarial review): a child whose FAILURE-REPORT getter
    raises while its ATTESTATION getter returns healthy statements must
    not convert a real failure into apparent success. The broken child
    is excluded from BOTH evidence getters for the step (memo reset at
    the next report drain), so the scheduler's gates see missing
    evidence and fail closed -- never a silent publish over a
    swallowed failure report."""
    attestation = attestation_for("r1", 42)
    child = FakeChild("c0", attestations=[attestation], raise_reports=True)
    healthy = FakeChild("c1", attestations=[attestation_for("r2", 43)])
    multi = make_multi([child, healthy])

    # The drain order the model runner uses: reports first, then
    # attestations.
    assert multi.get_load_failure_reports() == []
    out = multi.get_load_attestations()
    assert out == [attestation_for("r2", 43)]
    assert attestation not in out

    # The memo resets on the next report drain: a recovered child is
    # re-admitted (its reports flow again).
    child._raise_reports = False
    child._reports = [report_for("r1", 42)]
    assert multi.get_load_failure_reports() == [report_for("r1", 42)]
    assert multi.get_load_attestations() == [
        attestation,
        attestation_for("r2", 43),
    ]


def test_evidence_lists_are_copies():
    """The wrapper must not hand out a child's internal list (a caller
    mutating the aggregate would corrupt the child's state)."""
    child = FakeChild(reports=[report_for("r1", 1)])
    multi = make_multi([child])
    out = multi.get_load_failure_reports()
    out.clear()
    assert child.get_load_failure_reports() != []
