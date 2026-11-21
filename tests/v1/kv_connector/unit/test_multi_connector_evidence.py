# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MultiConnector preserves the structured load-failure report channel.

The wrapper delegates external loading to child connectors, so it must
also delegate the reporting contract's worker->scheduler half:
structured failure reports aggregated from the children without
dropping or merging (attribution and identity remain the scheduler's
judgment). A child whose getter raises is logged and skipped: its
reports are lost for the step, never fabricated.
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.v1.outputs import KVLoadFailureReport

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def report_for(req_id):
    return KVLoadFailureReport(
        request_id=req_id,
        failed_block_ids=[7],
        token_span=(112, 128),
        rollback_boundary=112,
        observed_failed_layers=[2],
        healthy_above_span=True,
        restored_extent=128,
        source_rank=None,
    )


class FakeChild:
    """A child connector with independently configurable reports."""

    def __init__(self, name="child", reports=(), raise_reports=False):
        self.name = name
        self._reports = list(reports)
        self._raise_reports = raise_reports

    def get_load_failure_reports(self):
        if self._raise_reports:
            raise RuntimeError(self.name + " report getter exploded")
        return list(self._reports)


def make_multi(children):
    """A MultiConnector with the given children (no vllm config, no
    worker)."""
    multi = object.__new__(MultiConnector)
    multi._connectors = list(children)
    return multi


def test_failure_reports_propagated_from_children():
    report = report_for("r1")
    children = [FakeChild("c0"), FakeChild("c1", reports=[report])]
    multi = make_multi(children)
    out = multi.get_load_failure_reports()
    assert out == [report]


def test_empty_child_evidence_yields_empty_lists():
    multi = make_multi([FakeChild(), FakeChild()])
    assert multi.get_load_failure_reports() == []


def test_conflicting_child_evidence_both_preserved():
    """Two children producing evidence for the same request (one of them
    unselected or misbehaving): both statements survive -- attribution
    and identity are the scheduler's judgment, not the wrapper's."""
    a = report_for("r1")
    b = report_for("r1")
    b.rollback_boundary = 96  # divergent metadata
    children = [FakeChild("c0", reports=[a]), FakeChild("c1", reports=[b])]
    multi = make_multi(children)
    out = multi.get_load_failure_reports()
    assert len(out) == 2
    assert a in out and b in out


def test_child_exception_skips_only_that_child():
    """A child raising in its report getter is logged and skipped: its
    reports are lost for the step (never fabricated as healthy), while
    the other children's reports still flow through."""
    report = report_for("r1")
    children = [
        FakeChild("c0", reports=[report]),
        FakeChild("c1", raise_reports=True),
    ]
    multi = make_multi(children)
    assert multi.get_load_failure_reports() == [report]


def test_all_children_raising_yields_empty_evidence():
    children = [
        FakeChild("c0", raise_reports=True),
        FakeChild("c1", raise_reports=True),
    ]
    multi = make_multi(children)
    assert multi.get_load_failure_reports() == []


def test_evidence_lists_are_copies():
    """The wrapper must not hand out a child's internal list (a caller
    mutating the aggregate would corrupt the child's state)."""
    child = FakeChild(reports=[report_for("r1")])
    multi = make_multi([child])
    out = multi.get_load_failure_reports()
    out.clear()
    assert child.get_load_failure_reports() != []
