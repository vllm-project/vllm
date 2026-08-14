# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pytest plugin that emits one OTLP span per collected test."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from ci_otel import Span, export_spans


@dataclass
class TestRun:
    start_ns: int
    end_ns: int = 0
    outcome: str = "unknown"


_runs: dict[str, TestRun] = {}
_spans: list[Span] = []


def _enabled() -> bool:
    return bool(os.getenv("VLLM_CI_TRACE_ID") and os.getenv("VLLM_CI_COMMAND_SPAN_ID"))


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]):
    if _enabled():
        _runs[nodeid] = TestRun(start_ns=time.time_ns())


def pytest_runtest_logreport(report):
    run = _runs.get(report.nodeid)
    if not run:
        return
    if report.failed:
        run.outcome = "failed"
    elif report.skipped and run.outcome != "failed":
        run.outcome = "skipped"
    elif report.when == "call" and run.outcome == "unknown":
        run.outcome = "passed"


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]):
    run = _runs.pop(nodeid, None)
    if not run:
        return
    run.end_ns = time.time_ns()
    _spans.append(_test_span(nodeid, run))


def _test_span(nodeid: str, run: TestRun) -> Span:
    outcome = run.outcome
    return Span(
        trace_id=os.environ["VLLM_CI_TRACE_ID"],
        span_id=os.urandom(8).hex(),
        parent_span_id=os.environ["VLLM_CI_COMMAND_SPAN_ID"],
        name="pytest.test",
        start_ns=run.start_ns,
        end_ns=run.end_ns or time.time_ns(),
        attributes={
            "ci.span.kind": "test",
            "test.nodeid": nodeid,
            "test.file": nodeid.split("::", 1)[0],
            "test.outcome": outcome,
        },
        status_code=2 if outcome == "failed" else 1,
    )


def pytest_sessionfinish(session, exitstatus: int):
    if not _enabled():
        return
    for nodeid, run in list(_runs.items()):
        _spans.append(_test_span(nodeid, run))
    _runs.clear()
    export_spans(_spans)
    _spans.clear()
