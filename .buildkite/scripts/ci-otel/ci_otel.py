# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-free OTLP exporter and pytest timing plugin for CI."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
import sys
import time
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ENDPOINT = os.getenv("CI_INFRA_OTEL_ENDPOINT", "https://ci.vllm.ai/api/otel/v1/traces")
AUDIENCE = os.getenv("CI_INFRA_OTEL_AUDIENCE", "https://ci.vllm.ai/api/otel")
MAX_BATCH_SIZE = 2_000


@dataclass(frozen=True)
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ns: int
    end_ns: int
    attributes: dict[str, str | int | bool]
    status_code: int = 1


def _varint(value: int) -> bytes:
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _field(number: int, wire_type: int) -> bytes:
    return _varint((number << 3) | wire_type)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _field(number, 2) + _varint(len(value)) + value


def _string_field(number: int, value: str) -> bytes:
    return _bytes_field(number, value.encode())


def _varint_field(number: int, value: int) -> bytes:
    return _field(number, 0) + _varint(value)


def _fixed64_field(number: int, value: int) -> bytes:
    return _field(number, 1) + struct.pack("<Q", value)


def _attribute(name: str, value: str | int | bool) -> bytes:
    if isinstance(value, bool):
        encoded_value = _varint_field(2, int(value))
    elif isinstance(value, int):
        encoded_value = _varint_field(3, value)
    else:
        encoded_value = _string_field(1, value)
    return _string_field(1, name) + _bytes_field(2, encoded_value)


def _resource_attributes() -> dict[str, str | int | bool]:
    build_number = os.getenv("BUILDKITE_BUILD_NUMBER", "0")
    attributes: dict[str, str | int | bool] = {
        "service.name": "vllm-ci",
        "buildkite.organization.slug": os.getenv("BUILDKITE_ORGANIZATION_SLUG", ""),
        "buildkite.pipeline.slug": os.getenv("BUILDKITE_PIPELINE_SLUG", ""),
        "buildkite.build.id": os.getenv("BUILDKITE_BUILD_ID", ""),
        "buildkite.build.number": int(build_number) if build_number.isdigit() else 0,
        "buildkite.build.branch": os.getenv("BUILDKITE_BRANCH", ""),
        "buildkite.build.commit": os.getenv("BUILDKITE_COMMIT", ""),
        "buildkite.job.id": os.getenv("BUILDKITE_JOB_ID", ""),
        "buildkite.job.label": os.getenv("BUILDKITE_LABEL", ""),
        "buildkite.agent.queue": os.getenv("BUILDKITE_AGENT_META_DATA_QUEUE", ""),
    }
    return {name: value for name, value in attributes.items() if value != ""}


def _encode_span(span: Span) -> bytes:
    encoded = b"".join(
        (
            _bytes_field(1, bytes.fromhex(span.trace_id)),
            _bytes_field(2, bytes.fromhex(span.span_id)),
            _bytes_field(4, bytes.fromhex(span.parent_span_id))
            if span.parent_span_id
            else b"",
            _string_field(5, span.name),
            _varint_field(6, 1),
            _fixed64_field(7, span.start_ns),
            _fixed64_field(8, span.end_ns),
        )
    )
    attributes = dict(span.attributes)
    if os.getenv("BUILDKITE_BUILD_URL") and os.getenv("BUILDKITE_JOB_ID"):
        attributes["buildkite.job.web_url"] = (
            f"{os.environ['BUILDKITE_BUILD_URL']}#{os.environ['BUILDKITE_JOB_ID']}"
        )
    for name, value in attributes.items():
        encoded += _bytes_field(9, _attribute(name, value))
    return encoded + _bytes_field(15, _varint_field(3, span.status_code))


def encode_request(spans: Iterable[Span]) -> bytes:
    resource = b"".join(
        _bytes_field(1, _attribute(name, value))
        for name, value in _resource_attributes().items()
    )
    scope_spans = _bytes_field(1, _string_field(1, "vllm.ci"))
    for span in spans:
        scope_spans += _bytes_field(2, _encode_span(span))
    return _bytes_field(1, _bytes_field(1, resource) + _bytes_field(2, scope_spans))


def new_context() -> tuple[str, str, str | None]:
    parts = os.getenv("TRACEPARENT", "").lower().split("-")
    valid = (
        len(parts) == 4
        and [len(part) for part in parts] == [2, 32, 16, 2]
        and all(all(char in "0123456789abcdef" for char in part) for part in parts)
        and parts[1] != "0" * 32
        and parts[2] != "0" * 16
    )
    if valid:
        return parts[1], secrets.token_hex(8), parts[2]
    build_id = os.getenv("BUILDKITE_BUILD_ID", "local")
    trace_id = hashlib.sha256(build_id.encode()).hexdigest()[:32]
    return trace_id, secrets.token_hex(8), None


def _spool_dir() -> Path | None:
    value = os.getenv("CI_INFRA_OTEL_SPOOL_DIR")
    return Path(value) if value else None


def record_spans(spans: Iterable[Span]) -> bool:
    try:
        spool_dir = _spool_dir()
        records = [json.dumps(asdict(span), separators=(",", ":")) for span in spans]
        if spool_dir is None or not records:
            return False
        spool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        with (spool_dir / f"spans-{os.getpid()}.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            output.write("\n".join(records) + "\n")
        return True
    except Exception as error:
        print(f"CI timing spool skipped: {error}", file=sys.stderr)
        return False


def load_spans() -> list[Span]:
    spool_dir = _spool_dir()
    if spool_dir is None or not spool_dir.is_dir():
        return []
    spans: list[Span] = []
    for spool_file in sorted(spool_dir.glob("spans-*.jsonl")):
        try:
            spans.extend(
                Span(**json.loads(record))
                for record in spool_file.read_text(encoding="utf-8").splitlines()
                if record
            )
        except Exception as error:
            print(
                f"CI timing spool ignored {spool_file.name}: {error}", file=sys.stderr
            )
    return spans


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("CI timing upload deadline exceeded")
    return remaining


def _oidc_token(deadline: float) -> str:
    access_token = os.getenv("BUILDKITE_AGENT_ACCESS_TOKEN", "")
    job_id = os.getenv("BUILDKITE_JOB_ID", "")
    if not access_token or not job_id:
        raise RuntimeError("Buildkite job credentials are unavailable")
    endpoint = os.getenv(
        "BUILDKITE_AGENT_ENDPOINT", "https://agent.buildkite.com/v3"
    ).rstrip("/")
    request = urllib.request.Request(
        f"{endpoint}/jobs/{job_id}/oidc/tokens",
        data=json.dumps({"audience": AUDIENCE, "lifetime": 300}).encode(),
        method="POST",
        headers={
            "Authorization": f"Token {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "vllm-ci-otel/1",
        },
    )
    with urllib.request.urlopen(
        request, timeout=_remaining_seconds(deadline)
    ) as response:
        token = json.loads(response.read()).get("token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Buildkite OIDC response did not contain a token")
    return token


def export_spans(spans: list[Span], timeout_seconds: float = 3.0) -> bool:
    if not spans or os.getenv("BUILDKITE") != "true":
        return False
    try:
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        token = _oidc_token(deadline)
        for offset in range(0, len(spans), MAX_BATCH_SIZE):
            request = urllib.request.Request(
                ENDPOINT,
                data=encode_request(spans[offset : offset + MAX_BATCH_SIZE]),
                method="POST",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-protobuf",
                    "User-Agent": "vllm-ci-otel/1",
                },
            )
            with urllib.request.urlopen(
                request, timeout=_remaining_seconds(deadline)
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"OTLP endpoint returned {response.status}")
        return True
    except Exception as error:
        print(f"CI timing upload skipped: {error}", file=sys.stderr)
        return False


_test_runs: dict[str, tuple[int, str]] = {}
_test_spans: list[Span] = []


def _test_span(nodeid: str, start_ns: int, outcome: str) -> Span:
    return Span(
        trace_id=os.environ["CI_INFRA_TRACE_ID"],
        span_id=secrets.token_hex(8),
        parent_span_id=os.environ["CI_INFRA_COMMAND_SPAN_ID"],
        name="pytest.test",
        start_ns=start_ns,
        end_ns=time.time_ns(),
        attributes={
            "ci.span.kind": "test",
            "test.nodeid": nodeid,
            "test.file": nodeid.split("::", 1)[0],
            "test.outcome": outcome,
        },
        status_code=2 if outcome == "failed" else 1,
    )


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]):
    try:
        if os.getenv("CI_INFRA_TRACE_ID") and os.getenv("CI_INFRA_COMMAND_SPAN_ID"):
            _test_runs[nodeid] = (time.time_ns(), "unknown")
    except Exception:
        pass


def pytest_runtest_logreport(report):
    try:
        if report.nodeid not in _test_runs:
            return
        start_ns, outcome = _test_runs[report.nodeid]
        if report.failed:
            outcome = "failed"
        elif report.skipped and outcome != "failed":
            outcome = "skipped"
        elif report.when == "call" and outcome == "unknown":
            outcome = "passed"
        _test_runs[report.nodeid] = (start_ns, outcome)
    except Exception:
        pass


def pytest_runtest_logfinish(nodeid: str, location: tuple[str, int | None, str]):
    try:
        run = _test_runs.pop(nodeid, None)
        if run:
            _test_spans.append(_test_span(nodeid, *run))
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus: int):
    try:
        for nodeid, run in list(_test_runs.items()):
            _test_spans.append(_test_span(nodeid, *run))
        _test_runs.clear()
        record_spans(_test_spans)
        _test_spans.clear()
    except Exception:
        pass


def _record_command(arguments: list[str]) -> None:
    trace_id, span_id, parent_span_id, start_ns, index, exit_code, label = arguments
    exit_status = int(exit_code)
    record_spans(
        [
            Span(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=None if parent_span_id == "-" else parent_span_id,
                name="ci.command",
                start_ns=int(start_ns),
                end_ns=time.time_ns(),
                attributes={
                    "ci.span.kind": "command",
                    "ci.command.index": int(index),
                    "ci.command.label": label,
                    "process.exit.code": exit_status,
                },
                status_code=1 if exit_status == 0 else 2,
            )
        ]
    )


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments == ["new-context"]:
        trace_id, span_id, parent_span_id = new_context()
        print(trace_id, span_id, parent_span_id or "-", time.time_ns())
    elif arguments == ["flush"]:
        export_spans(load_spans())
    elif len(arguments) == 8 and arguments[0] == "record-command":
        _record_command(arguments[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
