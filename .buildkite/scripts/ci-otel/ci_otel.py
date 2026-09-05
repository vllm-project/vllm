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
TEST_SPOOL_BATCH_SIZE = 100


def _parse_dockerfile_step(name: str) -> tuple[str, int, int, str] | None:
    """Parse "[stage index/total] label" without regex."""
    if not name.startswith("["):
        return None
    close = name.find("]")
    if close == -1:
        return None
    header = name[1:close]
    label = name[close + 1 :].lstrip()
    if not label:
        return None
    parts = header.rsplit(None, 1)
    if len(parts) != 2:
        return None
    stage, step = parts
    if "/" not in step:
        return None
    index_str, total_str = step.split("/", 1)
    if not index_str.isdigit() or not total_str.isdigit():
        return None
    return stage, int(index_str), int(total_str), label


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
    if value < 0:
        value += 1 << 64
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


def _read_spool() -> tuple[list[Span], list[Path]]:
    spool_dir = _spool_dir()
    if spool_dir is None or not spool_dir.is_dir():
        return [], []
    spans: list[Span] = []
    loaded_files: list[Path] = []
    for spool_file in sorted(spool_dir.glob("spans-*.jsonl")):
        try:
            spans.extend(
                Span(**json.loads(record))
                for record in spool_file.read_text(encoding="utf-8").splitlines()
                if record
            )
            loaded_files.append(spool_file)
        except Exception as error:
            print(
                f"CI timing spool ignored {spool_file.name}: {error}", file=sys.stderr
            )
    return spans, loaded_files


def load_spans() -> list[Span]:
    return _read_spool()[0]


def buildx_ref(metadata_path: str, target: str) -> str:
    """Read the exact Buildx history reference emitted by bake."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    target_metadata = metadata.get(target, metadata)
    reference = target_metadata.get("buildx.build.ref")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"Buildx metadata has no history reference for {target}")
    return reference


def _buildkit_tags(span: dict) -> dict[str, str | int | bool]:
    tags: dict[str, str | int | bool] = {}
    for tag in span.get("tags", []):
        key = tag.get("key")
        value = tag.get("value")
        if isinstance(key, str) and isinstance(value, (str, int, bool)):
            tags[key] = value
    return tags


def _buildkit_parent_ids(span: dict) -> list[str]:
    return [
        reference["spanID"]
        for reference in span.get("references", [])
        if reference.get("refType") == "CHILD_OF"
        and isinstance(reference.get("spanID"), str)
    ]


def _buildkit_span_id(trace_id: str, value: str) -> str:
    return hashlib.sha256(f"{trace_id}:{value}".encode()).hexdigest()[:16]


def buildkit_spans(payload: dict) -> list[Span]:
    """Normalize Buildx's Jaeger JSON into dashboard-oriented OTLP spans."""
    trace_id, _, _ = new_context()
    traces = payload.get("data", [])
    native_trace_id = (
        traces[0].get("traceID", "buildkit")
        if traces and isinstance(traces[0], dict)
        else "buildkit"
    )
    span_scope = (
        os.getenv("CI_INFRA_BUILDX_REF")
        or os.getenv("BUILDKITE_JOB_ID")
        or str(native_trace_id)
    )
    steps: list[tuple[str, int, int, str, dict, int, int, int]] = []

    for trace in traces:
        raw_spans = [
            span
            for span in trace.get("spans", [])
            if isinstance(span, dict) and isinstance(span.get("spanID"), str)
        ]
        spans_by_id = {span["spanID"]: span for span in raw_spans}
        children: dict[str, list[dict]] = {}
        for span in raw_spans:
            for parent_id in _buildkit_parent_ids(span):
                if parent_id in spans_by_id:
                    children.setdefault(parent_id, []).append(span)

        def envelope(
            span: dict, child_spans: dict[str, list[dict]] = children
        ) -> tuple[int, int]:
            start_us = int(span["startTime"])
            end_us = start_us + int(span["duration"])
            pending = list(child_spans.get(span["spanID"], []))
            seen: set[str] = set()
            while pending:
                child = pending.pop()
                child_id = child["spanID"]
                if child_id in seen:
                    continue
                seen.add(child_id)
                child_start = int(child["startTime"])
                start_us = min(start_us, child_start)
                end_us = max(end_us, child_start + int(child["duration"]))
                pending.extend(child_spans.get(child_id, []))
            return start_us * 1_000, end_us * 1_000

        selected: dict[str, tuple[dict, int, int]] = {}
        for span in raw_spans:
            name = span.get("operationName")
            if not isinstance(name, str) or name.startswith("cache request: "):
                continue
            step = _parse_dockerfile_step(name)
            is_internal = name.startswith("[internal] ")
            is_export = name.startswith(("exporting to ", "export to "))
            if not (step or is_internal or is_export):
                continue
            try:
                start_ns, end_ns = envelope(span)
            except (KeyError, TypeError, ValueError):
                continue
            vertex = _buildkit_tags(span).get("vertex")
            key = str(vertex) if vertex else f"{name}:{span['spanID']}"
            previous = selected.get(key)
            if previous is None or end_ns - start_ns > previous[2] - previous[1]:
                selected[key] = (span, start_ns, end_ns)

        for span, start_ns, end_ns in selected.values():
            name = span["operationName"]
            step = _parse_dockerfile_step(name)
            tags = _buildkit_tags(span)
            failed = (
                tags.get("error") is True or tags.get("otel.status_code") == "ERROR"
            )
            if step:
                stage, step_index, step_total, label = step
            else:
                stage = "BuildKit internals"
                label = name.removeprefix("[internal] ")
                step_index = step_total = 0
            steps.append(
                (
                    stage,
                    step_index,
                    step_total,
                    label,
                    tags,
                    start_ns,
                    end_ns,
                    2 if failed else 1,
                )
            )

    output: list[Span] = []
    build_ref = os.getenv("CI_INFRA_BUILDX_REF", "")
    for stage in dict.fromkeys(
        step[0] for step in steps if step[0] != "BuildKit internals"
    ):
        stage_steps = sorted(
            (step for step in steps if step[0] == stage), key=lambda step: step[5]
        )
        stage_id = _buildkit_span_id(trace_id, f"{span_scope}:stage:{stage}")
        stage_start = min(step[5] for step in stage_steps)
        stage_end = max(step[6] for step in stage_steps)
        stage_failed = any(step[7] == 2 for step in stage_steps)
        stage_attributes: dict[str, str | int | bool] = {
            "ci.span.kind": "docker-stage",
            "docker.stage": stage,
            "docker.step.count": len(stage_steps),
        }
        if build_ref:
            stage_attributes["docker.build.ref"] = build_ref
        output.append(
            Span(
                trace_id=trace_id,
                span_id=stage_id,
                parent_span_id=None,
                name="docker.stage",
                start_ns=stage_start,
                end_ns=stage_end,
                attributes=stage_attributes,
                status_code=2 if stage_failed else 1,
            )
        )
        for (
            stage_name,
            index,
            total,
            label,
            tags,
            start_ns,
            end_ns,
            status,
        ) in stage_steps:
            vertex = str(tags.get("vertex", ""))
            attributes: dict[str, str | int | bool] = {
                "ci.span.kind": "docker-instruction",
                "docker.stage": stage_name,
                "docker.step.label": label,
            }
            if index:
                attributes["docker.step.index"] = index
                attributes["docker.step.total"] = total
                attributes["docker.instruction"] = label.split(None, 1)[0]
            if vertex:
                attributes["docker.vertex"] = vertex
            output.append(
                Span(
                    trace_id=trace_id,
                    span_id=_buildkit_span_id(
                        trace_id,
                        f"{span_scope}:step:{stage_name}:{index}:{vertex}:{label}",
                    ),
                    parent_span_id=stage_id,
                    name="docker.instruction",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    attributes=attributes,
                    status_code=status,
                )
            )
    for stage_name, index, total, label, tags, start_ns, end_ns, status in (
        step for step in steps if step[0] == "BuildKit internals"
    ):
        vertex = str(tags.get("vertex", ""))
        attributes: dict[str, str | int | bool] = {
            "ci.span.kind": "docker-internal",
            "docker.stage": stage_name,
            "docker.step.label": label,
        }
        if vertex:
            attributes["docker.vertex"] = vertex
        output.append(
            Span(
                trace_id=trace_id,
                span_id=_buildkit_span_id(
                    trace_id, f"{span_scope}:internal:{vertex}:{label}"
                ),
                parent_span_id=None,
                name="docker.internal",
                start_ns=start_ns,
                end_ns=end_ns,
                attributes=attributes,
                status_code=status,
            )
        )
    return output


def record_buildkit_trace(trace_path: str) -> bool:
    payload = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    return record_spans(buildkit_spans(payload))


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


def flush_spans(timeout_seconds: float = 3.0) -> bool:
    spans, spool_files = _read_spool()
    if not spool_files:
        return True
    if not spans or not export_spans(spans, timeout_seconds):
        return False
    for spool_file in spool_files:
        try:
            spool_file.unlink()
        except FileNotFoundError:
            pass
        except Exception as error:
            print(f"CI timing spool cleanup skipped: {error}", file=sys.stderr)
    return True


_test_runs: dict[str, tuple[int, str]] = {}
_test_spans: list[Span] = []


def _spool_test_spans() -> None:
    if not _test_spans:
        return
    record_spans(_test_spans)
    _test_spans.clear()


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
            if len(_test_spans) >= TEST_SPOOL_BATCH_SIZE:
                _spool_test_spans()
    except Exception:
        pass


def pytest_sessionfinish(session, exitstatus: int):
    try:
        for nodeid, run in list(_test_runs.items()):
            _test_spans.append(_test_span(nodeid, *run))
        _test_runs.clear()
        _spool_test_spans()
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
    try:
        if arguments == ["new-context"]:
            trace_id, span_id, parent_span_id = new_context()
            print(trace_id, span_id, parent_span_id or "-", time.time_ns())
            return 0
        if arguments == ["flush"]:
            return 0 if flush_spans() else 1
        if len(arguments) == 3 and arguments[0] == "build-ref":
            print(buildx_ref(arguments[1], arguments[2]))
            return 0
        if len(arguments) == 2 and arguments[0] == "record-buildkit":
            return 0 if record_buildkit_trace(arguments[1]) else 1
        if len(arguments) == 8 and arguments[0] == "record-command":
            _record_command(arguments[1:])
            return 0
        print(
            "usage: ci_otel.py {new-context|flush|build-ref ...|"
            "record-buildkit ...|record-command ...}",
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        print(f"CI timing helper failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
