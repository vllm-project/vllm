#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-free OTLP exporter for command and pytest CI spans."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

ENDPOINT = os.getenv("VLLM_CI_OTEL_ENDPOINT", "https://ci.vllm.ai/api/otel/v1/traces")
AUDIENCE = os.getenv("VLLM_CI_OTEL_AUDIENCE", "https://ci.vllm.ai/api/otel")
MAX_BATCH_SIZE = 250


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


def _key(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _bytes_field(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _string_field(field: int, value: str) -> bytes:
    return _bytes_field(field, value.encode())


def _varint_field(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _fixed64_field(field: int, value: int) -> bytes:
    return _key(field, 1) + struct.pack("<Q", value)


def _fixed32_field(field: int, value: int) -> bytes:
    return _key(field, 5) + struct.pack("<I", value)


def _any_value(value: str | int | bool) -> bytes:
    if isinstance(value, bool):
        return _varint_field(2, int(value))
    if isinstance(value, int):
        return _varint_field(3, value)
    return _string_field(1, value)


def _attribute(key: str, value: str | int | bool) -> bytes:
    return _string_field(1, key) + _bytes_field(2, _any_value(value))


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
        "buildkite.step.key": os.getenv("BUILDKITE_STEP_KEY", ""),
        "buildkite.agent.queue": os.getenv("BUILDKITE_AGENT_META_DATA_QUEUE", ""),
    }
    return {key: value for key, value in attributes.items() if value != ""}


def _encode_span(span: Span) -> bytes:
    encoded = b"".join(
        [
            _bytes_field(1, bytes.fromhex(span.trace_id)),
            _bytes_field(2, bytes.fromhex(span.span_id)),
            _bytes_field(4, bytes.fromhex(span.parent_span_id))
            if span.parent_span_id
            else b"",
            _string_field(5, span.name),
            _varint_field(6, 1),
            _fixed64_field(7, span.start_ns),
            _fixed64_field(8, span.end_ns),
        ]
    )
    attributes = dict(span.attributes)
    build_url = os.getenv("BUILDKITE_BUILD_URL")
    job_id = os.getenv("BUILDKITE_JOB_ID")
    if build_url and job_id:
        attributes.setdefault("buildkite.job.web_url", f"{build_url}#{job_id}")
    for key, value in attributes.items():
        encoded += _bytes_field(9, _attribute(key, value))
    encoded += _bytes_field(15, _varint_field(3, span.status_code))
    encoded += _fixed32_field(16, 1)
    return encoded


def encode_request(spans: Iterable[Span]) -> bytes:
    resource = b"".join(
        _bytes_field(1, _attribute(key, value))
        for key, value in _resource_attributes().items()
    )
    scope = _string_field(1, "vllm.ci") + _string_field(2, "1")
    scope_spans = _bytes_field(1, scope)
    for span in spans:
        scope_spans += _bytes_field(2, _encode_span(span))
    resource_spans = _bytes_field(1, resource) + _bytes_field(2, scope_spans)
    return _bytes_field(1, resource_spans)


def new_context() -> tuple[str, str, str | None]:
    traceparent = os.getenv("TRACEPARENT", "").lower()
    parts = traceparent.split("-")
    valid_lengths = len(parts) == 4 and [len(part) for part in parts] == [2, 32, 16, 2]
    valid_hex = valid_lengths and all(
        all(character in "0123456789abcdef" for character in part) for part in parts
    )
    if valid_hex and parts[1] != "0" * 32 and parts[2] != "0" * 16:
        trace_id = parts[1]
        parent_span_id = parts[2]
    else:
        build_id = os.getenv("BUILDKITE_BUILD_ID", "local")
        trace_id = hashlib.sha256(build_id.encode()).hexdigest()[:32]
        parent_span_id = None
    return trace_id, secrets.token_hex(8), parent_span_id


def _oidc_token() -> str:
    result = subprocess.run(
        [
            "buildkite-agent",
            "oidc",
            "request-token",
            "--audience",
            AUDIENCE,
            "--lifetime",
            "300",
            "--claim",
            "build_id",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip()


def export_spans(spans: list[Span]) -> bool:
    if not spans or os.getenv("BUILDKITE", "") != "true":
        return False
    required = (
        "BUILDKITE_ORGANIZATION_SLUG",
        "BUILDKITE_PIPELINE_SLUG",
        "BUILDKITE_BUILD_ID",
        "BUILDKITE_BUILD_NUMBER",
        "BUILDKITE_JOB_ID",
        "BUILDKITE_BRANCH",
    )
    if any(not os.getenv(name) for name in required):
        return False

    try:
        token = _oidc_token()
        for offset in range(0, len(spans), MAX_BATCH_SIZE):
            payload = encode_request(spans[offset : offset + MAX_BATCH_SIZE])
            request = urllib.request.Request(
                ENDPOINT,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/x-protobuf",
                    "User-Agent": "vllm-ci-otel/1",
                },
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status != 200:
                    raise RuntimeError(f"OTLP endpoint returned {response.status}")
        return True
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        urllib.error.URLError,
    ) as error:
        print(f"CI timing upload skipped: {error}", file=sys.stderr)
        return False


def _command_span(args: argparse.Namespace) -> Span:
    attributes: dict[str, str | int | bool] = {
        "ci.span.kind": "command",
        "ci.command.index": args.index,
        "ci.command.label": args.label,
        "process.exit.code": args.exit_code,
    }
    return Span(
        trace_id=args.trace_id,
        span_id=args.span_id,
        parent_span_id=args.parent_span_id or None,
        name="ci.command",
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        attributes=attributes,
        status_code=1 if args.exit_code == 0 else 2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("new-context")
    command = subparsers.add_parser("command")
    command.add_argument("--trace-id", required=True)
    command.add_argument("--span-id", required=True)
    command.add_argument("--parent-span-id", default="")
    command.add_argument("--start-ns", required=True, type=int)
    command.add_argument("--end-ns", required=True, type=int)
    command.add_argument("--index", required=True, type=int)
    command.add_argument("--label", required=True)
    command.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    if args.command == "new-context":
        trace_id, span_id, parent_span_id = new_context()
        print(trace_id, span_id, parent_span_id or "-")
        return 0
    export_spans([_command_span(args)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
