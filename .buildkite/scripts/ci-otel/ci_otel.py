# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dependency-free OTLP exporter for command and pytest CI spans."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import secrets
import struct
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ENDPOINT = os.getenv("CI_INFRA_OTEL_ENDPOINT", "https://ci.vllm.ai/api/otel/v1/traces")
AUDIENCE = os.getenv("CI_INFRA_OTEL_AUDIENCE", "https://ci.vllm.ai/api/otel")
MAX_BATCH_SIZE = 2_000


def _upload_timeout_seconds() -> float:
    try:
        value = float(os.getenv("CI_INFRA_OTEL_UPLOAD_TIMEOUT", "3"))
        return value if value > 0 else 3.0
    except (TypeError, ValueError):
        return 3.0


UPLOAD_TIMEOUT_SECONDS = _upload_timeout_seconds()


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


def _spool_dir() -> Path | None:
    value = os.getenv("CI_INFRA_OTEL_SPOOL_DIR")
    return Path(value) if value else None


def record_spans(spans: Iterable[Span]) -> bool:
    """Append spans to a process-local spool file without doing network I/O."""
    try:
        spool_dir = _spool_dir()
        records = [json.dumps(asdict(span), separators=(",", ":")) for span in spans]
        if spool_dir is None or not records:
            return False
        spool_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        spool_file = spool_dir / f"spans-{os.getpid()}.jsonl"
        with spool_file.open("a", encoding="utf-8") as output:
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
            with spool_file.open(encoding="utf-8") as records:
                for record in records:
                    if record.strip():
                        spans.append(Span(**json.loads(record)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
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

    agent_endpoint = os.getenv(
        "BUILDKITE_AGENT_ENDPOINT", "https://agent.buildkite.com/v3"
    ).rstrip("/")
    request = urllib.request.Request(
        f"{agent_endpoint}/jobs/{job_id}/oidc/tokens",
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
        body = json.loads(response.read())
    token = body.get("token") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token:
        raise RuntimeError("Buildkite OIDC response did not contain a token")
    return token


def _safe_oidc_claims(token: str) -> str:
    """Describe non-secret identity claims without logging the signed token."""
    try:
        payload = token.split(".", 2)[1].encode()
        padding = b"=" * (-len(payload) % 4)
        encoded = (payload + padding).translate(bytes.maketrans(b"-_", b"+/"))
        claims = json.loads(binascii.a2b_base64(encoded))
        names = (
            "iss",
            "aud",
            "organization_slug",
            "pipeline_slug",
            "build_branch",
            "build_source",
            "build_number",
            "job_id",
        )
        return json.dumps(
            {name: claims.get(name) for name in names},
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        return "unavailable"


def export_spans(
    spans: list[Span], timeout_seconds: float = UPLOAD_TIMEOUT_SECONDS
) -> bool:
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
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        token = _oidc_token(deadline)
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
            try:
                with urllib.request.urlopen(
                    request, timeout=_remaining_seconds(deadline)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"OTLP endpoint returned {response.status}")
            except urllib.error.HTTPError as error:
                if error.code == 401:
                    raise RuntimeError(
                        "OTLP endpoint returned 401; OIDC claims: "
                        f"{_safe_oidc_claims(token)}"
                    ) from error
                raise
        return True
    except Exception as error:
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
    subparsers.add_parser("flush")
    command = subparsers.add_parser("record-command")
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
    if args.command == "record-command":
        record_spans([_command_span(args)])
        return 0
    if args.command == "flush":
        export_spans(load_spans())
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
