#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Write the immutable image-build provenance artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
from datetime import datetime

STATIC_FILES = ("build-graph.jsonl", "kernel-map.jsonl")
LOWER_HEX = frozenset("0123456789abcdef")


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in LOWER_HEX for character in value)


def _is_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    directory: pathlib.Path,
    *,
    repository_sha: str,
    image_tag: str,
    image_digest: str,
    created_at: str,
    buildkite_build_id: str | None = None,
) -> dict[str, object]:
    if not _is_lower_hex(repository_sha, 40):
        raise ValueError(f"not a 40-hex repository SHA: {repository_sha!r}")
    if not image_digest.startswith("sha256:") or not _is_lower_hex(
        image_digest.removeprefix("sha256:"), 64
    ):
        raise ValueError(f"not a sha256 image digest: {image_digest!r}")
    if not _is_utc_timestamp(created_at):
        raise ValueError(f"not a UTC RFC3339 timestamp: {created_at!r}")
    files = {}
    for name in STATIC_FILES:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing static provenance file: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return {
        "buildkite_build_id": buildkite_build_id,
        "created_at": created_at,
        "files": files,
        "image_tag": image_tag,
        "image_digest": image_digest,
        "kind": "static-build-provenance",
        "publisher_step_key": "image-build",
        "repository_sha": repository_sha,
        "schema_version": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--buildkite-build-id")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    document = build_manifest(
        args.directory,
        repository_sha=args.repository_sha,
        image_tag=args.image_tag,
        image_digest=args.image_digest,
        created_at=args.created_at,
        buildkite_build_id=args.buildkite_build_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=args.out.parent, suffix=".tmp", delete=False
    ) as stream:
        temporary = pathlib.Path(stream.name)
        stream.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
