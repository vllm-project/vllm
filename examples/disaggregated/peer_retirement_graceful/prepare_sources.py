# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fetch or copy two hash-pinned v0.26.0 files for offline CPU source tests."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from patch_vllm import HASHES

SOURCE_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/"
    "vllm/distributed/kv_transfer/kv_connector/v1/nixl/"
)
DESTINATION = Path(__file__).resolve().parent / ".sources"
MAX_BYTES = 2 * 1024 * 1024


def checked_sources(source_dir: Path | None = None) -> dict[str, bytes]:
    sources = {}
    for name, expected in HASHES.items():
        if source_dir is None:
            with urllib.request.urlopen(SOURCE_URL + name, timeout=30) as response:
                data = response.read(MAX_BYTES + 1)
        else:
            with (source_dir / name).open("rb") as source:
                data = source.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected vLLM v0.26.0 source: {name}")
        sources[name] = data
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Use an offline source copy")
    args = parser.parse_args()
    # Validate both files before publishing any local test fixture.
    sources = checked_sources(args.source_dir)
    DESTINATION.mkdir(exist_ok=True)
    for name, data in sources.items():
        (DESTINATION / name).write_bytes(data)
        print(f"{name}: {HASHES[name]}")


if __name__ == "__main__":
    main()
