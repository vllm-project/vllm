# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Require immutable references for external GitHub Actions and workflows."""

import sys
from collections.abc import Iterator
from pathlib import Path

import regex as re
import yaml


def unpinned_actions(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "uses" and isinstance(item, str):
                if item.startswith("./"):
                    continue
                pattern = (
                    r"docker://[^\s@]+@sha256:[0-9a-f]{64}"
                    if item.startswith("docker://")
                    else r"[^\s@]+@[0-9a-f]{40}"
                )
                if not re.fullmatch(pattern, item):
                    yield item
            else:
                yield from unpinned_actions(item)
    elif isinstance(value, list):
        for item in value:
            yield from unpinned_actions(item)


def main() -> int:
    failed = False
    for filename in sys.argv[1:]:
        for action in unpinned_actions(yaml.safe_load(Path(filename).read_text())):
            print(f"{filename}: pin {action} to a full commit SHA or image digest")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
