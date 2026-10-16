# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backport deepseek-ai/DeepGEMM#448 into the packaged SM90 JIT header.

The upstream fix by jason (leevan) keeps register A operands alive until
asynchronous WGMMA finishes. Remove this backport when the dependency pin
includes that fix. A separate output preserves user-supplied source trees.
"""

import argparse
from pathlib import Path

OLD_WAIT = """            ptx::warpgroup_wait<0>();
            if (s > 0)
"""
NEW_WAIT = """            ptx::warpgroup_commit_batch();
            // Keep register A operands live until the asynchronous WGMMA completes.
            ptx::warpgroup_wait<0>();
"""
COMMIT = "            ptx::warpgroup_commit_batch();\n"


def patch_header(source: Path, output: Path) -> None:
    text = source.read_text()
    if text.count(NEW_WAIT) != 1:
        if text.count(OLD_WAIT) != 1 or text.count(COMMIT) != 1:
            raise ValueError(
                "Unrecognized SM90 mHC header; check whether DeepGEMM#448 "
                "has been integrated before updating this backport."
            )
        text = text.replace(OLD_WAIT, "            if (s > 0)\n", 1)
        text = text.replace(COMMIT, NEW_WAIT, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    patch_header(args.source, args.output)
