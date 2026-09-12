#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run Track-B producer/consumer regression tests against this source tree.

Installs import stubs before pytest collects so an unbuilt checkout can still
exercise the shipped Python/Triton paths on a disposable local GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
# Checkout first; strip empty '' so cwd site discovery cannot shadow it.
sys.path = [p for p in sys.path if p not in ("", str(TESTS_DIR))]
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from trackb_import_stubs import install_trackb_import_stubs  # noqa: E402

install_trackb_import_stubs()

# Ensure no previously imported site-packages vllm remains cached.
for _name in list(sys.modules):
    if _name == "vllm" or _name.startswith("vllm."):
        del sys.modules[_name]

import pytest  # noqa: E402


def main() -> int:
    targets = [
        str(TESTS_DIR / "v1/attention/test_sparse_mla_token_to_req_indices.py"),
        str(TESTS_DIR / "kernels/attention/test_compute_global_topk_bounds.py"),
    ]
    return pytest.main(
        ["-v", "--tb=short", "--noconftest", "--import-mode=importlib", *targets, *sys.argv[1:]]
    )


if __name__ == "__main__":
    raise SystemExit(main())
