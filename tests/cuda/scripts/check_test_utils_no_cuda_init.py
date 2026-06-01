#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that importing shared test utils does not initialize CUDA."""

import sys
from pathlib import Path

import torch  # noqa: E402

assert not torch.cuda.is_initialized(), "CUDA initialized before import"


def find_repo_root() -> Path:
    for path in Path(__file__).resolve().parents:
        if (path / "pyproject.toml").is_file() and (
            path / "tests" / "utils.py"
        ).is_file():
            return path
    raise RuntimeError("Could not locate vLLM repository root")


# Buildkite runs CUDA tests from /vllm-workspace/tests, so the repository root
# is not guaranteed to be importable when this script runs as a subprocess.
sys.path.insert(0, str(find_repo_root()))

from tests.utils import create_new_process_for_each_test  # noqa: E402, F401

assert not torch.cuda.is_initialized(), "CUDA was initialized during tests.utils import"
print("OK")
