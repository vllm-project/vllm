# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATION = REPO_ROOT / "tests/models/language/generation"
HELPERS = [
    GENERATION / "_common_models.py",
    GENERATION / "_granite_models.py",
    GENERATION / "_hybrid_models.py",
]


@pytest.mark.parametrize("helper", HELPERS, ids=lambda p: p.name)
def test_shared_models_helper_is_never_collected(helper: Path):
    """Shared test bodies must not be collected as tests themselves.

    The ``_*_models.py`` modules hold the shared bodies of the split
    ``test_models`` files under ``generation/core/``, ``core_slow/``,
    ``hybrid/`` and ``extended/``. Their leading
    underscore keeps them out of pytest's ``test_*.py`` collection; even
    when a helper is passed to pytest explicitly, it must yield zero
    collected items so no case is ever double-collected.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(helper),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    collected = [
        line
        for line in result.stdout.splitlines()
        if "::" in line and line.startswith("tests/")
    ]
    assert result.returncode in (0, 5), (
        f"collection errored (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert not collected, (
        f"helper must never collect items, got {collected}:\n"
        f"{result.stdout}\n{result.stderr}"
    )
