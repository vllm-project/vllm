# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HYBRID_HELPER = REPO_ROOT / "tests/models/language/generation/_hybrid_models.py"


def test_hybrid_models_helper_is_never_collected():
    """The shared hybrid test body must not be collected as a test itself.

    ``_hybrid_models.py`` holds the shared body of the split ``test_models``
    slices under ``generation/hybrid/`` and ``generation/hybrid_granite/``.
    Its leading underscore keeps it out of pytest's ``test_*.py`` collection;
    even when the file is passed to pytest explicitly, it must yield zero
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
            str(HYBRID_HELPER),
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
