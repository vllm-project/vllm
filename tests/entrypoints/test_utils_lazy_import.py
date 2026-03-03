# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests to ensure entrypoints/utils.py doesn't trigger early
platform detection via accessing current_platform.

Uses a subprocess so we get a pristine interpreter with no risk of
mutating global state in the test process.
"""

import subprocess
import sys


def test_utils_import_no_platform_detection():
    """Importing vllm.entrypoints.utils must not trigger platform detection."""
    # Run in a subprocess for full isolation — no mock / patch needed.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ";".join(
                [
                    "import vllm.platforms",
                    "import vllm.entrypoints.utils",
                    "assert vllm.platforms._current_platform is None"
                    ", 'Importing vllm.entrypoints.utils triggered detection'",
                ]
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Subprocess failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
