# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the build summary CLI without compiling kernels."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools/report_build_time_ninja.py"


@pytest.mark.parametrize(
    "records, expected",
    [
        ("", ["No build steps found in ninja log."]),
        ("interrupted record\n", ["No build steps found in ninja log."]),
        (
            "10\t10\t0\tfast.o\tabc\n",
            ["1 build steps completed", "N/A parallelism", "average of N/A"],
        ),
        (
            "0\t1000\t0\ta.o\tabc\n1000\t2000\t0\tb.o\tdef\n",
            ["2 build steps completed", "1.0x parallelism", "average of 1.00/s"],
        ),
        (
            "10\t10\t0\ta.o\tabc\n10\t10\t0\tb.o\tdef\n",
            ["2 build steps completed", "N/A parallelism", "average of N/A"],
        ),
        (
            "0\t1000\t0\ta.o\tabc\n0\t1000\t0\tb.o\tdef\n",
            ["2 build steps completed", "2.0x parallelism", "average of 2.00/s"],
        ),
        (
            "0\t1000\t0\ta.o\tabc\n0\t1000\t0\tb.o\tabc\n",
            ["1 build steps completed", "1.0x parallelism", "average of 1.00/s"],
        ),
        (
            "0\t1000\t0\ta.o\tabc\n0\t500\t0\tb.o\tdef\n",
            ["1 build steps completed", "1.0x parallelism", "average of 2.00/s"],
        ),
    ],
    ids=[
        "header-only",
        "interrupted-record",
        "zero-duration",
        "serial",
        "concurrent-zero-duration",
        "parallel",
        "multiple-outputs",
        "incremental-build",
    ],
)
def test_build_summary_cli(tmp_path, records, expected):
    log = tmp_path / ".ninja_log"
    log.write_text("# ninja log v5\n" + records)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--log-file", str(log)],
        capture_output=True,
        text=True,
        check=True,
    )
    for text in expected:
        assert text in result.stdout
