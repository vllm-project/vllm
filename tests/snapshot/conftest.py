# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import suppress

import pytest


@pytest.fixture
def sleeping_child() -> Iterator[int]:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal; "
                "signal.signal(signal.SIGUSR2, signal.SIG_IGN); "
                "print('ready', flush=True); "
                "signal.pause()"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline() == "ready\n"
    child.stdout.close()
    try:
        yield child.pid
    finally:
        with suppress(ProcessLookupError):
            child.send_signal(signal.SIGKILL)
        child.wait(timeout=5)
