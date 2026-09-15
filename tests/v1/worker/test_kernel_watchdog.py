# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import time

from vllm.v1.worker.kernel_watchdog import wedged_kernel_watchdog


def test_watchdog_warns_when_section_stalls(caplog):
    """A section that outlives the timeout must produce a diagnostic naming
    the wedge signature; this is the only evidence a wedged non-output rank
    leaves behind (see #51035)."""
    with (
        caplog.at_level(logging.ERROR),
        wedged_kernel_watchdog(0.2, "execute_model on worker rank 3"),
    ):
        time.sleep(0.45)
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("execute_model on worker rank 3" in m for m in messages)
    assert any("wedged" in m and "#51035" in m for m in messages)


def test_watchdog_silent_when_section_completes(caplog):
    """No diagnostic may be emitted for a section that finishes in time."""
    with (
        caplog.at_level(logging.ERROR),
        wedged_kernel_watchdog(5.0, "execute_model on worker rank 0"),
    ):
        time.sleep(0.05)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_watchdog_disabled_for_nonpositive_timeout(caplog):
    """A non-positive timeout opts out entirely (no thread, no warning)."""
    with (
        caplog.at_level(logging.ERROR),
        wedged_kernel_watchdog(0, "execute_model on worker rank 0"),
    ):
        time.sleep(0.05)
    assert not caplog.records
