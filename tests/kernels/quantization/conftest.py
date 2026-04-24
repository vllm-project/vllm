# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""conftest for quantization kernel tests.

Adds session-finish hook that updates golden baseline JSON files in place
when ``--write-golden`` is set.  Existing entries are retained, and
re-measured entries overwrite their old values.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_GOLDEN_DIR = _HERE / "golden"

_ATTR = "_hybrid_w4a16_measured_results"
_TEMP_ATTR = "_hybrid_w4a16_temp_log"
_PRELOAD_ATTR = "_hybrid_w4a16_preloaded_gpus"


def get_measured_results(config: pytest.Config) -> dict[str, list[dict]]:
    """Return (creating if needed) the session-scoped baseline dict."""
    d = getattr(config, _ATTR, None)
    if d is None:
        d = {}
        setattr(config, _ATTR, d)
    return d


def get_temp_log(config: pytest.Config) -> list[tuple[float, str, float]]:
    """Return (creating if needed) the session-scoped temperature log."""
    log = getattr(config, _TEMP_ATTR, None)
    if log is None:
        log = []
        setattr(config, _TEMP_ATTR, log)
    return log


def preload_golden(config: pytest.Config, gpu: str) -> None:
    """Load existing golden JSON for *gpu* into the collector if not already done."""
    preloaded: set[str] = getattr(config, _PRELOAD_ATTR, set())
    if gpu in preloaded:
        return
    preloaded.add(gpu)
    setattr(config, _PRELOAD_ATTR, preloaded)
    results = get_measured_results(config)
    golden_path = _GOLDEN_DIR / f"hybrid_w4a16_{gpu}.json"
    if golden_path.exists():
        data = json.loads(golden_path.read_text())
        results[gpu] = data.get("shapes", [])


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Write temperature log if path was set
    import os

    temp_log_path = os.environ.get("TEMP_LOG_PATH", "")
    if temp_log_path:
        log = get_temp_log(session.config)
        if log:
            t0 = log[0][0]
            with open(temp_log_path, "w") as f:
                f.write("elapsed_s,label,temp_C\n")
                for ts, label, temp in log:
                    f.write(f"{ts - t0:.2f},{label},{temp:.1f}\n")

    if not session.config.getoption("--write-golden", default=False):
        return

    results = get_measured_results(session.config)
    if not results:
        return

    for gpu, shapes in results.items():
        out_path = _GOLDEN_DIR / f"hybrid_w4a16_{gpu}.json"
        data = {"gpu": gpu, "shapes": shapes}
        out_path.write_text(json.dumps(data, indent=2) + "\n")
