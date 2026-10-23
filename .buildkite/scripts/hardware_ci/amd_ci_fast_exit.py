# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pytest plugin for AMD CI jobs: leave without native teardown once pytest is done.

A pytest process can crash in native library destructors after its summary:
vllm/ci #96786 printed "236 passed" and then died with SIGSEGV (exit 139) during
interpreter shutdown, past the point where faulthandler reports. Once the outermost
session has finished, this runs the Python exit handlers (engine shutdown, Ray,
multiprocessing cleanup), flushes, and exits with pytest's own status, skipping
module teardown and native destructors. pytest's result is never changed, and a
command chain still stops on a real failure.

Off when a kernel recorder is loaded (it writes its trace during native teardown)
or with VLLM_CI_FAST_EXIT=0.
"""

import atexit
import os
import sys

import pytest

_outer = None
_session = None


def _recorder_loaded() -> bool:
    return bool(
        os.environ.get("ROCP_TOOL_LIBRARIES")
        or os.environ.get("CUDA_INJECTION64_PATH")
        or "rocprofiler" in os.environ.get("LD_PRELOAD", "")
    )


def pytest_configure(config):
    global _outer
    if _outer is None:  # pytester runs nested sessions in-process; act on the outer one
        _outer = config


def pytest_sessionfinish(session, exitstatus):
    global _session
    if session.config is _outer:
        _session = session


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if config is not _outer or _session is None:
        return
    if os.environ.get("VLLM_CI_FAST_EXIT") == "0" or _recorder_loaded():
        return
    status = int(_session.exitstatus)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        atexit._run_exitfuncs()
    except BaseException as exc:  # an exit handler failing must not mask the result
        print(f"[amd-ci] exit handler failed: {exc!r}", file=sys.stderr)
    print(
        f"[amd-ci] pytest finished with exit {status}; leaving without native teardown",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    os._exit(status)
