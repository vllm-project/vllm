# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for set_pdeathsig and orphan-worker prevention.

Verifies that:
1. set_pdeathsig() correctly configures prctl(PR_SET_PDEATHSIG) on Linux.
2. EngineCoreProc worker processes are terminated when their parent exits
   unexpectedly (regression test for
   https://github.com/vllm-project/vllm/issues/34643).
"""

import multiprocessing
import os
import signal
import sys
import time

import pytest

from vllm.utils.system_utils import set_pdeathsig

# ---------------------------------------------------------------------------
# Unit tests for set_pdeathsig()
# ---------------------------------------------------------------------------


def test_set_pdeathsig_no_op_on_non_linux():
    """set_pdeathsig is a no-op (no exception) on non-Linux platforms."""
    if sys.platform == "linux":
        pytest.skip("Platform is Linux; testing non-Linux no-op path elsewhere")
    # Should not raise
    set_pdeathsig(signal.SIGTERM)


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux-only")
def test_set_pdeathsig_succeeds_on_linux():
    """set_pdeathsig completes without error on Linux."""
    # Should not raise
    set_pdeathsig(signal.SIGTERM)
    set_pdeathsig(signal.SIGKILL)
    # Clear the pdeathsig (set to 0)
    set_pdeathsig(0)


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux-only")
def test_set_pdeathsig_reads_back_correctly():
    """Verify that prctl(PR_SET_PDEATHSIG) takes effect by reading it back."""
    import ctypes
    import ctypes.util

    PR_GET_PDEATHSIG = 2

    libc_name = ctypes.util.find_library("c")
    assert libc_name, "libc not found"
    libc = ctypes.CDLL(libc_name, use_errno=True)

    # Set via our helper
    set_pdeathsig(signal.SIGTERM)

    # Read back using PR_GET_PDEATHSIG
    value = ctypes.c_int(0)
    ret = libc.prctl(PR_GET_PDEATHSIG, ctypes.byref(value), 0, 0, 0)
    assert ret == 0, "PR_GET_PDEATHSIG failed"
    assert value.value == signal.SIGTERM

    # Clean up
    set_pdeathsig(0)
    ret = libc.prctl(PR_GET_PDEATHSIG, ctypes.byref(value), 0, 0, 0)
    assert ret == 0
    assert value.value == 0


# ---------------------------------------------------------------------------
# Integration test: orphan prevention
# ---------------------------------------------------------------------------


def _child_with_pdeathsig(result_file: str) -> None:
    """Child process: sets PR_SET_PDEATHSIG and writes its PID to a file."""
    set_pdeathsig(signal.SIGTERM)
    with open(result_file, "w") as f:
        f.write(str(os.getpid()))
    # Sleep long enough for the parent to die
    time.sleep(30)


def _parent_that_spawns_and_exits(child_target, result_file: str) -> None:
    """Grandparent's child: spawns a grandchild then exits immediately."""
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(target=child_target, args=(result_file,))
    child.start()
    # Give the child time to call set_pdeathsig and write its PID
    time.sleep(0.5)
    # Exit without waiting for the child — simulating a crash
    os._exit(0)


@pytest.mark.skipif(sys.platform != "linux", reason="prctl is Linux-only")
def test_child_dies_when_parent_exits(tmp_path):
    """When a parent process exits, a child that called set_pdeathsig(SIGTERM)
    receives SIGTERM and terminates rather than becoming an orphan.

    Process tree:
        pytest (grandparent)
          └── parent  (spawned here, exits immediately)
                └── child  (calls set_pdeathsig; should die with parent)
    """
    result_file = str(tmp_path / "child_pid.txt")

    ctx = multiprocessing.get_context("spawn")
    parent = ctx.Process(
        target=_parent_that_spawns_and_exits,
        args=(_child_with_pdeathsig, result_file),
    )
    parent.start()
    parent.join(timeout=5)
    assert parent.exitcode == 0, "Parent process did not exit cleanly"

    # Read the child's PID
    deadline = time.monotonic() + 3.0
    child_pid = None
    while time.monotonic() < deadline:
        if os.path.exists(result_file):
            with open(result_file) as f:
                text = f.read().strip()
            if text:
                child_pid = int(text)
                break
        time.sleep(0.05)

    assert child_pid is not None, "Child did not write its PID in time"

    # Give the child time to receive SIGTERM and exit
    time.sleep(1.0)

    # The child should no longer be alive
    try:
        os.kill(child_pid, 0)  # signal 0 = check existence
        pytest.fail(
            f"Child process {child_pid} is still alive after parent exited "
            "(orphan process detected)"
        )
    except ProcessLookupError:
        pass  # Expected: child is gone
    except PermissionError:
        # Process exists but we don't own it (shouldn't happen in test).
        pytest.fail(f"Child process {child_pid} still alive (PermissionError)")
