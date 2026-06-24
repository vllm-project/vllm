# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static tests for the forkserver setup block in api_server.py.

The forkserver setup runs inside ``build_async_engine_client`` which is an
async context manager that needs an EngineCore subprocess to exercise
end-to-end. These tests verify the structural invariants of the block
(ordering, no bare ``assert``, platform gating) by inspecting the module
source — cheap to run, catches the four HIGH-severity audit findings.
"""

import inspect
import re
import textwrap

import pytest


def _forkserver_block_source() -> str:
    """Return the body source of build_async_engine_client up to the
    end of the forkserver setup block (everything before
    ``# Context manager`` marker).
    """
    from vllm.entrypoints.openai import api_server

    src = inspect.getsource(api_server.build_async_engine_client)
    # Truncate at the engine_client lifecycle marker so we only inspect
    # the forkserver block.
    cut_marker = "# Context manager to handle engine_client lifecycle"
    assert cut_marker in src, "lifecycle marker missing — file refactored?"
    return src.split(cut_marker, 1)[0]


def test_forkserver_block_has_no_bare_assert():
    """HIGH finding (a): bare `assert` is stripped under `python -O`,
    silently disabling the CUDA-init guard. The fix uses `if ...: raise`.
    """
    block = _forkserver_block_source()
    # Match `assert` as a keyword (whitespace around). We allow `assert`
    # to appear inside a string literal / comment, but no statement-level
    # bare assert.
    assert_lines = [
        line
        for line in block.splitlines()
        if re.match(r"^\s*assert\b", line)
    ]
    assert not assert_lines, (
        "build_async_engine_client must not use bare `assert` for the "
        "CUDA-init guard — strip-on-`-O` would silently disable it. "
        f"Found: {assert_lines}"
    )


def test_forkserver_block_uses_runtime_error():
    """HIGH finding (a) positive: the guard must raise RuntimeError."""
    block = _forkserver_block_source()
    assert "raise RuntimeError" in block, (
        "Expected a `raise RuntimeError(...)` in the forkserver block "
        "(replacing the previous bare `assert`)"
    )


def test_cuda_init_check_runs_before_set_start_method():
    """HIGH finding (b): the CUDA-init check must precede
    `set_start_method("forkserver")` because set_start_method mutates
    the process-global default and any post-flip raise leaves the
    interpreter in an unrecoverable state.
    """
    block = _forkserver_block_source()
    # Find positions; both must be present, and the check must come first.
    raise_pos = block.find("raise RuntimeError")
    set_method_pos = block.find('set_start_method("forkserver"')
    assert raise_pos != -1, "RuntimeError raise missing"
    assert set_method_pos != -1, 'set_start_method("forkserver"...) missing'
    assert raise_pos < set_method_pos, (
        "CUDA-init guard must raise BEFORE set_start_method('forkserver') "
        "so a CUDA-already-initialized parent never flips the global mp "
        "default into an unrecoverable forkserver state."
    )


def test_maybe_force_spawn_runs_before_forkserver_block():
    """HIGH finding (c): `_maybe_force_spawn()` must be invoked at the
    top of the function so its env-var override takes effect BEFORE we
    decide whether to set up forkserver.
    """
    block = _forkserver_block_source()
    force_spawn_pos = block.find("_maybe_force_spawn()")
    forkserver_check_pos = block.find('"forkserver"')
    assert force_spawn_pos != -1, (
        "build_async_engine_client must call _maybe_force_spawn() before "
        "the forkserver setup so a forced spawn override is honored."
    )
    assert force_spawn_pos < forkserver_check_pos


def test_set_start_method_uses_force_kwarg():
    """HIGH finding (c): set_start_method must be called with `force=True`
    so the call is idempotent across re-entry (double-init, child re-imports).
    """
    block = _forkserver_block_source()
    assert (
        'set_start_method("forkserver", force=True)' in block
        or "set_start_method('forkserver', force=True)" in block
    ), (
        'set_start_method("forkserver") must be called with force=True so '
        "`_maybe_force_spawn`-driven state changes don't leave a stale "
        "global default and so re-entry is safe."
    )


def test_cuda_check_is_platform_gated():
    """The torch.cuda.is_initialized() probe must be gated on a
    cuda-alike platform — on XPU/CPU/etc. it's the wrong test and may
    spuriously trip or no-op confusingly.
    """
    block = _forkserver_block_source()
    assert "current_platform.is_cuda_alike()" in block, (
        "CUDA-init guard must be gated on current_platform.is_cuda_alike() "
        "so XPU / ROCm-non-cuda-alike / CPU paths skip the torch.cuda probe."
    )
