# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tools/pre_commit/check_gpu_sync_patterns.py."""

# Load the linter as a module under test.
import importlib.util
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LINTER = _ROOT / "tools" / "pre_commit" / "check_gpu_sync_patterns.py"
_spec = importlib.util.spec_from_file_location("check_gpu_sync_patterns", _LINTER)
assert _spec is not None and _spec.loader is not None
linter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(linter)


def _check(tmp_path: Path, source: str) -> int:
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return linter.scan_file(str(path))


# --- bad cases: the linter must flag ------------------------------------- #


@pytest.mark.parametrize(
    "snippet",
    [
        # from_numpy -> .to(cuda, non_blocking=True)
        "torch.from_numpy(x).to(device, non_blocking=True)",
        # from_numpy -> .cuda(non_blocking=True)
        "torch.from_numpy(x).cuda(non_blocking=True)",
        # explicit device="cpu" then push
        'torch.tensor(indices, dtype=torch.int32, device="cpu")'
        ".to(device, non_blocking=True)",
        # no pin_memory kwarg — implicit unpinned
        "torch.tensor(indices, dtype=torch.int32).to(device, non_blocking=True)",
        # zeros() without pin_memory
        "torch.zeros(1024).to(device, non_blocking=True)",
        # .copy_(unpinned source)
        "buf.copy_(torch.from_numpy(x), non_blocking=True)",
        # .copy_(torch.tensor(list)) — no pin_memory
        "buf.copy_(torch.tensor(indices, dtype=torch.int32), non_blocking=True)",
    ],
)
def test_bad_cases_flagged(tmp_path: Path, snippet: str) -> None:
    assert _check(tmp_path, snippet) == 1


# --- good cases: the linter must stay silent ----------------------------- #


@pytest.mark.parametrize(
    "snippet",
    [
        # No non_blocking kwarg — different concern
        "torch.from_numpy(x).to(device)",
        # Explicit pin_memory=True
        "torch.tensor(indices, pin_memory=True).to(device, non_blocking=True)",
        # Truthy pin_memory reference (e.g., a module-level flag)
        "torch.tensor(indices, pin_memory=PIN_MEMORY).to(device, non_blocking=True)",
        # Constructed directly on device (device=some_name)
        "torch.zeros(1024, device=device).to(device, non_blocking=True)",
        # Constructed on a literal non-cpu device string
        'torch.zeros(1024, device="cuda:0").to(device, non_blocking=True)',
        # D2H via .to("cpu", non_blocking=True) — torch allocates a pinned dest
        'gpu_t.to("cpu", non_blocking=True)',
        # Escape hatch: gpu-sync-ok pragma
        "torch.from_numpy(x).to(device, non_blocking=True)  # gpu-sync-ok: legacy",
        # .copy_ from a variable — we can't statically prove unpinnedness
        "buf.copy_(pinned_src, non_blocking=True)",
        # .to without non_blocking — blocking copy, not this bug class
        "torch.from_numpy(x).to(device)",
    ],
)
def test_good_cases_silent(tmp_path: Path, snippet: str) -> None:
    assert _check(tmp_path, snippet) == 0


def test_pragma_only_mutes_its_own_line(tmp_path: Path) -> None:
    source = """
    torch.from_numpy(a).to(device, non_blocking=True)  # gpu-sync-ok: reason
    torch.from_numpy(b).to(device, non_blocking=True)
    """
    assert _check(tmp_path, source) == 1  # still flags line without the pragma


def test_syntax_error_returns_zero(tmp_path: Path) -> None:
    # Malformed source shouldn't crash the linter or fail the file.
    assert _check(tmp_path, "def broken(:\n") == 0


# --- cross-variable flow: two-line shape ---------------------------------- #


@pytest.mark.parametrize(
    "snippet",
    [
        # Assign then push in the same function
        """
        def f(x):
            t = torch.from_numpy(x)
            return t.to(device, non_blocking=True)
        """,
        # Assign then use as .copy_ source
        """
        def f(buf, x):
            src = torch.from_numpy(x)
            buf.copy_(src, non_blocking=True)
        """,
        # Augmented assign preserves taint
        """
        def f(x):
            t = torch.from_numpy(x)
            t += 1
            return t.to(device, non_blocking=True)
        """,
    ],
)
def test_two_line_bad_flagged(tmp_path: Path, snippet: str) -> None:
    assert _check(tmp_path, snippet) == 1


@pytest.mark.parametrize(
    "snippet",
    [
        # Reassignment to a safe value clears the taint
        """
        def f(x):
            t = torch.from_numpy(x)
            t = safe_helper(t)
            return t.to(device, non_blocking=True)
        """,
        # Taint doesn't cross function boundaries
        """
        def f(x):
            t = torch.from_numpy(x)
            def inner():
                return t.to(device, non_blocking=True)
            return inner
        """,
        # Tuple unpack from an unknown call — don't taint anything
        """
        def f(x):
            a, b = something(x)
            return a.to(device, non_blocking=True)
        """,
    ],
)
def test_two_line_good_silent(tmp_path: Path, snippet: str) -> None:
    assert _check(tmp_path, snippet) == 0
