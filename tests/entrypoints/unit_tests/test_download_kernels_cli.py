# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess

import pytest

from vllm.entrypoints.cli import download_kernels


def test_replaces_kernels_from_another_flashinfer_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """After a vLLM upgrade the installed kernels no longer match FlashInfer,
    which then fails to import, so its CLI must run with the check bypassed."""
    calls = []

    def run(cmd, env, check):
        calls.append((cmd, env))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(download_kernels, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "run", run)

    assert download_kernels._download_flashinfer_kernels(dry_run=False) == 0
    ((cmd, env),) = calls
    assert cmd[1:] == ["-m", "flashinfer", "download-kernels"]
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"


def test_skips_without_flashinfer(monkeypatch: pytest.MonkeyPatch):
    """Builds without FlashInfer (CPU, ROCm, XPU) have nothing to download."""
    monkeypatch.setattr(download_kernels, "find_spec", lambda name: None)
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert download_kernels._download_flashinfer_kernels(dry_run=False) == 0
