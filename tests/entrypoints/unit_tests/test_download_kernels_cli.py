# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.metadata
import os
import subprocess
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re

import vllm.utils.flashinfer as fi
from vllm.entrypoints.cli import download_kernels


@pytest.mark.skipif(find_spec("flashinfer") is None, reason="requires FlashInfer")
def test_download_kernels_resolves_kernel_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    caplog_vllm: pytest.LogCaptureFixture,
    disable_log_dedup,
):
    """Runs the installed FlashInfer CLI, so a FlashInfer release that renames
    its kernel packages, CLI or version check fails here instead of leaving
    users with a warning that its own fix cannot clear."""
    # A vLLM upgrade leaves kernels from the previous FlashInfer behind, which
    # FlashInfer rejects on import. The command must work in that state.
    stale_cubin = tmp_path / "flashinfer_cubin"
    stale_cubin.mkdir()
    (stale_cubin / "__init__.py").write_text(
        f'__version__ = "0"\ndef get_cubin_dir(): return "{stale_cubin}"\n'
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(tmp_path), os.getenv("PYTHONPATH", "")])
    )

    assert download_kernels._download_flashinfer_kernels(dry_run=True) == 0
    kernels = dict(re.findall(r"Requirement: ([\w-]+)==(\S+)", capfd.readouterr().out))
    assert kernels

    installed = {"flashinfer-python": importlib.metadata.version("flashinfer-python")}

    def version(name: str) -> str:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    def warns() -> bool:
        caplog_vllm.clear()
        fi.warn_if_flashinfer_kernels_missing()
        return "Run `vllm download-kernels`" in caplog_vllm.text

    monkeypatch.delenv("VLLM_HAS_FLASHINFER_CUBIN", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(fi, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        fi,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, has_device_capability=lambda _: True),
    )

    assert warns()
    installed.update(dict.fromkeys(kernels, "0"))
    assert warns()
    installed.update(kernels)
    assert not warns()


def test_cuda_without_jit_cache_wheels_needs_only_cubin(
    monkeypatch: pytest.MonkeyPatch,
    caplog_vllm: pytest.LogCaptureFixture,
    disable_log_dedup,
):
    """FlashInfer publishes no flashinfer-jit-cache below CUDA 12.9, so the
    command must not fail on it and the warning must not ask for it."""
    commands = []
    installed = {"flashinfer-python": "1.0", "flashinfer-cubin": "1.0"}

    def version(name: str) -> str:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    def run(cmd: list[str], **kwargs) -> SimpleNamespace:
        commands.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv("VLLM_HAS_FLASHINFER_CUBIN", raising=False)
    monkeypatch.setattr(fi.torch.version, "cuda", "12.8")
    monkeypatch.setattr(download_kernels, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(fi, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        fi,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, has_device_capability=lambda _: True),
    )

    assert download_kernels._download_flashinfer_kernels(dry_run=False) == 0
    assert commands[0][-1] == "install-cubin-wheel"
    fi.warn_if_flashinfer_kernels_missing()
    assert "vllm download-kernels" not in caplog_vllm.text


def test_download_kernels_skips_without_flashinfer(monkeypatch: pytest.MonkeyPatch):
    """Builds without FlashInfer (CPU, ROCm, XPU) have nothing to download."""
    monkeypatch.setattr(download_kernels, "find_spec", lambda name: None)
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert download_kernels._download_flashinfer_kernels(dry_run=False) == 0
