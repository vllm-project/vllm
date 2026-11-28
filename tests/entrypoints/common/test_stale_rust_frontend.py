# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for detecting a `vllm-rs` binary built without the checkout's `rust/`."""

import shutil
import subprocess
from pathlib import Path

import pytest

import vllm
from vllm.entrypoints.common import stale_rust_frontend
from vllm.entrypoints.common.stale_rust_frontend import (
    find_unbuilt_rust_changes,
    warn_if_rust_frontend_stale,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(repo: Path, path: str, content: str) -> str:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content)
    _git(repo, "add", path)
    _git(repo, "commit", "-q", "-m", path)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    repo = tmp_path / "vllm-checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _commit(repo, "vllm/__init__.py", "")
    _commit(repo, "rust/src/lib.rs", "// v1\n")
    return repo


def _binary(package_dir: Path, version: str) -> str:
    path = package_dir / "vllm-rs"
    path.write_text(f"#!/bin/sh\necho 'vllm-rs {version}'\n")
    path.chmod(0o755)
    return str(path)


def _binary_printing(package_dir: Path, output: bytes) -> str:
    path = package_dir / "vllm-rs"
    path.write_bytes(b"#!/bin/sh\nprintf '" + output + b"'\n")
    path.chmod(0o755)
    return str(path)


def _version(commit: str, dirty: bool = False) -> str:
    return f"0.1.0.dev3+g{commit[:9]}" + (".d20261123" if dirty else "")


def test_reports_rust_changes_committed_after_the_binary(checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "rust/src/lib.rs", "// v2\n")
    _commit(checkout, "vllm/model.py", "")

    unbuilt = find_unbuilt_rust_changes(
        _binary(checkout / "vllm", _version(built)), str(checkout / "vllm")
    )

    assert unbuilt is not None
    assert unbuilt.binary_commit == built
    assert unbuilt.changed_paths == ["rust/src/lib.rs"]


def test_reports_uncommitted_rust_changes(checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    (checkout / "rust/src/lib.rs").write_text("// edited\n")

    unbuilt = find_unbuilt_rust_changes(
        _binary(checkout / "vllm", _version(built)), str(checkout / "vllm")
    )

    assert unbuilt is not None
    assert unbuilt.changed_paths == ["rust/src/lib.rs"]


def test_reports_untracked_rust_files(checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    # Cargo discovers a new build script without any tracked file changing.
    (checkout / "rust/build.rs").write_text("fn main() {}\n")

    unbuilt = find_unbuilt_rust_changes(
        _binary(checkout / "vllm", _version(built)), str(checkout / "vllm")
    )

    assert unbuilt is not None
    assert unbuilt.changed_paths == ["rust/build.rs"]


def test_reports_a_symlinked_packaged_binary(tmp_path, checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "rust/src/lib.rs", "// v2\n")
    # E.g. linked from another worktree's build.
    (checkout / "vllm/vllm-rs").symlink_to(_binary(tmp_path, _version(built)))

    unbuilt = find_unbuilt_rust_changes(
        str(checkout / "vllm/vllm-rs"), str(checkout / "vllm")
    )

    assert unbuilt is not None
    assert unbuilt.changed_paths == ["rust/src/lib.rs"]


def test_ignores_changes_outside_rust(checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "vllm/model.py", "")

    assert (
        find_unbuilt_rust_changes(
            _binary(checkout / "vllm", _version(built)), str(checkout / "vllm")
        )
        is None
    )


@pytest.mark.parametrize(
    "version",
    [
        pytest.param(None, id="dirty-build"),
        pytest.param("0.1.0", id="release-tag"),
        pytest.param("0.1.0+gdeadbeef12", id="unknown-commit"),
    ],
)
def test_no_evidence_without_a_known_clean_build_commit(checkout, version):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "rust/src/lib.rs", "// v2\n")
    if version is None:
        version = _version(built, dirty=True)

    assert (
        find_unbuilt_rust_changes(
            _binary(checkout / "vllm", version), str(checkout / "vllm")
        )
        is None
    )


def test_no_evidence_from_undecodable_version_output(checkout):
    _commit(checkout, "rust/src/lib.rs", "// v2\n")

    assert (
        find_unbuilt_rust_changes(
            _binary_printing(checkout / "vllm", rb"vllm-rs \377"),
            str(checkout / "vllm"),
        )
        is None
    )


def test_no_evidence_for_a_package_outside_the_checkout_sources(checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "rust/src/lib.rs", "// v2\n")
    # A non-editable install into a venv that lives inside the checkout.
    site_vllm = checkout / ".venv/lib/site-packages/vllm"
    site_vllm.mkdir(parents=True)

    assert (
        find_unbuilt_rust_changes(_binary(site_vllm, _version(built)), str(site_vllm))
        is None
    )


def test_no_evidence_for_a_binary_outside_the_package(tmp_path, checkout):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "rust/src/lib.rs", "// v2\n")

    assert (
        find_unbuilt_rust_changes(
            _binary(tmp_path, _version(built)), str(checkout / "vllm")
        )
        is None
    )


def test_warns_about_a_stale_binary(checkout, monkeypatch):
    built = _git(checkout, "rev-parse", "HEAD")
    _commit(checkout, "rust/src/lib.rs", "// v2\n")
    monkeypatch.setattr(vllm, "__file__", str(checkout / "vllm/__init__.py"))
    warnings: list[str] = []
    monkeypatch.setattr(
        stale_rust_frontend.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )

    warn_if_rust_frontend_stale(_binary(checkout / "vllm", _version(built)))

    assert len(warnings) == 1
    assert "differs from that commit in 1 file (rust/src/lib.rs)" in warnings[0]
    assert "`./tools/build_rust.sh`" in warnings[0]
