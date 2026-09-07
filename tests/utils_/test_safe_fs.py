# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm.utils.safe_fs import (
    _current_uid,
    _verify_dir_components,
    get_user_root_dir,
    prepare_private_dir,
    safe_open_file,
)


def _make_symlink(link: Path, target: Path) -> None:
    """Create a symlink, skipping the test if the platform forbids it."""
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation not permitted on this platform")


def test_get_user_root_dir_under_home():
    """Verify the user root directory is the vllm folder under home."""
    assert get_user_root_dir() == os.path.join(Path.home(), "vllm")


def test_current_uid_matches_os_when_available():
    """Verify _current_uid mirrors os.getuid() when it exists."""
    expected = os.getuid() if hasattr(os, "getuid") else -1
    assert _current_uid() == expected


def test_current_uid_uses_getuid_result():
    """Verify _current_uid returns the value from a (patched) os.getuid."""
    with patch.object(os, "getuid", return_value=42, create=True):
        assert _current_uid() == 42


def test_verify_dir_components_allows_missing_dirs(tmp_path):
    """Verify paths not created yet pass the check."""
    _verify_dir_components(str(tmp_path / "not" / "created"))


def test_verify_dir_components_allows_plain_dirs(tmp_path):
    """Verify a plain directory tree under the temp root passes."""
    (tmp_path / "sub").mkdir()
    _verify_dir_components(str(tmp_path / "sub"))


def test_verify_dir_components_rejects_symlink_below_temp_root(tmp_path):
    """Verify a symlinked component below the temp root is refused."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    _make_symlink(link, real)
    with pytest.raises(OSError, match="Refusing symlinked component"):
        _verify_dir_components(str(link))


def test_verify_dir_components_rejects_symlink_outside_temp_root(tmp_path):
    """Verify a symlink at the directory itself is refused outside the
    temp root."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    _make_symlink(link, real)
    temp_root = tmp_path / "temp_root"
    with patch.object(tempfile, "gettempdir", return_value=str(temp_root)):
        with pytest.raises(OSError, match="Refusing symlinked component"):
            _verify_dir_components(str(link))


def test_verify_dir_components_trusts_symlinked_temp_root(tmp_path):
    """Verify the temp root itself may be a symlink (e.g. /tmp on macOS)."""
    real = tmp_path / "real"
    real.mkdir()
    fake_root = tmp_path / "fake_root"
    _make_symlink(fake_root, real)
    with patch.object(tempfile, "gettempdir", return_value=str(fake_root)):
        # Components strictly below the temp root are checked; the root
        # itself is trusted.
        _verify_dir_components(str(fake_root / "sub"))


def test_prepare_private_dir_creates_dir(tmp_path):
    """Verify a missing directory is created with mode 0700."""
    d = tmp_path / "nested" / "dir"
    prepare_private_dir(str(d))
    assert d.is_dir()
    if os.name != "nt":
        assert d.stat().st_mode & 0o777 == 0o700


def test_prepare_private_dir_is_idempotent(tmp_path):
    """Verify preparing an existing directory again is safe."""
    d = tmp_path / "dir"
    d.mkdir()
    prepare_private_dir(str(d))
    prepare_private_dir(str(d))
    assert d.is_dir()


def test_prepare_private_dir_owner_is_current_user(tmp_path):
    """Verify the directory ends up owned by the current user."""
    if os.name == "nt":
        pytest.skip("ownership is not enforced on Windows")
    d = tmp_path / "owned"
    prepare_private_dir(str(d))
    assert d.stat().st_uid == os.getuid()


def test_prepare_private_dir_rejects_symlink(tmp_path):
    """Verify a symlinked directory is refused."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    _make_symlink(link, real)
    with pytest.raises(OSError):
        prepare_private_dir(str(link))


def test_prepare_private_dir_rejects_file(tmp_path):
    """Verify a path pointing to a regular file is refused."""
    f = tmp_path / "file"
    f.write_text("data")
    with pytest.raises(OSError):
        prepare_private_dir(str(f))
    assert f.read_text() == "data"


def test_safe_open_file_creates_and_appends(tmp_path):
    """Verify the file is created with mode 0600 and writes append."""
    f = tmp_path / "dump.log"
    with safe_open_file(str(f), "a") as fh:
        fh.write("first\n")
    with safe_open_file(str(f), "a") as fh:
        fh.write("second\n")
    assert f.read_text() == "first\nsecond\n"
    if os.name != "nt":
        assert f.stat().st_mode & 0o777 == 0o600


def test_safe_open_file_refuses_symlink(tmp_path):
    """Verify a symlinked file is refused instead of followed."""
    real = tmp_path / "real.log"
    real.write_text("sensitive")
    link = tmp_path / "link.log"
    _make_symlink(link, real)
    with pytest.raises(OSError):
        safe_open_file(str(link), "a")
    assert real.read_text() == "sensitive"
