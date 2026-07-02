# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the _ensure_getuser_safe() patch in env_override.py.

Validates that _ensure_getuser_safe() correctly patches getpass.getuser() to
return a safe UID-derived string in containers where the running UID has no
/etc/passwd entry (e.g. OpenShift arbitrary-UID pods), without affecting normal
environments where getpass.getuser() already works.
"""

import getpass
import os

import pytest

from vllm.env_override import _ensure_getuser_safe


@pytest.fixture(autouse=True)
def _restore_getuser():
    """Restore the original getpass.getuser after every test."""
    original = getpass.getuser
    yield
    getpass.getuser = original


def _raise(exc):
    """Return a zero-argument callable that raises *exc* when called."""

    def _raiser():
        raise exc

    return _raiser


class TestNormalEnvironment:
    """_ensure_getuser_safe() must be a no-op when getuser() works."""

    def test_getuser_not_replaced(self, monkeypatch):
        original = getpass.getuser
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        _ensure_getuser_safe()
        assert getpass.getuser is original

    def test_torchinductor_cache_dir_not_set(self, monkeypatch):
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        _ensure_getuser_safe()
        assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


class TestBrokenEnvironment:
    """_ensure_getuser_safe() must patch when getuser() raises."""

    @pytest.mark.parametrize(
        "exc",
        [
            KeyError("getpwuid(): uid not found: 1000770000"),
            ModuleNotFoundError("No module named 'pwd'"),
            OSError("getpwuid(): uid not found: 1000770000"),
        ],
    )
    def test_getuser_patched_for_each_exception_type(self, monkeypatch, exc):
        monkeypatch.setattr(getpass, "getuser", _raise(exc))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

        _ensure_getuser_safe()

        result = getpass.getuser()
        assert result == f"uid{os.getuid()}"

    def test_patched_getuser_returns_stable_string(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

        _ensure_getuser_safe()

        # Must return the same value on every call (closed-over constant).
        results = {getpass.getuser() for _ in range(10)}
        assert len(results) == 1

    def test_torchinductor_cache_dir_set_from_tmp(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        _ensure_getuser_safe()

        val = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        assert val == f"/tmp/torchinductor_uid{os.getuid()}"

    def test_torchinductor_cache_dir_set_from_xdg_cache_home(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/xdg/cache")

        _ensure_getuser_safe()

        val = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        assert val == f"/xdg/cache/torchinductor_uid{os.getuid()}"

    def test_existing_torchinductor_cache_dir_not_overwritten(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/my/custom/cache")

        _ensure_getuser_safe()

        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/my/custom/cache"


class TestIdempotency:
    """Calling _ensure_getuser_safe() multiple times must be safe."""

    def test_double_call_does_not_rewrap(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

        _ensure_getuser_safe()
        patched = getpass.getuser
        _ensure_getuser_safe()

        # Second call must not wrap the already-safe replacement again.
        assert getpass.getuser is patched


class TestVersionGate:
    """Patch must be skipped on torch >= 2.13 (pytorch#184208 is present)."""

    def test_skipped_on_torch_2_13(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        # Simulate torch >= 2.13 by making is_torch_equal_or_newer return True.
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: True
        )
        original = getpass.getuser

        _ensure_getuser_safe()

        # Nothing should have been patched.
        assert getpass.getuser is original

    def test_applied_below_torch_2_13(self, monkeypatch):
        monkeypatch.setattr(getpass, "getuser", _raise(KeyError("uid not found")))
        monkeypatch.setattr(
            "vllm.utils.torch_utils.is_torch_equal_or_newer", lambda _: False
        )
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

        _ensure_getuser_safe()

        # Patch must have been applied.
        assert getpass.getuser() == f"uid{os.getuid()}"
