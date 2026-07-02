# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import getpass
import os

import vllm.env_override as env_override


def _enable_torchinductor_cache_patch(monkeypatch):
    monkeypatch.setattr(
        env_override,
        "_torch_needs_torchinductor_cache_dir_patch",
        lambda: True,
    )


def test_torchinductor_default_cache_dir_patch_is_version_guarded(monkeypatch):
    from torch._inductor.runtime import cache_dir_utils

    original = lambda: "original"
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", original)
    monkeypatch.setattr(
        env_override,
        "_torch_needs_torchinductor_cache_dir_patch",
        lambda: False,
    )

    env_override._patch_torchinductor_default_cache_dir()

    assert cache_dir_utils.default_cache_dir is original


def test_torchinductor_cache_dir_patch_applies_through_torch_2_12(monkeypatch):
    monkeypatch.setattr(env_override.torch, "__version__", "2.12.1+cu128")

    assert env_override._torch_needs_torchinductor_cache_dir_patch()


def test_torchinductor_cache_dir_patch_skips_torch_2_13(monkeypatch):
    monkeypatch.setattr(env_override.torch, "__version__", "2.13.0")

    assert not env_override._torch_needs_torchinductor_cache_dir_patch()


def test_torchinductor_default_cache_dir_falls_back_for_unmapped_uid(
    monkeypatch, tmp_path
):
    from torch._inductor.runtime import cache_dir_utils, runtime_utils

    _enable_torchinductor_cache_patch(monkeypatch)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", lambda: "old")
    monkeypatch.setattr(
        runtime_utils,
        "default_cache_dir",
        cache_dir_utils.default_cache_dir,
        raising=False,
    )
    monkeypatch.setattr(env_override.os, "getuid", lambda: 12345, raising=False)
    monkeypatch.setattr(env_override.tempfile, "gettempdir", lambda: str(tmp_path))

    def raise_key_error():
        raise KeyError("uid not found")

    monkeypatch.setattr(getpass, "getuser", raise_key_error)
    env_override._patch_torchinductor_default_cache_dir()

    expected = str(tmp_path / "torchinductor_uid_12345")
    assert cache_dir_utils.default_cache_dir() == expected
    assert runtime_utils.default_cache_dir() == expected


def test_torchinductor_cache_dir_left_untouched_when_username_resolves(monkeypatch):
    from torch._inductor.runtime import cache_dir_utils

    _enable_torchinductor_cache_patch(monkeypatch)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    original = lambda: "original"
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", original)
    monkeypatch.setattr(getpass, "getuser", lambda: "alice")

    env_override._patch_torchinductor_default_cache_dir()

    # A resolvable username means torch's native behavior is left untouched:
    # neither default_cache_dir nor the cache-dir env var is changed.
    assert cache_dir_utils.default_cache_dir is original
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ


def test_torchinductor_cache_dir_env_var_is_seeded_for_unmapped_uid(
    monkeypatch, tmp_path
):
    from torch._inductor.runtime import cache_dir_utils, runtime_utils

    _enable_torchinductor_cache_patch(monkeypatch)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", lambda: "old")
    monkeypatch.setattr(
        runtime_utils, "default_cache_dir", lambda: "old", raising=False
    )
    monkeypatch.setattr(env_override.os, "getuid", lambda: 4242, raising=False)
    monkeypatch.setattr(env_override.tempfile, "gettempdir", lambda: str(tmp_path))

    def raise_key_error():
        raise KeyError("uid not found")

    monkeypatch.setattr(getpass, "getuser", raise_key_error)
    env_override._patch_torchinductor_default_cache_dir()

    # Seeding the env var is what keeps module-level callers that run at import
    # time alive (e.g. torch._dynamo.package's top-level ``DynamoCache``), since
    # ``cache_dir()`` returns the env var before ever calling default_cache_dir.
    expected = str(tmp_path / "torchinductor_uid_4242")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == expected
