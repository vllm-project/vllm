# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import getpass

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


def test_torchinductor_cache_dir_patch_applies_through_torch_2_13(monkeypatch):
    monkeypatch.setattr(env_override.torch, "__version__", "2.13.1+cu128")

    assert env_override._torch_needs_torchinductor_cache_dir_patch()


def test_torchinductor_cache_dir_patch_skips_torch_2_14_dev(monkeypatch):
    monkeypatch.setattr(env_override.torch, "__version__", "2.14.0.dev20260601")

    assert not env_override._torch_needs_torchinductor_cache_dir_patch()


def test_torchinductor_default_cache_dir_falls_back_for_unmapped_uid(
    monkeypatch, tmp_path
):
    from torch._inductor.runtime import cache_dir_utils, runtime_utils

    _enable_torchinductor_cache_patch(monkeypatch)
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", lambda: "old")
    monkeypatch.setattr(
        runtime_utils,
        "default_cache_dir",
        cache_dir_utils.default_cache_dir,
        raising=False,
    )
    monkeypatch.setattr(env_override.os, "getuid", lambda: 12345, raising=False)
    monkeypatch.setattr(env_override.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cache_dir_utils, "is_fbcode", lambda: False, raising=False)

    def raise_key_error():
        raise KeyError("uid not found")

    monkeypatch.setattr(getpass, "getuser", raise_key_error)
    env_override._patch_torchinductor_default_cache_dir()

    expected = str(tmp_path / "torchinductor_uid_12345")
    assert cache_dir_utils.default_cache_dir() == expected
    assert runtime_utils.default_cache_dir() == expected


def test_torchinductor_default_cache_dir_sanitizes_username(monkeypatch, tmp_path):
    from torch._inductor.runtime import cache_dir_utils, runtime_utils

    _enable_torchinductor_cache_patch(monkeypatch)
    monkeypatch.setattr(cache_dir_utils, "default_cache_dir", lambda: "old")
    monkeypatch.setattr(
        runtime_utils,
        "default_cache_dir",
        cache_dir_utils.default_cache_dir,
        raising=False,
    )
    monkeypatch.setattr(env_override.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cache_dir_utils, "is_fbcode", lambda: False, raising=False)

    monkeypatch.setattr(getpass, "getuser", lambda: 'bad/user:name*?"<>|')
    env_override._patch_torchinductor_default_cache_dir()

    expected = str(tmp_path / "torchinductor_bad_user_name______")
    assert cache_dir_utils.default_cache_dir() == expected
    assert runtime_utils.default_cache_dir() == expected
