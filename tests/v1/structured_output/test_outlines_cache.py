# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from importlib import util
from pathlib import Path

import pytest
from cachetools import LRUCache

import vllm.envs as envs

UTILS_PATH = (
    Path(__file__).parents[3] / "vllm" / "v1" / "structured_output" / "utils.py"
)

utils_spec = util.spec_from_file_location("outlines_cache_utils", UTILS_PATH)
assert utils_spec is not None
assert utils_spec.loader is not None
outlines_utils = util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(outlines_utils)

pytestmark = pytest.mark.cpu_test


def _clear_env_cache() -> None:
    if hasattr(envs.__getattr__, "cache_clear"):
        envs.__getattr__.cache_clear()


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("true", False),
    ],
)
def test_vllm_v1_use_outlines_cache_env_var(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: bool,
) -> None:
    if env_value is None:
        monkeypatch.delenv("VLLM_V1_USE_OUTLINES_CACHE", raising=False)
    else:
        monkeypatch.setenv("VLLM_V1_USE_OUTLINES_CACHE", env_value)
    _clear_env_cache()

    assert envs.VLLM_V1_USE_OUTLINES_CACHE is expected


def test_outlines_cache_disabled_does_not_import_diskcache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_V1_USE_OUTLINES_CACHE", "0")
    monkeypatch.setitem(sys.modules, "diskcache", None)
    _clear_env_cache()

    cache = outlines_utils.get_outlines_cache()

    assert isinstance(cache, LRUCache)


def test_outlines_cache_enabled_requires_diskcache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("VLLM_V1_USE_OUTLINES_CACHE", "1")
    monkeypatch.setenv("OUTLINES_CACHE_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "diskcache", None)
    _clear_env_cache()

    with pytest.raises(ImportError, match="outlines-cache"):
        outlines_utils.get_outlines_cache()


def test_outlines_cache_enabled_uses_diskcache_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeCache(dict):
        def __init__(
            self,
            directory: str,
            *,
            eviction_policy: str,
            cull_limit: int,
        ) -> None:
            super().__init__()
            self.directory = directory
            self.eviction_policy = eviction_policy
            self.cull_limit = cull_limit

        def set(self, key: str, value: object) -> None:
            self[key] = value

    fake_diskcache = types.ModuleType("diskcache")
    fake_diskcache.__dict__["Cache"] = FakeCache
    monkeypatch.setitem(sys.modules, "diskcache", fake_diskcache)
    monkeypatch.setattr(
        outlines_utils.importlib.metadata,
        "version",
        lambda package_name: f"{package_name}-test",
    )
    monkeypatch.setenv("VLLM_V1_USE_OUTLINES_CACHE", "1")
    monkeypatch.setenv("OUTLINES_CACHE_DIR", str(tmp_path))
    _clear_env_cache()

    cache = outlines_utils.get_outlines_cache()

    assert isinstance(cache, FakeCache)
    assert cache.directory == str(tmp_path)
    assert cache.eviction_policy == "none"
    assert cache.cull_limit == 0
    assert cache["__version__"] == "outlines_core-test"
