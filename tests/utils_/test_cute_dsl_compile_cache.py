# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

import vllm.utils.cute_dsl_compile_cache as compile_cache

# The real-DSL checks of this cache (a second process loads compiled modules
# from CUTE_DSL_CACHE_DIR but still traces) run in test_cute_dsl_aot_cache.py.


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_CUTE_DSL_COMPILE_CACHE", raising=False)
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)
    assert not compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert "CUTE_DSL_CACHE_DIR" not in os.environ


def test_missing_cute_dsl_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_CUTE_DSL_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_CACHE_DIR", raising=False)
    # A None entry makes the import fail, as when CuTe DSL is not installed.
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.dsl", None)
    assert not compile_cache.maybe_enable_cute_dsl_compile_cache()
    assert "CUTE_DSL_CACHE_DIR" not in os.environ


def test_cache_directory_that_cannot_exist_is_rejected(tmp_path):
    """CuTe DSL would fail every compile creating it, so the cache stays off."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    assert not compile_cache._ensure_dir(str(blocker / "cute_dsl"))
    assert compile_cache._ensure_dir(str(tmp_path / "cute_dsl"))
    os.chmod(tmp_path, 0o555)
    try:
        # An existing read-only directory still serves hits.
        assert compile_cache._ensure_dir(str(tmp_path / "cute_dsl"))
    finally:
        os.chmod(tmp_path, 0o755)
