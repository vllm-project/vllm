# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the autouse compile-cache isolation fixtures in tests/conftest.py.

Some tests below are order-dependent on purpose: they check that state left
behind by the previous test does not reach the next one.
"""

import os
import subprocess
import sys

import pytest

from vllm import envs

pytestmark = pytest.mark.skip_global_cleanup

_seen_cache_roots: list[str] = []


@pytest.mark.parametrize("run", range(2))
def test_compile_cache_root_is_fresh_per_test(run, isolated_compile_cache):
    cache_root = isolated_compile_cache
    assert os.environ["VLLM_COMPILE_CACHE_ROOT"] == cache_root
    assert cache_root == envs.VLLM_COMPILE_CACHE_ROOT
    assert os.listdir(cache_root) == []
    assert cache_root not in _seen_cache_roots
    for previous in _seen_cache_roots:
        assert not os.path.exists(previous)
    _seen_cache_roots.append(cache_root)
    os.makedirs(os.path.join(cache_root, "some_hash"))


def test_subprocess_inherits_compile_cache_root(isolated_compile_cache):
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ['VLLM_COMPILE_CACHE_ROOT'])",
        ],
        text=True,
    )
    assert out.strip() == isolated_compile_cache


def test_other_caches_stay_under_vllm_cache_root(isolated_compile_cache):
    assert not envs.VLLM_CACHE_ROOT.startswith(isolated_compile_cache)
    assert not envs.VLLM_COMPILE_CACHE_ROOT.startswith(envs.VLLM_CACHE_ROOT)


def test_leak_env_from_previous_test(isolated_compile_cache):
    # What compilation does via os.environ (decorators.py, compiler_interface.py)
    # and what an in-process EngineCore does via enable_envs_cache().
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/leaked/inductor_cache"
    os.environ["TRITON_CACHE_DIR"] = "/leaked/triton_cache"
    os.environ["DG_JIT_CACHE_DIR"] = "/leaked/deep_gemm"
    envs.enable_envs_cache()


def test_env_is_restored_after_previous_test(isolated_compile_cache):
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "DG_JIT_CACHE_DIR"):
        assert not (os.environ.get(key) or "").startswith("/leaked")
    assert isolated_compile_cache == envs.VLLM_COMPILE_CACHE_ROOT
