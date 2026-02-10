#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the envs_impl module."""

import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def test_basic_access():
    import vllm.envs as envs

    print("Testing basic access...")

    print(
        f"VLLM_HOST_IP: {envs.VLLM_HOST_IP!r} "
        f"(type: {type(envs.VLLM_HOST_IP).__name__})"
    )
    assert isinstance(envs.VLLM_HOST_IP, str)
    assert envs.VLLM_HOST_IP == ""

    print(f"LOCAL_RANK: {envs.LOCAL_RANK!r} (type: {type(envs.LOCAL_RANK).__name__})")
    assert isinstance(envs.LOCAL_RANK, int)
    assert envs.LOCAL_RANK == 0

    print(
        f"VLLM_USE_MODELSCOPE: {envs.VLLM_USE_MODELSCOPE!r} "
        f"(type: {type(envs.VLLM_USE_MODELSCOPE).__name__})"
    )
    assert isinstance(envs.VLLM_USE_MODELSCOPE, bool)
    assert not envs.VLLM_USE_MODELSCOPE

    print(f"VLLM_PORT: {envs.VLLM_PORT!r} (type: {type(envs.VLLM_PORT).__name__})")
    assert envs.VLLM_PORT is None

    print("✓ Basic access tests passed!\n")


def test_env_var_parsing():
    import importlib

    print("Testing environment variable parsing...")

    os.environ["VLLM_HOST_IP"] = "192.168.1.1"
    os.environ["LOCAL_RANK"] = "5"
    os.environ["VLLM_USE_MODELSCOPE"] = "1"
    os.environ["VLLM_PORT"] = "8000"

    import vllm.envs as envs

    if hasattr(envs, "__wrapped__"):
        importlib.reload(envs)

    if "vllm.envs" in sys.modules:
        del sys.modules["vllm.envs"]
    import vllm.envs as envs

    host_ip = envs.VLLM_HOST_IP
    print(f"VLLM_HOST_IP: {host_ip!r} (type: {type(host_ip).__name__})")
    assert isinstance(host_ip, str)
    assert host_ip == "192.168.1.1"

    local_rank = envs.LOCAL_RANK
    print(f"LOCAL_RANK: {local_rank!r} (type: {type(local_rank).__name__})")
    assert isinstance(local_rank, int)
    assert local_rank == 5

    use_modelscope = envs.VLLM_USE_MODELSCOPE
    print(
        f"VLLM_USE_MODELSCOPE: {use_modelscope!r} "
        f"(type: {type(use_modelscope).__name__})"
    )
    assert isinstance(use_modelscope, bool)
    assert use_modelscope

    port = envs.VLLM_PORT
    print(f"VLLM_PORT: {port!r} (type: {type(port).__name__})")
    assert isinstance(port, int)
    assert port == 8000

    del os.environ["VLLM_HOST_IP"]
    del os.environ["LOCAL_RANK"]
    del os.environ["VLLM_USE_MODELSCOPE"]
    del os.environ["VLLM_PORT"]

    print("✓ Environment variable parsing tests passed!\n")


def test_lazy_defaults():
    print("Testing lazy defaults...")

    if "VLLM_CACHE_ROOT" in os.environ:
        del os.environ["VLLM_CACHE_ROOT"]

    if "vllm.envs_impl" in sys.modules:
        del sys.modules["vllm.envs_impl"]
    if "vllm.envs_impl._variables" in sys.modules:
        del sys.modules["vllm.envs_impl._variables"]
    if "vllm.envs_impl.utils" in sys.modules:
        del sys.modules["vllm.envs_impl.utils"]

    import vllm.envs as envs

    cache_root = envs.VLLM_CACHE_ROOT
    print(f"VLLM_CACHE_ROOT: {cache_root!r} (type: {type(cache_root).__name__})")
    assert isinstance(cache_root, str)
    assert "vllm" in cache_root

    print("✓ Lazy default tests passed!\n")


def test_is_set():
    print("Testing is_set() function...")

    if "vllm.envs" in sys.modules:
        del sys.modules["vllm.envs"]
    import vllm.envs as envs

    assert not envs.is_set("VLLM_TEST_VAR_123")
    print(f"is_set('VLLM_TEST_VAR_123'): {envs.is_set('VLLM_TEST_VAR_123')}")

    os.environ["VLLM_TEST_VAR_123"] = "test"
    assert envs.is_set("VLLM_TEST_VAR_123")
    print(
        f"After setting: is_set('VLLM_TEST_VAR_123'): "
        f"{envs.is_set('VLLM_TEST_VAR_123')}"
    )

    del os.environ["VLLM_TEST_VAR_123"]

    print("✓ is_set() tests passed!\n")
