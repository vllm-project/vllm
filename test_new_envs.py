#!/usr/bin/env python3
"""Simple test script to verify the new envs module works.

This script should be run from the repository root:
    python test_new_envs.py

Or install vllm in editable mode first:
    pip install -e .
    python test_new_envs.py
"""

import os
import sys
from pathlib import Path

# Add vllm to path (relative to this script's location)
# This makes the test work when run from the repo root
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def test_basic_access():
    """Test basic environment variable access."""
    import vllm.envs as envs

    print("Testing basic access...")

    # Test string default
    print(f"VLLM_HOST_IP: {envs.VLLM_HOST_IP!r} (type: {type(envs.VLLM_HOST_IP).__name__})")
    assert isinstance(envs.VLLM_HOST_IP, str)
    assert envs.VLLM_HOST_IP == ""

    # Test int default
    print(f"LOCAL_RANK: {envs.LOCAL_RANK!r} (type: {type(envs.LOCAL_RANK).__name__})")
    assert isinstance(envs.LOCAL_RANK, int)
    assert envs.LOCAL_RANK == 0

    # Test bool default
    print(f"VLLM_USE_MODELSCOPE: {envs.VLLM_USE_MODELSCOPE!r} (type: {type(envs.VLLM_USE_MODELSCOPE).__name__})")
    assert isinstance(envs.VLLM_USE_MODELSCOPE, bool)
    assert envs.VLLM_USE_MODELSCOPE == False

    # Test Optional[int] default
    print(f"VLLM_PORT: {envs.VLLM_PORT!r} (type: {type(envs.VLLM_PORT).__name__})")
    assert envs.VLLM_PORT is None

    print("✓ Basic access tests passed!\n")


def test_env_var_parsing():
    """Test parsing from environment variables."""
    import importlib

    print("Testing environment variable parsing...")

    # Set some environment variables
    os.environ["VLLM_HOST_IP"] = "192.168.1.1"
    os.environ["LOCAL_RANK"] = "5"
    os.environ["VLLM_USE_MODELSCOPE"] = "1"
    os.environ["VLLM_PORT"] = "8000"

    # Reload the module to pick up env vars
    import vllm.envs as envs
    if hasattr(envs, "__wrapped__"):
        importlib.reload(envs)

    # Force re-import by deleting from sys.modules
    if "vllm.envs" in sys.modules:
        del sys.modules["vllm.envs"]
    import vllm.envs as envs

    # Test string parsing
    host_ip = envs.VLLM_HOST_IP
    print(f"VLLM_HOST_IP: {host_ip!r} (type: {type(host_ip).__name__})")
    assert isinstance(host_ip, str)
    assert host_ip == "192.168.1.1"

    # Test int parsing
    local_rank = envs.LOCAL_RANK
    print(f"LOCAL_RANK: {local_rank!r} (type: {type(local_rank).__name__})")
    assert isinstance(local_rank, int)
    assert local_rank == 5

    # Test bool parsing
    use_modelscope = envs.VLLM_USE_MODELSCOPE
    print(f"VLLM_USE_MODELSCOPE: {use_modelscope!r} (type: {type(use_modelscope).__name__})")
    assert isinstance(use_modelscope, bool)
    assert use_modelscope == True

    # Test Optional[int] parsing with custom function
    port = envs.VLLM_PORT
    print(f"VLLM_PORT: {port!r} (type: {type(port).__name__})")
    assert isinstance(port, int)
    assert port == 8000

    # Clean up
    del os.environ["VLLM_HOST_IP"]
    del os.environ["LOCAL_RANK"]
    del os.environ["VLLM_USE_MODELSCOPE"]
    del os.environ["VLLM_PORT"]

    print("✓ Environment variable parsing tests passed!\n")


def test_lazy_defaults():
    """Test lazy-initialized defaults."""
    print("Testing lazy defaults...")

    # Clear any existing env var
    if "VLLM_CACHE_ROOT" in os.environ:
        del os.environ["VLLM_CACHE_ROOT"]

    # Reload
    if "vllm.envs" in sys.modules:
        del sys.modules["vllm.envs"]
    if "vllm.envs._variables" in sys.modules:
        del sys.modules["vllm.envs._variables"]
    if "vllm.envs.utils" in sys.modules:
        del sys.modules["vllm.envs.utils"]

    import vllm.envs as envs

    cache_root = envs.VLLM_CACHE_ROOT
    print(f"VLLM_CACHE_ROOT: {cache_root!r} (type: {type(cache_root).__name__})")
    assert isinstance(cache_root, str)
    assert "vllm" in cache_root

    print("✓ Lazy default tests passed!\n")


def test_is_set():
    """Test the is_set() function."""
    print("Testing is_set() function...")

    # Reload
    if "vllm.envs" in sys.modules:
        del sys.modules["vllm.envs"]
    import vllm.envs as envs

    # Should not be set initially
    assert not envs.is_set("VLLM_TEST_VAR_123")
    print(f"is_set('VLLM_TEST_VAR_123'): {envs.is_set('VLLM_TEST_VAR_123')}")

    # Set it
    os.environ["VLLM_TEST_VAR_123"] = "test"
    assert envs.is_set("VLLM_TEST_VAR_123")
    print(f"After setting: is_set('VLLM_TEST_VAR_123'): {envs.is_set('VLLM_TEST_VAR_123')}")

    # Clean up
    del os.environ["VLLM_TEST_VAR_123"]

    print("✓ is_set() tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing new vllm.envs module")
    print("=" * 60 + "\n")

    try:
        test_basic_access()
        test_env_var_parsing()
        test_lazy_defaults()
        test_is_set()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
