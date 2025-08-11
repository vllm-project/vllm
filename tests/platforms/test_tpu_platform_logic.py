# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This file contains CPU-only unit tests for the TPU platform logic.
# It uses mocking to verify the dynamic loading and registration of
# tpu_commons components without requiring actual TPU hardware.

import builtins
import importlib
import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest

# The module we are testing
TPU_PLATFORM_MODULE = "vllm.platforms.tpu"


def reload_tpu_module():
    """
    Reloads the vllm.platforms.tpu module to re-trigger its import-time logic.
    This is essential for testing the dynamic platform selection.
    """
    if TPU_PLATFORM_MODULE in sys.modules:
        importlib.reload(sys.modules[TPU_PLATFORM_MODULE])
    else:
        importlib.import_module(TPU_PLATFORM_MODULE)
    return sys.modules[TPU_PLATFORM_MODULE]


@pytest.fixture(autouse=True)
def clear_registry_and_module_cache():
    """
    Fixture to ensure a clean state for each test. It clears the engine
    registry and unloads the tpu module before each test run.
    """
    from vllm.dependency_injection.registry import _engine_core_proc_registry

    # Unload the tpu module to ensure it's re-imported in each test
    if TPU_PLATFORM_MODULE in sys.modules:
        del sys.modules[TPU_PLATFORM_MODULE]

    _engine_core_proc_registry.clear()

    yield

    # Cleanup after test
    if TPU_PLATFORM_MODULE in sys.modules:
        del sys.modules[TPU_PLATFORM_MODULE]
    _engine_core_proc_registry.clear()


def test_vllm_platform_used_when_tpu_commons_not_installed():
    """
    Verify that the default vLLM TpuPlatform is used when tpu_commons
    is not installed (simulated by raising ImportError).
    """
    # To robustly simulate a missing module, we patch the import mechanism
    original_import = builtins.__import__

    def import_mock(name, *args, **kwargs):
        if name.startswith('tpu_commons'):
            raise ImportError(f"Mock import error for '{name}'")
        return original_import(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=import_mock):
        tpu_platform_module = reload_tpu_module()
        from vllm.dependency_injection.registry import (
            retrieve_engine_core_proc)

        # Check that the platform is the vLLM one
        assert "vllm" in tpu_platform_module.TpuPlatform.__module__

        # Check that the disaggregated engine is NOT registered
        with pytest.raises(ValueError, match="is not registered"):
            retrieve_engine_core_proc("disaggregated_tpu")


@pytest.mark.parametrize("is_supported", [True, False])
def test_tpu_commons_jax_backend_selected(is_supported):
    """
    Verify that the JAX backend from tpu_commons is selected and that the
    disaggregated engine is registered only when it is supported.
    """
    # Create a mock tpu_commons library
    mock_tpu_commons = MagicMock()
    mock_tpu_commons.platforms.tpu_jax.TpuPlatform = MagicMock(
        __name__="MockTpuJaxPlatform")
    mock_disagg_engine = MagicMock(__name__="MockDisaggEngine")
    mock_disagg_engine.is_supported.return_value = is_supported
    mock_tpu_commons.core.core_tpu.DisaggEngineCoreProc = mock_disagg_engine

    # Patch sys.modules to make our mock library importable
    mock_modules = {
        'tpu_commons': mock_tpu_commons,
        'tpu_commons.platforms': mock_tpu_commons.platforms,
        'tpu_commons.platforms.tpu_jax': mock_tpu_commons.platforms.tpu_jax,
        'tpu_commons.core': mock_tpu_commons.core,
        'tpu_commons.core.core_tpu': mock_tpu_commons.core.core_tpu,
    }
    with patch.dict('sys.modules', mock_modules), \
         patch('os.environ.get', return_value="jax"):
        tpu_platform_module = reload_tpu_module()
        from vllm.dependency_injection.registry import (
            retrieve_engine_core_proc)

        # Check that the platform is the mocked one from tpu_commons
        assert tpu_platform_module.TpuPlatform.__name__ == "MockTpuJaxPlatform"

        # The engine is always registered, discovery happens later.
        registered_engine = retrieve_engine_core_proc("disaggregated_tpu")
        assert registered_engine.__name__ == "MockDisaggEngine"


@pytest.mark.parametrize("is_supported", [True, False])
def test_tpu_commons_torchax_backend_selected(is_supported):
    """
    Verify that the torchax backend from tpu_commons is selected and that the
    disaggregated engine is registered only when it is supported.
    """
    # Create a mock tpu_commons library
    mock_tpu_commons = MagicMock()
    mock_tpu_commons.platforms.tpu_torchax.TpuPlatform = MagicMock(
        __name__="MockTpuTorchaxPlatform")
    mock_disagg_engine = MagicMock(__name__="MockDisaggEngine")
    mock_disagg_engine.is_supported.return_value = is_supported
    mock_tpu_commons.core.core_tpu.DisaggEngineCoreProc = mock_disagg_engine

    # Patch sys.modules to make our mock library importable
    mock_modules = {
        'tpu_commons': mock_tpu_commons,
        'tpu_commons.platforms': mock_tpu_commons.platforms,
        'tpu_commons.platforms.tpu_torchax':
        mock_tpu_commons.platforms.tpu_torchax,
        'tpu_commons.core': mock_tpu_commons.core,
        'tpu_commons.core.core_tpu': mock_tpu_commons.core.core_tpu,
    }
    with patch.dict('sys.modules', mock_modules), \
         patch('os.environ.get', return_value="torchax"):
        tpu_platform_module = reload_tpu_module()
        from vllm.dependency_injection.registry import (
            retrieve_engine_core_proc)

        # Check that the platform is the mocked one from tpu_commons
        assert tpu_platform_module.TpuPlatform.__name__ == (
            "MockTpuTorchaxPlatform")

        # The engine is always registered, discovery happens later.
        registered_engine = retrieve_engine_core_proc("disaggregated_tpu")
        assert registered_engine.__name__ == "MockDisaggEngine"


def test_fallback_and_warning_on_backend_load_failure():
    """
    Verify that vLLM falls back to its own platform and warns the user
    if tpu_commons is installed but the specified backend fails to load.
    """
    # Mock tpu_commons to exist, but make the specific backend module
    # (tpu_jax) un-importable by setting it to None in sys.modules.
    mock_tpu_commons = MagicMock()
    mock_modules = {
        'tpu_commons': mock_tpu_commons,
        'tpu_commons.platforms': mock_tpu_commons.platforms,
        'tpu_commons.platforms.tpu_jax': None,
    }
    with patch.dict('sys.modules', mock_modules), patch(
            'os.environ.get',
            return_value="jax"), warnings.catch_warnings(record=True) as w:
        # We expect a warning to be issued
        warnings.simplefilter("always")
        tpu_platform_module = reload_tpu_module()

        # Check that the platform fell back to the vLLM one
        assert "vllm" in tpu_platform_module.TpuPlatform.__module__

        # Check that a warning was raised
        assert len(w) > 0
        assert "failed to load backend" in str(w[-1].message)
