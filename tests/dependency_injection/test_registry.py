# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

# NOTE: We need to import the modules to be tested *after* we mock
# the modules they might import.


class MockEngineProc:
    """A mock engine class that can be imported by the tests."""

    def __init__(self, *args, **kwargs):
        pass

    def run_busy_loop(self):
        pass

    @staticmethod
    def run_engine_core(*args, **kwargs):
        pass

    @staticmethod
    def is_supported() -> bool:
        return True


@pytest.fixture
def cleanup_registry():
    """Fixture to clean up the registry after each test."""
    from vllm.dependency_injection.registry import (_LAZY_ENGINE_CORE_PROC_MAP,
                                                    _engine_core_proc_registry)
    original_lazy_map = _LAZY_ENGINE_CORE_PROC_MAP.copy()
    yield
    _engine_core_proc_registry.clear()
    _LAZY_ENGINE_CORE_PROC_MAP.clear()
    _LAZY_ENGINE_CORE_PROC_MAP.update(original_lazy_map)


# --- Tests for registration and basic retrieval ---


def test_register_and_get_engine(cleanup_registry):
    """Test that an engine can be registered and retrieved."""
    from vllm.dependency_injection.registry import (register_engine_core_proc,
                                                    retrieve_engine_core_proc)
    register_engine_core_proc("mock", MockEngineProc)
    assert retrieve_engine_core_proc("mock") is MockEngineProc


def test_get_non_existent_engine(cleanup_registry):
    """Test that getting a non-existent engine raises a ValueError."""
    from vllm.dependency_injection.registry import retrieve_engine_core_proc
    with pytest.raises(ValueError,
                       match="Engine core 'non_existent' is not registered."):
        retrieve_engine_core_proc("non_existent")


def test_register_duplicate_engine(cleanup_registry):
    """Test that registering a duplicate engine raises a ValueError."""
    from vllm.dependency_injection.registry import register_engine_core_proc
    register_engine_core_proc("mock", MockEngineProc)
    with pytest.raises(ValueError, match="is already registered"):
        register_engine_core_proc("mock", MockEngineProc)


@pytest.mark.parametrize("reserved_name", ["auto", "default"])
def test_register_reserved_name_fails(reserved_name: str, cleanup_registry):
    """Test that registering an engine with a reserved name fails."""
    from vllm.dependency_injection.registry import register_engine_core_proc
    with pytest.raises(ValueError, match="is reserved and cannot be used"):
        register_engine_core_proc(reserved_name, MockEngineProc)


# --- Tests for lazy loading in retrieve_engine_core_proc ---


def test_lazy_load_success(cleanup_registry):
    """Test that a lazy-loadable engine can be imported and retrieved."""
    from vllm.dependency_injection.registry import (_LAZY_ENGINE_CORE_PROC_MAP,
                                                    _engine_core_proc_registry,
                                                    retrieve_engine_core_proc)

    # This test file can be imported, and MockEngineProc is a valid class.
    # Note that we are testing the *current* module.
    module_path = __name__
    _LAZY_ENGINE_CORE_PROC_MAP["lazy_mock"] = {
        "path": f"{module_path}.MockEngineProc"
    }

    # Should not be in the main registry yet.
    assert "lazy_mock" not in _engine_core_proc_registry

    # Upon retrieval, it should be loaded and registered.
    engine_class = retrieve_engine_core_proc("lazy_mock")
    assert engine_class is MockEngineProc
    assert "lazy_mock" in _engine_core_proc_registry
    assert _engine_core_proc_registry["lazy_mock"] is MockEngineProc


def test_lazy_load_import_error(cleanup_registry):
    """Test that a helpful error is raised for a non-existent module."""
    from vllm.dependency_injection.registry import (_LAZY_ENGINE_CORE_PROC_MAP,
                                                    retrieve_engine_core_proc)
    _LAZY_ENGINE_CORE_PROC_MAP["bad_import"] = {
        "path": "non_existent_module.NonExistentClass"
    }

    with pytest.raises(ValueError, match="could not be imported from"):
        retrieve_engine_core_proc("bad_import")


def test_lazy_load_attribute_error(cleanup_registry):
    """Test that a helpful error is raised for a non-existent class."""
    from vllm.dependency_injection.registry import (_LAZY_ENGINE_CORE_PROC_MAP,
                                                    retrieve_engine_core_proc)
    module_path = __name__
    _LAZY_ENGINE_CORE_PROC_MAP["bad_attr"] = {
        "path": f"{module_path}.NonExistentClass"
    }

    with pytest.raises(ValueError, match="could not be imported from"):
        retrieve_engine_core_proc("bad_attr")


def test_lazy_load_with_extra_install(cleanup_registry):
    """Test that the error message includes the pip install hint."""
    from vllm.dependency_injection.registry import (_LAZY_ENGINE_CORE_PROC_MAP,
                                                    retrieve_engine_core_proc)
    _LAZY_ENGINE_CORE_PROC_MAP["tpu_engine"] = {
        "path": "vllm_tpu_plugin.TpuEngine",
        "extra": "tpu"
    }

    with pytest.raises(ValueError, match="pip install vllm\[tpu\]"):
        retrieve_engine_core_proc("tpu_engine")


# --- Tests for discover_supported_engine_core_proc ---


class UnsupportedEngine(MockEngineProc):

    @staticmethod
    def is_supported() -> bool:
        return False


class AnotherSupportedEngine(MockEngineProc):

    @staticmethod
    def is_supported() -> bool:
        return True


def test_discover_engine_empty_registry(cleanup_registry):
    """Test that no engine is found when the registry is empty."""
    from vllm.dependency_injection.registry import (
        discover_supported_engine_core_proc)
    assert discover_supported_engine_core_proc() is None


def test_discover_engine_none_supported(cleanup_registry):
    """Test that no engine is found when no registered engines are supported."""
    from vllm.dependency_injection.registry import (
        discover_supported_engine_core_proc, register_engine_core_proc)
    register_engine_core_proc("unsupported", UnsupportedEngine)
    assert discover_supported_engine_core_proc() is None


def test_discover_engine_one_supported(cleanup_registry):
    """Test that the correct engine is found when one is supported."""
    from vllm.dependency_injection.registry import (
        discover_supported_engine_core_proc, register_engine_core_proc)
    register_engine_core_proc("unsupported", UnsupportedEngine)
    register_engine_core_proc(
        "supported", MockEngineProc)  # MockEngineProc now returns True
    assert discover_supported_engine_core_proc() is MockEngineProc


def test_discover_engine_multiple_supported_fails(cleanup_registry):
    """Test that an error is raised if multiple engines are supported."""
    from vllm.dependency_injection.registry import (
        discover_supported_engine_core_proc, register_engine_core_proc)
    register_engine_core_proc("supported1", MockEngineProc)
    register_engine_core_proc("supported2", AnotherSupportedEngine)
    with pytest.raises(RuntimeError, match="Multiple custom engines support"):
        discover_supported_engine_core_proc()
