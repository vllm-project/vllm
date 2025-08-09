# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.registry import (_LAZY_ENGINE_MAP, IEngineCoreProc,
                                     _engine_registry, find_supported_engine,
                                     get_engine_core, register_engine_core)


class MockEngineProc(IEngineCoreProc):

    def __init__(self, *args, **kwargs):
        pass

    def run_busy_loop(self):
        pass

    @staticmethod
    def run_engine_core(*args, **kwargs):
        pass

    @staticmethod
    def is_supported() -> bool:
        # The base mock is not supported, specific mocks will override this.
        return False


class SupportedEngine(MockEngineProc):

    @staticmethod
    def is_supported() -> bool:
        return True


class AnotherSupportedEngine(MockEngineProc):

    @staticmethod
    def is_supported() -> bool:
        return True


@pytest.fixture
def cleanup_registry():
    """Fixture to clean up the registry after each test."""
    original_lazy_map = _LAZY_ENGINE_MAP.copy()
    yield
    _engine_registry.clear()
    _LAZY_ENGINE_MAP.clear()
    _LAZY_ENGINE_MAP.update(original_lazy_map)


def test_register_and_get_engine(cleanup_registry):
    """Test that an engine can be registered and retrieved."""
    register_engine_core("mock", MockEngineProc)
    assert get_engine_core("mock") is MockEngineProc


def test_get_non_existent_engine(cleanup_registry):
    """Test that getting a non-existent engine raises a ValueError."""
    with pytest.raises(ValueError,
                       match="Engine core 'non_existent' is not registered."):
        get_engine_core("non_existent")


def test_register_duplicate_engine(cleanup_registry):
    """Test that registering a duplicate engine raises a ValueError."""
    register_engine_core("mock", MockEngineProc)
    with pytest.raises(ValueError,
                       match="Engine core 'mock' is already registered."):
        register_engine_core("mock", MockEngineProc)


# --- Tests for find_supported_engine ---


def test_find_engine_empty_registry(cleanup_registry):
    """Test that no engine is found when the registry is empty."""
    assert find_supported_engine() is None


def test_find_engine_none_supported(cleanup_registry):
    """Test that no engine is found when no registered engines are supported."""
    register_engine_core("unsupported", MockEngineProc)
    assert find_supported_engine() is None


def test_find_engine_one_supported(cleanup_registry):
    """Test that the correct engine is found when one is supported."""
    register_engine_core("unsupported", MockEngineProc)
    register_engine_core("supported", SupportedEngine)
    assert find_supported_engine() is SupportedEngine


def test_find_engine_multiple_supported_fails(cleanup_registry):
    """Test that an error is raised if multiple engines are supported."""
    register_engine_core("supported1", SupportedEngine)
    register_engine_core("supported2", AnotherSupportedEngine)
    with pytest.raises(RuntimeError, match="Multiple custom engines support"):
        find_supported_engine()