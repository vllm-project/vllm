# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.engine.registry import (_LAZY_ENGINE_MAP, IEngineCoreProc,
                                     _engine_registry, get_engine_core,
                                     register_engine_core)


class MockEngineProc(IEngineCoreProc):

    def __init__(self, *args, **kwargs):
        pass

    def run_busy_loop(self):
        pass

    @staticmethod
    def run_engine_core(*args, **kwargs):
        pass


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
