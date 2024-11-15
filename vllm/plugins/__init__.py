import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Optional, Union

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, VllmConfig
else:
    CompilationConfig = None
    VllmConfig = None

logger = logging.getLogger(__name__)


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group='vllm.general_plugins')
    for plugin in discovered_plugins:
        logger.info("Found general plugin: %s", plugin.name)
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                func()
                logger.info("Loaded general plugin: %s", plugin.name)
            except Exception:
                logger.exception("Failed to load general plugin: %s",
                                 plugin.name)


_torch_compile_backend: Optional[Union[Callable, str]] = None


def set_torch_compile_backend(backend: Union[Callable, str]):
    global _torch_compile_backend
    _torch_compile_backend = backend


def get_torch_compile_backend() -> Optional[Union[Callable, str]]:
    return _torch_compile_backend


_compilation_config: Optional[CompilationConfig] = None


def set_compilation_config(config: Optional[CompilationConfig]):
    global _compilation_config
    _compilation_config = config


def get_compilation_config() -> Optional[CompilationConfig]:
    return _compilation_config


_current_vllm_config: Optional[VllmConfig] = None


@contextmanager
def set_current_vllm_config(vllm_config: VllmConfig):
    """
    Temporarily set the current VLLM config.
    Used during model initialization.
    We save the current VLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the VLLM config to determine how to dispatch.
    """
    global _current_vllm_config
    old_vllm_config = _current_vllm_config
    try:
        _current_vllm_config = vllm_config
        yield
    finally:
        _current_vllm_config = old_vllm_config


def get_current_vllm_config() -> VllmConfig:
    assert _current_vllm_config is not None, "Current VLLM config is not set."
    return _current_vllm_config
