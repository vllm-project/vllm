import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.config import CompilationConfig, VllmConfig

logger = logging.getLogger(__name__)

# make sure one process only loads plugins once
plugins_loaded = False


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True
    import sys
    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group='vllm.general_plugins')
    if len(discovered_plugins) == 0:
        logger.info("No plugins found.")
        return
    logger.info("Available plugins:")
    for plugin in discovered_plugins:
        logger.info("name=%s, value=%s, group=%s", plugin.name, plugin.value,
                    plugin.group)
    if allowed_plugins is None:
        logger.info("all available plugins will be loaded.")
        logger.info("set environment variable VLLM_PLUGINS to control"
                    " which plugins to load.")
    else:
        logger.info("plugins to load: %s", allowed_plugins)
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                func()
                logger.info("plugin %s loaded.", plugin.name)
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)


_compilation_config: Optional["CompilationConfig"] = None


def set_compilation_config(config: Optional["CompilationConfig"]):
    global _compilation_config
    _compilation_config = config


def get_compilation_config() -> Optional["CompilationConfig"]:
    return _compilation_config


_current_vllm_config: Optional["VllmConfig"] = None


@contextmanager
def set_current_vllm_config(vllm_config: "VllmConfig"):
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
        logger.debug("enabled custom ops: %s",
                     vllm_config.compilation_config.enabled_custom_ops)
        logger.debug("disabled custom ops: %s",
                     vllm_config.compilation_config.disabled_custom_ops)
        _current_vllm_config = old_vllm_config


def get_current_vllm_config() -> "VllmConfig":
    if _current_vllm_config is None:
        # in ci, usually when we test custom ops/modules directly,
        # we don't set the vllm config. In that case, we set a default
        # config.
        logger.warning("Current VLLM config is not set.")
        from vllm.config import VllmConfig
        return VllmConfig()
    return _current_vllm_config
