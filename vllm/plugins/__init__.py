import logging

import vllm.envs as envs

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

    allowed_plugins = envs.VLLM_GENERAL_PLUGINS

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
