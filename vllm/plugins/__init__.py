import logging
from typing import Optional, Type

import vllm.envs as envs
from vllm.executor.executor_base import ExecutorBase

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


_executor_cls_to_use: Optional[Type[ExecutorBase]] = None


def set_executor_cls(executor_cls: Type[ExecutorBase]):
    """
    Set the executor class to use. This is used by plugins to set the executor
    """
    global _executor_cls_to_use
    if _executor_cls_to_use is not None and \
        _executor_cls_to_use is not executor_cls:
        logger.warning(
            "Executor class has already been set to %s by other plugins."
            "Changing the executor class to %s now.", _executor_cls_to_use,
            executor_cls)
    _executor_cls_to_use = executor_cls


def get_executor_cls() -> Optional[Type[ExecutorBase]]:
    """
    Get the executor class to use. This is used by the main code to get the
    executor class to use.
    """
    return _executor_cls_to_use
