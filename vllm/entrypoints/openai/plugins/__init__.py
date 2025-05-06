# SPDX-License-Identifier: Apache-2.0
from vllm.logger import init_logger
from vllm.plugins import load_plugins_by_group

from .abs_server_plugin import ServerPlugin

logger = init_logger(__name__)

loaded_server_plugins: dict[str, ServerPlugin] = {}


def load_server_plugins() -> dict[str, ServerPlugin]:
    global loaded_server_plugins
    platform_plugins = load_plugins_by_group("vllm.server_plugins")

    activated_plugins = {}

    for name, plugin_cls in platform_plugins.items():
        try:
            assert isinstance(plugin_cls, type), "Plugin class must be a type"
            assert issubclass(
                plugin_cls, ServerPlugin
            ), "Plugin class must be a subclass of ServerPlugin"
            activated_plugins[name] = plugin_cls
        except AssertionError as e:
            logger.error("Failed to load plugin %s: %s", name, e)
            continue

    loaded_server_plugins = activated_plugins
    logger.info("Loaded server plugins: %s",
                list(loaded_server_plugins.keys()))

    return loaded_server_plugins


def get_server_plugins() -> dict[str, ServerPlugin]:
    """Get the loaded server plugins.

    Returns:
        dict[str, ServerPlugin]: A dictionary of loaded server plugins.
    """
    global loaded_server_plugins
    return loaded_server_plugins
