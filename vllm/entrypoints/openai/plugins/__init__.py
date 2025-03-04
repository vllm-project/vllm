# SPDX-License-Identifier: Apache-2.0
from vllm.plugins import load_plugins_by_group

from .abs_server_plugin import ServerPlugin


def load_server_plugins() -> dict[str, ServerPlugin]:
    platform_plugins = load_plugins_by_group("vllm.server_plugins")

    activated_plugins = {}

    for name, plugin_cls in platform_plugins.items():
        try:
            assert isinstance(plugin_cls, type)
            assert issubclass(plugin_cls, ServerPlugin)
            activated_plugins[name] = plugin_cls
        except Exception as e:
            print(f"Failed to load plugin {name}: {e}")

    return activated_plugins
