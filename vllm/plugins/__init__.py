# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import vllm.envs as envs

if TYPE_CHECKING:
    from vllm.entrypoints.serve.endpoint_plugin import EndpointPlugin
    from vllm.tasks import SupportedTask

logger = logging.getLogger(__name__)

# Default plugins group will be loaded in all processes(process0, engine core
# process and worker processes)
DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"
# IO processor plugins group will be loaded in process0 only
IO_PROCESSOR_PLUGINS_GROUP = "vllm.io_processor_plugins"
# Platform plugins group will be loaded in all processes when
# `vllm.platforms.current_platform` is called and the value not initialized,
PLATFORM_PLUGINS_GROUP = "vllm.platform_plugins"
# Stat logger plugins group will be loaded in process0 only when serve vLLM with
# async mode.
STAT_LOGGER_PLUGINS_GROUP = "vllm.stat_logger_plugins"
# Endpoint plugins group is loaded in the API server front end process only.
# Each entry point resolves to a factory returning an `EndpointPlugin`
# (see `vllm/entrypoints/serve/endpoint_plugin.py`).
ENDPOINT_PLUGINS_GROUP = "vllm.endpoint_plugins"

# make sure one process only loads plugins once
plugins_loaded = False


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    """Load plugins registered under the given entry point group."""
    from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", group)
        return {}

    # Check if the only discovered plugin is the default one
    is_default_group = group == DEFAULT_PLUGINS_GROUP
    # Use INFO for non-default groups and DEBUG for the default group
    log_level = logger.debug if is_default_group else logger.info

    log_level("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        log_level("- %s -> %s", plugin.name, plugin.value)

    if allowed_plugins is None:
        log_level(
            "All plugins in this group will be loaded. "
            "Set `VLLM_PLUGINS` to control which plugins to load."
        )

    plugins = dict[str, Callable[[], Any]]()
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("Loading plugin %s", plugin.name)

            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)

    return plugins


def load_general_plugins():
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global plugins_loaded
    if plugins_loaded:
        return
    plugins_loaded = True

    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)
    # general plugins, we only need to execute the loaded functions
    for func in plugins.values():
        func()


def load_endpoint_plugins(
    supported_tasks: "tuple[SupportedTask, ...] | None" = None,
) -> "list[EndpointPlugin]":
    """Discover, gate and instantiate `vllm.endpoint_plugins` entry points.

    Endpoint plugins add HTTP routes to the API server, so they default to
    not loading. Unlike other plugin groups, a plugin here is only
    considered when it is explicitly named in `VLLM_PLUGINS`. This is a
    stricter posture than `load_plugins_by_group` which "load everything unless
    an allowlist says otherwise". This posture is taken to handle potentially
    larger exposed network surface.

    A discovered plugin is loaded only if both hold:
      - it is named in `VLLM_PLUGINS` (enforced by not calling the loader
        at all when `VLLM_PLUGINS` is unset). Note that `VLLM_PLUGINS=""`
        parses to `[""]`, not `None`, so it is treated as a (non strict)
        allowlist that matches no plugin name, not as "unset".
      - its `required_tasks` is `None` or intersects `supported_tasks`.

    Args:
        supported_tasks: Tasks the server supports. `None` means no plugin
            with a non `None` `required_tasks` will be loaded.

    Returns:
        Instantiated plugins that passed gating in discovery order.
    """
    if envs.VLLM_PLUGINS is None:
        logger.warning(
            "VLLM_PLUGINS is not set. No endpoint plugins will be loaded. "
            "Endpoint plugins add HTTP routes and must be explicitly "
            "allowlisted via VLLM_PLUGINS to be loaded."
        )
        return []

    factories = load_plugins_by_group(ENDPOINT_PLUGINS_GROUP)

    endpoint_plugins: list[EndpointPlugin] = []
    for name, factory in factories.items():
        try:
            plugin = factory()
        except Exception:
            logger.exception("Failed to instantiate endpoint plugin %s", name)
            continue

        required_tasks = plugin.required_tasks
        if required_tasks is not None and (
            supported_tasks is None or not set(required_tasks) & set(supported_tasks)
        ):
            logger.info(
                "Skipping endpoint plugin %s: requires one of tasks %s, "
                "server supports %s",
                name,
                required_tasks,
                supported_tasks,
            )
            continue

        logger.info("Loaded endpoint plugin %s", name)
        endpoint_plugins.append(plugin)

    return endpoint_plugins
