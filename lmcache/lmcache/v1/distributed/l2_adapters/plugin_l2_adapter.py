# SPDX-License-Identifier: Apache-2.0
"""
Plugin L2 adapter -- dynamically loads an external adapter
class from a user-supplied Python module.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional
import importlib

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface as _L2AI
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

# Config class


class PluginL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for a plugin L2 adapter.

    Dynamically loads an adapter class from a user-supplied
    Python module at creation time.

    Fields:
    - module_path: Dotted Python import path of the module
        containing the adapter class.
    - class_name: Name of the class inside *module_path*
        that implements ``L2AdapterInterface``.
    - adapter_params: Arbitrary dict forwarded to the
        adapter class constructor.
    - config_class_name: Optional name of a config class
        inside *module_path* that subclasses
        ``L2AdapterConfigBase``.  When set, the factory
        builds a config instance via ``from_dict()`` and
        passes it (instead of a raw dict) to the adapter
        constructor -- matching the built-in convention.
    """

    def __init__(
        self,
        module_path: str,
        class_name: str,
        adapter_params: dict[str, Any] | None = None,
        config_class_name: str | None = None,
    ):
        self.module_path = module_path
        self.class_name = class_name
        self.adapter_params = adapter_params or {}
        self.config_class_name = config_class_name

    @classmethod
    def from_dict(cls, d: dict) -> "PluginL2AdapterConfig":
        module_path = d.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            raise ValueError("module_path must be a non-empty string")

        class_name = d.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("class_name must be a non-empty string")

        adapter_params = d.get("adapter_params", {})
        if not isinstance(adapter_params, dict):
            raise ValueError("adapter_params must be a dict")

        config_class_name = d.get("config_class_name")
        if config_class_name is not None and not isinstance(config_class_name, str):
            raise ValueError("config_class_name must be a string")

        return cls(
            module_path=module_path,
            class_name=class_name,
            adapter_params=adapter_params,
            config_class_name=config_class_name,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Plugin L2 adapter config fields:\n"
            "- module_path (str): dotted import path of "
            "the module containing the adapter class "
            "(required)\n"
            "- class_name (str): name of the adapter "
            "class inside the module (required)\n"
            "- adapter_params (dict): forwarded to the "
            "adapter constructor "
            "(optional, default {})\n"
            "- config_class_name (str): explicit config "
            "class name; when omitted the factory "
            "auto-discovers it (see plugin.md)\n"
            "\n"
            "Example JSON (raw dict):\n"
            '{"type": "plugin", '
            '"module_path": "my_plugin.l2", '
            '"class_name": "MyL2Adapter", '
            '"adapter_params": {"host": "localhost"}}\n'
            "\n"
            "Example JSON (with config class):\n"
            '{"type": "plugin", '
            '"module_path": "my_plugin.l2", '
            '"class_name": "MyL2Adapter", '
            '"config_class_name": "MyL2AdapterConfig", '
            '"adapter_params": {"host": "localhost"}}'
        )


# Factory function


def _create_plugin_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> _L2AI:
    """Dynamically load and create a plugin L2 adapter."""
    assert isinstance(config, PluginL2AdapterConfig)

    try:
        module = importlib.import_module(config.module_path)
    except ImportError as e:
        raise ImportError(
            "Could not import module '%s': %s" % (config.module_path, e)
        ) from e

    try:
        adapter_cls = getattr(module, config.class_name)
    except AttributeError as e:
        raise AttributeError(
            "Module '%s' has no class '%s': %s"
            % (config.module_path, config.class_name, e)
        ) from e

    if not (isinstance(adapter_cls, type) and issubclass(adapter_cls, _L2AI)):
        raise TypeError(
            "%s.%s is not a subclass of "
            "L2AdapterInterface" % (config.module_path, config.class_name)
        )

    cfg_cls = _resolve_config_class(
        module,
        config,
        adapter_cls,
    )
    kwargs: dict[str, object] = {}
    if l1_memory_desc is not None:
        kwargs["l1_memory_desc"] = l1_memory_desc

    # mypy sees ``adapter_cls`` as ``type[L2AdapterInterface]`` and
    # complains that we're passing a config object / dict where the
    # base class expects ``max_capacity_bytes: int``. The plugin
    # surface is intentionally dynamic, so silence both checks.
    if cfg_cls is not None:
        return adapter_cls(
            cfg_cls.from_dict(config.adapter_params),  # type: ignore[arg-type]
            **kwargs,  # type: ignore[call-arg]
        )

    return adapter_cls(
        config.adapter_params,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[call-arg]
    )


def _resolve_config_class(
    module: object,
    config: PluginL2AdapterConfig,
    adapter_cls: type,
) -> type[L2AdapterConfigBase] | None:
    """Resolve the config class for a plugin adapter.

    Discovery order:
    1. Explicit ``config.config_class_name`` field.
    2. Convention: ``config.class_name`` + ``"Config"``.
    3. ``config_class_name`` attribute on the adapter
       class itself.
    4. ``None`` -- fall back to raw dict mode.
    """
    candidates: list[str | None] = [
        config.config_class_name,
        config.class_name + "Config",
        getattr(adapter_cls, "config_class_name", None),
    ]
    for name in candidates:
        if not name:
            continue
        cls = getattr(module, name, None)
        if cls is None:
            continue
        if isinstance(cls, type) and issubclass(
            cls,
            L2AdapterConfigBase,
        ):
            return cls
    return None


# Self-register config type and adapter factory
register_l2_adapter_type("plugin", PluginL2AdapterConfig)
register_l2_adapter_factory("plugin", _create_plugin_adapter)
