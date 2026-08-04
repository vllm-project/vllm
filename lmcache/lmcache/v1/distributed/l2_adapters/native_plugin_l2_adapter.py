# SPDX-License-Identifier: Apache-2.0
"""
Native plugin L2 adapter config and factory.

Dynamically loads a third-party pybind-wrapped C++ connector
and wraps it with ``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Any, Optional
import importlib

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)


class NativePluginL2AdapterConfig(L2AdapterConfigBase):
    """Config for a native-plugin L2 adapter.

    Dynamically loads a third-party pybind-wrapped C++
    connector class and wraps it with
    ``NativeConnectorL2Adapter``.

    Fields:
    - module_path: Dotted Python import path of the
        module containing the connector class.
    - class_name: Name of the connector class inside
        *module_path*.
    - adapter_params: Arbitrary dict forwarded to the
        connector class constructor as keyword arguments.
    """

    def __init__(
        self,
        module_path: str,
        class_name: str,
        adapter_params: dict[str, Any] | None = None,
        max_capacity_gb: float = 0,
    ):
        self.module_path = module_path
        self.class_name = class_name
        self.adapter_params = adapter_params or {}
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "NativePluginL2AdapterConfig":
        module_path = d.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            raise ValueError("module_path must be a non-empty string")

        class_name = d.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("class_name must be a non-empty string")

        adapter_params = d.get("adapter_params", {})
        if not isinstance(adapter_params, dict):
            raise ValueError("adapter_params must be a dict")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)) or max_capacity_gb < 0:
            raise ValueError("max_capacity_gb must be a non-negative number")

        return cls(
            module_path=module_path,
            class_name=class_name,
            adapter_params=adapter_params,
            max_capacity_gb=float(max_capacity_gb),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Native plugin L2 adapter config fields:\n"
            "- module_path (str): dotted import path of "
            "the module containing the connector "
            "class (required)\n"
            "- class_name (str): name of the connector "
            "class inside the module (required)\n"
            "- adapter_params (dict): forwarded as "
            "kwargs to the connector constructor "
            "(optional, default {})\n"
            "\n"
            "Example JSON:\n"
            '{"type": "native_plugin", '
            '"module_path": "my_ext.connector", '
            '"class_name": "MyConnectorClient", '
            '"adapter_params": '
            '{"host": "localhost", "port": 1234}}\n'
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)"
        )


def _create_native_plugin_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Dynamically load a third-party native connector
    and wrap it with NativeConnectorL2Adapter."""
    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, NativePluginL2AdapterConfig)

    try:
        module = importlib.import_module(config.module_path)
    except ImportError as e:
        raise ImportError(
            "Could not import module '%s': %s" % (config.module_path, e)
        ) from e

    try:
        connector_cls = getattr(module, config.class_name)
    except AttributeError as e:
        raise AttributeError(
            "Module '%s' has no class '%s': %s"
            % (
                config.module_path,
                config.class_name,
                e,
            )
        ) from e

    native_client = connector_cls(**config.adapter_params)

    # Verify the native client exposes the required
    # interface methods.
    required_methods = [
        "event_fd",
        "submit_batch_get",
        "submit_batch_set",
        "submit_batch_exists",
        "drain_completions",
        "close",
    ]
    try:
        for method in required_methods:
            if not callable(getattr(native_client, method, None)):
                raise TypeError(
                    "%s.%s instance missing required method "
                    "'%s'"
                    % (
                        config.module_path,
                        config.class_name,
                        method,
                    )
                )
    except TypeError:
        # Close the connector to avoid resource leak
        # when validation fails.
        if callable(getattr(native_client, "close", None)):
            native_client.close()
        raise

    if not callable(getattr(native_client, "submit_batch_delete", None)):
        logger.warning(
            "%s.%s does not expose submit_batch_delete; "
            "L2 eviction delete will be a no-op.",
            config.module_path,
            config.class_name,
        )

    logger.info(
        "Created native plugin L2 adapter: %s.%s (params=%s)",
        config.module_path,
        config.class_name,
        config.adapter_params,
    )
    return NativeConnectorL2Adapter(
        native_client, max_capacity_gb=config.max_capacity_gb
    )


register_l2_adapter_type("native_plugin", NativePluginL2AdapterConfig)
register_l2_adapter_factory("native_plugin", _create_native_plugin_l2_adapter)
