# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, List
from vllm.utils import import_from_path
import os
from vllm.logger import init_logger

logger = init_logger(__name__)

class ExtensionManager:

    _registry: dict[str, dict[str, type]] = {}

    @staticmethod
    def _group_key(base_cls: type) -> str:
        return f"{base_cls.__module__}.{base_cls.__name__}"

    @staticmethod
    def register(base_cls: type, names: List[str]):

        def wrap(impl_cls: type):
            for name in names:
                group_key = ExtensionManager._group_key(base_cls)
                if name in ExtensionManager._registry[group_key]:
                    raise ValueError(f"Extension {name} already registered in group {group_key}")
                ExtensionManager._registry[group_key][name] = impl_cls
            return impl_cls

        return wrap

    @staticmethod
    def create(base_cls: type, name: str, *args, **kwargs) -> Any:
        group_key = ExtensionManager._group_key(base_cls)
        if extension_group := ExtensionManager._registry.get(group_key):
            if impl_cls := extension_group.get(name):
                return impl_cls(*args, **kwargs)
            else:
                raise ValueError(f"Extension {name} not found in group {group_key}")
        else:
            raise ValueError(f"Extension group {group_key} not found")

    @staticmethod
    def get_extension_class(base_cls: type, name: str) -> type:
        group_key = ExtensionManager._group_key(base_cls)
        if extension_group := ExtensionManager._registry.get(group_key):
            if impl_cls := extension_group.get(name):
                return impl_cls
            else:
                raise ValueError(f"Extension {name} not found in group {group_key}")
        else:
            raise ValueError(f"Extension group {group_key} not found")

    @staticmethod
    def get_valid_extension_names(base_cls: type) -> List[str]:
        group_key = ExtensionManager._group_key(base_cls)
        if extension_group := ExtensionManager._registry.get(group_key):
            return list(extension_group.keys())
        else:
            return []

    @staticmethod
    def import_extension(extension_path: type) -> None:
        """
        Import a user-defined extension by the path of the extension file.
        """
        module_name = os.path.splitext(os.path.basename(extension_path))[0]

        try:
            import_from_path(module_name, extension_path)
        except Exception:
            logger.exception("Failed to load module '%s' from %s.",
                                module_name, extension_path)
        return
