# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

from vllm.logger import init_logger
from vllm.utils import import_from_path

logger = init_logger(__name__)


class ExtensionManagerRegistry:
    _registry: dict[str, dict[str, type]] = {}

    @staticmethod
    def _group_key(base_cls: type) -> str:
        return f"{base_cls.__module__}.{base_cls.__name__}"

    @staticmethod
    def _register(base_cls: type, names: list[str]):

        def wrap(impl_cls: type):
            for name in names:
                if base_cls.__name__ not in ExtensionManagerRegistry._registry:
                    ExtensionManagerRegistry._registry[base_cls.__name__] = {}
                if name in ExtensionManagerRegistry._registry[
                        base_cls.__name__]:
                    raise ValueError(
                        f"Extension {name} already registered in group {base_cls.__name__}"  # noqa: E501
                    )
                ExtensionManagerRegistry._registry[
                    base_cls.__name__][name] = impl_cls
            return impl_cls

        return wrap

    @staticmethod
    def _create(base_cls: type, name: str, *args, **kwargs) -> Any:
        if extension_group := ExtensionManagerRegistry._registry.get(
                base_cls.__name__):
            if impl_cls := extension_group.get(name):
                return impl_cls(*args, **kwargs)
            else:
                raise ValueError(
                    f"Extension {name} not found in group {base_cls.__name__}")
        else:
            raise ValueError(f"Extension group {base_cls.__name__} not found")

    @staticmethod
    def _get_extension_class(base_cls: type, name: str) -> type:
        if extension_group := ExtensionManagerRegistry._registry.get(
                base_cls.__name__):
            if impl_cls := extension_group.get(name):
                return impl_cls
            else:
                raise ValueError(
                    f"Extension {name} not found in group {base_cls.__name__}")
        else:
            raise ValueError(
                f"Extension base class {base_cls.__name__} not found")

    @staticmethod
    def _get_valid_extension_names(base_cls: type) -> list[str]:
        if extension_group := ExtensionManagerRegistry._registry.get(
                base_cls.__name__):
            return list(extension_group.keys())
        else:
            return []

    @staticmethod
    def import_extension(extension_path: type) -> None:
        """
        Import a user-defined extension by the path of the extension file.

        Users should use the decorator to register implementations. 
        This method is kept for backward compatibility reasons.
        """
        module_name = os.path.splitext(os.path.basename(extension_path))[0]

        try:
            import_from_path(module_name, extension_path)
        except Exception:
            logger.exception("Failed to load module '%s' from %s.",
                             module_name, extension_path)
        return


class ExtensionManager:

    def __init__(self, base_cls: type) -> None:
        self.base_cls = base_cls

    def register(self, names: list[str]):
        return ExtensionManagerRegistry._register(self.base_cls, names)

    def create(self, name: str, *args, **kwargs) -> Any:
        return ExtensionManagerRegistry._create(self.base_cls, name, *args,
                                                **kwargs)

    def get_extension_class(self, name: str) -> type:
        return ExtensionManagerRegistry._get_extension_class(
            self.base_cls, name)

    def get_valid_extension_names(self) -> list[str]:
        return ExtensionManagerRegistry._get_valid_extension_names(
            self.base_cls)
