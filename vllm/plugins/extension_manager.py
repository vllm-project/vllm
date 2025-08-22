# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from typing import Any, Optional

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
    def _create_or_import(base_cls: type, name: str,
                          extension_path: Optional[str], *args,
                          **kwargs) -> Any:
        extension_group = ExtensionManagerRegistry._registry.get(
            base_cls.__name__)
        if extension_group is not None:
            if name not in extension_group and extension_path:
                logger.info(
                    f"Importing extension {name} from {extension_path}")
                importlib.import_module(extension_path)

            if impl_cls := extension_group.get(name):
                return impl_cls(*args, **kwargs)
            else:
                raise ValueError(
                    f"Extension {name} not found in group {base_cls.__name__}")
        else:
            raise ValueError(
                f"Extension base class {base_cls.__name__} not found")

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
    def import_extension(extension_path: str) -> None:
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


class ExtensionManager:

    def __init__(self, base_cls: type) -> None:
        if base_cls.__name__ in ExtensionManagerRegistry._registry:
            raise ValueError(
                f"Extension group {base_cls.__name__} already exists.")
        ExtensionManagerRegistry._registry[base_cls.__name__] = {}
        self.base_cls = base_cls

    def register(self, names: list[str]):
        return ExtensionManagerRegistry._register(self.base_cls, names)

    def create(self, name: str, *args, **kwargs) -> Any:
        return ExtensionManagerRegistry._create(self.base_cls, name, *args,
                                                **kwargs)

    def create_or_import(self, name: str, extension_path: Optional[str], *args,
                         **kwargs) -> Any:
        return ExtensionManagerRegistry._create_or_import(
            self.base_cls, name, extension_path, *args, **kwargs)

    def get_extension_class(self, name: str) -> type:
        return ExtensionManagerRegistry._get_extension_class(
            self.base_cls, name)

    def get_valid_extension_names(self) -> list[str]:
        return ExtensionManagerRegistry._get_valid_extension_names(
            self.base_cls)
