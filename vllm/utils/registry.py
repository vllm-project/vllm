# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


class ExtensionManager:
    """
    A registry for managing pluggable extension classes.

    This class provides a simple mechanism to register and instantiate
    extension classes by name. It is commonly used to implement plugin
    systems where different implementations can be swapped at runtime.

    Examples:
        Basic usage with a registry instance:

        >>> FOO_REGISTRY = ExtensionManager()
        >>> @FOO_REGISTRY.register("my_foo_impl")
        ... class MyFooImpl(Foo):
        ...     def __init__(self, value):
        ...         self.value = value
        >>> foo_impl = FOO_REGISTRY.load("my_foo_impl", value=123)

    """

    def __init__(self) -> None:
        """
        Initialize an empty extension registry.
        """
        self.name2class: dict[str, type] = {}

    def register(self, name: str):
        """
        Decorator to register a class with the given name.
        """

        def wrap(cls_to_register):
            self.name2class[name] = cls_to_register
            return cls_to_register

        return wrap

    def load(self, cls_name: str, *args, **kwargs) -> Any:
        """
        Instantiate and return a registered extension class by name.
        """
        cls = self.name2class.get(cls_name)
        assert cls is not None, f"Extension class {cls_name} not found"
        return cls(*args, **kwargs)
