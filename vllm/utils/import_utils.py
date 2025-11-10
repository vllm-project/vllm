# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers related to importing modules.

This is similar in concept to the `importlib` module.
"""

import importlib.metadata
import importlib.util
import os
import sys
from functools import cache
from types import ModuleType
from typing import Any

import regex as re
from typing_extensions import Never


# TODO: This function can be removed if transformer_modules classes are
# serialized by value when communicating between processes
def init_cached_hf_modules() -> None:
    """
    Lazy initialization of the Hugging Face modules.
    """
    from transformers.dynamic_module_utils import init_hf_modules

    init_hf_modules()


def import_pynvml():
    """
    Historical comments:

    libnvml.so is the library behind nvidia-smi, and
    pynvml is a Python wrapper around it. We use it to get GPU
    status without initializing CUDA context in the current process.
    Historically, there are two packages that provide pynvml:
    - `nvidia-ml-py` (https://pypi.org/project/nvidia-ml-py/): The official
        wrapper. It is a dependency of vLLM, and is installed when users
        install vLLM. It provides a Python module named `pynvml`.
    - `pynvml` (https://pypi.org/project/pynvml/): An unofficial wrapper.
        Prior to version 12.0, it also provides a Python module `pynvml`,
        and therefore conflicts with the official one. What's worse,
        the module is a Python package, and has higher priority than
        the official one which is a standalone Python file.
        This causes errors when both of them are installed.
        Starting from version 12.0, it migrates to a new module
        named `pynvml_utils` to avoid the conflict.
    It is so confusing that many packages in the community use the
    unofficial one by mistake, and we have to handle this case.
    For example, `nvcr.io/nvidia/pytorch:24.12-py3` uses the unofficial
    one, and it will cause errors, see the issue
    https://github.com/vllm-project/vllm/issues/12847 for example.
    After all the troubles, we decide to copy the official `pynvml`
    module to our codebase, and use it directly.
    """
    import vllm.third_party.pynvml as pynvml

    return pynvml


def import_from_path(module_name: str, file_path: str | os.PathLike):
    """
    Import a Python file according to its file path.

    Based on the official recipe:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named {module_name!r}")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully-qualified class name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


@cache
def get_vllm_optional_dependencies():
    metadata = importlib.metadata.metadata("vllm")
    requirements = metadata.get_all("Requires-Dist", [])
    extras = metadata.get_all("Provides-Extra", [])

    return {
        extra: [
            re.split(r";|>=|<=|==", req)[0]
            for req in requirements
            if req.endswith(f'extra == "{extra}"')
        ]
        for extra in extras
    }


class _PlaceholderBase:
    """
    Disallows downstream usage of placeholder modules.

    We need to explicitly override each dunder method because
    [`__getattr__`][vllm.utils.import_utils._PlaceholderBase.__getattr__]
    is not called when they are accessed.

    Info:
        [Special method lookup](https://docs.python.org/3/reference/datamodel.html#special-lookup)
    """

    def __getattr__(self, key: str) -> Never:
        """
        The main class should implement this to throw an error
        for attribute accesses representing downstream usage.
        """
        raise NotImplementedError

    # [Basic customization]

    def __lt__(self, other: object):
        return self.__getattr__("__lt__")

    def __le__(self, other: object):
        return self.__getattr__("__le__")

    def __eq__(self, other: object):
        return self.__getattr__("__eq__")

    def __ne__(self, other: object):
        return self.__getattr__("__ne__")

    def __gt__(self, other: object):
        return self.__getattr__("__gt__")

    def __ge__(self, other: object):
        return self.__getattr__("__ge__")

    def __hash__(self):
        return self.__getattr__("__hash__")

    def __bool__(self):
        return self.__getattr__("__bool__")

    # [Callable objects]

    def __call__(self, *args: object, **kwargs: object):
        return self.__getattr__("__call__")

    # [Container types]

    def __len__(self):
        return self.__getattr__("__len__")

    def __getitem__(self, key: object):
        return self.__getattr__("__getitem__")

    def __setitem__(self, key: object, value: object):
        return self.__getattr__("__setitem__")

    def __delitem__(self, key: object):
        return self.__getattr__("__delitem__")

    # __missing__ is optional according to __getitem__ specification,
    # so it is skipped

    # __iter__ and __reversed__ have a default implementation
    # based on __len__ and __getitem__, so they are skipped.

    # [Numeric Types]

    def __add__(self, other: object):
        return self.__getattr__("__add__")

    def __sub__(self, other: object):
        return self.__getattr__("__sub__")

    def __mul__(self, other: object):
        return self.__getattr__("__mul__")

    def __matmul__(self, other: object):
        return self.__getattr__("__matmul__")

    def __truediv__(self, other: object):
        return self.__getattr__("__truediv__")

    def __floordiv__(self, other: object):
        return self.__getattr__("__floordiv__")

    def __mod__(self, other: object):
        return self.__getattr__("__mod__")

    def __divmod__(self, other: object):
        return self.__getattr__("__divmod__")

    def __pow__(self, other: object, modulo: object = ...):
        return self.__getattr__("__pow__")

    def __lshift__(self, other: object):
        return self.__getattr__("__lshift__")

    def __rshift__(self, other: object):
        return self.__getattr__("__rshift__")

    def __and__(self, other: object):
        return self.__getattr__("__and__")

    def __xor__(self, other: object):
        return self.__getattr__("__xor__")

    def __or__(self, other: object):
        return self.__getattr__("__or__")

    # r* and i* methods have lower priority than
    # the methods for left operand so they are skipped

    def __neg__(self):
        return self.__getattr__("__neg__")

    def __pos__(self):
        return self.__getattr__("__pos__")

    def __abs__(self):
        return self.__getattr__("__abs__")

    def __invert__(self):
        return self.__getattr__("__invert__")

    # __complex__, __int__ and __float__ have a default implementation
    # based on __index__, so they are skipped.

    def __index__(self):
        return self.__getattr__("__index__")

    def __round__(self, ndigits: object = ...):
        return self.__getattr__("__round__")

    def __trunc__(self):
        return self.__getattr__("__trunc__")

    def __floor__(self):
        return self.__getattr__("__floor__")

    def __ceil__(self):
        return self.__getattr__("__ceil__")

    # [Context managers]

    def __enter__(self):
        return self.__getattr__("__enter__")

    def __exit__(self, *args: object, **kwargs: object):
        return self.__getattr__("__exit__")


class PlaceholderModule(_PlaceholderBase):
    """
    A placeholder object to use when a module does not exist.

    This enables more informative errors when trying to access attributes
    of a module that does not exist.
    """

    def __init__(self, name: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__name = name

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self, attr_path)

    def __getattr__(self, key: str) -> Never:
        name = self.__name

        try:
            importlib.import_module(name)
        except ImportError as exc:
            for extra, names in get_vllm_optional_dependencies().items():
                if name in names:
                    msg = f"Please install vllm[{extra}] for {extra} support"
                    raise ImportError(msg) from exc

            raise exc

        raise AssertionError(
            "PlaceholderModule should not be used "
            "when the original module can be imported"
        )


class _PlaceholderModuleAttr(_PlaceholderBase):
    def __init__(self, module: PlaceholderModule, attr_path: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__module = module
        self.__attr_path = attr_path

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self.__module, f"{self.__attr_path}.{attr_path}")

    def __getattr__(self, key: str) -> Never:
        getattr(self.__module, f"{self.__attr_path}.{key}")

        raise AssertionError(
            "PlaceholderModule should not be used "
            "when the original module can be imported"
        )


class LazyLoader(ModuleType):
    """
    `LazyLoader` module borrowed from [Tensorflow]
    (https://github.com/tensorflow/tensorflow/blob/main/tensorflow/python/util/lazy_loader.py)
    with an addition of "module caching".

    Lazily import a module, mainly to avoid pulling in large dependencies.
    Modules such as `xgrammar` might do additional side effects, so we
    only want to use this when it is needed, delaying all eager effects.
    """

    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        name: str,
    ):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module: ModuleType | None = None

        super().__init__(str(name))

    def _load(self) -> ModuleType:
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            # The additional add to sys.modules
            # ensures library is actually loaded.
            sys.modules[self._local_name] = module
        except ModuleNotFoundError as err:
            raise err from None

        # Update this object's dict so that if someone keeps a
        # reference to the LazyLoader, lookups are efficient
        # (__getattr__ is only called on lookups that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: Any) -> Any:
        if self._module is None:
            self._module = self._load()
        return getattr(self._module, item)

    def __dir__(self) -> list[str]:
        if self._module is None:
            self._module = self._load()
        return dir(self._module)


# Optional dependency detection utilities
@cache
def _has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.

    The result is cached so that subsequent queries for the same module incur
    no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None


def has_pplx() -> bool:
    """Whether the optional `pplx_kernels` package is available."""
    return _has_module("pplx_kernels")


def has_deep_ep() -> bool:
    """Whether the optional `deep_ep` package is available."""
    return _has_module("deep_ep")


def has_deep_gemm() -> bool:
    """Whether the optional `deep_gemm` package is available."""
    return _has_module("deep_gemm")


def has_triton_kernels() -> bool:
    """Whether the optional `triton_kernels` package is available."""
    return _has_module("triton_kernels")


def has_tilelang() -> bool:
    """Whether the optional `tilelang` package is available."""
    return _has_module("tilelang")


def has_arctic_inference() -> bool:
    """Whether the optional `arctic_inference` package is available."""

    return _has_module("arctic_inference")
