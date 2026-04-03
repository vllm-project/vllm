# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import hashlib
import inspect
import types
import weakref
from pathlib import Path
from typing import Any


@functools.lru_cache(maxsize=None)
def hash_file(path: Path) -> str:
    """Hash the contents of a file, cached per path."""
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def hash_source(*srcs: str | Any) -> str:
    """
    Utility method to hash the sources of functions or objects.
    :param srcs: strings or objects to add to the hash.
    Objects and functions have their source inspected.
    :return:
    """
    hasher = hashlib.sha256()
    for src in srcs:
        if src is None:
            src_str = "None"
        elif isinstance(src, str):
            src_str = src
        elif isinstance(src, Path):
            src_str = src.read_text()
        elif isinstance(src, (types.FunctionType, type)):
            src_str = inspect.getsource(src)
        else:
            # object instance
            src_str = inspect.getsource(src.__class__)
        hasher.update(src_str.encode("utf-8"))
    return hasher.hexdigest()


def weak_lru_cache(maxsize: int | None = 128, typed: bool = False):
    """
    LRU Cache decorator that keeps a weak reference to 'self'.
    This avoids memory leakage, which happens when functools.lru_cache
    stores a reference to self in the global cache.

    Taken from: https://stackoverflow.com/a/68052994/5082708
    """

    def wrapper(func):
        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def weak_cache(user_function, /):
    """Simple weak equivalent to functools.cache"""
    return weak_lru_cache(maxsize=None)(user_function)
