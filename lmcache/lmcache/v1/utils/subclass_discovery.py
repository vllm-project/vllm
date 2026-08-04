# SPDX-License-Identifier: Apache-2.0
"""Generic helper for plugin-style auto-discovery of concrete subclasses.

This module centralises the boilerplate that several packages used to
duplicate (CLI commands, controller-benchmark handlers, lookup-client
record strategies, health-monitor checks, remote connector adapters,
...).  Each of those packages walks its own submodules with
``pkgutil.iter_modules``, imports them, then iterates classes via
``inspect.getmembers`` to locate concrete subclasses of a given base
class.  ``discover_subclasses`` captures that pattern in a single
well-tested place so callers stay tiny and behave consistently.
"""

# Standard
from types import ModuleType
from typing import Callable, Iterator, Optional, Sequence, TypeVar, Union
import importlib
import inspect
import pkgutil

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

# Sentinel meaning "scan recursively without any depth limit".
_UNLIMITED = 0


def _resolve_package(package: Union[ModuleType, str]) -> ModuleType:
    if isinstance(package, str):
        return importlib.import_module(package)
    return package


def _normalize_levels(levels: Optional[Sequence[int]]) -> tuple[int, int]:
    """Normalize the ``levels`` argument to a ``(min_depth, max_depth)``
    pair. ``max_depth == 0`` is the sentinel for "no upper bound".

    Accepted forms:

    * ``None``                          -> ``(1, 1)`` (legacy behavior).
    * ``[]`` or ``[0, 0]``              -> ``(1, 0)`` -> unlimited depth.
    * ``[lo, hi]`` with ``1 <= lo <= hi`` -> ``(lo, hi)``.

    Any other shape raises ``ValueError`` so misconfigurations surface
    immediately at the call site instead of silently scanning the wrong
    set of modules.
    """
    if levels is None:
        return 1, 1
    if len(levels) == 0:
        return 1, _UNLIMITED
    if len(levels) != 2:
        raise ValueError(
            "levels must be either None, an empty sequence, or a "
            "two-element sequence [min, max]; got %r" % (list(levels),)
        )
    lo, hi = int(levels[0]), int(levels[1])
    if lo == 0 and hi == 0:
        return 1, _UNLIMITED
    if lo < 1 or hi < 1 or lo > hi:
        raise ValueError(
            "levels must satisfy 1 <= min <= max (or be [0, 0] for "
            "unlimited); got %r" % (list(levels),)
        )
    return lo, hi


def discover_subclasses(
    package: Union[ModuleType, str],
    base_class: type[T],
    *,
    module_filter: Optional[Callable[[str], bool]] = None,
    include_abstract: bool = False,
    require_defined_in_module: bool = True,
    on_import_error: Optional[Callable[[str, Exception], None]] = None,
    levels: Optional[Sequence[int]] = None,
) -> Iterator[type[T]]:
    """Yield concrete subclasses of *base_class* found by walking the
    submodules of *package*.

    Each subclass is yielded **at most once**, even when re-exported
    from several modules.

    Args:
        package: The package to scan, either as a module object or its
            fully-qualified dotted name.
        base_class: The base class whose concrete subclasses to collect.
        module_filter: Optional predicate over the *short* module name
            (i.e. without the package prefix).  Modules for which the
            predicate returns ``False`` are skipped.  Sub-packages are
            still descended into so a deep scan can keep filtering by
            leaf module name.  Defaults to ``None`` which keeps every
            module.
        include_abstract: When ``False`` (default) classes with
            unimplemented abstract methods are skipped.
        require_defined_in_module: When ``True`` (default) classes that
            were merely imported (re-exported) into a module are
            ignored; only classes whose ``__module__`` matches the
            module being scanned are yielded.  Set to ``False`` to keep
            the historical behaviour of accepting re-exported classes.
        on_import_error: Optional callback invoked as
            ``on_import_error(full_module_name, exc)`` when a submodule
            fails to import.  When omitted the error is logged at
            ``warning`` level and discovery proceeds with the next
            module.
        levels: Optional ``[min_depth, max_depth]`` window controlling
            which levels of submodules contribute classes.  *package*
            itself is depth ``0``; its direct submodules are depth
            ``1``; etc.  ``None`` (default) and ``[1, 1]`` are
            equivalent and only inspect direct children -- the
            historical behavior.  ``[1, 2]`` inspects direct children
            and grand-children, ``[2, 2]`` only the grand-children, and
            ``[]`` / ``[0, 0]`` recurse without any depth limit.
            A sub-package's ``__init__.py`` is scanned at the depth
            where the sub-package itself sits (e.g. ``bench/__init__.py``
            is depth 1), matching the pre-levels behavior where
            sub-packages were treated as ordinary modules.

    Raises:
        TypeError: If *package* is not a real package (no ``__path__``).
        ValueError: If *levels* is malformed; see :func:`_normalize_levels`.
    """
    pkg = _resolve_package(package)
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        raise TypeError(
            "discover_subclasses requires a package (with __path__), got %r" % (pkg,)
        )

    min_depth, max_depth = _normalize_levels(levels)
    seen: set[type] = set()

    def _walk(
        current_pkg: ModuleType,
        current_path: Sequence[str],
        depth: int,
    ) -> Iterator[type[T]]:
        # Stop descending once we are past the requested upper bound.
        if max_depth != _UNLIMITED and depth > max_depth:
            return

        def _scan_module(module: ModuleType) -> Iterator[type[T]]:
            """Yield classes from *module* that match all filters."""
            for _, obj in inspect.getmembers(module, inspect.isclass):
                try:
                    if not issubclass(obj, base_class) or obj is base_class:
                        continue
                except TypeError:
                    # typing.GenericAlias (e.g. ``list[Foo]``) passes
                    # inspect.isclass on Python < 3.12 but issubclass
                    # raises on ABCs; skip silently.
                    continue
                if not include_abstract and inspect.isabstract(obj):
                    continue
                if require_defined_in_module and (obj.__module__ != module.__name__):
                    continue
                if obj in seen:
                    continue
                seen.add(obj)
                yield obj

        for _, short_name, is_pkg in pkgutil.iter_modules(current_path):
            full_name = "%s.%s" % (current_pkg.__name__, short_name)
            if is_pkg:
                # Recurse into sub-packages so their leaf modules can
                # still be reached.  The sub-package ``__init__.py`` is
                # also scanned at the current depth -- this preserves
                # the pre-levels behaviour where sub-package init files
                # were treated just like ordinary leaf modules.
                can_scan_init = depth >= min_depth and (
                    max_depth == _UNLIMITED or depth <= max_depth
                )
                can_recurse = max_depth == _UNLIMITED or depth + 1 <= max_depth
                if not can_scan_init and not can_recurse:
                    continue
                try:
                    sub_pkg = importlib.import_module(full_name)
                except Exception as exc:
                    _report_import_error(full_name, exc)
                    continue
                sub_path = getattr(sub_pkg, "__path__", None)
                if sub_path is None:
                    continue
                if can_scan_init:
                    if module_filter is None or module_filter(short_name):
                        yield from _scan_module(sub_pkg)
                if can_recurse:
                    yield from _walk(sub_pkg, sub_path, depth + 1)
                continue

            # Leaf module: apply depth window + caller's name filter.
            if depth < min_depth:
                continue
            if max_depth != _UNLIMITED and depth > max_depth:
                continue
            if module_filter is not None and not module_filter(short_name):
                continue
            try:
                module = importlib.import_module(full_name)
            except Exception as exc:
                _report_import_error(full_name, exc)
                continue
            yield from _scan_module(module)

    def _report_import_error(full_name: str, exc: Exception) -> None:
        if on_import_error is not None:
            on_import_error(full_name, exc)
        else:
            logger.warning(
                "Failed to import module %s during subclass discovery: %s",
                full_name,
                exc,
            )

    yield from _walk(pkg, pkg_path, depth=1)
