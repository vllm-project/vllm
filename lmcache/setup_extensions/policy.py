# SPDX-License-Identifier: Apache-2.0
"""Policy-driven extension build orchestrator.

:class:`BuildPolicy` auto-discovers available platform profiles,
selects the best one via explicit env var or auto-detection with fallback,
and drives the common C++ + platform extension build pipeline.
"""

# Standard
from types import ModuleType
from typing import Callable, Iterator, Optional, TypeVar, Union
import importlib
import inspect
import pkgutil
import sys

# First Party
from setup_extensions.build_profiles import BuildProfile
from setup_extensions.common_cpp import build_common_cpp
from setup_extensions.storage_backend_profiles import (  # noqa: E402
    StorageBackendProfile,
)

# ---------------------------------------------------------------------------
# Generic subclass discovery (filesystem-based, from _discovery.py)
# ---------------------------------------------------------------------------

T = TypeVar("T")


def discover_subclasses(
    package: Union[ModuleType, str],
    base_class: type[T],
    *,
    on_import_error: Optional[Callable[[str, Exception], None]] = None,
) -> Iterator[type[T]]:
    """Yield concrete subclasses of *base_class* found in direct
    submodules of *package*.

    Args:
        package: The package to scan (module or dotted name).
        base_class: The base class whose concrete subclasses to collect.
        on_import_error: Optional callback ``(full_module_name, exc)``
            invoked on import failures.  When omitted the error is
            printed to stderr and discovery continues.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    pkg_path = getattr(package, "__path__", None)
    if pkg_path is None:
        raise TypeError(
            "discover_subclasses requires a package (with __path__), "
            "got %r" % (package,)
        )

    seen: set[type] = set()
    for _, short_name, _ in pkgutil.iter_modules(pkg_path):
        full_name = "%s.%s" % (package.__name__, short_name)
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:
            if on_import_error is not None:
                on_import_error(full_name, exc)
            else:
                print(
                    "warning: failed to import %s: %s" % (full_name, exc),
                    file=sys.stderr,
                )
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, base_class) or obj is base_class:
                continue
            if inspect.isabstract(obj):
                continue
            if obj in seen:
                continue
            seen.add(obj)
            yield obj


# ---------------------------------------------------------------------------
# Platform auto-discovery (filesystem-based, no hard-coded module list)
# ---------------------------------------------------------------------------


def _discover_platforms() -> list[BuildProfile]:
    """Auto-discover all platform build profiles.

    Walks ``setup_extensions.build_profiles`` via ``discover_subclasses``,
    which uses ``pkgutil.iter_modules`` to find submodules at the
    filesystem level.  Adding a new ``.py`` file under that package
    with a concrete ``BuildProfile`` subclass is all that is needed —
    no module list to maintain.
    """
    platforms: list[BuildProfile] = []
    for cls in discover_subclasses(
        "setup_extensions.build_profiles",
        BuildProfile,  # type: ignore[type-abstract]
    ):
        platforms.append(cls())
    return platforms


def _discover_storage_backends() -> list[StorageBackendProfile]:
    """Auto-discover all optional L2 storage backend build profiles.

    Walks ``setup_extensions.storage_backend_profiles`` via
    ``discover_subclasses``. Adding a new ``.py`` file under that
    package with a concrete ``StorageBackendProfile`` subclass is all
    that is needed.
    """
    backends: list[StorageBackendProfile] = []
    for cls in discover_subclasses(
        "setup_extensions.storage_backend_profiles",
        StorageBackendProfile,  # type: ignore[type-abstract]
    ):
        backends.append(cls())
    return backends


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------


class BuildPolicy:
    """Selects and builds extensions using platform-aware build profiles.

    Resolution order:
        1. If an explicit ``BUILD_WITH_*`` env var is set, use that profile
           unconditionally (no fallback). A warning is emitted when its
           toolchain cannot be auto-detected, but the build proceeds so
           that the underlying compiler produces the authoritative error.
        2. Otherwise auto-detect with fallback through candidates.
        3. If nothing is available, warn and continue without extensions.
    """

    def __init__(self) -> None:
        self._platforms = _discover_platforms()

    def resolve_profile(self) -> Optional[BuildProfile]:
        """Resolve the active build profile.

        Returns ``None`` when building sdist, native extensions are
        disabled, GPU extensions are disabled, or no profile was
        detected.
        """
        if (
            BuildProfile.is_building_sdist()
            or BuildProfile.is_native_ext_disabled()
            or BuildProfile.is_gpu_ext_disabled()
        ):
            return None

        # ---------------------------------------------------------------
        # Phase 1: explicit env var selection
        # ---------------------------------------------------------------
        explicitly_requested = [
            s for s in self._platforms if s.is_explicitly_requested()
        ]
        if len(explicitly_requested) > 1:
            names = ", ".join(s.name for s in explicitly_requested)
            raise RuntimeError("Multiple profiles explicitly requested: %s" % names)
        if explicitly_requested:
            profile = explicitly_requested[0]
            print("Using explicitly requested profile: %s" % profile.name)
            if not profile.detect():
                print(
                    "warning: profile '%s' was explicitly requested but its "
                    "toolchain was not auto-detected; proceeding anyway" % profile.name,
                    file=sys.stderr,
                )
            return profile

        # ---------------------------------------------------------------
        # Phase 2: auto-detect with fallback
        # ---------------------------------------------------------------
        print("No profile explicitly selected, auto-detecting...")
        for profile in self._platforms:
            if profile.detect():
                print("Auto-detected profile: %s" % profile.name)
                return profile

        # ---------------------------------------------------------------
        # Phase 3: nothing found
        # ---------------------------------------------------------------
        print(
            "warning: no profile detected, building without extensions",
            file=sys.stderr,
        )
        return None

    @staticmethod
    def collect_extensions(
        profile: Optional[BuildProfile],
    ) -> tuple[list, dict, Optional[str]]:
        """Build all extensions and return requirements file name.

        Args:
            profile: Resolved build profile, or ``None``.

        Returns:
            ``(ext_modules, cmdclass, requirements_file)`` tuple.
        """
        if BuildProfile.is_building_sdist():
            print("Not building extensions for sdist")
            return [], {}, None

        if BuildProfile.is_native_ext_disabled():
            return [], {}, None

        # ---- build common C++ extensions ----
        ext_modules, cmdclass = build_common_cpp(profile)

        # ---- build profile-specific extensions ----
        if profile and not BuildProfile.is_gpu_ext_disabled():
            em, cc = profile.build()
            ext_modules.extend(em)
            cmdclass.update(cc)

        # ---- build optional storage backends ----
        storage_flags = profile.default_cxx_flags() if profile else []
        ext_modules.extend(BuildPolicy.collect_storage_backends(storage_flags))

        # ---- requirements ----
        req_file = profile.requirements_file() if profile else None

        return ext_modules, cmdclass, req_file

    @staticmethod
    def collect_storage_backends(
        extra_cxx_flags: list[str],
    ) -> list:
        """Discover and build optional L2 storage backend extensions.

        Each storage backend can be enabled via its ``BUILD_WITH_*``
        env var or auto-detected through its SDK presence.
        """
        storage_backends = _discover_storage_backends()
        ext_modules = []
        for s in storage_backends:
            if s.is_explicitly_requested():
                print("Using explicitly requested storage backend: %s" % s.name)
                ext_modules.extend(s.build(extra_cxx_flags))
            elif s.detect():
                print("Auto-detected storage backend: %s" % s.name)
                ext_modules.extend(s.build(extra_cxx_flags))
        return ext_modules
