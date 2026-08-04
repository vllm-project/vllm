# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
from typing import Iterable, List, Optional
import importlib
import pkgutil

# Third Party
from fastapi import APIRouter

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def discover_api_routers(
    search_path: Path,
    package_name: str,
    suffix: str = "_api",
    exclude: Optional[Iterable[str]] = None,
) -> List[APIRouter]:
    """Scan *search_path* for modules whose name ends with *suffix*
    and return every ``router`` attribute that is an
    :class:`~fastapi.APIRouter`.

    Args:
        search_path: Filesystem directory to scan.
        package_name: Fully-qualified Python package name that
            corresponds to *search_path* (used by
            :func:`importlib.import_module`).
        suffix: Only modules whose name ends with this string
            are considered.  Defaults to ``"_api"``.
        exclude: Optional iterable of module base names to skip
            (e.g. ``{"run_script_api"}``).  Useful when a host
            package only wants a subset of the available routers.

    Returns:
        A list of discovered :class:`~fastapi.APIRouter` instances.
    """
    excluded = set(exclude or ())
    routers: List[APIRouter] = []
    for _, module_name, _ in pkgutil.iter_modules([str(search_path)]):
        if not module_name.endswith(suffix):
            continue
        if module_name in excluded:
            logger.info("Skipping excluded API module: %s", module_name)
            continue
        full_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_name)
        if hasattr(module, "router") and isinstance(module.router, APIRouter):
            routers.append(module.router)
            logger.info("Discovered API module: %s", module_name)
    return routers
