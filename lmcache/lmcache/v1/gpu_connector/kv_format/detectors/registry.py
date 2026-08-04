# SPDX-License-Identifier: Apache-2.0
"""The serving-engine -> detector table, discovered from this folder.

Every ``detectors/<engine>.py`` defines one :class:`EngineDetector` subclass.
This imports them all and indexes each by the ``engine_type`` it declares, so
adding an engine is just dropping a new file here -- nothing in this file
changes.
"""

# Standard
from pathlib import Path
import importlib
import pkgutil

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors.base import EngineDetector


def _discover_detectors() -> dict[EngineType, type[EngineDetector]]:
    """Import every detector module in this folder and index it by its engine."""
    detectors: dict[EngineType, type[EngineDetector]] = {}
    for module in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if module.name in ("base", "registry"):
            continue
        imported = importlib.import_module(f"{__package__}.{module.name}")
        for value in vars(imported).values():
            if (
                isinstance(value, type)
                and issubclass(value, EngineDetector)
                and value is not EngineDetector
            ):
                detectors[value.engine_type] = value
    return detectors


DETECTORS = _discover_detectors()


def get_detector(engine_type: EngineType) -> "EngineDetector | None":
    """Return a detector instance for *engine_type*, or ``None`` if none exists."""
    cls = DETECTORS.get(engine_type)
    return cls() if cls is not None else None
