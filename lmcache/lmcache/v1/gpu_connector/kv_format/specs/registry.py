# SPDX-License-Identifier: Apache-2.0
"""The ``EngineKVFormat`` -> spec table, discovered from this folder.

Every ``specs/<engine_kv_format>.py`` defines one :class:`KVFormatSpec`
subclass. This imports them all and indexes each by the ``engine_kv_format`` it
declares, so adding a format is just dropping a new file here -- nothing in this
file changes.
"""

# Standard
from pathlib import Path
import importlib
import pkgutil

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
import lmcache.c_ops as lmc_ops


def _discover_specs() -> dict["lmc_ops.EngineKVFormat", type[KVFormatSpec]]:
    """Import every spec module in this folder and index it by its format."""
    specs: dict["lmc_ops.EngineKVFormat", type[KVFormatSpec]] = {}
    for module in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if module.name in ("base", "registry"):
            continue
        imported = importlib.import_module(f"{__package__}.{module.name}")
        for value in vars(imported).values():
            if (
                isinstance(value, type)
                and issubclass(value, KVFormatSpec)
                and value is not KVFormatSpec
            ):
                specs[value.engine_kv_format] = value
    return specs


SPECS = _discover_specs()


def get_spec_class(fmt: "lmc_ops.EngineKVFormat") -> type[KVFormatSpec]:
    """Return the spec class for *fmt*, for static facts (``is_mla``, ...).

    Raises:
        ValueError: If *fmt* has no spec.
    """
    if fmt not in SPECS:
        raise ValueError(f"Unknown Engine KV Format: {fmt}")
    return SPECS[fmt]


def get_spec(
    kv_caches: DiscoverableKVCache, fmt: "lmc_ops.EngineKVFormat"
) -> KVFormatSpec:
    """Return a spec instance wrapping *kv_caches* of *fmt*."""
    return get_spec_class(fmt)(kv_caches)
