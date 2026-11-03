# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer-type instances of the ``kernels`` package.

The layer geometry (NoPE query width, KV sharing) is a module-level constant of
the kernel sources (``kernels/variant.py``), so each layer type loads the one
source tree under its own package name with its own ``variant`` module.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import NamedTuple

_KERNELS = Path(__file__).parent.parent / "mono"


class KernelVariant(NamedTuple):
    op: types.ModuleType
    layout: types.ModuleType
    geometry_config: types.ModuleType


def load_variant(nope: bool, kv_share: bool) -> KernelVariant:
    """The kernel package compiled for one ML4 layer type."""
    assert nope or not kv_share, "KV-sharing layers are NoPE layers"
    name = f"{__package__}._kernels_{'nope' if nope else 'rope'}{'_kvs' if kv_share else ''}"  # noqa: E501
    if name not in sys.modules:
        variant = types.ModuleType(f"{name}.variant")
        variant.NOPE_LAYER = nope  # type: ignore[attr-defined]
        variant.KV_SHARE = kv_share  # type: ignore[attr-defined]
        sys.modules[variant.__name__] = variant
        spec = importlib.util.spec_from_file_location(
            name, _KERNELS / "__init__.py", submodule_search_locations=[str(_KERNELS)]
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
    return KernelVariant(
        importlib.import_module(f"{name}.op"),
        importlib.import_module(f"{name}.layout"),
        importlib.import_module(f"{name}.geometry_config"),
    )
