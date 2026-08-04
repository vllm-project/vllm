# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy facade for the optional ``humming`` package.

vLLM code should import humming symbols from here so that ``import humming``
(which has import-time side effects) is deferred until first use. Add new
symbols by appending one entry to ``_EXPORTS`` as ``"module.path:attr"``,
or ``"module.path"`` for a whole-module re-export.
"""

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "dtypes": "humming.dtypes",
    "DataType": "humming.dtypes:DataType",
    "GemmType": "humming.config:GemmType",
    "HummingMethod": "humming.layer:HummingMethod",
    "HummingLayerMeta": "humming.layer:HummingLayerMeta",
    "BaseInputSchema": "humming.schema:BaseInputSchema",
    "BaseWeightSchema": "humming.schema:BaseWeightSchema",
    "HummingInputSchema": "humming.schema:HummingInputSchema",
    "HummingWeightSchema": "humming.schema:HummingWeightSchema",
    "quantize_weight": "humming.utils.weight:quantize_weight",
}


def __getattr__(name: str) -> Any:
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'vllm.utils.humming' has no attribute {name!r}")
    if ":" in spec:
        mod_path, attr = spec.split(":", 1)
        obj = getattr(importlib.import_module(mod_path), attr)
    else:
        obj = importlib.import_module(spec)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
