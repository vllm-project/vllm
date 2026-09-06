# SPDX-License-Identifier: Apache-2.0
"""Selective, in-place refresh of runtime-derived weight state."""

from __future__ import annotations

import torch


def refresh_derived_state(
    model: torch.nn.Module,
    updated_parameter_names: frozenset[str] | None = None,
) -> None:
    """Refresh explicitly opted-in modules without rebuilding runtime objects.

    Modules opt in by implementing ``refresh_derived_state`` and exposing a
    truthy ``supports_selective_reload`` capability on their quantization
    method. Unrecognized modules are intentionally skipped until they are
    audited, preserving the existing layerwise fallback.
    """
    names = updated_parameter_names
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        module_capability = getattr(module, "supports_selective_reload", None)
        method_capability = getattr(method, "supports_selective_reload", None)
        module_opted_in = callable(module_capability) and module_capability()
        method_opted_in = callable(method_capability) and method_capability()
        if not (module_opted_in or method_opted_in):
            continue
        refresh = getattr(module, "refresh_derived_state", None)
        if refresh is not None:
            refresh(names)
        elif method_opted_in:
            method.refresh_derived_state(module, names)
