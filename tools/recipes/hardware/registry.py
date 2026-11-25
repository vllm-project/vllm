# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Select optional Recipes backends by recipe hardware ID, not host hardware.

Backend modules are loaded lazily so plain conversion and shell flag selection
never import vLLM or PyTorch or probe unrelated hardware.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.tuning import Policy


@dataclass(frozen=True, slots=True)
class HardwareBackend:
    module_name: str
    label: str
    auto_fit_env: str | None = None
    auto_fit_flag: str = "--auto-fit-parallelism"

    def _load(self) -> ModuleType:
        return import_module(self.module_name)

    def detect(self) -> Any:
        return self._load().detect_hardware()

    def tuning_policies(self) -> tuple[Policy, ...]:
        return self._load().DEFAULT_POLICIES

    def sweep_kwargs(self, hardware: Any) -> dict[str, Any]:
        return self._load().sweep_kwargs(hardware)

    def fit_parallelism(self, config: dict[str, Any], hardware: Any) -> dict[str, Any]:
        fit = getattr(self._load(), "fit_parallelism", None)
        if fit is None:
            raise ValueError(
                f"Automatic TP/DP fallback is not supported for {self.label}"
            )
        return fit(config, hardware)


# New hardware support consists of implementing the backend contract and
# registering the recipe hardware IDs here. No host-wide auto-probe is used.
_BACKENDS = {
    "xeon6": HardwareBackend("hardware.cpu", "Xeon 6"),
    "arc_pro_b70": HardwareBackend(
        "hardware.xpu",
        "B70",
        auto_fit_env="RECIPE_AUTO_FIT_XPU_PARALLELISM",
        auto_fit_flag="--auto-fit-xpu-parallelism",
    ),
}
_DEFAULT_HARDWARE = "xeon6"


def get_hardware_backend(recipe_hardware: object) -> HardwareBackend:
    if not isinstance(recipe_hardware, str) or not recipe_hardware:
        raise ValueError(
            "Runtime tuning requires the resolved Recipes JSON to declare "
            "a non-empty `hardware` field."
        )
    backend = _BACKENDS.get(recipe_hardware)
    if backend is None:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            "Runtime tuning is not supported for recipe hardware "
            f"{recipe_hardware!r}. Currently supported: {supported}."
        )
    return backend


def get_default_hardware_backend() -> HardwareBackend:
    """Keep the historical CPU default for direct runtime-tuning callers."""
    return get_hardware_backend(_DEFAULT_HARDWARE)


def serve_flags(recipe_hardware: str, environ: Mapping[str, str]) -> tuple[str, ...]:
    """Default launcher flags without importing a backend or probing devices."""
    backend = _BACKENDS.get(recipe_hardware.lower())
    if backend is None:
        # Other recipes still support unmodified, plain conversion and serving.
        return ()
    flags = ["--detect-hardware"]
    if backend.auto_fit_env and environ.get(backend.auto_fit_env) in ("true", "1"):
        flags.append(backend.auto_fit_flag)
    return tuple(flags)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: registry.py RECIPE_HARDWARE")
    for flag in serve_flags(sys.argv[1], os.environ):
        print(flag)
