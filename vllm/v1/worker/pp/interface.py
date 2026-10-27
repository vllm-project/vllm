# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PPLayerSpec:
    base_boundary: int
    num_layers: int
    minimum_boundary: int = 1


@runtime_checkable
class SupportsDynamicPP(Protocol):
    """Models opt in by loading replicas and selecting a logical layer range."""

    def get_dynamic_pp_layer_spec(self) -> PPLayerSpec: ...

    def set_dynamic_pp_layer_range(self, start: int, end: int) -> None: ...
