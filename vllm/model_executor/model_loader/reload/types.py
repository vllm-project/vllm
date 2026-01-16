# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from dataclasses import dataclass, field

import torch

__all__ = ["LayerTensors", "LayerReloadingInfo"]

LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


@dataclass
class LayerReloadingInfo:
    # model format (meta), populated by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # kernel format (device)
    kernel_tensors: LayerTensors = ({}, {})

    # track how many restored elements are ready for loading
    load_numel: int | float = 0
    load_numel_total: int | float = float("inf")

    # stores arguments and tensors ready for loading
    loaded_weights: list[tuple[str, inspect.BoundArguments]] = field(
        default_factory=list
    )

    def reset(self):
        self.kernel_tensors = ({}, {})
        self.load_numel = 0
        self.load_numel_total = float("inf")
        self.loaded_weights = list()
