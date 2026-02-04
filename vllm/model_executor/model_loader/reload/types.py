# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from inspect import BoundArguments

import torch

__all__ = ["LayerTensors", "LayerReloadingInfo"]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


@dataclass
class LayerReloadingInfo:
    # model format (meta), populated by `record_metadata_for_reloading`
    restore_metadata: LayerTensors = field(default_factory=lambda: ({}, {}))

    # kernel format (device), used for layerwise reloading
    kernel_tensors: LayerTensors = field(default_factory=lambda: ({}, {}))

    # model format (device), used for initial layerwise loading
    model_format_tensors: LayerTensors = field(default_factory=lambda: ({}, {}))

    # track how many restored elements are ready for loading
    load_numel: int = 0
    load_numel_total: int | None = None

    # stores arguments and tensors ready for loading
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    def reset(self):
        self.__init__(restore_metadata=self.restore_metadata)  # type: ignore[misc]

    def can_process(self) -> bool:
        return self.load_numel_total is not None

    @property
    def has_model_format_tensors(self) -> bool:
        parameters, buffers = self.model_format_tensors
        return len(parameters) > 0 or len(buffers) > 0

    @property
    def has_kernel_tensors(self) -> bool:
        parameters, buffers = self.kernel_tensors
        return len(parameters) > 0 or len(buffers) > 0
