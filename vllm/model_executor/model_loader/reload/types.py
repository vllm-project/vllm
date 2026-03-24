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

    # kernel format (device)
    kernel_tensors: LayerTensors = field(default_factory=lambda: ({}, {}))

    # track how many restored elements are ready for loading
    load_numel: int = 0
    load_numel_total: int | None = None

    # stores arguments and tensors ready for loading
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    def reset(self):
        self.__init__(restore_metadata=self.restore_metadata)  # type: ignore[misc]

    def can_process(self) -> bool:
        return self.load_numel_total is not None
