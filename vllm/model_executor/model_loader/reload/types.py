# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter
from dataclasses import dataclass, field
from inspect import BoundArguments
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .plan import LoadPlan

__all__ = ["LayerTensors", "LayerReloadingInfo"]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


@dataclass
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # Whether layerwise loading is armed for this layer
    active: bool = False

    # The loader applications this layer expects, recorded during the initial
    # checkpoint load and empty until one has been.
    expected_loads: "LoadPlan" = field(default_factory=Counter)
    observed_loads: "LoadPlan" = field(default_factory=Counter)

    # used by `online_process_loader` to buffer args and tensors until ready to load
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # non-persistent buffer names captured with `kernel_tensors`, so buffer
    # persistence survives `_non_persistent_buffers_set` being mutated during reload
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)

    # Set once a layer publishes, which happens before the transaction ends.
    applied: bool = False

    def reset(self):
        expected_loads = self.expected_loads
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata, restore_device=self.restore_device
        )
        # The contract outlives a single transaction; only progress resets.
        self.expected_loads = expected_loads

    def can_load(self) -> bool:
        return self.active

    def received_any(self) -> bool:
        return bool(self.loaded_weights)

    def is_complete(self) -> bool:
        """Whether every expected application has arrived, which is never true
        without a contract, so such a layer defers to finalization."""
        return bool(self.expected_loads) and all(
            self.observed_loads[key] >= count
            for key, count in self.expected_loads.items()
        )
