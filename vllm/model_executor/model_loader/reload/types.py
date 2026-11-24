# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import BoundArguments
from typing import Protocol

import torch

__all__ = [
    "LayerTensors",
    "LayerReloadingInfo",
    "WeightReloadValidator",
    "WeightStream",
]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


class WeightStream(Protocol):
    """Loads the weights of a layer as they arrive, in place of the layerwise
    buffering, e.g. to quantize MoE experts one at a time. Created by the layer's
    quant method `create_weight_stream`."""

    @property
    def has_loaded(self) -> bool:
        """Whether any weight was loaded."""
        ...

    def load(
        self,
        layer: torch.nn.Module,
        param_name: str,
        loader: Callable,
        args: BoundArguments,
    ) -> object:
        """Load a weight of `param_name` with its (unwrapped) `loader`."""
        ...

    def is_complete(self, layer: torch.nn.Module) -> bool:
        """Whether all weights are loaded, so the layer can be processed."""
        ...


class WeightReloadValidator(ABC):
    """Opt-in validation for modules with weight reload restrictions."""

    @abstractmethod
    def validate_weight_reload(self) -> None:
        """Raise if reloading is unsupported, before any weights are written."""


@dataclass(slots=True)
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # track how many elements are ready for loading, used by `online_process_loader`
    load_numel: int = 0
    load_numel_total: int | None = None
    defer_processing: bool = False

    # used by `online_process_loader` to buffer args and tensors until ready to load
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    # loads the weights instead of `loaded_weights`, if the quant method has one
    weight_stream: WeightStream | None = None

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # non-persistent buffer names captured with `kernel_tensors`, so buffer
    # persistence survives `_non_persistent_buffers_set` being mutated during reload
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
