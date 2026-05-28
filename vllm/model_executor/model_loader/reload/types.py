# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass, field
from inspect import BoundArguments

import torch

__all__ = ["LayerTensors", "LayerReloadingInfo"]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


@dataclass
class LayerReloadingInfo:
    # model format metadata, recorded by `record_metadata_for_reloading`
    restore_metadata: LayerTensors

    # device to materialize layers with, recorded by `record_metadata_for_reloading`
    restore_device: torch.device

    # track how many elements are ready for loading, used by `online_process_loader`
    load_numel: int = 0
    load_numel_total: int | None = None

    # used by `online_process_loader` to buffer args and tensors until ready to load
    loaded_weights: list[tuple[str, BoundArguments]] = field(default_factory=list)

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # Optional callback fired at the top of `_layerwise_process`, before
    # buffered loaders are replayed onto materialized params. Lets a weight
    # transfer engine (e.g. the sharded RDT engine) prefetch slice tensors
    # in one batched RPC per layer.
    pre_replay_hook: Callable[["LayerReloadingInfo"], None] | None = None

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata, restore_device=self.restore_device
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
