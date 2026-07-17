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

    # non-persistent buffer names captured with `kernel_tensors`, so buffer
    # persistence survives `_non_persistent_buffers_set` being mutated during reload
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)

    # CUDA tensors reachable from the layer but NOT registered as parameters
    # or buffers (e.g. workspace, sort-indices, derived MLA weights, CUTLASS
    # stride descriptors).  Populated by `initialize_layerwise_reload` so that
    # `copy_back_extra_tensors` can preserve their device addresses across
    # reload, keeping captured CUDA-graph pointers valid.
    # Each entry is (dotted_path, old_tensor).
    extra_tensor_slots: list[tuple[str, torch.Tensor]] = field(
        default_factory=list)

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata, restore_device=self.restore_device
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
