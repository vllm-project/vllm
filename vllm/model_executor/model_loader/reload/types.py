# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from inspect import BoundArguments

import torch

__all__ = ["LayerTensors", "LayerReloadingInfo", "RecordedCopies"]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


class RecordedCopies(list[tuple[torch.Size, tuple[int, ...], int, torch.Tensor]]):
    """The copies a loader made into its parameter, buffered instead of its
    arguments: (size, stride, offset into the parameter, copied source)."""

    def replay(self, param: torch.Tensor) -> None:
        for size, stride, offset, source in self:
            view = param.data.as_strided(size, stride, param.storage_offset() + offset)
            view.copy_(source)


@dataclass
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
    loaded_weights: list[tuple[str, BoundArguments | RecordedCopies]] = field(
        default_factory=list
    )

    # kernel formatted tensors, copied into by `_layerwise_process` when reloading
    kernel_tensors: LayerTensors | None = None

    # non-persistent buffer names captured with `kernel_tensors`, so buffer
    # persistence survives `_non_persistent_buffers_set` being mutated during reload
    kernel_non_persistent_buffers: set[str] = field(default_factory=set)

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata, restore_device=self.restore_device
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
