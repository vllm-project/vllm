# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from enum import Enum, auto
from inspect import BoundArguments

import torch
from torch.nn.parameter import is_lazy

__all__ = [
    "LayerReloadMode",
    "LayerReloadPlan",
    "LayerReloadingInfo",
    "LayerTensors",
    "TensorReloadSignature",
]

# encodes both parameters and buffers separately
LayerTensors = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]


class LayerReloadMode(Enum):
    DIRECT = auto()
    RUNTIME_VIEW = auto()
    LAYERWISE = auto()


@dataclass(frozen=True)
class TensorReloadSignature:
    shape: torch.Size | None
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    data_ptr: int | None
    storage_ptr: int | None
    storage_offset: int
    storage_nbytes: int

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorReloadSignature":
        if is_lazy(tensor):
            return cls(
                shape=None,
                stride=(),
                dtype=tensor.dtype,
                device=tensor.device,
                data_ptr=None,
                storage_ptr=None,
                storage_offset=0,
                storage_nbytes=0,
            )
        storage = tensor.untyped_storage()
        return cls(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
            data_ptr=tensor.data_ptr(),
            storage_ptr=storage.data_ptr(),
            storage_offset=tensor.storage_offset(),
            storage_nbytes=storage.nbytes(),
        )


TensorReloadSignatures = tuple[
    dict[str, TensorReloadSignature], dict[str, TensorReloadSignature]
]


@dataclass(frozen=True)
class LayerReloadPlan:
    mode: LayerReloadMode
    runtime_signatures: TensorReloadSignatures


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

    # checkpoint-layout views are bound directly to kernel storage while reloading
    runtime_bound: bool = False

    # reusable strategy and runtime-storage invariants
    reload_plan: LayerReloadPlan | None = None

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
            reload_plan=self.reload_plan,
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
