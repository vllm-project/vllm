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

    # The set of parameter keys a correct load actually consumes, OBSERVED at
    # the first load (record_load_consumption), not predicted from layer
    # structure. This is the ground-truth "required" for completion: tensors
    # never routed through a loader (EP bookkeeping, shared copies) are absent
    # by construction, so no SKIP_TENSORS-style hand-maintained exclusion is
    # needed. It is a BASELINE -- survives reset(), like restore_metadata.
    required_keys: "set[str] | None" = None

    # Keys received during the current reload, accumulated by the online
    # loader wrapper. Transient: cleared by reset() each reload. Completion
    # by set reconciliation is received_keys >= required_keys.
    received_keys: set[str] = field(default_factory=set)

    def reset(self):
        # required_keys is the observed baseline and must survive across
        # reloads, like restore_metadata; everything else is per-reload state.
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata,
            restore_device=self.restore_device,
            required_keys=self.required_keys,
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
