# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from inspect import BoundArguments
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.reload_arena import ArenaIdentity

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

    # Per-layer arena storage identities snapshotted at the start of a reload,
    # verified at finalize. VERIFY-ONLY: unlike kernel_tensors, these are never
    # restored-to-meta or copied back. Arena slots already keep their storage
    # stable across the PWAL rebuild by construction (the arena reuses the same
    # buffer); this field only records what those addresses were so finalize
    # can prove none drifted. reset() clearing it each reload is correct -- the
    # arena, not this field, owns the storage, and a fresh snapshot is taken at
    # the next reload's initialize.
    arena_snapshot: "dict[str, ArenaIdentity] | None" = None

    def reset(self):
        self.__init__(  # type: ignore[misc]
            restore_metadata=self.restore_metadata, restore_device=self.restore_device
        )

    def can_load(self) -> bool:
        return self.load_numel_total is not None
