# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step work that linear-attention kernels need outside the forward."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import KVCacheConfig

# A kernel's layers in one KV cache group, as (layer name, layer).
GroupLayers = Sequence[tuple[str, Any]]


class StepLifecycle(Protocol):
    """A kernel's per-step work outside the forward, for all its KV cache groups.
    The kernel decides what runs; the model runner decides when."""

    def record_step(self, attn_metadata: Mapping[str, Any]) -> None:
        """Called once the step's attention metadata is built, before the
        forward. CUDA graph capture's metadata comes through here too."""


class StepLifecycleOwner(Protocol):
    @classmethod
    def make_step_lifecycle(
        cls, layers_by_group: Sequence[GroupLayers]
    ) -> StepLifecycle | None: ...


def make_step_lifecycles(
    kv_cache_config: KVCacheConfig, forward_context: Mapping[str, Any]
) -> list[StepLifecycle]:
    """One lifecycle per owner (`MambaBase.step_lifecycle_owner`) that needs
    one, built from its layers grouped by KV cache group."""
    by_owner: dict[type[StepLifecycleOwner], list[GroupLayers]] = {}
    for group in kv_cache_config.kv_cache_groups:
        in_group: dict[type[StepLifecycleOwner], list[tuple[str, Any]]] = {}
        for name in group.layer_names:
            layer = forward_context.get(name)
            if isinstance(layer, MambaBase) and layer.step_lifecycle_owner:
                in_group.setdefault(layer.step_lifecycle_owner, []).append(
                    (name, layer)
                )
        for owner, layers in in_group.items():
            by_owner.setdefault(owner, []).append(layers)
    lifecycles = [
        owner.make_step_lifecycle(groups) for owner, groups in by_owner.items()
    ]
    return [lifecycle for lifecycle in lifecycles if lifecycle is not None]
