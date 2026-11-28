# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config.lora import LoRAConfig

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase


class BaseLayerWithLoRA(nn.Module):
    def __getattr__(self, name):
        d = self.__dict__
        if name in d.get("_parameters", ()):
            return d["_parameters"][name]
        if name in d.get("_buffers", ()):
            return d["_buffers"][name]
        if name in d.get("_modules", ()):
            return d["_modules"][name]
        # Forward public misses to ``base_layer``; private names are framework
        # bookkeeping and must stay local.
        if not name.startswith("_"):
            base_layer = d.get("_modules", {}).get("base_layer")
            if base_layer is not None:
                return getattr(base_layer, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        """Load checkpoint weights into the wrapped base layer."""
        base_load_weights = getattr(self.base_layer, "load_weights", None)
        if callable(base_load_weights):
            return base_load_weights(weights)

        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self.base_layer).load_weights(weights)

    @overload
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_a(
        self, lora_a: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora a if splitting for tensor parallelism."""
        ...

    @overload
    def slice_lora_b(
        self, lora_b: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_b(
        self, lora_b: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]:
        """Slice lora b if splitting with tensor parallelism."""
        ...

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PreTrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def reset_lora(self, index: int):
        """Resets the lora weights at index back to 0."""
        ...

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        """Overwrites lora tensors at index."""
        ...

    def set_mapping(
        self,
        punica_wrapper,
    ):
        self.punica_wrapper: PunicaWrapperBase = punica_wrapper

    def _init_lora_rank_buffers(
        self,
        n_slices: int,
        max_loras: int,
        device: torch.device,
        *,
        track_a: bool = True,
    ) -> None:
        """Allocate per-slice rank tensors and CPU sub-max tracking flags."""
        if track_a:
            self.lora_a_ranks = torch.zeros(
                (n_slices, max_loras), dtype=torch.int32, device=device
            )
            self._a_submax_slots = [False] * max_loras
            self._has_submax_a_ranks_cpu = torch.zeros(
                1, dtype=torch.bool, device="cpu"
            )
        self.lora_b_ranks = torch.zeros(
            (n_slices, max_loras), dtype=torch.int32, device=device
        )
        self._b_submax_slots = [False] * max_loras
        self._has_submax_b_ranks_cpu = torch.zeros(1, dtype=torch.bool, device="cpu")

    def _clear_lora_rank_slot(self, index: int) -> None:
        """Reset stored ranks and sub-max tracking state for one adapter slot."""
        if hasattr(self, "lora_a_ranks"):
            self.lora_a_ranks[:, index] = 0
            self._a_submax_slots[index] = False
            self._has_submax_a_ranks_cpu[0] = any(self._a_submax_slots)
        if hasattr(self, "lora_b_ranks"):
            self.lora_b_ranks[:, index] = 0
            self._b_submax_slots[index] = False
            self._has_submax_b_ranks_cpu[0] = any(self._b_submax_slots)

    def clear_lora_ranks(self, index: int) -> None:
        """Clear rank tracking for a deactivated slot without zeroing weights."""
        if hasattr(self, "_b_submax_slots"):
            self._clear_lora_rank_slot(index)

    @staticmethod
    def _set_slot_rank(
        ranks_gpu: torch.Tensor,
        submax_slots: list[bool],
        has_submax_cpu: torch.Tensor,
        slot_index: int,
        slice_index: int,
        rank: int | None,
        full_rank: int,
    ) -> None:
        """Record a slice rank and mark the slot sub-max if unadapted or < full."""
        if rank is not None:
            ranks_gpu[slice_index, slot_index] = rank
        if rank is None or rank < full_rank:
            submax_slots[slot_index] = True
            has_submax_cpu[0] = True

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PreTrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError
