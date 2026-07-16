# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase


class BaseLayerWithLoRA(nn.Module):
    def named_modules(
        self,
        memo: set[nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, nn.Module]]:
        """Make the LoRA wrapper transparent in the module tree.

        LoRA wrapping moves a layer's parameters under ``base_layer``
        (e.g. ``qkv_proj.weight`` -> ``qkv_proj.base_layer.weight``).
        Checkpoint files and model-specific ``load_weights()`` methods
        use the original (un-prefixed) names.

        This override flattens ``base_layer`` out of the hierarchy so
        that :meth:`named_parameters` and :meth:`named_buffers` return
        the original names, making weight loading work transparently.
        """
        if memo is None:
            memo = set()
        if remove_duplicate and self in memo:
            return
        memo.add(self)
        yield prefix, self

        base: nn.Module | None = getattr(self, "base_layer", None)
        if base is not None and not (remove_duplicate and base in memo):
            memo.add(base)
            yield prefix, base
            for name, child in base._modules.items():
                if child is not None:
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    yield from child.named_modules(memo, child_prefix, remove_duplicate)

        for name, child in self._modules.items():
            if name == "base_layer" or child is None:
                continue
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(memo, child_prefix, remove_duplicate)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        """Forward checkpoint weights to the unwrapped base layer.

        Delegate to the base layer's own ``load_weights`` when it defines
        one — fused layers (e.g. ``MergedColumnParallelLinear``) read the
        shard id off the loaded tensor there, which an outside-in walk
        with ``AutoWeightsLoader`` would bypass. Fall back to loading the
        base layer's parameters directly otherwise.
        """
        base_load_weights = getattr(self.base_layer, "load_weights", None)
        if callable(base_load_weights):
            return base_load_weights(weights)

        from vllm.model_executor.models.utils import AutoWeightsLoader

        loader = AutoWeightsLoader(self.base_layer)
        return loader.load_weights(weights)

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
        model_config: PretrainedConfig | None = None,
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

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool:
        """Returns True if the layer can be replaced by this LoRA layer."""
        raise NotImplementedError
