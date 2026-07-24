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
    # Known LoRA stacked tensor attribute names allocated on GPU.
    # These are plain attributes (not nn.Parameter or registered buffers)
    # that must be explicitly re-zeroed after level-2 sleep/wake.
    _LORA_TENSOR_ATTRS = (
        "lora_a_stacked",
        "lora_b_stacked",
        "lora_embedding_a",
        "lora_embedding_b",
    )

    def named_modules(
        self,
        memo: set[nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterable[tuple[str, nn.Module]]:
        """Make the LoRA wrapper transparent in the module tree.

        LoRA wrapping moves parameters under ``base_layer``
        (e.g. ``qkv_proj.weight`` -> ``qkv_proj.base_layer.weight``).
        This override flattens ``base_layer`` out so that
        ``named_parameters()`` returns the original checkpoint names.
        """
        if memo is None:
            memo = set()
        if remove_duplicate and self in memo:
            return
        memo.add(self)
        yield prefix, self

        base: nn.Module | None = getattr(self, "base_layer", None)
        if base is not None:
            if not (remove_duplicate and base in memo):
                memo.add(base)
                yield prefix, base
                for name, child in base._modules.items():
                    if child is not None:
                        child_prefix = f"{prefix}.{name}" if prefix else name
                        yield from child.named_modules(
                            memo, child_prefix, remove_duplicate
                        )

        for name, child in self._modules.items():
            if name == "base_layer" or child is None:
                continue
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(
                memo, child_prefix, remove_duplicate
            )

    def zero_lora_state(self) -> None:
        """Re-zero LoRA stacked tensors after level-2 sleep/wake.

        These tensors are plain attributes (not nn.Parameter or registered
        buffers). After level-2 sleep, their GPU memory is discarded and
        remapped with undefined contents. reload_weights() only restores
        parameters and buffers, so these must be explicitly re-zeroed.
        """
        for attr_name in self._LORA_TENSOR_ATTRS:
            val = getattr(self, attr_name, None)
            if val is None:
                continue
            if isinstance(val, torch.Tensor) and val.device.type != "meta":
                val.zero_()
            elif isinstance(val, (tuple, list)):
                for t in val:
                    if isinstance(t, torch.Tensor) and t.device.type != "meta":
                        t.zero_()

    def restore_non_parameter_tensors(self) -> None:
        """Restore non-LoRA GPU tensors from their CPU sources after sleep.

        Some LoRA layers hold GPU tensors (e.g. sharded_to_full_mapping_gpu)
        that are neither nn.Parameters nor LoRA stacked tensors. After
        level-2 sleep their GPU memory contains undefined data. This method
        rebuilds them from their CPU-side sources.
        """
        cpu_mapping = getattr(self, "sharded_to_full_mapping", None)
        if cpu_mapping is not None:
            self.sharded_to_full_mapping_gpu = torch.tensor(
                cpu_mapping,
                device=getattr(self, "device", "cuda"),
                dtype=torch.long,
            )

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
