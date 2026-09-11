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

    def _get_lora_shard_buffers(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        raise NotImplementedError(
            f"Local LoRA shards are unsupported by {type(self).__name__}"
        )

    def get_lora_shard_shapes(
        self, rank: int
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        """Describe already partitioned factors in runtime packed order."""
        shapes = []
        for a, b in self._get_lora_shard_buffers(0):
            if not 0 < rank <= min(a.shape[-2], b.shape[-1]):
                raise ValueError("Local LoRA rank exceeds the allocated capacity")
            shapes.append(
                (
                    tuple(a.shape[:-2]) + (rank, a.shape[-1]),
                    tuple(b.shape[:-1]) + (rank,),
                )
            )
        return tuple(shapes)

    def validate_lora_shard(
        self,
        rank: int,
        lora_a: list[torch.Tensor],
        lora_b: list[torch.Tensor],
    ) -> None:
        shapes = self.get_lora_shard_shapes(rank)
        if len(lora_a) != len(shapes) or len(lora_b) != len(shapes):
            raise ValueError("Local LoRA factors must match every packed slice")
        buffers = self._get_lora_shard_buffers(0)
        owned_storage = {
            (tensor.device, tensor.untyped_storage().data_ptr())
            for pair in buffers
            for tensor in pair
        }
        for (a, b), (a_shape, b_shape), (a_buffer, b_buffer) in zip(
            zip(lora_a, lora_b), shapes, buffers
        ):
            for tensor in (a, b):
                if tensor.is_meta or tensor.layout != torch.strided:
                    raise ValueError(
                        "Local LoRA factors must be materialized dense tensors"
                    )
                if (
                    tensor.device,
                    tensor.untyped_storage().data_ptr(),
                ) in owned_storage:
                    raise ValueError(
                        "Local LoRA factors must not alias runtime buffers"
                    )
            if tuple(a.shape) != a_shape or tuple(b.shape) != b_shape:
                raise ValueError(
                    f"Local LoRA shape mismatch: expected {a_shape}, {b_shape}; "
                    f"received {tuple(a.shape)}, {tuple(b.shape)}"
                )
            if a.dtype != a_buffer.dtype or b.dtype != b_buffer.dtype:
                raise ValueError("Local LoRA dtype must match the runtime buffers")

    def set_lora_shard(
        self,
        index: int,
        rank: int,
        lora_a: list[torch.Tensor],
        lora_b: list[torch.Tensor],
    ) -> None:
        """Copy pre-scaled local factors without TP or EP slicing."""
        if index < 0:
            raise ValueError("Local LoRA slot index must be nonnegative")
        self.validate_lora_shard(rank, lora_a, lora_b)
        buffers = self._get_lora_shard_buffers(index)
        self.reset_lora(index)
        for a, b, (a_buffer, b_buffer) in zip(lora_a, lora_b, buffers):
            a_buffer[..., :rank, :].copy_(a, non_blocking=True)
            b_buffer[..., :rank].copy_(b, non_blocking=True)

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
