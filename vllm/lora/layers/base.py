# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.triton_utils import HAS_TRITON

if TYPE_CHECKING:
    from vllm.lora.punica_wrapper import PunicaWrapperBase


class BaseLayerWithLoRA(nn.Module):
    _expand_input_uses_lora_dtype = False
    _uses_lora_shrink = True

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
        model_config: PretrainedConfig | None = None,
    ) -> None:
        """Initializes lora matrices."""
        ...

    def register_jit_warmups(
        self,
        *,
        max_tokens: int,
        lora_slots: int,
        output_dtype: torch.dtype,
    ) -> None:
        """Register the standard Punica kernels used by this LoRA layer."""
        if not HAS_TRITON:
            return

        lora_a_weights = getattr(self, "lora_a_stacked", ())
        lora_b_weights = getattr(self, "lora_b_stacked", ())
        if isinstance(lora_a_weights, torch.Tensor):
            lora_a_weights = (lora_a_weights,)
        if isinstance(lora_b_weights, torch.Tensor):
            lora_b_weights = (lora_b_weights,)
        if not lora_a_weights or not lora_b_weights:
            return

        from vllm.lora.ops.triton_ops.lora_expand_op import _LORA_EXPAND_KERNEL
        from vllm.lora.ops.triton_ops.lora_shrink_op import _LORA_SHRINK_KERNEL

        lora_a_weights_3d = tuple(weight.squeeze(1) for weight in lora_a_weights)
        lora_b_weights_3d = tuple(weight.squeeze(1) for weight in lora_b_weights)
        expand_input_dtype = (
            lora_a_weights_3d[0].dtype
            if self._expand_input_uses_lora_dtype
            else torch.float32
        )
        same_stride = (
            len(
                {
                    (
                        weight.shape[1],
                        weight.stride(0),
                        weight.stride(1),
                        weight.stride(2),
                    )
                    for weight in lora_b_weights_3d
                }
            )
            == 1
        )

        if self._uses_lora_shrink:
            _LORA_SHRINK_KERNEL.register_warmup(
                max_tokens=max_tokens,
                max_loras=lora_slots + 1,
                input_dtype=lora_a_weights_3d[0].dtype,
                weight_dtype=lora_a_weights_3d[0].dtype,
                n=lora_a_weights_3d[0].shape[1],
                k=lora_a_weights_3d[0].shape[2],
                lora_d0_stride=lora_a_weights_3d[0].stride(0),
                lora_d1_stride=lora_a_weights_3d[0].stride(1),
                lora_d2_stride=lora_a_weights_3d[0].stride(2),
                slice_num=len(lora_a_weights_3d),
                lora_pointer_table=len(lora_a_weights_3d) > 1,
            )

        for add_inputs in (False, True):
            _LORA_EXPAND_KERNEL.register_warmup(
                max_tokens=max_tokens,
                max_loras=lora_slots + 1,
                input_dtype=expand_input_dtype,
                weight_dtype=lora_b_weights_3d[0].dtype,
                output_dtype=output_dtype,
                n=max(weight.shape[1] for weight in lora_b_weights_3d),
                k=lora_b_weights_3d[0].shape[2],
                lora_d0_stride=lora_b_weights_3d[0].stride(0),
                lora_d1_stride=lora_b_weights_3d[0].stride(1),
                lora_d2_stride=lora_b_weights_3d[0].stride(2),
                output_d0_stride=sum(weight.shape[1] for weight in lora_b_weights_3d),
                output_d1_stride=1,
                add_inputs=add_inputs,
                cast_type=True,
                slice_num=len(lora_b_weights_3d),
                same_stride=same_stride,
                lora_pointer_table=len(lora_b_weights_3d) > 1,
                metadata_table=not same_stride,
            )

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
