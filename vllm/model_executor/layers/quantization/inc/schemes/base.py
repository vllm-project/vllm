# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase
    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from ..inc import INCConfig
    from ..resolver import INCLayerConfig


class INCScheme(ABC):
    """One class per quant type. Single registration point for the factory.

    Each subclass defines:
      - can_handle(): when does this scheme apply?
      - get_linear_method(): required — how to quantize Linear layers
      - get_moe_method(): optional — how to quantize MoE layers
      - get_kvcache_method(): optional — how to quantize KV cache

    Schemes that don't support MoE/KVCache inherit the default raise.
    """

    @staticmethod
    @abstractmethod
    def can_handle(layer_config: "INCLayerConfig") -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_linear_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> "LinearMethodBase":
        raise NotImplementedError

    def get_moe_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> "FusedMoEMethodBase | None":
        """Optional. Override if this scheme supports MoE.
        Default raises NotImplementedError."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support MoE layers. "
            f"Layer config: {layer_config}"
        )

    def get_kvcache_method(
        self,
        config: "INCConfig",
        layer: "torch.nn.Module",
        prefix: str,
        layer_config: "INCLayerConfig",
    ) -> "QuantizationMethods":
        """Optional. Override if this scheme supports KV cache quantization.
        Default raises NotImplementedError."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support KV cache quantization. "
            f"Layer config: {layer_config}"
        )


class INCLinearScheme(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @abstractmethod
    def create_weights(
        self,
        layer: "torch.nn.Module",
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: "torch.dtype",
        **extra_weight_attrs,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: "torch.nn.Module") -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        bias: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        raise NotImplementedError
