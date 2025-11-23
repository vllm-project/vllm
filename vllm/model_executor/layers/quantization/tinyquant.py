# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TinyQuant: Fallback quantization method for unsupported formats.
This method is used when no specific quantization implementation is found.
"""

from typing import Any, Optional

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization import QuantizationMethods

from tinyquant.quantizer import get_quantizer

logger = init_logger(__name__)


class TinyQuantConfig(QuantizationConfig):
    """Config class for TinyQuant - fallback quantization method."""

    def __init__(
        self,
        quant_method: str,
        original_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.quant_method = quant_method
        self.original_config = original_config or {}
        
        self.quantizer = get_quantizer(quant_method)
        if not self.quantizer:
            raise ValueError(f"Unknown quantization method: {quant_method}")
        logger.info(
            f"TinyQuant: Successfully loaded quantizer for method '{quant_method}'"
        )

    def __repr__(self) -> str:
        return f"TinyQuantConfig(quant_method={self.quant_method})"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "tinyquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # Support common dtypes
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # No specific GPU requirement for fallback
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json", "quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TinyQuantConfig":
        """Create config from HF quantization_config.
        
        Этот метод автоматически извлекает метод квантизации из конфига
        и передает его в конструктор вместе с полным конфигом.
        """
        quant_method = config.get("quant_method", "unknown")
        
        logger.info(
            f"TinyQuantConfig.from_config: Creating config for method '{quant_method}'"
        )
        
        return cls(quant_method=quant_method, original_config=config)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        """Get quantization method for the layer."""
        from vllm.model_executor.layers.linear import LinearBase
        
        if isinstance(layer, LinearBase):
            if self.quantizer is not None:
                # Есть квантайзер - используем TinyQuantLinearMethod
                logger.info_once(
                    f"TinyQuant: Using quantizer '{self.quant_method}' for layers"
                )
                return TinyQuantLinearMethod(self)
            else:
                # Нет квантайзера - используем unquantized fallback
                logger.warning_once(
                    f"TinyQuant: No quantizer for '{self.quant_method}', using unquantized weights"
                )
                return None
        return None


class TinyQuantLinearMethod(LinearMethodBase):
    """Linear method for TinyQuant.
    
    Заглушка, которая использует unquantized веса.
    TODO: Реализовать реальную квантизацию с использованием self.quant_config.quantizer
    """
    
    def __init__(self, quant_config: TinyQuantConfig):
        self.quant_config = quant_config
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Создает веса для слоя (пока просто unquantized).
        
        TODO: Создать квантизованные веса используя self.quant_config.quantizer
        """
        from vllm.model_executor.layers.linear import (
            ModelWeightParameter,
            set_weight_attrs,
        )
        
        # Заглушка: создаем обычные unquantized веса
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        
        logger.info_once(
            f"TinyQuantLinearMethod: Created unquantized weights (placeholder). "
            f"Quantizer '{self.quant_config.quant_method}' available but not integrated yet."
        )
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Обработка весов после загрузки.
        
        TODO: Применить квантизацию используя self.quant_config.quantizer.quantize()
        """
        from vllm.platforms import current_platform
        
        # Заглушка: обработка как для unquantized
        if current_platform.is_cpu():
            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
            dispatch_cpu_unquantized_gemm(layer, remove_weight=True)
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass (пока просто обычный matmul).
        
        TODO: Использовать self.quant_config.quantizer.forward() для работы с квантизованными весами
        """
        from vllm.model_executor.layers.linear import dispatch_unquantized_gemm
        
        # Заглушка: используем обычный unquantized gemm
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
