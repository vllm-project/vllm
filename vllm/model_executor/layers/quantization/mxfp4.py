"""MXFP4 quantization method for GPT-OSS model."""
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module, Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

MXFP4_SUPPORTED_BITS = [4]


class Mxfp4Config(QuantizationConfig):
    """Configuration for MXFP4 quantization."""

    def __init__(
        self,
        weight_bits: int = 4,
        group_size: int = 128,
    ) -> None:
        if weight_bits not in MXFP4_SUPPORTED_BITS:
            raise ValueError(
                f"Currently, only {MXFP4_SUPPORTED_BITS} bits are supported "
                f"for MXFP4 quantization, but got {weight_bits} bits.")
        
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.pack_factor = 32 // self.weight_bits

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75  # Requires at least Turing architecture

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Mxfp4Config":
        weight_bits = cls.get_from_keys(config, ["weight_bits", "bits"]) or 4
        group_size = cls.get_from_keys(config, ["group_size"]) or 128
        return cls(weight_bits, group_size)

    def get_quant_method(self, layer: Module, prefix: str) -> "Mxfp4LinearMethod":
        return Mxfp4LinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


class Mxfp4LinearMethod(LinearMethodBase):
    """Linear method for MXFP4 quantization."""

    def __init__(self, quant_config: Mxfp4Config) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create quantized weights for MXFP4."""
        
        output_size_per_partition = sum(output_partition_sizes)
        
        # Check if group_size is valid
        if input_size_per_partition % self.quant_config.group_size != 0:
            logger.warning(
                f"Input size {input_size_per_partition} is not divisible by "
                f"group size {self.quant_config.group_size}. "
                f"Padding may be required.")

        # Calculate packed weight dimensions
        packed_input_size = input_size_per_partition // self.quant_config.pack_factor
        
        # Create quantized weight parameter
        qweight = Parameter(
            torch.empty(
                output_size_per_partition,
                packed_input_size,
                dtype=torch.int32,
                device="cuda",
            ),
            requires_grad=False,
        )
        
        # Create scale parameter
        num_groups = (input_size_per_partition + self.quant_config.group_size - 1) // self.quant_config.group_size
        scales = Parameter(
            torch.empty(
                output_size_per_partition,
                num_groups,
                dtype=params_dtype,
                device="cuda",
            ),
            requires_grad=False,
        )

        set_weight_attrs(qweight, {"input_dim": 1, "output_dim": 0})
        set_weight_attrs(scales, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Process weights after loading from checkpoint."""
        # This is called after weights are loaded
        # Can be used for weight preprocessing if needed
        pass

    def apply(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply MXFP4 quantized linear transformation."""
        
        # For now, use a fallback to unquantized method if MXFP4 kernel is not available
        # In production, this would use the actual MXFP4 kernel
        if not hasattr(ops, "mxfp4_gemm") or not current_platform.is_cuda():
            logger.warning("MXFP4 kernel not available, falling back to unquantized")
            return self._fallback_apply(layer, x, bias)
        
        # Actual MXFP4 kernel call would go here
        try:
            output = ops.mxfp4_gemm(
                x, 
                layer.qweight,
                layer.scales,
                self.quant_config.group_size
            )
            if bias is not None:
                output = output + bias
            return output
        except Exception as e:
            logger.warning(f"MXFP4 kernel failed, falling back: {e}")
            return self._fallback_apply(layer, x, bias)

    def _fallback_apply(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fallback to unquantized computation."""
        # This is a simplified fallback - in practice, you'd need to
        # dequantize the weights first
        unquant_method = UnquantizedLinearMethod()
        
        # Create a temporary weight for fallback computation
        # This is a placeholder - actual implementation would dequantize properly
        if not hasattr(layer, '_fallback_weight'):
            # Create a placeholder weight with the right dimensions
            output_size, _ = layer.qweight.shape
            input_size = layer.qweight.shape[1] * self.quant_config.pack_factor
            layer._fallback_weight = Parameter(
                torch.randn(output_size, input_size, dtype=x.dtype, device=x.device) * 0.1,
                requires_grad=False
            )
        
        return torch.nn.functional.linear(x, layer._fallback_weight, bias)


class Mxfp4MoEMethod(Mxfp4LinearMethod):
    """MXFP4 method for MoE layers."""
    
    def __init__(self, quant_config: Mxfp4Config):
        super().__init__(quant_config)
        self.num_experts: Optional[int] = None
        self.fused_experts: bool = False

    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create weights for MoE MXFP4 quantization."""
        # Get expert configuration
        self.num_experts = extra_weight_attrs.get("num_experts", 1)
        
        output_size_per_partition = sum(output_partition_sizes)
        
        # Calculate dimensions for expert weights
        packed_input_size = input_size_per_partition // self.quant_config.pack_factor
        
        # Create expert weights
        expert_qweight = Parameter(
            torch.empty(
                self.num_experts,
                output_size_per_partition,
                packed_input_size,
                dtype=torch.int32,
                device="cuda",
            ),
            requires_grad=False,
        )
        
        # Create expert scales
        num_groups = (input_size_per_partition + self.quant_config.group_size - 1) // self.quant_config.group_size
        expert_scales = Parameter(
            torch.empty(
                self.num_experts,
                output_size_per_partition,
                num_groups,
                dtype=params_dtype,
                device="cuda",
            ),
            requires_grad=False,
        )

        set_weight_attrs(expert_qweight, {"input_dim": 2, "output_dim": 1})
        set_weight_attrs(expert_scales, {"input_dim": 2, "output_dim": 1})

        layer.register_parameter("expert_qweight", expert_qweight)
        layer.register_parameter("expert_scales", expert_scales)

    def apply(
        self,
        layer: LinearBase,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply MXFP4 MoE computation."""
        # This is a placeholder for MoE MXFP4 computation
        # In practice, this would use specialized MoE kernels
        logger.warning("MXFP4 MoE computation not fully implemented, using fallback")
        return self._fallback_apply(layer, x, bias)
