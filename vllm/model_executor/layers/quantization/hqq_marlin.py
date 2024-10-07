from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kernels import (
    MPLinearLayerConfig, choose_mp_linear_kernel)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported, marlin_moe_permute_scales,
    marlin_repeat_scales_on_all_ranks, verify_marlin_supported)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

class HQQMarlinConfig(QuantizationConfig):
    """Config class for HQQ Marlin"""

     # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
    ) -> None:
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.quant_type = self.TYPE_MAP[(weight_bits)]

    def __repr__(self) -> str:
        return (f"HQQMarlinConfig(quant_type={self.quant_type}, "
                f"group_size={self.group_size})")

    @classmethod
    def get_name(cls) -> str:
        return "hqq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HQQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        #TODO
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "HQQMarlinMethod":
        return HQQMarlinMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []
    

class HQQMarlinMethod(LinearMethodBase):
    """Linear method for HQQ Marlin.
    """
    
    def __init__(
        self,
        quant_config: HQQMarlinConfig,
    ):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")

        scales_and_zp_size = input_size_per_partition // self.quant_config.group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)
        
        zeros_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        scales_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }

        zeros = PackedvLLMParameter(
                    input_dim=0,
                    output_dim=1,
                    packed_dim=1,
                    packed_factor=self.quant_config.pack_factor,
                    **zeros_args)

        scales = PackedvLLMParameter(
                    input_dim=0,
                    output_dim=1,
                    packed_dim=1,
                    packed_factor=self.quant_config.pack_factor,
                    **scales_args)
        
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("zeros", zeros)
        layer.register_parameter("scales", scales)

        # self.kernel = '.' #TODO


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        print("TODO")
        # self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.empty((0), dtype=x.dtype, device=x.device)
        # return self.kernel.apply_weights(layer, x, bias)
