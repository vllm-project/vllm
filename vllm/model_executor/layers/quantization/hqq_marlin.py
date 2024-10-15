from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch

import torch.nn.functional as F
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

big_printing_counter = 0

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
        if isinstance(layer, LinearBase):
            return HQQMarlinMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
    

class HQQMarlinMethod(LinearMethodBase):
    """Linear method for HQQ Marlin.
    """

    global_print_ctr = 0
    
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
        self.output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition

        self.input_size_per_partition = input_size_per_partition
        
        weight_loader = extra_weight_attrs.get("weight_loader")

        # print("WEIGHT LOADER:", weight_loader)

        scales_and_zp_size = input_size_per_partition // self.quant_config.group_size

        group_in_tensor_size = (self.output_size_per_partition * self.input_size_per_partition) // self.quant_config.group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                self.output_size_per_partition,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            # data=torch.empty(
            #     group_in_tensor_size // 2,
            #     self.quant_config.group_size,
            #     dtype=torch.uint8,
            # ),
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=1,#self.quant_config.pack_factor,
            weight_loader=weight_loader)
        
        zeros = GroupQuantScaleParameter(
            data=torch.empty(
                self.output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            # data=torch.empty(
            #     group_in_tensor_size,
            #     1,
            #     dtype=params_dtype,
            # ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                self.output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            # data=torch.empty(
            #     group_in_tensor_size,
            #     1,
            #     dtype=params_dtype,
            # ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)

        # print("qweight size:", qweight.shape)
        
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("zeros", zeros)
        layer.register_parameter("scales", scales)

        # self.kernel = '.' #TODO


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # torch.set_printoptions(profile="full")
        # print("layer qweight:", layer.qweight.shape)
        # print(layer.qweight.data.transpose(1, 0)[0])
        # # self.kernel.process_weights_after_loading(layer)
        # raise ValueError("stop")
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("input size:", x.shape)
        # (layer.unpack() - meta['zero'])*meta['scale]).reshape(meta['shape'])

        ## this is unpack function copied from hqq repo
        def unpack_4bit_u8(W_q: torch.Tensor, dtype=torch.uint8) ->torch.Tensor:  # uint8/2 > uint8
            step = W_q.shape[0]
            tmp = torch.empty([2 * step, W_q.shape[1]], dtype=dtype, device=W_q.device)

            tmp[:step] = (W_q & 0b11110000) >> 4
            tmp[step:] = W_q & 0b00001111

            return tmp
        ##

        # lowbits = torch.full((layer.qweight.shape), 15, device=x.device)
        # shifts = torch.full((layer.qweight.shape), 4, device=x.device)
        # unpacked = torch.concat([layer.qweight.bitwise_and(lowbits).to(torch.int8),
        #     layer.qweight.bitwise_right_shift(shifts).to(torch.int8)], dim=0)
        unpacked = layer.qweight.reshape(-1, 64) #unpack_4bit_u8(layer.qweight.reshape(-1, 64), dtype=x.dtype)
        scales = layer.scales.reshape(-1, 1)#.repeat_interleave(64, dim=1)
        zeros = layer.zeros.reshape(-1, 1)#.repeat_interleave(64, dim=1)
        # torch.set_printoptions(sci_mode=False)
        # print("scales:", scales, scales.shape)
        # print("zeros:", zeros, zeros.shape)
        # # print(unpacked.shape, zeros.shape, scales.shape)
        # print("mydeq:", unpacked)
        b = (unpacked - zeros) * scales
        b = b.reshape(self.output_size_per_partition, self.input_size_per_partition)
        # print("unpacked:", unpacked, unpacked.shape)
        if HQQMarlinMethod.global_print_ctr < 1:
            torch.set_printoptions(profile="full")
            torch.set_printoptions(sci_mode=False)
            # print("unpacked size:", layer.qweight.reshape(-1, 64).shape, "->", unpacked.shape)
            # print(layer.qweight.reshape(-1, 64).transpose(1, 0)[0])
            # print(unpacked.transpose(1, 0)[0])
            # print("act wq:", layer.qweight[0])
            # print("scales:", layer.scales.reshape(-1, 1).transpose(1, 0))
            # print("zeros:", layer.zeros.reshape(-1, 1).transpose(1, 0))
            # print("mydeq:", b.transpose(1, 0)[0], b.shape)
            HQQMarlinMethod.global_print_ctr += 1
            # raise ValueError("stop")
        # # print(x.shape, b.shape)
        # print("act wq:", layer.qweight)
        # print(x.dtype, b.dtype)
        return F.linear(x, b, bias)
        # return torch.empty((x.shape[0], self.output_size_per_partition),
        #                    dtype=x.dtype,
        #                    device=x.device)
