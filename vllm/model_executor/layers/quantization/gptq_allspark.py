# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import Any, Dict, List, Optional, Set

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    check_allspark_supported, check_allspark_supported_dtype_shape)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class GPTQAllSparkConfig(QuantizationConfig):
    """Config class for GPTQ AllSpark
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = 32 // weight_bits
        self.is_sym = is_sym
        self.lm_head_quantized = lm_head_quantized

    def __repr__(self) -> str:
        return (f"GPTQAllSparkConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"lm_head_quantized={self.lm_head_quantized}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_allspark"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Later it will be expanded to 70
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQAllSparkConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, is_sym,
                   lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_allspark_gptq_compatible(hf_quant_cfg)
        is_valid_user_quant = (user_quant is None
                               or user_quant == "gptq_allspark")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to AllSpark GPTQ format. "
                   "Using AllSpark GPTQ kernel.")
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["GPTQAllSparkLinearMethod"]:
        if isinstance(layer, LinearBase) or (isinstance(layer, ParallelLMHead)
                                             and self.lm_head_quantized):
            return GPTQAllSparkLinearMethod(self)
        return None

    @classmethod
    def is_allspark_gptq_compatible(cls, quant_config: Dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        if quant_method != "gptq":
            return False

        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        desc_act = quant_config.get("desc_act")

        if not current_platform.is_cuda():
            return False

        if (num_bits is None or group_size is None or desc_act is None):
            return False

        status, _ = check_allspark_supported(num_bits, group_size, desc_act)
        return status


class GPTQAllSparkLinearMethod(LinearMethodBase):
    """Linear method for GPTQ AllSpark 

    Args:
        quant_config: The GPTQ AllSpark quantization config.
    """
    _kernel_backends_being_used: Set[str] = set()

    def __init__(self, quant_config: GPTQAllSparkConfig):
        self.name = "GPTQAllSparkLinearMethod"
        status, msg = check_allspark_supported(quant_config.weight_bits,
                                               quant_config.group_size,
                                               quant_config.desc_act)
        if not status:
            assert msg is not None
            raise ValueError(msg)
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
    ):
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        self.prefix = extra_weight_attrs.get("prefix")
        assert self.prefix is not None

        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        scales_and_zp_size = input_size // group_size
        scales_and_zp_input_dim = None
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1
                and not self.quant_config.desc_act):
            scales_and_zp_size = input_size_per_partition // group_size
            scales_and_zp_input_dim = 0

        check_allspark_supported_dtype_shape(input_size_per_partition,
                                             output_size_per_partition,
                                             self.quant_config.group_size,
                                             params_dtype)

        if self.name not in self._kernel_backends_being_used:
            logger.info("Using %s for quantization", self.name)
            self._kernel_backends_being_used.add(self.name)

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

        g_idx = RowvLLMParameter(data=torch.tensor(
            [
                i // self.quant_config.group_size
                for i in range(input_size_per_partition)
            ],
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)

        qzeros_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            scales = GroupQuantScaleParameter(output_dim=1,
                                              input_dim=0,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        properties = torch.cuda.get_device_properties(qweight.device.index)
        sm_count = properties.multi_processor_count
        sm_version = properties.major * 10 + properties.minor

        gemm_args = {}
        gemm_args['group_size'] = self.quant_config.group_size
        gemm_args['sm_count'] = sm_count
        gemm_args['sm_version'] = sm_version
        gemm_args['CUBLAS_M_THRESHOLD'] = int(
            os.environ.get('CUBLAS_M_THRESHOLD', 1024))
        gemm_args['has_zp'] = not self.quant_config.is_sym
        gemm_args['n32k16_reorder'] = False
        gemm_args['n'] = output_size_per_partition
        layer.gemm_args = gemm_args

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process the parameters in gptq format into the specific
        # format required by allspark kernel
        # unpack qweight to K x N uint8
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.qweight.data = layer.qweight.data.t().contiguous().view(
            dtype=torch.uint8)
        layer.qweight.data = layer.qweight.data.t().contiguous()

        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        # unpack qzeros and convert qzeros to params_dtype
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.qzeros.data = layer.qzeros.data.view(dtype=torch.uint8).to(
            layer.scales.data.dtype)
        num_bits = self.quant_config.weight_bits
        layer.qzeros.data -= 2**(num_bits - 1) - 1

        layer.scales = Parameter(layer.scales.data, requires_grad=False)
        # reorder KN weight as N32K16 format for Ampere A16W8
        gemm_args = layer.gemm_args
        if (self.quant_config.weight_bits == 8
                and gemm_args['sm_version'] >= 80
                and gemm_args['sm_version'] < 90):
            self.reorder_weights_as_N32K16(layer)
            gemm_args['n32k16_reorder'] = True

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        gemm_args = layer.gemm_args
        out_shape = x.shape[:-1] + (gemm_args['n'], )
        reshaped_x = x.reshape(-1, x.shape[-1])

        weight_name = self.prefix if self.prefix is not None else ""
        if re.search(r'\.\d+\.', weight_name):
            weight_name_pattern = re.sub(r'\.\d+\.', '.', weight_name, count=1)
        else:
            weight_name_pattern = weight_name

        if self.quant_config.weight_bits == 8:
            output = ops.allspark_a16w8_gemm(
                reshaped_x, layer.qweight, layer.scales, layer.qzeros,
                gemm_args['n'], gemm_args['group_size'], gemm_args['sm_count'],
                gemm_args['sm_version'], gemm_args['CUBLAS_M_THRESHOLD'],
                gemm_args['has_zp'], gemm_args['n32k16_reorder'],
                weight_name_pattern)
        else:
            raise RuntimeError(
                "AllSpark now only supports 8bit quantization"
                f"but got weight_bits = {self.quant_config.weight_bits}")

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)

    def reorder_weights_as_N32K16(self, layer: torch.nn.Module):
        qweight_data, weight_scales_data, zero_point_data = \
            ops.gptq_allspark_rearrange_weight(
            layer.qweight, layer.scales, layer.qzeros,
            layer.gemm_args['has_zp'])
        layer.qweight.data = qweight_data
        layer.scales.data = weight_scales_data
        layer.qzeros = zero_point_data
