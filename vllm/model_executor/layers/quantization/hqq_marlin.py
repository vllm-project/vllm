from typing import Any, Dict, List, Optional

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
    marlin_make_empty_g_idx, marlin_permute_scales)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    MarlinWorkspace)
from vllm.model_executor.layers.quantization.utils.quant_utils import gptq_pack
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           ModelWeightParameter)
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
        return "hqq"

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
        weight_bits = cls.get_from_keys(config, ["nbits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        return cls(weight_bits, group_size)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        #TODO
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["HQQMarlinMethod"]:
        if isinstance(layer, LinearBase):
            return HQQMarlinMethod(self)
        return None

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
        self.output_size_per_partition = sum(output_partition_sizes)

        self.input_size_per_partition = input_size_per_partition

        weight_loader = extra_weight_attrs.get("weight_loader")

        self.scales_and_zp_size = (input_size_per_partition //
                                   self.quant_config.group_size)

        # Quantized weights
        qweight = ModelWeightParameter(data=torch.empty(
            self.output_size_per_partition,
            input_size_per_partition,
            dtype=torch.uint8,
        ),
                                       input_dim=1,
                                       output_dim=0,
                                       weight_loader=weight_loader)

        zeros = GroupQuantScaleParameter(data=torch.empty(
            self.output_size_per_partition,
            self.scales_and_zp_size,
            dtype=params_dtype,
        ),
                                         input_dim=1,
                                         output_dim=0,
                                         weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            self.output_size_per_partition,
            self.scales_and_zp_size,
            dtype=params_dtype,
        ),
                                          input_dim=1,
                                          output_dim=0,
                                          weight_loader=weight_loader)

        layer.register_parameter("W_q", qweight)
        layer.register_parameter("zero", zeros)
        layer.register_parameter("scale", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dev = layer.W_q.device
        qweight_t = layer.W_q.transpose(1, 0)

        gptq_w_q = gptq_pack(qweight_t, 4, self.input_size_per_partition,
                             self.output_size_per_partition)

        sort_indices = torch.empty(0, dtype=torch.int, device=gptq_w_q.device)
        marlin_w_q = ops.gptq_marlin_repack(
            gptq_w_q,
            sort_indices,
            self.input_size_per_partition,
            self.output_size_per_partition,
            4,
        ).to(dev)
        marlin_s = marlin_permute_scales(layer.scale.transpose(1, 0),
                                         self.input_size_per_partition,
                                         self.output_size_per_partition,
                                         self.quant_config.group_size).to(dev)
        marlin_zp = marlin_permute_scales(layer.zero.transpose(1, 0),
                                          self.input_size_per_partition,
                                          self.output_size_per_partition,
                                          self.quant_config.group_size).to(dev)

        layer.g_idx = marlin_make_empty_g_idx(dev)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(dev)

        layer.marlin_qweight = marlin_w_q
        layer.marlin_zeros = marlin_zp
        layer.marlin_scales = marlin_s

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        workspace = MarlinWorkspace(self.output_size_per_partition,
                                    GPTQ_MARLIN_MIN_THREAD_N,
                                    GPTQ_MARLIN_MAX_PARALLEL)

        marlin_out = ops.gptq_marlin_gemm(
            x,
            layer.marlin_qweight,
            layer.marlin_scales,
            layer.marlin_zeros,
            layer.g_idx,
            layer.g_idx_sort_indices,
            workspace.scratch,
            scalar_types.uint4,
            x.shape[0],
            self.output_size_per_partition,
            self.input_size_per_partition,
            True,  # is_k_full
            True,  # has_zp
            False,  # use 32-bit reduce
            True,  # use float zp
        )

        if bias is not None:
            marlin_out.add_(bias)

        return marlin_out
