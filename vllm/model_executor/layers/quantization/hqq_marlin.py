from typing import Any, Dict, List, Optional, Tuple

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
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           HQQQweightParameter,
                                           HQQZeroScaleParameter)
from vllm.model_executor.utils import set_weight_attrs
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
        self.pack_factor = 8 // weight_bits  # packed into uint8
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
        wq_params = (config["quant_config"]["weight_quant_params"])
        weight_bits = cls.get_from_keys(wq_params, ["nbits"])
        group_size = cls.get_from_keys(wq_params, ["group_size"])
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


# Empty HQQ parameter, will be ignored during loading
class HQQEmptyParameter(BasevLLMParameter):

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        pass

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        pass

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        pass


def error_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    raise ValueError("No loader provided for HQQ parameter!")


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

        weight_loader = extra_weight_attrs.get("weight_loader", error_loader)

        self.scales_and_zp_size = (input_size_per_partition //
                                   self.quant_config.group_size)

        # Quantized weights
        qweight = HQQQweightParameter(
            data=torch.empty(
                self.output_size_per_partition //
                self.quant_config.pack_factor,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        set_weight_attrs(qweight, {
            "is_hqq_weight": True,
            "shard_offsets:": [],
        })

        zeros = HQQZeroScaleParameter(data=torch.empty(
            self.output_size_per_partition,
            self.scales_and_zp_size,
            dtype=params_dtype,
        ),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        scales = HQQZeroScaleParameter(data=torch.empty(
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

        # Ignore extra parameters in the HQQ model.
        # To be added as needed.
        ignore_parameters = ("axis", "channel_wise", "compute_dtype",
                             "encoded_state_dict", "group_size", "nbits",
                             "offload_meta", "optimize", "packing",
                             "quant_scale", "quant_zero", "round_zero",
                             "shape", "stores_quant_config",
                             "unpack_view_dtype", "view_as_float")
        for name in ignore_parameters:
            layer.register_parameter(
                name,
                HQQEmptyParameter(data=torch.empty(0),
                                  weight_loader=weight_loader))

    # Unpack weights from the HQQ format and repack them to GPTQ -> Marlin
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        dev = layer.W_q.device

        # unpack function from https://github.com/mobiusml/hqq
        def unpack_4bit_u8(
            W_q: torch.Tensor,
            shard_offsets: List[Tuple[int, int]],
        ) -> torch.Tensor:  # uint8/2 > uint8
            dtype = torch.uint8
            tmp = torch.empty([2 * W_q.shape[0], W_q.shape[1]],
                              dtype=dtype,
                              device=W_q.device)
            for (offset, size) in shard_offsets:
                tmp_offset = 2 * offset
                tmp[tmp_offset:tmp_offset +
                    size] = (W_q[offset:offset + size] & 0b11110000) >> 4
                tmp[tmp_offset + size:tmp_offset +
                    2 * size] = (W_q[offset:offset + size] & 0b00001111)
            return tmp

        # Unpack from 4-bit to 8-bit
        shard_offsets = getattr(layer.W_q, "shard_offsets", [])
        qweight_t = unpack_4bit_u8(layer.W_q, shard_offsets).transpose(1, 0)

        # Repack to GPTQ
        gptq_w_q = gptq_pack(qweight_t, 4, self.input_size_per_partition,
                             self.output_size_per_partition)

        # Repack to Marlin
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

        scales = layer.marlin_scales
        zeros = layer.marlin_zeros
        orig_type = x.dtype

        if orig_type != torch.float16:
            x = x.to(torch.float16)
            scales = scales.to(torch.float16)
            zeros = zeros.to(torch.float16)

        marlin_out = ops.gptq_marlin_gemm(
            x,
            layer.marlin_qweight,
            scales,
            zeros,
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

        if orig_type != torch.float16:
            return marlin_out.to(orig_type)
        else:
            return marlin_out
