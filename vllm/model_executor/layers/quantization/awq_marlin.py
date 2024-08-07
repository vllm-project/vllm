from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear, awq_to_marlin_zero_points, check_marlin_supported,
    marlin_make_empty_g_idx, marlin_make_workspace, marlin_permute_scales,
    replace_tensor, verify_marlin_supported, verify_marlin_supports_shape)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class AWQMarlinConfig(QuantizationConfig):
    """Config class for AWQ Marlin"""

    # num_bits -> type
    TYPE_MAP = {
        4: scalar_types.uint4,
        8: scalar_types.uint8,
    }

    def __init__(self, weight_bits: int, group_size: int, has_zp: bool,
                 lm_head_quantized: bool) -> None:
        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.has_zp = has_zp
        self.lm_head_quantized = lm_head_quantized

        if weight_bits not in self.TYPE_MAP:
            raise ValueError(f"Unsupported num_bits = {weight_bits}. "
                             f"Supported num_bits = {self.TYPE_MAP.keys()}")

        self.quant_type = self.TYPE_MAP[weight_bits]

        verify_marlin_supported(self.quant_type,
                                group_size=self.group_size,
                                has_zp=self.has_zp)

    def __repr__(self) -> str:
        return (f"AWQMarlinConfig(quant_type={self.quant_type}, "
                f"group_size={self.group_size}, "
                f"has_zp={self.has_zp}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "awq_marlin"

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
    def from_config(cls, config: Dict[str, Any]) -> "AWQMarlinConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        has_zp = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, has_zp, lm_head_quantized)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_awq_marlin_compatible(hf_quant_cfg)
        is_valid_user_quant = (user_quant is None or user_quant == "marlin"
                               or user_quant == "awq_marlin")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info("Detected that the model can run with awq_marlin"
                        ", however you specified quantization=awq explicitly,"
                        " so forcing awq. Use quantization=awq_marlin for"
                        " faster inference")
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["AWQMarlinLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return AWQMarlinLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        has_zp = quant_config.get("zero_point", None)

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if (num_bits is None or group_size is None or has_zp is None):
            return False

        if num_bits not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(quant_type=cls.TYPE_MAP[num_bits],
                                      group_size=group_size,
                                      has_zp=has_zp,
                                      min_capability=cls.get_min_capability())


class AWQMarlinLinearMethod(LinearMethodBase):
    """Linear method for AWQ Marlin.

    Args:
        quant_config: The AWQ Marlin quantization config.
    """

    def __init__(self, quant_config: AWQMarlinConfig) -> None:
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
        del output_size
        output_size_per_partition = sum(output_partition_sizes)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size)

        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })

        num_groups = input_size_per_partition // group_size

        qzeros = Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros, {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            })

        scales = Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    # TODO: Update this docs
    # Checkpoints are serialized in AutoAWQ format, which is different from the
    # marlin format. This function is called after the weights are loaded.
    # Here, we handle the repacking
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device

        # Allocate marlin workspace
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Repack weights from AWQ format to marlin format.
        marlin_qweight = ops.awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits)
        replace_tensor(layer, "qweight", marlin_qweight)

        # Permute scales from AWQ format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size)
        replace_tensor(layer, "scales", marlin_scales)

        # Permute zero-points from AWQ format to marlin format.
        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits)
        replace_tensor(layer, "qzeros", marlin_zp)

        # Not-used
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_awq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.qzeros,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            quant_type=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias)
