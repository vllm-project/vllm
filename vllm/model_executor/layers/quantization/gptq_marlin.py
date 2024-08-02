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
    apply_gptq_marlin_linear, check_marlin_supported, marlin_is_k_full,
    marlin_make_empty_g_idx, marlin_make_workspace, marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks, marlin_sort_g_idx, replace_tensor,
    verify_marlin_supported, verify_marlin_supports_shape)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(self, weight_bits: int, group_size: int, desc_act: bool,
                 is_sym: bool, lm_head_quantized: bool) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError("Unsupported quantization config: "
                             f"bits={weight_bits}, sym={is_sym}")

        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

        # Verify supported on platform.
        verify_marlin_supported(quant_type=self.quant_type,
                                group_size=self.group_size)

    def __repr__(self) -> str:
        return (f"GPTQMarlinConfig(quant_type={self.quant_type}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}, "
                f"lm_head_quantized={self.lm_head_quantized})")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_marlin"

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
    def from_config(cls, config: Dict[str, Any]) -> "GPTQMarlinConfig":
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
        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (user_quant is None or user_quant == "marlin"
                               or user_quant == "gptq_marlin")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info("Detected that the model can run with gptq_marlin"
                        ", however you specified quantization=gptq explicitly,"
                        " so forcing gptq. Use quantization=gptq_marlin for"
                        " faster inference")
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["GPTQMarlinLinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return GPTQMarlinLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        sym = quant_config.get("sym", None)
        desc_act = quant_config.get("desc_act", None)

        if quant_method != "gptq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if (num_bits is None or group_size is None or sym is None
                or desc_act is None):
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(quant_type=cls.TYPE_MAP[(num_bits, sym)],
                                      group_size=group_size,
                                      min_capability=cls.get_min_capability())


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
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
        is_row_parallel = input_size != input_size_per_partition

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

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(self.quant_config.desc_act,
                                             self.quant_config.group_size,
                                             is_row_parallel):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                **extra_weight_attrs,
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        # Activation order
        g_idx = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(
            g_idx,
            {
                **extra_weight_attrs, "input_dim": 0,
                "ignore_warning": True
            },
        )

        # Scales
        scales = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
            },
        )

        # Quantized zero-points
        qzeros = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
                device="meta",
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.is_k_full = marlin_is_k_full(self.quant_config.desc_act,
                                           is_row_parallel)

    # Checkpoints are serialized in AutoGPTQ format, which is different from the
    # marlin format. This function is called after the weights are loaded.
    # Here, we handle the repacking, including the activation reordering case.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device

        # Allocate marlin workspace
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Handle sorting for activation reordering if needed.
        if self.quant_config.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(layer.g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
            replace_tensor(layer, "g_idx", g_idx)
        else:
            layer.g_idx = marlin_make_empty_g_idx(device)
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        layer.zp = marlin_make_empty_g_idx(device)

        # Repack weights from autogptq format to marlin format.
        marlin_qweight = ops.gptq_marlin_repack(
            layer.qweight,
            perm=layer.g_idx_sort_indices,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits)
        replace_tensor(layer, "qweight", marlin_qweight)

        # Permute scales from autogptq format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=(layer.input_size if self.quant_config.desc_act else
                    layer.input_size_per_partition),
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size)
        replace_tensor(layer, "scales", marlin_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_gptq_marlin_linear(
            input=x,
            weight=layer.qweight,
            weight_scale=layer.scales,
            weight_zp=layer.zp,
            g_idx=layer.g_idx,
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            wtype=self.quant_config.quant_type,
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            is_k_full=layer.is_k_full,
            bias=bias)
