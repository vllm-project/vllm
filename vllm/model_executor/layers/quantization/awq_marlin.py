from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    apply_awq_marlin_linear, awq_to_marlin_zero_points, check_marlin_supported,
    marlin_make_empty_g_idx, marlin_make_workspace, marlin_moe_permute_scales,
    marlin_permute_scales, moe_awq_to_marlin_zero_points,
    verify_marlin_supported, verify_marlin_supports_shape)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform
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
        self.weight_bits = weight_bits

        if self.weight_bits not in self.TYPE_MAP:
            raise ValueError(f"Unsupported num_bits = {self.weight_bits}. "
                             f"Supported num_bits = {self.TYPE_MAP.keys()}")

        self.quant_type = self.TYPE_MAP[self.weight_bits]

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
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return AWQMarlinLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return AWQMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_awq_marlin_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        has_zp = quant_config.get("zero_point")

        if not current_platform.is_cuda():
            return False

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if (num_bits is None or group_size is None or has_zp is None):
            return False

        if num_bits not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(quant_type=cls.TYPE_MAP[num_bits],
                                      group_size=group_size,
                                      has_zp=has_zp)


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
        weight_loader = extra_weight_attrs.get("weight_loader")

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

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        num_groups = input_size_per_partition // group_size

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        scales = GroupQuantScaleParameter(data=torch.empty(
            num_groups,
            output_size_per_partition,
            dtype=params_dtype,
        ),
                                          input_dim=0,
                                          output_dim=1,
                                          weight_loader=weight_loader)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    # TODO: Update this docs
    # Checkpoints are serialized in AutoAWQ format, which is different from the
    # marlin format. This function is called after the weights are loaded.
    # Here, we handle the repacking
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device
        layer.qweight = torch.nn.Parameter(layer.qweight.data,
                                           requires_grad=False)
        layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
                                          requires_grad=False)
        layer.scales = torch.nn.Parameter(layer.scales.data,
                                          requires_grad=False)

        # Allocate marlin workspace
        layer.workspace = marlin_make_workspace(
            layer.output_size_per_partition, device)

        # Repack weights from AWQ format to marlin format.
        marlin_qweight = ops.awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits)
        replace_parameter(layer, "qweight", marlin_qweight)

        # Permute scales from AWQ format to marlin format.
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size)
        replace_parameter(layer, "scales", marlin_scales)

        # Permute zero-points from AWQ format to marlin format.
        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits)
        replace_parameter(layer, "qzeros", marlin_zp)

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


class AWQMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: AWQMarlinConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        extra_weight_attrs.update({
            "is_transposed":
            True,
            "quant_method":
            FusedMoeWeightScaleSupported.GROUP.value,
        })

        w13_qweight = Parameter(torch.empty(num_experts,
                                            hidden_size,
                                            2 * intermediate_size //
                                            self.quant_config.pack_factor,
                                            dtype=torch.int32),
                                requires_grad=False)
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = Parameter(torch.empty(num_experts,
                                           intermediate_size,
                                           hidden_size //
                                           self.quant_config.pack_factor,
                                           dtype=torch.int32),
                               requires_grad=False)
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size // self.quant_config.group_size

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = Parameter(torch.empty(num_experts,
                                           num_groups_w13,
                                           intermediate_size * 2,
                                           dtype=params_dtype),
                               requires_grad=False)
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = Parameter(torch.empty(num_experts,
                                          num_groups_w2,
                                          hidden_size,
                                          dtype=params_dtype),
                              requires_grad=False)
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = Parameter(torch.empty(num_experts,
                                           num_groups_w13,
                                           2 * intermediate_size //
                                           self.quant_config.pack_factor,
                                           dtype=torch.int32),
                               requires_grad=False)
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = Parameter(torch.empty(num_experts,
                                          num_groups_w2,
                                          hidden_size //
                                          self.quant_config.pack_factor,
                                          dtype=torch.int32),
                              requires_grad=False)
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.w13_qweight.shape[0]
        device = layer.w13_qweight.device

        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

        marlin_w13_qweight = ops.awq_marlin_moe_repack(
            layer.w13_qweight,
            layer.w13_g_idx_sort_indices,
            size_k=layer.w13_qweight.shape[1],
            size_n=layer.w13_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)

        marlin_w2_qweight = ops.awq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            size_k=layer.w2_qweight.shape[1],
            size_n=layer.w2_qweight.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        # Why does this take the intermediate size for size_k?
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
        )

        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            size_k=layer.intermediate_size_per_partition,
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        marlin_w13_zp = moe_awq_to_marlin_zero_points(
            layer.w13_qzeros,
            size_k=layer.w13_qzeros.shape[1],
            size_n=layer.w13_qzeros.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits)
        replace_parameter(layer, "w13_qzeros", marlin_w13_zp)

        marlin_w2_zp = moe_awq_to_marlin_zero_points(
            layer.w2_qzeros,
            size_k=layer.w2_qzeros.shape[1],
            size_n=layer.w2_qzeros.shape[2] * self.quant_config.pack_factor,
            num_bits=self.quant_config.weight_bits)
        replace_parameter(layer, "w2_qzeros", marlin_w2_zp)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:

        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            fused_marlin_moe)

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function)

        return fused_marlin_moe(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            layer.w13_scales,
            layer.w2_scales,
            router_logits,
            topk_weights,
            topk_ids,
            w1_zeros=layer.w13_qzeros,
            w2_zeros=layer.w2_qzeros,
            num_bits=self.quant_config.weight_bits,
        )
