# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/sgl-project/sglang/pull/3730

import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.nn import Module
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped)

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    apply_w8a8_block_int8_linear)
from vllm.model_executor.utils import set_weight_attrs

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class BlockInt8Config(QuantizationConfig):
    """Config class for INT8."""

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
    ) -> None:
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized
        if is_checkpoint_int8_serialized:
            logger.warning(
                "Detected int8 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError("Unsupported activation scheme"
                             f" {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_int8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports "
                    "int8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 "
                    "dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic "
                    "activation scheme for now, but got "
                    "{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "blockwise_int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BlockInt8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "int8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config,
                                                 ["weight_block_size"], None)
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return BlockInt8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return BlockInt8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class BlockInt8LinearMethod(LinearMethodBase):
    """Linear method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support block-wise int8 quantization and int8 checkpoint

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: BlockInt8Config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None
        assert self.quant_config.is_checkpoint_int8_serialized

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: Optional[List[int]],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # assert output_partition_sizes is not None, (
        #     "output_partition_sizes must be provided for quantization")
        output_partition_sizes = output_partition_sizes or []
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()

        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # Required by row parallel
        if (tp_size > 1
                and input_size // input_size_per_partition == tp_size
                and input_size_per_partition % block_k != 0):
            raise ValueError(
                f"Weight input_size_per_partition = "
                f"{input_size_per_partition} is not divisible by "
                f"weight quantization block_k = {block_k}.")
        # Required by collum parallel or enabling merged weights
        if (tp_size > 1 and output_size // output_size_per_partition
                == tp_size) or len(output_partition_sizes) > 1:
            for output_partition_size in output_partition_sizes:
                if output_partition_size % block_n != 0:
                    raise ValueError(
                        f"Weight output_partition_size = "
                        f"{output_partition_size} is not divisible by "
                        f"weight quantization block_n = {block_n}."
                    )

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.int8
            if self.quant_config.is_checkpoint_int8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition,
                dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE

        scale = BlockQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        assert self.quant_config.activation_scheme == "dynamic"
        layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        # Use torch Parameter to avoid cuda graph capturing issue
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.data, requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.quant_config.weight_block_size is not None
        return apply_w8a8_block_int8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )


class BlockInt8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic activation scale.

    Limitations:
    Only support block-wise int8 quantization and int8 checkpoint

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config
        assert self.quant_config.weight_block_size is not None
        assert self.quant_config.is_checkpoint_int8_serialized

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.quant_config.is_checkpoint_int8_serialized:
            params_dtype = torch.int8
        tp_size = get_tensor_model_parallel_world_size()

        block_n, block_k = (
            self.quant_config.weight_block_size[0],
            self.quant_config.weight_block_size[1],
        )
        # NOTE(HandH1998): To ensure proper alignment of the block-wise
        # quantization scales, the output_size of the weights for both
        # the gate and up layers must be divisible by block_n.
        # Required by collum parallel or enabling merged weights
        if intermediate_size_per_partition % block_n != 0:
            raise ValueError(
                f"The output_size of gate's and up's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_n = {block_n}."
            )

        if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
            # Required by row parallel
            raise ValueError(
                f"The input_size of down's weight = "
                f"{intermediate_size_per_partition} is not divisible by "
                f"weight quantization block_k = {block_k}."
            )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition,
                hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size,
                intermediate_size_per_partition, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) //
                     block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert self.quant_config.activation_scheme == "dynamic"
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Block quant doesn't need to process weights after loading
        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        # Expert selection
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )

        # Expert fusion with INT8 quantization
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_int8_w8a8=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=(layer.w13_weight_scale_inv),
            w2_scale=(layer.w2_weight_scale_inv),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=self.quant_config.weight_block_size,
        )
