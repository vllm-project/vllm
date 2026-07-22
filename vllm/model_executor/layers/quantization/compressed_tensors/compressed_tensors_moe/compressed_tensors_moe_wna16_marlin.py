# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEExpertsModular,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_TYPES_MAP,
    WNA16_ZP_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static32GroupScale,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.symmetric = weight_quant.symmetric
        # Extract properties from weight_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.actorder = weight_quant.actorder

        self.quant_type = (
            WNA16_SUPPORTED_TYPES_MAP[self.num_bits]
            if self.symmetric
            else WNA16_ZP_SUPPORTED_TYPES_MAP[self.num_bits]
        )

        self.marlin_input_dtype = get_marlin_input_dtype(layer_name)

        if self.num_bits == 4:
            if self.group_size == 32:
                scale = kInt4Static32GroupScale
            else:
                scale = kInt4StaticGroupScale
        elif self.num_bits == 8:
            scale = kInt8StaticGroupScale
        else:
            raise ValueError(
                "CompressedTensorsWNA16MarlinMoEMethod only supports int4 and int8 now."
            )

        weight_key = QuantKey(self.quant_type, scale, symmetric=self.symmetric)

        # Select WNA16 MoE backend via oracle.
        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            quant_config=self.weight_quant,
            may_have_zp=not self.symmetric,
            may_have_bias=False,
        )
        self.is_marlin = self.wna16_backend in [
            WNA16MoEBackend.MARLIN,
            WNA16MoEBackend.BATCHED_MARLIN,
        ]
        self.is_transposed = self.wna16_backend != WNA16MoEBackend.FLASHINFER_TRTLLM

    def get_weight_shape(
        self,
        weight_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        num_groups_w2: int | None = None,
        num_groups_w13: int | None = None,
    ) -> tuple[int, int, int]:
        """
        Get the shape of the weight based on the weight name, number of experts
        hidden size, intermediate size per partition, number of groups for w2,
        and number of groups for w13. Pass in num_groups_w2 and num_groups_w13
        for weight scales/zero_points.
        """
        if weight_name in ("w13_scale", "w13_zp"):
            assert num_groups_w13 is not None, (
                "num_groups_w13 must be provided for weight scales/zero_points"
            )
        if weight_name in ("w2_scale", "w2_zp"):
            assert num_groups_w2 is not None, (
                "num_groups_w2 must be provided for weight scales/zero_points"
            )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        shape_map = {
            "w13_weight": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    hidden_size // self.packed_factor,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w13_scale": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    num_groups_w13,
                ),
                "Marlin": (
                    num_experts,
                    num_groups_w13,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w13_zp": {
                "Marlin": (
                    num_experts,
                    num_groups_w13,
                    w13_num_shards
                    * intermediate_size_per_partition
                    // self.packed_factor,
                ),
            },
            "w2_weight": {
                "Flashinfer": (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    intermediate_size_per_partition // self.packed_factor,
                    hidden_size,
                ),
            },
            "w2_scale": {
                "Flashinfer": (num_experts, hidden_size, num_groups_w2),
                "Marlin": (num_experts, num_groups_w2, hidden_size),
            },
            "w2_zp": {
                "Marlin": (
                    num_experts,
                    num_groups_w2,
                    hidden_size // self.packed_factor,
                ),
            },
        }
        backend_key = "Marlin" if self.is_transposed else "Flashinfer"
        return shape_map[weight_name][backend_key]

    @staticmethod
    def _w2_scale_sharding(
        actorder,
        group_size: int,
        intermediate_size_per_partition: int,
        intermediate_size_full: int,
    ) -> tuple[bool, int, bool]:
        """Decide how to shard w2 group scales across TP for WNA16 Marlin MoE.

        Only ``actorder="group"`` permutes activations by ``g_idx`` at runtime
        and therefore needs the full-K (unsharded) w2 scales plus ``is_k_full``.
        ``actorder="weight"``/``"static"`` (and ``None``) reorder weights at
        quantization time, so scales shard normally per TP rank.
        """
        load_full_w2 = (actorder == "group") and group_size != -1
        w2_scales_size = (
            intermediate_size_full if load_full_w2 else intermediate_size_per_partition
        )
        is_k_full = (actorder != "group") or (
            intermediate_size_per_partition == intermediate_size_full
        )
        return load_full_w2, w2_scales_size, is_k_full

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": self.is_transposed, "quant_method": self.strategy}
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                *self.get_weight_shape(
                    "w13_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                *self.get_weight_shape(
                    "w2_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        load_full_w2, w2_scales_size, self.is_k_full = self._w2_scale_sharding(
            self.actorder,
            self.group_size,
            intermediate_size_per_partition,
            intermediate_size_full,
        )

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            if hidden_size % self.group_size != 0:
                raise ValueError(
                    "CompressedTensors WNA16 Marlin MoE requires hidden_size "
                    f"({hidden_size}) to be divisible by group_size "
                    f"({self.group_size})."
                )
            if (
                not load_full_w2
                and intermediate_size_per_partition % self.group_size != 0
            ):
                raise ValueError(
                    "CompressedTensors WNA16 Marlin MoE with static group "
                    "scales requires the MoE intermediate size per "
                    "tensor-parallel partition "
                    f"({intermediate_size_per_partition}) to be divisible by "
                    f"group_size ({self.group_size}). Scale groups would "
                    "otherwise cross TP shard boundaries; use a compatible TP "
                    "size or enable expert parallelism."
                )
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        w13_scale = torch.nn.Parameter(
            torch.ones(
                *self.get_weight_shape(
                    "w13_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w13=num_groups_w13,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(
                *self.get_weight_shape(
                    "w2_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w2=num_groups_w2,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        if not self.symmetric:
            w13_zp = torch.nn.Parameter(
                torch.zeros(
                    *self.get_weight_shape(
                        "w13_zp",
                        num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        num_groups_w13=num_groups_w13,
                    ),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_zero_point", w13_zp)
            set_weight_attrs(w13_zp, extra_weight_attrs)

            w2_zp = torch.nn.Parameter(
                torch.zeros(
                    *self.get_weight_shape(
                        "w2_zp",
                        num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        num_groups_w2=num_groups_w2,
                    ),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_zero_point", w2_zp)
            set_weight_attrs(w2_zp, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process weights using the shared oracle infrastructure
        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_backend,
            layer=layer,
            quant_config=self.weight_quant,
            input_dtype=self.marlin_input_dtype,
            w13=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_g_idx=layer.w13_weight_g_idx,
            w2_g_idx=layer.w2_weight_g_idx,
            w13_qzeros=getattr(layer, "w13_weight_zero_point", None),
            w2_qzeros=getattr(layer, "w2_weight_zero_point", None),
        )

        if converted is None:
            # In-place backends (e.g. Humming) are not wired through this
            # marlin-only method; fail clearly rather than unpacking None.
            raise NotImplementedError(
                f"{type(self).__name__} does not support the "
                f"{self.wna16_backend.value} MoE backend."
            )

        (
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_g_idx_processed,
            w2_g_idx_processed,
            w13_g_idx_sort_indices,
            w2_g_idx_sort_indices,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            _,  # w13_bias
            _,  # w2_bias
        ) = converted

        # Replace common parameters
        replace_parameter(layer, "w13_weight_packed", w13_qweight)
        replace_parameter(layer, "w2_weight_packed", w2_qweight)
        replace_parameter(layer, "w13_weight_scale", w13_scales)
        replace_parameter(layer, "w2_weight_scale", w2_scales)

        # CPU fused_experts_cpu requires zero points even for symmetric quant
        if not self.symmetric or self.wna16_backend == WNA16MoEBackend.CPU:
            replace_parameter(layer, "w13_weight_zero_point", w13_qzeros)
            replace_parameter(layer, "w2_weight_zero_point", w2_qzeros)

        # Marlin-specific parameters (not needed for Flashinfer)
        if self.is_marlin:
            if w13_g_idx_processed is not None:
                replace_parameter(layer, "w13_weight_g_idx", w13_g_idx_processed)
            if w2_g_idx_processed is not None:
                replace_parameter(layer, "w2_weight_g_idx", w2_g_idx_processed)
            if w13_g_idx_sort_indices is not None:
                replace_parameter(
                    layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices
                )
            if w2_g_idx_sort_indices is not None:
                replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

            # Register input global scales if present
            if w13_input_global_scale is not None:
                layer.register_parameter(
                    "w13_input_global_scale",
                    torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
                )
            if w2_input_global_scale is not None:
                layer.register_parameter(
                    "w2_input_global_scale",
                    torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
                )

            # Marlin workspace — only needed for Marlin-family backends, not emulation.
            if (
                self.experts_cls is not None
                and issubclass(self.experts_cls, FusedMoEExpertsModular)
                and self.wna16_backend != WNA16MoEBackend.EMULATION
            ):
                layer.workspace = marlin_make_workspace_new(
                    layer.w13_weight_g_idx.device, 4
                )

        # Alias packed weights to w13_weight/w2_weight for the modular kernel interface
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        assert self.experts_cls is not None
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        # Add Marlin-specific arguments
        marlin_args: dict[str, Any] = {}
        if self.is_marlin:
            marlin_args = {
                "w13_g_idx": layer.w13_weight_g_idx,
                "w2_g_idx": layer.w2_weight_g_idx,
                "w13_g_idx_sort_indices": layer.w13_g_idx_sort_indices,
                "w2_g_idx_sort_indices": layer.w2_g_idx_sort_indices,
                "is_k_full": self.is_k_full,
            }

        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
            **marlin_args,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_size=self.group_size,
            num_bits=self.num_bits,
            w1_zp=getattr(layer, "w13_weight_zero_point", None),
            w2_zp=getattr(layer, "w2_weight_zero_point", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
