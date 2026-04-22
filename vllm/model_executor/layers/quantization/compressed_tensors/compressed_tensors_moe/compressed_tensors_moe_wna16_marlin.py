# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.oracle.wna16 import (
    WNA16MoeBackend,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_mxint4_moe import (
    prepare_static_weights_for_trtllm_mxint4_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
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
        assert weight_quant.symmetric, (
            "Only symmetric quantization is supported for MoE"
        )
        # Extract properties from weight_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.actorder = weight_quant.actorder

        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[self.num_bits]

        self.marlin_input_dtype = get_marlin_input_dtype(layer_name)

        # Select WNA16 MoE backend via oracle.
        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=self.moe,
            num_bits=self.num_bits,
            group_size=self.group_size or -1,
        )

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
        for weight scales.
        """
        if weight_name == "w13_scale":
            assert num_groups_w13 is not None, (
                "num_groups_w13 must be provided for weight scales"
            )
        if weight_name == "w2_scale":
            assert num_groups_w2 is not None, (
                "num_groups_w2 must be provided for weight scales"
            )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        is_flashinfer = self.wna16_backend == WNA16MoeBackend.FLASHINFER
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
        }
        backend_key = "Flashinfer" if is_flashinfer else "Marlin"
        return shape_map[weight_name][backend_key]

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
        is_transposed = self.wna16_backend != WNA16MoeBackend.FLASHINFER
        extra_weight_attrs.update(
            {"is_transposed": is_transposed, "quant_method": self.strategy}
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

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1
        w2_scales_size = (
            intermediate_size_full if load_full_w2 else intermediate_size_per_partition
        )

        self.is_k_full = (not self.actorder) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
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
        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device
        if self.wna16_backend == WNA16MoeBackend.FLASHINFER:
            dict_weights_mxint4 = prepare_static_weights_for_trtllm_mxint4_moe(
                layer.w13_weight_packed,
                layer.w13_weight_scale,
                layer.w2_weight_packed,
                layer.w2_weight_scale,
            )
            replace_parameter(
                layer, "w13_weight_packed", dict_weights_mxint4["gemm1_weights"]
            )
            replace_parameter(
                layer, "w13_weight_scale", dict_weights_mxint4["gemm1_scales"]
            )
            replace_parameter(
                layer, "w2_weight_packed", dict_weights_mxint4["gemm2_weights"]
            )
            replace_parameter(
                layer, "w2_weight_scale", dict_weights_mxint4["gemm2_scales"]
            )
            # Alias packed weights to w13_weight/w2_weight for the modular
            # kernel interface.
            layer.w13_weight = layer.w13_weight_packed
            layer.w2_weight = layer.w2_weight_packed

            # Setup moe_kernel for the Flashinfer monolithic path.
            assert self.experts_cls is not None
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            self.moe_kernel = make_wna16_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.wna16_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )
            return

        is_a_8bit = (
            self.marlin_input_dtype is not None
            and self.marlin_input_dtype.itemsize == 1
        )

        if self.marlin_input_dtype == torch.float8_e4m3fn:
            # NOTE: for non-zp quantization format only
            ops.marlin_int4_fp8_preprocess(layer.w13_weight_packed, inplace=True)
            ops.marlin_int4_fp8_preprocess(layer.w2_weight_packed, inplace=True)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data * 512
            layer.w2_weight_scale.data = layer.w2_weight_scale.data * 512

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)

        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)

        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_weight_scale,
            size_k=layer.w13_weight_packed.shape[2],
            size_n=layer.w13_weight_scale.shape[2],
            group_size=self.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.marlin_input_dtype == torch.int8 and layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
            layer.register_parameter(
                "w13_input_global_scale",
                torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
            )
        replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_weight_scale,
            size_k=layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            size_n=layer.w2_weight_scale.shape[2],
            group_size=self.group_size,
            is_a_8bit=is_a_8bit,
        )
        if self.marlin_input_dtype == torch.int8 and layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )
            layer.register_parameter(
                "w2_input_global_scale",
                torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
            )
        replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)

        layer.workspace = marlin_make_workspace_new(device, 4)

        # Alias packed weights to w13_weight/w2_weight for the modular
        # kernel interface.
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        # Setup moe_kernel for the 4-bit Marlin path.
        if self.num_bits == 4 and self.experts_cls is not None:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            assert self.moe_quant_config is not None
            self.moe_kernel = make_wna16_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                backend=self.wna16_backend,
                experts_cls=self.experts_cls,
                w13_g_idx=layer.w13_weight_g_idx,
                w2_g_idx=layer.w2_weight_g_idx,
                w13_g_idx_sort_indices=layer.w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=layer.w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.num_bits != 4:
            return None
        return make_wna16_moe_quant_config(
            layer=layer,
            group_size=self.group_size,
        )

    @property
    def is_monolithic(self) -> bool:
        return self.wna16_backend == WNA16MoeBackend.FLASHINFER

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert self.wna16_backend == WNA16MoeBackend.FLASHINFER
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
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        if self.moe_kernel is not None:
            # 4-bit Marlin: use the modular kernel abstraction.
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
                shared_experts_input=shared_experts_input,
            )
        # Non-4-bit Marlin fallback: call fused_marlin_moe() directly.
        return fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            None,
            None,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            input_global_scale1=getattr(layer, "w13_input_global_scale", None),
            input_global_scale2=getattr(layer, "w2_input_global_scale", None),
            quant_type_id=self.quant_type.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            activation=layer.activation,
            expert_map=layer.expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            input_dtype=self.marlin_input_dtype,
            is_k_full=self.is_k_full,
            inplace=not self.moe.disable_inplace,
        )
