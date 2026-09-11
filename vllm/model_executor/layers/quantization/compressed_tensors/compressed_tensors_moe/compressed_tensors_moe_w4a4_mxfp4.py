# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
    narrow_mxfp4_candidates,
    pack_deepgemm_mxfp4_scales,
    select_mxfp4_moe_backend,
    select_mxfp4_moe_backend_from,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class CompressedTensorsW4A4Mxfp4MoEMethod(CompressedTensorsMoEMethod):
    # Candidate backends in preference order. Must stay in sync with the weight
    # preparation below: CUTLASS swizzles scales, DeepGEMM packs them, XPU
    # consumes the checkpoint packing, and Marlin repacks weights and scales.
    # The oracle drops candidates the deployment cannot use (device, expert
    # parallelism, activation format), which is how DeepEP V2's PaddedStandard
    # dispatch reaches DeepGemmFP4Experts instead of a Standard-only kernel.
    CANDIDATE_BACKENDS = (
        Mxfp4MoeBackend.CUTLASS_MXFP4_MXFP4,
        Mxfp4MoeBackend.DEEPGEMM_MXFP4,
        Mxfp4MoeBackend.XPU,
        Mxfp4MoeBackend.MARLIN,
    )

    def __init__(self, moe):
        super().__init__(moe)
        self.group_size = 32
        self.experts_cls: type[mk.FusedMoEExperts]
        if moe.moe_backend == "b12x":
            # b12x has its own precision policy (VLLM_B12X_MOE_FP4_FORCE_A16).
            self.mxfp4_backend, experts_cls = select_mxfp4_moe_backend(moe)
            assert experts_cls is not None
            self.experts_cls = experts_cls
            return

        candidates = list(self.CANDIDATE_BACKENDS)
        if moe.moe_backend != "auto":
            candidates = narrow_mxfp4_candidates(moe.moe_backend, candidates)
            if not candidates:
                raise ValueError(
                    f"moe_backend={moe.moe_backend!r} is not supported for "
                    "compressed-tensors MXFP4 MoE; expected one of "
                    f"{[b.value for b in self.CANDIDATE_BACKENDS]}."
                )
        self.mxfp4_backend, self.experts_cls = select_mxfp4_moe_backend_from(
            moe, candidates
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                requires_grad=False,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.moe.w13_num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.group_size,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # CUTLASS and XPU use W4A4; DeepGEMM W4A8; b12x W4A8 or W4A16;
        # Marlin W4A16.
        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            layer=layer,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w13_weight_packed")

        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight_packed.data, requires_grad=False
        )
        delattr(layer, "w2_weight_packed")

        if self.mxfp4_backend == Mxfp4MoeBackend.CUTLASS_MXFP4_MXFP4:
            # Swizzle weight scales from flat checkpoint layout [E, N, K//32]
            # to CUTLASS tiled layout [E, numMTiles*numKTiles*512].
            from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
                swizzle_mxfp4_scales,
            )

            E = layer.w13_weight_scale.shape[0]
            w13_N = layer.w13_weight_scale.shape[1]
            w13_scale_K = layer.w13_weight_scale.shape[2]
            w13_K = w13_scale_K * 32

            w2_M = layer.w2_weight_scale.shape[1]
            w2_scale_N = layer.w2_weight_scale.shape[2]
            w2_N = w2_scale_N * 32

            swizzled_w13 = []
            swizzled_w2 = []
            for e_idx in range(E):
                s13 = layer.w13_weight_scale[e_idx]
                sw13 = swizzle_mxfp4_scales(s13, w13_N, w13_K)
                swizzled_w13.append(sw13.reshape(w13_N, w13_scale_K))
                s2 = layer.w2_weight_scale[e_idx]
                sw2 = swizzle_mxfp4_scales(s2, w2_M, w2_N)
                swizzled_w2.append(sw2.reshape(w2_M, w2_scale_N))
            layer.w13_weight_scale = torch.nn.Parameter(
                torch.stack(swizzled_w13), requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(
                torch.stack(swizzled_w2), requires_grad=False
            )
        elif self.mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:
            w13_scale, w2_scale = pack_deepgemm_mxfp4_scales(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
            )
            layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
        elif self.mxfp4_backend in (
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.BATCHED_MARLIN,
        ):
            logger.warning_once(
                "Your GPU does not have native support for FP4 computation "
                "but FP4 quantization is being used. Weight-only FP4 "
                "compression will be used leveraging the Marlin kernel. "
                "This may degrade performance for compute-heavy workloads."
            )
            prepare_moe_fp4_layer_for_marlin(layer)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config is not None:
            self.moe_kernel = make_mxfp4_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                experts_cls=self.experts_cls,
                mxfp4_backend=self.mxfp4_backend,
                routing_tables=layer._expert_routing_tables(),
            )
            self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
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
