# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.nn.functional as F

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    AiterExperts,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts
else:
    TritonExperts = None  # type: ignore


logger = init_logger(__name__)


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

        self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

        # FlashInfer CUTLASS MoE is only supported on Hopper and later GPUS
        self.flashinfer_cutlass_moe_enabled = (
            has_flashinfer_cutlass_fused_moe()
            and envs.VLLM_USE_FLASHINFER_MOE_FP16
            and self.moe.moe_parallel_config.use_ep
            and self.moe.moe_parallel_config.dp_size == 1
            and current_platform.get_device_capability()[0] >= 9
        )
        if self.flashinfer_cutlass_moe_enabled:
            logger.info_once(
                "Enabling FlashInfer CUTLASS MoE for UnquantizedFusedMoEMethod"
            )
        else:
            if (
                self.moe.moe_parallel_config.use_ep
                and self.moe.moe_parallel_config.dp_size == 1
            ):
                logger.info_once(
                    "FlashInfer CUTLASS MoE is available for EP"
                    " but not enabled, consider setting"
                    " VLLM_USE_FLASHINFER_MOE_FP16=1 to enable it.",
                    scope="local",
                )
            elif self.moe.moe_parallel_config.dp_size > 1:
                logger.info_once(
                    "FlashInfer CUTLASS MoE is currently not available for DP.",
                    scope="local",
                )

    @property
    def supports_eplb(self) -> bool:
        return True

    @property
    def allow_inplace(self) -> bool:
        return True

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalize | None:
        if self.rocm_aiter_moe_enabled:
            return None
        else:
            return super().maybe_make_prepare_finalize(routing_tables)

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                quant_config=self.moe_quant_config,
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return TritonExperts(self.moe_quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.moe.is_act_and_mul:
            w13_up_dim = 2 * intermediate_size_per_partition
        else:
            w13_up_dim = intermediate_size_per_partition
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(num_experts, w13_up_dim, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        if self.moe.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        if (
            envs.VLLM_ROCM_MOE_PADDING
            and current_platform.is_rocm()
            and weight.stride(-1) == 1
            and (weight.stride(-2) * weight.element_size()) % 512 == 0
        ):
            num_pad = 256 // weight.element_size()
            weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
            torch.cuda.empty_cache()

        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        # Padding the weight for better performance on ROCm
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)

        if current_platform.is_xpu():
            import intel_extension_for_pytorch as ipex

            ep_rank_start = self.moe.ep_rank * self.moe.num_local_experts
            layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
                experts_start_id=ep_rank_start,
            )
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.fused_moe import cpu_fused_moe

            if current_platform.get_cpu_architecture() == CpuArchEnum.X86:
                from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

                dtype_w13 = layer.w13_weight.dtype
                _, n_w13, k_w13 = layer.w13_weight.size()
                dtype_w2 = layer.w2_weight.dtype
                _, n_w2, k_w2 = layer.w2_weight.size()
                if (
                    envs.VLLM_CPU_SGL_KERNEL
                    and check_cpu_sgl_kernel(n_w13, k_w13, dtype_w13)
                    and check_cpu_sgl_kernel(n_w2, k_w2, dtype_w2)
                ):
                    packed_w13_weight = torch.ops._C.convert_weight_packed(
                        layer.w13_weight
                    )
                    assert packed_w13_weight.size() == layer.w13_weight.size()
                    layer.w13_weight.copy_(packed_w13_weight)
                    del packed_w13_weight
                    packed_w2_weight = torch.ops._C.convert_weight_packed(
                        layer.w2_weight
                    )
                    assert packed_w2_weight.size() == layer.w2_weight.size()
                    layer.w2_weight.copy_(packed_w2_weight)
                    layer.cpu_fused_moe = cpu_fused_moe.SGLFusedMOE(layer)
                else:
                    layer.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
            else:
                layer.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)
        elif current_platform.is_cuda_alike():
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
            if self.rocm_aiter_moe_enabled:
                shuffled_w13, shuffled_w2 = rocm_aiter_ops.shuffle_weights(
                    layer.w13_weight.data, layer.w2_weight.data
                )
                replace_parameter(layer, "w13_weight", shuffled_w13)
                replace_parameter(layer, "w2_weight", shuffled_w2)

                self.use_inplace = True
                self.kernel = mk.FusedMoEModularKernel(
                    MoEPrepareAndFinalizeNoEP(),
                    AiterExperts(self.moe_quant_config),
                    shared_experts=None,
                )

            elif self.flashinfer_cutlass_moe_enabled:
                self.use_inplace = False
                # Swap halves to arrange as [w3; w1] (kernel expectation)
                w13_weight = swap_w13_to_w31(layer.w13_weight.data)
                replace_parameter(layer, "w13_weight", w13_weight)

                self.kernel = mk.FusedMoEModularKernel(
                    MoEPrepareAndFinalizeNoEP(),
                    FlashInferExperts(
                        out_dtype=layer.params_dtype,
                        quant_config=self.moe_quant_config,
                        tp_rank=self.moe.moe_parallel_config.tp_rank,
                        tp_size=self.moe.moe_parallel_config.tp_size,
                        ep_rank=self.moe.moe_parallel_config.ep_rank,
                        ep_size=self.moe.moe_parallel_config.ep_size,
                    ),
                )
            else:
                self.use_inplace = True
                self.kernel = mk.FusedMoEModularKernel(
                    MoEPrepareAndFinalizeNoEP(),
                    TritonExperts(self.moe_quant_config),
                    shared_experts=None,
                )

    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            router=router,
            layer=layer,
            x=x,
            router_logits=router_logits,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        result = self.kernel(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=self.use_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )

        return result

    def forward_cpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            layer.enable_eplb is not False
            or layer.expert_load_view is not None
            or layer.logical_to_physical_map is not None
            or layer.logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for CPU.")

        return layer.cpu_fused_moe(
            layer,
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            layer.global_num_experts,
            layer.expert_map,
            layer.custom_routing_function,
            layer.scoring_func,
            layer.routed_scaling_factor,
            layer.e_score_correction_bias,
            layer.apply_router_weight_on_input,
            layer.activation,
        )

    def forward_xpu(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            layer.enable_eplb is not False
            or layer.expert_load_view is not None
            or layer.logical_to_physical_map is not None
            or layer.logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for XPU.")
        return layer.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            layer.topk_group,
            layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
        )

    if current_platform.is_cpu():
        forward_native = forward_cpu
    elif current_platform.is_xpu():
        forward_native = forward_xpu
    else:
        forward_native = forward_cuda
