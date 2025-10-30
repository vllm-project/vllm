# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Literal, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    biased_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEModularKernel,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data,
    is_rocm_aiter_fusion_shared_expert_enabled,
    is_rocm_aiter_moe_enabled,
)
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingSimulator
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    is_flashinfer_supporting_global_sf,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.import_utils import has_deep_ep, has_pplx
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import current_stream, direct_register_custom_op
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if current_platform.is_cuda_alike():
    from .fused_batched_moe import BatchedTritonExperts
    from .fused_moe import TritonExperts, eplb_map_to_physical_and_record, fused_experts

    if has_pplx():
        from .pplx_prepare_finalize import (
            PplxPrepareAndFinalize,
            pplx_hidden_dim_scale_bytes,
        )
    if has_deep_ep():
        from .deepep_ht_prepare_finalize import DeepEPHTPrepareAndFinalize
        from .deepep_ll_prepare_finalize import (
            DEEPEP_QUANT_BLOCK_SHAPE,
            DeepEPLLPrepareAndFinalize,
        )
else:
    fused_experts = None  # type: ignore
    FusedMoEPermuteExpertsUnpermute = object  # type: ignore
    FusedMoEPrepareAndFinalize = object  # type: ignore

    def _eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids

    eplb_map_to_physical_and_record = _eplb_map_to_physical_and_record

if is_rocm_aiter_moe_enabled():
    from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
        rocm_aiter_grouped_topk as grouped_topk_aiter,
    )
else:
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore

logger = init_logger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.fused_experts: FusedMoEModularKernel | None = None
        self.topk_indices_dtype = None

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    @staticmethod
    def _maybe_make_prepare_finalize(
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig | None,
    ) -> FusedMoEPrepareAndFinalize | None:
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None

        prepare_finalize: FusedMoEPrepareAndFinalize | None = None

        # TODO: could allow this now
        assert not moe.use_flashinfer_cutlass_kernels, "Must be created in modelopt.py"

        if moe.use_pplx_kernels:
            assert quant_config is not None

            hidden_dim_bytes, hidden_scale_bytes = pplx_hidden_dim_scale_bytes(
                moe.max_num_tokens,
                moe.hidden_dim,
                moe.in_dtype,
                quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape,
            )

            all_to_all_args = dict(
                max_num_tokens=moe.max_num_tokens,
                num_experts=moe.num_experts,
                experts_per_token=moe.experts_per_token,  # topk
                rank=all2all_manager.rank,
                world_size=all2all_manager.world_size,
                # dp_size actually means tp_size, bug in pplx kernels
                dp_size=all2all_manager.tp_group.world_size,
                hidden_dim=moe.hidden_dim,
                hidden_dim_bytes=hidden_dim_bytes,
                hidden_dim_scale_bytes=hidden_scale_bytes,
            )

            num_dispatchers = (
                all2all_manager.world_size // all2all_manager.tp_group.world_size
            )

            # Intranode pplx a2a takes a group name while internode does not.
            if not all2all_manager.internode:
                all_to_all_args["group_name"] = all2all_manager.cpu_group.group_name

            handle = all2all_manager.get_handle(all_to_all_args)

            prepare_finalize = PplxPrepareAndFinalize(
                handle,
                max_num_tokens=moe.max_num_tokens,
                num_local_experts=moe.num_local_experts,
                num_dispatchers=num_dispatchers,
            )
        elif moe.use_deepep_ht_kernels:
            assert moe.dp_size == all2all_manager.dp_world_size

            all_to_all_args = dict()
            handle = all2all_manager.get_handle(all_to_all_args)
            prepare_finalize = DeepEPHTPrepareAndFinalize(
                handle,
                num_dispatchers=all2all_manager.world_size,
                dp_size=all2all_manager.dp_world_size,
                rank_expert_offset=all2all_manager.rank * moe.num_local_experts,
            )

        elif moe.use_deepep_ll_kernels:
            assert quant_config is not None
            all_to_all_args = dict(
                max_num_tokens_per_dp_rank=moe.max_num_tokens,
                token_hidden_size=moe.hidden_dim,
                num_ep_ranks=all2all_manager.world_size,
                num_global_experts=moe.num_experts,
                num_local_experts=moe.num_experts // all2all_manager.world_size,
            )
            handle = all2all_manager.get_handle(all_to_all_args)

            # Note: We may want to use FP8 dispatch just to reduce
            # data movement.
            use_fp8_dispatch = (
                quant_config.quant_dtype == current_platform.fp8_dtype()
                and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
            )

            prepare_finalize = DeepEPLLPrepareAndFinalize(
                handle,
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=all2all_manager.world_size,
                use_fp8_dispatch=use_fp8_dispatch,
            )

        return prepare_finalize

    def maybe_make_prepare_finalize(self) -> FusedMoEPrepareAndFinalize | None:
        if self.moe.moe_parallel_config.use_all2all_kernels:
            return FusedMoEMethodBase._maybe_make_prepare_finalize(
                self.moe, self.moe_quant_config
            )
        else:
            return None

    # Note: init_prepare_finalize should only be called by
    # prepare_communication_buffer_for_model.
    def init_prepare_finalize(self, layer: torch.nn.Module):
        assert self.moe is not None

        # We must get the quant config here so that the layer is
        # completely initialized, i.e. all weights loaded and post
        # processed.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        prepare_finalize = self.maybe_make_prepare_finalize()

        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            assert self.topk_indices_dtype is None
            assert self.fused_experts is None, (
                f"Attempt to override experts for {id(self)}!"
            )
            self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
            experts = self.select_gemm_impl(prepare_finalize, layer)
            self.fused_experts = FusedMoEModularKernel(
                prepare_finalize,
                experts,
                layer.shared_experts,
            )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} must select appropriate gemm "
            "implementation based on the prepare_finalize"
        )

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def using_modular_kernel(self) -> bool:
        return self.fused_experts is not None

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@CustomOp.register("unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
        if self.rocm_aiter_moe_enabled:
            from .rocm_aiter_fused_moe import rocm_aiter_fused_experts

            self.rocm_aiter_fused_experts = rocm_aiter_fused_experts
        else:
            self.rocm_aiter_fused_experts = None  # type: ignore

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
            from functools import partial

            from .flashinfer_cutlass_moe import flashinfer_cutlass_moe

            self.flashinfer_cutlass_moe = partial(
                flashinfer_cutlass_moe,
                quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
                tp_rank=self.moe.moe_parallel_config.tp_rank,
                tp_size=self.moe.moe_parallel_config.tp_size,
                ep_rank=self.moe.moe_parallel_config.ep_rank,
                ep_size=self.moe.moe_parallel_config.ep_size,
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
            self.flashinfer_cutlass_moe = None  # type: ignore

    def maybe_make_prepare_finalize(self) -> FusedMoEPrepareAndFinalize | None:
        if self.rocm_aiter_moe_enabled:
            return None
        else:
            return super().maybe_make_prepare_finalize()

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
        # Lazy import to avoid importing triton.
        from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
            shuffle_weights,
        )

        if self.rocm_aiter_moe_enabled:
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight.data, layer.w2_weight.data
            )

            layer.w13_weight.data = shuffled_w13
            layer.w2_weight.data = shuffled_w2

        if self.flashinfer_cutlass_moe_enabled:
            # Swap halves to arrange as [w3; w1] (kernel expectation)
            w1_w, w3_w = torch.chunk(layer.w13_weight.data, 2, dim=1)
            w13_weight_swapped = torch.cat([w3_w, w1_w], dim=1)
            layer.w13_weight.data = w13_weight_swapped.contiguous()

        if current_platform.is_xpu():
            import intel_extension_for_pytorch as ipex

            layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
                layer.w13_weight,
                layer.w2_weight,
                use_prepack=True,
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
                    layer.cpu_fused_moe = cpu_fused_moe.IPEXFusedMOE(layer)
            else:
                layer.cpu_fused_moe = cpu_fused_moe.CPUFusedMOE(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None
            assert isinstance(layer, FusedMoE)

        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            enable_eplb=enable_eplb,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.moe.has_bias:
            return biased_moe_quant_config(
                layer.w13_bias,
                layer.w2_bias,
            )
        else:
            return FUSED_MOE_UNQUANTIZED_CONFIG

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)

        topk_weights, topk_ids, zero_expert_result = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype,
            enable_eplb=enable_eplb,
            expert_map=expert_map,
            expert_load_view=expert_load_view,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            global_num_experts=global_num_experts,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
            num_fused_shared_experts=layer.num_fused_shared_experts,
        )

        if self.rocm_aiter_moe_enabled:
            assert self.fused_experts is None
            result = self.rocm_aiter_fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=expert_map,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        elif self.flashinfer_cutlass_moe_enabled:
            return self.flashinfer_cutlass_moe(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        elif self.fused_experts is not None:
            result = self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        else:
            assert fused_experts is not None
            result = fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=activation,
                quant_config=self.moe_quant_config,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )

        if zero_expert_num != 0 and zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            enable_eplb is not False
            or expert_load_view is not None
            or logical_to_physical_map is not None
            or logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for CPU.")
        return layer.cpu_fused_moe(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            global_num_experts,
            expert_map,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            apply_router_weight_on_input,
            activation,
        )

    def forward_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (
            enable_eplb is not False
            or expert_load_view is not None
            or logical_to_physical_map is not None
            or logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for XPU.")
        assert custom_routing_function is None
        return layer.ipex_fusion(
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
        )

    def forward_tpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not use_grouped_topk
        assert num_expert_group is None
        assert topk_group is None
        assert custom_routing_function is None
        assert apply_router_weight_on_input is False
        if scoring_func != "softmax":
            raise NotImplementedError(
                "Only softmax scoring function is supported for TPU."
            )
        if e_score_correction_bias is not None:
            raise NotImplementedError(
                "Expert score correction bias is not supported for TPU."
            )
        assert activation == "silu", f"{activation} is not supported for TPU."
        assert routed_scaling_factor == 1.0, (
            f"routed_scaling_factor {routed_scaling_factor} is not supported for TPU."
        )
        if (
            enable_eplb is not False
            or expert_load_view is not None
            or logical_to_physical_map is not None
            or logical_replica_count is not None
        ):
            raise NotImplementedError("Expert load balancing is not supported for TPU.")
        return fused_moe_pallas(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk=top_k,
            gating_output=router_logits,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            renormalize=renormalize,
        )

    if current_platform.is_tpu():
        forward_native = forward_tpu
    elif current_platform.is_cpu():
        forward_native = forward_cpu
    elif current_platform.is_xpu():
        forward_native = forward_xpu
    else:
        forward_native = forward_cuda


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if is_rocm_aiter_moe_enabled():
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping from global to local index. Contains -1 for experts not
            assigned to the current rank.

    Returns:
        str: A string mapping from local to global index.
            Using str to support hashing for logging once only.
    """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    quant_config: QuantizationConfig | None,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        quant_config: Fused MoE quantization configuration.
        moe_parallel_config: Fused MoE parallelization strategy configuration.
        is_lora_enabled: True if the engine is enabled with LoRA. This
            is used in the case of mxfp4 quantization in selecting the
            MxFP4Backend.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """

    if moe_parallel_config.use_deepep_ht_kernels:
        hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )

    if moe_parallel_config.use_deepep_ll_kernels:
        hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    # we are padding globally so EP buffer allocation works
    if quant_config and quant_config.get_name() == "mxfp4":
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend,
            get_mxfp4_backend,
        )

        current_mxfp4_backend = get_mxfp4_backend(is_lora_enabled)
        if (
            current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        ):
            hidden_size = round_up(hidden_size, 128)
        elif (
            current_platform.is_rocm()
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
    ):
        super().__init__()

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.info_once("Disabling MoE shared_experts cuda stream")
            self.shared_experts_stream = None
        else:
            self.shared_experts_stream = torch.cuda.Stream()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts + num_redundant_experts
        self.logical_num_experts = num_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size,
            moe_in_dtype,
            quant_config,
            self.moe_parallel_config,
            is_lora_enabled=self.vllm_config.lora_config is not None,
        )

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: torch.Tensor | None = None
        self.logical_to_physical_map: torch.Tensor | None = None
        self.logical_replica_count: torch.Tensor | None = None

        # ROCm aiter shared experts fusion
        self.num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None
            and is_rocm_aiter_fusion_shared_expert_enabled()
            else 0
        )
        if (
            not is_rocm_aiter_fusion_shared_expert_enabled()
            and self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            expert_placement_strategy = (
                vllm_config.parallel_config.expert_placement_strategy
            )
            if expert_placement_strategy == "round_robin":
                # TODO(Bruce): will support round robin expert placement with
                # EPLB enabled in the future.
                round_robin_supported = (
                    (num_expert_group is not None and num_expert_group > 1)
                    and num_redundant_experts == 0
                    and not self.enable_eplb
                )

                if not round_robin_supported:
                    logger.warning(
                        "Round-robin expert placement is only supported for "
                        "models with multiple expert groups and no redundant "
                        "experts. Falling back to linear expert placement."
                    )
                    expert_placement_strategy = "linear"

            self.expert_map: torch.Tensor | None
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map),
            )
        else:
            self.local_num_experts, self.expert_map, self.expert_mask = (
                self.global_num_experts,
                None,
                None,
            )

        self.top_k = top_k

        self._init_aiter_shared_experts_topK_buffer(
            vllm_config=vllm_config, dp_size=dp_size_
        )

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=vllm_config.lora_config is not None,
        )
        self.moe_config: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.quant_config = quant_config

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: QuantizeMethodBase | None = None
        quant_method = (
            UnquantizedFusedMoEMethod(moe)
            if quant_config is None
            else quant_config.get_quant_method(self, prefix)
        )
        if quant_method is None:
            quant_method = UnquantizedFusedMoEMethod(moe)

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        if not self.moe_config.is_act_and_mul:
            # Avoid circular import
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8MoEMethod,
            )

            if not isinstance(
                quant_method, (UnquantizedFusedMoEMethod, ModelOptFp8MoEMethod)
            ):
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for unquantized "
                    "and ModelOpt FP8 moe for now"
                )
            if not current_platform.is_cuda():
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for CUDA for now"
                )

        if self.enable_eplb:
            from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

            if not isinstance(quant_method, (Fp8MoEMethod, UnquantizedFusedMoEMethod)):
                # TODO: Add support for additional quantization methods.
                # The implementation for other quantization methods does not
                # contain essential differences, but the current quant API
                # design causes duplicated work when extending to new
                # quantization methods, so I'm leaving it for now.
                # If you plan to add support for more quantization methods,
                # please refer to the implementation in `Fp8MoEMethod`.
                raise NotImplementedError(
                    "EPLB is only supported for FP8 quantization for now."
                )

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": self.global_num_experts,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return None

    @property
    def gate(self) -> torch.nn.Module | None:
        return None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (
            self.moe_quant_config is not None
            and self.moe_quant_config.quant_dtype == "nvfp4"
            and self.moe_config.use_flashinfer_cutlass_kernels
        )

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or (self.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
        )

    @property
    def is_internal_router(self) -> bool:
        # By default, router/gate is called before FusedMoE forward pass
        return False

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self.expert_map is not None
        with self.expert_map.device:
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._init_aiter_shared_experts_topK_buffer(
                vllm_config=get_current_vllm_config(), dp_size=get_dp_group().world_size
            )

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
        self,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        param: torch.Tensor,
        tp_rank: int,
    ):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def _init_aiter_shared_experts_topK_buffer(
        self, vllm_config: VllmConfig, dp_size: int
    ):
        if is_rocm_aiter_fusion_shared_expert_enabled():
            if self.num_fused_shared_experts > 0:
                init_aiter_topK_meta_data(
                    n_routed_experts=self.global_num_experts,
                    n_shared_experts=self.num_fused_shared_experts,
                    top_k=self.top_k,
                    tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                    tp_size=self.ep_size if self.use_ep else self.tp_size,
                    shared_experts_score=1.0,
                    max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens
                    * dp_size,
                    is_EP=self.use_ep,
                )
            self.local_num_experts += self.num_fused_shared_experts

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        allow_flashinfer = getattr(self.quant_method, "allow_flashinfer", False)
        moe_backend = getattr(self.quant_method, "flashinfer_moe_backend", None)

        use_global_sf = (
            allow_flashinfer
            and is_flashinfer_supporting_global_sf(moe_backend)
            and "input_scale" in weight_name
            and quant_method_name == "ModelOptNvFp4FusedMoE"
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=param,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                success = self.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded %s for expert %d into %s",
                        param_name,
                        expert_id,
                        self.layer_name,
                    )
                    yield param_name

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        weights = list(self.named_parameters())
        assert all(weight.is_contiguous() for _, weight in weights)

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith("_shared_experts.")
            # exclude parameters from non-expert submodules (e.g. gate/shared)
            and not name.startswith("_gate.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

        if self.moe_quant_config is None:
            self.moe_quant_config = self.quant_method.moe_quant_config

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.vllm_config.parallel_config.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.hidden_size)
            logits_shape = (2, moe.max_num_tokens, self.logical_num_experts)
        else:
            states_shape = (moe.max_num_tokens, self.hidden_size)
            logits_shape = (moe.max_num_tokens, self.logical_num_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        indices_type: torch.dtype | None = None,
        enable_eplb: bool = False,
        expert_map: torch.Tensor | None = None,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
        global_num_experts: int | None = None,
        zero_expert_num: int | None = None,
        zero_expert_type: str | None = None,
        num_fused_shared_experts: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids, zero_expert_result)
                (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                The weights, expert ids, and zero expert computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            if is_rocm_aiter_moe_enabled():
                if not is_rocm_aiter_fusion_shared_expert_enabled():
                    assert num_fused_shared_experts == 0
                grouped_topk_impl = partial(
                    grouped_topk_aiter,
                    num_fused_shared_experts=num_fused_shared_experts,
                )
            else:
                grouped_topk_impl = grouped_topk
            topk_weights, topk_ids = grouped_topk_impl(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
            )
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=e_score_correction_bias.data,
                topk=top_k,
                renormalize=renormalize,
            )
            if routed_scaling_factor is not None:
                topk_weights *= routed_scaling_factor
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None

            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        assert topk_ids.dtype == indices_type or indices_type is None

        # Compute zero expert result if needed
        if (
            zero_expert_num is not None
            and zero_expert_num > 0
            and zero_expert_type is not None
            and global_num_experts is not None
        ):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        return topk_weights, topk_ids, zero_expert_result

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        assert self.quant_method is not None
        return (
            self.quant_method.fused_experts is not None
            and self.quant_method.fused_experts.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        if self.shared_experts is None:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
            return fused_output[..., :og_hidden_states]
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits
                )
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
            return (
                shared_output[..., :og_hidden_states],
                fused_output[..., :og_hidden_states],
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # If there are shared experts but we are not using a modular kernel,
            # the shared experts must be called here
            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if self.shared_experts_stream is not None:
                    # For chunked, we start the shared experts stream here
                    # (Note that no concurrency with the router/gate)
                    self.shared_experts_stream.wait_stream(current_stream())

                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that staged_hidden_states clone() is necessary
                        # here to avoid conflict with the main stream
                        shared_output = self.shared_experts(
                            staged_hidden_states.clone()
                        )
                else:
                    shared_output = self.shared_experts(staged_hidden_states)

            else:
                shared_output = None

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=staged_hidden_states,
                router_logits=staged_router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not is_rocm_aiter_moe_enabled()
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                # Here we finish the shared experts stream
                if self.shared_experts_stream is not None:
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                assert self.shared_experts is None
                final_hidden_states, zero_expert_result = final_hidden_states
                if zero_expert_result is not None:
                    final_hidden_states += zero_expert_result

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

        self.ensure_moe_quant_config_init()
        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not isinstance(self.quant_method.fused_experts, FusedMoEModularKernel)
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        if (
            has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
        ):
            # Start the separate shared experts stream here since we want
            # to run in parallel with the router/gate (next op below)
            self.shared_experts_stream.wait_stream(current_stream())

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            return self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )

        do_naive_dispatch_combine: bool = (
            self.dp_size > 1 and not self.quant_method.using_modular_kernel
        )

        # If there are shared experts but we are not using a modular kernel, the
        # shared experts must be called here
        if has_separate_shared_experts:
            assert self.shared_experts is not None

            if self.shared_experts_stream is not None:
                # Run shared experts in parallel on a separate stream
                with torch.cuda.stream(self.shared_experts_stream):
                    # Note that hidden_states clone() is necessary here to avoid
                    # conflict with the main stream
                    shared_output = self.shared_experts(hidden_states.clone())
            else:
                shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not is_rocm_aiter_moe_enabled()
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                # Wait for the parallel shared experts stream to finish here
                if self.shared_experts_stream is not None:
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(
                states: torch.Tensor, do_combine: bool = True
            ) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if (
                    not self.is_sequence_parallel
                    and self.reduce_results
                    and (self.tp_size > 1 or self.ep_size > 1)
                ):
                    states = self.maybe_all_reduce_tensor_model_parallel(states)

                return states

            if self.shared_experts is not None:
                return (
                    reduce_output(final_hidden_states[0], do_combine=False),
                    reduce_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return reduce_output(final_hidden_states) + zero_expert_result
            else:
                return reduce_output(final_hidden_states)

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_",
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}"
        )

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
