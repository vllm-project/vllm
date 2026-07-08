# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsMxfp4,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsMxFp4
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    make_mxfp4_moe_kernel,
    make_mxfp4_moe_quant_config,
)
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)


class INCXPUWNA16MoEMethod(MoeWNA16Method):
    """W4A16 INT4-symmetric group MoE executed by the native XPU kernel.

    Inherits weight creation / loading (GPTQ-named uint8 layout) from
    :class:`MoeWNA16Method` and overrides :meth:`apply` to dispatch to
    :class:`XPUExpertsWNA16`, which wraps ``xpu_fused_moe(is_int4=True)``.
    """

    def __init__(self, quant_config, moe) -> None:
        super().__init__(quant_config, moe)
        self._xpu_experts = None

    def _get_xpu_experts(self):
        if self._xpu_experts is None:
            from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
                XPUExpertsWNA16,
            )

            assert self.moe_quant_config is not None, (
                "moe_quant_config must be initialised before apply(); it is "
                "populated from get_fused_moe_quant_config() after weight load."
            )
            self._xpu_experts = XPUExpertsWNA16(self.moe, self.moe_quant_config)
        return self._xpu_experts

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:

        experts = self._get_xpu_experts()
        output = torch.empty_like(x)
        # XPUExpertsWNA16 runs the full fused MoE inside xpu_fused_moe, so the
        # modular workspaces are unused; pass empty placeholders.
        empty = x.new_empty(0)
        experts.apply(
            output=output,
            hidden_states=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            a1q_scale=None,
            a2_scale=None,
            workspace13=empty,
            workspace2=empty,
            expert_tokens_meta=None,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )
        return output


class INCMxfp4MoEMethod(FusedMoEMethodBase):
    """W4A4 MXFP4 group MoE for AutoRound ``auto_round:llm_compressor`` exports.

    Registers the packed MXFP4 layout (uint8 ``weight_packed`` + uint8 E8M0
    ``weight_scale``, ``group_size=32``) and dispatches the fused MoE to the
    best backend for the current device: CUTLASS (true W4A4 on supported
    GPUs), the native XPU kernel, or Marlin weight-only as a fallback. The
    per-expert ``gate_proj`` / ``up_proj`` / ``down_proj`` tensors are folded
    into the stacked ``w13`` / ``w2`` parameters by ``make_expert_params_mapping``.
    """

    def __init__(self, moe) -> None:
        super().__init__(moe)
        self.group_size = 32
        self.mxfp4_backend = Mxfp4MoeBackend.MARLIN
        # Use CUTLASS if the device supports it, otherwise XPU on Intel GPUs,
        # otherwise fall back to weight-only Marlin.
        self.use_cutlass_mxfp4 = CutlassExpertsMxfp4._supports_current_device()
        self.experts_cls: type
        if self.use_cutlass_mxfp4:
            logger.info_once("Using CutlassExpertsMxfp4 for AutoRound MXFP4 MoE")
            self.experts_cls = CutlassExpertsMxfp4
        elif current_platform.is_xpu():
            self.mxfp4_backend = Mxfp4MoeBackend.XPU
            self.experts_cls = XPUExpertsMxFp4
            logger.info_once("Using XPUExpertsMxFp4 for AutoRound MXFP4 MoE on XPU")
        else:
            logger.info_once("Using MarlinExperts for AutoRound MXFP4 MoE")
            self.experts_cls = MarlinExperts

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype

        # gate + up fused on the output dim; two FP4 packed per input byte.
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
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
                intermediate_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Per-group E8M0 block scales (group_size=32), stored as uint8.
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
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
        if self.use_cutlass_mxfp4:
            # W4A4: both weights and activations quantized to MXFP4.
            return mxfp4_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        # Weight-only (Marlin) or native XPU kernel.
        return make_mxfp4_moe_quant_config(
            mxfp4_backend=self.mxfp4_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
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

        if self.use_cutlass_mxfp4:
            # Swizzle weight scales from flat checkpoint layout [E, N, K//32]
            # to the CUTLASS tiled layout.
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
        elif current_platform.is_xpu():
            # The XPU fused-MoE kernel consumes the packed layout directly; no
            # swizzle / repack / transpose is required.
            pass
        else:
            logger.warning_once(
                "This device lacks native FP4 compute; using weight-only FP4 "
                "via the Marlin kernel, which may reduce performance for "
                "compute-heavy workloads."
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
