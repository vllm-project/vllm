# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    make_wna16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    MoeWNA16Method,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig

logger = init_logger(__name__)


class INCARKWNA16MoEMethod(MoeWNA16Method):
    def __init__(
        self,
        quant_config: MoeWNA16Config,
        moe: FusedMoEConfig,
    ) -> None:
        if quant_config.weight_bits != 4 or quant_config.has_zp:
            raise NotImplementedError(
                "ARK WNA16 MoE only supports symmetric int4 W4A16 for now."
            )

        super().__init__(quant_config, moe)

        from .inc_ark_ops import get_ark_state

        is_available, error_str, ark, _ = get_ark_state()
        xpu_lib = getattr(ark, "xpu_lib", None) if ark is not None else None
        has_moe_kernel = (
            is_available
            and ark is not None
            and hasattr(ark, "moe")
            and xpu_lib is not None
            and hasattr(xpu_lib, "moe_gemm_decode")
            and hasattr(xpu_lib, "moe_gemm_prefill")
        )
        if not has_moe_kernel:
            reason = error_str or "ARK MoE kernels are unavailable."
            raise ImportError(f"Failed to initialize ARK WNA16 MoE. {reason}")

        self.ark = ark
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.local_to_global_experts: tuple[int, ...] | None = None
        self.group_size: int | None = None

        logger.info_once("Using ARK XPU WNA16 MoE kernel.")

    def process_weights_after_loading(self, layer) -> None:
        # ARK int4 MoE with asym=False expects signed nibbles [-8, 7].
        # Loaded GPTQ weights are unsigned nibbles [0, 15], so flip the sign
        # bit of each nibble, equivalent to q - 8 in 4-bit two's-complement.
        w13_qweight = torch.bitwise_xor(
            layer.w13_qweight.detach().contiguous(),
            0x88,
        )
        w2_qweight = torch.bitwise_xor(
            layer.w2_qweight.detach().contiguous(),
            0x88,
        )

        replace_parameter(layer, "w13_qweight", w13_qweight)
        replace_parameter(layer, "w2_qweight", w2_qweight)
        replace_parameter(layer, "w13_scales", layer.w13_scales.detach().contiguous())
        replace_parameter(layer, "w2_scales", layer.w2_scales.detach().contiguous())

        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        num_local_experts = layer.w13_weight.shape[0]
        if layer.expert_map is None:
            self.local_to_global_experts = tuple(range(num_local_experts))
        else:
            local_to_global = [-1] * num_local_experts
            expert_map_cpu = layer.expert_map.detach().cpu()
            for global_expert_id, local_expert_id_tensor in enumerate(expert_map_cpu):
                local_expert_id = int(local_expert_id_tensor)
                if 0 <= local_expert_id < num_local_experts:
                    local_to_global[local_expert_id] = global_expert_id
            self.local_to_global_experts = tuple(local_to_global)

        self.group_size = layer.group_size

    def get_fused_moe_quant_config(
        self,
        layer,
    ) -> FusedMoEQuantConfig | None:
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            group_size=layer.group_size,
            num_bits=self.quant_config.weight_bits,
        )

    def _ark_moe(
        self,
        activations: torch.Tensor,
        weights: torch.Tensor,
        scales: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        return self.ark.moe(
            activations.contiguous(),
            weights,
            num_tokens_per_expert,
            scales=scales,
            zeros=None,
            weight_bits=self.quant_config.weight_bits,
            group_size=group_size,
            asym=False,
            phase="auto",
        )

    def _make_compact_inputs(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del layer

        assert self.local_to_global_experts is not None
        local_to_global = self.local_to_global_experts

        token_indices_per_expert: list[torch.Tensor] = []
        topk_slots_per_expert: list[torch.Tensor] = []
        num_tokens_per_expert_list: list[int] = []

        for global_expert_id in local_to_global:
            if global_expert_id < 0:
                num_tokens_per_expert_list.append(0)
                continue

            token_indices, topk_slots = torch.where(topk_ids == global_expert_id)
            num_tokens = token_indices.numel()
            num_tokens_per_expert_list.append(num_tokens)

            if num_tokens > 0:
                token_indices_per_expert.append(token_indices)
                topk_slots_per_expert.append(topk_slots)

        num_tokens_per_expert = torch.tensor(
            num_tokens_per_expert_list,
            dtype=torch.int32,
            device=x.device,
        )

        if not token_indices_per_expert:
            empty_indices = torch.empty((0,), dtype=torch.long, device=x.device)
            compact_x = x.new_empty((0, x.shape[-1]))
            return compact_x, num_tokens_per_expert, empty_indices, empty_indices

        token_indices = torch.cat(token_indices_per_expert)
        topk_slots = torch.cat(topk_slots_per_expert)
        compact_x = x.index_select(0, token_indices)

        return compact_x, num_tokens_per_expert, token_indices, topk_slots

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input

        assert self.group_size is not None
        compact_x, num_tokens_per_expert, token_indices, topk_slots = (
            self._make_compact_inputs(
                x,
                topk_ids,
                topk_weights,
                layer,
            )
        )

        output = torch.zeros_like(x)
        if compact_x.numel() == 0:
            return output

        if layer.apply_router_weight_on_input:
            if topk_ids.shape[1] != 1:
                raise NotImplementedError(
                    "apply_router_weight_on_input is only supported for topk=1."
                )
            route_weights = topk_weights[token_indices, topk_slots].to(compact_x.dtype)
            compact_x = compact_x * route_weights.unsqueeze(-1)

        compact_w13 = self._ark_moe(
            compact_x,
            layer.w13_weight,
            layer.w13_scales,
            num_tokens_per_expert,
            self.group_size,
        )

        activation = layer.activation
        activated_size = (
            compact_w13.shape[-1] // 2 if activation.is_gated else compact_w13.shape[-1]
        )
        compact_activated = compact_w13.new_empty(
            (compact_w13.shape[0], activated_size)
        )
        apply_moe_activation(
            activation,
            compact_activated,
            compact_w13,
        )

        compact_out = self._ark_moe(
            compact_activated,
            layer.w2_weight,
            layer.w2_scales,
            num_tokens_per_expert,
            self.group_size,
        )

        if not layer.apply_router_weight_on_input:
            route_weights = topk_weights[token_indices, topk_slots].to(
                compact_out.dtype
            )
            compact_out = compact_out * route_weights.unsqueeze(-1)

        output.index_add_(0, token_indices, compact_out.to(output.dtype))
        return output


class INCWNA16MoEScheme:
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.layer_config = layer_config

    def get_method(self, layer: torch.nn.Module):
        if current_platform.is_cpu():
            return self._build_cpu_method(layer)
        if self.layer_config.is_gptq:
            return self._build_gptq_method(layer)
        if self.layer_config.is_awq:
            return self._build_awq_method(layer)
        raise NotImplementedError(
            f"WNA16 MoE does not support config {self.layer_config}"
        )

    def _build_cpu_method(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.fused_moe import (
            UnquantizedFusedMoEMethod,
        )

        return UnquantizedFusedMoEMethod(layer.moe_config)

    def _build_gptq_method(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.quantization.auto_gptq import (
            AutoGPTQMoEMethod,
        )
        from vllm.model_executor.layers.quantization.moe_wna16 import (
            MoeWNA16Config,
            MoeWNA16Method,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_moe_marlin_supports_layer,
        )

        use_marlin = (self.layer_config.bits, self.layer_config.sym) in {
            (4, True),
            (8, True),
        } and check_moe_marlin_supports_layer(
            layer,
            self.layer_config.group_size,
        )

        if use_marlin:
            return AutoGPTQMoEMethod(
                AutoGPTQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    desc_act=False,
                    is_sym=self.layer_config.sym,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                ),
                layer.moe_config,
            )

        moe_config = MoeWNA16Config.from_config(
            {
                "quant_method": "gptq",
                "bits": self.layer_config.bits,
                "group_size": self.layer_config.group_size,
                "sym": self.layer_config.sym,
                "lm_head": False,
            }
        )
        return MoeWNA16Method(moe_config, layer.moe_config)

    def _build_awq_method(self, layer: torch.nn.Module):
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQMoEMethod
        from vllm.model_executor.layers.quantization.moe_wna16 import (
            MoeWNA16Config,
            MoeWNA16Method,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_moe_marlin_supports_layer,
        )

        use_marlin = self.layer_config.bits in (
            4,
            8,
        ) and check_moe_marlin_supports_layer(
            layer,
            self.layer_config.group_size,
        )

        if use_marlin:
            return AutoAWQMoEMethod(
                AutoAWQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=self.layer_config.group_size,
                    zero_point=not self.layer_config.sym,
                    lm_head_quantized=False,
                    modules_to_not_convert=[],
                    full_config={},
                ),
                layer.moe_config,
            )

        moe_config = MoeWNA16Config.from_config(
            {
                "quant_method": "awq",
                "bits": self.layer_config.bits,
                "group_size": self.layer_config.group_size,
                "zero_point": not self.layer_config.sym,
                "lm_head": False,
            }
        )
        return MoeWNA16Method(moe_config, layer.moe_config)
