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
            and hasattr(ark, "MoeSymmetricGemm")
            and xpu_lib is not None
        )
        if not has_moe_kernel:
            reason = error_str or "ARK MoE kernels are unavailable."
            raise ImportError(f"Failed to initialize ARK WNA16 MoE. {reason}")

        assert ark is not None
        self.ark = ark
        self.ark_moe_op = ark.moe
        self.weight_bits = quant_config.weight_bits
        self.remap_hidden_states_op = None
        self.moe_gather_op = None
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.group_size: int | None = None
        self.local_num_experts: int | None = None
        self.global_num_experts: int | None = None
        self.expert_map: torch.Tensor | None = None
        self.w13_weight: torch.Tensor | None = None
        self.w13_scales: torch.Tensor | None = None
        self.w2_weight: torch.Tensor | None = None
        self.w2_scales: torch.Tensor | None = None
        self.rows_per_expert: torch.Tensor | None = None
        self.unpermuted_row_to_permuted_row: torch.Tensor | None = None
        self.router_weight_ones: torch.Tensor | None = None
        self.apply_router_weight_on_input: bool | None = None
        self.router_weights_fn = self._keep_router_weights
        self.activation = None
        self.inter_size: int = 0
        self.inter_size_scale: int = 1
        self.w13_moe = None
        self.w2_moe = None

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
        self.group_size = layer.group_size
        self.local_num_experts = num_local_experts
        self.global_num_experts = layer.global_num_experts
        self.expert_map = layer.expert_map
        self.w13_weight = layer.w13_weight
        self.w13_scales = layer.w13_scales
        self.w2_weight = layer.w2_weight
        self.w2_scales = layer.w2_scales
        self.rows_per_expert = torch.empty(
            (num_local_experts,),
            dtype=torch.int32,
            device=layer.w13_weight.device,
        )
        self.unpermuted_row_to_permuted_row = None
        self.router_weight_ones = None
        self.apply_router_weight_on_input = layer.apply_router_weight_on_input
        if self.apply_router_weight_on_input:
            if layer.top_k != 1:
                raise NotImplementedError(
                    "apply_router_weight_on_input is only supported for topk=1."
                )
            self.router_weights_fn = self._apply_router_weight_to_input
        else:
            self.router_weights_fn = self._keep_router_weights
        self.activation = layer.activation
        self.inter_size = layer.w13_weight.shape[-2] // 2
        self.inter_size_scale = 1 if layer.activation.is_gated else 2
        self.remap_hidden_states_op = torch.ops._moe_C.remap_hidden_states
        self.moe_gather_op = torch.ops._moe_C.moe_gather

        self.w13_moe = self.ark.MoeSymmetricGemm.prepare(
            layer.w13_weight,
            layer.w13_scales,
            weight_bits=4,
            group_size=layer.group_size,
            activation_dtype=layer.w13_scales.dtype,
        )

        self.w2_moe = self.ark.MoeSymmetricGemm.prepare(
            layer.w2_weight,
            layer.w2_scales,
            weight_bits=4,
            group_size=layer.group_size,
            activation_dtype=layer.w2_scales.dtype,
        )

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

    def _get_rows_per_expert(
        self,
        local_num_experts: int,
        device: torch.device,
    ) -> torch.Tensor:
        rows_per_expert = self.rows_per_expert
        if rows_per_expert is None or rows_per_expert.device != device:
            rows_per_expert = torch.empty(
                (local_num_experts,),
                dtype=torch.int32,
                device=device,
            )
            self.rows_per_expert = rows_per_expert
        rows_per_expert.zero_()
        return rows_per_expert

    def _get_unpermuted_row_to_permuted_row(
        self,
        num_rows: int,
        topk: int,
        device: torch.device,
    ) -> torch.Tensor:
        mapping = self.unpermuted_row_to_permuted_row
        if (
            mapping is None
            or mapping.device != device
            or mapping.shape[0] < num_rows
            or mapping.shape[1] != topk
        ):
            mapping = torch.empty(
                (num_rows, topk),
                dtype=torch.int32,
                device=device,
            )
            self.unpermuted_row_to_permuted_row = mapping
        return mapping[:num_rows]

    def _get_router_weight_ones(
        self,
        num_rows: int,
        topk: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        router_weight_ones = self.router_weight_ones
        if (
            router_weight_ones is None
            or router_weight_ones.device != device
            or router_weight_ones.dtype != dtype
            or router_weight_ones.shape[0] < num_rows
            or router_weight_ones.shape[1] != topk
        ):
            router_weight_ones = torch.ones(
                (num_rows, topk),
                dtype=dtype,
                device=device,
            )
            self.router_weight_ones = router_weight_ones
        return router_weight_ones[:num_rows]

    def _keep_router_weights(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        num_rows: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del num_rows, topk
        return x, topk_weights

    def _apply_router_weight_to_input(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        num_rows: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x * topk_weights.to(x.dtype)
        return x, self._get_router_weight_ones(
            num_rows,
            topk,
            topk_weights.dtype,
            topk_weights.device,
        )

    def apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del layer, shared_experts, shared_experts_input

        num_rows, hidden_size = x.shape
        topk = topk_ids.shape[1]

        x, gather_topk_weights = self.router_weights_fn(
            x,
            topk_weights,
            num_rows,
            topk,
        )

        num_moe_inputs = num_rows * topk
        output = torch.empty_like(x)

        if num_moe_inputs == 0:
            return output

        remapped_hidden_states = torch.empty(
            (num_moe_inputs, hidden_size),
            dtype=x.dtype,
            device=x.device,
        )
        rows_per_expert = self._get_rows_per_expert(self.local_num_experts, x.device)
        unpermuted_row_to_permuted_row = (
            self._get_unpermuted_row_to_permuted_row(num_rows, topk, x.device)
        )

        self.remap_hidden_states_op(
            hidden_states=x,
            hidden_states_scales=None,
            remapped_hidden_states=remapped_hidden_states,
            remapped_hidden_states_scales=None,
            expert_map=self.expert_map,
            rows_per_expert=rows_per_expert,
            unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
            topk_ids=topk_ids,
            total_experts_num=self.global_num_experts,
            local_experts_num=self.local_num_experts,
        )

        gemm1_output = self.w13_moe.apply(
            remapped_hidden_states,
            rows_per_expert,
            phase="auto",
        )

        act_output = gemm1_output.new_empty(
            (gemm1_output.shape[0], self.inter_size * self.inter_size_scale)
        )
        
        apply_moe_activation(
            self.activation,
            act_output,
            gemm1_output,
        )

        gemm2_output = self.w2_moe.apply(
            act_output,
            rows_per_expert,
            phase="auto",
        )

        self.moe_gather_op(
            output,
            gemm2_output,
            gather_topk_weights,
            unpermuted_row_to_permuted_row,
            self.local_num_experts,
        )
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
