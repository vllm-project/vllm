# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


class XPUExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.fused_moe_impl: XpuFusedMoe | None = None
        self._weight_bindings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._layer: RoutedExperts | None = None
        self._lora_ones = torch.empty(0)
        is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
        self._expects_unquantized_inputs = is_xe2_or_xe3

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self._expects_unquantized_inputs

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def _weights_match(self, w1: torch.Tensor, w2: torch.Tensor) -> bool:
        return (
            self.fused_moe_impl is not None
            and self._weight_bindings is not None
            and w1 is self._weight_bindings[0]
            and w2 is self._weight_bindings[1]
        )

    def initialize_weights(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk: int,
        activation: MoEActivation,
        expert_map: torch.Tensor | None,
    ) -> None:
        """Reuse the provider while its source weight bindings remain unchanged."""
        if self._weights_match(w1, w2):
            return
        weight_bindings = (w1, w2)
        if self._layer is not None:
            quant_config = self._layer.quant_method.get_fused_moe_quant_config(
                self._layer
            )
            assert quant_config is not None
            self.quant_config = quant_config
        if (
            self.quant_config is not None
            and self.quant_config.weight_quant_dtype == "mxfp4"
        ):
            w1 = w1.view(torch.float4_e2m1fn_x2)
            w2 = w2.view(torch.float4_e2m1fn_x2)
        # XpuFusedMoe relayouts the loaded [E, N, K] weights and scales
        # in place for this device's grouped GEMM.
        self.fused_moe_impl = XpuFusedMoe(
            w13=w1,
            w13_scales=self.w1_scale,
            w13_bias=self.w1_bias,
            w2=w2,
            w2_scales=self.w2_scale,
            w2_bias=self.w2_bias,
            n_experts_per_token=topk,
            activation=activation.value,
            num_experts=self.moe_config.num_local_experts,
            ep_rank=self.moe_config.ep_rank,
            ep_size=self.moe_config.ep_size,
            expert_map=expert_map,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
        )
        self._weight_bindings = weight_bindings

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.moe_config.is_lora_enabled and apply_router_weight_on_input:
            raise NotImplementedError(
                "XPU MoE LoRA requires routing weights on the output."
            )
        if not self._weights_match(w1, w2):
            self.initialize_weights(w1, w2, topk_ids.size(-1), activation, expert_map)
        assert self.fused_moe_impl is not None
        if self.moe_config.ep_size > 1:
            # Avoid topk_ids = -1, remap them to an
            # expert not owned by this rank so they are skipped.
            sentinel = global_num_experts - 1 if self.moe_config.ep_rank == 0 else 0
            topk_ids = torch.where(topk_ids < 0, sentinel, topk_ids)
        self._apply_experts(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_map,
            a1q_scale,
        )

    def _apply_experts(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
    ) -> None:
        assert self.fused_moe_impl is not None
        hooks = {}
        if self._lora_context is not None:
            hooks = self._lora_hooks(
                self._lora_context,
                hidden_states,
                topk_weights,
                topk_ids,
                expert_map,
            )
        self.fused_moe_impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
            **hooks,
        )

    def _lora_hooks(
        self,
        ctx: MoELoRAContext,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> dict[str, Callable[..., None]]:
        """Builds XpuFusedMoe hooks that add the LoRA deltas in place.

        gemm1_output / act_output / gemm2_output hold one row per
        (token, topk) pair grouped by expert; `row_map` gives each pair's
        row (-1 for skipped pairs), so the LoRA kernels read and accumulate
        into those rows directly.
        """
        impl = self.fused_moe_impl
        assert impl is not None
        num_tokens, top_k = topk_ids.shape
        # The LoRA config lookup only reads the logical [E, N, K] shapes. The
        # stored weights cannot provide them for every recipe (XpuFusedMoe
        # relayouts bf16 to [E, K, N]; int4 is packed), so pass shape-only
        # meta tensors. They allocate no memory, and the real weights held by
        # fused_moe_impl are untouched.
        inter_size = impl.inter_size * impl.inter_size_scale
        hidden_size = hidden_states.shape[-1]
        w1_shape = torch.empty(
            (ctx.local_num_experts, 2 * impl.inter_size, hidden_size), device="meta"
        )
        w2_shape = torch.empty(
            (ctx.local_num_experts, hidden_size, inter_size), device="meta"
        )
        # moe_gather applies the routing weights to base and LoRA outputs.
        if self._lora_ones.shape[0] < num_tokens:
            self._lora_ones = torch.ones_like(topk_weights)
        unweighted = self._lora_ones[:num_tokens]
        w13_meta: tuple = ()

        def activation_hook(gemm1_output: torch.Tensor, row_map: torch.Tensor):
            nonlocal w13_meta
            w13_meta = self.apply_w13_lora(
                ctx,
                y=gemm1_output,
                x=hidden_states,
                topk_ids=topk_ids,
                topk_weights=unweighted,
                expert_map=expert_map,
                w1=w1_shape,
                w2=w2_shape,
                num_tokens=num_tokens,
                top_k_num=top_k,
                row_map=row_map.view(-1),
            )

        def output_hook(
            gemm2_output: torch.Tensor,
            act_output: torch.Tensor,
            row_map: torch.Tensor,
        ):
            sorted_token_ids, expert_ids, num_tokens_post_padded, token_lora_map = (
                w13_meta
            )
            self.apply_w2_lora(
                ctx,
                y=gemm2_output.view(num_tokens, top_k, -1),
                x=act_output,
                topk_weights=unweighted,
                sorted_token_ids_lora=sorted_token_ids,
                expert_ids_lora=expert_ids,
                num_tokens_post_padded_lora=num_tokens_post_padded,
                token_lora_mapping=token_lora_map,
                num_tokens=num_tokens,
                w1=w1_shape,
                w2=w2_shape,
                top_k_num=top_k,
                row_map=row_map.view(-1),
            )

        return dict(activation_hook=activation_hook, output_hook=output_hook)


class XPUExpertsFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare layout before reload restores the captured parameter storage."""
        self._layer = cast("RoutedExperts", layer)
        self.initialize_weights(
            self._layer.w13_weight,
            self._layer.w2_weight,
            self.moe_config.experts_per_token,
            self._layer.activation,
            self._layer.expert_map,
        )
        # Reload can restore different Parameters; post-load or first use
        # rebinds only when those final objects differ from this preparation.


class XPUExpertsMxFp8(XPUExpertsFp8):
    @staticmethod
    def supports_lora() -> bool:
        return False

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        assert quant_config.quant_dtype == "mxfp8"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp8Static, None),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsBlockFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsWNA16(XPUExperts):
    """W4A16 INT4-symmetric MoE backed by `xpu_fused_moe(is_int4=True)`.

    Weight layout when `is_int4=True` (per `xpu_fused_moe` docstring):
        w13: [num_experts, 2*inter_size, hidden_size]   contiguous int4-packed
        w13_scales: [num_experts, 2*inter_size, hidden_size // group_size]
        w2:  [num_experts, hidden_size, inter_size]     contiguous int4-packed
        w2_scales:  [num_experts, hidden_size, inter_size // group_size]

    Pairs with `INCXPULinearMethod` for the linear layers; together they
    cover full-attn + MoE on Intel XPU end-to-end without IPEX.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kInt4Static, None),
            (kInt4Static32, None),
        )


class XPUExpertsMxFp4(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # K = a1q.size(-1). When activations are pre-quantized packed mxfp4,
        # K is the packed hidden_size (= logical / 2); the kernel output is at
        # logical hidden_size (2 * K). When unquantized (bf16), K is already
        # the logical size.
        logical_K = K if self.expects_unquantized_inputs else 2 * K
        return (0,), (0,), (M, logical_K)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A
