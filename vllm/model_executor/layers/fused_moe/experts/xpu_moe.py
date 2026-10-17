# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
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

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


def prepare_fp8_moe_layer_for_xpu(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if w13_scale is not None and w13_scale.ndim == 3:
        w13_scale = w13_scale.transpose(-1, -2).contiguous()
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = w2_scale.transpose(-1, -2).contiguous()
    return (
        w13.transpose(-1, -2).contiguous(),
        w13_scale,
        w2.transpose(-1, -2).contiguous(),
        w2_scale,
    )


class XPUExpertsBase(mk.FusedMoEExpertsModular):
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
        is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
        if not is_xe2_or_xe3:
            raise NotImplementedError(
                "XPUExperts is only supported on Intel Xe2/Xe3 GPUs"
            )
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
        if self.fused_moe_impl is None:
            topk = topk_ids.size(-1)
            if (
                self.quant_config is not None
                and self.quant_config.weight_quant_dtype == "mxfp4"
            ):
                w1 = w1.view(torch.float4_e2m1fn_x2)
                w2 = w2.view(torch.float4_e2m1fn_x2)
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
        assert self.fused_moe_impl is not None
        if self.moe_config.ep_size > 1:
            # Avoid topk_ids = -1, remap them to an
            # expert not owned by this rank so they are skipped.
            sentinel = global_num_experts - 1 if self.moe_config.ep_rank == 0 else 0
            topk_ids = torch.where(topk_ids < 0, sentinel, topk_ids)
        self._apply(
            output,
            hidden_states,
            topk_weights,
            topk_ids,
            a1q_scale,
            workspace13,
            workspace2,
            apply_router_weight_on_input,
        )

    def _apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert self.fused_moe_impl is not None
        self.fused_moe_impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
        )


class XPUExperts(LoRAExpertsMixin, XPUExpertsBase):
    def _apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        if self._lora_context is not None:
            if apply_router_weight_on_input or a1q_scale is not None:
                raise NotImplementedError(
                    "XPU MoE LoRA requires unquantized inputs "
                    "and output routing weights"
                )
            self._apply_with_lora(
                output, hidden_states, topk_weights, topk_ids, workspace13, workspace2
            )
            return
        super()._apply(
            output,
            hidden_states,
            topk_weights,
            topk_ids,
            a1q_scale,
            workspace13,
            workspace2,
            apply_router_weight_on_input,
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
        if not self.moe_config.is_lora_enabled:
            return (0,), (0,), (M, K)
        rows = M * topk
        act_dim = self.adjust_N_for_activation(N, activation)
        # GEMM2 input includes a sink row for nonlocal routes. Its output must
        # also be disjoint from the final output, which can alias workspace13.
        gemm2_offset = max((rows + 1) * act_dim, M * K)
        workspace13 = (max(rows * N, gemm2_offset + rows * K),)
        workspace2 = (max(rows * N, rows * (act_dim + K)),)
        return workspace13, workspace2, (M, K)

    def _apply_with_lora(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
    ) -> None:
        ctx = self._lora_context
        impl = self.fused_moe_impl
        assert ctx is not None and impl is not None
        if ctx.fully_sharded:
            raise NotImplementedError("Fully sharded LoRA is not supported by XPU MoE")
        num_tokens, hidden_size = hidden_states.shape
        topk = topk_ids.shape[1]
        rows = num_tokens * topk
        if rows == 0:
            return
        counts = torch.zeros(
            impl.num_experts, dtype=torch.int32, device=hidden_states.device
        )
        routing = torch.empty_like(topk_ids, dtype=torch.int32)
        remapped = _resize_cache(workspace13, (rows, hidden_size))
        torch.ops._moe_C.remap_hidden_states(
            hidden_states=hidden_states,
            hidden_states_scales=None,
            remapped_hidden_states=remapped,
            remapped_hidden_states_scales=None,
            expert_map=impl.expert_map,
            rows_per_expert=counts,
            unpermuted_row_to_permuted_row=routing,
            topk_ids=topk_ids,
            total_experts_num=impl.total_experts_num,
            local_experts_num=impl.local_experts_num,
        )
        grouped = _resize_cache(workspace2, (rows, 2 * impl.inter_size))
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=remapped,
            ptr_A_scale=None,
            ptr_B=impl.w13,
            ptr_B_scale=impl.gemm1_wei_scales,
            ptr_bias=impl.w13_bias,
            ptr_D=grouped,
            rows_per_expert=counts,
            N=2 * impl.inter_size,
            K=hidden_size,
            num_experts=impl.num_experts,
        )
        # Native GEMMs group by expert; the LoRA kernels address original
        # token/top-k rows. Nonlocal routes must never read undefined data.
        route = routing.flatten().long()
        local = route >= 0
        original = _resize_cache(workspace13, grouped.shape)
        torch.index_select(grouped, 0, route.clamp_min(0), out=original)
        original.masked_fill_(~local[:, None], 0)
        lora_width = sum(b.shape[-2] for b in ctx.w13_lora_b_stacked)
        delta = _resize_cache(workspace2, (rows, lora_width)).zero_()
        w1 = impl.w13 if impl.is_int4 else impl.w13.transpose(-1, -2)
        w2 = impl.w2 if impl.is_int4 else impl.w2.transpose(-1, -2)
        metadata = self.apply_w13_lora(
            ctx,
            y=delta,
            x=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=impl.expert_map,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=topk,
            add_inputs=False,
        )
        if self.moe_config.activation == MoEActivation.SWIGLUOAI:
            # Both packed adapters and the two-slice helper emit interleaved
            # gate/up channels. Padding follows the last logical pair.
            original[:, :lora_width].add_(delta)
        elif impl.inter_size_scale == 1:
            # Each gate/up half may have independent padding in the base weights.
            width = lora_width // 2
            original[:, :width].add_(delta[:, :width])
            original[:, impl.inter_size : impl.inter_size + width].add_(
                delta[:, width:]
            )
        else:
            original[:, :lora_width].add_(delta)
        if impl.gemm1_clamp_limit is not None and impl.gemm1_clamp_limit > 0:
            original[:, : impl.inter_size].clamp_(max=impl.gemm1_clamp_limit)
            original[:, impl.inter_size :].clamp_(
                min=-impl.gemm1_clamp_limit, max=impl.gemm1_clamp_limit
            )
        activated = _resize_cache(
            workspace2, (rows, impl.inter_size * impl.inter_size_scale)
        )
        impl.act_func(activated, original)
        # The extra row receives nonlocal routes; valid grouped rows are unique.
        # This avoids a device-to-host synchronization from boolean indexing.
        grouped_input = _resize_cache(workspace13, (rows + 1, activated.shape[1]))
        grouped_input.index_copy_(0, route.masked_fill(~local, rows), activated)
        gemm2_offset = max(grouped_input.numel(), output.numel())
        grouped_output = _resize_cache(
            workspace13.flatten()[gemm2_offset:], (rows, hidden_size)
        )
        torch.ops._xpu_C.cutlass_grouped_gemm_interface(
            ptr_A=grouped_input[:rows],
            ptr_A_scale=None,
            ptr_B=impl.w2,
            ptr_B_scale=impl.gemm2_wei_scales,
            ptr_bias=impl.w2_bias,
            ptr_D=grouped_output,
            rows_per_expert=counts,
            N=hidden_size,
            K=activated.shape[1],
            num_experts=impl.num_experts,
        )
        # Preserve the native FP32 routing-weight multiply and accumulation.
        # Multiplying each route in BF16/FP16 first adds an extra rounding.
        torch.ops._moe_C.moe_gather(
            output, grouped_output, topk_weights, routing, impl.num_experts
        )
        combined = _resize_cache(
            workspace2.flatten()[activated.numel() :],
            (num_tokens, topk, hidden_size),
        ).zero_()
        self.apply_w2_lora(
            ctx,
            y=combined,
            x=activated[:, : ctx.w2_lora_a_stacked[0].shape[-1]].contiguous(),
            topk_weights=topk_weights,
            sorted_token_ids_lora=metadata[0],
            expert_ids_lora=metadata[1],
            num_tokens_post_padded_lora=metadata[2],
            token_lora_mapping=metadata[3],
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=topk,
        )
        output.copy_(combined.sum(dim=1, dtype=torch.float32).add_(output))


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


class XPUExpertsMxFp4(XPUExpertsBase):
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
