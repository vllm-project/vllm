# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based MoE expert implementations."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.lora_experts_mixin import (
    LoRAExpertsMixin,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    invoke_fused_moe_wna16_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    is_deep_gemm_e8m0_used,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


class TritonExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Whether quantized MOE runs natively, or through
        # higher-precision + activation QDQ.
        self.quantization_emulation = False
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # INT8 requires at least 7.5 (Turing).
        device_supports_int8 = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((7, 5))
        )

        supported: list[tuple[QuantKey | None, QuantKey | None]] = [(None, None)]
        if device_supports_int8:
            supported.append((kInt8StaticChannelSym, kInt8DynamicTokenSym))
        if current_platform.supports_fp8():
            supported += [
                (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                (kFp8StaticChannelSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8StaticTensorSym),
                (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            ]
        return (weight_key, activation_key) in supported

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    @staticmethod
    def _supports_batch_invariance():
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def activation(
        self, activation: MoEActivation, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        gemm1_clamp_limit = self.quant_config.gemm1_clamp_limit
        if activation == MoEActivation.SILU and gemm1_clamp_limit is not None:
            swiglu_limit_func(output, input, float(gemm1_clamp_limit))
            return

        super().activation(activation, output, input)

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
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M, topk, max(activation_out_dim, K))
        workspace2 = (M, topk, max(N, K))
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
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        cache2_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, cache2_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                global_num_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
            )
        )

        # LoRA w13: applied to intermediate_cache1 before activation. When
        # the LoRA layer requested a dual-stream schedule, we run base w13
        # GEMM on the default stream and the LoRA fast-path on aux_stream;
        # the LoRA writes its delta into a fresh zero buffer (add_inputs=
        # False) and we sum it into intermediate_cache1 after both finish.

        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context

        def _base_w13_fn():
            invoke_fused_moe_triton_kernel(
                hidden_states,
                w1,
                intermediate_cache1,
                a1q_scale if a1q_scale is not None else self.a1_scale,
                self.w1_scale,
                None,  # topk_weights
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                False,  # mul_routed_weights
                top_k_num,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w1_bias,
            )

        if lora_context is not None and lora_context.aux_stream is not None:
            # add_inputs=False: kernel overwrites lora_delta_w13. zeros (not
            # empty) so untouched rows -- e.g. blocks where every program
            # early-exits because lora_id<0 -- stay at zero and the trailing
            # add_() is a no-op there.
            lora_delta_w13 = torch.zeros_like(intermediate_cache1)

            def _lora_w13_fn():
                return self.apply_w13_lora(
                    lora_context,
                    y=lora_delta_w13,
                    x=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=num_tokens,
                    top_k_num=top_k_num,
                    add_inputs=False,
                )

            assert lora_context.events is not None
            _, lora_meta = maybe_execute_in_parallel(
                _base_w13_fn,
                _lora_w13_fn,
                lora_context.events[0],
                lora_context.events[1],
                lora_context.aux_stream,
            )
            (
                sorted_token_ids_lora,
                expert_ids_lora,
                num_tokens_post_padded_lora,
                token_lora_mapping,
            ) = lora_meta
            intermediate_cache1.add_(lora_delta_w13)
        else:
            _base_w13_fn()
            if lora_context is not None:
                (
                    sorted_token_ids_lora,
                    expert_ids_lora,
                    num_tokens_post_padded_lora,
                    token_lora_mapping,
                ) = self.apply_w13_lora(
                    lora_context,
                    y=intermediate_cache1,
                    x=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    expert_map=expert_map,
                    w1=w1,
                    w2=w2,
                    num_tokens=num_tokens,
                    top_k_num=top_k_num,
                )

        a2q_scale: torch.Tensor | None = None

        # Fuse SiLU+Mul + FP8 block quantize into a single kernel
        # when conditions permit (gated SiLU, fp8 block quant with
        # group_size=128, no LoRA requiring the BF16 intermediate).
        if (
            activation == MoEActivation.SILU
            and self.quant_config.use_fp8_w8a8
            and self.block_shape == [128, 128]
            and lora_context is None
            and not is_deep_gemm_e8m0_used()
        ):
            qintermediate_cache2, a2q_scale = ops.silu_and_mul_per_block_quant(
                intermediate_cache1.view(-1, N),
                group_size=128,
                quant_dtype=current_platform.fp8_dtype(),
            )
        else:
            self.activation(
                activation, intermediate_cache2, intermediate_cache1.view(-1, N)
            )

            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
                intermediate_cache2,
                a2_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
                quantization_emulation=self.quantization_emulation,
            )

        # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
        # unquantized intermediate_cache2 as the lora_a input.  Reuses the
        # sorted_token_ids_lora computed above. Same dual-stream pattern as
        # the w13 pair: base GEMM on default stream, LoRA delta on aux,
        # join via .add_() into intermediate_cache3.
        def _base_w2_fn():
            invoke_fused_moe_triton_kernel(
                qintermediate_cache2,
                w2,
                intermediate_cache3,
                a2q_scale,
                self.w2_scale,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                not apply_router_weight_on_input,
                1,
                config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w2_bias,
            )

        if lora_context is not None and lora_context.aux_stream is not None:
            lora_delta_w2 = torch.zeros_like(intermediate_cache3)

            def _lora_w2_fn():
                self.apply_w2_lora(
                    lora_context,
                    y=lora_delta_w2,
                    x=intermediate_cache2,
                    topk_weights=topk_weights,
                    sorted_token_ids_lora=sorted_token_ids_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                    token_lora_mapping=token_lora_mapping,
                    num_tokens=num_tokens,
                    w1=w1,
                    w2=w2,
                    top_k_num=top_k_num,
                    add_inputs=False,
                )

            assert lora_context.events is not None
            maybe_execute_in_parallel(
                _base_w2_fn,
                _lora_w2_fn,
                lora_context.events[2],
                lora_context.events[3],
                lora_context.aux_stream,
            )
            intermediate_cache3.add_(lora_delta_w2)
        else:
            _base_w2_fn()
            if lora_context is not None:
                self.apply_w2_lora(
                    lora_context,
                    y=intermediate_cache3,
                    x=intermediate_cache2,
                    topk_weights=topk_weights,
                    sorted_token_ids_lora=sorted_token_ids_lora,
                    expert_ids_lora=expert_ids_lora,
                    num_tokens_post_padded_lora=num_tokens_post_padded_lora,
                    token_lora_mapping=token_lora_mapping,
                    num_tokens=num_tokens,
                    w1=w1,
                    w2=w2,
                    top_k_num=top_k_num,
                )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        ops.moe_sum(input, output)


class TritonWNA16Experts(TritonExperts):
    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W = [
            kInt4Static,
            kInt8Static,
            kInt4Static32,
            # other group sizes?
        ]
        return weight_key in SUPPORTED_W

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUSTEP,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.GELU_TANH_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Why?
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

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
        # Check constraints.
        if self.quant_config.use_int4_w4a16:
            assert hidden_states.size(-1) // 2 == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1) // 2} == {w1.size(2)}"
            )
        else:
            assert hidden_states.size(-1) == w1.size(2), (
                f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
            )

        assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
        assert hidden_states.dim() == 2
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ]

        E, num_tokens, N, K, top_k_num = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )

        if global_num_experts == -1:
            global_num_experts = E

        config = try_get_optimal_moe_config(
            w1.size(),
            w2.size(),
            top_k_num,
            self.quant_config.config_name(hidden_states.dtype),
            num_tokens,
            block_shape=self.block_shape,
        )

        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        elif (
            hidden_states.dtype == torch.float8_e4m3fn
            or hidden_states.dtype == torch.float8_e4m3fnuz
        ):
            compute_type = tl.bfloat16
        else:
            raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

        # Note that the output tensor might be in workspace1
        intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        intermediate_cache2 = _resize_cache(
            workspace13, (num_tokens * top_k_num, activation_out_dim)
        )
        intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        invoke_fused_moe_wna16_triton_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            self.w1_scale,
            self.quant_config.w1_zp,
            None,  # topk_weights
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,  # mul_routed_weights
            top_k_num,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        self.activation(
            activation, intermediate_cache2, intermediate_cache1.view(-1, N)
        )

        a2q_scale: torch.Tensor | None = None

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            intermediate_cache2,
            a2_scale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
        )

        invoke_fused_moe_wna16_triton_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            self.w2_scale,
            self.quant_config.w2_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )

        # separate function is required for MoE + LoRA
        self.moe_sum(intermediate_cache3, output)
