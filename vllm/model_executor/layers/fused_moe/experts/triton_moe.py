# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton-based MoE expert implementations."""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation_supported,
)
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
    _weights_host_resident,
    get_moe_configs,
    invoke_fused_moe_triton_kernel,
    invoke_fused_moe_wna16_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.moe_output import route_major_moe_output
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
    FP8_DTYPE,
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kInt4Static32Asym,
    kInt4StaticAsym,
    kInt8DynamicTensorSym,
    kInt8DynamicTokenSym,
    kInt8Static,
    kInt8StaticChannelSym,
    kInt8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel


@triton.jit
def _silu_mul_quant_sorted_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    num_tokens_post_padded_ptr,
    intermediate: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    if row_start >= tl.load(num_tokens_post_padded_ptr):
        return
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * GROUP_K + tl.arange(0, GROUP_K)
    gate = tl.load(input_ptr + rows[:, None] * 2 * intermediate + cols[None, :])
    up = tl.load(
        input_ptr + rows[:, None] * 2 * intermediate + intermediate + cols[None, :]
    )
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    # match silu_and_mul: round SiLU before multiplying the other half.
    silu = tl.div_rn(gate, 1.0 + tl.extra.cuda.libdevice.exp(-gate))
    silu = silu.to(input_ptr.dtype.element_ty).to(tl.float32)
    result = (silu * up).to(input_ptr.dtype.element_ty).to(tl.float32)
    scale_raw = tl.maximum(tl.max(tl.abs(result), 1), 1e-10) * (1.0 / 448.0)
    scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))
    quantized = tl.minimum(tl.maximum(tl.div_rn(result, scale[:, None]), -448.0), 448.0)
    tl.store(
        output_ptr + rows[:, None] * intermediate + cols[None, :],
        quantized.to(tl.float8e4nv),
    )
    tl.store(scale_ptr + rows * (intermediate // GROUP_K) + tl.program_id(1), scale)


class TritonExperts(LoRAExpertsMixin, mk.FusedMoEExpertsModular):
    """Triton-based fused MoE expert implementation."""

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported or not current_platform.is_cuda_alike():
            return supported, reason

        padded_num_experts = (moe_config.num_experts + 31) // 32 * 32
        if padded_num_experts >= 1024:
            return False, (
                "kernel's moe_align_block_size requires fewer than 1024 padded experts"
            )
        return True, None

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Whether quantized MOE runs natively, or through
        # higher-precision + activation QDQ.
        self.quantization_emulation = False
        self._enable_sorted_a_tma = True
        super().__init__(moe_config, quant_config)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Defer activation quantization to apply() only when LoRA is active AND
        # tokens are dispatched across ranks (DP+EP all2all).
        return (
            self._lora_context is not None
            and self.quant_dtype is not None
            and self.moe_config.moe_parallel_config.use_all2all_kernels
        )

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
        # INT8 requires at least 7.5 (Turing) on CUDA. ROCm CDNA GPUs
        # (e.g. MI2xx/MI3xx/gfx950) provide native INT8 matrix-core support and
        # the Triton int8_w8a8 fused MoE kernel handles them.
        # XPU reaches the same Triton kernel through the same launcher, and both
        # int8 activation-quant paths work there: per-token uses the Triton
        # `per_token_quant_int8`, per-tensor resolves to the XPU branch of
        # `scaled_int8_quant`, which is plain elementwise arithmetic.
        device_supports_int8 = (
            (
                current_platform.is_cuda()
                and current_platform.has_device_capability((7, 5))
            )
            or current_platform.is_rocm()
            or current_platform.is_xpu()
        )

        supported: list[tuple[QuantKey | None, QuantKey | None]] = [(None, None)]
        if device_supports_int8:
            # Activations are consumed as float and quantized to int8
            # dynamically inside the kernel, so only dynamic-activation int8
            # schemes are supported (static-activation int8 is not).
            supported += [
                # per-channel weight + dynamic per-token activation
                (kInt8StaticChannelSym, kInt8DynamicTokenSym),
                # per-tensor weight + dynamic per-tensor activation
                (kInt8StaticTensorSym, kInt8DynamicTensorSym),
            ]
        if current_platform.supports_fp8():
            supported += [
                (kFp8Static128BlockSym, kFp8Dynamic128Sym),
                (kFp8StaticChannelSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8DynamicTokenSym),
                (kFp8StaticTensorSym, kFp8StaticTensorSym),
                (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            ]
            # Block-quantized FP8 with arbitrary block shape: the Triton
            # kernels take the block shape as a runtime argument.
            if (
                weight_key is not None
                and activation_key == kFp8Dynamic128Sym
                and weight_key.dtype == FP8_DTYPE
                and weight_key.symmetric
                and weight_key.scale.static
                and weight_key.scale.dtype == torch.float32
                and weight_key.scale.group_shape.row > 1
                and weight_key.scale.group_shape.col > 1
            ):
                return True
        return (weight_key, activation_key) in supported

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return apply_moe_activation_supported(activation)

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

    def _apply_router_weight_in_gemm2(self, apply_router_weight_on_input: bool) -> bool:
        return not apply_router_weight_on_input

    def _finalize_expert_output(
        self,
        expert_output: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        self.moe_sum(expert_output, output, topk_ids, expert_map)

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        **kwargs,
    ) -> None:
        activation_config = self.activation_config
        if (
            activation == MoEActivation.SILU
            and activation_config.clamp_limit is not None
        ):
            swiglu_limit_func(output, input, activation_config.clamp_limit)
            return

        # SWIGLUOAI_UNINTERLEAVE routes to torch.ops._C.silu_and_mul_with_clamp
        # via apply_moe_activation() and requires clamp_limit to be set.
        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            assert activation_config.clamp_limit is not None, (
                "SWIGLUOAI_UNINTERLEAVE requires gemm1_clamp_limit"
            )

        super().activation(
            activation,
            output,
            input,
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
            torch.int8,
        ]

        # We declared expects_unquantized_inputs (LoRA + DP/EP all2all), so the
        # prepare step deferred activation quantization to this kernel:
        # `hidden_states` arrives unquantized. Keep the unquantized tensor for
        # the LoRA shrink input and quantize a copy here for the base GEMM
        # (mirrors what the prepare step would have done, but after the
        # all-gather so the layout matches the gathered topk_ids / token map).
        lora_unquantized_hidden_states: torch.Tensor | None = None
        if self.expects_unquantized_inputs:
            assert a1q_scale is None
            lora_unquantized_hidden_states = hidden_states
            hidden_states, a1q_scale = moe_kernel_quantize_input(
                hidden_states,
                self.a1_scale,
                self.quant_dtype,
                self.per_act_token_quant,
                self.block_shape,
                quantization_emulation=self.quantization_emulation,
            )

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
            or hidden_states.dtype == torch.int8
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
        hooked = self._uses_post_gemm2_hook(num_tokens)
        if hooked and (apply_router_weight_on_input or self._lora_context is not None):
            raise NotImplementedError(
                "post_gemm2_hook needs unweighted GEMM2 rows; it supports neither "
                "router weights on the input nor LoRA."
            )

        # Under EP, drop the top-k slots routed to remote experts at alignment
        # time instead of launching `off_experts == -1` GEMM blocks that only
        # write zeros; the pad-aware `moe_sum` below skips the same slots.
        # Their rows in the workspace caches are then never written, so every
        # consumer of those caches must be row-local: a dynamic per-tensor
        # scale for the second GEMM's activation would take its amax over the
        # untouched rows, and LoRA aligns the full `topk_ids` itself.
        a2_scale_is_global = (
            self.quant_dtype is not None
            and a2_scale is None
            and self.block_shape is None
            and (not self.per_act_token_quant or self.quantization_emulation)
        )
        skip_invalid = (
            expert_map is not None
            and self._lora_context is None
            and not a2_scale_is_global
        )

        # Include fused shared-expert rows while preserving EP remapping.
        num_align_experts = w1.shape[0] if expert_map is None else global_num_experts
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                num_tokens,
                top_k_num,
                num_align_experts,
                expert_map,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                block_shape=self.block_shape,
                ignore_invalid_experts=skip_invalid,
                weights_host_resident=_weights_host_resident(w1, w2),
            )
        )

        from vllm.model_executor.layers.fused_moe import get_config

        down_configs = (
            get_moe_configs(
                E,
                cache2_dim,
                self.quant_config.config_name(hidden_states.dtype),
                128,
                128,
                down_moe=True,
            )
            if self.quant_config.use_fp8_w8a8
            and self.block_shape == [128, 128]
            and not get_config()
            else None
        )
        tma_down_config = (
            dict(down_configs[min(down_configs, key=lambda m: abs(m - num_tokens))])
            if down_configs
            else None
        )

        use_sorted_a = bool(
            tma_down_config
            and tma_down_config.pop("USE_TMA", False)
            and self._enable_sorted_a_tma
            and activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
            and self._lora_context is None
            and not hooked
            and not self.quantization_emulation
            and is_deep_gemm_e8m0_used()
            and a2_scale is None
            and sorted_token_ids is not None
            and expert_map is None
            and workspace2.dtype == torch.bfloat16
            and current_platform.is_cuda()
            and current_platform.is_device_capability(90)
            and config["BLOCK_SIZE_M"] % 16 == 0
            and w2.is_contiguous()
            and w2.size(1) >= 4096
            and w2.size(1) % tma_down_config["BLOCK_SIZE_N"] == 0
            and cache2_dim % 128 == 0
        )
        if use_sorted_a:
            assert sorted_token_ids is not None
            num_sorted_rows = sorted_token_ids.numel()
            # GEMM fills padding inside active blocks; quantization skips the tail.
            intermediate_cache1 = workspace2.new_empty((num_sorted_rows, 1, N))

        # LoRA w13: applied to intermediate_cache1 before activation. When
        # the LoRA layer requested a dual-stream schedule, we run base w13
        # GEMM on the default stream and the LoRA fast-path on aux_stream;
        # the LoRA writes its delta into a fresh zero buffer (add_inputs=
        # False) and we sum it into intermediate_cache1 after both finish.
        #
        # The LoRA shrink kernel needs unquantized, gathered-layout
        # activations. When activation quant was deferred to this kernel
        # (expects_unquantized_inputs), the input we quantized above is exactly
        # that, so use it directly. Otherwise fall back to the context stash
        # (e.g. weight-only quant), guarding on a row-count match so a
        # DP-gathered layout never indexes a local stash out of bounds.
        sorted_token_ids_lora = None
        expert_ids_lora = None
        num_tokens_post_padded_lora = None
        token_lora_mapping = None
        lora_context = self._lora_context
        if lora_unquantized_hidden_states is not None:
            lora_x = lora_unquantized_hidden_states
        elif (
            lora_context is not None
            and lora_context.original_hidden_states is not None
            and lora_context.original_hidden_states.shape[0] == hidden_states.shape[0]
        ):
            lora_x = lora_context.original_hidden_states
        else:
            lora_x = hidden_states

        # TODO: The fallback to self.a1_scale was added for deferred static
        # activation quantization in https://github.com/vllm-project/vllm/pull/40857.
        # Activation emulation relies solely on `a1q_scale` output of
        # `moe_kernel_quantize_input` - this should be adapted to
        # always solely rely on `a1q_scale`.
        input_scale = (
            a1q_scale
            if self.quantization_emulation
            else (a1q_scale if a1q_scale is not None else self.a1_scale)
        )

        def _base_w13_fn():
            invoke_fused_moe_triton_kernel(
                hidden_states,
                w1,
                intermediate_cache1,
                input_scale,
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
                C_is_sorted=use_sorted_a,
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
                    x=lora_x,
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
                    x=lora_x,
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
        # The fused kernel has no clamp parameter, so a configured
        # SwiGLU clamp limit falls through to the unfused path.
        if use_sorted_a:
            qintermediate_cache2 = intermediate_cache1.new_empty(
                (num_sorted_rows, cache2_dim),
                dtype=current_platform.fp8_dtype(),
            )
            a2q_scale = qintermediate_cache2.new_empty(
                (num_sorted_rows, cache2_dim // 128),
                dtype=torch.float32,
            )
            _silu_mul_quant_sorted_kernel[
                (triton.cdiv(num_sorted_rows, 16), cache2_dim // 128)
            ](
                intermediate_cache1,
                qintermediate_cache2,
                a2q_scale,
                num_tokens_post_padded,
                cache2_dim,
                BLOCK_M=16,
                GROUP_K=128,
            )
        elif (
            activation == MoEActivation.SILU
            and self.activation_config.clamp_limit is None
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
            if use_sorted_a:
                assert tma_down_config is not None
                down_config = {
                    **tma_down_config,
                    "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
                }
            else:
                down_config = config
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
                not hooked
                and self._apply_router_weight_in_gemm2(apply_router_weight_on_input),
                1,
                down_config,
                compute_type=compute_type,
                use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
                use_int8_w8a8=self.quant_config.use_int8_w8a8,
                use_int8_w8a16=self.quant_config.use_int8_w8a16,
                use_int4_w4a16=self.quant_config.use_int4_w4a16,
                per_channel_quant=self.per_act_token_quant,
                block_shape=self.block_shape,
                B_bias=self.w2_bias,
                A_is_sorted=use_sorted_a,
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

        if hooked:
            self._run_post_gemm2_hook(
                route_major_moe_output(
                    intermediate_cache3, topk_weights, topk_ids, expert_map
                ),
                output,
            )
        else:
            self._finalize_expert_output(
                intermediate_cache3, output, topk_ids, expert_map
            )

    def moe_sum(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        if expert_map is not None:
            # Skip the slots whose expert is not on this rank: the rows the
            # alignment dropped, or the zeros the `-1` blocks wrote.
            ops.moe_sum(input, output, topk_ids, expert_map)
        else:
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
            kInt4StaticAsym,
            kInt8Static,
            kInt4Static32,
            kInt4Static32Asym,
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
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
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

        # Include fused shared-expert rows while preserving EP remapping.
        num_align_experts = w1.shape[0] if expert_map is None else global_num_experts
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], num_align_experts, expert_map
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
