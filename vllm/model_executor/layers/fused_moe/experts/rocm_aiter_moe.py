# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from functools import lru_cache

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)


class QuantMethod(IntEnum):
    # This allows interfacing with AITER QuantType Enum
    # without importing the QuantType from AITER globally.

    # Note that these quantization methods are
    # supported in AITER package. However,
    # not all are used in this module.

    NO = 0  # a16w16
    PER_TENSOR = 1  # w8a8 (pre_Tensor)
    PER_TOKEN = 2  # w8a8/w8a4 (per_Token)
    BLOCK_1X32 = 3  # fp4x2
    BLOCK_1X128 = 4  # block quantized w8a8 (per_1x128)
    BLOCK_128x128 = 5  # block quantized w8a8 (per_128x128)


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


aiter_topK_meta_data: tuple[torch.Tensor, torch.Tensor] | None = None


@lru_cache(maxsize=1)
def init_aiter_topK_meta_data(
    n_routed_experts: int,
    n_shared_experts: int,
    top_k: int,
    tp_rank: int,
    tp_size: int,
    shared_experts_score: float = 1.0,
    max_num_tokens: int = 32768,
    is_EP: bool = False,
):
    global aiter_topK_meta_data
    fake_expertid = n_routed_experts + n_shared_experts

    # all layers reuse same buffer
    # This extra element when EP is enabled is used as a sentinel
    # to mask out shared expert processing for tokens not owned by
    # the current EP rank. This is necessary to avoid double-processing
    # of shared experts.
    total_topk_ids = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.int32,
        device="cuda",
    )
    ns_topk_ids, s_topk_ids = total_topk_ids.split(
        [top_k, n_shared_experts + is_EP], dim=1
    )
    shared_expert_ids = [n_routed_experts + i for i in range(n_shared_experts + is_EP)]
    if is_EP:
        s_topk_ids_list = [
            [fake_expertid] * (n_shared_experts + is_EP)
        ] * max_num_tokens
        for i in range(tp_rank, max_num_tokens, tp_size):
            s_topk_ids_list[i] = shared_expert_ids
    else:
        s_topk_ids_list = [
            list(range(n_routed_experts, fake_expertid))
        ] * max_num_tokens
    s_topk_ids[:] = torch.tensor(s_topk_ids_list, dtype=torch.int32, device="cuda")

    total_topk_weights = torch.empty(
        (max_num_tokens, top_k + n_shared_experts + is_EP),
        dtype=torch.float32,
        device="cuda",
    )
    ns_topk_weights, s_topk_weights = total_topk_weights.split(
        [top_k, n_shared_experts + is_EP], dim=1
    )
    s_topk_weights.fill_(shared_experts_score)
    assert aiter_topK_meta_data is None, "AITER topK meta data is already initialized"
    aiter_topK_meta_data = (total_topk_weights, total_topk_ids)


def inject_shared_expert_weights(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    num_fused_shared_experts: int,
    shared_expert_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge routed topk results with the shared expert buffer and inject
    dynamic per-token shared expert gate values for AITER fusion.

    For routers that already return the combined buffer (e.g. GroupedTopKRouter
    via rocm_aiter_grouped_topk), only the dynamic weight injection is needed.
    For routers that return only routed slots (e.g. FusedTopKRouter), this also
    copies the routed results into the pre-allocated combined buffer.
    """
    if num_fused_shared_experts == 0:
        return topk_weights, topk_ids

    assert aiter_topK_meta_data is not None, (
        "aiter_topK_meta_data is not initialized but "
        "num_fused_shared_experts > 0. Ensure init_aiter_topK_meta_data "
        "is called before routing."
    )

    total_topk_weights, total_topk_ids = aiter_topK_meta_data
    token = topk_weights.shape[0]

    assert total_topk_weights.shape[0] >= token, (
        f"AITER topK meta data supports {total_topk_weights.shape[0]} "
        f"tokens, but got {token} tokens."
    )

    total_topk_weights_slice = total_topk_weights[:token]
    total_topk_ids_slice = total_topk_ids[:token]

    if topk_weights.shape[1] == topk:
        total_topk_weights_slice[:, :topk] = topk_weights
        total_topk_ids_slice[:, :topk] = topk_ids
        topk_weights = total_topk_weights_slice
        topk_ids = total_topk_ids_slice

    if shared_expert_weights is not None:
        topk_weights[:, topk : topk + num_fused_shared_experts] = shared_expert_weights[
            :token
        ]

    return topk_weights, topk_ids


def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = hidden_states.shape[0]
    device = hidden_states.device
    if (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        and num_fused_shared_experts > 0
    ):
        assert aiter_topK_meta_data is not None, (
            "AITER topK meta data is not initialized. "
            "Please ensure that init_aiter_topK_meta_data "
            "is called before this function."
        )
        total_topk_weights, total_topk_ids = aiter_topK_meta_data
        assert total_topk_weights.shape[0] >= token, (
            f"AITER topK meta data support {total_topk_weights.shape[0]} "
            f"tokens which is determined by max_num_batched_tokens, "
            f"but got {token} tokens now."
        )
        total_topk_weights = total_topk_weights[:token]
        total_topk_ids = total_topk_ids[:token]
        topk_weights, _ = total_topk_weights.split(
            [topk, total_topk_weights.shape[1] - topk], dim=1
        )
        topk_ids, _ = total_topk_ids.split(
            [topk, total_topk_ids.shape[1] - topk], dim=1
        )
    else:
        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((token, topk), dtype=torch.float32, device=device)

    if e_score_correction_bias is not None:
        rocm_aiter_ops.biased_grouped_topk(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
    else:
        assert scoring_func == "softmax" or scoring_func == "sigmoid"
        rocm_aiter_ops.grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    if (
        rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        and num_fused_shared_experts > 0
    ):
        return total_topk_weights, total_topk_ids
    return topk_weights, topk_ids


def rocm_aiter_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: FusedMoEConfig,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
    a1q_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    moe_sorting_dispatch_policy: int = 0,
) -> torch.Tensor:
    """ROCm AITER fused MoE expert computation."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    if activation == MoEActivation.SILU:
        activation_method = ActivationMethod.SILU
    elif activation == MoEActivation.GELU:
        activation_method = ActivationMethod.GELU
    elif activation == MoEActivation.SWIGLUOAI:
        activation_method = rocm_aiter_ops.get_aiter_activation_type("swiglu")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    expert_mask = expert_map if expert_map is not None else None

    # w8a8 per-channel quantization
    if (
        quant_config.per_act_token_quant
        and apply_router_weight_on_input
        and quant_config.use_fp8_w8a8
    ):
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert topk_weights.dim() == 2, (
            "`topk_weights` should be in shape (num_tokens, topk)"
        )
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when `apply_router_weight_on_input` is True"
        )
        assert num_local_tokens is None, (
            "AITER tkw1 kernel does not support `num_local_tokens`"
        )

        return rocm_aiter_ops.asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale=quant_config.w1_scale,
            fc2_scale=quant_config.w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            a16=False,
            per_tensor_quant_scale=None,
            expert_mask=expert_mask,
            activation_method=activation_method,
        )

    else:
        quant_method = QuantMethod.NO.value
        # mxfp4 i.e. w4a4, w4a16 uses BLOCK_1X32
        # mxfp6 and mxfp8 are unsupported in AITER currently and use emulation instead
        if quant_config.use_mxfp4_w4a4 or quant_config.use_mxfp4_w4a16:
            quant_method = QuantMethod.BLOCK_1X32.value
        # w8a8 block-scaled
        if quant_config.block_shape is not None and quant_config.use_fp8_w8a8:
            assert not apply_router_weight_on_input, (
                "apply_router_weight_on_input is not supported for block scaled moe"
            )
            assert quant_config.w1_scale is not None
            assert quant_config.w2_scale is not None
            quant_method = QuantMethod.BLOCK_128x128.value
        elif quant_config.use_fp8_w8a8 and quant_config.per_out_ch_quant:
            quant_method = QuantMethod.PER_TOKEN.value
        elif quant_config.use_fp8_w8a8:
            # Currently only per tensor quantization method is enabled.
            quant_method = QuantMethod.PER_TENSOR.value

        if apply_router_weight_on_input:
            assert topk_weights.dim() == 2, (
                "`topk_weights` should be in shape (num_tokens, topk)"
            )
            _, topk = topk_weights.shape
            assert topk == 1, (
                "Only support topk=1 when `apply_router_weight_on_input` is True"
            )

        # Compute padding on-the-fly for CK MXFP4 kernels
        hidden_pad = 0
        intermediate_pad = 0
        assert moe_config.hidden_dim_unpadded is not None
        assert moe_config.intermediate_size_per_partition_unpadded is not None
        hidden_pad = hidden_states.shape[1] - moe_config.hidden_dim_unpadded
        intermediate_pad = (
            moe_config.intermediate_size_per_partition
            - moe_config.intermediate_size_per_partition_unpadded
        )
        # Round hidden_pad/intermediate_pad to match AITER's CK/FlyDSL MoE
        # dispatch (currently pinned to v0.1.13.post1):
        # https://github.com/ROCm/aiter/blob/v0.1.13.post1/aiter/fused_moe.py#L1073
        # https://github.com/ROCm/aiter/blob/v0.1.13.post1/aiter/fused_moe.py#L1099
        # TODO: Revisit this once we bump AITER to 0.1.15 with padding fixes
        # for CK/FlyDSL MoE GEMM e.g. https://github.com/ROCm/aiter/pull/3401
        hidden_pad = hidden_pad // 128 * 128
        intermediate_pad = (
            intermediate_pad // 64 * 64 * (2 if moe_config.tp_size == 1 else 1)
        )

        # https://github.com/ROCm/aiter/pull/3123 specialized the AITER stage1 GEMMs
        # for interleaved vs separated gate and up weights.
        # For gpt-oss i.e. use_mxfp4_w4a16=True, the weights are shuffled by
        # `rocm_aiter_ops.shuffle_weight_a16w4` in `oracle/mxfp4.py`,
        # which always sets `is_guinterleave=True`.
        # Hence, we pass in GateMode.INTERLEAVE to match the weight shuffling.
        gate_mode = ""
        if quant_config.use_mxfp4_w4a16:
            try:
                from aiter.ops.flydsl.moe_common import GateMode

                gate_mode = GateMode.INTERLEAVE.value
            except ImportError:
                pass

        return rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            quant_method=quant_method,
            activation_method=activation_method,
            w1_scale=quant_config.w1_scale,
            w2_scale=quant_config.w2_scale,
            a1_scale=quant_config.a1_scale if a1q_scale is None else a1q_scale,
            a2_scale=quant_config.a2_scale,
            doweight_stage1=apply_router_weight_on_input,
            num_local_tokens=num_local_tokens,
            output_dtype=output_dtype,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            gate_mode=gate_mode,
            bias1=quant_config.w1_bias if quant_config.use_mxfp4_w4a16 else None,
            bias2=quant_config.w2_bias if quant_config.use_mxfp4_w4a16 else None,
            moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        )


class AiterExperts(mk.FusedMoEExpertsModular):
    @property
    def expects_unquantized_inputs(self) -> bool:
        # When paired with MoRI, the prepare/finalize handles FP8
        # quantization during dispatch to reduce network traffic,
        # so we should not defer input quantization.
        # Otherwise, AITER fused MoE kernels handle input quantization
        # internally via a single fused kernel.
        return not self.moe_config.use_mori_kernels

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not is_supported and not rocm_aiter_ops.is_fused_moe_enabled():
            reason = (
                f"{reason}. AITER MoE is not enabled — "
                "set VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 "
                "to enable it"
            )
        return is_supported, reason

    @staticmethod
    def _supports_current_device() -> bool:
        return rocm_aiter_ops.is_fused_moe_enabled()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        if (weight_key, activation_key) not in SUPPORTED_W_A:
            return False
        # CK MXFP4 MoE kernels are only supported on gfx950.
        if weight_key == kMxfp4Static:
            from vllm.platforms.rocm import on_gfx950

            if not on_gfx950():
                return False
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

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
        # Workspaces are managed internally by AITER.
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
        # TODO(rob): rocm_aiter_fused_experts uses self.quant_config's
        # a_scales for static quantization. Update this to fit better
        # with the interface once all quant integrations are complete.

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        else:
            num_local_tokens = None

        result = rocm_aiter_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            quant_config=self.quant_config,
            moe_config=self.moe_config,
            a1q_scale=a1q_scale,
            num_local_tokens=num_local_tokens,
            output_dtype=output.dtype,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
        )
        # avoid redundant copy when output is a view of the result
        if (
            output.shape == result.shape
            and output.dtype == result.dtype
            and output.device == result.device
            and output.is_contiguous()
            and result.is_contiguous()
            and output._base is None
        ):
            output.set_(result)
        else:
            output.copy_(result)


class AiterBatchedExpertsFp8(mk.FusedMoEExpertsModular):
    """Adapt AITER's Standard-layout FP8 fused MoE kernel to the
    ``BatchedExperts`` activation format used by multi-node DP/EP deployments
    (e.g. ``--all2all-backend deepep_low_latency``).

    Strategy (reshape wrapper, no new GPU kernel):

    1. Receive ``hidden_states`` of shape ``(E_local, M_e, K)`` already
       sorted/batched by expert by the prepare step.
    2. Flatten to ``(E_local * M_e, K)`` and construct a synthetic ``topk_ids``
       so that tokens ``[i*M_e, (i+1)*M_e)`` map to local expert ``i``.
    3. Leave padding slots beyond each ``expert_num_tokens[i]`` in the
       flattened stream. The Standard AITER kernel computes those rows; the
       BatchedExperts finalizer only combines the valid prefix for each expert.
    4. Delegate to the existing ``rocm_aiter_fused_experts`` (Standard layout)
       which already supports FP8 W8A8 (per-tensor, per-token, block 128x128).
    5. The output buffer is a contiguous ``(E_local, M_e, N)`` tensor passed
       by the runtime; we obtain a 2-D view ``(E_local * M_e, N)`` and write
       through it, so no separate reshape-back step is needed.

    Notes / caveats (verified against ``NaiveBatchedExperts`` /
    ``CutlassBatchedExpertsFp8`` in this codebase):

    - In the BatchedExperts contract ``topk_weights`` / ``topk_ids`` passed to
      ``apply()`` are the *original* router outputs (shape ``(M_router, K)``);
      they are **not** used for expert routing inside ``apply()`` because the
      prepare step has already sorted and batched the tokens. The surrounding
      BatchedExperts prepare/finalize path owns router-weight application: when
      ``apply_router_weight_on_input`` is true, ``hidden_states`` are already
      weighted before this wrapper; otherwise finalize applies the original
      weights during reduction. Therefore we construct synthetic
      per-flattened-token weights of 1.0 here and keep the inner AITER
      router-weight flag disabled.

    - ``expert_map`` is unused because BatchedExperts prepare/finalize has
      already resolved the global-to-local expert mapping. The inner AITER
      call receives synthetic local expert ids 0..E_local-1 and
      ``expert_map=None`` so those ids address the local weight slabs directly.

    - Performance: extra ``arange.repeat_interleave`` + ``reshape`` per layer,
      no extra device-side data movement (flatten is a view). A native AITER
      batched kernel is a follow-up non-goal. This wrapper unblocks multi-node
      DP/EP FP8 deployments on AMD MI300X (gfx942) which would otherwise fail
      the FP8 oracle's activation-format check.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int,
        num_dispatchers: int,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )
        # Build an inner Standard-layout AITER experts instance to delegate to.
        # We intentionally pass *no* max_num_tokens / num_dispatchers to the
        # inner (Standard) experts, since AiterExperts (Standard) asserts
        # those must be None — see FusedMoEExperts.__init__ in modular_kernel.py.
        self._inner = AiterExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The BatchedExperts prepare/finalize implementations that produce
        # this activation format (DeepEP low-latency, NIXL) explicitly do
        # *not* support ``defer_input_quant=True`` (see
        # ``DeepEPLLPrepareAndFinalize.prepare_async`` — it raises if
        # defer_input_quant is requested). So we must accept already-quantized
        # inputs from the prepare step, regardless of what the inner
        # Standard-layout AiterExperts would normally request.
        return False

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    @staticmethod
    def is_supported_config(
        cls, moe_config, weight_key, activation_key, activation_format
    ):
        is_supported, reason = super().is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not is_supported and not rocm_aiter_ops.is_fused_moe_enabled():
            reason = (
                f"{reason}. AITER MoE is not enabled — "
                "set VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_MOE=1 "
                "to enable it"
            )
        return is_supported, reason

    @staticmethod
    def _supports_current_device() -> bool:
        return rocm_aiter_ops.is_fused_moe_enabled()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # Only FP8 W8A8 schemes are unblocked by this wrapper today. MXFP4 is
        # excluded since the BatchedExperts producers used in DP/EP MoE
        # (DeepEP LL, NIXL EP) currently only dispatch FP8 / bf16, not
        # MXFP4-packed scales.
        SUPPORTED_W_A: list[tuple[QuantKey | None, QuantKey | None]] = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return AiterExperts._supports_activation(activation)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # BatchedExperts is itself the format used by EP via DeepEP-LL / NIXL,
        # so allow it for those parallel configs. We mirror the flashinfer
        # exclusion that AiterExperts also applies, since the inner kernel
        # would not be able to handle those.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # Let the BatchedExperts prepare/finalize (e.g. DeepEP-LL combine)
        # handle topk weight application + reduction. This matches the
        # behavior of every other BatchedExperts implementation in tree
        # (BatchedTritonExperts, CutlassBatchedExpertsFp8, NaiveBatchedExperts).
        return TopKWeightAndReduceDelegate()

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
        # AITER manages its own internal workspaces (kernel-side), so the
        # only buffer we need the modular kernel runtime to allocate for us
        # is the fused-output tensor itself, which for BatchedExperts must
        # be shaped (E_local, M_e_total, K) — see e.g.
        # NaiveBatchedExperts.workspace_shapes.
        assert self.num_dispatchers is not None
        assert self.max_num_tokens is not None
        num_dp = self.num_dispatchers
        workspace1 = (0,)
        workspace2 = (0,)
        # M here is max_num_tokens (per dispatcher); multiply by num_dp to
        # match the convention used by the BatchedExperts producers.
        output = (local_num_experts, self.max_num_tokens * num_dp, K)
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
        # Shape contract: see class docstring.
        assert hidden_states.dim() == 3, (
            f"AiterBatchedExpertsFp8 expects 3-D batched hidden_states "
            f"(E_local, M_e, K), got {tuple(hidden_states.shape)}"
        )
        assert output.dim() == 3, (
            f"AiterBatchedExpertsFp8 expects 3-D batched output "
            f"(E_local, M_e, N), got {tuple(output.shape)}"
        )
        assert expert_tokens_meta is not None, (
            "AiterBatchedExpertsFp8 requires expert_tokens_meta from the "
            "BatchedExperts prepare step (e.g. DeepEPLL dispatch)"
        )
        # The outer BatchedExperts prepare/finalize contract owns router
        # weighting. If apply_router_weight_on_input is true, hidden_states are
        # already weighted before this wrapper; otherwise finalize applies the
        # real topk_weights. The inner Standard AITER call only sees synthetic
        # all-ones weights, so keep its flag path disabled below.
        E_local, M_e, K = hidden_states.shape
        assert w1.size(0) == E_local, (
            f"w1 expert dim {w1.size(0)} != hidden_states E_local {E_local}"
        )

        device = hidden_states.device

        # 1) Flatten activations to (E_local * M_e, K). View-only.
        flat_in = hidden_states.reshape(E_local * M_e, K)
        flat_out = output.reshape(E_local * M_e, output.size(-1))

        # 2) Build synthetic per-flattened-token routing.
        # Tokens [i*M_e, (i+1)*M_e) → local expert i.
        synth_ids = (
            torch.arange(E_local, device=device, dtype=torch.int32)
            .repeat_interleave(M_e)
            .unsqueeze(-1)
        )

        # Synthetic weights: all 1.0 — the real router weights get applied
        # later in finalize (TopKWeightAndReduceDelegate). topk_weights.dtype
        # is float32 in AITER (see rocm_aiter_fused_experts which casts to
        # float32 anyway), so use that.
        synth_weights = torch.ones(
            (E_local * M_e, 1), device=device, dtype=torch.float32
        )

        # 3) Padding slots remain in the flattened stream. The Standard AITER
        # kernel computes every row, including rows beyond
        # expert_num_tokens[i]. BatchedExperts finalizers combine only the
        # valid prefix for each expert, so those padded outputs are ignored.

        # 4) Handle quantization-scale layouts. The BatchedExperts producers
        # ship per-block scales shaped (E_local, M_e, K/block_k) or per-tensor
        # (1,) or per-token (E_local, M_e, 1). The inner AITER kernel expects
        # 2-D scales matching the *flattened* activation layout
        # (E_local * M_e, K/block_k) or (E_local * M_e, 1).
        flat_a1q_scale: torch.Tensor | None = None
        if a1q_scale is not None:
            if a1q_scale.dim() == 3:
                flat_a1q_scale = a1q_scale.reshape(
                    a1q_scale.size(0) * a1q_scale.size(1), a1q_scale.size(2)
                )
            else:
                # Scalar / per-tensor: leave untouched, AITER handles it.
                flat_a1q_scale = a1q_scale

        # 5) Delegate to the Standard-layout AITER kernel. The inner apply()
        # writes into `flat_out`, which is a view of `output`, so no copy
        # back is needed.
        self._inner.apply(
            output=flat_out,
            hidden_states=flat_in,
            w1=w1,
            w2=w2,
            topk_weights=synth_weights,
            topk_ids=synth_ids,
            activation=activation,
            # global_num_experts == E_local in batched layout: synth_ids
            # already address local slabs directly.
            global_num_experts=E_local,
            # No expert_map: BatchedExperts is already-local; see class docstring.
            expert_map=None,
            a1q_scale=flat_a1q_scale,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            # expert_tokens_meta is dropped intentionally: the inner kernel
            # uses it only to set `num_local_tokens`, which it then forwards
            # as a *flat* (E_local,) tensor counting tokens per expert in the
            # flat layout. Since our flat layout is contiguous per expert
            # with stride M_e, the original expert_num_tokens vector still
            # has the right semantics — but the inner AiterExperts.apply()
            # passes it as `num_local_tokens` to AITER's fused_moe op, which
            # is a Standard-layout concept. We pass it through to preserve
            # any kernel-side opportunistic skipping AITER may do.
            expert_tokens_meta=expert_tokens_meta,
            # Router weights are handled by the outer BatchedExperts
            # prepare/finalize path. Passing True here would send synthetic
            # all-ones weights through AITER's unsupported top-1 flag path.
            apply_router_weight_on_input=False,
        )
