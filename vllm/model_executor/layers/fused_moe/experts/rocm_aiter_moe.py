# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum, IntEnum
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, NamedTuple

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
    kInt4Static32,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from aiter import ActivationType


class AiterQuantType(IntEnum):
    """Mirror of `aiter.QuantType`, usable without importing AITER."""

    NO = 0  # a16w16
    PER_TENSOR = 1  # w8a8 (per_Tensor)
    PER_TOKEN = 2  # w8a8/w8a4 (per_Token)
    BLOCK_1X32 = 3  # fp4x2
    BLOCK_1X128 = 4  # block quantized w8a8 (per_1x128)
    BLOCK_128x128 = 5  # block quantized w8a8 (per_128x128)


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


def aiter_moe_pads(
    hidden_pad: int,
    intermediate_pad: int,
    activation: MoEActivation,
    use_mxfp4_w4a4: bool,
    use_mxfp4_w4a16: bool,
    tp_size: int,
) -> tuple[int, int]:
    """Round the padding deltas to what AITER's MoE dispatch expects.

    SITU's A16W4 FlyDSL kernel pads per gate/up half, so its pads pass through
    unrounded. Otherwise the pads are rounded to AITER's CK/FlyDSL tile
    granularity, and at tp_size == 1 the intermediate pad is doubled for the
    fused gate/up weight. The W4A4 SiLU and W4A16 paths double it inside AITER,
    so it is passed undoubled; doubling it here as well masks live
    intermediate rows.

    Where AITER doubles it (v0.1.28.post1):
    - CK-Tile stage 1, aiter/fused_moe.py:
      `n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1)`.
    - FlyDSL stage 1 takes `inter_dim_pad` unchanged and doubles it in the
      kernel, aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:
      `inter_dim_pad_total = 2 * inter_dim_pad` is subtracted from the fused
      gate/up width. Stage 2 skips the padded K tail with the undoubled value
      (`pad_ku_skip_s2 = min(k_unroll, inter_dim_pad // K_per_ku_s2)`).
    So both stage 1 kernels expect the undoubled pad. Recheck these lines on an
    AITER bump.
    """
    if activation == MoEActivation.SITU:
        return hidden_pad, intermediate_pad
    double = (
        tp_size == 1
        and not use_mxfp4_w4a16
        and not (use_mxfp4_w4a4 and activation == MoEActivation.SILU)
    )
    return (
        hidden_pad // 128 * 128,
        intermediate_pad // 64 * 64 * (2 if double else 1),
    )


_AITER_ACTIVATION_NAMES = {
    MoEActivation.SILU: "silu",
    MoEActivation.GELU: "gelu",
    MoEActivation.SWIGLUOAI: "swiglu",
    MoEActivation.SWIGLUOAI_UNINTERLEAVE: "swiglu",
    MoEActivation.SITU: "situ",
}


def _resolve_aiter_activation(
    activation: MoEActivation,
) -> tuple["ActivationType", bool | None]:
    """Map a vLLM activation to AITER's activation_method, plus a gate/up
    interleave hint (only SWIGLUOAI_UNINTERLEAVE sets it)."""
    if activation not in _AITER_ACTIVATION_NAMES:
        raise ValueError(f"Unsupported activation: {activation}")
    activation_method = rocm_aiter_ops.get_aiter_activation_type(
        _AITER_ACTIVATION_NAMES[activation]
    )
    if activation_method is None:
        raise ValueError(f"AITER does not support activation: {activation}")
    activation_interleave = (
        False if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE else None
    )
    return activation_method, activation_interleave


def _assert_topk1_for_router_weight_on_input(topk_weights: torch.Tensor) -> None:
    assert topk_weights.dim() == 2, (
        "`topk_weights` should be in shape (num_tokens, topk)"
    )
    assert topk_weights.shape[-1] == 1, (
        "Only support topk=1 when `apply_router_weight_on_input` is True"
    )


def _write_moe_output(output: torch.Tensor, result: torch.Tensor) -> None:
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


def derive_aiter_quant_type(quant_config: FusedMoEQuantConfig) -> AiterQuantType:
    """`AiterExperts._AITER_QUANT_TYPE` for callers that bypass the subclasses."""
    if quant_config.use_mxfp4_w4a4 or quant_config.use_mxfp4_w4a16:
        return AiterQuantType.BLOCK_1X32
    if quant_config.block_shape is not None and quant_config.use_fp8_w8a8:
        return AiterQuantType.BLOCK_128x128
    if quant_config.use_fp8_w8a8 and quant_config.per_out_ch_quant:
        return AiterQuantType.PER_TOKEN
    if quant_config.use_fp8_w8a8:
        return AiterQuantType.PER_TENSOR
    return AiterQuantType.NO


def mxfp4_w4a16_extra_kwargs(
    quant_config: FusedMoEQuantConfig, moe_config: FusedMoEConfig
) -> dict:
    """`AiterMxfp4Experts._extra_fused_moe_kwargs` for callers that bypass it."""
    if not quant_config.use_mxfp4_w4a16:
        return {}

    from aiter.ops.flydsl.moe_common import GateMode

    # Opt-in DeepSeek V4.1 a4w4 weights are shuffled SEPARATED at load time.
    if moe_config.use_mxfp4_w4a4_dsv4:
        from aiter import dtypes

        return {
            "gate_mode_override": GateMode.SEPARATED.value,
            "q_dtype_a": dtypes.fp4x2,
            "bias1": quant_config.w1_bias,
            "bias2": quant_config.w2_bias,
        }
    return {
        "gate_mode_override": GateMode.INTERLEAVE.value,
        "bias1": quant_config.w1_bias,
        "bias2": quant_config.w2_bias,
    }


def _aiter_gate_mode(
    activation: MoEActivation,
    activation_interleave: bool | None,
    gate_mode_override: str | None,
) -> str:
    """The gate_mode handed to AITER's fused MoE; "" leaves AITER's default."""
    # AITER's stage1 GEMM needs gate_mode to match how weights were shuffled
    # at load time (oracle/mxfp4.py).
    from aiter.ops.flydsl.moe_common import GateMode

    if activation == MoEActivation.SITU:
        return (
            GateMode.INTERLEAVE.value
            if rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved()
            else GateMode.SEPARATED.value
        )
    if gate_mode_override is not None:
        return gate_mode_override
    if activation_interleave is not None:
        return (
            GateMode.INTERLEAVE.value
            if activation_interleave
            else GateMode.SEPARATED.value
        )
    return ""


class AiterInputFormat(Enum):
    """An activation format AITER's fused MoE consumes as it is, so an
    all-to-all dispatch can carry it and the experts skip quantizing."""

    # Also the answer when AITER's choice is unknown: it accepts BF16 everywhere.
    BF16 = "bf16"
    # FP8 with one FP32 scale per 128 elements (per_1x128).
    FP8 = "fp8"
    # FP8 with one e8m0 scale per 32 elements (per_1x32).
    MXFP8 = "mxfp8"
    # Packed FP4 (two per byte) with one e8m0 scale per 32 elements (per_1x32).
    MXFP4 = "mxfp4"


class AiterInputChoice(NamedTuple):
    format: AiterInputFormat
    reason: str
    # A configuration change could lift a BF16 fallback, so it is worth a warning.
    actionable: bool = False


@lru_cache(maxsize=1)
def _aiter_resolve_activation_dtype():
    """AITER's `resolve_activation_dtype` (ROCm/aiter#5221), where it exists."""
    try:
        from aiter.fused_moe import resolve_activation_dtype
    except ImportError:
        return None
    return resolve_activation_dtype


def aiter_moe_input_format(
    moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig | None
) -> AiterInputChoice:
    """The activation format AITER's fused MoE consumes for this layer.

    Asks AITER's `resolve_activation_dtype` with the quant type, weight dtype,
    activation and gate mode `AiterExperts` hand to `fused_moe`. A dispatch
    format is fixed before any batch exists, so the token count is not passed:
    AITER then answers None wherever its choice depends on it, and the answer
    is BF16, which every AITER path accepts.
    """
    if quant_config is None:
        return AiterInputChoice(AiterInputFormat.BF16, "unquantized MoE")
    quant_type = derive_aiter_quant_type(quant_config)
    if quant_type == AiterQuantType.BLOCK_1X32:
        weight_dtype = torch.float4_e2m1fn_x2
    elif quant_type == AiterQuantType.BLOCK_128x128:
        weight_dtype = current_platform.fp8_dtype()
    elif quant_type == AiterQuantType.NO:
        return AiterInputChoice(AiterInputFormat.BF16, "MoE takes bf16 activations")
    else:
        return AiterInputChoice(
            AiterInputFormat.BF16,
            f"no pre-quantized input format for AITER {quant_type.name} MoE",
        )

    resolve_activation_dtype = _aiter_resolve_activation_dtype()
    if resolve_activation_dtype is None:
        return AiterInputChoice(
            AiterInputFormat.BF16,
            "this AITER has no resolve_activation_dtype (ROCm/aiter#5221)",
            actionable=True,
        )

    from aiter import dtypes
    from aiter.ops.flydsl.moe_common import GateMode

    activation = moe_config.activation
    activation_method, activation_interleave = _resolve_aiter_activation(activation)
    extra = mxfp4_w4a16_extra_kwargs(quant_config, moe_config)
    gate_mode = _aiter_gate_mode(
        activation, activation_interleave, extra.get("gate_mode_override")
    )
    # rocm_aiter_ops.fused_moe omits an empty or unsupported gate_mode.
    if not gate_mode or not rocm_aiter_ops.fused_moe_supports_gate_mode():
        gate_mode = GateMode.SEPARATED.value

    q_dtype_a = extra.get("q_dtype_a")
    if q_dtype_a is None:
        q_dtype_a = resolve_activation_dtype(
            int(quant_type),
            weight_dtype,
            activation=int(activation_method),
            gate_mode=gate_mode,
        )
    if q_dtype_a is None:
        bound = (
            "GPTOSS_SWIGLU_MXFP4_BF16_BOUND"
            if activation
            in (MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE)
            and gate_mode == GateMode.SEPARATED.value
            else "AITER_BF16_FP8_MOE_BOUND"
        )
        return AiterInputChoice(
            AiterInputFormat.BF16,
            "AITER picks the MoE activation dtype by batch size here, which a "
            f"dispatch format cannot follow; set {bound}=0 to pin the quantized "
            "dtype",
            actionable=True,
        )
    if q_dtype_a == dtypes.bf16:
        return AiterInputChoice(AiterInputFormat.BF16, "AITER fused MoE consumes bf16")
    fmt = None
    if q_dtype_a in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        # The two FP8 formats differ only in the scales, which the quant type sets.
        fmt = {
            AiterQuantType.BLOCK_128x128: AiterInputFormat.FP8,
            AiterQuantType.BLOCK_1X32: AiterInputFormat.MXFP8,
        }.get(quant_type)
    elif q_dtype_a == dtypes.fp4x2 and quant_type == AiterQuantType.BLOCK_1X32:
        fmt = AiterInputFormat.MXFP4
    if fmt is None:
        return AiterInputChoice(
            AiterInputFormat.BF16,
            f"no dispatch format carries {q_dtype_a} for AITER {quant_type.name}",
        )
    return AiterInputChoice(
        fmt, f"AITER fused MoE consumes {fmt.value} ({quant_type.name})"
    )


def rocm_aiter_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_config: FusedMoEConfig,
    quant_type: AiterQuantType,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    expert_mask: torch.Tensor | None = None,
    quant_config: FusedMoEQuantConfig | None = None,
    a1q_scale: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    moe_sorting_dispatch_policy: int = 0,
    gate_mode_override: str | None = None,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    q_dtype_a: torch.dtype | None = None,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
) -> torch.Tensor:
    """ROCm AITER fused MoE expert computation."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    activation_method, activation_interleave = _resolve_aiter_activation(activation)
    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    if apply_router_weight_on_input:
        _assert_topk1_for_router_weight_on_input(topk_weights)

    # Compute padding on-the-fly for CK MXFP4 kernels
    hidden_pad = 0
    intermediate_pad = 0
    assert moe_config.hidden_dim_unpadded is not None
    assert moe_config.intermediate_size_per_partition_unpadded is not None
    hidden_size = hidden_states.shape[1]
    if hidden_states.dtype == torch.float4_e2m1fn_x2:
        # Pre-quantized MXFP4 from MoRI dispatch, packed two per byte.
        hidden_size *= 2
    hidden_pad = hidden_size - moe_config.hidden_dim_unpadded
    intermediate_pad = (
        (
            moe_config.intermediate_size_per_partition
            - moe_config.intermediate_size_per_partition_unpadded
        )
        if moe_config.intermediate_pad is None
        else moe_config.intermediate_pad
    )

    hidden_pad, intermediate_pad = aiter_moe_pads(
        hidden_pad,
        intermediate_pad,
        activation,
        quant_config.use_mxfp4_w4a4,
        quant_config.use_mxfp4_w4a16,
        moe_config.tp_size,
    )

    # AITER's bias-less W4A16 FlyDSL stage 1 leaves the padded intermediate
    # columns unwritten (ROCm/aiter#6598); the padded weights are zero, so
    # declare no pad there. SITU zeroes its own scratch on a non-zero pad.
    if (
        quant_config.use_mxfp4_w4a16
        and bias1 is None
        and activation != MoEActivation.SITU
    ):
        intermediate_pad = 0

    gate_mode = _aiter_gate_mode(activation, activation_interleave, gate_mode_override)

    return rocm_aiter_ops.fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
        quant_method=quant_type.value,
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
        bias1=bias1,
        bias2=bias2,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        swiglu_limit=(
            0.0 if moe_config.swiglu_limit is None else float(moe_config.swiglu_limit)
        ),
        beta=moe_config.activation_situ_beta,
        linear_beta=moe_config.activation_situ_linear_beta,
        shared_w1=shared_w1,
        shared_w2=shared_w2,
        shared_w1_scale=shared_w1_scale,
        shared_w2_scale=shared_w2_scale,
        shared_expert_id=shared_expert_id,
        q_dtype_a=q_dtype_a,
    )


class AiterExperts(mk.FusedMoEExpertsModular):
    consumes_expert_mask = True

    _AITER_QUANT_TYPE: AiterQuantType
    _SUPPORTS_ROUTER_WEIGHT_ON_INPUT = True

    def _extra_fused_moe_kwargs(self) -> dict:
        """Leaf-specific extra kwargs for `rocm_aiter_fused_experts`."""
        return {}

    @cached_property
    def mori_epv2_dispatch_format(self) -> AiterInputFormat:
        """What MORI EPv2 dispatches to these experts: BF16, or a format
        quantized before dispatch that AITER consumes as it is."""
        if not self.moe_config.use_mori_epv2_kernels:
            return AiterInputFormat.BF16
        from vllm.model_executor.layers.fused_moe.prepare_finalize.mori_epv2 import (
            mori_epv2_dispatch_format,
        )

        return mori_epv2_dispatch_format(self.moe_config, self.quant_config).format

    @property
    def expects_unquantized_inputs(self) -> bool:
        # When paired with MoRI, the prepare/finalize quantizes during dispatch
        # to reduce network traffic: MoRI EPv1 to FP8 or MXFP4, MoRI EPv2 to the
        # format AITER consumes, when it has one. Otherwise, AITER fused MoE
        # kernels handle input quantization internally via a single fused kernel.
        return not (
            self.moe_config.use_mori_epv1_kernels
            or self.mori_epv2_dispatch_format != AiterInputFormat.BF16
        )

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
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
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
        # MoRI dispatches MXFP4 activations packed two per byte, so K is
        # half the hidden size there.
        packed_fp4_input = (
            self.moe_config.use_mori_epv1_kernels and self.quant_config.use_mxfp4_w4a4
        ) or self.mori_epv2_dispatch_format == AiterInputFormat.MXFP4
        output = (M, K * 2 if packed_fp4_input else K)
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
        assert self._SUPPORTS_ROUTER_WEIGHT_ON_INPUT or not (
            apply_router_weight_on_input
        ), f"{type(self).__name__} does not support apply_router_weight_on_input"

        if expert_tokens_meta is not None:
            num_local_tokens = expert_tokens_meta.expert_num_tokens
        elif self.mori_epv2_dispatch_format == AiterInputFormat.FP8:
            # AITER transposes the scales of FP8 block input over a row count it
            # requires; MORI EPv2 passes none, and the experts read every row.
            num_local_tokens = hidden_states.new_full(
                (1,), hidden_states.size(0), dtype=torch.int32
            )
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
            expert_mask=expert_map,
            quant_config=self.quant_config,
            moe_config=self.moe_config,
            quant_type=self._AITER_QUANT_TYPE,
            a1q_scale=a1q_scale,
            num_local_tokens=num_local_tokens,
            output_dtype=output.dtype,
            moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
            **self._extra_fused_moe_kwargs(),
        )
        _write_moe_output(output, result)


class AiterUnquantizedExperts(AiterExperts):
    """Unquantized (bf16/fp16) weights and activations."""

    _AITER_QUANT_TYPE = AiterQuantType.NO

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [(None, None)]
        return (weight_key, activation_key) in SUPPORTED_W_A


class AiterFp8BlockExperts(AiterExperts):
    """128x128 block-quantized weight scale, 128-grouped dynamic
    activation scale."""

    _AITER_QUANT_TYPE = AiterQuantType.BLOCK_128x128
    _SUPPORTS_ROUTER_WEIGHT_ON_INPUT = False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [(kFp8Static128BlockSym, kFp8Dynamic128Sym)]
        return (weight_key, activation_key) in SUPPORTED_W_A


class AiterFp8TensorExperts(AiterExperts):
    """Per-tensor static weight scale, either per-tensor static or
    per-tensor dynamic activation scale."""

    _AITER_QUANT_TYPE = AiterQuantType.PER_TENSOR

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticTensorSym, kFp8StaticTensorSym),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class AiterFp8ChannelExperts(AiterExperts):
    """Per-output-channel static weight scale, per-token dynamic
    activation scale."""

    _AITER_QUANT_TYPE = AiterQuantType.PER_TOKEN

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [(kFp8StaticChannelSym, kFp8DynamicTokenSym)]
        return (weight_key, activation_key) in SUPPORTED_W_A

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
        if not apply_router_weight_on_input:
            return super().apply(
                output=output,
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a1q_scale=a1q_scale,
                a2_scale=a2_scale,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # tkw1 applies topk_weights to FC1's output instead of FC2's.
        assert (
            expert_tokens_meta is None or expert_tokens_meta.expert_num_tokens is None
        ), "AITER tkw1 kernel does not support `num_local_tokens`"
        # tkw1 quantizes the activation itself, so a dispatch-side `a1q_scale`
        # would be silently dropped, unlike on the generic route.
        assert a1q_scale is None, (
            "AITER tkw1 kernel does not support pre-quantized activations"
        )
        _assert_topk1_for_router_weight_on_input(topk_weights)
        activation_method, _ = _resolve_aiter_activation(activation)
        result = rocm_aiter_ops.asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            fc1_scale=self.quant_config.w1_scale,
            fc2_scale=self.quant_config.w2_scale,
            fc1_smooth_scale=None,
            fc2_smooth_scale=None,
            a16=False,
            per_tensor_quant_scale=None,
            expert_mask=expert_map,
            activation_method=activation_method,
        )
        _write_moe_output(output, result)


class AiterMxfp4Experts(AiterExperts):
    """MXFP4-quantized weights, with either unquantized (bf16/fp16) or
    MXFP4-quantized dynamic activations. gfx950-only (not gfx1250), except for
    Kimi-K3's MXFP4 experts requantized to packed int4 at load on gfx942."""

    _AITER_QUANT_TYPE = AiterQuantType.BLOCK_1X32

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        from vllm.platforms.rocm import on_gfx942, on_gfx950, on_gfx1250

        if (weight_key, activation_key) == (kInt4Static32, None):
            return on_gfx942()
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        if (weight_key, activation_key) not in SUPPORTED_W_A:
            return False
        return on_gfx950() and not on_gfx1250()

    def _extra_fused_moe_kwargs(self) -> dict:
        return mxfp4_w4a16_extra_kwargs(self.quant_config, self.moe_config)
