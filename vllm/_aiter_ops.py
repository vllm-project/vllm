# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer


def if_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if
    ROCm AITER package is supported on gfx9 archs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # checks the platform, device arch and aiter library existance.
        from importlib.util import find_spec

        from vllm.platforms.rocm import on_gfx9

        if current_platform.is_rocm() and on_gfx9() and find_spec("aiter") is not None:
            return func(*args, **kwargs)
        else:
            # Return None or do nothing if not supported
            return None

    return wrapper


def _rocm_aiter_fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation_method: int = 0,
    quant_method: int = 0,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    activation = ActivationType(activation_method)
    quant_type = QuantType(quant_method)

    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask,
        activation,
        quant_type,
        doweight_stage1,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
    )


def _rocm_aiter_fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation_method: int = 0,
    quant_method: int = 0,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _rocm_aiter_asm_moe_tkw1_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    fc1_scale: torch.Tensor | None = None,
    fc2_scale: torch.Tensor | None = None,
    fc1_smooth_scale: torch.Tensor | None = None,
    fc2_smooth_scale: torch.Tensor | None = None,
    a16: bool = False,
    per_tensor_quant_scale: torch.Tensor | None = None,
    expert_mask: torch.Tensor | None = None,
    activation_method: int = 0,
) -> torch.Tensor:
    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1

    activation = ActivationType(activation_method)

    return asm_moe_tkw1(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        fc1_scale=fc1_scale,
        fc2_scale=fc2_scale,
        fc1_smooth_scale=fc1_smooth_scale,
        fc2_smooth_scale=fc2_smooth_scale,
        a16=a16,
        per_tensor_quant_scale=per_tensor_quant_scale,
        expert_mask=expert_mask,
        activation=activation,
    )


def _rocm_aiter_asm_moe_tkw1_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    fc1_scale: torch.Tensor | None = None,
    fc2_scale: torch.Tensor | None = None,
    fc1_smooth_scale: torch.Tensor | None = None,
    fc2_smooth_scale: torch.Tensor | None = None,
    a16: bool = False,
    per_tensor_quant_scale: torch.Tensor | None = None,
    expert_mask: torch.Tensor | None = None,
    activation_method: int = 0,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _rocm_aiter_topk_softmax_impl(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> None:
    from aiter import topk_softmax

    topk_softmax(
        topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
    )


def _rocm_aiter_topk_softmax_fake(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> None:
    pass


def _rocm_aiter_biased_grouped_topk_impl(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
) -> None:
    from aiter import biased_grouped_topk

    biased_grouped_topk(
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        need_renorm,
        routed_scaling_factor,
    )


def _rocm_aiter_biased_grouped_topk_fake(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
) -> None:
    pass


def _rocm_aiter_grouped_topk_impl(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
) -> None:
    is_softmax = scoring_func == "softmax"
    from aiter import grouped_topk

    grouped_topk(
        gating_output,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        need_renorm,
        is_softmax,
        routed_scaling_factor,
    )


def _rocm_aiter_grouped_topk_fake(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
) -> None:
    pass


def _rocm_aiter_mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    kv_last_page_lens: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
) -> None:
    from aiter.mla import mla_decode_fwd

    mla_decode_fwd(
        q,
        kv_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )


def _rocm_aiter_mla_decode_fwd_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    kv_last_page_lens: torch.Tensor | None = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
) -> None:
    pass


def _rocm_aiter_gemm_w8a8_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter import gemm_a8w8_CK

    # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
    # a to be [M, K]
    # b to be [N, K]
    # CutlassScaledMMLinearKernel prepare weight `w_q` in [K, N] format
    return gemm_a8w8_CK(A, B, As, Bs, bias, output_dtype)


def _rocm_aiter_gemm_w8a8_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    m = A.shape[0]
    n = B.shape[0]
    Y = torch.empty(m, n, dtype=output_dtype, device=A.device)
    return Y


def _rocm_aiter_gemm_w8a8_blockscale_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)


def _rocm_aiter_gemm_w8a8_blockscale_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    m = A.shape[0]
    n = B.shape[0]
    Y = torch.empty(m, n, dtype=output_dtype, device=A.device)
    return Y


def _rocm_aiter_rms_norm_impl(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from aiter import rms_norm

    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)


def _rocm_aiter_rms_norm_fake(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    return torch.empty_like(x)


def _rocm_aiter_rmsnorm2d_fwd_with_add_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter import rmsnorm2d_fwd_with_add

    residual_out = torch.empty_like(residual)
    output = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(
        output,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return output, residual_out


def _rocm_aiter_rmsnorm2d_fwd_with_add_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class rocm_aiter_ops:
    _AITER_ENABLED = envs.VLLM_ROCM_USE_AITER
    _LINEAR_ENABLED = envs.VLLM_ROCM_USE_AITER_LINEAR
    _RMSNORM_ENABLED = envs.VLLM_ROCM_USE_AITER_RMSNORM
    _FMOE_ENABLED = envs.VLLM_ROCM_USE_AITER_MOE
    _MLA_ENABLED = envs.VLLM_ROCM_USE_AITER_MLA
    _PG_ATTN_ENABLED = envs.VLLM_ROCM_USE_AITER_PAGED_ATTN
    _MHA_ENABLED = envs.VLLM_ROCM_USE_AITER_MHA
    _TRITON_UNIFIED_ATTN_ENABLED = envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION
    _FP8BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP8BMM
    _FP4_GEMM_DYNAMIC_QUANT_ASM = envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
    _TRITON_ROTARY_EMBED = envs.VLLM_ROCM_USE_AITER_TRITON_ROPE
    _MOE_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS

    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        """Verifies device specs and availability of aiter main env variable."""
        return cls._AITER_ENABLED

    @classmethod
    @if_aiter_supported
    def is_linear_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._LINEAR_ENABLED

    @classmethod
    @if_aiter_supported
    def is_linear_fp8_enaled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls.is_linear_enabled() and current_platform.is_fp8_fnuz()

    @classmethod
    @if_aiter_supported
    def is_rmsnorm_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._RMSNORM_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fused_moe_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._FMOE_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fusion_moe_shared_experts_enabled(cls) -> bool:
        return cls.is_fused_moe_enabled() and cls._MOE_SHARED_EXPERTS_ENABLED

    @classmethod
    @if_aiter_supported
    def is_mla_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._MLA_ENABLED

    @classmethod
    @if_aiter_supported
    def is_mha_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._MHA_ENABLED

    @classmethod
    @if_aiter_supported
    def is_pa_attn_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._PG_ATTN_ENABLED

    @classmethod
    @if_aiter_supported
    def is_triton_unified_attn_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._TRITON_UNIFIED_ATTN_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fp8bmm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP8BMM_ENABLED

    @classmethod
    @if_aiter_supported
    def is_asm_fp4_gemm_dynamic_quant_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP4_GEMM_DYNAMIC_QUANT_ASM

    @classmethod
    @if_aiter_supported
    def is_triton_rotary_embed_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._TRITON_ROTARY_EMBED

    @staticmethod
    @if_aiter_supported
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            tags = (
                tuple()
                if is_torch_equal_or_newer("2.7.0")
                else (torch.Tag.needs_fixed_stride_order,)
            )

            # register all the custom ops here
            direct_register_custom_op(
                op_name="rocm_aiter_asm_moe_tkw1",
                op_func=_rocm_aiter_asm_moe_tkw1_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_asm_moe_tkw1_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_fused_moe",
                op_func=_rocm_aiter_fused_moe_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_fused_moe_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_topk_softmax",
                op_func=_rocm_aiter_topk_softmax_impl,
                mutates_args=["topk_weights", "topk_indices", "token_expert_indices"],
                fake_impl=_rocm_aiter_topk_softmax_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_biased_grouped_topk",
                op_func=_rocm_aiter_biased_grouped_topk_impl,
                mutates_args=["topk_weights", "topk_ids"],
                fake_impl=_rocm_aiter_biased_grouped_topk_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_grouped_topk",
                op_func=_rocm_aiter_grouped_topk_impl,
                mutates_args=["topk_weights", "topk_ids"],
                fake_impl=_rocm_aiter_grouped_topk_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_mla_decode_fwd",
                op_func=_rocm_aiter_mla_decode_fwd_impl,
                mutates_args=["o"],
                fake_impl=_rocm_aiter_mla_decode_fwd_fake,
                tags=tags,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_w8a8",
                op_func=_rocm_aiter_gemm_w8a8_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_gemm_w8a8_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_w8a8_blockscale",
                op_func=_rocm_aiter_gemm_w8a8_blockscale_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_gemm_w8a8_blockscale_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rms_norm",
                op_func=_rocm_aiter_rms_norm_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_rms_norm_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm2d_fwd_with_add",
                op_func=_rocm_aiter_rmsnorm2d_fwd_with_add_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_rmsnorm2d_fwd_with_add_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            _OPS_REGISTERED = True

    @staticmethod
    def rms_norm2d_with_add(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        variance_epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            x, residual, weight, variance_epsilon
        )

    @staticmethod
    def rms_norm(
        x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_rms_norm(x, weight, variance_epsilon)

    @staticmethod
    def gemm_w8a8(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_w8a8(A, B, As, Bs, bias, output_dtype)

    @staticmethod
    def gemm_w8a8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_w8a8_blockscale(
            A, B, As, Bs, output_dtype
        )

    @staticmethod
    def fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weight: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_mask: torch.Tensor | None = None,
        activation_method: int = 0,
        quant_method: int = 0,
        doweight_stage1: bool = False,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            expert_mask,
            activation_method,
            quant_method,
            doweight_stage1,
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
        )

    @staticmethod
    def asm_moe_tkw1(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: torch.Tensor | None = None,
        fc2_scale: torch.Tensor | None = None,
        fc1_smooth_scale: torch.Tensor | None = None,
        fc2_smooth_scale: torch.Tensor | None = None,
        a16: bool = False,
        per_tensor_quant_scale: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
        activation_method: int = 0,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_asm_moe_tkw1(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            a16,
            per_tensor_quant_scale,
            expert_mask,
            activation_method,
        )

    @staticmethod
    def topk_softmax(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> tuple[torch.Tensor, ...]:
        torch.ops.vllm.rocm_aiter_topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )
        return topk_weights, topk_indices

    @staticmethod
    def biased_grouped_topk(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        torch.ops.vllm.rocm_aiter_biased_grouped_topk(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            need_renorm,
            routed_scaling_factor,
        )

    @staticmethod
    def grouped_topk(
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
    ) -> None:
        torch.ops.vllm.rocm_aiter_grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            need_renorm,
            scoring_func,
            routed_scaling_factor,
        )

    @staticmethod
    def mla_decode_fwd(
        q: torch.Tensor,
        kv_buffer: torch.Tensor,
        o: torch.Tensor,
        sm_scale: float,
        qo_indptr: torch.Tensor,
        max_seqlen_qo: int,
        kv_indptr: torch.Tensor | None = None,
        kv_indices: torch.Tensor | None = None,
        kv_last_page_lens: torch.Tensor | None = None,
        logit_cap: float = 0.0,
    ):
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(
            q,
            kv_buffer.view(-1, 1, 1, q.shape[-1]),
            o,
            qo_indptr,
            max_seqlen_qo,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
        )

    @staticmethod
    def triton_fp4_gemm_dynamic_qaunt(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        if x_scales is None:
            x_q, x_s = dynamic_mxfp4_quant(x)
        else:
            x_q = x
            x_s = x_scales

        y = torch.empty(
            x_q.shape[0], weight.shape[0], device=x_q.device, dtype=out_dtype
        )

        gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, y)
        return y

    @staticmethod
    def triton_rotary_embed(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        head_size: int,
        rotary_dim: int,
        is_neox_style: bool,
    ):
        from aiter.ops.triton.rope import rope_cached_thd_positions_2c_fwd_inplace

        num_tokens = positions.numel()
        cos, sin = cos_sin_cache.chunk(2, dim=-1)
        query_shape = query.shape
        key_shape = key.shape
        rotate_style = 0 if is_neox_style else 1

        query = query.view(num_tokens, -1, head_size)
        key = key.view(num_tokens, -1, head_size)
        query_ = query[..., :rotary_dim]
        key_ = key[..., :rotary_dim]
        positions = positions.view(*query.shape[:1])
        rope_cached_thd_positions_2c_fwd_inplace(
            positions,
            sin,
            cos,
            query_,
            key_,
            rotate_style,
            reuse_freqs_front_part=True,
            is_nope_first=False,
        )
        query = query.view(query_shape)
        key = key.view(key_shape)

    @staticmethod
    def triton_fp8_bmm(
        X: torch.Tensor,
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
        group_size: int = 128,
        bias: torch.Tensor | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
        splitK: int | None = None,
        YQ: torch.Tensor | None = None,
        transpose_bm: bool | None = False,
        config: dict | None = None,
    ) -> torch.Tensor:
        # ruff: noqa: E501 # isort: skip
        from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
            batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as aiter_triton_fp8_bmm,
        )

        return aiter_triton_fp8_bmm(
            X,
            WQ,
            w_scale,
            group_size=group_size,
            bias=bias,
            dtype=dtype,
            splitK=splitK,
            YQ=YQ,
            transpose_bm=transpose_bm,
            config=config,
        )

    @staticmethod
    def per_1x128_fp8_quant(
        input_2d: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Only applies quantization method for fp8 data type only."""
        from aiter import QuantType, dtypes, get_hip_quant

        aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
        return aiter_per1x128_quant(input_2d.contiguous(), quant_dtype=dtypes.fp8)

    def shuffle_weight(
        self, tensor: torch.Tensor, layout: tuple[int, int] = (16, 16)
    ) -> torch.Tensor:
        from aiter.ops.shuffle import shuffle_weight

        return shuffle_weight(tensor, layout=layout)

    @staticmethod
    def shuffle_weights(
        *tensors: torch.Tensor, layout: tuple[int, int] = (16, 16)
    ) -> tuple[torch.Tensor, ...]:
        """
        Applies shuffle_weight function from AITER to each
        input tensor and returns them.

        Rearranges (shuffles) the input tensor/s
        into a specified block layout for optimized computation.

        Args:
            *tensors: Variable number of torch.Tensor objects.
            layout: A pair of integers specifying the block sizes used to divide
                the tensors during shuffling. Default is (16, 16).

        Returns:
        A Tuple of shuffled tensors.
        """
        from aiter.ops.shuffle import shuffle_weight

        return tuple(shuffle_weight(tensor, layout=layout) for tensor in tensors)


rocm_aiter_ops.register_ops_once()
