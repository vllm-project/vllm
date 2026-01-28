# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable

import torch
from torch._ops import OpOverload

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_aiter_sparse_attn_indexer,
    rocm_aiter_sparse_attn_indexer_fake,
)

_FP8_DTYPE = current_platform.fp8_dtype()


def is_aiter_found() -> bool:
    from importlib.util import find_spec

    return find_spec("aiter") is not None


# `find_spec` is not torch.compile compatible.
# In cases where aiter availability might have
# been checked in forward passes that are torch compiled.
# we keep this global outside to not cause torch compile breaks.
IS_AITER_FOUND = is_aiter_found()


def is_aiter_found_and_supported() -> bool:
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_gfx9

        return on_gfx9()
    return False


def if_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if
    ROCm AITER package is supported on gfx9 archs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # checks the platform, device arch and aiter library existence.

        if is_aiter_found_and_supported():
            return func(*args, **kwargs)

        return None

    return wrapper


# Can't use dtypes.fp8 directly inside an op
# because it returns wrong result on gfx942.
# This is a workaround to get the correct FP8 dtype.
# This might because that the get_gfx() is wrapped as a custom op.
if is_aiter_found_and_supported():
    from aiter import dtypes

    AITER_FP8_DTYPE = dtypes.fp8


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
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
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
        num_local_tokens=num_local_tokens,
        dtype=output_dtype,
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
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if output_dtype is not None:
        return torch.empty_like(hidden_states, dtype=output_dtype)
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


def _rocm_aiter_topk_sigmoid_impl(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
) -> None:
    from aiter import topk_sigmoid

    topk_sigmoid(topk_weights, topk_indices, gating_output)


def _rocm_aiter_topk_sigmoid_fake(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
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


# Cache whether aiter supports FP8 MLA parameters
_AITER_MLA_SUPPORTS_FP8: bool | None = None


def _check_aiter_mla_fp8_support() -> bool:
    """Check if aiter.mla.mla_decode_fwd supports q_scale and kv_scale parameters."""
    global _AITER_MLA_SUPPORTS_FP8
    if _AITER_MLA_SUPPORTS_FP8 is None:
        try:
            import inspect

            from aiter.mla import mla_decode_fwd

            sig = inspect.signature(mla_decode_fwd)
            _AITER_MLA_SUPPORTS_FP8 = (
                "q_scale" in sig.parameters and "kv_scale" in sig.parameters
            )
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            ValueError,
            TypeError,
        ):
            # ImportError/ModuleNotFoundError: aiter.mla module not available
            # AttributeError: mla_decode_fwd doesn't exist
            # ValueError: mla_decode_fwd has no signature (e.g., built-in)
            # TypeError: mla_decode_fwd is not a callable
            _AITER_MLA_SUPPORTS_FP8 = False
    return _AITER_MLA_SUPPORTS_FP8


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
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> None:
    from aiter.mla import mla_decode_fwd

    kwargs = {
        "sm_scale": sm_scale,
        "logit_cap": logit_cap,
    }

    # Only pass q_scale and kv_scale if the aiter library supports them
    if _check_aiter_mla_fp8_support():
        kwargs["q_scale"] = q_scale
        kwargs["kv_scale"] = kv_scale

    mla_decode_fwd(
        q,
        kv_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        **kwargs,
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
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
) -> None:
    pass


def _rocm_aiter_gemm_a8w8_impl(
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
    # CutlassInt8ScaledMMLinearKernel prepare weight `w_q` in [K, N] format
    return gemm_a8w8_CK(A, B, As, Bs, bias, output_dtype)


def _rocm_aiter_gemm_a8w8_fake(
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


def _rocm_aiter_triton_gemm_a8w8_blockscale_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)


def _rocm_aiter_triton_gemm_a8w8_blockscale_fake(
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


def _rocm_aiter_gemm_a8w8_blockscale_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)


def _rocm_aiter_gemm_a8w8_blockscale_fake(
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
    out = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(
        out,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return out, residual_out


def _rocm_aiter_rmsnorm2d_fwd_with_add_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_out = torch.empty_like(residual)
    out = torch.empty_like(x)
    return out, residual_out


def _rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    assert quant_dtype in [torch.int8, _FP8_DTYPE]

    y_scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    residual_out = torch.empty_like(x)

    rocm_aiter.rmsnorm2d_fwd_with_add_dynamicquant(
        out,
        x,
        residual,
        residual_out,
        y_scale,
        weight,
        epsilon,
        use_model_sensitive_rmsnorm=0,
    )

    return out, residual_out, y_scale


def _rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    residual_out = torch.empty_like(x)

    return out, residual_out, y_scale


def _rocm_aiter_rmsnorm_fused_dynamic_quant_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    assert quant_dtype in [torch.int8, _FP8_DTYPE]

    y_scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)

    rocm_aiter.rmsnorm2d_fwd_with_dynamicquant(
        out, x, y_scale, weight, epsilon, use_model_sensitive_rmsnorm=0
    )

    return out, y_scale


def _rocm_aiter_rmsnorm_fused_dynamic_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)

    return out, y_scale


def _rocm_aiter_per_tensor_quant_impl(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import per_tensor_quant_hip

    return per_tensor_quant_hip(x, scale, quant_dtype)


def _rocm_aiter_per_tensor_quant_fake(
    x: torch.Tensor,
    quant_dtype: torch.dtype,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=quant_dtype), torch.empty(
        1, dtype=torch.float32, device=x.device
    )


def _rocm_aiter_per_token_quant_impl(
    x: torch.Tensor, quant_dtype: torch.dtype, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import dynamic_per_token_scaled_quant

    assert quant_dtype in [torch.int8, _FP8_DTYPE]

    out_shape = x.shape
    out = torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device)
    if scale is None:
        scale = torch.empty((*out_shape[:-1], 1), dtype=torch.float32, device=x.device)
    dynamic_per_token_scaled_quant(
        out,
        x,
        scale,
        scale_ub=None,
        shuffle_scale=False,
        num_rows=None,
        num_rows_factor=1,
    )
    return out, scale


def _rocm_aiter_per_token_quant_fake(
    x: torch.Tensor, quant_dtype: torch.dtype, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    out_shape = x.shape
    return (
        torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device),
        torch.empty((*out_shape[:-1], 1), dtype=torch.float32, device=x.device),
    )


def _rocm_aiter_rmsnorm_with_add_fp8_group_quant_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    (x_quant, x_quant_scales), _, _, res = fused_rms_fp8_group_quant(
        x,
        weight,
        variance_epsilon,
        None,
        None,
        None,
        group_size=group_size,
        dtype_quant=AITER_FP8_DTYPE,
        res1=residual,
    )
    return (
        x_quant,
        res,
        x_quant_scales,
    )


def _rocm_aiter_rmsnorm_with_add_fp8_group_quant_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    scale_shape = (M, (N + group_size - 1) // group_size)
    return (
        torch.empty_like(x, dtype=AITER_FP8_DTYPE, device=x.device),
        torch.empty_like(residual, device=residual.device),
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
    )


def _rocm_aiter_rmsnorm_fp8_group_quant_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    (x_quant, x_quant_scales), _, _, res = fused_rms_fp8_group_quant(
        x,
        weight,
        variance_epsilon,
        None,
        None,
        None,
        group_size=group_size,
        dtype_quant=AITER_FP8_DTYPE,
        res1=None,
    )
    return (x_quant, x_quant_scales)


def _rocm_aiter_rmsnorm_fp8_group_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    scale_shape = (M, (N + group_size - 1) // group_size)
    return (
        torch.empty_like(x, dtype=AITER_FP8_DTYPE, device=x.device),
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
    )


def _rocm_aiter_group_fp8_quant_impl(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0, "Input shape must be divisible by group size"
    from aiter import QuantType, get_hip_quant

    aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
    return aiter_per1x128_quant(x.contiguous(), quant_dtype=AITER_FP8_DTYPE)


def _rocm_aiter_group_fp8_quant_fake(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=AITER_FP8_DTYPE, device=x.device)
    out_bs = torch.empty(
        (
            M,
            (N + group_size - 1) // group_size,
        ),
        dtype=torch.float32,
        device=x.device,
    )
    return x_fp8, out_bs


def _rocm_aiter_act_mul_and_fp8_group_quant_impl(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.activation import act_mul_and_fp8_group_quant

    return act_mul_and_fp8_group_quant(
        x,
        activation="silu",
        group_size=group_size,
        dtype_quant=AITER_FP8_DTYPE,
    )


def _rocm_aiter_act_mul_and_fp8_group_quant_fake(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    assert N % 2 == 0
    N_half = N // 2
    x_fp8 = torch.empty((M, N_half), dtype=AITER_FP8_DTYPE, device=x.device)
    out_bs = torch.empty(
        (
            M,
            (N_half + group_size - 1) // group_size,
        ),
        dtype=torch.float32,
        device=x.device,
    )
    return x_fp8, out_bs


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class rocm_aiter_ops:
    """ROCm AITER operations wrapper for AMD GPU acceleration in vLLM.

    This class centralizes the import and registration of AITER ops,
    and provides a unified interface for checking if AITER is enabled.
    Operations are only available on supported gfx9
    architectures when aiter is installed.

    The class uses environment variables to control which features are enabled,
    allowing fine-grained control over which AITER optimizations are used.

    Environment Variables:
        VLLM_ROCM_USE_AITER: Main toggle for all AITER operations.
        VLLM_ROCM_USE_AITER_LINEAR: Controls GEMM and quantization ops.
        VLLM_ROCM_USE_AITER_RMSNORM: Controls RMSNorm operations.
        VLLM_ROCM_USE_AITER_MOE: Controls MoE (Mixture of Experts) ops.
        VLLM_ROCM_USE_AITER_MLA: Controls MLA (Multi-head Latent Attention) ops.
        VLLM_ROCM_USE_AITER_MHA: Controls MHA ops including flash_attn_varlen.
        VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION: Controls Triton unified attention.
        VLLM_ROCM_USE_AITER_FP8BMM: Controls FP8 batched matrix multiply.
        VLLM_ROCM_USE_AITER_FP4_ASM_GEMM: Controls FP4 assembly GEMM.
        VLLM_ROCM_USE_AITER_TRITON_ROPE: Controls Triton rotary embeddings.
        VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS: Controls shared expert fusion.
        VLLM_ROCM_USE_AITER_TRITON_GEMM: Controls Triton unquantized GEMM.

    Note:
        The environment variables are assigned when the module is imported,
        so you can't change the environment variables after the module is imported.
        This is done out of performance consideration. Accessing environment variables
        is expensive as described in issue https://github.com/vllm-project/vllm/issues/17067
        so we don't want to do it repeatedly, especially in the hot path (the forward pass).
        You can call the refresh_env_variables() function to reload the env variables
        after monkey patching the env variables in the unit test.

    Check Functions:
        All check functions (is_*_enabled) are decorated with @if_aiter_supported,
        which verifies: (1) platform is ROCm, (2) device arch is gfx9, and
        (3) aiter library is installed. The check function then also verifies
        the corresponding environment variable is enabled.
        i.e.                                             ___
        is_enabled() == current_platform.is_rocm() and      |     checked by
                        current_platform.is_on_gfx9() and   | @if_aiter_supported
                        IS_AITER_FOUND and   _______________|
                        cls._AITER_ENABLED   -----> Check by the logic in `is_enabled()`

    Example:
        from vllm._aiter_ops import rocm_aiter_ops

        # Check if aiter is enabled before using operations
        if rocm_aiter_ops.is_enabled():
            result = rocm_aiter_ops.rms_norm(x, weight, epsilon)

    Operations:
        - RMS normalization: rms_norm, rms_norm2d_with_add
        - GEMM operations: gemm_a8w8, gemm_a8w8_blockscale
        - Fused MoE: fused_moe, asm_moe_tkw1
        - Routing: topk_softmax, biased_grouped_topk, grouped_topk
        - MLA decode: mla_decode_fwd
        - Quantization: per_tensor_quant, per_token_quant, group_fp8_quant
        - Triton ops: triton_rotary_embed, triton_fp8_bmm, triton_gemm_a8w8_blockscale
    """

    # Check if the env variable is set
    _AITER_ENABLED = envs.VLLM_ROCM_USE_AITER
    _LINEAR_ENABLED = envs.VLLM_ROCM_USE_AITER_LINEAR
    _RMSNORM_ENABLED = envs.VLLM_ROCM_USE_AITER_RMSNORM
    _FMOE_ENABLED = envs.VLLM_ROCM_USE_AITER_MOE
    _MLA_ENABLED = envs.VLLM_ROCM_USE_AITER_MLA
    _MHA_ENABLED = envs.VLLM_ROCM_USE_AITER_MHA
    _SHUFFLE_KV_CACHE_ENABLED = envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT
    _TRITON_UNIFIED_ATTN_ENABLED = envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION
    # TODO: Consolidate under _LINEAR_ENABLED
    _FP8BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP8BMM
    _FP4BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP4BMM
    # TODO: Consolidate under _LINEAR_ENABLED
    _FP4_GEMM_DYNAMIC_QUANT_ASM = envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
    # TODO: Consolidate under VLLM_ROCM_USE_AITER_ROPE
    _TRITON_ROTARY_EMBED = envs.VLLM_ROCM_USE_AITER_TRITON_ROPE
    _MOE_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
    # TODO: Consolidate under _LINEAR_ENABLED
    _TRITON_UNQUANT_GEMM = envs.VLLM_ROCM_USE_AITER_TRITON_GEMM

    @classmethod
    def refresh_env_variables(cls):
        """
        Since the environment variables are assigned when the module is imported,
        This is a helper function to reload all the env variables from
        the environment variables.
        for example, after monkey patching the env variables in the unit test,
        you can call this function to reload the env variables.
        """
        cls._AITER_ENABLED = envs.VLLM_ROCM_USE_AITER
        cls._LINEAR_ENABLED = envs.VLLM_ROCM_USE_AITER_LINEAR
        cls._RMSNORM_ENABLED = envs.VLLM_ROCM_USE_AITER_RMSNORM
        cls._FMOE_ENABLED = envs.VLLM_ROCM_USE_AITER_MOE
        cls._MLA_ENABLED = envs.VLLM_ROCM_USE_AITER_MLA
        cls._MHA_ENABLED = envs.VLLM_ROCM_USE_AITER_MHA
        cls._SHUFFLE_KV_CACHE_ENABLED = envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT
        cls._TRITON_UNIFIED_ATTN_ENABLED = envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION
        cls._FP8BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP8BMM
        cls._FP4BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP4BMM
        cls._FP4_GEMM_DYNAMIC_QUANT_ASM = envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
        cls._TRITON_ROTARY_EMBED = envs.VLLM_ROCM_USE_AITER_TRITON_ROPE
        cls._MOE_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        cls._TRITON_UNQUANT_GEMM = envs.VLLM_ROCM_USE_AITER_TRITON_GEMM

    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        return cls._AITER_ENABLED

    @classmethod
    @if_aiter_supported
    def is_linear_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._LINEAR_ENABLED

    @classmethod
    @if_aiter_supported
    def is_linear_fp8_enabled(cls) -> bool:
        return cls.is_linear_enabled()

    @classmethod
    @if_aiter_supported
    def is_rmsnorm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._RMSNORM_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fused_moe_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FMOE_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fusion_moe_shared_experts_enabled(cls) -> bool:
        return cls.is_fused_moe_enabled() and cls._MOE_SHARED_EXPERTS_ENABLED

    @classmethod
    @if_aiter_supported
    def is_mla_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._MLA_ENABLED

    @classmethod
    @if_aiter_supported
    def is_mha_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._MHA_ENABLED

    @classmethod
    @if_aiter_supported
    def is_shuffle_kv_cache_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._SHUFFLE_KV_CACHE_ENABLED

    @classmethod
    @if_aiter_supported
    def is_triton_unified_attn_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._TRITON_UNIFIED_ATTN_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fp8bmm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP8BMM_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fp4bmm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP4BMM_ENABLED

    @classmethod
    @if_aiter_supported
    def is_asm_fp4_gemm_dynamic_quant_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP4_GEMM_DYNAMIC_QUANT_ASM

    @classmethod
    @if_aiter_supported
    def is_triton_rotary_embed_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._TRITON_ROTARY_EMBED

    @classmethod
    @if_aiter_supported
    def is_triton_gemm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._TRITON_UNQUANT_GEMM

    @staticmethod
    @if_aiter_supported
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
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
                op_name="rocm_aiter_topk_sigmoid",
                op_func=_rocm_aiter_topk_sigmoid_impl,
                mutates_args=["topk_weights", "topk_indices"],
                fake_impl=_rocm_aiter_topk_sigmoid_fake,
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
            )

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_a8w8",
                op_func=_rocm_aiter_gemm_a8w8_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_gemm_a8w8_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_triton_gemm_a8w8_blockscale",
                op_func=_rocm_aiter_triton_gemm_a8w8_blockscale_impl,
                fake_impl=_rocm_aiter_triton_gemm_a8w8_blockscale_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_a8w8_blockscale",
                op_func=_rocm_aiter_gemm_a8w8_blockscale_impl,
                fake_impl=_rocm_aiter_gemm_a8w8_blockscale_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rms_norm",
                op_func=_rocm_aiter_rms_norm_impl,
                fake_impl=_rocm_aiter_rms_norm_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm2d_fwd_with_add",
                op_func=_rocm_aiter_rmsnorm2d_fwd_with_add_impl,
                fake_impl=_rocm_aiter_rmsnorm2d_fwd_with_add_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm_fused_dynamic_quant",
                op_func=_rocm_aiter_rmsnorm_fused_dynamic_quant_impl,
                fake_impl=_rocm_aiter_rmsnorm_fused_dynamic_quant_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm_fused_add_dynamic_quant",
                op_func=_rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl,
                fake_impl=_rocm_aiter_rmsnorm_fused_add_dynamic_quant_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm_fp8_group_quant",
                op_func=_rocm_aiter_rmsnorm_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_rmsnorm_fp8_group_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm_with_add_fp8_group_quant",
                op_func=_rocm_aiter_rmsnorm_with_add_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_rmsnorm_with_add_fp8_group_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_act_mul_and_fp8_group_quant",
                op_func=_rocm_aiter_act_mul_and_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_act_mul_and_fp8_group_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_group_fp8_quant",
                op_func=_rocm_aiter_group_fp8_quant_impl,
                fake_impl=_rocm_aiter_group_fp8_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_per_tensor_quant",
                op_func=_rocm_aiter_per_tensor_quant_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_per_tensor_quant_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_per_token_quant",
                op_func=_rocm_aiter_per_token_quant_impl,
                fake_impl=_rocm_aiter_per_token_quant_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_sparse_attn_indexer",
                op_func=rocm_aiter_sparse_attn_indexer,
                mutates_args=["topk_indices_buffer"],
                fake_impl=rocm_aiter_sparse_attn_indexer_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            _OPS_REGISTERED = True

    @staticmethod
    def get_rmsnorm_fused_add_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add.default

    @staticmethod
    def get_rmsnorm_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_rms_norm.default

    @staticmethod
    def get_rmsnorm_fused_add_dynamic_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_rmsnorm_fused_add_dynamic_quant.default

    @staticmethod
    def get_rmsnorm_fused_dynamic_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_rmsnorm_fused_dynamic_quant.default

    @staticmethod
    def get_rmsnorm_group_fused_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_rmsnorm_fp8_group_quant.default

    @staticmethod
    def get_rmsnorm_group_add_fused_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_rmsnorm_with_add_fp8_group_quant.default

    @staticmethod
    def get_per_token_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_per_token_quant.default

    @staticmethod
    def get_group_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_group_fp8_quant.default

    @staticmethod
    def get_act_mul_fused_fp8_group_quant_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant.default

    @staticmethod
    def rms_norm(
        x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_rms_norm(x, weight, variance_epsilon)

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
    def gemm_a8w8(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_a8w8(A, B, As, Bs, bias, output_dtype)

    @staticmethod
    def triton_gemm_a8w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_triton_gemm_a8w8_blockscale(
            A, B, As, Bs, output_dtype
        )

    @staticmethod
    def gemm_a8w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
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
        num_local_tokens: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
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
            num_local_tokens,
            output_dtype,
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
    def topk_sigmoid(
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool,
    ) -> tuple[torch.Tensor, ...]:
        torch.ops.vllm.rocm_aiter_topk_sigmoid(
            topk_weights, topk_indices, gating_output
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
        q_scale: torch.Tensor | None = None,
        kv_scale: torch.Tensor | None = None,
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
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

    @staticmethod
    def per_tensor_quant(
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.rocm_aiter_per_tensor_quant(x, quant_dtype, scale)

    @staticmethod
    def per_token_quant(
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.rocm_aiter_per_token_quant(x, quant_dtype, scale)

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
            query_,
            key_,
            cos,
            sin,
            positions,
            rotate_style,
            reuse_freqs_front_part=True,
            nope_first=False,
        )
        query = query.view(query_shape)
        key = key.view(key_shape)

    @staticmethod
    def batched_gemm_a16wfp4(
        X: torch.Tensor,
        W: torch.Tensor,
        w_scale: torch.Tensor,
        Y: torch.Tensor,
        transpose_bm: bool | None = False,
        prequant: bool | None = False,
        y_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ruff: noqa: E501 # isort: skip
        from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

        return batched_gemm_a16wfp4(
            X,
            W,
            w_scale,
            y=Y,
            transpose_bm=transpose_bm,
            prequant=prequant,
            y_scale=y_scale,
        )

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
    def group_fp8_quant(
        input_2d: torch.Tensor,
        group_size: int = 128,
    ) -> tuple[torch.Tensor, ...]:
        assert group_size == 128, "Group size must be 128"
        return torch.ops.vllm.rocm_aiter_group_fp8_quant(input_2d, group_size)

    @staticmethod
    def is_triton_gemm_w8a8_tuned(n: int, k: int) -> bool:
        return (n, k) in [
            (1024, 8192),
            (2112, 7168),
            (3072, 1536),
            (32768, 8192),
            (4096, 7168),
            (4608, 7168),
            (512, 7168),
            (7168, 2048),
            (7168, 256),
            (8192, 1024),
            (8192, 32768),
        ]

    @staticmethod
    def is_triton_gemm_afp4wfp4_presh_ws_tuned(n: int, k: int) -> bool:
        return (n, k) in [
            (8192, 4096),
            (1280, 8192),
            (16384, 53248),
            (106496, 16384),
            (57344, 8192),
            (8192, 2048),
            (2560, 8192),
            (10240, 8192),
            (16384, 16384),
            (8192, 28672),
            (28672, 8192),
            (18432, 16384),
            (8192, 1024),
            (7168, 8192),
            (5120, 8192),
            (8192, 8192),
            (8192, 7168),
            (14336, 8192),
            (8192, 14336),
            (8192, 3584),
        ]

    @staticmethod
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
