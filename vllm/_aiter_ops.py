# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer


def is_aiter_found() -> bool:
    from importlib.util import find_spec

    return find_spec("aiter") is not None


# `find_spec` is not torch.compile compatible.
# In cases where aiter availability might have
# been checked in forward passes that are torch compiled.
# we keep this global outside to not cause torch compile breaks.
IS_AITER_FOUND = is_aiter_found()

# Can't use dtypes.fp8 directly inside an op
# because it returns wrong result on gfx942.
# This is a workaround to get the correct FP8 dtype.
# This might because that the get_gfx() is wrapped as a custom op.
if IS_AITER_FOUND:
    from aiter import dtypes

    AITER_FP8_DTYPE = dtypes.fp8


def if_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if
    ROCm AITER package is supported on gfx9 archs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # checks the platform, device arch and aiter library existence.

        if current_platform.is_rocm() and IS_AITER_FOUND:
            from vllm.platforms.rocm import on_gfx9

            if on_gfx9():
                return func(*args, **kwargs)

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
        except Exception:
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
    # CutlassScaledMMLinearKernel prepare weight `w_q` in [K, N] format
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
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

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


def _rocm_aiter_triton_gemm_afp4wfp4_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    return gemm_afp4wfp4(A, B, As, Bs.T, dtype=output_dtype)


def _rocm_aiter_triton_gemm_afp4wfp4_fake(
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


def _rocm_aiter_triton_gemm_a16w8_blockscale_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter.ops.triton.gemm.basic.gemm_a16w8_blockscale import gemm_a16w8_blockscale

    return gemm_a16w8_blockscale(A, B, Bs, dtype=output_dtype)


def _rocm_aiter_triton_gemm_a16w8_blockscale_fake(
    A: torch.Tensor,
    B: torch.Tensor,
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


def _rocm_aiter_rmsnorm_with_add_fp8_group_quant_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_group_quant

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
    return (x_quant, x_quant_scales, res)


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
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
        torch.empty_like(residual, device=residual.device),
    )


def _rocm_aiter_rmsnorm_fp8_group_quant_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_group_quant

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


def _rocm_aiter_2rmsnorm_1fp8_group_quant_impl(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight1: torch.Tensor,
    variance_epsilon1: float,
    weight2: torch.Tensor,
    variance_epsilon2: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_group_quant

    (x_quant, x_quant_scales), _, x2_out, _ = fused_rms_fp8_group_quant(
        x1,
        weight1,
        variance_epsilon1,
        x2,
        weight2,
        variance_epsilon2,
        group_size=group_size,
        dtype_quant=AITER_FP8_DTYPE,
        res1=None,
    )
    return (x_quant, x_quant_scales, x2_out)


def _rocm_aiter_2rmsnorm_1fp8_group_quant_fake(
    x1: torch.Tensor,
    x2: torch.Tensor,
    weight1: torch.Tensor,
    variance_epsilon1: float,
    weight2: torch.Tensor,
    variance_epsilon2: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x1.shape
    scale_shape = (
        M,
        (N + group_size - 1) // group_size,
    )
    return (
        torch.empty_like(x1, dtype=AITER_FP8_DTYPE, device=x1.device),
        torch.empty(scale_shape, dtype=torch.float32, device=x1.device),
        torch.empty_like(x2, dtype=x2.dtype, device=x1.device),
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

def _rocm_aiter_triton_fused_shared_expert_fp8_impl(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_gate_up: torch.Tensor,
    weight_scale_gate_up: torch.Tensor,
    hidden_states_moe_gate: torch.Tensor,
    weight_moe_gate: torch.Tensor,
    bias_shared: torch.Tensor,
    bias_moe_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_a16w16 import fused_gemm_a8w8_blockscale_a16w16
    from aiter.ops.triton.quant.fused_fp8_quant import fused_reduce_act_mul_fp8_group_quant
    
    shared_output, router_logits = fused_gemm_a8w8_blockscale_a16w16(hidden_states_shared, weight_gate_up, hidden_states_shared_scale, weight_scale_gate_up, hidden_states_moe_gate, weight_moe_gate, 
                                    bias_fp8=bias_shared, bias_bf16=bias_moe_gate, dtype=hidden_states_moe_gate.dtype, skip_reduce=True)
    if shared_output.dim() == 3:
        (shared_output_q, shared_output_s), router_logits = fused_reduce_act_mul_fp8_group_quant(shared_output, activation="silu", x2=router_logits, group_size=128, dtype_quant=AITER_FP8_DTYPE)
    else:
        (shared_output_q, shared_output_s), _ = fused_reduce_act_mul_fp8_group_quant(shared_output, activation="silu", x2=None, group_size=128, dtype_quant=AITER_FP8_DTYPE)
    return shared_output_q, shared_output_s, router_logits

def _rocm_aiter_triton_fused_shared_expert_fp8_fake(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_gate_up: torch.Tensor,
    weight_scale_gate_up: torch.Tensor,
    hidden_states_moe_gate: torch.Tensor,
    weight_moe_gate: torch.Tensor,
    bias_shared: torch.Tensor,
    bias_moe_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_shared.shape[0]
    N = weight_gate_up.shape[0]
    N_moe = weight_moe_gate.shape[0]
    device = hidden_states_shared.device
    group_size = 128
    assert N % 2 == 0
    N_half = N // 2
    assert N_half == N_moe, f"{weight_moe_gate.shape}"
    shared_output_q = torch.empty((M, N_half), dtype=AITER_FP8_DTYPE, device=device)
    shared_output_s = torch.empty((M, (N_half + group_size - 1) // group_size), dtype=torch.float32, device=device)
    router_logits = torch.empty((M, N_moe), dtype=hidden_states_moe_gate.dtype, device=device)
    return shared_output_q, shared_output_s, router_logits


def _rocm_aiter_triton_fused_down_proj_mul_add_fp8_impl(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_down_proj: torch.Tensor,
    weight_scale_down_proj: torch.Tensor,
    routed_scaling_factor: float,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_mul_add import fused_gemm_a8w8_blockscale_mul_add

    out = fused_gemm_a8w8_blockscale_mul_add(hidden_states_shared, weight_down_proj, hidden_states_shared_scale, weight_scale_down_proj, routed_scaling_factor, final_hidden_states, fuse_type=1)
    return out


def _rocm_aiter_triton_fused_down_proj_mul_add_fp8_fake(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_down_proj: torch.Tensor,
    weight_scale_down_proj: torch.Tensor,
    routed_scaling_factor: float,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(final_hidden_states)
    return out

def _rocm_aiter_triton_fused_shared_expert_fp4_impl(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_gate_up: torch.Tensor,
    weight_scale_gate_up: torch.Tensor,
    hidden_states_moe_gate: torch.Tensor,
    weight_moe_gate: torch.Tensor,
    bias_shared: torch.Tensor,
    bias_moe_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import fused_gemm_afp4wfp4_a16w16
    from aiter.ops.triton.quant.fused_mxfp4_quant import fused_reduce_act_mul_and_mxfp4_quant
    
    shared_output, router_logits = fused_gemm_afp4wfp4_a16w16(hidden_states_shared, weight_gate_up, hidden_states_shared_scale, weight_scale_gate_up.T, hidden_states_moe_gate, weight_moe_gate, 
                                    is_fp4_preshuffled=False, bias_fp4=bias_shared, bias_bf16=bias_moe_gate, dtype=hidden_states_moe_gate.dtype, skip_reduce=True)
    if shared_output.dim() == 3:
        (shared_output_q, shared_output_s), router_logits = fused_reduce_act_mul_and_mxfp4_quant(shared_output, activation="silu", x2=router_logits, shuffle=False, scale_shuffle_padding=False, dtype=hidden_states_moe_gate.dtype)
    else:
        (shared_output_q, shared_output_s), _ = fused_reduce_act_mul_and_mxfp4_quant(shared_output, activation="silu", x2=None, shuffle=False, scale_shuffle_padding=False, dtype=hidden_states_moe_gate.dtype)

    # assert bias_shared is None
    # shared_output = gemm_afp4wfp4(hidden_states_shared, weight_gate_up, hidden_states_shared_scale, weight_scale_gate_up.T)
    # router_logits = gemm_a16w16(hidden_states_moe_gate, weight_moe_gate, bias=bias_moe_gate) 
    # shared_output_q, shared_output_s = act_mul_and_mxfp4_quant(shared_output, activation="silu", shuffle=False, scale_shuffle_padding=False)
    
    return shared_output_q, shared_output_s, router_logits


def _rocm_aiter_triton_fused_shared_expert_fp4_fake(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_gate_up: torch.Tensor,
    weight_scale_gate_up: torch.Tensor,
    hidden_states_moe_gate: torch.Tensor,
    weight_moe_gate: torch.Tensor,
    bias_shared: torch.Tensor,
    bias_moe_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_shared.shape[0]
    N = weight_gate_up.shape[0]
    N_moe = weight_moe_gate.shape[0]
    device = hidden_states_shared.device
    group_size = 32
    assert N % 4 == 0
    N_half = N // 2
    assert N_half == 256, f"{weight_gate_up.shape}"
    assert N_half == N_moe, f"{weight_moe_gate.shape}"
    shared_output_q = torch.empty((M, N_half // 2), dtype=torch.uint8, device=device)
    shared_output_s = torch.empty((M, (N_half + group_size - 1) // group_size), dtype=torch.uint8, device=device)
    router_logits = torch.empty((M, N_moe), dtype=hidden_states_moe_gate.dtype, device=device)
    return shared_output_q, shared_output_s, router_logits


def _rocm_aiter_triton_fused_down_proj_mul_add_fp4_impl(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_down_proj: torch.Tensor,
    weight_scale_down_proj: torch.Tensor,
    routed_scaling_factor: float,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_mul_add import fused_gemm_afp4wfp4_mul_add

    out = fused_gemm_afp4wfp4_mul_add(hidden_states_shared, weight_down_proj, hidden_states_shared_scale, weight_scale_down_proj.T, routed_scaling_factor, final_hidden_states, fuse_type=1)
    return out


def _rocm_aiter_triton_fused_down_proj_mul_add_fp4_fake(
    hidden_states_shared: torch.Tensor,
    hidden_states_shared_scale: torch.Tensor,
    weight_down_proj: torch.Tensor,
    weight_scale_down_proj: torch.Tensor,
    routed_scaling_factor: float,
    final_hidden_states: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(final_hidden_states)
    return out

def _rocm_aiter_triton_qkv_a_proj_layernorm_fp8_impl(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant.fused_fp8_quant import fused_reduce_rms_fp8_group_quant
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    import aiter as rocm_aiter
    qkv_lora = gemm_a8w8_blockscale(hidden_states_quant, weight_qkv_a_proj, hidden_states_quant_scale, weight_scale_qkv_a_proj, skip_reduce=True)
    q_c, kv_c, k_pe = qkv_lora.split([q_lora_rank, kv_lora_rank, qk_rope_head_dim],
                                        dim=-1,
                                    )
    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        M = hidden_states_quant.shape[0]
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = fused_reduce_rms_fp8_group_quant(q_c, q_a_layernorm_weight, q_a_layernorm_variance_epsilon, 
                                            kv_c, kv_a_layernorm_weight, kv_a_layernorm_variance_epsilon, k_pe_reduced,
                                            group_size=128,
                                            dtype_quant=AITER_FP8_DTYPE, 
                                            dtype=torch.bfloat16,
                                            res1=None,
                                            out3=k_pe_reduced_out)
    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out
    return q_c, q_c_scale, kv_c_normed, k_pe

def _rocm_aiter_triton_qkv_a_proj_layernorm_fp8_fake(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank), dtype=AITER_FP8_DTYPE, device=device)
    q_c_scale = torch.empty((M, (q_lora_rank + 128 - 1) // 128), dtype=torch.float32, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe

def _rocm_aiter_triton_qkv_a_proj_layernorm_fp4_impl(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant.fused_mxfp4_quant import fused_reduce_rms_mxfp4_quant
        
    qkv_lora = gemm_afp4wfp4(hidden_states_quant, weight_qkv_a_proj, hidden_states_quant_scale, weight_scale_qkv_a_proj.T, skip_reduce=True)
    q_c, kv_c, k_pe = qkv_lora.split([q_lora_rank, kv_lora_rank, qk_rope_head_dim],
                                        dim=-1,
                                    )
    k_pe_reduced = None
    k_pe_reduced_out = None
    if k_pe.dim() == 3:
        M = hidden_states_quant.shape[0]
        device = hidden_states_quant.device
        k_pe_reduced = k_pe
        k_pe_reduced_out = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
    (q_c, q_c_scale), _, kv_c_normed, _, k_pe_reduced_out = fused_reduce_rms_mxfp4_quant(q_c, q_a_layernorm_weight, q_a_layernorm_variance_epsilon, 
                                            kv_c, kv_a_layernorm_weight, kv_a_layernorm_variance_epsilon, k_pe_reduced,
                                            res1=None,
                                            shuffle=False,
                                            scale_shuffle_padding=False,
                                            dtype=torch.bfloat16,
                                            out3=k_pe_reduced_out)
    
    if k_pe_reduced_out is not None:
        k_pe = k_pe_reduced_out
    return q_c, q_c_scale, kv_c_normed, k_pe

def _rocm_aiter_triton_qkv_a_proj_layernorm_fp4_fake(
    hidden_states_quant: torch.Tensor,
    hidden_states_quant_scale: torch.Tensor,
    weight_qkv_a_proj: torch.Tensor,
    weight_scale_qkv_a_proj: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    q_a_layernorm_variance_epsilon: float,
    kv_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_variance_epsilon: float,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = hidden_states_quant.shape[0]
    device = hidden_states_quant.device
    q_c = torch.empty((M, q_lora_rank // 2), dtype=torch.uint8, device=device)
    q_c_scale = torch.empty((M, (q_lora_rank + 32 - 1) // 32), dtype=torch.float32, device=device)
    kv_c_normed = torch.empty((M, kv_lora_rank), dtype=torch.bfloat16, device=device)
    k_pe = torch.empty((M, q_lora_rank + kv_lora_rank + qk_rope_head_dim), dtype=torch.bfloat16, device=device)[..., :qk_rope_head_dim]
    return q_c, q_c_scale, kv_c_normed, k_pe

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
    _FP4BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP4BMM
    _FP4_GEMM_DYNAMIC_QUANT_ASM = envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM
    _TRITON_ROTARY_EMBED = envs.VLLM_ROCM_USE_AITER_TRITON_ROPE
    _MOE_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
    _TRITON_SHARED_EXPERTS_ENABLED = envs.VLLM_ROCM_USE_AITER_TRITON_FUSION_SHARED_EXPERTS
    _TRITON_UNQUANT_GEMM = envs.VLLM_ROCM_USE_AITER_TRITON_GEMM

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
        return cls.is_linear_enabled()

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
    def is_fusion_triton_shared_experts_enabled(cls) -> bool:
        return cls.is_fused_moe_enabled() and cls._TRITON_SHARED_EXPERTS_ENABLED

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
    def is_fp4bmm_enabled(cls) -> bool:
        """ "Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._FP4BMM_ENABLED and current_platform.supports_mx()

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
                op_name="rocm_aiter_triton_gemm_afp4wfp4",
                op_func=_rocm_aiter_triton_gemm_afp4wfp4_impl,
                fake_impl=_rocm_aiter_triton_gemm_afp4wfp4_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_triton_gemm_a16w8_blockscale",
                op_func=_rocm_aiter_triton_gemm_a16w8_blockscale_impl,
                fake_impl=_rocm_aiter_triton_gemm_a16w8_blockscale_fake,
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
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm_fp8_group_quant",
                op_func=_rocm_aiter_rmsnorm_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_rmsnorm_fp8_group_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_2rmsnorm_1fp8_group_quant",
                op_func=_rocm_aiter_2rmsnorm_1fp8_group_quant_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_2rmsnorm_1fp8_group_quant_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm_with_add_fp8_group_quant",
                op_func=_rocm_aiter_rmsnorm_with_add_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_rmsnorm_with_add_fp8_group_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_group_fp8_quant",
                op_func=_rocm_aiter_group_fp8_quant_impl,
                fake_impl=_rocm_aiter_group_fp8_quant_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_act_mul_and_fp8_group_quant",
                op_func=_rocm_aiter_act_mul_and_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_act_mul_and_fp8_group_quant_fake,
            )
            
            direct_register_custom_op(
                op_name="rocm_aiter_triton_fused_shared_expert_fp8",
                op_func=_rocm_aiter_triton_fused_shared_expert_fp8_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_triton_fused_shared_expert_fp8_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_triton_fused_down_proj_mul_add_fp8",
                op_func=_rocm_aiter_triton_fused_down_proj_mul_add_fp8_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_triton_fused_down_proj_mul_add_fp8_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            
            direct_register_custom_op(
                op_name="rocm_aiter_triton_fused_shared_expert_fp4",
                op_func=_rocm_aiter_triton_fused_shared_expert_fp4_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_triton_fused_shared_expert_fp4_fake,
                dispatch_key=current_platform.dispatch_key,
            )
            
            direct_register_custom_op(
                op_name="rocm_aiter_triton_fused_down_proj_mul_add_fp4",
                op_func=_rocm_aiter_triton_fused_down_proj_mul_add_fp4_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_triton_fused_down_proj_mul_add_fp4_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_triton_qkv_a_proj_layernorm_fp8",
                op_func=_rocm_aiter_triton_qkv_a_proj_layernorm_fp8_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_triton_qkv_a_proj_layernorm_fp8_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_triton_qkv_a_proj_layernorm_fp4",
                op_func=_rocm_aiter_triton_qkv_a_proj_layernorm_fp4_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_triton_qkv_a_proj_layernorm_fp4_fake,
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
    def triton_fp4_gemm_dynamic_qaunt(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = torch.bfloat16,
        x_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
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
        from aiter.ops.triton.rope.rope import rope_cached_thd_positions_2c_fwd_inplace

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
    def triton_fp4_bmm(
        X: torch.Tensor,
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
        dtype: torch.dtype | None = torch.bfloat16,
        YQ: torch.Tensor | None = None,
        transpose_bm: bool | None = False,
        config: dict | None = None,
        y_scale: dict | None = None,
    ) -> torch.Tensor:
        # ruff: noqa: E501 # isort: skip
        from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
            batched_gemm_a16wfp4 as aiter_triton_fp4_bmm,
        )

        return aiter_triton_fp4_bmm(
            X,
            WQ,
            w_scale,
            dtype=dtype,
            y=YQ,
            config=config,
            transpose_bm=transpose_bm,
            prequant=True,
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
        from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
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
    def triton_gemm_a16w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        Bs: torch.Tensor,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_triton_gemm_a16w8_blockscale(
            A, B, Bs, output_dtype
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
