# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Protocol

import torch
from torch._ops import OpOverload
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils.import_utils import PlaceholderModule
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_aiter_sparse_attn_indexer,
    rocm_aiter_sparse_attn_indexer_fake,
)

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

# fp8_dtype is not cached.
# on ROCm the fp8_dtype always calls is_fp8_fnuz
# which is a host op, so we cache it once here.
FP8_DTYPE = current_platform.fp8_dtype()


def is_aiter_found() -> bool:
    from importlib.util import find_spec

    return find_spec("aiter") is not None


# `find_spec` is not torch.compile compatible.
# In cases where aiter availability might have
# been checked in forward passes that are torch compiled.
# we keep this global outside to not cause torch compile breaks.
IS_AITER_FOUND = is_aiter_found()


class AiterCustomAllreduceProto(Protocol):
    max_size: int
    world_size: int
    fully_connected: bool

    @contextmanager
    def capture(self): ...
    def close(self) -> None: ...
    def fused_ar_rms(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        *,
        w: torch.Tensor,
        eps: float,
        registered: bool = False,
        use_1stage: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def should_custom_ar(self, inp: torch.Tensor) -> bool: ...


def is_aiter_found_and_supported() -> bool:
    """Check if AITER library is available and platform supports it.

    Checks: platform (ROCm), device arch (gfx9), and library existence.
    Does NOT check environment variables - that's handled by rocm_aiter_ops.is_enabled().

    This function determines if aiter CAN be used, not if it SHOULD be used.

    Separation of concerns:
    - This function: Can aiter work on this system? (platform + library availability)
    - rocm_aiter_ops.is_enabled(): Should aiter be used by default? (adds env var check)
    - Backend selection: Can explicitly request aiter regardless of env var

    This allows explicit backend selection via attention_config to work even when
    VLLM_ROCM_USE_AITER=0, while preventing unwanted JIT warnings for auto-discovery.
    """
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_mi3xx

        return on_mi3xx()
    return False


@functools.cache
def _load_gemm_tuned_configs(
    q_dtype_w: torch.dtype, csv_path: str
) -> set[tuple[int, int, int]]:
    try:
        df = pd.read_csv(csv_path).drop_duplicates()
        df = df[df["q_dtype_w"] == str(q_dtype_w)]
        return set(zip(df["N"].astype(int), df["K"].astype(int), df["M"].astype(int)))
    except Exception:
        return set()


def _check_kernel_tuned(N: int, K: int, q_dtype_w: torch.dtype, csv_path: str) -> bool:
    configs = _load_gemm_tuned_configs(q_dtype_w, csv_path)
    l_m = (
        [1, 2, 4]
        + list(range(8, 513, 8))
        + [1024, 1536]
        + [2**i for i in range(11, 19)]
    )
    return any((N, K, M) in configs for M in l_m)


def if_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if
    ROCm AITER package is supported and enabled on gfx9 archs.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_aiter_found_and_supported():
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
    num_local_tokens: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
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
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
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
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
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
    num_shared_experts: int = 0,
    shared_expert_scoring_func: str = "",
) -> None:
    from aiter import topk_softmax

    topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        num_shared_experts,
        shared_expert_scoring_func,
    )


def _rocm_aiter_topk_softmax_fake(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    num_shared_experts: int = 0,
    shared_expert_scoring_func: str = "",
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


def _rocm_aiter_fused_topk_impl(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.fused_moe import fused_topk

    # fused_topk returns (topk_weights, topk_indices)
    return fused_topk(x, router_logits, top_k, gate_up)


def _rocm_aiter_fused_topk_fake(
    x: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    gate_up: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = x.shape[0]
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=x.device
    )
    topk_indices = torch.empty((num_tokens, top_k), dtype=torch.int32, device=x.device)
    return topk_weights, topk_indices


# Cache whether aiter supports FP8 MLA parameters
_AITER_MLA_SUPPORTS_FP8: bool | None = None
_AITER_HAS_FUSED_QK_RMSNORM: bool | None = None


def check_aiter_fused_qk_rmsnorm() -> bool:
    """Check if aiter provides fused_qk_rmsnorm (requires AITer >= PR #2442)."""
    global _AITER_HAS_FUSED_QK_RMSNORM
    if _AITER_HAS_FUSED_QK_RMSNORM is None:
        try:
            from aiter.ops.fused_qk_norm_rope_cache_quant import (  # noqa: F401
                fused_qk_rmsnorm,
            )

            _AITER_HAS_FUSED_QK_RMSNORM = True
        except (ImportError, ModuleNotFoundError, AttributeError):
            _AITER_HAS_FUSED_QK_RMSNORM = False
    return _AITER_HAS_FUSED_QK_RMSNORM


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
    work_meta_data: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info_set: torch.Tensor | None = None,
    reduce_indptr: torch.Tensor | None = None,
    reduce_final_map: torch.Tensor | None = None,
    reduce_partial_map: torch.Tensor | None = None,
) -> None:
    from aiter.mla import mla_decode_fwd

    kwargs: dict[str, float | torch.Tensor | None] = {
        "sm_scale": sm_scale,
        "logit_cap": logit_cap,
    }

    # Only pass q_scale and kv_scale if the aiter library supports them
    if _check_aiter_mla_fp8_support():
        kwargs["q_scale"] = q_scale
        kwargs["kv_scale"] = kv_scale

    if work_meta_data is not None:
        assert work_indptr is not None, (
            "work_indptr must be provided with work_meta_data"
        )
        assert work_info_set is not None, (
            "work_info_set must be provided with work_meta_data"
        )
        assert reduce_indptr is not None, (
            "reduce_indptr must be provided with work_meta_data"
        )
        assert reduce_final_map is not None, (
            "reduce_final_map must be provided with work_meta_data"
        )
        assert reduce_partial_map is not None, (
            "reduce_partial_map must be provided with work_meta_data"
        )
        kwargs["work_meta_data"] = work_meta_data
        kwargs["work_indptr"] = work_indptr
        kwargs["work_info_set"] = work_info_set
        kwargs["reduce_indptr"] = reduce_indptr
        kwargs["reduce_final_map"] = reduce_final_map
        kwargs["reduce_partial_map"] = reduce_partial_map

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
    work_meta_data: torch.Tensor | None = None,
    work_indptr: torch.Tensor | None = None,
    work_info_set: torch.Tensor | None = None,
    reduce_indptr: torch.Tensor | None = None,
    reduce_final_map: torch.Tensor | None = None,
    reduce_partial_map: torch.Tensor | None = None,
) -> None:
    pass


def _rocm_aiter_w8a8_gemm_impl(
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


def _rocm_aiter_w8a8_gemm_fake(
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


def _rocm_aiter_preshuffled_per_token_w8a8_gemm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter import gemm_a8w8_bpreshuffle

    output = gemm_a8w8_bpreshuffle(A, B, As, Bs, None, output_dtype)
    if bias is not None:
        output.add_(bias)
    return output


def _rocm_aiter_preshuffled_per_token_w8a8_gemm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    m = A.shape[0]
    n = B.shape[0]
    return torch.empty(m, n, dtype=output_dtype, device=A.device)


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


def _rocm_aiter_rmsnorm_fused_add_dynamic_quant_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import aiter as rocm_aiter

    assert quant_dtype in [torch.int8, FP8_DTYPE]

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

    assert quant_dtype in [torch.int8, FP8_DTYPE]

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


def _rocm_aiter_fused_allreduce_rmsnorm_impl(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    assert aiter_ar is not None, "aiter allreduce must be initialized"

    total_bytes = input_.numel() * input_.element_size()
    hidden_dim = input_.shape[-1]
    token_num = input_.shape[0]
    if input_.dtype in (torch.bfloat16, torch.float16):
        pack_size = 16 // input_.element_size()
        hidden_ok = hidden_dim % pack_size == 0 and hidden_dim // pack_size <= 1024
    else:
        hidden_ok = False
    token_ok = token_num <= 80
    world_size = aiter_ar.world_size
    full_nvlink = aiter_ar.fully_connected

    if world_size == 2:
        size_ok = True
    elif full_nvlink and world_size <= 4:
        size_ok = total_bytes < 256 * 1024
    elif full_nvlink and world_size <= 8:
        size_ok = total_bytes < 128 * 1024
    else:
        size_ok = False

    use_1stage = hidden_ok and token_ok and size_ok

    result = aiter_ar.fused_ar_rms(
        input_,
        residual,
        w=weight,
        eps=epsilon,
        registered=torch.cuda.is_current_stream_capturing(),
        use_1stage=use_1stage,
    )
    assert result is not None
    return result[0], result[1]


def _rocm_aiter_fused_allreduce_rmsnorm_fake(
    input_: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input_), torch.empty_like(residual)


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

    assert quant_dtype in [torch.int8, FP8_DTYPE]

    out_shape = x.shape
    out = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
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
        torch.empty(x.shape, dtype=quant_dtype, device=x.device),
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
        dtype_quant=FP8_DTYPE,
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
        torch.empty_like(x, dtype=FP8_DTYPE, device=x.device),
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
        dtype_quant=FP8_DTYPE,
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
        torch.empty_like(x, dtype=FP8_DTYPE, device=x.device),
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
    )


def _rocm_aiter_fused_rms_gated_fp8_group_quant_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor,
    eps: float,
    norm_before_gate: bool,
    activation: str,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gated-RMSNorm + FP8 group quantization via aiter Triton kernel."""
    from aiter.ops.triton.quant import fused_rms_gated_fp8_group_quant

    return fused_rms_gated_fp8_group_quant(
        x,
        weight,
        bias,
        z,
        eps,
        norm_before_gate=norm_before_gate,
        activation=activation,
        out_dtype=FP8_DTYPE,
        group_size=group_size,
    )


def _rocm_aiter_fused_rms_gated_fp8_group_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    z: torch.Tensor,
    eps: float,
    norm_before_gate: bool,
    activation: str,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    scale_shape = (M, (N + group_size - 1) // group_size)
    return (
        torch.empty_like(x, dtype=FP8_DTYPE, device=x.device),
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
    )


def _rocm_aiter_group_fp8_quant_impl(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % group_size == 0, "Input shape must be divisible by group size"
    from aiter import QuantType, get_hip_quant

    aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
    return aiter_per1x128_quant(x.contiguous(), quant_dtype=FP8_DTYPE)


def _rocm_aiter_group_fp8_quant_fake(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    x_fp8 = torch.empty((M, N), dtype=FP8_DTYPE, device=x.device)
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
        dtype_quant=FP8_DTYPE,
    )


def _rocm_aiter_act_mul_and_fp8_group_quant_fake(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    assert N % 2 == 0
    N_half = N // 2
    x_fp8 = torch.empty((M, N_half), dtype=FP8_DTYPE, device=x.device)
    out_bs = torch.empty(
        (
            M,
            (N_half + group_size - 1) // group_size,
        ),
        dtype=torch.float32,
        device=x.device,
    )
    return x_fp8, out_bs


def _rocm_aiter_triton_add_rmsnorm_pad_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: torch.Tensor,
    x_pad_to_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad

    return fused_add_rmsnorm_pad(
        x,
        weight,
        variance_epsilon,
        residual,
        x_pad_to_multiple=x_pad_to_multiple,
    )


def _rocm_aiter_triton_add_rmsnorm_pad_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
    residual: torch.Tensor,
    x_pad_to_multiple: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    if x_pad_to_multiple > 0:
        N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    else:
        N_out = N
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    residual_out = torch.empty_like(residual)
    return out, residual_out


def _fused_mla_dual_rms_norm_impl(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x2: torch.Tensor,
    x2_weight: torch.Tensor,
    x1_epsilon: float,
    x2_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from aiter.ops.fused_qk_norm_rope_cache_quant import fused_qk_rmsnorm
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "fused_qk_rmsnorm requires a newer AITer version "
            "(>= PR #2442). Please upgrade aiter or disable the "
            "fuse_mla_dual_rms_norm pass."
        ) from exc

    return fused_qk_rmsnorm(
        q=x1,
        q_weight=x1_weight,
        q_eps=x1_epsilon,
        k=x2,
        k_weight=x2_weight,
        k_eps=x2_epsilon,
    )


def _fused_mla_dual_rms_norm_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x2: torch.Tensor,
    x2_weight: torch.Tensor,
    x1_epsilon: float,
    x2_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty_like(x1), torch.empty_like(x2))


def _rocm_aiter_gemm_a8wfp4_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from aiter.ops.triton.gemm_a8wfp4 import gemm_a8wfp4

    M, N = x.shape[0], w.shape[0]
    y = torch.empty(M, N, dtype=out_dtype, device=x.device)
    gemm_a8wfp4(
        x=x,
        w=w,
        y=y,
        x_scales=x_scales,
        w_scales=w_scales,
        dtype=out_dtype,
        config=None,
    )
    return y


def _rocm_aiter_gemm_a8wfp4_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(x.shape[0], w.shape[0], dtype=out_dtype, device=x.device)


def _triton_rotary_embedding_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    offsets: torch.Tensor | None = None,
) -> None:
    # Modifies query and key in-place
    from aiter.ops.triton.rope.rope import (
        rope_cached_thd_positions_offsets_2c_fwd_inplace,
    )

    num_tokens = positions.numel()
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    query_shape = query.shape
    key_shape = key.shape
    rotate_style = 0 if is_neox else 1
    rotary_dim = head_size

    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)
    query_ = query[..., :rotary_dim]
    key_ = key[..., :rotary_dim]
    positions = positions.view(*query.shape[:1])
    rope_cached_thd_positions_offsets_2c_fwd_inplace(
        query_,
        key_,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part=True,
        nope_first=False,
    )
    query = query.view(query_shape)
    key = key.view(key_shape)


def _triton_rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool,
    offsets: torch.Tensor | None = None,
) -> None:
    return


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
            result = rocm_aiter_ops.per_token_quant(x, FP8_DTYPE)

    Operations:
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
    # Lazily probed: whether aiter.topk_softmax supports the
    # num_shared_experts / shared_expert_scoring_func args (7-arg form).
    _TOPK_SOFTMAX_FUSED_SIGMOID: bool | None = None

    _ALL_REDUCE_MAX_SIZE: int = 8192 * 1024 * 8 * 2
    _CUSTOM_ALL_REDUCE: AiterCustomAllreduceProto | None = None

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

    @staticmethod
    def get_aiter_activation_type(activation_str: str):
        """
        Given an activation type as a string, returns the corresponding aiter ActivationType enum.
        Supported activation types: "no", "none", "silu", "gelu", "swiglu".
        Returns None if the mapping fails.

        Args:
            activation_str (str): Activation type as string.

        Returns:
            Aiter ActivationType enum value, or None if not found.
        """
        # Import only locally, since aiter may not always be available.
        try:
            from aiter import ActivationType
        except ImportError:
            return None

        if not isinstance(activation_str, str):
            return None

        name = activation_str.strip().lower()
        mapping = {
            "none": ActivationType.No,
            "no": ActivationType.No,
            "silu": ActivationType.Silu,
            "gelu": ActivationType.Gelu,
            "swiglu": ActivationType.Swiglu,
        }
        return mapping.get(name)

    @staticmethod
    def get_aiter_quant_type(quant_type_str: str):
        """
        Given a quantization type as a string, returns the corresponding aiter QuantType enum.
        Supported quantization types: "no", "per_tensor", "per_token", "per_1x32", "per_1x128", "per_128x128".
        Returns None if the mapping fails.

        Args:
            quant_type_str (str): Quantization type as string.

        Returns:
            Aiter QuantType enum value, or None if not found.
        """
        try:
            from aiter import QuantType
        except ImportError:
            return None

        if not isinstance(quant_type_str, str):
            return None

        name = quant_type_str.strip().lower()
        mapping = {
            "no": QuantType.No,
            "per_tensor": QuantType.per_Tensor,
            "per_token": QuantType.per_Token,
            "per_1x32": QuantType.per_1x32,
            "per_1x128": QuantType.per_1x128,
            "per_128x128": QuantType.per_128x128,
        }
        return mapping.get(name)

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
    def is_fused_moe_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FMOE_ENABLED

    @classmethod
    @if_aiter_supported
    def is_fusion_moe_shared_experts_enabled(cls) -> bool:
        return cls.is_fused_moe_enabled() and cls._MOE_SHARED_EXPERTS_ENABLED

    @classmethod
    @if_aiter_supported
    def topk_softmax_supports_fused_sigmoid(cls) -> bool:
        """Check if topk_softmax supports fused shared expert activation."""
        if cls._TOPK_SOFTMAX_FUSED_SIGMOID is None:
            try:
                import inspect

                from aiter import topk_softmax

                params = inspect.signature(topk_softmax).parameters
                if "num_shared_experts" in params:
                    cls._TOPK_SOFTMAX_FUSED_SIGMOID = True
                else:
                    # @compile_ops wrapper loses the original signature.
                    # Fall back to the torch custom op schema.
                    import torch

                    schema = getattr(
                        getattr(torch.ops.aiter, "topk_softmax", None), "default", None
                    )
                    schema_str = str(getattr(schema, "_schema", ""))
                    cls._TOPK_SOFTMAX_FUSED_SIGMOID = "num_shared_experts" in schema_str
            except (ImportError, ValueError):
                cls._TOPK_SOFTMAX_FUSED_SIGMOID = False
        return cls._TOPK_SOFTMAX_FUSED_SIGMOID

    @classmethod
    @if_aiter_supported
    def fuse_sigmoid_in_kernel(cls, aiter_topK_meta_data: object) -> bool:
        """Whether fused shared-expert sigmoid in the topk kernel is usable.

        Combines the cached static capability checks (FSE enabled, fused-moe
        enabled, topk_softmax supports fused sigmoid) with the runtime
        readiness check (topK meta-data buffer initialized).

        ``aiter_topK_meta_data`` is accepted as a parameter rather than
        imported internally so callers cannot hit initialization-order
        issues where the module-level global has not been set yet.
        """
        return (
            cls.is_fusion_moe_shared_experts_enabled()
            and cls.topk_softmax_supports_fused_sigmoid()
            and aiter_topK_meta_data is not None
        )

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
        return cls._SHUFFLE_KV_CACHE_ENABLED

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
        from vllm.platforms.rocm import on_gfx950

        return cls._AITER_ENABLED and cls._FP4BMM_ENABLED and on_gfx950()

    @classmethod
    @if_aiter_supported
    def is_asm_fp4_gemm_dynamic_quant_enabled(cls) -> bool:
        from vllm.platforms.rocm import on_gfx950

        return cls._AITER_ENABLED and cls._FP4_GEMM_DYNAMIC_QUANT_ASM and on_gfx950()

    @classmethod
    @if_aiter_supported
    def is_triton_rotary_embed_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._TRITON_ROTARY_EMBED

    @classmethod
    @if_aiter_supported
    def is_triton_gemm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._TRITON_UNQUANT_GEMM

    @classmethod
    @if_aiter_supported
    def is_tgemm_enabled(cls) -> bool:
        from vllm.platforms.rocm import on_gfx950

        return cls.is_linear_enabled() and on_gfx950()

    @classmethod
    def initialize_aiter_allreduce(
        cls, group: ProcessGroup, device: torch.device
    ) -> None:
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            cls._CUSTOM_ALL_REDUCE = AiterCustomAllreduce(group, device)
        except Exception:
            cls._CUSTOM_ALL_REDUCE = None

    @classmethod
    def get_aiter_allreduce(cls) -> AiterCustomAllreduceProto | None:
        return cls._CUSTOM_ALL_REDUCE

    @classmethod
    def destroy_aiter_allreduce(cls) -> None:
        if cls._CUSTOM_ALL_REDUCE is not None:
            cls._CUSTOM_ALL_REDUCE.close()
            cls._CUSTOM_ALL_REDUCE = None

    @classmethod
    def get_aiter_allreduce_max_size(cls) -> int | None:
        # effective max input size (based on upstream aiter version: v0.1.10.post3)
        # https://github.com/ROCm/aiter/blob/6a0e7b26ccf33164785531212cc2ec2cde0b9243/aiter/dist/device_communicators/custom_all_reduce.py#L272-L273
        return int(cls._ALL_REDUCE_MAX_SIZE / 2)

    @classmethod
    @if_aiter_supported
    def are_gdn_triton_kernels_available(cls) -> bool:
        """Check if AITER Triton kernels for GDN attention are importable.

        These are optional Triton kernels (conv1d fast-path, gated delta net)
        used by GatedDeltaNetAttention's decode fast-path.  They may be absent
        in older aiter builds.
        """
        if not cls._AITER_ENABLED:
            return False
        try:
            import aiter.ops.triton.causal_conv1d_update_single_token  # noqa: F401
            import aiter.ops.triton.gated_delta_net  # noqa: F401
            from aiter.ops.triton.quant import (  # noqa: F401
                fused_rms_gated_fp8_group_quant,
            )

            return True
        except (ImportError, ModuleNotFoundError):
            return False

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
                op_name="rocm_aiter_fused_topk",
                op_func=_rocm_aiter_fused_topk_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_fused_topk_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_mla_decode_fwd",
                op_func=_rocm_aiter_mla_decode_fwd_impl,
                mutates_args=["o"],
                fake_impl=_rocm_aiter_mla_decode_fwd_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_w8a8_gemm",
                op_func=_rocm_aiter_w8a8_gemm_impl,
                fake_impl=_rocm_aiter_w8a8_gemm_fake,
            )

            direct_register_custom_op(
                op_name="_rocm_aiter_preshuffled_per_token_w8a8_gemm",
                op_func=_rocm_aiter_preshuffled_per_token_w8a8_gemm_impl,
                fake_impl=_rocm_aiter_preshuffled_per_token_w8a8_gemm_fake,
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
                op_name="rocm_aiter_fused_rms_gated_fp8_group_quant",
                op_func=_rocm_aiter_fused_rms_gated_fp8_group_quant_impl,
                fake_impl=_rocm_aiter_fused_rms_gated_fp8_group_quant_fake,
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
                op_name="rocm_aiter_triton_add_rmsnorm_pad",
                op_func=_rocm_aiter_triton_add_rmsnorm_pad_impl,
                fake_impl=_rocm_aiter_triton_add_rmsnorm_pad_fake,
                dispatch_key=current_platform.dispatch_key,
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

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_a8wfp4",
                op_func=_rocm_aiter_gemm_a8wfp4_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_gemm_a8wfp4_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            # Register rocm aiter rotary embedding custom op
            direct_register_custom_op(
                op_name="rocm_aiter_triton_rotary_embedding",
                op_func=_triton_rotary_embedding_impl,
                mutates_args=["query", "key"],  # These tensors are modified in-place
                fake_impl=_triton_rotary_embedding_fake,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_fused_allreduce_rmsnorm",
                op_func=_rocm_aiter_fused_allreduce_rmsnorm_impl,
                fake_impl=_rocm_aiter_fused_allreduce_rmsnorm_fake,
            )

            direct_register_custom_op(
                op_name="fused_mla_dual_rms_norm",
                op_func=_fused_mla_dual_rms_norm_impl,
                mutates_args=[],
                fake_impl=_fused_mla_dual_rms_norm_fake,
            )

            _OPS_REGISTERED = True

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
    def get_fused_rms_gated_fp8_group_quant_op() -> OpOverload:
        """Return the fused gated-RMSNorm + FP8 group quant custom op."""
        return torch.ops.vllm.rocm_aiter_fused_rms_gated_fp8_group_quant.default

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
    def get_triton_add_rmsnorm_pad_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_triton_add_rmsnorm_pad.default

    @staticmethod
    def get_triton_rotary_embedding_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_triton_rotary_embedding.default

    @staticmethod
    def get_fused_allreduce_rmsnorm_op() -> OpOverload:
        return torch.ops.vllm.rocm_aiter_fused_allreduce_rmsnorm.default

    @staticmethod
    def get_fused_mla_dual_rms_norm_op() -> OpOverload:
        return torch.ops.vllm.fused_mla_dual_rms_norm.default

    @staticmethod
    def w8a8_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_w8a8_gemm(A, B, As, Bs, bias, output_dtype)

    @staticmethod
    def preshuffled_per_token_w8a8_gemm(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm._rocm_aiter_preshuffled_per_token_w8a8_gemm(
            A, B, As, Bs, bias, output_dtype
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
        hidden_pad: int = 0,
        intermediate_pad: int = 0,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
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
            hidden_pad,
            intermediate_pad,
            bias1,
            bias2,
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
        num_shared_experts: int = 0,
        shared_expert_scoring_func: str = "",
    ) -> tuple[torch.Tensor, ...]:
        torch.ops.vllm.rocm_aiter_topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            num_shared_experts,
            shared_expert_scoring_func,
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
        if correction_bias.dtype != gating_output.dtype:
            correction_bias = correction_bias.to(gating_output.dtype)
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
    def fused_topk(
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        gate_up: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.rocm_aiter_fused_topk(x, router_logits, top_k, gate_up)

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
        work_meta_data: torch.Tensor | None = None,
        work_indptr: torch.Tensor | None = None,
        work_info_set: torch.Tensor | None = None,
        reduce_indptr: torch.Tensor | None = None,
        reduce_final_map: torch.Tensor | None = None,
        reduce_partial_map: torch.Tensor | None = None,
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
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
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
    def gemm_a8wfp4(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_a8wfp4(
            x, w, x_scales, w_scales, out_dtype
        )

    @staticmethod
    def triton_fp4_gemm_dynamic_quant(
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
    def triton_rope_and_cache(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        flash_layout: bool,
        apply_scale: bool,
    ):
        from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache

        cos, sin = cos_sin_cache.chunk(2, dim=-1)
        fused_qk_rope_reshape_and_cache(
            query,
            key,
            value,
            key_cache,
            value_cache,
            layer_slot_mapping,
            positions,
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox,
            flash_layout=flash_layout,
            apply_scale=apply_scale,
            q_out=query,
            k_out=key,
            output_zeros=False,
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    def is_shuffled_per_token_w8a8_gemm_tuned(
        N: int, K: int, q_dtype_w: torch.dtype
    ) -> bool:
        import aiter.ops.gemm_op_a8w8 as aiter_gemm_a8w8_ops

        csv_path = (
            aiter_gemm_a8w8_ops.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
        )
        return _check_kernel_tuned(N, K, q_dtype_w, csv_path)

    @staticmethod
    def is_per_token_w8a8_gemm_tuned(N: int, K: int, q_dtype_w: torch.dtype) -> bool:
        import aiter.ops.gemm_op_a8w8 as aiter_gemm_a8w8_ops

        csv_path = aiter_gemm_a8w8_ops.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE
        return _check_kernel_tuned(N, K, q_dtype_w, csv_path)

    @staticmethod
    def shuffle_weight(
        tensor: torch.Tensor, layout: tuple[int, int] = (16, 16)
    ) -> torch.Tensor:
        from aiter.ops.shuffle import shuffle_weight

        return shuffle_weight(tensor, layout=layout)

    @staticmethod
    def shuffle_weight_a16w4(
        tensor: "torch.Tensor",
        nLane: int,
        gate_up: bool,
    ) -> "torch.Tensor":
        """
        Shuffles the weight tensor into (A16W4) layout for AITER kernels.

        Args:
            tensor: The input weight tensor to be shuffled.
            layout: The block layout to use, defaults to (16, 4).

        Returns:
            torch.Tensor: The shuffled tensor.
        """
        from aiter.ops.shuffle import shuffle_weight_a16w4

        return shuffle_weight_a16w4(tensor, nLane, gate_up)

    @staticmethod
    def shuffle_scale_a16w4(
        tensor: "torch.Tensor",
        num_experts: int,
        gate_up: bool,
    ) -> "torch.Tensor":
        """
        Shuffles the scale tensor into (A16W4) layout for AITER kernels.

        Args:
            tensor: The input scale tensor to be shuffled.
            num_experts: Number of experts, needed for reshaping logic.
            gate_up: Whether the scale is for w13 (True) or w2 (False).

        Returns:
            torch.Tensor: The shuffled scale tensor.
        """
        from aiter.ops.shuffle import shuffle_scale_a16w4

        return shuffle_scale_a16w4(tensor, num_experts, gate_up)

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

    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int | None = None,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int, int] | None = None,
        alibi_slopes: torch.Tensor | None = None,
        return_lse: bool = False,
        out: torch.Tensor | None = None,
    ):
        """
        Flash attention with variable length sequences.

        This function is NOT wrapped with @is_aiter_supported decorator
        to allow explicit backend selection via attention_config to work
        even when VLLM_ROCM_USE_AITER=0.

        Note: This performs lazy import of aiter.flash_attn_varlen_func
        """
        from aiter import flash_attn_varlen_func

        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            out=out,
        )

    @staticmethod
    def pa_fwd_asm(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables_stride0: int,
        K_QScale: torch.Tensor,
        V_QScale: torch.Tensor,
        out_: torch.Tensor,
    ):
        """
        Paged attention forward pass using assembly kernel.

        This function is NOT wrapped with @is_aiter_supported decorator
        to allow explicit backend selection via attention_config to work
        even when VLLM_ROCM_USE_AITER=0.

        Note: This performs lazy import of aiter.pa_fwd_asm
        """
        from aiter import pa_fwd_asm

        return pa_fwd_asm(
            Q=Q,
            K=K,
            V=V,
            block_tables=block_tables,
            context_lens=context_lens,
            block_tables_stride0=block_tables_stride0,
            K_QScale=K_QScale,
            V_QScale=V_QScale,
            out_=out_,
        )

    @staticmethod
    def paged_attention_common(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        tmp_out: torch.Tensor,
        max_logits: torch.Tensor,
        exp_sums: torch.Tensor,
        max_seq_len: int,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables_stride0: int,
        scale: float,
        K_QScale_hip: torch.Tensor,
        V_QScale_hip: torch.Tensor,
        K_QScale_asm: torch.Tensor,
        V_QScale_asm: torch.Tensor,
        out_: torch.Tensor,
        kv_cache_dtype: str,
    ):
        """
        Paged attention common function.

        This function is NOT wrapped with @is_aiter_supported decorator
        to allow explicit backend selection via attention_config to work
        even when VLLM_ROCM_USE_AITER=0.

        Note: This performs lazy import of aiter.paged_attention_common
        """
        from aiter import paged_attention_common

        return paged_attention_common(
            Q=Q,
            K=K,
            V=V,
            tmp_out=tmp_out,
            max_logits=max_logits,
            exp_sums=exp_sums,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            context_lens=context_lens,
            block_tables_stride0=block_tables_stride0,
            scale=scale,
            K_QScale_hip=K_QScale_hip,
            V_QScale_hip=V_QScale_hip,
            K_QScale_asm=K_QScale_asm,
            V_QScale_asm=V_QScale_asm,
            out_=out_,
            kv_cache_dtype=kv_cache_dtype,
        )

    @staticmethod
    def mhc_pre(
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for mHC pre block.

        Args:
            residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
            fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
            hc_scale: shape (3,), dtype torch.float32
            hc_base: shape (hc_mult3,), dtype torch.float32
            rms_eps: RMS normalization epsilon
            hc_pre_eps: pre-mix epsilon
            hc_sinkhorn_eps: sinkhorn epsilon
            hc_post_mult_value: post-mix multiplier value
            sinkhorn_repeat: number of sinkhorn iterations
            n_splits: split-k factor;

        Returns:
            post_mix: shape (..., hc_mult), dtype torch.float32
            comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
            layer_input: shape (..., hidden_size), dtype torch.bfloat16
        """
        from aiter.ops.mhc import mhc_pre

        # Validate shapes
        assert residual.dtype == torch.bfloat16
        assert fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32

        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        hc_mult2 = hc_mult * hc_mult
        hc_mult3 = hc_mult * 2 + hc_mult2

        hc_hidden_size = hc_mult * hidden_size
        assert fn.shape[0] == hc_mult3
        assert fn.shape[1] == hc_hidden_size
        assert hc_scale.shape == (3,)
        assert hc_base.shape == (hc_mult3,)

        outer_shape = residual.shape[:-2]

        residual_flat = residual.view(-1, hc_mult, hidden_size)

        num_tokens = residual_flat.shape[0]
        if num_tokens == 0:
            return (
                torch.empty(
                    num_tokens,
                    hc_mult,
                    1,
                    dtype=torch.float32,
                    device=residual_flat.device,
                ),
                torch.empty(
                    num_tokens,
                    hc_mult,
                    hc_mult,
                    dtype=torch.float32,
                    device=residual_flat.device,
                ),
                torch.empty(
                    num_tokens,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=residual_flat.device,
                ),
            )

        # AITER's Python wrapper allocates intermediate/output tensors without
        # explicit device arguments, so run it under the residual tensor's device.
        with torch.device(residual_flat.device):
            post_mix, comb_mix, layer_input = mhc_pre(
                residual_flat,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    @staticmethod
    def hc_head(
        hs_flat: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        out: torch.Tensor,
        hidden_size: int,
        rms_eps: float,
        hc_eps: float,
        hc_mult: int,
    ) -> None:
        """Run hc_head through AITER mhc_pre and write the result to out."""
        assert hs_flat.dtype == torch.bfloat16
        assert fn.dtype == torch.float32
        assert hc_scale.dtype == torch.float32
        assert hc_base.dtype == torch.float32
        assert hs_flat.shape[-2:] == (hc_mult, hidden_size)
        assert fn.shape == (hc_mult, hc_mult * hidden_size)
        assert hc_scale.shape == (1,)
        assert hc_base.shape == (hc_mult,)

        num_tokens = hs_flat.shape[0]
        if num_tokens == 0:
            return

        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

        full_fn = torch.zeros(
            hc_mult3,
            hc_mult * hidden_size,
            dtype=fn.dtype,
            device=fn.device,
        )
        full_fn[:hc_mult] = fn

        full_base = torch.zeros(hc_mult3, dtype=hc_base.dtype, device=hc_base.device)
        full_base[:hc_mult] = hc_base

        full_scale = torch.zeros(3, dtype=hc_scale.dtype, device=hc_scale.device)
        full_scale[0] = hc_scale[0]

        _, _, layer_input = rocm_aiter_ops.mhc_pre(
            hs_flat,
            full_fn,
            full_scale,
            full_base,
            rms_eps,
            hc_eps,
            0.0,
            1.0,
            0,
        )
        out.copy_(layer_input)

    @staticmethod
    def mhc_post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        from aiter.ops.mhc import mhc_post

        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        residual_flat = residual.view(-1, hc_mult, hidden_size)
        num_tokens = residual_flat.shape[0]
        out = torch.empty_like(residual_flat)
        mhc_post(
            out,
            x.view(num_tokens, hidden_size),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult, 1),
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
        )
        return out.view_as(residual)


rocm_aiter_ops.register_ops_once()
