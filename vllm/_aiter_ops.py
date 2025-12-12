# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
from collections.abc import Callable

import torch

import vllm.envs as envs
from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

_FP8_DTYPE = current_platform.fp8_dtype()


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


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    tile_offset = (
        offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
        + offset % HEAD_TILE_SIZE
    )
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    dst_ptr = (
        kv_cache_ptr
        + block_id * kv_cache_value_stride
        + tile_block_id * BLOCK_TILE_SIZE * head_dim
        + tile_block_offset * HEAD_TILE_SIZE
    )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    kv_cache_value = kv_cache[:, : block_size * head_dim]
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    src_cache_offset = (
        block_id * kv_cache_stride
        + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
        + tiled_block_offset * HEAD_TILE_SIZE
    )
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    tiled_src_offset = (
        offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
        + offset % HEAD_TILE_SIZE
    )
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        head_dim,
        block_tile_size,
        head_tile_size,
    )


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


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    kv_cache, scale = kv_cache[..., :dim], kv_cache[..., dim:]
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    if rocm_aiter_ops.is_enabled():
        batch_size, next_n, heads, head_dim = q_fp8.shape
        num_blocks, block_size, _, _ = kv_cache_fp8.shape

        from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

        out_logits = torch.full(
            [batch_size * next_n, max_model_len],
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        deepgemm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_logits,
            context_lens,
            block_tables,
            max_model_len,
            ChunkK=256,
            Preshuffle=block_size == 64,
            KVBlockSize=block_size,
            WavePerEU=2,
        )
        return out_logits
    else:
        return fp8_paged_mqa_logits_torch(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv, scale = kv
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    @functools.lru_cache
    def has_mqa_logits_module():
        return importlib.util.find_spec("aiter.ops.triton.fp8_mqa_logits") is not None

    if rocm_aiter_ops.is_enabled() and has_mqa_logits_module():
        from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

        kv, scale = kv
        return fp8_mqa_logits(q, kv, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    fp8_dtype = current_platform.fp8_dtype()
    _k_fp8 = _flattened_kv[..., :head_dim].view(fp8_dtype).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    indexer_k_quant_and_cache_triton(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=k.device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=k.device,
                dtype=torch.uint8,
            )
            cp_gather_indexer_k_quant_cache_triton(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
                chunk.token_to_seq,
            )

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]
            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


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
                mutates_args=["scale"],
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
    def triton_gemm_a8w8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale

        return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)

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
