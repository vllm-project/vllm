# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake

if hasattr(torch.ops._xpu_C, "fp8_gemm"):

    @register_fake("_xpu_C::fp8_gemm")
    def _fp8_gemm_fake(
        q_input: torch.Tensor,
        q_weight: torch.Tensor,
        out_dtype: torch.dtype,
        input_scales: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = q_input.view(-1, q_input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=out_dtype, device=q_input.device)


if hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):

    @register_fake("_xpu_C::fp8_gemm_w8a16")
    def _fp8_gemm_w8a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):

    @register_fake("_xpu_C::int4_gemm_w4a8")
    def _int4_gemm_w4a8_fake(
        input: torch.Tensor,
        input_scales: torch.Tensor,
        input_zero_points: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        group_size: int,
        g_idx: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=torch.float16, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):

    @register_fake("_xpu_C::int4_gemm_w4a16")
    def _int4_gemm_w4a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        bias: torch.Tensor | None,
        weight_scale: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


def _gdn_attention_core_xpu_impl(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op wrapping the XPU SYCL GDN kernel for torch.compile."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    num_actual_tokens = attn_metadata.num_actual_tokens
    num_accepted_tokens = attn_metadata.num_accepted_tokens

    num_prefills = attn_metadata.num_prefills
    num_decodes = attn_metadata.num_decodes
    num_spec_decodes = attn_metadata.num_spec_decodes

    has_initial_state = attn_metadata.has_initial_state

    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
    non_spec_state_indices_tensor = (
        non_spec_state_indices_tensor.contiguous()
        if non_spec_state_indices_tensor is not None
        else None
    )

    spec_query_start_loc = attn_metadata.spec_query_start_loc
    spec_token_indx = attn_metadata.spec_token_indx
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501

    spec_sequence_masks = attn_metadata.spec_sequence_masks
    if spec_sequence_masks is not None:
        if non_spec_token_indx is not None:
            non_spec_token_indx = non_spec_token_indx.to(torch.int32)
        if spec_token_indx is not None:
            spec_token_indx = spec_token_indx.to(torch.int32)

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=num_prefills,  # type: ignore[attr-defined]
        num_decodes=num_decodes,  # type: ignore[attr-defined]
        num_spec_decodes=num_spec_decodes,  # type: ignore[attr-defined]
        has_initial_state=has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_token_indx=non_spec_token_indx,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        spec_query_start_loc=spec_query_start_loc,  # type: ignore[attr-defined]
        spec_token_indx=spec_token_indx,  # type: ignore[attr-defined]
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,  # type: ignore[attr-defined]
        num_actual_tokens=num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )


def _gdn_attention_core_xpu_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _xpu_ops_deepseek_scaling_rope_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    offsets: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert key is not None
    return torch.ops._xpu_C.deepseek_scaling_rope(
        positions, query, key, offsets, cos_sin_cache, rotary_dim, is_neox_style
    )


def _xpu_ops_deepseek_scaling_rope_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    offsets: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _xpu_fp8_bmm_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """XPU FP8 batched GEMM implementation for ``torch.ops.vllm.xpu_fp8_bmm``.

    Computes batched matrix multiplication over the leading group dimension:
    ``[G, M, K] @ [G, K, N] -> [G, M, N]``.

    Args:
        a: FP8 activation tensor with shape ``[G, M, K]``.
            Does not need to be contiguous.
        b: FP8 weight tensor with shape ``[G, K, N]``.
            Does not need to be contiguous.
        out_dtype: Output dtype accepted by the kernel (typically
            ``torch.bfloat16`` for the DeepSeek-V4 O-proj path).
        a_scale: Activation scale tensor for ``a``.
            In current DeepSeek-V4 XPU usage it is block-scaled with shape
            ``[G, M, K/bs]`` (``bs`` is the quant block size, e.g. 128).
            Must be contiguous.
        b_scale: Weight scale tensor for ``b``.
            In current DeepSeek-V4 XPU usage it is block-scaled with shape
            ``[G, K/bs, N/bs]`` (``bs`` is the quant block size, e.g. 128).
            Must be contiguous.
        bias: Optional bias tensor. Pass ``None`` when no bias is required.

    Returns:
        Output tensor with shape ``[G, M, N]`` and dtype ``out_dtype``.

    Notes:
        This implementation centralizes access to
        ``torch.ops._xpu_C.fp8_bmm``. Both scales must be contiguous, while
        ``a`` and ``b`` may be non-contiguous views.
    """
    return torch.ops._xpu_C.fp8_bmm(a, b, out_dtype, a_scale, b_scale, bias)


def _xpu_fp8_bmm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # [G, M, K] @ [G, K, N] => [G, M, N]
    return torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]),
        dtype=out_dtype,
        device=a.device,
    )


def _xpu_fp8_mqa_logits_impl(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._xpu_C.fp8_mqa_logits(
        q,
        k_quant,
        k_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )


def _xpu_fp8_mqa_logits_fake(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (q.shape[0], k_quant.shape[0]),
        dtype=torch.float32,
        device=q.device,
    )


def _xpu_fp8_paged_mqa_logits_impl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    return torch.ops._xpu_C.fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    )


def _xpu_fp8_paged_mqa_logits_fake(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n = q.shape[:2]
    return torch.empty(
        (batch_size * next_n, max_model_len),
        dtype=torch.float32,
        device=q.device,
    )


def _topk_topp_sample_impl(
    random_sampled: torch.Tensor,
    logits_to_return: torch.Tensor | None,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: str,
    seeds: torch.Tensor | None,
    lambda_: float = 1.0,
) -> None:
    torch.ops._xpu_C.topk_topp_sampler(
        random_sampled, logits_to_return, logits, k, p, logprobs_mode, seeds, lambda_
    )
    return


def _topk_topp_sample_fake(
    random_sampled: torch.Tensor,
    logits_to_return: torch.Tensor | None,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: str,
    seeds: torch.Tensor | None,
    lambda_: float = 1.0,
) -> None:
    return


def _xpu_mxfp8_quantize_impl(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP8_BLOCK_SIZE = 32
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    if dtype is not None:
        assert dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
            f"Unsupported dtype for xpu_mxfp8_quantize: {dtype}. "
            f"Expected torch.float8_e4m3fn or torch.float8_e5m2."
        )
    else:
        dtype = current_platform.fp8_dtype()

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    eps = 1e-10

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(
        x,
        x_q,
        x_s,
        MXFP8_BLOCK_SIZE,
        eps,
        fp8_min,
        fp8_max,
        True,
        False,
        False,  # dummy_is_scale_transposed, dummy_is_tma_aligned
    )
    x_s = x_s.to(torch.float8_e8m0fnu)
    return x_q, x_s


def _xpu_mxfp8_quantize_fake(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = current_platform.fp8_dtype()

    MXFP8_BLOCK_SIZE = 32

    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.zeros(shape, device=x.device, dtype=torch.float32)

    return x.to(dtype), x_s.to(torch.float8_e8m0fnu)


def _xpu_mxfp4_quantize_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP4_BLOCK_SIZE = 32
    eps = 1e-10
    assert x.ndim == 2, "input must be 2-D"
    assert x.shape[-1] % MXFP4_BLOCK_SIZE == 0, (
        f"last dimension {x.shape[-1]} must be divisible by group_size "
        f"{MXFP4_BLOCK_SIZE}"
    )
    assert x.is_contiguous(), "input groups must be contiguous"

    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE, eps)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s


def _xpu_mxfp4_quantize_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP4_BLOCK_SIZE = 32
    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, tl.math.log(tl.math.exp(x) + 1.0), x)


@triton.jit
def _selective_scan_fwd_kernel(
    # Pointers to input tensors
    u_ptr,
    delta_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    delta_bias_ptr,
    # Pointers to output tensors (out aliases delta, out_z aliases z)
    out_ptr,
    out_z_ptr,
    # SSM states
    ssm_states_ptr,
    # Optional pointers
    query_start_loc_ptr,
    cache_indices_ptr,
    has_initial_state_ptr,
    # APC pointers
    block_idx_first_ptr,
    block_idx_last_ptr,
    initial_state_idx_ptr,
    cu_chunk_seqlen_ptr,
    last_chunk_indices_ptr,
    # Dimensions
    batch: tl.int32,
    dim: tl.int32,
    seqlen: tl.int32,
    dstate: tl.int32,
    n_groups: tl.int32,
    dim_ngroups_ratio: tl.int32,
    # Strides for u (and out, since out = delta which has same layout)
    u_batch_stride: tl.int64,
    u_d_stride: tl.int64,
    # Strides for delta
    delta_batch_stride: tl.int64,
    delta_d_stride: tl.int64,
    # Strides for A
    A_d_stride: tl.int64,
    A_dstate_stride: tl.int64,
    # Strides for B
    B_batch_stride: tl.int64,
    B_group_stride: tl.int64,
    B_dstate_stride: tl.int64,
    # Strides for C
    C_batch_stride: tl.int64,
    C_group_stride: tl.int64,
    C_dstate_stride: tl.int64,
    # Strides for z
    z_batch_stride: tl.int64,
    z_d_stride: tl.int64,
    # Strides for out
    out_batch_stride: tl.int64,
    out_d_stride: tl.int64,
    # Strides for out_z
    out_z_batch_stride: tl.int64,
    out_z_d_stride: tl.int64,
    # Strides for ssm_states
    ssm_batch_stride: tl.int64,
    ssm_dim_stride: tl.int64,
    ssm_dstate_stride: tl.int64,
    # Cache strides
    cache_indices_stride: tl.int64,
    # Scalar params
    null_block_id: tl.int64,
    block_size: tl.int32,
    # Compile-time constants
    delta_softplus: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DELTA_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_CACHE_INDICES: tl.constexpr,
    CACHE_ENABLED: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)
    group_idx = dim_idx // dim_ngroups_ratio

    # Determine sequence boundaries
    if IS_VARLEN:
        seq_start = tl.load(query_start_loc_ptr + batch_idx).to(tl.int32)
        seq_end = tl.load(query_start_loc_ptr + batch_idx + 1).to(tl.int32)
        actual_seqlen = seq_end - seq_start
    else:
        seq_start = 0
        actual_seqlen = seqlen

    # Determine cache index for ssm_states
    if CACHE_ENABLED:
        init_state_idx = tl.load(initial_state_idx_ptr + batch_idx).to(tl.int32)
        load_cache_slot = tl.load(
            cache_indices_ptr + batch_idx * cache_indices_stride + init_state_idx
        ).to(tl.int64)
        if load_cache_slot == null_block_id:
            return
    elif HAS_CACHE_INDICES:
        cache_index = tl.load(cache_indices_ptr + batch_idx).to(tl.int64)
        if cache_index == null_block_id:
            return
        load_cache_slot = cache_index
    else:
        load_cache_slot = batch_idx.to(tl.int64)

    # Load D value
    D_val = 0.0
    if HAS_D:
        D_val = tl.load(D_ptr + dim_idx).to(tl.float32)

    # Load delta_bias value
    delta_bias_val = 0.0
    if HAS_DELTA_BIAS:
        delta_bias_val = tl.load(delta_bias_ptr + dim_idx).to(tl.float32)

    # Load A values for this dim - shape (dstate,)
    dstate_offs = tl.arange(0, BLOCK_DSTATE)
    dstate_mask = dstate_offs < dstate
    A_vals = tl.load(
        A_ptr + dim_idx * A_d_stride + dstate_offs * A_dstate_stride,
        mask=dstate_mask,
        other=0.0,
    ).to(tl.float32)

    # Initialize state vector
    state = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    # Load initial state if available
    has_init = False
    if has_initial_state_ptr is not None:
        has_init = tl.load(has_initial_state_ptr + batch_idx)
    if has_init:
        state = tl.load(
            ssm_states_ptr
            + load_cache_slot * ssm_batch_stride
            + dim_idx * ssm_dim_stride
            + dstate_offs * ssm_dstate_stride,
            mask=dstate_mask,
            other=0.0,
        ).to(tl.float32)

    # Compute base addresses for u and delta
    if IS_VARLEN:
        u_base = u_ptr + dim_idx * u_d_stride + seq_start * u_batch_stride
        delta_base = (
            delta_ptr + dim_idx * delta_d_stride + seq_start * delta_batch_stride
        )
        out_base = out_ptr + dim_idx * out_d_stride + seq_start * out_batch_stride
        B_base = B_ptr + group_idx * B_group_stride + seq_start * B_batch_stride
        C_base = C_ptr + group_idx * C_group_stride + seq_start * C_batch_stride
    else:
        u_base = u_ptr + batch_idx * u_batch_stride + dim_idx * u_d_stride
        delta_base = (
            delta_ptr + batch_idx * delta_batch_stride + dim_idx * delta_d_stride
        )
        out_base = out_ptr + batch_idx * out_batch_stride + dim_idx * out_d_stride
        B_base = B_ptr + batch_idx * B_batch_stride + group_idx * B_group_stride
        C_base = C_ptr + batch_idx * C_batch_stride + group_idx * C_group_stride

    if HAS_Z:
        if IS_VARLEN:
            z_base = z_ptr + dim_idx * z_d_stride + seq_start * z_batch_stride
            out_z_base = (
                out_z_ptr + dim_idx * out_z_d_stride + seq_start * out_z_batch_stride
            )
        else:
            z_base = z_ptr + batch_idx * z_batch_stride + dim_idx * z_d_stride
            out_z_base = (
                out_z_ptr + batch_idx * out_z_batch_stride + dim_idx * out_z_d_stride
            )

    # Determine chunk boundaries for APC mode
    if CACHE_ENABLED:
        last_chunk_idx = tl.load(last_chunk_indices_ptr + batch_idx).to(tl.int32)
        if batch_idx == 0:
            first_chunk_idx = 0
        else:
            first_chunk_idx = (
                tl.load(last_chunk_indices_ptr + batch_idx - 1).to(tl.int32) + 1
            )
        n_chunks = last_chunk_idx - first_chunk_idx + 1
        first_chunk_tokens = tl.load(cu_chunk_seqlen_ptr + first_chunk_idx + 1).to(
            tl.int32
        ) - tl.load(cu_chunk_seqlen_ptr + first_chunk_idx).to(tl.int32)
        block_idx_first = tl.load(block_idx_first_ptr + batch_idx).to(tl.int32)
        chunk_start_offset = 0
        if n_chunks > 1 and first_chunk_tokens < block_size:
            chunk_start_offset = block_size - first_chunk_tokens
        current_position = block_idx_first * block_size + chunk_start_offset
    else:
        n_chunks = 1
        first_chunk_idx = 0

    # Sequential scan over the sequence
    tokens_processed = 0
    for chunk in range(0, n_chunks if CACHE_ENABLED else 1):
        if CACHE_ENABLED:
            chunk_tokens = tl.load(
                cu_chunk_seqlen_ptr + first_chunk_idx + chunk + 1
            ).to(tl.int32) - tl.load(cu_chunk_seqlen_ptr + first_chunk_idx + chunk).to(
                tl.int32
            )
        else:
            chunk_tokens = actual_seqlen

        for local_pos in range(chunk_tokens):
            pos = tokens_processed + local_pos
            # Load u value
            u_val = tl.load(u_base + pos).to(tl.float32)

            # Load delta value
            delta_val = tl.load(delta_base + pos).to(tl.float32)

            # Apply delta bias
            if HAS_DELTA_BIAS:
                delta_val = delta_val + delta_bias_val

            # Apply softplus
            if delta_softplus:
                delta_val = _softplus(delta_val)

            delta_u = delta_val * u_val

            # Compute dA = exp(delta * A) for all dstate elements
            dA = tl.exp(delta_val * A_vals)

            # Load B values for this position
            B_vals = tl.load(
                B_base + dstate_offs * B_dstate_stride + pos,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

            # Load C values for this position
            C_vals = tl.load(
                C_base + dstate_offs * C_dstate_stride + pos,
                mask=dstate_mask,
                other=0.0,
            ).to(tl.float32)

            # Update state: state = dA * state + delta * u * B
            state = dA * state + delta_u * B_vals

            # Compute output: out = sum(state * C) + D * u
            out_val = tl.sum(state * C_vals, axis=0)
            if HAS_D:
                out_val = out_val + D_val * u_val

            # Store output
            tl.store(out_base + pos, out_val.to(out_ptr.dtype.element_ty))

            if HAS_Z:
                z_val = tl.load(z_base + pos).to(tl.float32)
                out_z_val = out_val * z_val / (1.0 + tl.exp(-z_val))
                tl.store(
                    out_z_base + pos,
                    out_z_val.to(out_z_ptr.dtype.element_ty),
                )

        tokens_processed += chunk_tokens

        # Store intermediate state for APC mode
        if CACHE_ENABLED:
            if chunk == n_chunks - 1:
                store_slot = tl.load(
                    cache_indices_ptr
                    + batch_idx * cache_indices_stride
                    + tl.load(block_idx_last_ptr + batch_idx).to(tl.int32)
                ).to(tl.int64)
            else:
                block_idx_done = (current_position + chunk_tokens - 1) // block_size
                store_slot = tl.load(
                    cache_indices_ptr
                    + batch_idx * cache_indices_stride
                    + block_idx_done
                ).to(tl.int64)

            tl.store(
                ssm_states_ptr
                + store_slot * ssm_batch_stride
                + dim_idx * ssm_dim_stride
                + dstate_offs * ssm_dstate_stride,
                state.to(ssm_states_ptr.dtype.element_ty),
                mask=dstate_mask,
            )
            current_position += chunk_tokens

    # Store final state for non-APC mode
    if not CACHE_ENABLED:
        tl.store(
            ssm_states_ptr
            + load_cache_slot * ssm_batch_stride
            + dim_idx * ssm_dim_stride
            + dstate_offs * ssm_dstate_stride,
            state.to(ssm_states_ptr.dtype.element_ty),
            mask=dstate_mask,
        )


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class xpu_ops:
    @staticmethod
    @torch.compile
    def dynamic_per_token_int8_quant_ref(
        input: torch.Tensor, use_sym_quant: bool, bits: int
    ):
        original_sizes = input.size()
        # view is not safe in torch.compile if input is not contiguous
        input = input.reshape(
            -1, original_sizes[-1]
        )  # Flatten except for the last dimension
        qmin = -(2 ** (bits - 1)) if use_sym_quant else 0
        qmax = 2 ** (bits - 1) - 1 if use_sym_quant else 2**bits - 1
        min_val = torch.min(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
        max_val = torch.max(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
        if use_sym_quant:
            scale = (
                torch.maximum(torch.abs(min_val), torch.abs(max_val)) / qmax
            ).clamp(min=1e-5)
            zero_point = torch.zeros_like(scale).to(dtype=torch.int32)
        else:
            scale = ((max_val - min_val) / qmax).clamp(min=1e-5)
            zero_point = -1 * torch.round(min_val / scale).to(dtype=torch.int32)
        scale = scale.to(dtype=input.dtype)
        quantized = torch.clamp(
            torch.round(input / scale.to(dtype=torch.float32) + zero_point),
            qmin,
            qmax,
        ).to(dtype=torch.int8 if use_sym_quant else torch.uint8)
        return (
            quantized.view(original_sizes),
            scale.view(original_sizes[:-1] + (1,)),
            zero_point.view(original_sizes[:-1] + (1,)),
        )

    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
        window_size: list[int] | None = None,
        softcap: float | None = 0.0,
        seqused_k: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        # passed in qwen vl
        dropout_p: float = 0.0,
        # The following parameters are not used in xpu kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        return_softmax_lse: bool | None = False,
        s_aux: torch.Tensor | None = None,
        return_attn_probs: bool | None = False,
        dynamic_causal: torch.Tensor | None = None,
        mask_mod: Callable | None = None,
        aux_tensors: list | None = None,
        **kwargs,
    ):
        assert cu_seqlens_k is not None or seqused_k is not None, (
            "cu_seqlens_k or seqused_k must be provided"
        )
        assert cu_seqlens_k is None or seqused_k is None, (
            "cu_seqlens_k and seqused_k cannot be provided at the same time"
        )
        assert block_table is None or seqused_k is not None, (
            "when enable block_table, seqused_k is needed"
        )
        assert block_table is not None or cu_seqlens_k is not None, (
            "when block_table is disabled, cu_seqlens_k is needed"
        )
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])  # noqa: F841

        return flash_attn_varlen_func(
            out=out,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    @staticmethod
    def get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        cache_leftpad: torch.Tensor | None = None,
        page_size: int | None = None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        has_softcap=False,
        num_splits=0,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
        sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for xpu_ops, returning None."
        )
        return None

    @staticmethod
    def selective_scan_fwd(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D_: torch.Tensor | None,
        z_: torch.Tensor | None,
        delta_bias_: torch.Tensor | None,
        delta_softplus: bool,
        query_start_loc: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        ssm_states: torch.Tensor,
        null_block_id: int,
        block_size: int = 1024,
        block_idx_first_scheduled_token: torch.Tensor | None = None,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        cu_chunk_seqlen: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
    ) -> None:
        varlen = query_start_loc is not None
        batch_size = (
            (query_start_loc.shape[0] - 1)
            if query_start_loc is not None
            else u.shape[0]
        )
        dim = u.shape[0] if varlen else u.shape[1]
        total_seqlen = u.shape[1] if varlen else u.shape[2]
        dstate = A.size(1)
        n_groups = B.size(0) if varlen else B.size(1)
        dim_ngroups_ratio = dim // n_groups

        has_z = z_ is not None
        has_D = D_ is not None
        has_delta_bias = delta_bias_ is not None
        has_cache_indices = cache_indices is not None
        cache_enabled = block_idx_first_scheduled_token is not None

        # out and out_z alias delta and z respectively
        out = delta
        out_z = z_ if z_ is not None else delta  # won't be used if not has_z

        BLOCK_DSTATE = triton.next_power_of_2(dstate)

        # Compute strides
        if varlen:
            u_batch_stride = u.stride(1)
            u_d_stride = u.stride(0)
            delta_batch_stride = delta.stride(1)
            delta_d_stride = delta.stride(0)
            B_batch_stride = B.stride(2)
            B_group_stride = B.stride(0)
            B_dstate_stride = B.stride(1)
            C_batch_stride = C.stride(2)
            C_group_stride = C.stride(0)
            C_dstate_stride = C.stride(1)
            out_batch_stride = out.stride(1)
            out_d_stride = out.stride(0)
            if z_ is not None:
                z_batch_stride = z_.stride(1)
                z_d_stride = z_.stride(0)
                out_z_batch_stride = out_z.stride(1)
                out_z_d_stride = out_z.stride(0)
            else:
                z_batch_stride = 0
                z_d_stride = 0
                out_z_batch_stride = 0
                out_z_d_stride = 0
        else:
            u_batch_stride = u.stride(0)
            u_d_stride = u.stride(1)
            delta_batch_stride = delta.stride(0)
            delta_d_stride = delta.stride(1)
            B_batch_stride = B.stride(0)
            B_group_stride = B.stride(1)
            B_dstate_stride = B.stride(2)
            C_batch_stride = C.stride(0)
            C_group_stride = C.stride(1)
            C_dstate_stride = C.stride(2)
            out_batch_stride = out.stride(0)
            out_d_stride = out.stride(1)
            if z_ is not None:
                z_batch_stride = z_.stride(0)
                z_d_stride = z_.stride(1)
                out_z_batch_stride = out_z.stride(0)
                out_z_d_stride = out_z.stride(1)
            else:
                z_batch_stride = 0
                z_d_stride = 0
                out_z_batch_stride = 0
                out_z_d_stride = 0

        ssm_batch_stride = ssm_states.stride(0)
        ssm_dim_stride = ssm_states.stride(1)
        ssm_dstate_stride = ssm_states.stride(2)
        cache_indices_stride = (
            cache_indices.stride(0) if cache_indices is not None else 0
        )

        grid = (batch_size, dim)
        _selective_scan_fwd_kernel[grid](
            u,
            delta,
            A,
            B,
            C,
            D_ if has_D else u,  # dummy, won't be dereferenced
            z_ if has_z else u,  # dummy
            delta_bias_ if has_delta_bias else u,  # dummy
            out,
            out_z,
            ssm_states,
            query_start_loc if varlen else u,  # dummy
            cache_indices if has_cache_indices else u,  # dummy
            has_initial_state,
            # APC pointers
            block_idx_first_scheduled_token if cache_enabled else u,
            block_idx_last_scheduled_token if cache_enabled else u,
            initial_state_idx if cache_enabled else u,
            cu_chunk_seqlen if cache_enabled else u,
            last_chunk_indices if cache_enabled else u,
            # Dimensions
            batch_size,
            dim,
            total_seqlen,
            dstate,
            n_groups,
            dim_ngroups_ratio,
            # Strides
            u_batch_stride,
            u_d_stride,
            delta_batch_stride,
            delta_d_stride,
            A.stride(0),
            A.stride(1),
            B_batch_stride,
            B_group_stride,
            B_dstate_stride,
            C_batch_stride,
            C_group_stride,
            C_dstate_stride,
            z_batch_stride,
            z_d_stride,
            out_batch_stride,
            out_d_stride,
            out_z_batch_stride,
            out_z_d_stride,
            ssm_batch_stride,
            ssm_dim_stride,
            ssm_dstate_stride,
            cache_indices_stride,
            null_block_id,
            block_size,
            # Compile-time constants
            delta_softplus=delta_softplus,
            HAS_D=has_D,
            HAS_Z=has_z,
            HAS_DELTA_BIAS=has_delta_bias,
            IS_VARLEN=varlen,
            HAS_CACHE_INDICES=has_cache_indices,
            CACHE_ENABLED=cache_enabled,
            BLOCK_DSTATE=BLOCK_DSTATE,
        )

    @staticmethod
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            # register all the custom ops here
            direct_register_custom_op(
                op_name="xpu_ops_deepseek_scaling_rope",
                op_func=_xpu_ops_deepseek_scaling_rope_impl,
                mutates_args=[],
                fake_impl=_xpu_ops_deepseek_scaling_rope_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="xpu_mxfp8_quantize",
                op_func=_xpu_mxfp8_quantize_impl,
                fake_impl=_xpu_mxfp8_quantize_fake,
            )

            direct_register_custom_op(
                op_name="xpu_mxfp4_quantize",
                op_func=_xpu_mxfp4_quantize_impl,
                fake_impl=_xpu_mxfp4_quantize_fake,
            )

            direct_register_custom_op(
                op_name="xpu_fp8_bmm",
                op_func=_xpu_fp8_bmm_impl,
                fake_impl=_xpu_fp8_bmm_fake,
            )

            direct_register_custom_op(
                op_name="xpu_fp8_mqa_logits",
                op_func=_xpu_fp8_mqa_logits_impl,
                fake_impl=_xpu_fp8_mqa_logits_fake,
            )

            direct_register_custom_op(
                op_name="xpu_fp8_paged_mqa_logits",
                op_func=_xpu_fp8_paged_mqa_logits_impl,
                fake_impl=_xpu_fp8_paged_mqa_logits_fake,
            )

            direct_register_custom_op(
                op_name="gdn_attention_core_xpu",
                op_func=_gdn_attention_core_xpu_impl,
                mutates_args=["core_attn_out", "z"],
                fake_impl=_gdn_attention_core_xpu_fake,
            )

            direct_register_custom_op(
                op_name="xpu_topk_topp_sampler",
                op_func=_topk_topp_sample_impl,
                fake_impl=_topk_topp_sample_fake,
            )

            _OPS_REGISTERED = True


xpu_ops.register_ops_once()
