# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

from vllm.logger import init_logger
from vllm.platforms import current_platform
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

    # TODO: xpu does not support speculative decoding yet
    assert attn_metadata.spec_sequence_masks is None  # type: ignore[attr-defined]

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
        num_prefills=attn_metadata.num_prefills,  # type: ignore[attr-defined]
        num_decodes=attn_metadata.num_decodes,  # type: ignore[attr-defined]
        has_initial_state=attn_metadata.has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=attn_metadata.non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=attn_metadata.non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        num_actual_tokens=attn_metadata.num_actual_tokens,  # type: ignore[attr-defined]
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
        x, x_q, x_s, MXFP8_BLOCK_SIZE, eps, fp8_min, fp8_max, True
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
    def indexer_k_quant_and_cache(
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str | None,
    ) -> None:
        head_dim = k.shape[-1]
        k = k.view(-1, head_dim)  # [total_tokens, head_dim]

        def group_quant_torch(
            x: torch.Tensor,
            group_size: int,
            eps: float = 1e-10,
            dtype: torch.dtype | None = None,
            column_major_scales: bool = False,
            out_q: torch.Tensor | None = None,
            use_ue8m0: bool | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if use_ue8m0 is None:
                # Default fallback - could import is_deep_gemm_e8m0_used if needed
                use_ue8m0 = False

            if dtype is None:
                dtype = current_platform.fp8_dtype()

            # Validate inputs
            assert x.shape[-1] % group_size == 0, (
                f"Last dimension {x.shape[-1]} must be divisible by "
                f"group_size {group_size}"
            )
            assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

            # Prepare output tensor
            if out_q is None:
                x_q = torch.empty_like(x, dtype=dtype)
            else:
                assert out_q.shape == x.shape
                x_q = out_q

            # Reshape input for group processing
            # Original shape: (..., last_dim)
            # Target shape: (..., num_groups, group_size)
            original_shape = x.shape
            num_groups = original_shape[-1] // group_size

            # Reshape to separate groups
            group_shape = original_shape[:-1] + (num_groups, group_size)
            x_grouped = x.view(group_shape)

            # Compute per-group absolute maximum values
            # Shape: (..., num_groups)
            abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
            abs_max = torch.maximum(
                abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype)
            )

            # Compute scales
            FP8_MAX = torch.finfo(dtype).max
            FP8_MIN = torch.finfo(dtype).min
            scale_raw = abs_max / FP8_MAX

            if use_ue8m0:
                # For UE8M0 format, scales must be powers of 2
                scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
            else:
                scales = scale_raw

            # Expand scales for broadcasting with grouped data
            # Shape: (..., num_groups, 1)
            scales_expanded = scales.unsqueeze(-1)

            # Quantize the grouped data
            x_scaled = x_grouped / scales_expanded
            x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)
            x_quantized = x_clamped.to(dtype)

            # Reshape back to original shape
            x_q.copy_(x_quantized.view(original_shape))

            # Prepare scales tensor in requested format
            if column_major_scales:
                # Column-major: (num_groups,) + batch_dims
                # Transpose the scales to put group dimension first
                scales_shape = (num_groups,) + original_shape[:-1]
                x_s = scales.permute(-1, *range(len(original_shape) - 1))
                x_s = x_s.contiguous().view(scales_shape)
            else:
                # Row-major: batch_dims + (num_groups,)
                x_s = scales.contiguous()

            # Ensure scales are float32
            return x_q, x_s.float()

        k_fp8, k_scale = group_quant_torch(
            k,
            group_size=quant_block_size,
            column_major_scales=False,
            use_ue8m0=(scale_fmt == "ue8m0"),
        )

        k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
        scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
        k = torch.cat(
            [k_fp8_bytes, scale_bytes], dim=-1
        )  # [total_tokens, head_dim + 4]

        slot_mapping = slot_mapping.flatten()
        # kv_cache: [num_block, block_size, head_dim + 4]
        kv_cache.view(-1, kv_cache.shape[-1]).index_copy_(0, slot_mapping, k)

        if not hasattr(xpu_ops, '_insert_debug_count'):
            xpu_ops._insert_debug_count = 0
        xpu_ops._insert_debug_count += 1
        if xpu_ops._insert_debug_count <= 3:
            print(f"[XPU indexer_k_quant_and_cache] tokens={k_fp8.shape[0]} "
                  f"head_dim={head_dim} quant_block_size={quant_block_size} "
                  f"scale_range=[{k_scale.min().item():.6f}, {k_scale.max().item():.6f}] "
                  f"scale_fmt={scale_fmt} "
                  f"kv_cache_shape={kv_cache.shape} slot_mapping[:5]={slot_mapping[:5].tolist()}")

    @staticmethod
    def cp_gather_indexer_k_quant_cache(
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> None:
        """
        Gather K and scale from quantized indexer cache.

        Cache layout is BLOCK-PACKED (same as CUDA):
          kv_cache: [num_blocks, block_size, head_dim + scale_per_token] uint8
          But physically, the Triton compressor writes:
            FP8 data at: block_start + pos * head_dim
            Scale at: block_start + block_size * head_dim + pos * 4
          This is equivalent to viewing the block as:
            k_region: cache_flat[block_start : block_start + bs*dim]
            scale_region: cache_flat[block_start + bs*dim : block_start + bs*(dim+4)]
        """
        batch_size = block_table.size(0)
        num_tokens = dst_k.size(0)
        head_dim = dst_k.size(1)
        cache_block_size = kv_cache.size(1)
        scale_per_token = dst_scale.size(1)  # = 4
        block_stride = kv_cache.stride(0)  # bytes per block (may include padding)
        head_dim = dst_k.size(1)  # = TOKEN_STRIDE used by Triton compressor
        cache_block_size = kv_cache.size(1)
        scale_per_token = dst_scale.size(1)  # = 4

        # The cache uses BLOCK-PACKED layout where Triton kernels write with
        # raw pointer math:
        #   K for pos j: block_start + j * head_dim
        #   Scale for pos j: block_start + block_size * head_dim + j * scale_per_token
        # block_start = physical_block_idx * block_stride
        # The tensor may have inter-block padding (stride(0) > block_size * width).
        # Use as_strided to create a flat byte view of the full storage range.
        num_blocks = kv_cache.size(0)
        total_bytes = num_blocks * block_stride
        cache_flat = torch.as_strided(kv_cache, (total_bytes,), (1,))

        # For each token index [0, num_tokens), find batch and in-seq position
        token_indices = torch.arange(num_tokens, device=dst_k.device)
        batch_indices = (
            torch.searchsorted(cu_seq_lens, token_indices, right=True) - 1
        )
        batch_indices = torch.clamp(batch_indices, 0, batch_size - 1)

        # In-batch sequence position (0-based)
        inbatch_seq_pos = token_indices - cu_seq_lens[batch_indices]

        # Block index in table and in-block offset
        block_indices_in_table = inbatch_seq_pos // cache_block_size
        inblock_offsets = inbatch_seq_pos % cache_block_size

        # Physical block indices
        physical_block_indices = block_table[
            batch_indices, block_indices_in_table
        ]

        # Compute byte offsets in block-packed layout
        block_starts = physical_block_indices.to(torch.int64) * block_stride
        # FP8 K at: block_start + pos * head_dim
        k_offsets = block_starts + inblock_offsets.to(torch.int64) * head_dim
        # Scale at: block_start + block_size * head_dim + pos * 4
        scale_offsets = (
            block_starts
            + cache_block_size * head_dim
            + inblock_offsets.to(torch.int64) * scale_per_token
        )

        # Gather K bytes [num_tokens, head_dim]
        k_byte_indices = k_offsets[:, None] + torch.arange(
            head_dim, device=dst_k.device
        )[None, :]
        k_bytes = cache_flat[k_byte_indices.flatten()].view(num_tokens, head_dim)
        dst_k_uint8 = dst_k.view(torch.uint8).view(-1, head_dim)
        dst_k_uint8[:] = k_bytes

        # Gather scale bytes [num_tokens, 4]
        scale_byte_indices = scale_offsets[:, None] + torch.arange(
            scale_per_token, device=dst_k.device
        )[None, :]
        scale_bytes = cache_flat[scale_byte_indices.flatten()].view(
            num_tokens, scale_per_token
        )
        dst_scale[:] = scale_bytes

    @staticmethod
    def fp8_mqa_logits(
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FP8 MQA logits (non-paged, for prefill indexer).

        Pure PyTorch implementation for XPU. Adapted from ROCm fallback.

        Args:
            q: [M, H, D] fp8 query
            kv: (k_fp8 [N, D] fp8, k_scales [N] float32)
            weights: [M, H] float32 routing weights
            cu_seqlen_ks: [M] int32 start indices (inclusive)
            cu_seqlen_ke: [M] int32 end indices (exclusive)

        Returns:
            logits: [M, N] float32
        """
        k_fp8, scale = kv
        seq_len_kv = k_fp8.shape[0]
        k = k_fp8.to(torch.bfloat16)
        q_bf16 = q.to(torch.bfloat16)
        device = q.device

        mask_lo = (
            torch.arange(seq_len_kv, device=device)[None, :]
            >= cu_seqlen_ks[:, None]
        )
        mask_hi = (
            torch.arange(seq_len_kv, device=device)[None, :]
            < cu_seqlen_ke[:, None]
        )
        mask = mask_lo & mask_hi

        # score: [H, M, N] = einsum("mhd,nd->hmn")
        score = torch.einsum("mhd,nd->hmn", q_bf16, k).float() * scale
        # weights: [M, H] -> [H, M, 1]
        logits = (
            score.relu() * weights.unsqueeze(-1).transpose(0, 1)
        ).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))

        return logits

    @staticmethod
    def fp8_paged_mqa_logits(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_metadata: torch.Tensor | None,
        max_model_len: int,
    ) -> torch.Tensor:
        """Compute FP8 paged MQA logits (for decode indexer).

        Pure PyTorch, handles BLOCK-PACKED cache layout:
          kv_cache: [num_blocks, block_size, head_dim + 4] uint8
          But physically, within each block:
            FP8 K at: block_start + pos * head_dim
            Scale at: block_start + block_size * head_dim + pos * 4

        Args:
            q: [B, next_n, H, D] or [B*next_n, H, D] bf16/fp8 query
            kv_cache: [num_blocks, block_size, head_dim + 4] uint8 (may be 4D)
            weights: [B*next_n, H] float32
            seq_lens: [B, next_n] or [B] int32 context lengths
            block_table: [B, max_blocks_per_seq] int32
            schedule_metadata: unused on XPU
            max_model_len: int

        Returns:
            logits: [B*next_n, max_model_len] float32
        """
        from vllm.utils.math_utils import cdiv

        fp8_dtype = current_platform.fp8_dtype()
        # kv_cache may be 4D [num_blocks, block_size, 1, head_width] after unsqueeze
        if kv_cache.ndim == 4:
            kv_cache = kv_cache.squeeze(2)
        block_size = kv_cache.shape[1]
        dim = kv_cache.shape[2] - 4  # head_dim
        block_stride = kv_cache.stride(0)  # bytes per block

        # Handle q shape
        if q.ndim == 4:
            batch_size, next_n = q.shape[0], q.shape[1]
        elif q.ndim == 3:
            batch_size = q.shape[0]
            next_n = 1
            q = q.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected q shape: {q.shape}")

        # Handle seq_lens
        if seq_lens.ndim == 2:
            context_lens = seq_lens[:, -1]  # [B]
        else:
            context_lens = seq_lens  # [B]

        device = q.device
        logits = torch.full(
            [batch_size * next_n, max_model_len],
            float("-inf"),
            device=device,
            dtype=torch.float32,
        )

        # Flatten cache for raw byte access (block-packed layout)
        num_cache_blocks = kv_cache.shape[0]
        total_bytes = num_cache_blocks * block_stride
        cache_flat = torch.as_strided(kv_cache, (total_bytes,), (1,))

        for i in range(batch_size):
            seq_len = int(context_lens[i].item())
            if seq_len <= 0:
                continue
            num_pages = cdiv(seq_len, block_size)
            pages = block_table[i, :num_pages]
            padded_seq_len = num_pages * block_size

            # Gather K and scale from block-packed layout
            # For each page, K region: [page_start, page_start + bs*dim)
            # Scale region: [page_start + bs*dim, page_start + bs*dim + bs*4)
            page_starts = pages.to(torch.int64) * block_stride

            # K indices: for each page p, for each pos j: page_start + j*dim + d
            pos_offsets = torch.arange(block_size, device=device)
            dim_offsets = torch.arange(dim, device=device)
            # [num_pages, block_size, dim]
            k_indices = (
                page_starts[:, None, None]
                + pos_offsets[None, :, None] * dim
                + dim_offsets[None, None, :]
            )
            k_uint8 = cache_flat[k_indices.reshape(-1)].reshape(
                padded_seq_len, dim
            )
            k_fp8 = k_uint8.view(fp8_dtype).to(torch.float32)

            # Scale indices: for each page p, for each pos j:
            #   page_start + bs*dim + j*4 + byte
            scale_byte_offsets = torch.arange(4, device=device)
            scale_indices = (
                page_starts[:, None, None]
                + block_size * dim
                + pos_offsets[None, :, None] * 4
                + scale_byte_offsets[None, None, :]
            )
            scale_uint8 = cache_flat[scale_indices.reshape(-1)].reshape(
                padded_seq_len, 4
            )
            k_scale = scale_uint8.view(torch.float32)  # [padded_seq_len, 1]

            for n in range(next_n):
                q_i = q[i, n].to(torch.float32)  # [H, D]
                w_i = weights[i * next_n + n]  # [H]

                # Compute per-head scores: [H, padded_seq_len]
                scores = torch.mm(q_i, k_fp8[:padded_seq_len].T)  # [H, S]
                scores = scores * k_scale[:padded_seq_len].T  # broadcast scale
                scores = torch.relu(scores)
                # Weight and sum over heads
                weighted = (scores * w_i[:, None]).sum(dim=0)  # [S]
                logits[i * next_n + n, :seq_len] = weighted[:seq_len]

        return logits

    @staticmethod
    def top_k_per_row_prefill(
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        strdide1: int,
        topk_tokens: int,
    ) -> torch.Tensor:
        real_topk = min(topk_tokens, logits.shape[-1])
        topk_indices = logits.topk(real_topk, dim=-1)[1].to(torch.int32)
        topk_indices -= cu_seqlen_ks[:, None]
        mask_lo = topk_indices >= 0
        mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
        mask = torch.full_like(
            topk_indices, False, dtype=torch.bool, device=topk_indices.device
        )
        mask = mask_lo & mask_hi
        topk_indices.masked_fill_(~mask, -1)
        raw_topk_indices[: topk_indices.shape[0], : topk_indices.shape[1]] = (
            topk_indices
        )

    @staticmethod
    def top_k_per_row_decode(
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> torch.Tensor:
        device = logits.device
        batch_size = seq_lens.size(0)
        # padded query len
        padded_num_tokens = batch_size * next_n
        positions = (
            torch.arange(logits.shape[-1], device=device)
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )
        row_indices = torch.arange(padded_num_tokens, device=device) // next_n
        next_n_offset = torch.arange(padded_num_tokens, device=device) % next_n

        # seq_lens can be 1D [B] or 2D [B, next_n]. Normalize to per-token
        # context lengths.
        if seq_lens.ndim == 2:
            # 2D: each row has per-spec-token context lengths
            # Gather the correct context_len for each (batch, spec_pos) pair
            seq_lens_flat = seq_lens.reshape(-1)  # [B * next_n]
            # Each padded token i maps to seq_lens[i // next_n, i % next_n]
            per_token_lens = seq_lens_flat[
                row_indices * next_n + next_n_offset
            ]  # [padded_num_tokens]
            index_end_pos = (per_token_lens - 1).unsqueeze(1)
        else:
            index_end_pos = (
                seq_lens[row_indices] - next_n + next_n_offset
            ).unsqueeze(1)
        # index_end_pos: [B * N, 1]
        mask = positions <= index_end_pos
        # mask: [B * N, L]
        logits = logits.masked_fill(~mask, float("-inf"))
        real_topk = min(topk_tokens, logits.shape[-1])
        topk_indices = logits.topk(real_topk, dim=-1)[1].to(torch.int32)  # [B * N, K]
        # ensure we don't set indices for the top k
        # that is out of range(masked already)
        # this will happen if context length is shorter than K
        topk_indices[topk_indices > index_end_pos] = -1
        raw_topk_indices[: topk_indices.shape[0], : topk_indices.shape[1]] = (
            topk_indices
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
