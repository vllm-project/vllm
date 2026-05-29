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
        x, x_q, x_s, MXFP8_BLOCK_SIZE, eps, fp8_min, fp8_max, True,
        False, False
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
