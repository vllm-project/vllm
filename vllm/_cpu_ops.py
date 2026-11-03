# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrappers for custom ops implemented only by the CPU backend
(csrc/cpu). Ops shared with other backends stay in vllm._custom_ops."""

from enum import IntEnum
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform

current_platform.import_kernels()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    torch.ops._C.mla_decode_kvcache(out, query, kv_cache, scale, block_tables, seq_lens)


def gather_block_table_cpu(
    block_table: torch.Tensor,
    output: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_blocks: torch.Tensor,
    num_reqs_padded: int,
) -> None:
    torch.ops._C.gather_block_table_cpu(
        block_table,
        output,
        idx_mapping,
        num_blocks,
        num_reqs_padded,
    )


def compute_slot_mapping_cpu(
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    mapping_enabled: bool,
    pad_slot_id: int,
) -> None:
    torch.ops._C.compute_slot_mapping_cpu(
        block_table,
        slot_mapping,
        idx_mapping,
        query_start_loc,
        positions,
        block_size,
        mapping_enabled,
        pad_slot_id,
    )


def prepare_prefill_inputs_cpu(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    torch.ops._C.prepare_prefill_inputs_cpu(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )


def get_prompt_logprobs_token_ids_cpu(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.get_prompt_logprobs_token_ids_cpu(
        num_tokens,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
    )


def prepare_pos_seq_lens_cpu(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    torch.ops._C.prepare_pos_seq_lens_cpu(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        pos,
        seq_lens,
    )


def prepare_mrope_positions_cpu(
    positions: torch.Tensor,
    prefill_positions: torch.Tensor,
    prefill_delta: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    torch.ops._C.prepare_mrope_positions_cpu(
        positions,
        prefill_positions,
        prefill_delta,
        idx_mapping,
        query_start_loc,
        prefill_lens,
        num_computed_tokens,
    )


def combine_sampled_and_draft_tokens_cpu(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
    num_new_sampled_tokens: int,
) -> torch.Tensor:
    return torch.ops._C.combine_sampled_and_draft_tokens_cpu(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
        num_new_sampled_tokens,
    )


def post_update_cpu(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    torch.ops._C.post_update_cpu(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )


def gumbel_sample_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    apply_temperature: bool,
    is_drafting: bool,
    use_fp64: bool,
) -> torch.Tensor:
    return torch.ops._C.gumbel_sample_cpu(
        logits,
        expanded_idx_mapping,
        temperature,
        seeds,
        positions,
        apply_temperature,
        is_drafting,
        use_fp64,
    )


def apply_logit_bias_cpu(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    positions: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    restore_when_all_masked: torch.Tensor,
    stop_token_ids: torch.Tensor,
    check_all_masked_rows: bool,
) -> None:
    torch.ops._C.apply_logit_bias_cpu(
        logits,
        expanded_idx_mapping,
        positions,
        num_allowed_token_ids,
        allowed_token_ids,
        num_logit_bias,
        logit_bias_token_ids,
        logit_bias,
        min_lens,
        num_stop_token_ids,
        restore_when_all_masked,
        stop_token_ids,
        check_all_masked_rows,
    )


def causal_conv1d_update_cpu_vec(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = 0,
) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_update_cpu_vec(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        query_start_loc,
        pad_slot_id,
    )


def selective_state_update_cpu(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    state_batch_indices: torch.Tensor | None,
    dst_state_batch_indices: torch.Tensor | None,
    null_block_id: int,
    out: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
):
    torch.ops._C.selective_state_update_cpu(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        dst_state_batch_indices,
        null_block_id,
        out,
        num_accepted_tokens,
        cu_seqlens,
    )


def mamba_chunk_scan_fwd_cpu(
    out: torch.Tensor,
    final_states: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
) -> None:
    """Prefill SSM scan kernel. out and final_states are written in-place."""
    torch.ops._C.mamba_chunk_scan_fwd_cpu(
        out,
        final_states,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        cu_seqlens,
    )


def weight_packed_linear_cpu(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    n: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty((*x.shape[0:-1], n), dtype=x.dtype)
    torch.ops._C.weight_packed_linear(
        output.reshape(-1, n), x.reshape(-1, x.size(-1)), packed_weight, bias, True
    )
    return output


class CPUActMethod(IntEnum):
    SILU_AND_MUL = 0
    SWIGLU = 1
    GELU_AND_MUL = 2
    GELU_TANH_AND_MUL = 3


class CPUQuantMethod(IntEnum):
    UNQUANT = 0
    INT8_W8A8 = 1
    FP8_W8A16 = 2
    INT4_W4A8 = 3
    MXFP4 = 4
    FP8_W8A8 = 5


if hasattr(torch.ops._C, "dynamic_4bit_int_moe"):

    @register_fake("_C::dynamic_4bit_int_moe")
    def dynamic_4bit_int_moe_fake(
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13_packed: torch.Tensor,
        w2_packed: torch.Tensor,
        hidden_size: int,
        intermediate_size: int,
        group_size: int,
        apply_router_weight_on_input: bool,
        activation_kind: int,
    ) -> torch.Tensor:
        return x.new_empty((x.size(0), hidden_size))


def fused_experts_cpu(
    out: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    moe_comp_method: CPUQuantMethod,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    w1_zero: torch.Tensor | None,
    w2_zero: torch.Tensor | None,
    block_size: list[int] | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    alpha: float | None = None,
    limit: float | None = None,
    is_vnni: bool = True,
    a1_scale: torch.Tensor | None = None,
    act_method: CPUActMethod = CPUActMethod.SILU_AND_MUL,
) -> None:
    torch.ops._C.fused_experts_cpu(
        out,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_comp_method,
        w1_scale,
        w2_scale,
        w1_zero,
        w2_zero,
        a1_scale,
        block_size,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
        act_method,
    )


if hasattr(torch.ops._C, "int8_scaled_mm_with_quant"):

    @register_fake("_C::int8_scaled_mm_with_quant")
    def int8_scaled_mm_with_quant_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        scales2: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        is_vnni: bool,
    ) -> torch.Tensor:
        M = mat1.size(0)
        N = mat2.size(0)
        return torch.empty((M, N), dtype=out_dtype)


class CPUQuantAlgo(IntEnum):
    AWQ = 0
    GPTQ = 1


if hasattr(torch.ops._C, "convert_weight_packed_scale_zp"):

    @register_fake("_C::convert_weight_packed_scale_zp")
    def convert_weight_packed_scale_zp_fake(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        quant_method_4bit: CPUQuantAlgo,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty_like(qweight),
            torch.empty_like(qzeros),
            torch.empty_like(scales),
        )


def convert_weight_packed_scale_zp(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    quant_method_4bit: CPUQuantAlgo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops._C.convert_weight_packed_scale_zp(
        qweight,
        qzeros,
        scales,
        quant_method_4bit,
    )


if hasattr(torch.ops._C, "int4_scaled_mm_cpu"):

    @register_fake("_C::int4_scaled_mm_cpu")
    def int4_scaled_mm_cpu_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        w_zeros: torch.Tensor,
        w_scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        N = w_scales.size(0) * w_scales.size(-1)
        return torch.empty((x.size(0), N), dtype=x.dtype, device=x.device)


def int4_scaled_mm_cpu(
    x: torch.Tensor,
    w: torch.Tensor,
    w_zeros: torch.Tensor,
    w_scales: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1]) if len(x_shape) > 2 else x

    out = torch.ops._C.int4_scaled_mm_cpu(
        x_2d,
        w,
        w_zeros,
        w_scales,
        bias,
    )
    out = out.reshape(x_shape[:-1] + (out.size(-1),)) if len(x_shape) > 2 else out
    return out


if hasattr(torch.ops._C, "fp8_scaled_mm_cpu"):

    @register_fake("_C::fp8_scaled_mm_cpu")
    def fp8_scaled_mm_cpu_fake(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        scales2: torch.Tensor,
        block_size: list[int],
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        is_vnni: bool,
    ) -> torch.Tensor:
        M = mat1.size(0)
        N = mat2.size(0)
        return torch.empty((M, N), dtype=out_dtype, device=mat1.device)


_supports_cpu_fp8_w8a16 = bool(hasattr(torch.ops._C, "fp8_scaled_mm_cpu"))


def fp8_scaled_mm_cpu(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    scales2: torch.Tensor,
    block_size: list[int],
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    is_vnni: bool,
) -> torch.Tensor:
    return torch.ops._C.fp8_scaled_mm_cpu(
        mat1, mat2, scales2, block_size, bias, out_dtype, is_vnni
    )


# FP8 W8A8 CPU kernels
@cache
def cpu_has_amx_fp8() -> bool:
    """Whether this CPU has native AMX-FP8 MMA support."""
    if not hasattr(torch.ops._C, "cpu_has_amx_fp8"):
        return False
    return bool(torch.ops._C.cpu_has_amx_fp8())


if hasattr(torch.ops._C, "fp8_scaled_mm_with_quant"):

    @register_fake("_C::fp8_scaled_mm_with_quant")
    def fp8_scaled_mm_with_quant_fake(
        act: torch.Tensor,
        act_scales: torch.Tensor | None,
        channelwise: bool,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        M = act.reshape(-1, act.size(-1)).size(0)
        N = weight.size(0) * weight.size(-1)
        return torch.empty((M, N), dtype=output_dtype, device=act.device)


def fp8_scaled_mm_with_quant(
    act: torch.Tensor,
    act_scales: torch.Tensor | None,
    channelwise: bool,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops._C.fp8_scaled_mm_with_quant(
        act, act_scales, channelwise, weight, weight_scales, bias, output_dtype
    )


def chunk_gated_delta_rule_cpu(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    head_first: bool,
    use_qk_l2norm_in_kernel: bool,
    initial_state_indices: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.chunk_gated_delta_rule_cpu(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel,
        initial_state_indices,
        eps,
    )


def fused_sigmoid_gating_delta_rule_update_cpu(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    return torch.ops._C.fused_sigmoid_gating_delta_rule_update_cpu(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        softplus_beta,
        softplus_threshold,
    )


def fused_sigmoid_gating_delta_rule_update_spec_cpu(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    return torch.ops._C.fused_sigmoid_gating_delta_rule_update_spec_cpu(
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        initial_state_source,
        spec_state_indices,
        num_accepted_tokens,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        softplus_beta,
        softplus_threshold,
    )


def fused_gdn_gating_cpu(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.fused_gdn_gating_cpu(
        A_log,
        a,
        b,
        dt_bias,
    )


def causal_conv1d_weight_pack(
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_weight_pack(
        weight,
    )


def causal_conv1d_fwd_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    silu_activation: bool,
    is_weight_packed: bool,
) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_fwd_cpu(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        silu_activation,
        -1,
        is_weight_packed,
    )


def causal_conv1d_update_cpu(
    x: torch.Tensor,
    conv_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    silu_activation: bool,
    conv_state_indices: torch.Tensor | None,
    is_weight_packed: bool,
    num_accepted_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_update_cpu(
        x,
        conv_states,
        weight,
        bias,
        silu_activation,
        num_accepted_tokens,
        conv_state_indices,
        -1,
        is_weight_packed,
    )


class CPUDNNLGEMMHandler:
    def __init__(self) -> None:
        self.handler_tensor: torch.Tensor | None = None
        self.n = -1
        self.k = -1
        self.dtor = torch.ops._C.release_dnnl_matmul_handler

    def __del__(self):
        if self.handler_tensor is not None:
            self.dtor(self.handler_tensor.item())


_supports_onednn = bool(hasattr(torch.ops._C, "create_onednn_mm_handler"))


def is_onednn_acl_supported():
    return torch.ops._C.is_onednn_acl_supported()


def create_onednn_mm(
    weight: torch.Tensor,  # [K, N]
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    # store the handler pointer in a tensor it doesn't get inlined
    handler.handler_tensor = torch.tensor(
        torch.ops._C.create_onednn_mm_handler(weight, primitive_cache_size),
        dtype=torch.int64,
    )
    return handler


def onednn_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty((*x.shape[0:-1], dnnl_handler.n), dtype=x.dtype)
    torch.ops._C.onednn_mm(
        output, x.reshape(-1, dnnl_handler.k), bias, dnnl_handler.handler_tensor
    )

    return output


def create_onednn_scaled_mm(
    weight: torch.Tensor,  # [K, N]
    weight_scales: torch.Tensor,
    output_type: torch.dtype,
    dynamic_quant: bool,
    use_azp: bool,
    primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    # store the handler pointer in a tensor so it doesn't get inlined
    handler.handler_tensor = torch.tensor(
        torch.ops._C.create_onednn_scaled_mm_handler(
            weight,
            weight_scales,
            output_type,
            dynamic_quant,
            use_azp,
            primitive_cache_size,
        ),
        dtype=torch.int64,
    )
    return handler


def onednn_scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize the input tensor to int8 and return the quantized tensor and
    scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
        Output int8 tensor, scales, and optionally azp.

    """
    output = torch.empty_like(input, dtype=torch.int8)
    token_num = input.numel() // input.shape[-1]
    input = input.view((token_num, input.shape[-1]))
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((token_num, 1), device=input.device, dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales, input_azp)
    return output, input_scales, input_azp


def onednn_scaled_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor | None,
    input_zp: torch.Tensor | None,
    input_zp_adj: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    torch.ops._C.onednn_scaled_mm(
        output,
        x,
        input_scale,
        input_zp,
        input_zp_adj,
        bias,
        dnnl_handler.handler_tensor,
    )

    return output


def cpu_attn_get_scheduler_metadata(
    num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    dtype: torch.dtype,
    query_start_loc: torch.Tensor,
    causal: bool,
    sliding_window_size: int,
    isa: str,
    enable_kv_split: bool,
    dynamic_causal: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> torch.Tensor:
    scheduler_metadata = torch.ops._C.get_scheduler_metadata(
        num_reqs,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_lens,
        dtype,
        query_start_loc,
        causal,
        sliding_window_size,
        isa,
        enable_kv_split,
        dynamic_causal,
        kv_cache_dtype,
    )
    return scheduler_metadata


def cpu_attn_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    isa: str,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_cache_dtype: str = "auto",
) -> None:
    torch.ops._C.cpu_attn_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        isa,
        k_scale,
        v_scale,
        kv_cache_dtype,
    )


def cpu_attention_with_kv_cache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int,
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor | None,
    dynamic_causal: torch.Tensor | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    kv_cache_dtype: str = "auto",
) -> None:
    torch.ops._C.cpu_attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        causal,
        alibi_slopes,
        sliding_window,
        block_table,
        softcap,
        scheduler_metadata,
        s_aux,
        dynamic_causal,
        k_scale,
        v_scale,
        kv_cache_dtype,
    )


def cpu_mla_decode(
    query: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    output: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    loc: torch.Tensor | None,
    attn_logits: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    is_cross_attn: bool,
    sliding_window_size: int,
    encoder_lens: torch.Tensor | None,
    sinks: torch.Tensor | None,
) -> None:
    torch.ops._C.decode_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        key,
        value,
        loc,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
        is_cross_attn,
        sliding_window_size,
        encoder_lens,
        sinks,
    )


def cpu_mla_extend(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor | None,
    v_extend: torch.Tensor | None,
    o_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    max_len_extend: int,
    sm_scale: float,
    logit_cap: float,
    is_cross_attn: bool,
    sliding_window_size: int,
    encoder_lens: torch.Tensor | None,
    sinks: torch.Tensor | None,
    tree_mask: torch.Tensor | None = None,
) -> None:
    torch.ops._C.extend_attention_cpu(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        max_len_extend,
        sm_scale,
        logit_cap,
        is_cross_attn,
        sliding_window_size,
        encoder_lens,
        sinks,
        tree_mask,
    )


def bmm_cpu(
    out: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    is_vnni: bool,
    scale: torch.Tensor | None = None,
) -> None:
    torch.ops._C.bmm_cpu(out, mat1, mat2, is_vnni, scale)


def amx_mla_concat_and_cache(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops._C.concat_and_cache_mla_cpu(kv_c_normed, k_pe, kv_cache, slot_mapping)


if hasattr(torch.ops._C, "hc_pre_fused_cpu"):

    @register_fake("_C::hc_pre_fused_cpu")
    def hc_pre_fused_cpu_fake(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t, _, d = x.shape
        y = x.new_empty((t, d))
        post = x.new_empty((t, hc_mult), dtype=torch.float32)
        comb = x.new_empty((t, hc_mult, hc_mult), dtype=torch.float32)
        return y, post, comb


def hc_pre_fused_cpu(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    rms_eps: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops._C.hc_pre_fused_cpu(
        x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps
    )


if hasattr(torch.ops._C, "hc_post_fused_cpu"):

    @register_fake("_C::hc_post_fused_cpu")
    def hc_post_fused_cpu_fake(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        t, d = x.shape
        hc = residual.size(1)
        return x.new_empty((t, hc, d))


def hc_post_fused_cpu(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.hc_post_fused_cpu(x, residual, post, comb)


if hasattr(torch.ops._C, "hc_head_fused_cpu"):

    @register_fake("_C::hc_head_fused_cpu")
    def hc_head_fused_cpu_fake(
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_eps: float,
        norm_eps: float,
    ) -> torch.Tensor:
        t, _, d = x.shape
        return x.new_empty((t, d))


def hc_head_fused_cpu(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    return torch.ops._C.hc_head_fused_cpu(x, hc_fn, hc_scale, hc_base, hc_eps, norm_eps)


if hasattr(torch.ops._C, "fused_qnorm_rope_kv_insert_cpu"):

    @register_fake("_C::fused_qnorm_rope_kv_insert_cpu")
    def fused_qnorm_rope_kv_insert_cpu_fake(
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        swa_kv_cache_2d: torch.Tensor,
        slot_mapping: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        q_head_padded: int,
        eps: float,
        cache_block_size: int,
    ) -> torch.Tensor:
        num_tokens_full, _, head_dim = q.shape
        return q.new_empty((num_tokens_full, q_head_padded, head_dim))


def fused_qnorm_rope_kv_insert_cpu(
    q: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_head_padded: int,
    eps: float,
    cache_block_size: int,
) -> torch.Tensor:
    return torch.ops._C.fused_qnorm_rope_kv_insert_cpu(
        q,
        kv,
        positions,
        swa_kv_cache_2d,
        slot_mapping,
        cos_sin_cache,
        q_head_padded,
        eps,
        cache_block_size,
    )


def flash_mla_with_kvcache_cpu(
    out: torch.Tensor,
    q: torch.Tensor,
    window_cache_2d: torch.Tensor,
    window_slots: torch.Tensor,
    window_block_size: int,
    compressed_cache_2d: torch.Tensor,
    compressed_slots: torch.Tensor,
    compressed_block_size: int,
    attn_sink: torch.Tensor,
    scale: float,
) -> None:
    torch.ops._C.flash_mla_with_kvcache_cpu(
        out,
        q,
        window_cache_2d,
        window_slots,
        window_block_size,
        compressed_cache_2d,
        compressed_slots,
        compressed_block_size,
        attn_sink,
        scale,
    )


def save_partial_states_cpu(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops._C.save_partial_states_cpu(
        kv, score, ape, positions, state_cache, slot_mapping
    )


def compress_norm_rope_store_cpu(
    state_cache: torch.Tensor,
    gather_slots: torch.Tensor,
    positions: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    torch.ops._C.compress_norm_rope_store_cpu(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache_2d,
        kv_cache_block_size,
        compress_ratio,
    )


def compress_norm_rope_store_indexer_cpu(
    state_cache: torch.Tensor,
    gather_slots: torch.Tensor,
    positions: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    torch.ops._C.compress_norm_rope_store_indexer_cpu(
        state_cache,
        gather_slots,
        positions,
        kv_slot_mapping,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache_2d,
        kv_cache_block_size,
        compress_ratio,
    )


if hasattr(torch.ops._C, "fp8_paged_mqa_logits_cpu"):

    @register_fake("_C::fp8_paged_mqa_logits_cpu")
    def fp8_paged_mqa_logits_cpu_fake(
        q_fp8: torch.Tensor,
        kvcache_fp8: torch.Tensor,
        weight: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        block_size: int,
        max_seq_len: int,
    ) -> torch.Tensor:
        batch_size = q_fp8.size(0)
        return q_fp8.new_empty((batch_size, max_seq_len), dtype=torch.float32)


def fp8_paged_mqa_logits_cpu(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    block_size: int,
    max_seq_len: int,
) -> torch.Tensor:
    return torch.ops._C.fp8_paged_mqa_logits_cpu(
        q_fp8, kvcache_fp8, weight, seq_lens, page_table, block_size, max_seq_len
    )


def topk_transform_512_cpu(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: torch.Tensor | None = None,
) -> None:
    torch.ops._C.topk_transform_512_cpu(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )


def fused_indexer_q_rope_quant_cpu(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_q_fp8: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    index_weights_out: torch.Tensor,
) -> None:
    torch.ops._C.fused_indexer_q_rope_quant_cpu(
        positions,
        index_q,
        index_q_cos_sin_cache,
        index_q_fp8,
        index_weights,
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
    )


if hasattr(torch.ops._C, "biased_topk_cpu"):

    @register_fake("_C::biased_topk_cpu")
    def biased_topk_cpu_fake(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        renormalize: bool,
        scoring_func: str,
        num_fused_shared_experts: int,
        routed_scaling_factor: float | None,
        apply_routed_scaling_factor_on_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.size(0)
        topk_weights = hidden_states.new_empty((num_tokens, topk), dtype=torch.float32)
        topk_ids = hidden_states.new_empty((num_tokens, topk), dtype=torch.int32)
        return topk_weights, topk_ids


def biased_topk_cpu(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    num_fused_shared_experts: int,
    routed_scaling_factor: float | None,
    apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.biased_topk_cpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk,
        renormalize,
        scoring_func,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )


if hasattr(torch.ops._C, "hash_topk_cpu"):

    @register_fake("_C::hash_topk_cpu")
    def hash_topk_cpu_fake(
        gating_output: torch.Tensor,
        tid2eid: torch.Tensor,
        topk: int,
        scoring_func: str,
        num_fused_shared_experts: int,
        num_experts: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = gating_output.size(0)
        topk_weights = gating_output.new_empty((num_tokens, topk), dtype=torch.float32)
        topk_ids = gating_output.new_empty((num_tokens, topk), dtype=torch.int32)
        return topk_weights, topk_ids


def hash_topk_cpu(
    gating_output: torch.Tensor,
    tid2eid: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    num_experts: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.hash_topk_cpu(
        gating_output,
        tid2eid,
        topk,
        scoring_func,
        num_fused_shared_experts,
        num_experts,
        routed_scaling_factor,
    )


if hasattr(torch.ops._C, "inverse_gptj_rope_o_proj_cpu"):

    @register_fake("_C::inverse_gptj_rope_o_proj_cpu")
    def inverse_gptj_rope_o_proj_cpu_fake(
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rope_dim: int,
    ) -> torch.Tensor:
        return o.new_empty(o.shape, dtype=torch.float32)


def inverse_gptj_rope_o_proj_cpu(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    return torch.ops._C.inverse_gptj_rope_o_proj_cpu(
        o, positions, cos_sin_cache, rope_dim
    )


def cpu_gemm_wna16(
    input: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    bias: torch.Tensor | None,
    pack_factor: int,
    isa_hint: str,
) -> torch.Tensor:
    # Match int4_scaled_mm_cpu: flatten >2-D activations to [M, K] for the
    # C++ kernel, then restore the original leading dims on the output.
    x_shape = input.shape
    x_2d = input.reshape(-1, x_shape[-1]) if len(x_shape) > 2 else input
    out = torch.empty((x_2d.size(0), scales.size(1)), dtype=input.dtype)
    torch.ops._C.cpu_gemm_wna16(
        x_2d,
        q_weight,
        out,
        scales,
        zeros,
        bias,
        pack_factor,
        isa_hint,
    )
    out = out.reshape(x_shape[:-1] + (out.size(-1),)) if len(x_shape) > 2 else out
    return out


def cpu_prepack_moe_weight(
    weight: torch.Tensor,
    isa: str,
) -> torch.Tensor:
    output = torch.empty_like(weight)
    torch.ops._C.prepack_moe_weight(weight, output, isa)
    return output


def cpu_prepack_moe_weight_int8(
    weight: torch.Tensor,
    isa: str,
) -> torch.Tensor:
    output = torch.empty_like(weight)
    torch.ops._C.prepack_moe_weight_int8(weight, output, isa)
    return output


def cpu_fused_moe(
    out: torch.Tensor,
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act: str,
    isa: str,
    skip_weighted: bool = False,
) -> None:
    torch.ops._C.cpu_fused_moe(
        out,
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        skip_weighted,
        act,
        isa,
    )


def cpu_fused_moe_int8(
    out: torch.Tensor,
    input: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_bias: torch.Tensor | None,
    w2_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    act: str,
    isa: str,
    skip_weighted: bool = False,
) -> None:
    torch.ops._C.cpu_fused_moe_int8(
        out,
        input,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        skip_weighted,
        act,
        isa,
    )
