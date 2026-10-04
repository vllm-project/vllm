# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Triton operations for Qwen4Exp position-learning enhancement."""

from typing import Literal

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from ..runtime import has_active_triton_cpu_backend


def _require_cpu_triton(*tensors: torch.Tensor) -> None:
    if not has_active_triton_cpu_backend():
        raise RuntimeError("CPU Qwen4Exp PLE requires an active Triton CPU backend")
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise RuntimeError("CPU Qwen4Exp PLE requires CPU tensors")


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "binary_search_iters"])
def _ple_ngram_ids_kernel(
    input_ids_ptr,
    query_start_ptr,
    context_ptr,
    multipliers_ptr,
    sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_tokens,
    num_reqs,
    eos_token_id,
    binary_search_iters,
    NGRAM_CONTEXT_LEN: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    ngram_heads: tl.constexpr = NGRAM_CONTEXT_LEN * HEADS_PER_NGRAM
    block_h: tl.constexpr = triton.next_power_of_2(ngram_heads)
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < num_tokens

    request_lo = tl.full([BLOCK_T], 1, tl.int32)
    request_hi = tl.full([BLOCK_T], num_reqs + 1, tl.int32)
    for _ in range(binary_search_iters):
        middle = (request_lo + request_hi) // 2
        boundary = tl.load(
            query_start_ptr + middle,
            mask=token_mask & (middle <= num_reqs),
            other=0,
        )
        boundary_precedes_token = boundary <= token_offsets
        request_lo = tl.where(boundary_precedes_token, middle + 1, request_lo)
        request_hi = tl.where(boundary_precedes_token, request_hi, middle)
    request = tl.minimum(request_lo - 1, num_reqs - 1).to(tl.int64)
    request_start = tl.load(query_start_ptr + request, mask=token_mask, other=0)
    chunk_position = token_offsets - request_start

    current_token = tl.load(input_ids_ptr + token_offsets, mask=token_mask, other=0).to(
        tl.int64
    )
    mixed = current_token[:, None] * tl.load(multipliers_ptr)

    head = tl.arange(0, block_h)
    head_mask = head < ngram_heads
    ngram_order = head // HEADS_PER_NGRAM + 2

    crossed_eos = tl.zeros([BLOCK_T], tl.int1)
    for shift in tl.static_range(1, NGRAM_CONTEXT_LEN + 1):
        in_chunk = chunk_position >= shift
        context_column = NGRAM_CONTEXT_LEN - shift + chunk_position
        chunk_token = tl.load(
            input_ids_ptr + token_offsets - shift,
            mask=token_mask & in_chunk,
            other=0,
        )
        context_token = tl.load(
            context_ptr + request * NGRAM_CONTEXT_LEN + context_column,
            mask=token_mask & (~in_chunk),
            other=0,
        )
        candidate = tl.where(in_chunk, chunk_token, context_token).to(tl.int64)
        candidate = tl.where(crossed_eos, eos_token_id, candidate)
        crossed_eos = crossed_eos | (candidate == eos_token_id)
        term = candidate[:, None] * tl.load(multipliers_ptr + shift)
        mixed = mixed ^ tl.where((ngram_order > shift)[None, :], term, 0)

    sizes = tl.load(sizes_ptr + head, mask=head_mask, other=1)[None, :]
    head_offsets = tl.load(offsets_ptr + head, mask=head_mask, other=0)[None, :]
    remainders = mixed % sizes
    remainders = tl.where(remainders < 0, remainders + sizes, remainders)
    ids = remainders + head_offsets
    tl.store(
        output_ptr + token_offsets[:, None] * ngram_heads + head[None, :],
        ids,
        mask=token_mask[:, None] & head_mask[None, :],
    )


def _ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    output: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> None:
    tensors = (
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        output,
    )
    _require_cpu_triton(*tensors)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("PLE n-gram tensors must be contiguous")
    if input_ids.dtype != torch.int32 or query_start_loc.dtype != torch.int32:
        raise ValueError("PLE n-gram input IDs and request offsets must be int32")
    if ngram_context.dtype != torch.int32:
        raise ValueError("PLE n-gram context must be int32")
    if any(
        tensor.dtype != torch.int64
        for tensor in (
            layer_multipliers,
            ngram_heads_vocab_sizes,
            ngram_heads_offsets,
            output,
        )
    ):
        raise ValueError("PLE n-gram hash parameters and output must be int64")
    if ngram_context.ndim != 2 or query_start_loc.ndim != 1:
        raise ValueError("PLE n-gram context and request offsets must be 2D and 1D")

    num_tokens = input_ids.numel()
    num_reqs = query_start_loc.numel() - 1
    context_len = ngram_context.shape[1]
    ngram_heads = context_len * heads_per_ngram
    if num_reqs < 0 or ngram_context.shape[0] != num_reqs:
        raise ValueError("PLE n-gram context must have one row per request")
    if context_len <= 0 or heads_per_ngram <= 0:
        raise ValueError("PLE n-gram context and heads per n-gram must be positive")
    if layer_multipliers.numel() != context_len + 1:
        raise ValueError("PLE n-gram multipliers must cover every context position")
    if (
        ngram_heads_vocab_sizes.numel() != ngram_heads
        or ngram_heads_offsets.numel() != ngram_heads
    ):
        raise ValueError("PLE n-gram hash parameters must cover every head")
    if output.shape != (num_tokens, ngram_heads):
        raise ValueError("PLE n-gram output has an unexpected shape")
    if num_tokens and num_reqs == 0:
        raise ValueError("PLE n-gram tokens require at least one request")

    block_tokens = 8
    if num_tokens:
        _ple_ngram_ids_kernel[(triton.cdiv(num_tokens, block_tokens),)](
            input_ids,
            query_start_loc,
            ngram_context,
            layer_multipliers,
            ngram_heads_vocab_sizes,
            ngram_heads_offsets,
            output,
            num_tokens,
            num_reqs,
            eos_token_id,
            binary_search_iters=num_reqs.bit_length(),
            NGRAM_CONTEXT_LEN=context_len,
            HEADS_PER_NGRAM=heads_per_ngram,
            BLOCK_T=block_tokens,
            num_cpu_threads=0,
        )


def _ple_ngram_ids_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    output: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
) -> None:
    del (
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        output,
        eos_token_id,
        heads_per_ngram,
    )


direct_register_custom_op(
    op_name="qwen4_exp_cpu_ple_ngram_ids",
    op_func=_ple_ngram_ids,
    mutates_args=["output"],
    fake_impl=_ple_ngram_ids_fake,
)


def ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    layer_multipliers: torch.Tensor,
    ngram_heads_vocab_sizes: torch.Tensor,
    ngram_heads_offsets: torch.Tensor,
    eos_token_id: int,
    heads_per_ngram: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if output is None:
        output = torch.empty(
            (input_ids.numel(), ngram_context.shape[1] * heads_per_ngram),
            dtype=torch.int64,
            device=input_ids.device,
        )
    torch.ops.vllm.qwen4_exp_cpu_ple_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
        layer_multipliers,
        ngram_heads_vocab_sizes,
        ngram_heads_offsets,
        output,
        eos_token_id,
        heads_per_ngram,
    )
    return output


@triton.jit
def _ple_gate_kernel(
    key_ptr,
    value_ptr,
    hidden_ptr,
    nk_ptr,
    nq_ptr,
    ncw_ptr,
    gated_ptr,
    normed_ptr,
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    stream = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offsets = stream * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    key = tl.load(key_ptr + token * key_rs + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    query = tl.load(hidden_ptr + token * HC * H + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    norm_key = tl.load(nk_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm_query = tl.load(nq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    key = (key * tl.rsqrt(tl.sum(key * key) / H + eps) * (1.0 + norm_key)).to(dtype)
    query = (query * tl.rsqrt(tl.sum(query * query) / H + eps) * (1.0 + norm_query)).to(
        dtype
    )
    products = (key.to(tl.float32) * query.to(tl.float32)).to(dtype)
    similarity = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    similarity = (similarity / tl.sqrt(float(H))).to(dtype).to(tl.float32)
    sign = tl.where(similarity < 0, -1.0, 0.0)
    sign = tl.where(similarity > 0, 1.0, sign)
    magnitude = tl.sqrt(tl.maximum(tl.abs(similarity), 1e-6)).to(dtype)
    gate = tl.sigmoid(sign * magnitude).to(dtype).to(tl.float32)

    value = tl.load(value_ptr + token * value_rs + lanes, mask=mask, other=0.0).to(
        tl.float32
    )
    gated = (gate * value).to(dtype)
    gated_float = gated.to(tl.float32)
    norm_conv = tl.load(ncw_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = (
        gated_float
        * tl.rsqrt(tl.sum(gated_float * gated_float) / H + eps)
        * (1.0 + norm_conv)
    )

    tl.store(gated_ptr + token * HC * H + offsets, gated, mask=mask)
    tl.store(normed_ptr + token * HC * H + offsets, normalized, mask=mask)


def _ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = (
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
    )
    _require_cpu_triton(*tensors)
    if any(tensor.device != key.device for tensor in tensors):
        raise ValueError("PLE gate tensors must share one device")
    if key.dtype != value.dtype or key.dtype != hidden.dtype:
        raise ValueError("key, value, and hidden must have the same dtype")
    if key.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("PLE gate supports BF16 and FP16 inputs")
    if key.ndim != 2 or value.ndim != 2 or hidden.shape != key.shape:
        raise ValueError("PLE gate expects matching two-dimensional activations")
    hidden_size = value.shape[1]
    if not hidden_size or key.shape[1] % hidden_size:
        raise ValueError("key and hidden must contain whole hidden-size groups")
    if key.stride(1) != 1 or value.stride(1) != 1 or not hidden.is_contiguous():
        raise ValueError("PLE gate activations must be contiguous in hidden size")

    num_tokens = hidden.shape[0]
    hc_count = hidden.shape[1] // hidden_size
    gated = torch.empty_like(hidden)
    normalized = torch.empty_like(hidden)
    if num_tokens:
        _ple_gate_kernel[(num_tokens, hc_count)](
            key,
            value,
            hidden,
            norm_key_w,
            norm_query_w,
            norm_conv_w,
            gated,
            normalized,
            key.stride(0),
            value.stride(0),
            eps,
            H=hidden_size,
            HC=hc_count,
            BLOCK_H=triton.next_power_of_2(hidden_size),
            num_cpu_threads=0,
        )
    return gated, normalized


def _ple_gate_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del key, value, norm_key_w, norm_query_w, norm_conv_w, eps
    return torch.empty_like(hidden), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="qwen4_exp_cpu_ple_gate",
    op_func=_ple_gate,
    mutates_args=[],
    fake_impl=_ple_gate_fake,
)


def ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.qwen4_exp_cpu_ple_gate(
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        eps,
    )


@triton.jit(do_not_specialize=["num_reqs", "bs_iters"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    residual_ptr,
    state_idx_ptr,
    query_start_ptr,
    num_accepted_ptr,
    has_initial_ptr,
    token_idx_ptr,
    num_reqs,
    bs_iters,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_TOKEN_MAP: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    token = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < C
    output_token = (
        tl.load(token_idx_ptr + token).to(tl.int64) if HAS_TOKEN_MAP else token
    )

    if MODE == "decode":
        request = token
        query_start = token
        query_offset = tl.full([], 0, tl.int32)
        state_offset = tl.full([], 0, tl.int32)
        in_range = True
    else:
        lower = tl.full([], 1, tl.int32)
        upper = tl.full([], num_reqs + 1, tl.int32)
        for _ in range(bs_iters):
            middle = (lower + upper) // 2
            boundary = tl.load(
                query_start_ptr + middle, mask=middle <= num_reqs, other=0
            )
            lower = tl.where(boundary <= token, middle + 1, lower)
            upper = tl.where(boundary <= token, upper, middle)
        request = tl.minimum(lower - 1, num_reqs - 1)
        query_start = tl.load(query_start_ptr + request)
        query_offset = (token - query_start).to(tl.int32)
        in_range = token < tl.load(query_start_ptr + num_reqs)
        if MODE == "spec":
            num_accepted = tl.load(num_accepted_ptr + request)
            state_offset = tl.minimum(
                tl.maximum(num_accepted - 1, 0), SPEC_QUERY_LEN - 1
            ).to(tl.int32)
        else:
            state_offset = tl.full([], 0, tl.int32)

    state_id = tl.load(state_idx_ptr + request * state_idx_stride).to(tl.int64)
    state_valid = state_id != NULL_STATE_ID
    safe_state_id = tl.where(state_valid, state_id, 0)
    if HAS_INIT:
        has_initial = tl.load(has_initial_ptr + request, mask=state_valid, other=0) != 0
    else:
        has_initial = state_valid
    if MODE == "spec":
        read_state = state_valid
        output_valid = in_range
    else:
        read_state = state_valid & has_initial
        output_valid = in_range & state_valid

    state_base = state_ptr + safe_state_id * state_bs
    accumulator = tl.zeros([BLOCK_C], tl.float32)
    for kernel_index in tl.static_range(0, KERNEL_SIZE):
        history_offset = query_offset + DILATION * kernel_index
        from_state = history_offset <= STATE_LEN - 1
        state_value = tl.load(
            state_base
            + (history_offset + state_offset) * state_ws
            + channels * state_cs,
            mask=channel_mask & read_state & from_state,
            other=0.0,
        )
        input_token = query_start + history_offset - STATE_LEN
        if HAS_TOKEN_MAP:
            input_token = tl.load(
                token_idx_ptr + input_token,
                mask=output_valid & (~from_state),
                other=input_token,
            ).to(tl.int64)
        input_value = tl.load(
            x_ptr + input_token * C + channels,
            mask=channel_mask & output_valid & (~from_state),
            other=0.0,
        )
        tap = tl.where(from_state, state_value, input_value).to(tl.float32)
        weight = tl.load(
            weight_ptr + channels * KERNEL_SIZE + kernel_index,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += weight * tap

    convolution = accumulator.to(residual_ptr.dtype.element_ty).to(tl.float32)
    convolution = convolution * tl.sigmoid(convolution)
    convolution = tl.where(output_valid, convolution, 0.0).to(
        residual_ptr.dtype.element_ty
    )
    residual = tl.load(
        residual_ptr + output_token * C + channels,
        mask=channel_mask,
        other=0.0,
    )
    tl.store(
        residual_ptr + output_token * C + channels,
        residual + convolution,
        mask=channel_mask,
    )

    if MODE == "decode":
        decode_input = tl.load(
            x_ptr + output_token * C + channels,
            mask=channel_mask & state_valid,
            other=0.0,
        )
        for state_index in tl.static_range(0, STATE_LEN):
            if state_index < STATE_LEN - 1:
                next_state = tl.load(
                    state_base + (state_index + 1) * state_ws + channels * state_cs,
                    mask=channel_mask & state_valid & has_initial,
                    other=0.0,
                )
            else:
                next_state = decode_input
            tl.store(
                state_base + state_index * state_ws + channels * state_cs,
                next_state,
                mask=channel_mask & state_valid,
            )


@triton.jit
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    query_start_ptr,
    num_accepted_ptr,
    has_initial_ptr,
    token_idx_ptr,
    state_idx_stride,
    state_bs,
    state_ws,
    state_cs,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STATE_LEN: tl.constexpr,
    SPEC_QUERY_LEN: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    MODE: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_TOKEN_MAP: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
):
    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < C

    state_id = tl.load(state_idx_ptr + request * state_idx_stride).to(tl.int64)
    state_valid = state_id != NULL_STATE_ID
    safe_state_id = tl.where(state_valid, state_id, 0)
    query_start = tl.load(query_start_ptr + request)
    query_end = tl.load(query_start_ptr + request + 1)
    query_length = (query_end - query_start).to(tl.int32)
    if MODE == "spec":
        num_accepted = tl.load(num_accepted_ptr + request)
        state_offset = tl.minimum(
            tl.maximum(num_accepted - 1, 0), SPEC_QUERY_LEN - 1
        ).to(tl.int32)
        shift = 1
    else:
        state_offset = tl.full([], 0, tl.int32)
        shift = query_length

    has_initial = (
        tl.load(has_initial_ptr + request, mask=state_valid, other=0) != 0
        if HAS_INIT
        else True
    )
    source_valid = state_valid if MODE == "spec" else state_valid & has_initial
    write_width: tl.constexpr = STATE_WIDTH if MODE == "spec" else STATE_LEN
    state_base = state_ptr + safe_state_id * state_bs
    for state_index in tl.static_range(0, write_width):
        history_offset = shift + state_index
        from_state = history_offset <= STATE_LEN - 1
        do_write = (
            state_index < STATE_LEN + query_length - 1
            if MODE == "spec"
            else query_length > 0
        )
        state_value = tl.load(
            state_base
            + (state_offset + history_offset) * state_ws
            + channels * state_cs,
            mask=channel_mask & from_state & source_valid,
            other=0.0,
        )
        input_token = query_start + history_offset - STATE_LEN
        if HAS_TOKEN_MAP:
            input_token = tl.load(
                token_idx_ptr + input_token,
                mask=(~from_state) & do_write,
                other=input_token,
            ).to(tl.int64)
        input_value = tl.load(
            x_ptr + input_token * C + channels,
            mask=channel_mask & (~from_state) & do_write,
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            state_base + state_index * state_ws + channels * state_cs,
            value,
            mask=channel_mask & state_valid & do_write,
        )


def _ple_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    conv_mode: str,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    """Add short-convolution output and update CPU-resident state."""
    required = [inputs, residual, conv_state, conv_weights, state_indices]
    required.extend(
        tensor
        for tensor in (
            query_start_loc,
            num_accepted_tokens,
            has_initial_states,
            token_indices,
        )
        if tensor is not None
    )
    _require_cpu_triton(*required)
    if inputs.ndim != 2 or residual.shape != inputs.shape:
        raise ValueError("inputs and residual must be matching 2D tensors")
    if conv_weights.ndim != 2 or conv_weights.shape[0] != inputs.shape[1]:
        raise ValueError("conv_weights must have one row per input channel")
    if not inputs.is_contiguous() or not residual.is_contiguous():
        raise ValueError("PLE convolution inputs and residual must be contiguous")
    if not conv_weights.is_contiguous():
        raise ValueError("PLE convolution weights must be contiguous")

    token_count, channels = inputs.shape
    kernel_size = conv_weights.shape[1]
    state_len = (kernel_size - 1) * dilation
    kernel_spec_query_len = spec_query_len if conv_mode == "spec" else 1
    state_width = state_len + kernel_spec_query_len - 1
    if token_indices is not None:
        token_count = token_indices.numel()
    if (
        conv_state.ndim != 3
        or conv_state.shape[1] != channels
        or conv_state.shape[2] < state_width
    ):
        raise ValueError(
            "conv_state must have shape [slots, channels, window], with "
            f"channels={channels} and window >= {state_width}"
        )
    if dilation <= 0 or kernel_size <= 0:
        raise ValueError("PLE convolution kernel size and dilation must be positive")

    if conv_mode == "decode":
        num_reqs = token_count
        binary_search_iters = 1
        has_initial_states_arg = has_initial_states is not None
    elif conv_mode == "spec":
        if query_start_loc is None or num_accepted_tokens is None:
            raise ValueError(
                "query_start_loc and num_accepted_tokens are required for spec decode"
            )
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_initial_states_arg = False
    elif conv_mode == "prefill":
        if query_start_loc is None or has_initial_states is None:
            raise ValueError(
                "query_start_loc and has_initial_states are required for prefill"
            )
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_initial_states_arg = True
    else:
        raise ValueError(f"Unsupported short-conv mode: {conv_mode}")

    state_bs, state_cs, state_ws = conv_state.stride()
    state_idx_stride = state_indices.stride(0)
    block_channels = min(256, triton.next_power_of_2(channels))
    token_map = token_indices if token_indices is not None else state_indices
    if token_count:
        _ple_conv_kernel[(token_count, triton.cdiv(channels, block_channels))](
            inputs,
            conv_state,
            conv_weights,
            residual,
            state_indices,
            query_start_loc,
            num_accepted_tokens,
            has_initial_states,
            token_map,
            num_reqs,
            binary_search_iters,
            state_idx_stride,
            state_bs,
            state_ws,
            state_cs,
            C=channels,
            BLOCK_C=block_channels,
            STATE_LEN=state_len,
            DILATION=dilation,
            KERNEL_SIZE=kernel_size,
            SPEC_QUERY_LEN=kernel_spec_query_len,
            MODE=conv_mode,
            HAS_INIT=has_initial_states_arg,
            HAS_TOKEN_MAP=token_indices is not None,
            NULL_STATE_ID=NULL_BLOCK_ID,
            num_cpu_threads=0,
        )
    if conv_mode != "decode" and num_reqs:
        _ple_conv_writeback_kernel[(num_reqs, triton.cdiv(channels, block_channels))](
            inputs,
            conv_state,
            state_indices,
            query_start_loc,
            num_accepted_tokens,
            has_initial_states,
            token_map,
            state_idx_stride,
            state_bs,
            state_ws,
            state_cs,
            C=channels,
            BLOCK_C=block_channels,
            STATE_LEN=state_len,
            SPEC_QUERY_LEN=kernel_spec_query_len,
            STATE_WIDTH=state_width,
            MODE=conv_mode,
            HAS_INIT=has_initial_states_arg,
            HAS_TOKEN_MAP=token_indices is not None,
            NULL_STATE_ID=NULL_BLOCK_ID,
            num_cpu_threads=0,
        )


def _ple_conv_fake(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    conv_mode: str,
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    del (
        inputs,
        residual,
        conv_state,
        conv_weights,
        state_indices,
        conv_mode,
        dilation,
        query_start_loc,
        num_accepted_tokens,
        has_initial_states,
        spec_query_len,
        token_indices,
    )


direct_register_custom_op(
    op_name="qwen4_exp_cpu_ple_conv",
    op_func=_ple_conv,
    mutates_args=["residual", "conv_state"],
    fake_impl=_ple_conv_fake,
)


def ple_conv(
    inputs: torch.Tensor,
    residual: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    mode: Literal["decode", "spec", "prefill"],
    dilation: int,
    query_start_loc: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    has_initial_states: torch.Tensor | None = None,
    spec_query_len: int = 1,
    token_indices: torch.Tensor | None = None,
) -> None:
    torch.ops.vllm.qwen4_exp_cpu_ple_conv(
        inputs,
        residual,
        conv_state,
        conv_weights,
        state_indices,
        mode,
        dilation,
        query_start_loc,
        num_accepted_tokens,
        has_initial_states,
        spec_query_len,
        token_indices,
    )


__all__ = ["ple_conv", "ple_gate", "ple_ngram_ids"]
