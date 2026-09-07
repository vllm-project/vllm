# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Qwen4Exp PLE kernels.

N-gram IDs hash each suffix with
``offset[h] + (xor_i(token[t-i] * multiplier[i]) % size[h])``. The gate computes
``d = dot(RMSNorm(key), RMSNorm(query)) / sqrt(H)`` and
``g = sigmoid(sign(d) * sqrt(max(abs(d), 1e-6)))``. Short convolution adds
``silu(sum_k weight[k] * history[t + k * dilation])`` to the gated output and
updates its persistent history.
"""

from typing import Literal

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


# N-gram ID generation
@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "binary_search_iters"])
def _ple_ngram_ids_kernel(
    input_ids_ptr,
    qsl_ptr,
    ctx_ptr,
    multipliers_ptr,
    sizes_ptr,
    offsets_ptr,
    out_ptr,
    num_tokens,
    num_reqs,
    eos_token_id,
    binary_search_iters,
    NGRAM_CONTEXT_LEN: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    NGRAM_HEADS: tl.constexpr = NGRAM_CONTEXT_LEN * HEADS_PER_NGRAM
    BLOCK_H: tl.constexpr = triton.next_power_of_2(NGRAM_HEADS)
    pid = tl.program_id(0)
    token_offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < num_tokens

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # A flattened token offset does not identify its request when query chunks
    # have different lengths. Binary-search the request boundaries instead.
    p_lo = tl.full([BLOCK_T], 1, tl.int32)
    p_hi = tl.full([BLOCK_T], num_reqs + 1, tl.int32)
    for _ in range(binary_search_iters):
        mid = (p_lo + p_hi) // 2
        qmid = tl.load(qsl_ptr + mid, mask=token_mask & (mid <= num_reqs), other=0)
        pred = qmid <= token_offsets
        p_lo = tl.where(pred, mid + 1, p_lo)
        p_hi = tl.where(pred, p_hi, mid)
    req = tl.minimum(p_lo - 1, num_reqs - 1).to(tl.int64)
    request_start = tl.load(qsl_ptr + req, mask=token_mask, other=0)
    chunk_pos = token_offsets - request_start

    # Initialize every head with the current-token term.
    current_token = tl.load(input_ids_ptr + token_offsets, mask=token_mask, other=0).to(
        tl.int64
    )
    current_multiplier = tl.load(multipliers_ptr)
    mixed = current_token[:, None] * current_multiplier

    g = tl.arange(0, BLOCK_H)
    head_mask = g < NGRAM_HEADS
    ngram_order = g // HEADS_PER_NGRAM + 2

    # Walk predecessors from newest to oldest. At chunk boundaries they come
    # from ngram_context; otherwise they come from this step's input_ids.
    crossed = tl.zeros([BLOCK_T], tl.int1)
    for shift in tl.static_range(1, NGRAM_CONTEXT_LEN + 1):
        in_step = chunk_pos >= shift
        ctx_col = NGRAM_CONTEXT_LEN - shift + chunk_pos
        step_token = tl.load(
            input_ids_ptr + token_offsets - shift,
            mask=token_mask & in_step,
            other=0,
        )
        context_token = tl.load(
            ctx_ptr + req * NGRAM_CONTEXT_LEN + ctx_col,
            mask=token_mask & (~in_step),
            other=0,
        )
        candidate = tl.where(in_step, step_token, context_token).to(tl.int64)
        # Older positions remain behind the first EOS boundary.
        candidate = tl.where(crossed, eos_token_id, candidate)
        crossed = crossed | (candidate == eos_token_id)
        multiplier = tl.load(multipliers_ptr + shift)
        term = candidate[:, None] * multiplier
        mixed = mixed ^ tl.where((ngram_order > shift)[None, :], term, 0)

    # Map each hash into its embedding-table partition.
    sizes = tl.load(sizes_ptr + g, mask=head_mask, other=1)[None, :]
    head_offsets = tl.load(offsets_ptr + g, mask=head_mask, other=0)[None, :]
    # Hash products may overflow int64; preserve torch.remainder semantics.
    remainders = mixed % sizes
    remainders = tl.where(remainders < 0, remainders + sizes, remainders)
    ids = remainders + head_offsets
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        out_ptr + token_offsets[:, None] * NGRAM_HEADS + g[None, :],
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
    input_ids = input_ids.reshape(-1)
    num_tokens = input_ids.shape[0]
    num_reqs = query_start_loc.numel() - 1
    ctx_len = ngram_context.shape[1]
    BLOCK_T = 8
    launch_pdl = current_platform.is_arch_support_pdl()
    _ple_ngram_ids_kernel[(triton.cdiv(num_tokens, BLOCK_T),)](
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
        NGRAM_CONTEXT_LEN=ctx_len,
        HEADS_PER_NGRAM=heads_per_ngram,
        BLOCK_T=BLOCK_T,
        launch_pdl=launch_pdl,
        num_warps=4,
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
    _ple_ngram_ids(
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


# Gating and normalization
@triton.jit
def _ple_gate_kernel(
    key_ptr,  # [T, HC*H] with row stride key_rs
    value_ptr,  # [T, H] with row stride value_rs
    hidden_ptr,  # [T, HC*H]
    nk_ptr,  # [HC*H] norm_key weight
    nq_ptr,  # [HC*H] norm_query weight
    ncw_ptr,  # [HC*H] norm_conv weight
    gated_ptr,  # [T, HC*H] out
    normed_ptr,  # [T, HC*H] out
    key_rs,
    value_rs,
    eps,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_H)
    mask = lanes < H
    offs = s * H + lanes
    dtype: tl.constexpr = key_ptr.dtype.element_ty

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    k = tl.load(key_ptr + t * key_rs + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(hidden_ptr + t * HC * H + offs, mask=mask, other=0.0).to(tl.float32)
    nk = tl.load(nk_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    nq = tl.load(nq_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Match eager materialization at each intermediate tensor boundary.
    k_n = (k * tl.rsqrt(tl.sum(k * k) / H + eps) * (1.0 + nk)).to(dtype)
    q_n = (q * tl.rsqrt(tl.sum(q * q) / H + eps) * (1.0 + nq)).to(dtype)
    products = (k_n.to(tl.float32) * q_n.to(tl.float32)).to(dtype)
    dot = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    d = dot / tl.sqrt(float(H))
    d = d.to(dtype).to(tl.float32)
    sign = tl.where(d < 0, -1.0, 0.0)
    sign = tl.where(d > 0, 1.0, sign)
    magnitude = tl.sqrt(tl.maximum(tl.abs(d), 1e-6)).to(dtype)
    g = tl.sigmoid(sign * magnitude)
    g = g.to(dtype).to(tl.float32)

    v = tl.load(value_ptr + t * value_rs + lanes, mask=mask, other=0.0).to(tl.float32)
    gated = (g * v).to(dtype)
    gf = gated.to(tl.float32)
    ncw = tl.load(ncw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = gf * tl.rsqrt(tl.sum(gf * gf) / H + eps) * (1.0 + ncw)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(gated_ptr + t * HC * H + offs, gated, mask=mask)
    tl.store(normed_ptr + t * HC * H + offs, normed, mask=mask)


def _ple_gate(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden.shape[0]
    h = value.shape[-1]
    hc = hidden.shape[-1] // h
    assert key.stride(1) == 1 and value.stride(1) == 1
    assert hidden.is_contiguous()
    if key.dtype != value.dtype or key.dtype != hidden.dtype:
        raise ValueError("key, value, and hidden must have the same dtype")
    if key.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("PLE gate supports BF16 and FP16 inputs")
    gated = torch.empty_like(hidden)
    normed = torch.empty_like(hidden)
    _ple_gate_kernel[(num_tokens, hc)](
        key,
        value,
        hidden,
        norm_key_w,
        norm_query_w,
        norm_conv_w,
        gated,
        normed,
        key.stride(0),
        value.stride(0),
        eps,
        H=h,
        HC=hc,
        BLOCK_H=triton.next_power_of_2(h),
        num_warps=4,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return gated, normed


def _ple_gate_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    hidden: torch.Tensor,
    norm_key_w: torch.Tensor,
    norm_query_w: torch.Tensor,
    norm_conv_w: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="qwen4_exp_ple_gate",
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
    return torch.ops.vllm.qwen4_exp_ple_gate(
        key, value, hidden, norm_key_w, norm_query_w, norm_conv_w, eps
    )


# Dilated short convolution
@triton.jit(do_not_specialize=["num_reqs", "bs_iters", "has_token_map"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map,
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
    NULL_STATE_ID: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Mode-specific launches use logical row indices. Mixed batches map them
    # back to the original token order without gathering inputs or residuals.
    output_t = tl.load(token_idx_ptr + t, mask=has_token_map, other=t).to(tl.int64)

    if MODE == "decode":
        r = t
        q_start = t
        j = tl.full([], 0, tl.int32)
        slot_off = tl.full([], 0, tl.int32)
        in_range = True
    else:
        # Locate the request containing this token.
        lo = tl.full([], 1, tl.int32)
        hi = tl.full([], num_reqs + 1, tl.int32)
        for _ in range(bs_iters):
            mid = (lo + hi) // 2
            qmid = tl.load(qsl_ptr + mid, mask=mid <= num_reqs, other=0)
            pred = qmid <= t
            lo = tl.where(pred, mid + 1, lo)
            hi = tl.where(pred, hi, mid)
        r = tl.minimum(lo - 1, num_reqs - 1)
        q_start = tl.load(qsl_ptr + r)
        j = (t - q_start).to(tl.int32)
        total_real = tl.load(qsl_ptr + num_reqs)
        in_range = t < total_real
        if MODE == "spec":
            num_acc = tl.load(num_acc_ptr + r)
            # Roll back draft state to the last accepted token.
            slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(
                tl.int32
            )
        else:
            slot_off = tl.full([], 0, tl.int32)

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    sid_safe = tl.where(state_ok, sid, 0)
    if HAS_INIT:
        has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    else:
        has_init = state_ok
    if MODE == "spec":
        # A new spec chunk can produce output from its own tokens without a
        # prior state; null slots in other modes denote inactive rows.
        read_state = state_ok
        out_ok = in_range
    else:
        read_state = state_ok & has_init
        out_ok = in_range & state_ok

    # state[sid, w, c] may be a strided view into a shared pool.
    base_state = state_ptr + sid_safe * state_bs
    acc = tl.zeros([BLOCK_C], tl.float32)
    for k in tl.static_range(0, KERNEL_SIZE):
        h = j + DILATION * k
        from_state = h <= STATE_LEN - 1
        state_tap = tl.load(
            base_state + (h + slot_off) * state_ws + c_offs * state_cs,
            mask=c_mask & read_state & from_state,
            other=0.0,
        )
        input_t = q_start + h - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & out_ok & (~from_state),
            other=input_t,
        ).to(tl.int64)
        input_tap = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & out_ok & (~from_state),
            other=0.0,
        )
        tap = tl.where(from_state, state_tap, input_tap).to(tl.float32)
        weight = tl.load(
            w_ptr + c_offs * KERNEL_SIZE + k,
            mask=c_mask,
            other=0.0,
        ).to(tl.float32)
        acc += weight * tap

    # F.conv1d materializes its output dtype before SiLU.
    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + output_t * C + c_offs,
        mask=c_mask,
        other=0.0,
    )
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        residual_ptr + output_t * C + c_offs,
        residual + conv_output,
        mask=c_mask,
    )

    if MODE == "decode":
        # Decode has one program per request and channel block, so no peer can
        # still be reading this state slice when it is updated.
        decode_input = tl.load(
            x_ptr + output_t * C + c_offs,
            mask=c_mask & state_ok,
            other=0.0,
        )
        for i in tl.static_range(0, STATE_LEN):
            if i < STATE_LEN - 1:
                next_state = tl.load(
                    base_state + (i + 1) * state_ws + c_offs * state_cs,
                    mask=c_mask & state_ok & has_init,
                    other=0.0,
                )
            else:
                next_state = decode_input
            tl.store(
                base_state + i * state_ws + c_offs * state_cs,
                next_state,
                mask=c_mask & state_ok,
            )


# Prefill and spec write back separately so every token reads the old state.
@triton.jit(do_not_specialize=["has_token_map"])
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    token_idx_ptr,
    has_token_map,
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
    NULL_STATE_ID: tl.constexpr,
):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sid = tl.load(state_idx_ptr + r * state_idx_stride).to(tl.int64)
    state_ok = sid != NULL_STATE_ID
    if not state_ok:
        return

    q_start = tl.load(qsl_ptr + r)
    q_end = tl.load(qsl_ptr + r + 1)
    qlen = (q_end - q_start).to(tl.int32)
    if MODE == "spec":
        num_acc = tl.load(num_acc_ptr + r)
        # Use the same rollback window as the output kernel.
        slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(
            tl.int32
        )
        shift = 1
    else:
        slot_off = tl.full([], 0, tl.int32)
        shift = qlen
        if qlen <= 0:
            return

    has_init = tl.load(has_init_ptr + r) != 0 if HAS_INIT else True
    src_ok = state_ok if MODE == "spec" else state_ok & has_init

    WRITE_W: tl.constexpr = STATE_WIDTH if MODE == "spec" else STATE_LEN
    base_state = state_ptr + sid * state_bs
    for i in tl.static_range(0, WRITE_W):
        m = shift + i
        from_state = m <= STATE_LEN - 1
        # Short or graph-padded spec chunks preserve the unused state tail.
        do_write = i < STATE_LEN + qlen - 1 if MODE == "spec" else True
        state_value = tl.load(
            base_state + (slot_off + m) * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_t = q_start + m - STATE_LEN
        input_t = tl.load(
            token_idx_ptr + input_t,
            mask=has_token_map & (~from_state) & do_write,
            other=input_t,
        ).to(tl.int64)
        input_value = tl.load(
            x_ptr + input_t * C + c_offs,
            mask=c_mask & (~from_state) & do_write,
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask & do_write,
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
    """Add short-convolution output to ``residual`` and update its state."""
    BLOCK_C = 512
    kernel_spec_query_len = spec_query_len if mode == "spec" else 1
    T, C = inputs.shape
    K = conv_weights.shape[1]
    state_len = (K - 1) * dilation
    state_width = state_len + kernel_spec_query_len - 1
    if token_indices is not None:
        T = token_indices.numel()
    if conv_state.shape[1] != C or conv_state.shape[2] < state_width:
        raise ValueError(
            "conv_state must have shape [slots, channels, window], with "
            f"channels={C} and window >= {state_width}"
        )
    state_bs, state_cs, state_ws = conv_state.stride()

    if mode == "decode":
        num_reqs = T
        binary_search_iters = 1
        has_initial_states_arg = has_initial_states is not None
    elif mode == "spec":
        if query_start_loc is None or num_accepted_tokens is None:
            raise ValueError(
                "query_start_loc and num_accepted_tokens are required for spec decode"
            )
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_initial_states_arg = False
    elif mode == "prefill":
        if query_start_loc is None or has_initial_states is None:
            raise ValueError(
                "query_start_loc and has_initial_states are required for prefill"
            )
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        has_initial_states_arg = True
    else:
        raise ValueError(f"Unsupported short-conv mode: {mode}")

    num_warps = 4 if mode == "prefill" else 8
    launch_pdl = current_platform.is_arch_support_pdl()
    # Pure-prefill indices can be a strided block-table column view.
    state_idx_stride = state_indices.stride(0)

    # Constexpr flags eliminate accesses to optional None arguments. Without a
    # token map, state_indices is an unused but device-resident placeholder.
    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        conv_weights,
        residual,
        state_indices,
        query_start_loc,
        num_accepted_tokens,
        has_initial_states,
        token_indices if token_indices is not None else state_indices,
        token_indices is not None,
        num_reqs,
        binary_search_iters,
        state_idx_stride,
        state_bs,
        state_ws,
        state_cs,
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        SPEC_QUERY_LEN=kernel_spec_query_len,
        MODE=mode,
        HAS_INIT=has_initial_states_arg,
        NULL_STATE_ID=NULL_BLOCK_ID,
        launch_pdl=launch_pdl,
        num_warps=num_warps,
    )
    # conv state update is fused with the kernel above for decode
    if mode != "decode":
        _ple_conv_writeback_kernel[(num_reqs, triton.cdiv(C, BLOCK_C))](
            inputs,
            conv_state,
            state_indices,
            query_start_loc,
            num_accepted_tokens,
            has_initial_states,
            token_indices if token_indices is not None else state_indices,
            token_indices is not None,
            state_idx_stride,
            state_bs,
            state_ws,
            state_cs,
            C=C,
            BLOCK_C=BLOCK_C,
            STATE_LEN=state_len,
            SPEC_QUERY_LEN=kernel_spec_query_len,
            STATE_WIDTH=state_width,
            MODE=mode,
            HAS_INIT=has_initial_states_arg,
            NULL_STATE_ID=NULL_BLOCK_ID,
            num_warps=num_warps,
        )
