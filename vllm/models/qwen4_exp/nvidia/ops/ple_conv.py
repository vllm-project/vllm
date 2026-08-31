# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused PLE dilated short-convolution kernels.

For each token and channel, the output kernel computes

  out[t] = residual[t] + silu(sum_k w[k] * hist[j + k*dilation])
  hist[m] = state[sid, slot_off + m, c]  if m < state_len
            x[qsl[r] + m - state_len, c] otherwise

State may use either [slot, window, channel] or [slot, channel, window]
layout. Decode, speculative decode, and prefill differ in their state-window
offset and write-back rule. The write-back runs separately because it
overwrites state read by the output kernel. Slot zero is never updated.
"""

from typing import Literal

import torch

from vllm.triton_utils import tl, triton

NULL_BLOCK_ID = tl.constexpr(0)

BLOCK_C = 512
NUM_WARPS = 8


@triton.jit(do_not_specialize=["num_reqs", "bs_iters"])
def _ple_conv_kernel(
    x_ptr,
    state_ptr,
    w_ptr,
    residual_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    num_reqs,
    bs_iters,
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
):
    t = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

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
            slot_off = tl.minimum(tl.maximum(num_acc - 1, 0), SPEC_QUERY_LEN - 1).to(
                tl.int32
            )
        else:
            slot_off = tl.full([], 0, tl.int32)

    sid = tl.load(state_idx_ptr + r).to(tl.int64)
    state_ok = sid != NULL_BLOCK_ID
    sid_safe = tl.where(state_ok, sid, 0)
    if HAS_INIT:
        has_init = tl.load(has_init_ptr + r, mask=state_ok, other=0) != 0
    else:
        has_init = state_ok
    if MODE == "spec":
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
        input_tap = tl.load(
            x_ptr + (q_start + h - STATE_LEN) * C + c_offs,
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

    conv = acc.to(residual_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    conv_output = tl.where(out_ok, y, 0.0).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + t * C + c_offs,
        mask=c_mask,
        other=0.0,
    )
    tl.store(
        residual_ptr + t * C + c_offs,
        residual + conv_output,
        mask=c_mask,
    )


@triton.jit(do_not_specialize=["num_reqs"])
def _ple_conv_writeback_kernel(
    x_ptr,
    state_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    num_reqs,
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
):
    r = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sid = tl.load(state_idx_ptr + r).to(tl.int64)
    state_ok = sid != NULL_BLOCK_ID
    if not state_ok:
        return

    if MODE == "decode":
        q_start = r
        qlen = 1
        slot_off = tl.full([], 0, tl.int32)
        shift = 1
    else:
        q_start = tl.load(qsl_ptr + r)
        q_end = tl.load(qsl_ptr + r + 1)
        qlen = (q_end - q_start).to(tl.int32)
        if MODE == "spec":
            num_acc = tl.load(num_acc_ptr + r)
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
        do_write = i < STATE_LEN + qlen - 1 if MODE == "spec" else True
        state_value = tl.load(
            base_state + (slot_off + m) * state_ws + c_offs * state_cs,
            mask=c_mask & from_state & src_ok,
            other=0.0,
        )
        input_value = tl.load(
            x_ptr + (q_start + m - STATE_LEN) * C + c_offs,
            mask=c_mask & (~from_state) & do_write,
            other=0.0,
        )
        value = tl.where(from_state, state_value, input_value)
        tl.store(
            base_state + i * state_ws + c_offs * state_cs,
            value,
            mask=c_mask & do_write,
        )


def _conv_dimensions(
    inputs: torch.Tensor,
    conv_weights: torch.Tensor,
    dilation: int,
    spec_query_len: int,
) -> tuple[int, int, int, int, int]:
    num_tokens, channels = inputs.shape
    kernel_size = conv_weights.shape[1]
    state_len = (kernel_size - 1) * dilation
    state_width = state_len + spec_query_len - 1
    return num_tokens, channels, kernel_size, state_len, state_width


def _get_conv_state_strides(
    conv_state: torch.Tensor,
    channels: int,
    state_width: int,
) -> tuple[int, int, int]:
    is_sd_layout = (
        conv_state.shape[1] >= state_width and conv_state.shape[2] == channels
    )
    is_ds_layout = (
        conv_state.shape[1] == channels and conv_state.shape[2] >= state_width
    )
    if is_sd_layout == is_ds_layout:
        raise ValueError(
            "conv_state must have shape [slots, window, channels] or "
            "[slots, channels, window]"
        )
    state_dim, channel_dim = (1, 2) if is_sd_layout else (2, 1)
    return (
        conv_state.stride(0),
        conv_state.stride(state_dim),
        conv_state.stride(channel_dim),
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
) -> None:
    """Run short convolution in place for decode, spec decode, or prefill."""
    kernel_spec_query_len = spec_query_len if mode == "spec" else 1
    T, C, K, state_len, state_width = _conv_dimensions(
        inputs, conv_weights, dilation, kernel_spec_query_len
    )
    state_bs, state_ws, state_cs = _get_conv_state_strides(conv_state, C, state_width)

    if mode == "decode":
        query_start_loc_ptr = state_indices
        num_accepted_tokens_ptr = state_indices
        has_initial_states_ptr = (
            has_initial_states if has_initial_states is not None else state_indices
        )
        num_reqs = T
        binary_search_iters = 1
        num_writeback_reqs = T
        has_initial_states_arg = has_initial_states is not None
    elif mode == "spec":
        if query_start_loc is None or num_accepted_tokens is None:
            raise ValueError(
                "query_start_loc and num_accepted_tokens are required for spec decode"
            )
        query_start_loc_ptr = query_start_loc
        num_accepted_tokens_ptr = num_accepted_tokens
        has_initial_states_ptr = query_start_loc
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        num_writeback_reqs = num_reqs
        has_initial_states_arg = False
    elif mode == "prefill":
        if query_start_loc is None or has_initial_states is None:
            raise ValueError(
                "query_start_loc and has_initial_states are required for prefill"
            )
        query_start_loc_ptr = query_start_loc
        num_accepted_tokens_ptr = query_start_loc
        has_initial_states_ptr = has_initial_states
        num_reqs = state_indices.numel()
        binary_search_iters = max(num_reqs, 1).bit_length()
        num_writeback_reqs = num_reqs
        has_initial_states_arg = True
    else:
        raise ValueError(f"Unsupported short-conv mode: {mode}")

    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        conv_weights,
        residual,
        state_indices,
        query_start_loc_ptr,
        num_accepted_tokens_ptr,
        has_initial_states_ptr,
        num_reqs,
        binary_search_iters,
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
        num_warps=NUM_WARPS,
    )
    _ple_conv_writeback_kernel[(num_writeback_reqs, triton.cdiv(C, BLOCK_C))](
        inputs,
        conv_state,
        state_indices,
        query_start_loc_ptr,
        num_accepted_tokens_ptr,
        has_initial_states_ptr,
        num_reqs,
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
        num_warps=NUM_WARPS,
    )
