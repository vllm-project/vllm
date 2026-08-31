# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused PLE dilated short-convolution kernels.

For each token and channel, the output kernel computes

  out[t] = silu(sum_k w[k] * hist[j + k*dilation])
  hist[m] = state[sid, slot_off + m, c]  if m < state_len
            x[qsl[r] + m - state_len, c] otherwise

State uses the [slot, window, channel] layout. Decode, speculative decode,
and prefill differ only in their state-window offset and write-back rule.
The write-back runs as a separate kernel because it overwrites state read by
the output kernel. Slot zero is reserved and is never updated.
"""

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
    out_ptr,
    state_idx_ptr,
    qsl_ptr,
    num_acc_ptr,
    has_init_ptr,
    num_reqs,
    bs_iters,
    state_bs,
    state_ws,
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
            base_state + (h + slot_off) * state_ws + c_offs,
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

    conv = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    y = conv * tl.sigmoid(conv)
    y = tl.where(out_ok, y, 0.0)
    tl.store(
        out_ptr + t * C + c_offs,
        y.to(out_ptr.dtype.element_ty),
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
            base_state + (slot_off + m) * state_ws + c_offs,
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
            base_state + i * state_ws + c_offs,
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


def ple_conv_decode(
    x_d: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_d: torch.Tensor,
    has_initial_states_d: torch.Tensor | None,
    dilation: int,
) -> torch.Tensor:
    """Decode path: x_d [T, C], conv_state [S, W, C] (SD layout)."""
    T, C, K, state_len, state_width = _conv_dimensions(x_d, conv_weights, dilation, 1)
    out = torch.empty_like(x_d)
    has_init_ptr = (
        has_initial_states_d
        if has_initial_states_d is not None
        else state_indices_tensor_d  # unused dummy pointer
    )
    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        x_d,
        conv_state,
        conv_weights,
        out,
        state_indices_tensor_d,
        state_indices_tensor_d,
        state_indices_tensor_d,
        has_init_ptr,
        T,
        1,
        conv_state.stride(0),
        conv_state.stride(1),
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        SPEC_QUERY_LEN=1,
        MODE="decode",
        HAS_INIT=has_initial_states_d is not None,
        num_warps=NUM_WARPS,
    )
    _ple_conv_writeback_kernel[(T, triton.cdiv(C, BLOCK_C))](
        x_d,
        conv_state,
        state_indices_tensor_d,
        state_indices_tensor_d,
        state_indices_tensor_d,
        has_init_ptr,
        T,
        conv_state.stride(0),
        conv_state.stride(1),
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        SPEC_QUERY_LEN=1,
        STATE_WIDTH=state_width,
        MODE="decode",
        HAS_INIT=has_initial_states_d is not None,
        num_warps=NUM_WARPS,
    )
    return out


def ple_conv_spec(
    x_spec: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    spec_state_indices_tensor: torch.Tensor,
    spec_query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    dilation: int,
    spec_query_len: int,
) -> torch.Tensor:
    """Run speculative decode over inputs of shape [tokens, channels]."""
    T, C, K, state_len, state_width = _conv_dimensions(
        x_spec, conv_weights, dilation, spec_query_len
    )
    R = spec_state_indices_tensor.numel()
    out = torch.empty_like(x_spec)
    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        x_spec,
        conv_state,
        conv_weights,
        out,
        spec_state_indices_tensor,
        spec_query_start_loc,
        num_accepted_tokens,
        spec_query_start_loc,  # unused dummy pointer
        R,
        max(R, 1).bit_length(),
        conv_state.stride(0),
        conv_state.stride(1),
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        SPEC_QUERY_LEN=spec_query_len,
        MODE="spec",
        HAS_INIT=False,
        num_warps=NUM_WARPS,
    )
    _ple_conv_writeback_kernel[(R, triton.cdiv(C, BLOCK_C))](
        x_spec,
        conv_state,
        spec_state_indices_tensor,
        spec_query_start_loc,
        num_accepted_tokens,
        spec_query_start_loc,  # unused dummy pointer
        R,
        conv_state.stride(0),
        conv_state.stride(1),
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        SPEC_QUERY_LEN=spec_query_len,
        STATE_WIDTH=state_width,
        MODE="spec",
        HAS_INIT=False,
        num_warps=NUM_WARPS,
    )
    return out


def ple_conv_prefill(
    x_p: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    has_initial_states_p: torch.Tensor,
    query_start_loc_p: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """Run prefill over flat variable-length input sequences."""
    T, C, K, state_len, state_width = _conv_dimensions(x_p, conv_weights, dilation, 1)
    R = state_indices_tensor_p.numel()
    out = torch.empty_like(x_p)
    has_init_ptr = has_initial_states_p
    _ple_conv_kernel[(T, triton.cdiv(C, BLOCK_C))](
        x_p,
        conv_state,
        conv_weights,
        out,
        state_indices_tensor_p,
        query_start_loc_p,
        query_start_loc_p,
        has_init_ptr,
        R,
        max(R, 1).bit_length(),
        conv_state.stride(0),
        conv_state.stride(1),
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        DILATION=dilation,
        KERNEL_SIZE=K,
        SPEC_QUERY_LEN=1,
        MODE="prefill",
        HAS_INIT=True,
        num_warps=NUM_WARPS,
    )
    _ple_conv_writeback_kernel[(R, triton.cdiv(C, BLOCK_C))](
        x_p,
        conv_state,
        state_indices_tensor_p,
        query_start_loc_p,
        query_start_loc_p,
        has_init_ptr,
        R,
        conv_state.stride(0),
        conv_state.stride(1),
        C=C,
        BLOCK_C=BLOCK_C,
        STATE_LEN=state_len,
        SPEC_QUERY_LEN=1,
        STATE_WIDTH=state_width,
        MODE="prefill",
        HAS_INIT=True,
        num_warps=NUM_WARPS,
    )
    return out
