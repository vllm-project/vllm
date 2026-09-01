# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch

from vllm.triton_utils import tl, triton


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None
        or args["block_table"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
        "HAS_TABLE": lambda args: args["block_table"] is not None,
        "HAS_PACKED_ANCHORS": lambda args: args["packed_anchors"] is not None,
        "PACKED": lambda args: args["mixed_qkv"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    mixed_qkv,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_indices_output,
    num_accepted_tokens,
    block_table,
    read_anchor,
    write_anchor,
    packed_anchors,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_indices_output_seq: tl.constexpr,
    stride_block_table_seq: tl.constexpr,
    stride_mixed_qkv_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    HAS_TABLE: tl.constexpr,
    HAS_PACKED_ANCHORS: tl.constexpr,
    PACKED: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        # no tokens to process for this sequence
        return

    if HAS_PACKED_ANCHORS:
        # Packed (read, write) anchor pair: one 64-bit load, low word = read
        # anchor, high word = write anchor. The write-side block-table base is
        # hoisted out of the token loop (loop-invariant).
        b_pair = tl.load(packed_anchors + i_n)
        o_anchor_r = b_pair & 0xFFFFFFFF
        p_bt_w = block_table + i_n * stride_block_table_seq + (b_pair >> 32)

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    if PACKED:
        # q/k/v live interleaved per token in a single packed buffer; the
        # per-tensor offsets mirror fused_recurrent_gated_delta_rule_packed_decode.
        p_mixed = mixed_qkv + bos * stride_mixed_qkv_tok
        p_q = p_mixed + i_h * K + o_k
        p_k = p_mixed + (H * K) + i_h * K + o_k
        p_v = p_mixed + (2 * H * K) + i_hv * V + o_v
    else:
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    if not IS_KDA:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv
    else:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k

    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            # Load state index and check for invalid entries
            if HAS_PACKED_ANCHORS:
                state_idx = tl.load(
                    block_table + i_n * stride_block_table_seq + o_anchor_r + i_t
                ).to(tl.int64)
            elif HAS_TABLE:
                # Derive the read slot in-kernel from the block table:
                # block_table[i_n, read_anchor[i_n] + i_t].
                o_r = tl.load(read_anchor + i_n).to(tl.int64)
                state_idx = tl.load(
                    block_table + i_n * stride_block_table_seq + o_r + i_t
                ).to(tl.int64)
            else:
                state_idx = tl.load(
                    ssm_state_indices + i_n * stride_indices_seq + i_t
                ).to(tl.int64)
            # Skip if state index is invalid (NULL_BLOCK_ID=0)
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        x = tl.load(p_a).to(tl.float32) + tl.load(p_dt_bias).to(tl.float32)
        softplus_x = tl.where(
            beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x

        # compute beta_output = sigmoid(b)
        b_beta = tl.sigmoid(b_b.to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_h *= tl.exp(b_g)
        else:
            b_h *= tl.exp(b_g[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # keep the states for multi-query tokens
        if INPLACE_FINAL_STATE:
            # Load state index and check for invalid entries
            if HAS_PACKED_ANCHORS:
                final_state_idx = tl.load(p_bt_w + i_t).to(tl.int64)
            elif HAS_TABLE:
                # Derive the write slot in-kernel from the block table:
                # block_table[i_n, write_anchor[i_n] + i_t].
                o_w = tl.load(write_anchor + i_n).to(tl.int64)
                final_state_idx = tl.load(
                    block_table + i_n * stride_block_table_seq + o_w + i_t
                ).to(tl.int64)
            else:
                final_state_idx = tl.load(
                    ssm_state_indices_output + i_n * stride_indices_output_seq + i_t
                ).to(tl.int64)
            # Only store if state index is valid (not NULL_BLOCK_ID=0)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Update pointers for next timestep
        if PACKED:
            p_q += stride_mixed_qkv_tok
            p_k += stride_mixed_qkv_tok
            p_v += stride_mixed_qkv_tok
        else:
            p_q += H * K
            p_k += H * K
            p_v += HV * V
        p_o += HV * V
        p_b += HV
        p_a += HV


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    ssm_state_indices_output: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    read_anchor: torch.Tensor | None = None,
    write_anchor: torch.Tensor | None = None,
    packed_anchors: torch.Tensor | None = None,
    mixed_qkv: torch.Tensor | None = None,
    num_qk_heads: int | None = None,
    head_qk_dim: int | None = None,
    num_v_heads: int | None = None,
    head_v_dim: int | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating
    computation and the recurrent delta rule update for better performance.

    When ``block_table`` (2D, per-seq rows) plus per-seq ``read_anchor`` /
    ``write_anchor`` are given instead of ``ssm_state_indices``, the kernel
    derives its own state slots in-kernel: read slot
    ``block_table[i, read_anchor[i] + i_t]`` (with ``i_t`` selected by
    ``num_accepted_tokens`` in spec mode) and write slot
    ``block_table[i, write_anchor[i] + i_t]`` per token — equivalent to the
    host-side ``block_table.gather(1, anchor.unsqueeze(1) + arange(T))``.
    The two anchors may instead be given packed as ``packed_anchors``
    (``(num_seqs, 2)`` int32, ``[:, 0]`` read / ``[:, 1]`` write): the kernel
    then fetches both with a single 64-bit load and hoists the write-side
    table base out of the token loop. Results are identical.

    When ``mixed_qkv`` (2D ``(num_tokens, 2 * key_dim + value_dim)``, the
    packed conv output) is given instead of ``q``/``k``/``v``, the kernel
    reads q/k/v straight from the packed buffer with per-tensor offsets
    (same addressing as ``fused_recurrent_gated_delta_rule_packed_decode``),
    skipping the host-side rearrange copies. The head geometry cannot be
    inferred from a packed buffer, so ``num_qk_heads``/``head_qk_dim``/
    ``num_v_heads``/``head_v_dim`` are required in this mode.
    """
    if mixed_qkv is not None:
        assert q is None and k is None and v is None, (
            "mixed_qkv and q/k/v are mutually exclusive"
        )
        assert mixed_qkv.stride(-1) == 1, (
            "mixed_qkv must be contiguous in the last dim"
        )
        assert (
            num_qk_heads is not None
            and head_qk_dim is not None
            and num_v_heads is not None
            and head_v_dim is not None
        ), "head geometry kwargs are required with mixed_qkv"
        H, K, HV, V = num_qk_heads, head_qk_dim, num_v_heads, head_v_dim
        if cu_seqlens is not None:
            B, T = 1, mixed_qkv.shape[0]
        else:
            B, T = mixed_qkv.shape[0], 1
        stride_mixed_qkv_tok = mixed_qkv.stride(0)
    else:
        B, T, H, K, V = *k.shape, v.shape[-1]
        HV = v.shape[2]
        stride_mixed_qkv_tok = 0
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if cu_seqlens is not None and q is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]}"
            f" when using `cu_seqlens`. Please flatten variable-length"
            f" inputs before processing."
        )
    if scale is None:
        scale = K**-0.5
    else:
        assert scale > 0, "scale must be positive"

    if mixed_qkv is not None:
        o = mixed_qkv.new_empty(NK, B, T, HV, V)
    else:
        o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = o.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    # The kernel indexes both 2-D index tensors with an implicit token stride of
    # 1 (`... + i_n * stride_seq + i_t`), so a non-contiguous last dim (e.g. a
    # transposed/strided view) would silently read wrong offsets. Normalize to
    # contiguous here — a no-op on the current (gather-produced, contiguous)
    # all-mode spec path, a safety net otherwise.
    if ssm_state_indices is not None and ssm_state_indices.stride(-1) != 1:
        ssm_state_indices = ssm_state_indices.contiguous()
    if (ssm_state_indices_output is not None
            and ssm_state_indices_output.stride(-1) != 1):
        ssm_state_indices_output = ssm_state_indices_output.contiguous()

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    # all-mode dual-anchor: write the final state via a separate output index tensor
    # while reading h0 via ssm_state_indices. Defaults to in-place (output == input).
    if ssm_state_indices_output is None:
        ssm_state_indices_output = ssm_state_indices
    stride_indices_output_seq = (
        ssm_state_indices_output.stride(0)
        if ssm_state_indices_output is not None
        else 1
    )

    # Direct-block-table mode: the kernel derives its own read/write slots
    # from block_table + per-seq anchors (no host-gathered index tensors).
    if block_table is not None:
        assert ssm_state_indices is None and ssm_state_indices_output is None, (
            "block_table and ssm_state_indices are mutually exclusive"
        )
        if packed_anchors is not None:
            assert read_anchor is None and write_anchor is None, (
                "packed_anchors and read_anchor/write_anchor are "
                "mutually exclusive"
            )
            assert (
                packed_anchors.ndim == 2
                and packed_anchors.shape[-1] == 2
                and packed_anchors.dtype == torch.int32
                and packed_anchors.stride(-1) == 1
                and packed_anchors.stride(0) == 2
            ), "packed_anchors must be a contiguous (num_seqs, 2) int32 tensor"
            # Reinterpret each (read, write) int32 pair as one int64 so the
            # kernel fetches both anchors with a single load.
            packed_anchors = packed_anchors.view(torch.int64).squeeze(-1)
        else:
            assert read_anchor is not None and write_anchor is not None
        assert block_table.stride(-1) == 1
        stride_block_table_seq = block_table.stride(0)
    else:
        assert packed_anchors is None, "packed_anchors requires block_table"
        stride_block_table_seq = 0

    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        q=q.contiguous() if q is not None else None,
        k=k.contiguous() if k is not None else None,
        v=v.contiguous() if v is not None else None,
        mixed_qkv=mixed_qkv,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_indices_output=ssm_state_indices_output,
        num_accepted_tokens=num_accepted_tokens,
        block_table=block_table,
        read_anchor=read_anchor,
        write_anchor=write_anchor,
        packed_anchors=packed_anchors,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_indices_output_seq=stride_indices_output_seq,
        stride_block_table_seq=stride_block_table_seq,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
