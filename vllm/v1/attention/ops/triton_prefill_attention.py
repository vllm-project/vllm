# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/sgl-project/sglang/blob/97cb762bb65ebf05025eb342de03c184660427a3/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py
# Changes:
# - Add support for sliding window attention

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for prefill.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/f2a54f0912293f683bf1d1695fd12c4098a5bf82/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py#L1
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING_WINDOW_Q: tl.constexpr,
    SLIDING_WINDOW_K: tl.constexpr,
    Lk: tl.constexpr,
    HEAD_STRIDE_ALIGNED_8: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Head-axis byte offsets. For the packed [seq, heads, dim] layout the head
    # stride equals head_dim; when head_dim is a multiple of 8 but not 16
    # (e.g. 72) Triton's integer-arg auto-specialization does not attach
    # tt.divisibility=8 to stride_*h (its threshold is 16), so AxisInfo treats
    # the Q/K/V global loads as 2-byte aligned and Coalesce emits scalar
    # buffer_load_u16 instead of vectorized buffer_load_b128. Hinting
    # multiple_of(.., 8) on the head offset restores 16-byte alignment so the
    # D-contiguous loads coalesce. The wrapper only sets the flag when the
    # actual runtime strides are %8==0, so it stays sound for non-contiguous
    # Q/K/V views. Mirrors aiter PR #3424.
    off_h_q = cur_head * stride_qh
    off_h_k = cur_kv_head * stride_kh
    off_h_v = cur_kv_head * stride_vh
    off_h_o = cur_head * stride_oh
    if HEAD_STRIDE_ALIGNED_8:
        off_h_q = tl.multiple_of(off_h_q, 8)
        off_h_k = tl.multiple_of(off_h_k, 8)
        off_h_v = tl.multiple_of(off_h_v, 8)
        off_h_o = tl.multiple_of(off_h_o, 8)

    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + off_h_q
        + offs_d[None, :]
    )
    off_k = offs_n[None, :] * stride_kbs + off_h_k + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + off_h_v + offs_d[None, :]

    mask_d = offs_d < Lk

    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :]),
        other=0.0,
    )

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Split-D path: a non-power-of-2 head_dim (e.g. 72) would otherwise force
    # BLOCK_DMODEL = next_pow2(Lk) = 128, so the qk/pv WMMAs run 8 K-passes
    # when only ceil(Lk/16) are needed. Covering D as a power-of-2 main block
    # plus a power-of-2 tail block (e.g. 64 + 16 = 80 for Lk=72) drops that to
    # 5 passes. Constexpr-guarded: BLOCK_DMODEL_TAIL == 0 reproduces the
    # single-block kernel exactly (dead-code eliminated) for power-of-2 dims.
    if BLOCK_DMODEL_TAIL > 0:
        offs_dt = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL_TAIL)
        mask_dt = offs_dt < Lk
        off_qt = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
            + off_h_q
            + offs_dt[None, :]
        )
        off_kt = offs_n[None, :] * stride_kbs + off_h_k + offs_dt[:, None]
        off_vt = offs_n[:, None] * stride_vbs + off_h_v + offs_dt[None, :]
        qt = tl.load(
            Q + off_qt,
            mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_dt[None, :]),
            other=0.0,
        )
        kt_ptrs = K + off_kt
        vt_ptrs = V + off_vt

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if BLOCK_DMODEL_TAIL > 0:
        acc_t = tl.zeros([BLOCK_M, BLOCK_DMODEL_TAIL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    # Calculate the end position for attention computation
    end_n = cur_batch_seq_len

    # Apply causal attention pruning and sliding window attention pruning
    end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M) if IS_CAUSAL else end_n

    # Calculate the start position for backward sliding window
    start_n_limit = 0
    end_n_limit = block_mask * end_n

    for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
        # -- prepare attention mask ----
        # Position indices in the sequence
        pos_q = offs_m[:, None]  # Query positions [BLOCK_M, 1]
        pos_k = start_n + offs_n[None, :]  # Key positions [1, BLOCK_N]

        # Valid sequence mask
        mask = pos_k < cur_batch_seq_len
        # Causal mask
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        # Bidirectional sliding window masks
        sliding_mask_q = (
            pos_q - pos_k <= SLIDING_WINDOW_Q if SLIDING_WINDOW_Q > 0 else None
        )
        sliding_mask_k = (
            pos_k - pos_q <= SLIDING_WINDOW_K if SLIDING_WINDOW_K > 0 else None
        )
        if sliding_mask_q is not None:
            mask &= sliding_mask_q
        if sliding_mask_k is not None:
            mask &= sliding_mask_k

        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=(pos_k < cur_batch_seq_len) & (mask_d[:, None]),
            other=0.0,
        )

        qk = tl.dot(q, k)
        if BLOCK_DMODEL_TAIL > 0:
            kt = tl.load(
                kt_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                mask=(pos_k < cur_batch_seq_len) & (mask_dt[:, None]),
                other=0.0,
            )
            qk += tl.dot(qt, kt)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        if BLOCK_DMODEL_TAIL > 0:
            acc_t = acc_t * alpha[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        if BLOCK_DMODEL_TAIL > 0:
            vt = tl.load(
                vt_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                mask=((start_n + offs_n[:, None]) < cur_batch_seq_len)
                & (mask_dt[None, :]),
                other=0.0,
            )
            acc_t = tl.dot(p, vt, acc_t)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + off_h_o
        + offs_d[None, :]
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_d[None, :])
    )
    if BLOCK_DMODEL_TAIL > 0:
        acc_t = acc_t / l_i[:, None]
        off_o_tail = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
            + off_h_o
            + offs_dt[None, :]
        )
        tl.store(
            Out + off_o_tail,
            acc_t,
            mask=(offs_m[:, None] < cur_batch_seq_len) & (mask_dt[None, :]),
        )


def _is_rdna() -> bool:
    """Check if running on RDNA architecture (gfx11xx, gfx12xx)."""
    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx1x

        return on_gfx1x()
    except ImportError:
        return False


def get_block_size(dtype: torch.dtype, head_dim: int | None = None) -> int:
    """Query-tile size BLOCK_M (also the KV tile unless get_block_n differs)."""
    # SigLIP / Qwen3-VL ViT (head_dim=72) on gfx1151: with the split-D kernel
    # (D covered as 64+16, see _split_head_dim) BM=64 is fastest. The companion
    # KV tile / warps were re-tuned after the head-stride alignment hint
    # vectorized the loads (see get_block_n / get_num_warps: BN=16, NW=2).
    if _is_rdna() and head_dim == 72 and dtype in (torch.bfloat16, torch.float16):
        return 64
    if dtype == torch.float32:
        return 32
    elif current_platform.is_cuda_alike() and current_platform.has_device_capability(
        80
    ):
        return 128
    elif _is_rdna():
        # RDNA (gfx11xx, gfx12xx) performs better with smaller block sizes
        # This matches the tuning used by flash_attn's triton backend for RDNA
        return 32
    else:
        return 64


def get_block_n(dtype: torch.dtype, head_dim: int | None = None) -> int:
    """KV-tile size BLOCK_N. Defaults to BLOCK_M (square tile); the head_dim=72
    split-D ViT path wants an asymmetric BM=64 / BN=16 tile. Re-tuned after the
    head-stride 16B alignment hint (HEAD_STRIDE_ALIGNED_8) vectorized the Q/K/V
    loads and removed register spills: with spills gone the narrower BN=16 / NW=2
    tile wins (BN=32/NW=4/WE=6 was sized for the old spilling codegen)."""
    if _is_rdna() and head_dim == 72 and dtype in (torch.bfloat16, torch.float16):
        return 16
    return get_block_size(dtype, head_dim)


def get_num_warps(head_dim: int) -> int:
    """Get optimal number of warps based on architecture and head dimension."""
    if _is_rdna():
        # RDNA tuning: Block=32, Warps=8 is optimal for ViT attention.
        # Tested on Radeon 8060S (gfx1151) with Qwen2.5-VL-7B.
        # Tested configs: Block in {16,32,64}, Warps in {2,4,8,16}.
        if head_dim == 72:
            # SigLIP / Qwen3-VL, split-D kernel. Re-tuned after the head-stride
            # alignment hint vectorized the loads (0 reg spills): BM=64 + BN=16
            # + NW=2 (no waves_per_eu override) is fastest at B=1,S=3520,H=16,
            # D=72 bf16 (~2.36 ms vs ~2.85 ms at the old NW=4/WE=6 tile).
            return 2
        return 8
    else:
        return 4 if head_dim <= 64 else 8


def get_waves_per_eu(head_dim: int) -> int | None:
    """Per-shape waves_per_eu override for the ViT prefill kernel on RDNA."""
    if _is_rdna() and head_dim == 72:
        # No override after the alignment-hint re-tune: with vectorized loads /
        # 0 spills the default occupancy beats the old waves_per_eu=6 (which was
        # tuned for the spilling BM=32/NW=2 codegen).
        return None
    return None


def _split_head_dim(Lk: int) -> tuple[int, int]:
    """Pick (BLOCK_DMODEL, BLOCK_DMODEL_TAIL) covering head_dim ``Lk``.

    For a power-of-2 Lk, returns (Lk, 0) -- the single-block kernel.
    Otherwise covers Lk with the largest power-of-2 main block (< Lk) plus a
    power-of-2 tail block, so the WMMA K/N extent is ceil to a multiple of 16
    near Lk (e.g. 72 -> 64 + 16 = 80) instead of next_pow2(Lk) = 128.
    """
    npo2 = triton.next_power_of_2(Lk)
    if npo2 == Lk:
        return npo2, 0
    main = npo2 // 2  # largest power of 2 strictly below Lk
    tail = max(16, triton.next_power_of_2(Lk - main))
    return main, tail


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
):
    """
    q, k, v: [b * s, head, head_dim]
    b_start_loc: [b]
    b_seq_len: [b]
    out: [b * s, head, head_dim]
    """
    Lq, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]

    block_m = get_block_size(q.dtype, head_dim=Lk)
    block_n = get_block_n(q.dtype, head_dim=Lk)

    sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
    # rescale with 1/ln(2) for triton exp2
    sm_scale *= RCP_LN2

    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, block_m))
    num_warps = get_num_warps(Lk)

    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0

    waves_per_eu = get_waves_per_eu(Lk)
    extra_kwargs = {}
    if waves_per_eu is not None:
        extra_kwargs["waves_per_eu"] = waves_per_eu

    block_dmodel, block_dmodel_tail = _split_head_dim(Lk)

    # Hint 16-byte alignment on the head axis so the D-contiguous Q/K/V loads
    # coalesce to buffer_load_b128 instead of scalar buffer_load_u16. Checked
    # against the actual runtime strides (not head_dim) so it stays sound for
    # non-contiguous Q/K/V/O views. See _fwd_kernel / aiter PR #3424.
    head_stride_aligned_8 = (
        q.stride(1) % 8 == 0
        and k.stride(1) % 8 == 0
        and v.stride(1) % 8 == 0
        and o.stride(1) % 8 == 0
    )

    _fwd_kernel[grid](
        q,
        k,
        v,
        sm_scale,
        b_start_loc,
        b_seq_len,
        o,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_M=block_m,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DMODEL_TAIL=block_dmodel_tail,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW_Q=sliding_window_q,
        SLIDING_WINDOW_K=sliding_window_k,
        num_warps=num_warps,
        num_stages=1,
        Lk=Lk,
        HEAD_STRIDE_ALIGNED_8=head_stride_aligned_8,
        **extra_kwargs,
    )
