# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary gfx942 fallback for AITER's fp8_mqa_logits kernel.

This module vendors AITER's Triton fp8_mqa_logits kernel with the gfx942
tile-size workaround from ROCm/aiter#3257. It is used only while vLLM's
pinned AITER version lacks that fix.

TODO: Remove this vendored copy once vLLM pins an AITER version that includes
ROCm/aiter#3257 bugfix for gfx942.
"""

import torch

from vllm.triton_utils import tl, triton

# gfx942 (MI300X) has 64 KiB of LDS per CU. We accept the default
# (BLOCK_KV=128, num_stages=2) tile only when *both* of these hold:
#
# 1. Occupancy gate. With num_warps=4 we target two workgroups co-resident
#    on a CU -> per-WG LDS budget = 32 KiB. Triton keeps Q in registers
#    (loop-invariant) and the fp32 scores accumulator in VGPRs (heavy
#    VALU), so only the pipelined KV tile lives in LDS. A 0.9 safety
#    factor leaves headroom for the pipeline bookkeeping the compiler
#    adds on top of the raw tile (measured: 512 B).
#
# 2. Hardware ceiling. Defensive upper bound that also counts Q and
#    scores against the 64 KiB CU limit, in case a Triton version (older
#    or future) decides to spill them to LDS. False positives here only
#    shrink the tile; false negatives are JIT-aborts, so we lean
#    conservative.
#
# LDS ARITHMETIC (corrected -- the original model charged the KV tile at
# 2 bytes/element, which describes an fp16-expanded tile, not the native
# fp8 one this kernel actually feeds to the MFMA):
#
#   Old:  kv_bytes = head_size * BLOCK_KV * NUM_STAGES
#                  = 128 * 128 * 2 = 32768 B  >  28.8 KiB budget -> REJECT
#
#   That expression used NUM_STAGES=2 in the slot where a bytes-per-
#   element factor belongs. It happens to equal the right answer for a
#   single-buffered *fp16* tile -- i.e. the software fp8->fp16 expansion
#   path -- so it modelled the wrong operand width. The KV operand is one
#   fp8 byte per element, and Triton's stream pipeliner does not put a
#   second full copy of the tile in LDS for this single-dot loop (the
#   second stage lives in registers); only a small amount of pipeline
#   bookkeeping is added on top of the one resident tile:
#
#   New:  kv_bytes = head_size * BLOCK_KV * KV_ELEM_BYTES + PIPELINE_SLACK
#                  = 128 * 128 * 1 + 512 = 16896 B  <  28.8 KiB -> ACCEPT
#
#   16896 B is not an estimate: it is exactly what this Triton build
#   reports in `kernel.metadata.shared` for (BLOCK_KV=128, num_stages=2,
#   H=64, D=128) -- one 128x128 fp8 tile (16384 B) plus 512 B of slack.
#   So the DSv4 indexer shape fits the default tile with ~12 KiB of
#   headroom, and the only reason it was being rejected was the
#   fp16-expansion assumption.
#
#   The gate still does its job at the next size up: BLOCK_KV=256 needs
#   256*128*1 + 512 = 33280 B > 28.8 KiB and is correctly rejected (it
#   also measures slower, 9.82 ms vs 7.30 ms at M=8000).
_GFX942_CU_LDS_BYTES = 64 * 1024
_GFX942_PER_WG_LDS_BUDGET_BYTES = _GFX942_CU_LDS_BYTES * 9 // 20  # ~28.8 KiB
# The KV operand reaches the MFMA as fp8 (1 byte/element) on gfx942.
_GFX942_KV_ELEM_BYTES = 1
# Measured LDS the Triton stream pipeliner adds on top of the raw tile.
_GFX942_PIPELINE_SLACK_BYTES = 512


def _gfx942_default_tile_fits_lds(num_heads: int, head_size: int) -> bool:
    """Return True iff (BLOCK_KV=128, num_stages=2) fits in MI300X LDS."""
    BLOCK_KV = 128
    kv_bytes = (
        head_size * BLOCK_KV * _GFX942_KV_ELEM_BYTES + _GFX942_PIPELINE_SLACK_BYTES
    )
    scores_bytes = num_heads * BLOCK_KV * 4
    q_bytes = num_heads * head_size
    fits_occupancy = kv_bytes < _GFX942_PER_WG_LDS_BUDGET_BYTES
    fits_hardware = q_bytes + kv_bytes + scores_bytes <= _GFX942_CU_LDS_BYTES
    return fits_occupancy and fits_hardware


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # block sizes
    BLOCK_KV: tl.constexpr,
    OOW_FILL: tl.constexpr,
):
    row_id = tl.program_id(0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )

    # ---- native gfx942 fp8 MFMA path -------------------------------------
    # Operands arrive as OCP e4m3 (`torch.float8_e4m3fn`). CDNA3 has a native
    # fp8 MFMA only for the *fnuz* encodings, so without this Triton emits a
    # ~2500-instruction software fp8->fp16 upconvert per KV tile and runs
    # `v_mfma_f32_32x32x8_f16`. e4m3fn and e4m3fnuz share the exact same 1-4-3
    # bit layout and differ only in exponent bias (7 vs 8), so a raw BITCAST
    # reinterprets every value as exactly v/2 (bit-exact, denormals included).
    # One exception: byte 0x80 is -0.0 in OCP e4m3 but NaN in e4m3fnuz, so it
    # must be scrubbed to 0x00 first.
    #
    # Q is loop-invariant (loaded once per program) so scrubbing it here is
    # free. Two bitcast operands make the dot produce (q.k)/4; the correcting
    # *4 is folded into the loop-invariant per-head `w_block` below. That is
    # exact because 1/4 is a positive power of two, so relu(x/4) == relu(x)/4:
    #     relu((q.k/4) * kv_scale) * (4*w) == relu((q.k) * kv_scale) * w.
    q_bits = tl.load(q_ptrs, cache_modifier=".cg").to(tl.uint8, bitcast=True)
    q_bits = tl.where(q_bits == 0x80, 0, q_bits).to(tl.uint8)
    q_block = q_bits.to(tl.float8e4b8, bitcast=True)

    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    # `* 4.0` undoes the 2x-per-operand bias shift of the fnuz reinterpretation.
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32) * 4.0

    # Load start/end for each row in this block
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    # ---- -inf epilogue for the out-of-window lanes of this row -----------
    # `logits` is allocated with torch.empty (no host prefill), so each
    # program is responsible for writing -inf to the part of its own row that
    # falls outside [start_ind, end_ind). See the launcher for the A/B.
    if OOW_FILL:
        ninf = tl.full([BLOCK_KV], float("-inf"), tl.float32)
        for off in tl.range(0, start_ind, BLOCK_KV):
            cols = off + tl.arange(0, BLOCK_KV)
            tl.store(
                logits_row_ptrs + cols * stride_logits_k,
                ninf,
                mask=cols < start_ind,
            )
        for off in tl.range(end_ind, seq_len_kv, BLOCK_KV):
            cols = off + tl.arange(0, BLOCK_KV)
            tl.store(
                logits_row_ptrs + cols * stride_logits_k,
                ninf,
                mask=cols < seq_len_kv,
            )

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )

    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        # K needs the same 0x80 (OCP -0.0 / fnuz NaN) scrub Q gets, but K is
        # re-read once per OUTPUT ROW while it is only [N, D] bytes, so doing
        # it here re-scrubs identical bytes M times. It is hoisted to a single
        # host-side `torch.clamp_min(k_fp8.view(int8), -127)` pass in the
        # launcher (same 0x80 -> 0x81 mapping, bit-identical numerics), which
        # deleted 258 of the 884 inner-loop instructions (29 %: 128
        # v_max_i16_sdwa + 96 v_or_b32_sdwa + 34 v_lshrrev of byte-lane fixup).
        # So here we just load and bitcast straight to fnuz.
        kv_block = tl.load(kv_ptrs).to(tl.float8e4b8, bitcast=True)
        kv_scales = tl.load(kv_scales_ptrs)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        # both operands are fnuz -> `v_mfma_f32_*_fp8_fp8` (no upconvert)
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        # Multiply by kv_scales (broadcast along rows)
        scores = scores * kv_scales[None, :]
        # ReLU
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
        scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked load
    kv_col_mask = kv_col_offsets < end_ind
    # Same hoist as the main loop: K is already scrubbed on the host.
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0).to(
        tl.float8e4b8, bitcast=True
    )
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    # Multiply by kv_scales (broadcast along rows)
    scores = scores * kv_scales[None, :]
    # ReLU
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
    scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_gfx942(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits on MI300X (gfx942) using the vendored kernel.

    Drop-in replacement for ``aiter.ops.triton.attention.fp8_mqa_logits.
    fp8_mqa_logits`` on MI300X. Selects ``(BLOCK_KV, num_stages)`` based on
    whether the default tile fits within the 64 KiB LDS budget of a gfx942
    CU (see module docstring).

    Args:
        q: Query tensor of shape ``[M, H, D]``, FP8 dtype.
        k_fp8: Key tensor of shape ``[N, D]``, FP8 dtype.
        kv_scales: K scales of shape ``[N]`` (or ``[N, 1]`` -- viewed as
            ``[N]``), float32.
        weights: Per-head weights of shape ``[M, H]``, float32.
        cu_starts: Start indices (inclusive) of shape ``[M]``, int32.
        cu_ends: End indices (exclusive) of shape ``[M]``, int32.

    Returns:
        Logits of shape ``[M, N]``, float32 -- positions outside
        ``[cu_starts[i], cu_ends[i])`` for row ``i`` are pre-filled with
        ``-inf`` so the caller can run a top-k without masking.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    # The kernel walks ``kv_scales`` as a 1-D contiguous array of size N
    # (it indexes by ``kv_scales_ptr + kv_col_offsets``). The vLLM caller
    # passes a ``[N, 4]`` uint8 view-cast-to-float32 which lands as
    # ``[N, 1]`` contiguous -- byte-identical to ``[N]`` -- but flatten
    # explicitly to keep the kernel's pointer arithmetic intent clear.
    kv_scales_1d = kv_scales.reshape(-1)

    # ---- one-shot host-side fnuz NaN scrub of K --------------------------
    # The kernel bitcasts fp8 operands from OCP e4m3 to e4m3fnuz to reach the
    # native `v_mfma_f32_16x16x32_fp8_fp8`. The two encodings share a bit
    # layout and differ only in bias, EXCEPT that byte 0x80 is -0.0 in OCP but
    # NaN in fnuz, and one NaN byte poisons its whole output column. Viewed as
    # int8, 0x80 == INT8_MIN, so `clamp_min(-127)` neutralises it (0x80 ->
    # 0x81) and is the identity on every other byte.
    #
    # K is [N, D] but was being scrubbed once per output row inside the loop,
    # i.e. M times over the same bytes -- 29 % of the inner-loop instructions.
    # Doing it once here costs one [N, D]-byte elementwise pass (a few us) and
    # is measured at 1.19x end-to-end (0.5673 -> 0.4848 ms at M=4096) with the
    # clamp itself inside the timed region.
    #
    # `clamp_min` returns a NEW tensor -- never mutate the caller's k_fp8 in
    # place, the harness reuses inputs across iterations. The int8 dtype-view
    # requires the last dim to be contiguous, so densify first if needed; the
    # strides are re-read from the new tensor below.
    if k_fp8.stride(-1) != 1:
        k_fp8 = k_fp8.contiguous()
    k_fp8 = torch.clamp_min(k_fp8.view(torch.int8), -127).view(k_fp8.dtype)

    # Positions outside [cu_starts[i], cu_ends[i]) must read ``-inf`` -- this
    # matches AITER's ``fp8_mqa_logits`` semantics and is what the top-k
    # consumer expects. AITER pre-fills the whole [M, N] output with
    # ``torch.full(-inf)``; that costs a full M*N fp32 HBM write plus a
    # FillFunctor dispatch INSIDE the timed region (measured 8.7 / 20.4 /
    # 55.9 / 52.0 us for cases 0-3, ~1.5-3 % end-to-end now that the kernel
    # itself is 7x faster), and every element the kernel writes is written
    # twice.
    #
    # Instead allocate uninitialised and let each program -inf-fill the
    # out-of-window lanes of its OWN row (``OOW_FILL``): row i owns
    # [0, start_ind) and [end_ind, seq_len_kv). Same total store traffic in
    # the worst case, but it is fused into the kernel's own stores, needs no
    # second dispatch, and no host-side fullness test (a
    # ``cu_starts.amax().item()`` probe is a device->host sync costing 60-67 us
    # flat, strictly worse than the fill and not HIP-graph safe).
    #
    # Measured (full-benchmark geomean over 3 runs each): torch.full ->
    # 0.7149 / 0.7095 / 0.7162 ms, kernel epilogue -> 0.7064 / 0.7056 /
    # 0.6998 ms. A win on every case, so the prefill is retired.
    logits = torch.empty((seq_len, seq_len_kv), dtype=torch.float32, device=q.device)

    if _gfx942_default_tile_fits_lds(num_heads, head_size):
        block_kv = 128
        num_stages = 2
    else:
        block_kv = 64
        num_stages = 1

    # MFMA instruction shape. AITER keys this off `seq_len` (32 above 1024,
    # 16 at or below), but the right axis is the *dot shape*, not the
    # sequence length: nonkdim only selects between v_mfma_*_16x16x* and
    # v_mfma_*_32x32x*, and the dot here is [NUM_HEADS, HEAD_SIZE] x
    # [HEAD_SIZE, BLOCK_KV] -- seq_len does not appear in it at all.
    #
    # For the DSv4 indexer shape (H=64, D=128) the 16x16 instruction wins
    # at every measured seq_len, because the 32x32 form needs a larger
    # accumulator and spends more AGPR/VGPR for the same MACs. Measured on
    # gfx942 at BLOCK_KV=128, num_stages=2:
    #     M=8000   nonkdim=32 -> 7.99 ms      nonkdim=16 -> 7.30 ms
    # i.e. the `seq_len > 1024` branch was picking the slower instruction
    # on exactly the shapes where it costs the most. Restrict the 32 choice
    # to shapes we have not characterised, and take 16 whenever the tile
    # dimensions are the ones it was measured on.
    if num_heads <= 64 and head_size <= 128:
        matrix_instr_nonkdim = 16
    elif seq_len <= 1024:
        matrix_instr_nonkdim = 16
    else:
        matrix_instr_nonkdim = 32

    # waves_per_eu trims the VGPR allocation to force more waves resident
    # per EU. AITER ships 2; 3 is uniformly faster on this shape once the
    # tile is right (measured, all four benchmark cases at BLOCK_KV=128 /
    # num_stages=2 / nonkdim=16):
    #     M=2048  0.589 -> 0.553 ms      M=4096  2.033 -> 1.914 ms
    #     M=8000  7.299 -> 6.871 ms      M=8192  7.617 -> 7.159 ms
    # 4 is a hard regression (11.6 ms at M=8000) -- the trimming starts
    # costing more than the extra occupancy buys.
    waves_per_eu = 3

    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = k_fp8.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=k_fp8,
        kv_scales_ptr=kv_scales_1d,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        OOW_FILL=True,
        num_warps=4,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    return logits
