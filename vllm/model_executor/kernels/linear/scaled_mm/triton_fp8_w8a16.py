# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based W8A16 (fp8 weight, bf16/fp16 activation) GEMM for ROCm gfx90a.

Dequantizes float8_e4m3fn weights inside the GEMM's inner loop and feeds the
result straight to MFMA, so the fp8 bytes are the only weight traffic and the
activation is never quantized. This is the W8A16 case: the point is that `x`
arrives unquantized and stays that way.

Two decode variants are implemented, both exact; the default is measured, not
assumed. See WHICH DECODE SHIPS below.

Weight layout expected by this kernel (post-process_weights_after_loading):
  weight:       [K, N]  float8_e4m3fn, viewed as uint8 by the kernel.
                Arrives as `layer.weight.t()` from the scheme, i.e. a view of a
                C-contiguous [N, K] buffer, so K is the contiguous dim
                (stride_bk == 1, stride_bn == K). Strides are passed
                explicitly; no layout is assumed. See _WEIGHT_LAYOUT below.
  weight_scale: [N]  float32, one scale per output channel.

Checkpoint layout from CompressedTensorsW8A16Fp8.create_weights:
  weight:       [N, K]  float8_e4m3fn
  weight_scale: [N, 1]  float32   (per-tensor is expanded to channel by the
                scheme's process_weights_after_loading before we see it)


WHICH DECODE SHIPS
------------------
Both variants are exact on all 254 finite codes; the default is bf16-native
because it is faster, not because it is safer. Measured head to head on this
card with interleaved A/B sampling (so serving drift hits both arms equally),
median of 40 per sample, three independent sweeps. Ratio of fp16 wall clock to
bf16, so below 1.000 would mean the fp16 bit trick is faster:

    gate_up  34816x5120  M=1   1.013 1.018 1.010
    gate_up  34816x5120  M=8   1.014 1.021 1.010
    down_proj 5120x17408 M=1   1.080 1.079 1.101
    down_proj 5120x17408 M=8   1.097 1.100 1.083
    o_proj    4096x4096  M=1   1.054 1.052 1.056
    q_proj    6144x5120  M=1   1.059 1.062 1.063

The bit trick never wins. It ties on gate_up (1.3%, inside serving noise) and
loses 5-10% everywhere else, so the shipping rule -- fp16 only on a consistent
>5% win -- selects bf16-native. That is also the simpler kernel by some
distance: no fold factor, no subnormal hazard on either operand, and no
activation conversion. The fp16 path is kept behind `decode="fp16"` because it
was the original design, it is fully verified, and it is the fallback if a
future Triton changes the native cast lowering.

The intuition for why the cheaper decode does not win: at these shapes the
kernel is streaming weights, not saturating the VALU, so ~1.5 extra ops per
weight are largely hidden -- while the fp16 path's extra bf16->fp16 activation
convert and its in-loop `* 256.0` are not free. MFMA rate is not the
tiebreaker either: v_mfma_f32_16x16x16bf16_1k measures 165.5 TFLOP/s against
16x16x16f16's 161.5 on gfx90a (spec says both 181), so the dtype choice is a
decode-cost question only.


MEASUREMENT BASIS
-----------------
Every throughput figure in this file comes from the one table below, so that
there is a single number per shape to keep current. It is quoted as fp8 weight
bytes moved per second, at M=1, median of 30, on a card that is concurrently
serving -- so treat small differences as noise.

                       [N,K]-backed view    [K,N] contiguous copy
    gate_up  34816x5120     611.7 GB/s            454.7 GB/s
    down_proj 5120x17408    373.7 GB/s (stale)    224.2 GB/s (stale)
    o_proj    4096x4096     169.1 GB/s (stale)    142.3 GB/s (stale)

STALE ROWS: down_proj and o_proj have N < 16384, so at the time they were
taken they ran (16, 16, 128). The ladder now gives every narrow-rung shape
(16, 32, 64), which the tile sweep puts 7-9% ahead, so both rows understate
the current kernel and need re-taking. gate_up was already on (16, 32, 64) at
M=1 and is unaffected. Left in place rather than deleted because the
view-vs-copy RATIO is what they are cited for, and both columns moved together.

MEASURED AT the triton_fp8_w8a16_gemm wrapper: host wall clock with a
torch.cuda.synchronize inside the timed region. That boundary includes the
output allocation and the launch/sync cost of every call, so it is lower than
the kernel alone and higher than what a layer actually sees. For gate_up at
M=1 the same work measures roughly:

    kernel only (profiler)        ~754 GB/s
    this table (wrapper)           611.7 GB/s
    through apply_weights          ~517 GB/s

The kernel-only and apply_weights figures are from the separate profiling
pass, not from this file's sweep; they are quoted here only to fix what the
wrapper number does and does not contain. Compare like with like: a number
taken at one boundary must not be set against a number taken at another.


TILE LADDER
-----------
Three tiles are in play, all float64-oracle-verified before being timed
(4.5e-07 max rel err each, i.e. identical):

    narrow  (16, 32, 64)  num_warps=2
    bm32    (32, 32, 64)  num_warps=2
    wide    (64, 64, 32)  default warps

Fitted to a measured grid of 11 N-values x 10 M-values, and it reproduces the
per-cell best tile on ALL 110 cells (0.00% mean loss, 0.00% worst). That is a
fit to this grid, not a proof: the thresholds are shape boundaries, and the
one at N=20000 is interpolated -- bm32 measured best at 16384 and wide at
34816, with nothing measured between.

HOW IT IS SAMPLED, which turned out to matter more than anything in the
ladder. An earlier version of this grid rotated the three tiles call-to-call,
to keep serving drift from favouring whichever arm ran first. That rotation
silently handicapped the narrow tile by ~22%, because narrow at M>16 has
grid_m=2 and re-reads its weights, so it depends on those weights still being
in L2 -- and the wide tile's 64x64x32 access pattern evicts them between
samples. bm32 (grid_m=1) does not care and was unaffected. Measured at
N=5120, K=5120, M=24, median of 41:

    configuration        narrow     bm32     wide
    isolated (alone)      99.9us   109.8us  137.6us
    narrow+bm32          101.5us   111.1us       -
    narrow+wide          123.7us        -   137.9us
    narrow+bm32+wide     123.6us   110.0us  138.5us

Production runs one tile repeatedly for a given layer and never alternates, so
the isolated column is the real one and the interleaved column is an artifact.
Sampling now runs each arm in blocks long enough for cache state to be that
arm's own, alternating blocks so drift still hits every arm, with 25 warmup
iterations discarded per block (an isolated bm32 first-touch reads 151us before
settling to 110us).

Fixing that removed a phantom. The old grid showed the best tile oscillating
with N -- bm32, bm32, narrow, narrow, ..., bm32, narrow, bm32 -- which looked
like wave-quantisation ripple and was documented here as an argument that no
threshold schedule could work. It was the sampling. The corrected surface is
monotonic in both M and N, which is why the thresholds above fit it exactly.

The M<=16 rung is unaffected by any of this and is where the money is: narrow
beat wide at every N by 1.3-1.8x, and it won despite the handicap, so the true
margin is if anything larger.

Autotuning is still the better long-term answer than more thresholds, but for
a different reason than previously recorded here: not because the surface is
ragged (it is not), but because these boundaries are fitted to K=5120 and to
one card, and K enters through how long each workgroup runs.


VARIANT 1 (DEFAULT): native fp8e4nv -> bf16 cast, bf16 dot
----------------------------------------------------------
`b.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)`. Triton lowers this on AMD
as a packed magic-bias sequence -- mask 0x7fff7fff, lshr 4, fmul 2^120, see
ConvertFpCastOpToLLVM.cpp OcpF8ToBf16SW -- at roughly 4.5 ops per value. It
produces TRUE values, so no fold comes back out in the epilogue.

It is exact including the e4m3 denormal codes, which is not obvious and was
verified rather than assumed: a denormal code lands in the bf16 SUBNORMAL
range after the shift, and the fmul by 2^120 renormalizes it exactly, because
bf16's 7 mantissa bits hold e4m3's 3 significant bits with room to spare.
Code 0x01, for instance, shifts to bf16 mantissa 16 with exponent 0, i.e.
(16/128)*2^-126 = 2^-129, and 2^-129 * 2^120 = 2^-9, which is the true value.
Measured: 254/254 codes exact through the pure cast, and 254/254 exact through
tl.dot at BLOCK (16,16,128), (16,32,64) and (64,64,32).

Because the decoded values are true bf16 values and e4m3's smallest magnitude
is 2^-9, nothing on the weight side comes near bf16's 2^-126 subnormal floor,
so the MFMA flush documented below cannot bite. The activation is consumed as
bf16 with no conversion, so it cannot bite there either.


VARIANT 2: e4m3fn -> fp16 by bit manipulation
---------------------------------------------
An e4m3fn byte is [s eeee mmm]: sign, 4-bit exponent (bias 7), 3-bit mantissa.
An fp16 is [s eeeee mmmmmmmmmm]: sign, 5-bit exponent (bias 15), 10-bit
mantissa. The fields are compatible up to a shift and a bias, so the convert
is three integer ops instead of a conversion instruction:

    t     = b_u8.to(uint16) << 7   # e -> fp16 exp[13:10], m -> fp16 mant[9:7],
                                   # sign lands at bit 14
    t     = t + (t & 0x4000)       # carry: sign moves 14 -> 15, and the add
                                   # clears bit 14 (the top exponent bit) in
                                   # the process, which is exactly what we want
    b_f16 = bitcast(t, fp16)

The resulting fp16 exponent field is [0, eeee], so the value is
2^(eeee-15) * 1.mmm against the true 2^(eeee-7) * 1.mmm: every decoded weight
is the true weight times 2^-8. The factor 2^8 is a constant, so it has to come
back out -- see the `* 256.0` below.

The top exponent bit cannot be set instead of cleared to avoid the 2^-8 in the
first place: eeee goes up to 15, so [1, eeee] would carry into the sign bit.
2^-8 is the largest fold this construction admits.

Verified on GPU against a float64 oracle built from the raw bytes: all 254
finite codes decode bit-exactly (0 mismatches over a 254x254 one-hot probe,
with and without per-channel scales).


SUBNORMAL FLUSH -- why variant 2 needs a `* 256.0` in the loop
--------------------------------------------------------------
This section is about variant 2 only. The default bf16 path is immune, for the
reasons given under variant 1 -- but the hardware fact is permanent and worth
recording, because anyone writing an fp16 MFMA kernel on this target will hit
it.

The e4m3 denormal codes 0x01..0x07 are 1..7 * 2^-9. Biased by 2^-8 they become
2^-17..7*2^-17, which are fp16 SUBNORMALS (fp16 min normal is 2^-14).

Probed on this hardware (MI210, gfx90a, torch 2.11.0+rocm7.14.0, triton 3.7.1):
v_mfma_f32_*f16 FLUSHES SUBNORMAL INPUTS TO ZERO. Feeding the biased values
straight to tl.dot returned exactly 0.0 for every code in 0x01..0x07 and the
correct value from 0x08 (= 2^-14, the first normal) upward. The result was
identical for all five tile shapes in the ladder below, so it is a property of
the MFMA path and not of one instruction selection. Isolated on raw operands,
the boundary is exactly the fp16 normal floor:

    operand   through tl.dot   through a plain VALU multiply
    2^-13     1.2207e-04       1.2207e-04
    2^-14     6.1035e-05       6.1035e-05     <- min normal, fine
    2^-15     0.0000e+00       3.0518e-05     <- first subnormal, flushed
    2^-17     0.0000e+00       7.6294e-06
    2^-24     0.0000e+00       5.9605e-08

The VALU does NOT flush: the same bitcast multiplied by 256.0 in fp16 before
the dot returns the exact e4m3 value for those same codes. So the fix is one
fp16 multiply inside the loop, which cancels the 2^-8 at the source and leaves
no subnormal anywhere:

    b_f16 = bitcast(t, fp16) * 256.0     # exact: power-of-two rescale,
                                         # 2^-17 -> 2^-9, max 1.75 -> 448

That multiply is the whole cost of correctness here. Without it the low
seventh of the weight range silently becomes zero, which does not crash -- it
emits fluent wrong text.

FOLD FACTOR: because the `* 256.0` already undoes the bias, the per-channel
scale is used as-is under BOTH variants (fp32, upcast once at load). Do NOT
fold 2^8 into the scale; it would double-apply under variant 2 and be simply
wrong under variant 1.

ACTIVATION RANGE, variant 2 only: bf16 -> fp16 is exact for |x| in
[2^-14, 65504]. Above that the activation overflows fp16 to inf; below it the
same MFMA flush zeroes the element. Both are verified end-to-end rather than
clamped -- a clamp in the inner loop costs ~8%. Measured cost of the low end
by scaling a random activation vector (K=4096) against the float64 oracle:

    |a| ~ 1e+00    0.0% of A subnormal    err 9.2e-07
    |a| ~ 1e-02    0.5%                   err 3.1e-04
    |a| ~ 1e-04   45.8%                   err 2.5e-01
    |a| ~ 1e-05  100.0%                   err 1.0e+00

It is an absolute floor rather than a relative one, and it only bites once a
large fraction of the whole activation vector is below 2^-14 -- a regime
post-RMSNorm hidden states do not reach. None of this applies to the default
variant, which keeps the activation in bf16 and never converts it: re-running
that identical sweep on the bf16 path gives 9.2e-07 / 1.3e-06 / 1.5e-06 /
9.8e-07 / 9.4e-07, i.e. flat, with the 100%-subnormal case as accurate as the
|a|~1 case. Removing that cliff is a large part of why bf16-native ships.


BF16 MULTIPLY LANDMINE
----------------------
Do not introduce a scalar bf16 multiply anywhere in this kernel, in particular
not in the epilogue. AMD lowers scalar bf16 fmul via v_dot2_bf16_bf16, which
truncates instead of rounding to nearest-even, giving a systematic 1-ulp bias
(triton#11217). The epilogue here deliberately scales the fp32 accumulator by
an fp32 scale and adds an fp32-upcast bias; keep it that way. The `* 256.0` in
variant 2 is fp16, not bf16, and is exact anyway because it is a power of two.


NaN POLICY
----------
0x7F and 0xFF are the only NaN codes in e4m3fn. NEITHER variant propagates
them -- both the bit trick and Triton's native cast turn 0x7F into a finite
480.0 and 0xFF into -480.0 (measured). Rather than pay for a check in the
inner loop, process_weights_after_loading rejects any checkpoint containing
those codes -- loudly, at load, at zero runtime cost. A checkpoint with NaN
weights is broken anyway.


PRIOR ART, AND WHAT WAS REJECTED
--------------------------------
The variant-2 bit trick is upstream-canonical rather than novel: fp8-Marlin's
dequant (csrc/quantization/gptq_marlin/dequant.h) uses the identical
construction with the identical constants -- (q & 0x80008000) | ((q &
0x7F007F00) >> 1) with bias 256.0 for fp16, and >> 4 with 2^120 for bf16 --
in packed-pair form over two values at once. That packed form is a further
optimisation available here and is deliberately not taken yet: it needs the
weights repacked so each fp8 sits in the high byte of a 16-bit lane, which is
a load-time layout change, and the measurement above says the decode is not
the bottleneck at these shapes anyway.

Rejected:
  * Triton's native `.to(tl.float16)` for fp8e4nv. It lowers to ~22 ops
    including an 8-entry denormal select-chain, against 4.5 for the bf16 cast.
    Counted, not measured; there is no plausible way it competes.
  * AITER's Triton a16w8 kernel, as a whole. It uses the generic `.to()` cast
    AND applies the per-channel scale inside the K loop rather than hoisting
    it to the epilogue, so it does per-weight what this kernel does per
    output. Adopting it would have meant undoing both.
"""

from collections.abc import Sequence

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

# Default decode variant -- see "WHICH DECODE SHIPS" in the module header for
# the measurement this is set from. Both variants are exact on all 254 finite
# codes, so this is a speed choice only and flipping it is a one-line edit.
# fp16 activations always take "fp16" regardless (see the wrapper).
_DECODE = "bf16"


@triton.jit
def triton_fp8_w8a16_gemm_kernel(
    # Pointers
    a_ptr,  # [M, K]  bf16/fp16 activations
    b_ptr,  # [K, N]  uint8 (raw float8_e4m3fn bytes)
    s_ptr,  # [N]     fp32 per-channel weight scales
    bias_ptr,  # [N]  bf16/fp16, unused when HAS_BIAS is False
    c_ptr,  # [M, N]  bf16/fp16 output
    # Dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    # Decode variant. False = the fp16 bit trick (DEQUANT above); True =
    # Triton's native fp8e4nv -> bf16 cast with a bf16 dot. constexpr, so only
    # the selected branch is ever compiled. See _DECODE in the wrapper.
    DECODE_BF16: tl.constexpr,
    # Block sizes (tuned for gfx90a, wavefront=64 -- see the ladder in the
    # wrapper below)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused W8A16 GEMM: C[M,N] = A[M,K] @ (dequant(B)[K,N] * scale[N])

    B holds raw float8_e4m3fn bytes. Dequant is the bit trick documented in the
    module header: shift into the fp16 field layout, carry the sign, rescale by
    2^8 to undo the exponent-bias difference.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # ---- Load activations A: [BLOCK_M, BLOCK_K] ----
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # ---- Load raw fp8 weight bytes B: [BLOCK_K, BLOCK_N] ----
        # other=0 is code 0x00, which decodes to +0.0, so masked lanes
        # contribute nothing.
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = mask_k[:, None] & mask_n[None, :]
        b_u8 = tl.load(b_ptrs, mask=mask_b, other=0)

        if DECODE_BF16:
            # ---- Native cast, bf16 dot ----
            # Triton lowers fp8e4nv -> bf16 on AMD as a packed magic-bias
            # sequence (mask 0x7fff7fff, lshr 4, fmul 2^120 -- see
            # ConvertFpCastOpToLLVM.cpp OcpF8ToBf16SW), ~4.5 ops per value.
            # It yields TRUE values, so there is no fold to undo, and e4m3
            # denormals are ordinary normals in bf16 (floor 2^-126), so the
            # MFMA subnormal flush cannot bite. Verified exact on all 254
            # finite codes through tl.dot at three tile shapes.
            # The activation is already bf16 here (enforced by the wrapper);
            # converting fp16 -> bf16 would lose mantissa bits, so that
            # combination never selects this branch.
            b_dot = b_u8.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
            a_dot = a
        else:
            # ---- Dequantize e4m3fn -> fp16 (module header for derivation) ---
            t = b_u8.to(tl.uint16) << 7
            t = t + (t & 0x4000)
            # The `* 256.0` is not a scale fold, it is the subnormal fix: MFMA
            # on gfx90a flushes fp16 subnormal operands to zero and the biased
            # forms of codes 0x01..0x07 are subnormal. Exact (power of two, no
            # overflow: max 1.75 -> 448).
            b_dot = t.to(tl.float16, bitcast=True) * 256.0
            # bf16 -> fp16 is exact over the activation range; see the module
            # header for the overflow bound.
            a_dot = a.to(tl.float16)

        accumulator += tl.dot(a_dot, b_dot, out_dtype=tl.float32)

    # ---- Epilogue: per-channel scale (+ bias), in fp32 ----
    # Hoisted out of the loop: the scale is per output channel, so it is
    # O(BLOCK_M*BLOCK_N) once here instead of O(BLOCK_K*BLOCK_N) per K-tile.
    # Deliberately fp32 throughout -- see BF16 MULTIPLY LANDMINE in the header;
    # a bf16 multiply here would pick up a systematic 1-ulp truncation bias.
    scales = tl.load(s_ptr + offs_n, mask=mask_n, other=0.0)
    accumulator = accumulator * scales[None, :]

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator = accumulator + bias[None, :].to(tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def triton_fp8_w8a16_gemm(
    a: torch.Tensor,  # [M, K] bf16/fp16
    b_fp8: torch.Tensor,  # [K, N] float8_e4m3fn (or uint8 view)
    scales: torch.Tensor,  # [N] fp32
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    decode: str | None = None,
) -> torch.Tensor:
    """
    Fused W8A16 GEMM with float8_e4m3fn weights.

    Args:
        a:        Activations [M, K], bfloat16 or float16. Must be contiguous.
        b_fp8:    Weights [K, N], float8_e4m3fn. Any strides; the kernel reads
                  it through explicit strides.
        scales:   Per-output-channel scales [N], float32, contiguous.
        bias:     Optional bias [N].
        out_dtype: Output dtype; defaults to a.dtype.
        decode:   "bf16" (native cast, bf16 dot) or "fp16" (bit trick, fp16
                  dot). None picks _DECODE, the measured default. Both are
                  exact on all 254 finite codes; this only trades speed. fp16
                  activations always force "fp16" -- see _DECODE.

    Returns:
        Output [M, N] in out_dtype.
    """
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    assert scales.is_contiguous(), "Scales must be contiguous"

    if decode is None:
        decode = _DECODE
    # fp16 activations cannot take the bf16 branch: fp16 -> bf16 drops 3
    # mantissa bits, and unlike bf16 -> fp16 it is not exact.
    if a.dtype is not torch.bfloat16:
        decode = "fp16"
    assert decode in ("bf16", "fp16"), f"unknown decode variant {decode!r}"

    M, K = a.shape
    assert b_fp8.shape[0] == K, f"b shape {tuple(b_fp8.shape)} does not match K={K}"
    N = b_fp8.shape[1]
    assert scales.numel() == N, f"expected {N} scales, got {scales.numel()}"

    if out_dtype is None:
        out_dtype = a.dtype
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    # Read the fp8 bytes as uint8; Triton has no float8_e4m3fn load that
    # preserves the raw bits, and the whole decode is integer work anyway.
    b_u8 = b_fp8.view(torch.uint8)

    num_warps = None

    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a

        if on_gfx90a():
            # Cloned from the gfx90a ladder in mixed_precision/triton_w4a16.py,
            # which was searched over 69 configurations at M=1 on this exact
            # card. That search was run against int4 decode; fp8 decode is
            # cheaper (4 VALU ops per weight against ~19), so the balance
            # between decode cost and occupancy is not identical and a retune
            # may pay. Shipping the known-good ladder first: it is an
            # occupancy fix at heart -- BLOCK_N=64 leaves a narrow-N layer with
            # fewer workgroups than the card has CUs (104) -- and that argument
            # does not depend on what the decode costs.
            #
            # (M, N)-keyed; see TILE LADDER in the module header for the
            # measured grid this comes from and how it is sampled.
            if M <= 16:
                # narrow wins at every N measured, by 1.3-1.8x. Decode case.
                BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 64
                num_warps = 2
            elif M <= 32:
                if N < 14336:
                    BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 64
                    num_warps = 2
                elif N < 20000:
                    # Narrow band where BLOCK_M=32 wins: grid_m drops to 1 so
                    # the weights are read once, and N is not yet wide enough
                    # for the wide tile to take over. Upper bound is
                    # INTERPOLATED -- measured bm32 at 16384 and wide at
                    # 34816, nothing between.
                    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 64
                    num_warps = 2
                else:
                    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            elif M <= 48:
                if N < 12288:
                    BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 64
                    num_warps = 2
                else:
                    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            elif M <= 64:
                if N < 5120:
                    BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 64
                    num_warps = 2
                else:
                    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                # Inherited, and the only rung this grid does NOT cover: it
                # was measured to M=64. (128, 128, 32) lost to (64, 64, 32) at
                # every cell up to there, so this boundary is assumption, not
                # measurement. Worth a sweep if prefill throughput matters.
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        else:
            if M <= 32:
                BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        if M <= 32:
            BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
        elif M <= 64:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    launch_opts = {} if num_warps is None else {"num_warps": num_warps}

    triton_fp8_w8a16_gemm_kernel[grid](
        a,
        b_u8,
        scales,
        bias if bias is not None else scales,  # dummy ptr; Triton needs one
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b_u8.stride(0),
        b_u8.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
        DECODE_BF16=decode == "bf16",
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        **launch_opts,
    )
    return c


class TritonW8A16Fp8LinearKernel(FP8ScaledMMLinearKernel):
    """
    Triton W8A16 fp8 GEMM for ROCm gfx90a (MI210).

    Consumes bf16/fp16 activations directly -- no activation quantization --
    and dequantizes float8_e4m3fn weights in the GEMM inner loop.
    """

    _SUPPORTED_WEIGHT_QUANT_KEYS = {
        # TENSOR is promoted to CHANNEL by CompressedTensorsW8A16Fp8's
        # process_weights_after_loading before apply time, but the config is
        # built with the pre-promotion key, so both have to be accepted here.
        kFp8StaticChannelSym,
        kFp8StaticTensorSym,
    }

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Gated on gfx90a, on evidence rather than on validity. The dequant is
        # portable IEEE bit manipulation, but the subnormal-flush workaround
        # and the tile ladder are both measured facts about CDNA2 MFMA and
        # about a 104-CU card. Nobody has measured gfx942 or RDNA here, so
        # they keep their existing kernels. Widen once someone does.
        if not current_platform.is_rocm():
            return False, "TritonW8A16Fp8Linear requires ROCm"
        from vllm.platforms.rocm import on_gfx90a

        if not on_gfx90a():
            return False, "TritonW8A16Fp8Linear is only tuned/verified on gfx90a"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in cls._SUPPORTED_WEIGHT_QUANT_KEYS:
            # BLOCK in particular is deliberately rejected: it does not reach
            # this kernel list at all (it routes through
            # _POSSIBLE_FP8_BLOCK_KERNELS), and its per-block scale would need
            # the scale reload moved back inside the K loop.
            return (
                False,
                "TritonW8A16Fp8Linear only supports per-channel and per-tensor "
                "weight quantization",
            )
        if c.weight_quant_key.dtype != torch.float8_e4m3fn:
            # Not dead code, despite the keys above already pinning a dtype:
            # kFp8StaticChannelSym is built from current_platform.fp8_dtype(),
            # which is float8_e4m3fnUZ on the ROCm targets where is_fp8_fnuz()
            # holds. fnuz has exponent bias 8, no signed zero, and NaN only at
            # 0x80 -- the shift/carry derivation in the module header is for
            # e4m3fn and would silently mis-decode every fnuz weight by a
            # factor of two. e5m2 likewise needs its own derivation. Reject.
            return (
                False,
                "TritonW8A16Fp8Linear only supports float8_e4m3fn weights, got "
                f"{c.weight_quant_key.dtype}",
            )
        if c.input_dtype not in (torch.bfloat16, torch.float16):
            return False, "TritonW8A16Fp8Linear only supports bf16/fp16 activations"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        # Deliberately not FP8ScaledMMLinearKernel.__init__: it constructs a
        # QuantFP8 for the activation, and this kernel never quantizes the
        # activation. Same reimplementation of the grandparent body as
        # XPUW8A16FP8LinearKernel.
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Validate the weight codes and materialize an fp32 [N] scale.

        The scheme hands us `weight` already canonicalized to [K, N] -- it is a
        `.t()` view of the C-contiguous [N, K] checkpoint buffer, so K is the
        contiguous dimension.

        _WEIGHT_LAYOUT: that view is left alone deliberately, not made
        contiguous. With K contiguous, one column of a [BLOCK_K, BLOCK_N] tile
        is BLOCK_K consecutive bytes (128 at the decode tile), which coalesces;
        calling .contiguous() would flip that to BLOCK_N consecutive bytes per
        row -- 16 at the same tile -- and would also double the layer's peak
        memory during load. Both layouts were measured: see MEASUREMENT BASIS
        in the module header, whose two columns are exactly this comparison.

        The view wins on every shape there, so the copy would cost memory and
        time for nothing.

        SCOPE, stated precisely because the earlier wording overclaimed it:
        that comparison was run at M=1, i.e. only on the narrow rung, where
        BLOCK_K=64 exceeds BLOCK_N=32. BLOCK_K > BLOCK_N is the condition the
        argument rests on, and it does NOT hold at the upper rungs -- (64, 64,
        32) and (128, 128, 32) both have BLOCK_N >= BLOCK_K, which is exactly
        the documented kill condition. So the layout choice is measured for
        decode and UNMEASURED for prefill; M=32/64 is an open check, not a
        settled result.
        """
        w = layer.weight

        # NaN codes do not survive the bit trick -- 0x7F would decode to a
        # finite 480.0 -- so reject them at load rather than check per weight
        # in the inner loop. Loud, once, free.
        if ((w.data.view(torch.uint8) & 0x7F) == 0x7F).any():
            raise ValueError(
                "TritonW8A16Fp8Linear: weight contains float8_e4m3fn NaN codes "
                "(0x7F/0xFF). This kernel's bit-trick dequant decodes them to a "
                "finite 480.0 instead of propagating NaN, so the checkpoint is "
                "rejected rather than silently mis-evaluated."
            )

        # Per-channel scale as contiguous fp32 [N]. Checkpoints store this as
        # [N, 1], and depending on the loader path it can arrive bf16, whose
        # 8 mantissa bits would visibly quantize the output; upcast once here
        # where it costs nothing.
        s = layer.weight_scale.data.to(torch.float32).reshape(-1).contiguous()
        expected_n = w.shape[1]
        if s.numel() != expected_n:
            raise ValueError(
                f"TritonW8A16Fp8Linear: expected {expected_n} per-channel weight "
                f"scales for a [K={w.shape[0]}, N={expected_n}] weight, got "
                f"{s.numel()}"
            )
        replace_parameter(layer, "weight_scale", s)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x is consumed directly, unquantized -- that is the entire point of
        # W8A16. No _get_layer_params/quant_fp8/super().apply_weights here.
        weight = layer.weight
        scales = layer.weight_scale

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out_shape = x.shape[:-1] + (weight.shape[1],)

        out_dtype = self.config.out_dtype or x.dtype
        output = triton_fp8_w8a16_gemm(
            a=x_2d,
            b_fp8=weight,
            scales=scales,
            bias=bias,
            out_dtype=out_dtype,
        )
        return output.reshape(out_shape)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # Dead: required by the FP8ScaledMMLinearKernel ABC but never reached,
        # because apply_weights above is overridden and never calls it. Same
        # stub as MarlinFP8ScaledMMLinearKernel and XPUW8A16FP8LinearKernel.
        pass
