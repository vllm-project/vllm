# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton-based NVFP4 W4A16 (e2m1 weight, bf16/fp16 activation) GEMM for ROCm
gfx90a.

STATUS: ported from the standalone prototype/benchmark at
`nvfp4_final.py` (session scratchpad, fp8-decode's #19 measurement:
254-code-equivalent correctness pass on synthetic K=512,N=256 data, ~3.5
decode ops/weight, beats an Emulation-style dequant+matmul baseline 2.26x
wall-clock on gate_up-shaped data). Cleared for the GPU pass, which found
this kernel had in fact NEVER COMPILED before that pass -- two fatal bugs
in this file's byte-pair-along-K rewrite that a CPU-only oracle cannot see
(it models the math, not Triton's own dtype semantics): a module-global
`GROUP_SIZE` referenced inside @triton.jit bodies (NameError at trace
time, fixed by an explicit `GS: tl.constexpr` kernel arg) and a uint8
`codes` tensor shifted by 9/12 bits past its own width (Triton silently
discards the high bits rather than promoting; fixed by widening to int32
before shifting). Both are fixed in this version; see each kernel's own
docstring for the full story and fp8-decode's re-verification for the
result. Independent of the GPU-execution question, three things changed
from the validated CPU-oracle-checked prototype and specifically needed
GPU re-verification (flagged inline at each site below):
  1. Packing layout: the prototype's synthetic `make()` packs 8 e2m1 codes
     per int32 (`codes[:, j::8] << (4*j)`). The REAL checkpoint (compressed
     tensors' `weight_packed` / ModelOpt's `weight`) packs 2 codes per
     UINT8 byte, low-nibble-first. CORRECTION: `nvfp4.py`'s attempted
     cross-check against `torch.float4_e2m1fn_x2` never actually ran --
     `.to(torch.float32)` raises `NotImplementedError` on this build, so
     that script fell through to its OCP-spec-table path instead (it has a
     try/except around exactly this, and would have said so had anyone
     checked its output rather than assumed the cross-check passed). The
     nibble convention here rests on the OCP MX FP4 spec plus a
     self-consistent round-trip through the spec's own table, NOT on a
     torch built-in cross-check as previously (wrongly) claimed. The GPU
     pass, once cleared, validates this against a real checkpoint tensor
     directly, which settles it properly.
     The unpack below is rewritten for byte-pair packing; the underlying
     bit-shift E2M1 decode and subnormal-lift math are unchanged from the
     validated prototype.
  2. weight_global_scale: the prototype's synthetic harness only modeled
     the per-16-group e4m3 scale, no global scalar (its test data had none).
     Real NVFP4 checkpoints have both (base.py: "per-block weight scales
     ... and scalar global scales"). Added here as a third, straightforward
     fp32 multiply in the epilogue -- not present in the validated
     prototype, needs its own correctness check once GPU access clears.
  3. FIXED, pre-GPU, during verification prep: the inner kernel (BLOCK_K >
     16, i.e. M>=9 serving) inherited a real bug from the prototype's own
     scale-fold placement -- it fed the MFMA `true_e2m1 * true_scale *
     2**-8`, which underflows fp16's normal floor for 30 of 254 e4m3 scale
     codes and silently zeroed weights (whole blocks, for the smallest
     scale codes) via the same subnormal-input MFMA flush documented below.
     The hoist kernel (BLOCK_K <= 16) was never affected -- its scale
     correction happens in fp32 on the accumulator, post-dot, not in fp16
     pre-dot. Full derivation and the fix in `_nvfp4_w4a16_inner`'s own
     docstring. Verified 0/254 codes against fp8-decode's float64 oracle,
     op-count neutral (the moved multiply lands on the scale, 1/16th the
     elements of the weight tile).

Everything else (the E2M1 bit-trick decode, the subnormal-flush defense,
the two-stage scale fold, the hoist/inner dual-kernel split) is ported
as-measured from nvfp4_final.py, not re-derived.


THE VALIDATED MATH (ported from nvfp4_final.py, credited there)
-----------------------------------------------------------------
e2m1 byte layout (OCP MX FP4, 1 sign + 2 exp[bias 1] + 1 mantissa, no
inf/NaN): nibble n = [s e1 e0 m]. fp16 bits placing the exponent+mantissa
field at fp16's own field positions: `((n & 7) << 9) | ((n & 8) << 12)`.
Decoding that raw bit pattern as fp16 gives `true_e2m1_value * 2**-14`
(fp16 bias 15 vs e2m1 bias 1, 2**(1-15) = 2**-14) -- and for e2m1's actual
magnitudes (0.5..6) that raw decode lands in or near fp16's SUBNORMAL
range (fp16 min normal 2**-14). gfx90a's v_mfma_f32_16x16x16f16 flushes
subnormal MFMA *inputs* to zero (measured, see triton_fp8_w8a16.py's
"SUBNORMAL FLUSH" section for the general finding on this card). The fix
here is the same shape: multiply by 2**14 with a plain VALU op (which does
NOT flush, only the MFMA path does) BEFORE the value ever reaches `tl.dot`.
That multiply is not an extra scale -- it exactly cancels the -14 bias, so
after it the value is the true e2m1 value, now safely inside fp16's normal
range.

The per-16-group scale byte is e4m3 (float8_e4m3fn), decoded via the
identical bit-trick used in triton_fp8_w8a16.py's variant 2
(`(byte<<7); +=(t&0x4000)`, bias -8 this time since e4m3's bias is 7 vs
fp16's 15: 2**(7-15) = 2**-8), corrected by multiplying by 256.0. This is
the "two-stage scale fold" team-lead referenced: weight-side fold (x2**14,
inline in the K-loop, doubles as the subnormal-lift) and scale-side fold
(x2**8, applied once per scale value), independent of each other.

Two kernel variants, both correct, ported from nvfp4_final.py's
`nvfp4_hoist` / `nvfp4_inner`, selected by BLOCK_K vs the group size (16):
  - BLOCK_K <= 16 (one scale-group per K-tile): the per-group scale is
    identical for the whole tile, so it hoists out of the K-loop into a
    single per-(BLOCK_M,BLOCK_N) epilogue multiply -- O(BLOCK_M*BLOCK_N)
    once instead of O(BLOCK_K*BLOCK_N) per K-tile. Same hoist idea as the
    W4A16 int4 kernel's own scale hoist (2ec31f0ba6).
  - BLOCK_K > 16 (multiple scale-groups per K-tile): scale changes within
    the tile, so it can't be hoisted the same way -- applied per-element
    to the decoded weight before the dot, with NSUB = BLOCK_K // 16
    sub-groups broadcast across the tile.


PACKING LAYOUT (real checkpoint, both compressed-tensors and ModelOpt)
------------------------------------------------------------------------
Weight arrives from the scheme as `(N, K // 2)` uint8, PACKED ALONG K
(input_dim=1) -- i.e. each byte holds two CONSECUTIVE K-indices for the
SAME output channel N, low nibble = lower K-index. This rests on the OCP
MX FP4 spec's own nibble ordering, not a runtime cross-check against
`torch.float4_e2m1fn_x2` -- that path in `nvfp4.py` never actually ran on
this build (`.to(torch.float32)` raises `NotImplementedError`), see the
correction at the top of this file. The GPU pass validates this directly
against a real checkpoint tensor, which is the check that actually settles
it. This differs from the int4 W4A16 kernel, which packs along N
(8 output channels per int32) -- NVFP4 packing needs no interleave-based
unshuffle across N at all, only a lo/hi nibble split per byte, which is
simpler than the int4 case.

Neither CompressedTensorsW4A4Fp4 nor ModelOptNvFp4W4A16LinearMethod
transposes the weight in process_weights_after_loading (checked both,
neither calls `.t()`, unlike CompressedTensorsW8A16Fp8's FP8 scheme) --
so this kernel's own process_weights_after_loading does it, the same
`.t()`-and-leave-as-a-view choice triton_fp8_w8a16.py made for its weight
(see _WEIGHT_LAYOUT there: the view wins over a forced-contiguous copy on
every measured shape). Scale gets the same treatment for the same reason.
"""

from collections.abc import Sequence

import torch

from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig

GROUP_SIZE = 16


@triton.jit
def _nvfp4_w4a16_hoist(
    a_ptr,  # [M, K]  bf16/fp16 activations
    b_ptr,  # [K//2, N]  uint8, packed e2m1, 2 codes/byte along K (low nibble = lower K)
    s_ptr,  # [K//16, N]  uint8, raw float8_e4m3fn bytes, per-16-group scale
    global_scale,  # python float / 0-d tensor, fp32 -- weight_global_scale
    c_ptr,  # [M, N]  bf16/fp16 output
    M, N, K,
    sam, sak, sbk, sbn, ssk, ssn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GS: tl.constexpr,
):
    """BLOCK_K <= 16: exactly one scale-group per K-tile, scale hoisted to
    the epilogue. Ported from nvfp4_final.py's `nvfp4_hoist`, packing
    rewritten for byte-pair-along-K layout (see module header).

    GS is GROUP_SIZE (module global, =16) passed explicitly as a
    tl.constexpr kernel arg -- the module global itself can't be read from
    inside a @triton.jit function body on this Triton version, it raises
    NameError at trace time. See FATAL BUG A in this file's commit history
    for the full story; this kernel never compiled before that fix."""
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    BLOCK_KP: tl.constexpr = BLOCK_K // 2  # packed-byte width of this K-tile

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ks in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = ks * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        offs_kp = ks * BLOCK_KP + tl.arange(0, BLOCK_KP)
        mask_kp = (ks * BLOCK_K + 2 * tl.arange(0, BLOCK_KP)) < K

        a = tl.load(
            a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak,
            mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0,
        )

        # Load packed bytes as [BLOCK_N, BLOCK_KP] (N-rows, K-pair-cols) --
        # orientation is a pointer-arithmetic choice only, not a physical
        # layout requirement; see module header on why weight stays a
        # `.t()` view rather than a forced-contiguous copy.
        bp = tl.load(
            b_ptr + offs_kp[None, :] * sbk + offs_n[:, None] * sbn,
            mask=mask_kp[None, :] & mask_n[:, None], other=0,
        )
        lo = bp & 0xF
        hi = (bp >> 4) & 0xF
        # interleave along the last axis: lo[0],hi[0],lo[1],hi[1],... =
        # increasing K order, matching the low-nibble-first convention.
        codes = tl.interleave(lo, hi)  # [BLOCK_N, BLOCK_K], uint8

        # FATAL BUG B, found on the GPU pass: codes is uint8 here, and the
        # shifts below are by 9 and 12 bits -- both past uint8's own width,
        # so Triton silently discards the high bits instead of promoting
        # (it does warn; nobody was watching for it pre-GPU). Widen to
        # int32 BEFORE shifting, mirroring the int4 W4A16 kernel's own
        # proven pattern for exactly this hazard. Un-widened, this produced
        # 380/478 zeros plus NaN on the remainder on real hardware -- not a
        # subtle numerical drift, a silent bit-discard.
        codes32 = codes.to(tl.int32)

        # e2m1 -> fp16 raw bits, then lift by 2**14 (VALU multiply, does
        # NOT flush subnormals -- see module header). This IS the bias
        # correction, not an extra factor: after this vb holds the true
        # e2m1 value, already normal-range-safe for the MFMA below.
        vb = (((codes32 & 7) << 9) | ((codes32 & 8) << 12)).to(tl.int16).to(tl.float16, bitcast=True)
        vb = vb * 16384.0
        vb = tl.trans(vb)  # [BLOCK_K, BLOCK_N] for the dot

        dot = tl.dot(a.to(tl.float16), vb, out_dtype=tl.float32)

        # One scale-group covers this whole K-tile (BLOCK_K <= GS).
        g = (ks * BLOCK_K) // GS
        sb = tl.load(s_ptr + g * ssk + offs_n * ssn, mask=mask_n, other=0).to(tl.uint16)
        t = sb << 7
        t = t + (t & 0x4000)
        s16 = t.to(tl.float16, bitcast=True)
        # x256.0 undoes the e4m3->fp16 bias (see header); global_scale is
        # the checkpoint's separate scalar NVFP4 global scale, fp32
        # throughout per triton_fp8_w8a16.py's BF16 MULTIPLY LANDMINE
        # (never do this multiply in bf16 -- truncates instead of RNE).
        acc += dot * (s16.to(tl.float32) * 256.0 * global_scale)[None, :]

    c = acc.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    mask_c = (offs_m[:, None] < M) & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


@triton.jit
def _nvfp4_w4a16_inner(
    a_ptr, b_ptr, s_ptr, global_scale, c_ptr,
    M, N, K,
    sam, sak, sbk, sbn, ssk, ssn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NSUB: tl.constexpr, GS: tl.constexpr,
):
    """BLOCK_K > 16: multiple scale-groups per K-tile (NSUB = BLOCK_K //
    16), scale applied per-element before the dot since it can't be
    hoisted to a single epilogue multiply. Ported from nvfp4_final.py's
    `nvfp4_inner`; packing rewritten for byte-pair-along-K (see header).

    BUG FOUND DURING VERIFICATION PREP (2026-08-15, fp8-decode's CPU oracle
    + measured cliff, both cited in the fix commit) -- inherited from the
    prototype, not introduced here, and NOT caught by the earlier debug
    probes because those used |scale| ~ 1e-3..1, above the failure floor.

    The prototype's placement folded the weight's own x2**14 lift onto the
    SCALE instead of onto `vb` (comment used to say "the product `vb * sf`
    still lands in the safe range before `tl.dot` sees it" -- that claim
    was wrong, not just imprecise). With that placement, `sf` = true_scale
    * 2**6 (true_scale * 2**-8, e4m3's own bias, times the borrowed 2**14),
    and `vb` alone = true_e2m1 * 2**-14 (never lifted). The PRODUCT fed to
    `tl.dot` was therefore `vb * sf` = true_e2m1 * true_scale * 2**-8 --
    still biased by 2**-8, not the true weight -- and for any |true_scale|
    < 2**-5, that product falls below fp16's normal floor (2**-14) even at
    e2m1's largest magnitude (6.0): 6.0 * 2**-5 * 2**-8 = 2**-9.4, already
    subnormal; smaller e2m1 codes or smaller scales push further under.
    30 of 254 e4m3 scale codes have |value| < 2**-5; scale code 0x01 is far
    enough under the floor that it zeroed entire 16-element blocks outright
    via the same v_mfma_f32_16x16x16f16 subnormal-input flush documented in
    the class docstring and in triton_fp8_w8a16.py -- not a new hardware
    fact, a new way to accidentally feed it a subnormal operand.

    FIX: give the whole 2**14 to `vb` (matching the hoist kernel exactly --
    `vb` becomes the true e2m1 value, |v| in [0.5, 6], always fp16-normal)
    and only 2**8 to the scale (`s16` becomes the true scale value, |s| in
    [2**-9, 448] for e4m3's actual dynamic range, also always fp16-normal
    on its own). `vb * s16` is then already the TRUE dequantized weight --
    both factors individually normal-range BEFORE the multiply, computed by
    the VALU (which does not flush), so the product can't reintroduce a
    subnormal operand the way the old single combined-and-still-biased
    product could. The epilogue's separate `* 256.0` is deleted -- there is
    no scale bias left to correct, only the checkpoint's own global_scale.
    Op-count neutral: the moved multiply now lands on the scale, which has
    1/16th the elements of the weight tile.

    WHY THE HOIST KERNEL (BLOCK_K <= 16) WAS IMMUNE: its scale-side x256.0
    correction happens in fp32, on the fp32 accumulator, AFTER `tl.dot` --
    not in fp16, not before an MFMA. fp32's subnormal floor (~2**-126) is
    unreachable by any real e4m3 scale magnitude, so there was never a
    subnormal fp16 operand for that path to feed the MFMA in the first
    place. `vb` there was already lifted by 2**14 pre-dot, same as this
    fix now does here.

    LESSON: a late-discovered hardware fact (the MFMA subnormal-input
    flush) has to be re-verified against EVERY code path it could touch,
    not just the one that surfaced it. It was first found and fixed in the
    FP8 kernel's variant 2, ported correctly to this kernel's hoist path
    (which mirrors variant 2's own hoisted-scale-in-fp32 structure), but
    the inner path's different fold placement was never re-checked against
    the same failure mode until this pass -- verified now against 0/254
    codes on fp8-decode's oracle rather than assumed fixed by analogy.

    TWO MORE FATAL BUGS FOUND ON THE ACTUAL GPU PASS (2026-08-15), both in
    this file's byte-pair-along-K rewrite (the CPU oracle above models the
    math, not Triton's own dtype semantics, so neither was visible to it --
    this kernel had never actually compiled or executed before this fix):

    (A) GROUP_SIZE (this module's plain-int global) was referenced directly
    inside this @triton.jit body -- Triton raises NameError tracing it, so
    the kernel never compiled. Fixed by adding `GS: tl.constexpr` as an
    explicit kernel argument, passed from the launcher (`GROUP_SIZE` stays
    a module global for the host-side uses in `triton_nvfp4_w4a16_gemm`).

    (B) the byte-pair decode below shifts `codes` (uint8, from
    `tl.interleave`) by 9 and 12 bits -- both past uint8's own width.
    Triton silently discards the high bits rather than promoting the type;
    unwidened, this produced 380/478 zeros plus NaN on the rest against
    real hardware. Fixed by widening to int32 before shifting
    (`codes32 = codes.to(tl.int32)`), mirroring the int4 W4A16 kernel's own
    proven pattern for the identical hazard.
    """
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    BLOCK_KP: tl.constexpr = BLOCK_K // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ks in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = ks * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        offs_kp = ks * BLOCK_KP + tl.arange(0, BLOCK_KP)
        mask_kp = (ks * BLOCK_K + 2 * tl.arange(0, BLOCK_KP)) < K

        a = tl.load(
            a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak,
            mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0,
        )
        bp = tl.load(
            b_ptr + offs_kp[None, :] * sbk + offs_n[:, None] * sbn,
            mask=mask_kp[None, :] & mask_n[:, None], other=0,
        )
        lo = bp & 0xF
        hi = (bp >> 4) & 0xF
        codes = tl.interleave(lo, hi)  # [BLOCK_N, BLOCK_K], uint8
        # Widen before shifting by 9/12 -- past uint8's own width, Triton
        # silently discards the high bits otherwise (FATAL BUG B, see the
        # docstring above).
        codes32 = codes.to(tl.int32)
        vb = (((codes32 & 7) << 9) | ((codes32 & 8) << 12)).to(tl.int16).to(tl.float16, bitcast=True)
        # x16384.0 HERE, on the weight, not on the scale (see the BUG FOUND
        # docstring above) -- vb is now the true e2m1 value, |v| in
        # [0.5, 6], always fp16-normal, same as the hoist kernel does.
        vb = vb * 16384.0
        vb = tl.trans(vb)  # [BLOCK_K, BLOCK_N]

        g0 = (ks * BLOCK_K) // GS
        og = g0 + tl.arange(0, NSUB)
        sb = tl.load(
            s_ptr + og[:, None] * ssk + offs_n[None, :] * ssn,
            mask=offs_n[None, :] < N, other=0,
        ).to(tl.uint16)
        t = sb << 7
        t = t + (t & 0x4000)
        # x256.0, not x16384.0 -- undoes ONLY the e4m3->fp16 bias, giving
        # the true scale value, |s| in [2**-9, 448] for e4m3's actual
        # dynamic range, always fp16-normal on its own. vb and s16 are each
        # individually safe BEFORE this multiply, so their product (formed
        # by the VALU, not the MFMA) can't reintroduce a subnormal operand
        # the way the old combined-and-still-biased product could.
        s16 = t.to(tl.float16, bitcast=True) * 256.0
        sf = tl.reshape(
            tl.broadcast_to(s16[:, None, :], (NSUB, GS, BLOCK_N)),
            (BLOCK_K, BLOCK_N),
        )
        # vb * sf is now the TRUE dequantized weight -- no bias left to
        # correct in the epilogue, only the checkpoint's own global scale.
        acc += tl.dot(a.to(tl.float16), vb * sf, out_dtype=tl.float32)

    # No e4m3->fp16 bias correction here anymore -- it was already applied
    # per-element above. Only the checkpoint's own fp32 global scale.
    acc = acc * global_scale

    c = acc.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    mask_c = (offs_m[:, None] < M) & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_c)


def triton_nvfp4_w4a16_gemm(
    a: torch.Tensor,  # [M, K] bf16/fp16, contiguous
    b_packed: torch.Tensor,  # [K//2, N] uint8, packed e2m1 (see module header)
    scale_bytes: torch.Tensor,  # [K//16, N] uint8, raw e4m3 bytes
    global_scale: float,  # checkpoint's scalar NVFP4 global scale
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.is_contiguous(), "Activation matrix must be contiguous"
    M, K = a.shape
    Kp, N = b_packed.shape
    assert Kp * 2 == K, f"packed weight K//2={Kp} does not match activation K={K}"
    assert scale_bytes.shape == (K // GROUP_SIZE, N), (
        f"scale shape {tuple(scale_bytes.shape)} != ({K // GROUP_SIZE}, {N})"
    )
    if out_dtype is None:
        out_dtype = a.dtype
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    b_u8 = b_packed.view(torch.uint8) if b_packed.dtype != torch.uint8 else b_packed
    s_u8 = scale_bytes.view(torch.uint8) if scale_bytes.dtype != torch.uint8 else scale_bytes

    # Tile ladder: NOT independently retuned for NVFP4 -- inherited from
    # the gfx90a int4 W4A16 ladder's shape (mixed_precision/triton_w4a16.py)
    # as a starting point pending its own sweep, same as that kernel's own
    # "Known soft spot for whoever retunes" note for shapes it didn't
    # search either. BLOCK_K=16 defaults to the hoist variant since
    # GROUP_SIZE=16 lets every tile hoist its scale; this is a principled
    # default (one scale-group per tile, no per-element scale work) rather
    # than a searched one.
    num_warps = None
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx90a

        if on_gfx90a():
            if M <= 8:
                BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 16
            elif M <= 64:
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            else:
                BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
            num_warps = 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 16
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 16

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    launch_opts = {} if num_warps is None else {"num_warps": num_warps}

    common = dict(
        M=M, N=N, K=K,
        sam=a.stride(0), sak=a.stride(1),
        sbk=b_u8.stride(0), sbn=b_u8.stride(1),
        ssk=s_u8.stride(0), ssn=s_u8.stride(1),
        scm=c.stride(0), scn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        **launch_opts,
    )
    if BLOCK_K <= GROUP_SIZE:
        _nvfp4_w4a16_hoist[grid](a, b_u8, s_u8, global_scale, c, GS=GROUP_SIZE, **common)
    else:
        _nvfp4_w4a16_inner[grid](
            a, b_u8, s_u8, global_scale, c,
            NSUB=BLOCK_K // GROUP_SIZE, GS=GROUP_SIZE, **common,
        )
    return c


class TritonNvFp4LinearKernel(NvFp4LinearKernel):
    """Triton NVFP4 W4A16 GEMM for ROCm gfx90a (MI210).

    Consumes bf16/fp16 activations directly (no activation quantization --
    this is the weight-only path, `use_a16=True` at the scheme level) and
    dequantizes packed e2m1 weights in the GEMM inner loop. See module
    docstring for the full math and its GPU-untested status.
    """

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        # Gated on gfx90a specifically, same rationale as
        # TritonW8A16Fp8LinearKernel: the subnormal-flush workaround and
        # tile ladder are measured facts about this card, not proven
        # elsewhere. Widen once someone measures gfx942/RDNA.
        if not current_platform.is_rocm():
            return False, "TritonNvFp4Linear requires ROCm"
        from vllm.platforms.rocm import on_gfx90a

        if not on_gfx90a():
            return False, "TritonNvFp4Linear is only tuned/verified on gfx90a"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        # NvFp4LinearLayerConfig carries no fields (see base.py) -- every
        # NVFP4 layer shares the same packed-uint8 + per-16 e4m3 scale +
        # scalar global-scale structure, so there is nothing here to
        # reject on. Matches EmulationNvFp4LinearKernel's own
        # unconditional True.
        return True, None

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        # Deliberately not calling a shared quant-activation base __init__
        # -- there isn't one to skip here (NvFp4LinearKernel.__init__
        # already only asserts can_implement/is_supported and stores
        # config), unlike FP8ScaledMMLinearKernel's QuantFP8 construction.
        # Kept as an explicit override anyway for symmetry with
        # TritonW8A16Fp8LinearKernel / XPUW8A16FP8LinearKernel and in case
        # a future NvFp4LinearKernel base grows one.
        super().__init__(config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transpose packed weight and scale to K-major, leave as views.

        Checkpoint layout (neither CompressedTensorsW4A4Fp4 nor
        ModelOptNvFp4W4A16LinearMethod transposes before calling in --
        confirmed by reading both, see module header):
          weight:       [N, K//2]  uint8, e2m1 packed 2/byte along K
          weight_scale: [N, K//16] float8_e4m3fn, per-group scale

        Kernel wants (K-major, matching the activation's own K-contiguous
        convention and triton_fp8_w8a16.py's precedent for this):
          weight:       [K//2, N]  uint8   (`.t()` view)
          weight_scale: [K//16, N] uint8   (`.t()` view, raw e4m3 bytes)

        Left as `.t()` views rather than forced `.contiguous()` copies --
        triton_fp8_w8a16.py measured the view winning on every shape it
        tried for an analogous transpose; not independently re-measured
        for NVFP4's different packing density, flagged for the GPU pass.
        """
        w = layer.weight
        replace_parameter(layer, "weight", w.t())

        s = layer.weight_scale
        s_bytes = s.view(torch.uint8) if s.dtype != torch.uint8 else s
        replace_parameter(layer, "weight_scale", s_bytes.t())

        # weight_global_scale is already reduced to a single fp32 scalar
        # by the calling scheme (CompressedTensorsW4A4Fp4 / ModelOpt both
        # do `.max()` before this runs) -- consumed as-is, no transform.
        #
        # CONTRACT, stated explicitly after an adjudicated ambiguity
        # (2026-08-15, fp8-decode's fixture testing + this kernel's own
        # regression test below): this kernel receives a MULTIPLICATIVE
        # global scale. CompressedTensorsW4A4Fp4.process_weights_after_loading
        # has already inverted the on-disk divisor -- CT stores 1/scale on
        # disk, see compressed_tensors_w4a4_nvfp4.py:110-114
        # (`# Process weight global scale (CT stores as divisors, i.e.
        # 1/scale)` / `1.0 / weight_global_scale`) -- so `apply_weights`
        # below must MULTIPLY by this value, never divide. A harness that
        # feeds raw on-disk bytes directly to this kernel, bypassing the
        # scheme's own inversion, will get exactly backwards results; this
        # was checked and ruled out for the shipped kernel, not assumed.
        if not hasattr(layer, "weight_global_scale"):
            raise ValueError(
                "TritonNvFp4Linear: layer has no weight_global_scale; both "
                "the CompressedTensorsW4A4Fp4 and ModelOptNvFp4W4A16 schemes "
                "populate this before calling the kernel, so its absence "
                "means this layer was routed here incorrectly."
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x consumed directly, unquantized -- the W4A16 contract, same as
        # TritonW8A16Fp8LinearKernel.apply_weights.
        weight = layer.weight
        weight_scale = layer.weight_scale
        global_scale = float(layer.weight_global_scale)

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out_shape = x.shape[:-1] + (weight.shape[1],)

        output = triton_nvfp4_w4a16_gemm(
            a=x_2d,
            b_packed=weight,
            scale_bytes=weight_scale,
            global_scale=global_scale,
            out_dtype=x.dtype,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)
