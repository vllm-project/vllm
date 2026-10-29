# RDNA3 (gfx1100) Type Conversion ISA Reference

Verified empirically on **hipcc 7.2.1 / AMD clang 22.0.0git / ROCm 7.2**, compiled
with `-O3 --offload-arch=gfx1100`.
Instruction validity confirmed via `llvm-mc -mcpu=gfx1100` assembler.

---

## Complete v_cvt Instruction Set for gfx1100

### All 44 supported `v_cvt_*` instructions

#### VOP1 — Scalar conversions (29 instructions)

| Instruction | Direction | Category |
|---|---|---|
| `v_cvt_f16_f32_e32` | FP32 → FP16 | Float narrowing |
| `v_cvt_f16_i16_e32` | INT16 → FP16 | **Int→Float (key for INT8 KV-cache)** |
| `v_cvt_f16_u16_e32` | UINT16 → FP16 | Int→Float |
| `v_cvt_f32_f16_e32` | FP16 → FP32 | Float widening |
| `v_cvt_f32_f64_e32` | FP64 → FP32 | Float narrowing |
| `v_cvt_f32_i32_e32` | INT32 → FP32 | Int→Float |
| `v_cvt_f32_u32_e32` | UINT32 → FP32 | Int→Float |
| `v_cvt_f32_ubyte0_e32` | UBYTE0 → FP32 | **Byte0 of DWORD → Float** |
| `v_cvt_f32_ubyte1_e32` | UBYTE1 → FP32 | **Byte1 of DWORD → Float** |
| `v_cvt_f32_ubyte2_e32` | UBYTE2 → FP32 | **Byte2 of DWORD → Float** |
| `v_cvt_f32_ubyte3_e32` | UBYTE3 → FP32 | **Byte3 of DWORD → Float** |
| `v_cvt_f64_f32_e32` | FP32 → FP64 | Float widening |
| `v_cvt_f64_i32_e32` | INT32 → FP64 | Int→Float |
| `v_cvt_f64_u32_e32` | UINT32 → FP64 | Int→Float |
| `v_cvt_flr_i32_f32_e32` | FP32 → INT32 | Float→Int (floor) |
| `v_cvt_floor_i32_f32_e32` | FP32 → INT32 | Float→Int (floor, alias) |
| `v_cvt_i16_f16_e32` | FP16 → INT16 | Float→Int |
| `v_cvt_i32_f32_e32` | FP32 → INT32 | Float→Int (truncate) |
| `v_cvt_i32_f64_e32` | FP64 → INT32 | Float→Int |
| `v_cvt_i32_i16_e32` | INT16 → INT32 | **Sign-extend** |
| `v_cvt_nearest_i32_f32_e32` | FP32 → INT32 | Float→Int (nearest even) |
| `v_cvt_norm_i16_f16_e32` | FP16 → INORM16 | Normalize to [-32768,32767] |
| `v_cvt_norm_u16_f16_e32` | FP16 → UNORM16 | Normalize to [0,65535] |
| `v_cvt_off_f32_i4_e32` | I4 → FP32 | Offset table (for interpolation) |
| `v_cvt_rpi_i32_f32_e32` | FP32 → INT32 | Float→Int (round +inf) |
| `v_cvt_u16_f16_e32` | FP16 → UINT16 | Float→Int |
| `v_cvt_u32_f32_e32` | FP32 → UINT32 | Float→Int |
| `v_cvt_u32_f64_e32` | FP64 → UINT32 | Float→Int |
| `v_cvt_u32_u16_e32` | UINT16 → UINT32 | **Zero-extend** |

#### VOP3/VOP3P — Packed conversions (15 instructions)

| Instruction | Direction | Category |
|---|---|---|
| `v_cvt_pk_i16_f32` | 2×FP32 → 2×INT16 | Packed float→int |
| `v_cvt_pk_i16_i32` | 2×INT32 → 2×INT16 | Packed int narrow |
| `v_cvt_pk_norm_i16_f16` | 2×FP16 → 2×INORM16 | Packed normalize |
| `v_cvt_pk_norm_i16_f32` | 2×FP32 → 2×INORM16 | Packed normalize |
| `v_cvt_pk_norm_u16_f16` | 2×FP16 → 2×UNORM16 | Packed normalize |
| `v_cvt_pk_norm_u16_f32` | 2×FP32 → 2×UNORM16 | Packed normalize |
| `v_cvt_pk_rtz_f16_f32` | 2×FP32 → 2×FP16 | **Packed f32→f16 (round-to-zero)** |
| `v_cvt_pk_u16_f32` | 2×FP32 → 2×UINT16 | Packed float→int |
| `v_cvt_pk_u16_u32` | 2×UINT32 → 2×UINT16 | Packed int narrow |
| `v_cvt_pk_u8_f32` | FP32 → UINT8 (byte slot) | Pack byte into DWORD |
| `v_cvt_pknorm_i16_f16` | 2×FP16 → 2×INORM16 | Alias for pk_norm variant |
| `v_cvt_pknorm_i16_f32` | 2×FP32 → 2×INORM16 | Alias for pk_norm variant |
| `v_cvt_pknorm_u16_f16` | 2×FP16 → 2×UNORM16 | Alias for pk_norm variant |
| `v_cvt_pknorm_u16_f32` | 2×FP32 → 2×UNORM16 | Alias for pk_norm variant |
| `v_cvt_pkrtz_f16_f32` | 2×FP32 → 2×FP16 | Alias for pk_rtz variant |

### All 23 UNSUPPORTED `v_cvt_*` on gfx1100

These exist in the LLVM backend for **gfx94x (CDNA3/MI300)** and/or **gfx1200+**
but are **rejected by the assembler** for gfx1100:

| Instruction | Why not on gfx1100 |
|---|---|
| `v_cvt_f16_bf8` | FP8 ← CDNA3+ only |
| `v_cvt_f16_fp8` | FP8 ← CDNA3+ only |
| `v_cvt_f32_bf16` | BF16 native cvt ← gfx1200+ |
| `v_cvt_f32_bf8` | FP8/BF8 ← CDNA3+ only |
| `v_cvt_f32_fp8` | FP8 ← CDNA3+ only |
| `v_cvt_pk_bf16_f32` | BF16 packed ← CDNA3+/gfx1200+ |
| `v_cvt_pk_bf8_f16` | FP8/BF8 ← CDNA3+ only |
| `v_cvt_pk_bf8_f32` | FP8/BF8 ← CDNA3+ only |
| `v_cvt_pk_f16_bf8` | FP8/BF8 ← CDNA3+ only |
| `v_cvt_pk_f16_f32` | Use `v_cvt_pk_rtz_f16_f32` instead |
| `v_cvt_pk_f16_fp8` | FP8 ← CDNA3+ only |
| `v_cvt_pk_f32_bf8` | FP8/BF8 ← CDNA3+ only |
| `v_cvt_pk_f32_fp8` | FP8 ← CDNA3+ only |
| `v_cvt_pk_fp8_f16` | FP8 ← CDNA3+ only |
| `v_cvt_pk_fp8_f32` | FP8 ← CDNA3+ only |
| `v_cvt_sr_bf16_f32` | Stochastic rounding ← CDNA3+/gfx1200+ |
| `v_cvt_sr_bf8_f16` | SR + FP8 ← CDNA3+ only |
| `v_cvt_sr_bf8_f32` | SR + FP8 ← CDNA3+ only |
| `v_cvt_sr_f16_f32` | Stochastic rounding ← CDNA3+/gfx1200+ |
| `v_cvt_sr_fp8_f16` | SR + FP8 ← CDNA3+ only |
| `v_cvt_sr_fp8_f32` | SR + FP8 ← CDNA3+ only |
| `v_cvt_sr_pk_bf16_f32` | SR packed ← CDNA3+/gfx1200+ |
| `v_cvt_sr_pk_f16_f32` | SR packed ← CDNA3+/gfx1200+ |

### Notable gaps on gfx1100

- **Zero FP8/BF8 instructions** — All FP8 conversions are software-emulated
- **No `v_cvt_f32_bf16`** — BF16→FP32 must use the bit-shift trick (`v_lshlrev_b32 <<16`)
- **No `v_cvt_pk_bf16_f32`** — FP32→BF16 packed requires manual RNE rounding (~5 ops)
- **No `v_cvt_pk_f16_f32`** — Must use `v_cvt_pk_rtz_f16_f32` (round-to-zero, not RNE)
- **No stochastic rounding** — All `v_cvt_sr_*` are CDNA3+/gfx1200+

Source: `/tmp/test_all_cvt.cu` (micro-kernels isolating each conversion).
Disassembly: `hipcc -S --cuda-device-only`.

---

## Quick Summary

| Conversion | ISA instruction(s) | VALU ops | Branches | Notes |
|---|---|---|---|---|
| **INT8 → FP16** | `v_cvt_f16_i16_e32` | **1** | 0 | Load does sign-extend (`global_load_d16_i8`) |
| **UINT8 → FP16** | `v_cvt_f16_u16_e32` | **1** | 0 | Load zero-extends (`global_load_d16_u8`) |
| **INT8 → FP32** | `v_cvt_f32_i32_e32` | **1** | 0 | Load sign-extends to 32-bit (`global_load_i8`) |
| **INT16 → FP16** | `v_cvt_f16_i16_e32` | **1** | 0 | |
| **INT16 → FP32** | `v_cvt_f32_i32_e32` | **1** | 0 | |
| **INT32 → FP32** | `v_cvt_f32_i32_e32` | **1** | 0 | |
| **INT32 → FP16** | `v_cvt_f32_i32` + `v_cvt_f16_f32` | **2** | 0 | No direct i32→f16 |
| **UINT32 → FP32** | `v_cvt_f32_u32_e32` | **1** | 0 | |
| **FP16 → FP32** | `v_cvt_f32_f16_e32` | **1** | 0 | |
| **FP32 → FP16** | `v_cvt_f16_f32_e32` | **1** | 0 | |
| **BF16 → FP32** | `v_lshlrev_b32_e32 v, 16, v` | **1** | 0 | Just a left-shift! (same bits, zero lower 16) |
| **FP32 → BF16** | `v_bfe + v_or + v_cmp_u + v_add3 + v_cndmask` | **5** | 0 | RNE rounding + NaN propagation |
| **FP16 → BF16** | `v_cvt_f32_f16` + (f32→bf16 logic) | **6** | 0 | Via FP32 |
| **BF16 → FP16** | `v_lshlrev_b32` + `v_cvt_f16_f32` | **2** | 0 | Via FP32 |
| **FP16 → INT8** | `v_cvt_i16_f16_e32` | **1** | 0 | Truncation to 8-bit on store |
| **FP16 → INT16** | `v_cvt_i16_f16_e32` | **1** | 0 | |
| **FP32 → INT8** | `v_rndne_f32` + `v_cvt_i32_f32` | **2** | 0 | Explicit RNE before truncation |
| **FP32 → INT32** | `v_rndne_f32` + `v_cvt_i32_f32` | **2** | 0 | |
| **FP32 → FP64** | `v_cvt_f64_f32_e32` | **1** | 0 | |
| **FP64 → FP32** | `v_cvt_f32_f64_e32` | **1** | 0 | |
| **FP64 → FP16** | (software emulation) | **~30** | 2 | No native f64→f16 |
| **FP8 E4M3 → FP32** (HIP) | (software: CLZ + denormal normalize) | **~20** | 3 | `__hip_cvt_fp8_to_halfraw` |
| **FP8 E5M2 → FP32** (HIP) | (software: CLZ + denormal normalize) | **~25** | 3 | `__hip_cvt_fp8_to_halfraw` |
| **FP32 → FP8 E4M3** (HIP) | (software: full round + satfinite) | **~40+** | 4 | `__hip_cvt_halfraw_to_fp8` |
| **FP32 → FP8 E5M2** (HIP) | (software: full round + satfinite) | **~40+** | 4 | `__hip_cvt_halfraw_to_fp8` |
| **FP8 E4M3 → FP16** (bitmanip) | `v_lshrrev + v_and + v_lshlrev×3 + v_or×2 + v_add_nc_u16` | **~8** | 1 | Manual, skip denormals |
| **FP8 E5M2 → FP16** (bitmanip) | `v_lshlrev_b16 v0.l, 8, v0.l` | **1** | 0 | Same exponent bias as FP16! |
| **4×INT8 → 4×FP16** (packed) | 3× `v_cvt_f16_i16` + misc extract | **~10** | 0 | Vectorized 32-bit load |

---

## Key Findings for KV-Cache Design on RDNA3

### Best paths (native single-instruction)

```
INT8  → FP16:  v_cvt_f16_i16_e32      (1 cycle)
UINT8 → FP16:  v_cvt_f16_u16_e32      (1 cycle)
UINT8 → FP32:  v_cvt_f32_ubyte{0,1,2,3}  (1 cycle, reads byte lane from DWORD)
FP16  → FP32:  v_cvt_f32_f16_e32      (1 cycle)
FP32  → FP16:  v_cvt_f16_f32_e32      (1 cycle)
BF16  → FP32:  v_lshlrev_b32 <<16     (1 cycle, bit-trick)
```

### Vectorized 4×UINT8 → 4×FP32 (ubyte trick)

`v_cvt_f32_ubyte{0,1,2,3}` reads each byte lane of a DWORD without any
extract/shift instructions. The compiler uses this automatically:

```asm
global_load_b32 v3, ...              ; 1 load: 4 packed uint8
v_cvt_f32_ubyte0_e32 v0, v3          ; byte[0] → f32 (1 VALU)
v_cvt_f32_ubyte1_e32 v1, v3          ; byte[1] → f32 (1 VALU)
v_cvt_f32_ubyte2_e32 v2, v3          ; byte[2] → f32 (1 VALU)
v_cvt_f32_ubyte3_e32 v3, v3          ; byte[3] → f32 (1 VALU)
global_store_b128 v[4:5], v[0:3]     ; 1 store: 4×f32 as 128-bit
```

**4 VALU + 0 extracts** for 4 unsigned byte→float conversions. Combined with
`v_cvt_f16_f32` you get 4×UINT8→4×FP16 in 8 VALU ops total (2 ops/element).

For **signed INT8**: the compiler uses `v_bfe_i32` (sign-extend extract) +
`v_cvt_f16_i16` per element. Total for 4 bytes: ~7-10 VALU (slightly worse
than unsigned due to sign-extension, but still vastly better than FP8).

**Best KV-cache pattern for UINT8 per-tensor on RDNA3:**

```
Load 4 bytes as DWORD → v_cvt_f32_ubyte{0..3} → v_cvt_f16_f32 ×4
Scale applied post-matmul (0 extra ops in inner loop)
Total: 2 VALU/element, fully pipelined, zero branches
```

### FP8 E5M2 surprise: 1-instruction bitmanip

E5M2 and FP16 share the same exponent width (5 bits) and bias (15). Conversion
is a **pure left-shift by 8 bits** — zero-padding the mantissa:

```
FP8 E5M2: [S][EEEEE][MM]         (8 bits)
FP16:     [S][EEEEE][MMMMMMMMMM] (16 bits, lower 8 mant bits = 0)

ISA: v_lshlrev_b16 v0.l, 8, v0.l   ← 1 VALU, 1 cycle
```

This means **E5M2 KV-cache is essentially free** to dequantize on RDNA3.
Trade-off: E5M2 has less mantissa precision (2 bits) vs E4M3 (3 bits).

### FP8 E4M3: always emulated

No shortcut exists because the exponent bias differs (E4M3 bias=7 vs FP16 bias=15).
Minimum cost is ~8 integer ops (bit-manipulation without denormal handling).
HIP's native path (`__hip_cvt_fp8_to_halfraw`) adds full denormal support via
`v_clz_i32_u32` + normalization shifts, reaching ~20 ops + 3 divergent branches.

### Recommendation for KV-cache quantization on gfx1100

**Priority order (cost of dequant in attention inner loop):**

1. **FP8 E5M2 + bitmanip** — 1 VALU (`v_lshlrev_b16`), 50% VRAM saving, 2-bit mantissa
2. **INT8 per-tensor** — 1 VALU (`v_cvt_f16_i16_e32`), 50% VRAM saving, 7-bit effective range
3. **BF16 (no quant)** — 0 VALU (native), 0% VRAM saving
4. **FP8 E4M3 bitmanip** — 8 VALU + 1 branch, 50% VRAM saving
5. **FP8 E4M3 HIP native** — 20 VALU + 3 branches, 50% VRAM saving (CURRENT vLLM path)

---

## Detailed ISA Listings

### INT8 → FP16 (1 instruction)

```asm
; Load byte with hardware sign-extension to 16-bit lane
global_load_d16_i8 v0, v[3:4], off
s_waitcnt vmcnt(0)
; Native conversion: signed int16 → float16
v_cvt_f16_i16_e32 v0.l, v0.l
global_store_b16 v[1:2], v0, off
```

### UINT8 → FP16 (1 instruction)

```asm
global_load_d16_u8 v0, v[3:4], off
s_waitcnt vmcnt(0)
v_cvt_f16_u16_e32 v0.l, v0.l
global_store_b16 v[1:2], v0, off
```

### BF16 → FP32 (1 instruction, bit-trick)

```asm
; BF16 is upper 16 bits of FP32 — just shift left
global_load_u16 v3, v[3:4], off
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v2, 16, v3
global_store_b32 v[0:1], v2, off
```

### FP32 → BF16 (5 instructions, RNE rounding)

```asm
global_load_b32 v3, v[3:4], off
s_waitcnt vmcnt(0)
v_bfe_u32 v4, v3, 16, 1          ; extract bit 16 (round-to-nearest-even)
v_or_b32_e32 v2, 0x400000, v3    ; NaN quieting
v_cmp_u_f32_e32 vcc_lo, v3, v3   ; is NaN?
v_add3_u32 v4, v4, v3, 0x7fff    ; round (banker's rounding)
v_cndmask_b32_e32 v2, v4, v2, vcc_lo  ; select NaN path if needed
global_store_d16_hi_b16 v[0:1], v2, off ; store upper 16 bits
```

### FP8 E5M2 → FP16 bitmanip (1 instruction!)

```asm
; Same exponent bias (15) and width (5) — just left-shift 8
global_load_d16_u8 v0, v[3:4], off
s_waitcnt vmcnt(0)
v_lshlrev_b16 v0.l, 8, v0.l
global_store_b16 v[1:2], v0, off
```

### FP8 E4M3 → FP16 bitmanip (8 ops + 1 branch)

```asm
global_load_d16_hi_u8 v0, v[0:1], off
s_waitcnt vmcnt(0)
v_lshrrev_b16 v0.l, 3, v0.h       ; extract exponent field
v_and_b16 v1.l, v0.l, 15          ; mask 4-bit exponent
v_mov_b16_e32 v0.l, 0             ; default = zero
v_cmpx_ne_u16_e32 0, v1.l         ; skip if exp==0 (zero/denormal)
s_cbranch_execz .LBB_skip
; -- Normal case --
v_lshlrev_b16 v0.l, 8, v0.h       ; sign << 8
v_lshlrev_b16 v1.l, 10, v1.l      ; exp << 10
v_and_b16 v0.h, v0.h, 7           ; extract 3-bit mantissa
v_and_b16 v0.l, 0x8000, v0.l      ; isolate sign bit (bit 15)
v_lshlrev_b16 v0.h, 7, v0.h       ; mantissa << 7 (to fp16 position)
v_or_b16 v0.l, v1.l, v0.l         ; combine exp | sign
v_or_b16 v0.l, v0.l, v0.h         ; combine | mantissa
v_add_nc_u16 v0.l, 0x2000, v0.l   ; rebias: +8 << 10 = 0x2000
.LBB_skip:
global_store_b16 v[1:2], v0, off
```

### FP8 E4M3 → FP32 HIP native (~20 ops + 3 branches)

```asm
; Full IEEE-compliant conversion with denormal handling
global_load_d16_u8 v0, v[3:4], off
v_mov_b32_e32 v3, 0
s_waitcnt vmcnt(0)
v_cmpx_ne_u16_e32 0, v0.l             ; branch: skip if zero
s_cbranch_execz .LBB_zero
v_mov_b32_e32 v3, 0x7fc02000          ; default: NaN
v_cmpx_ne_u16_e32 0x80, v0.l          ; branch: skip if neg-zero
s_cbranch_execz .LBB_negzero
v_dual_mov_b32 v4, 0 :: v_dual_and_b32 v3, 7, v0   ; mantissa
v_bfe_u32 v5, v0, 3, 4                ; exponent
v_cmpx_eq_u32_e32 0, v5               ; branch: is denormal?
; -- Denormal normalization --
v_clz_i32_u32_e32 v5, v3              ; count leading zeros
v_min_u32_e32 v5, 32, v5
v_subrev_nc_u32_e32 v6, 28, v5        ; compute shift amount
v_sub_nc_u32_e32 v5, 29, v5           ; compute adjusted exponent
v_lshlrev_b64 v[3:4], v6, v[3:4]     ; normalize mantissa
v_and_b32_e32 v3, 7, v3              ; mask to 3 bits
; -- Reassemble FP16 bits --
v_lshlrev_b32_e32 v0, 8, v0           ; sign to position
v_lshl_add_u32 v4, v5, 10, 0x1c00    ; exponent + bias adjust
v_and_or_b32 v0, 0x8000, v0, v4      ; sign | exp
v_lshl_or_b32 v0, v3, 7, v0          ; | mantissa
v_cvt_f32_f16_e32 v3, v0.l            ; THEN convert fp16→fp32 (!!)
.LBB_negzero:
.LBB_zero:
global_store_b32 v[0:1], v3, off
```

### 4×INT8 → 4×FP16 vectorized

```asm
global_load_b32 v0, v[3:4], off        ; load 4 bytes packed
s_waitcnt vmcnt(0)
v_bfe_i32 v4, v0, 0, 8                ; extract byte 0 (sign-extend)
v_lshrrev_b32_e32 v3, 8, v0           ; byte 1
v_mov_b16_e32 v5.l, v0.h              ; byte 2
v_ashrrev_i32_e32 v6, 24, v0          ; byte 3 (arithmetic shift)
v_cvt_f16_i16_e32 v5.l, v0.l          ; convert byte 0
v_bfe_i32 v4, v5, 0, 8                ; extract byte 2
v_bfe_i32 v3, v3, 0, 8                ; extract byte 1
v_cvt_f32_i32_e32 v6, v6              ; byte 3 → f32 (compiler quirk)
v_cvt_f16_i16_e32 v7.h, v3.l          ; convert byte 1
v_cvt_f16_f32_e32 v4.h, v6            ; byte 3: f32 → f16
v_cvt_f16_i16_e32 v3.l, v0.l          ; convert byte 2
; ... pack into uint64 and store
global_store_b64 v[0:1], v[2:3], off
```

Note: compiler uses `v_cvt_f16_i16` for 3 of 4 bytes but routes byte 3
through `v_cvt_f32_i32 + v_cvt_f16_f32` (2 ops). Total: ~10 VALU for 4 elements.

---

## Memory Load Variants (hardware type conversion on load)

gfx1100 supports type-converting loads that do sign/zero extension in the
memory subsystem at no VALU cost:

| Instruction | Loads | Extends to | Notes |
|---|---|---|---|
| `global_load_d16_i8` | 1 byte | INT16 (sign-ext) | Writes to `.l` half of VGPR |
| `global_load_d16_u8` | 1 byte | UINT16 (zero-ext) | Writes to `.l` half of VGPR |
| `global_load_d16_hi_u8` | 1 byte | UINT16 (zero-ext) | Writes to `.h` half of VGPR |
| `global_load_d16_b16` | 2 bytes | 16-bit (no conv) | Raw fp16/int16 load |
| `global_load_i8` | 1 byte | INT32 (sign-ext) | Full 32-bit VGPR |
| `global_load_i16` | 2 bytes | INT32 (sign-ext) | Full 32-bit VGPR |
| `global_load_u8` | 1 byte | UINT32 (zero-ext) | Full 32-bit VGPR |
| `global_load_u16` | 2 bytes | UINT32 (zero-ext) | Full 32-bit VGPR |

These combine with conversion instructions for the optimal pattern:

```
global_load_d16_i8 → v_cvt_f16_i16  (INT8→FP16: 1 load + 1 VALU)
global_load_i8     → v_cvt_f32_i32  (INT8→FP32: 1 load + 1 VALU)
```

---

## Implications for Triton Codegen

When Triton compiles `data.to(tl.float16)` on FP8 tensors targeting gfx1100,
it likely goes through the HIP `__hip_cvt_fp8_to_halfraw` path — the **worst**
option (~20 ops). A Triton `tl.inline_asm` or a pre-conversion in a HIP kernel
using INT8 storage would be dramatically faster.

For the `triton_unified_attention.py` KV-cache dequant path, the optimal strategies are:

1. **Store as E5M2** → `v_lshlrev_b16` in Triton (if expressible as bitcast + shift)
2. **Store as INT8** → Triton's `data.to(tl.int16).to(tl.float16)` may emit `v_cvt_f16_i16`
3. **Store as E4M3 + use inline bitmanip** → ~8 ops (manual, avoids HIP soft-float)

---

## Build / Reproduce

```bash
# Copy source to container
docker cp /tmp/test_all_cvt.cu vllm-vllm1-1:/tmp/

# Compile to ISA assembly (device-only)
docker exec vllm-vllm1-1 bash -c \
  "/opt/rocm/bin/hipcc -x hip --offload-arch=gfx1100 -O3 -S \
   /tmp/test_all_cvt.cu -o /tmp/test_all_cvt_gfx1100.s --cuda-device-only"

# View specific kernel ISA
docker exec vllm-vllm1-1 bash -c \
  "sed -n '/^cvt_i8_f16:/,/s_endpgm/p' /tmp/test_all_cvt_gfx1100.s"
```

---

*Generated 2026-05-01 from empirical ISA disassembly on ROCm 7.2 / hipcc 7.2.1 / gfx1100.*
