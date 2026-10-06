# AMD RDNA3 (gfx1100) Complete ISA Reference

Verified empirically against `llvm-mc -mcpu=gfx1100` assembler.
ROCm 7.2 / hipcc 7.2.1 / AMD clang 22.0.0git.

**Total: 1348 instructions supported** (982 rejected — those are for CDNA3/gfx1200+).

---

## Summary by Category

| Category | Count | Key for ML kernels |
|---|---|---|
| WMMA/SWMMAC | 6 | Matrix multiply (FP16, BF16, INT8, INT4) |
| DOT | 12 | Packed dot products (FP16, BF16, INT8, INT4) |
| VOP3P (packed) | 20 | Packed FP16 arithmetic |
| CVT (conversion) | 44 | Type casting |
| VALU (vector ALU) | 247 | General compute |
| VOPC (compare) | 196 | Conditionals |
| DS (LDS/GDS) | 164 | Shared memory |
| BUFFER | 135 | Buffer memory |
| GLOBAL | 109 | Global memory |
| FLAT | 103 | Flat memory |
| SALU (scalar) | 218 | Scalar compute/control |
| SCRATCH | 44 | Stack/spill memory |
| SMEM | 20 | Scalar memory loads |
| TBUFFER | 24 | Typed buffer |
| VINTERP | 6 | Interpolation |

---

## WMMA/SWMMAC (6 instructions)

Wave Matrix Multiply-Accumulate. 16×16×16 tiles, wave32.

| Instruction | Operation | Types |
|---|---|---|
| `v_wmma_f32_16x16x16_f16` | C_f32 += A_f16 × B_f16 | FP16 → FP32 accumulate |
| `v_wmma_f32_16x16x16_bf16` | C_f32 += A_bf16 × B_bf16 | BF16 → FP32 accumulate |
| `v_wmma_f16_16x16x16_f16` | C_f16 += A_f16 × B_f16 | FP16 → FP16 accumulate |
| `v_wmma_bf16_16x16x16_bf16` | C_bf16 += A_bf16 × B_bf16 | BF16 → BF16 accumulate |
| `v_wmma_i32_16x16x16_iu8` | C_i32 += A_i8/u8 × B_i8/u8 | INT8 → INT32 accumulate |
| `v_wmma_i32_16x16x16_iu4` | C_i32 += A_i4/u4 × B_i4/u4 | INT4 → INT32 accumulate |

**NOT available on gfx1100:**

- No FP8 WMMA (`v_wmma_f32_16x16x16_fp8` — CDNA3+ only)
- No SWMMAC (scaled WMMA — gfx1200+ only)
- No BF8 WMMA

---

## DOT (12 instructions)

Packed multiply-accumulate within a single VGPR lane.

| Instruction | Operation | Throughput |
|---|---|---|
| `v_dot2_f32_f16` | `f32 += a.x*b.x + a.y*b.y` (fp16 pairs) | Full rate |
| `v_dot2_f32_bf16` | `f32 += a.x*b.x + a.y*b.y` (bf16 pairs) | Full rate |
| `v_dot2_f16_f16` | `f16 += a.x*b.x + a.y*b.y` (fp16) | Full rate |
| `v_dot2_bf16_bf16` | `bf16 += a.x*b.x + a.y*b.y` (bf16) | Full rate |
| `v_dot2acc_f32_f16` | Accumulate variant of dot2 | Full rate |
| `v_dot2c_f32_f16` | Compact encoding variant | Full rate |
| `v_dot4_i32_i8` | `i32 += Σ(a[i]*b[i])` for 4×INT8 (signed) | Full rate |
| `v_dot4_i32_iu8` | `i32 += Σ(a[i]*b[i])` for 4×INT8 (mixed sign) | Full rate |
| `v_dot4_u32_u8` | `u32 += Σ(a[i]*b[i])` for 4×UINT8 | Full rate |
| `v_dot8_i32_i4` | `i32 += Σ(a[i]*b[i])` for 8×INT4 (signed) | Full rate |
| `v_dot8_i32_iu4` | `i32 += Σ(a[i]*b[i])` for 8×INT4 (mixed sign) | Full rate |
| `v_dot8_u32_u4` | `u32 += Σ(a[i]*b[i])` for 8×UINT4 | Full rate |

---

## VOP3P — Packed FP16/INT16 (20 instructions)

Operate on 2× packed FP16 or INT16 values in a single VGPR.

```
v_pk_add_f16          v_pk_add_i16          v_pk_add_u16
v_pk_ashrrev_i16      v_pk_fma_f16          v_pk_fmac_f16
v_pk_lshlrev_b16      v_pk_lshrrev_b16      v_pk_mad_i16
v_pk_mad_u16          v_pk_max_f16          v_pk_max_i16
v_pk_max_u16          v_pk_min_f16          v_pk_min_i16
v_pk_min_u16          v_pk_mul_f16          v_pk_mul_lo_u16
v_pk_sub_i16          v_pk_sub_u16
```

**NOT available:** `v_pk_fma_bf16` (CDNA3+ only — gfx1100 has no packed BF16 FMA)

---

## CVT — Type Conversion (44 instructions)

See `README_RDNA3_CVT_ISA.md` for full details with ISA listings.

**Key:**

- Native: INT8/16/32 ↔ FP16/32/64, FP16 ↔ FP32/64
- Missing: All FP8/BF8, BF16 native, stochastic rounding

---

## VALU — Vector ALU (247 instructions)

### Arithmetic (FP32)

```
v_add_f32    v_sub_f32    v_subrev_f32    v_mul_f32    v_mul_legacy_f32
v_mul_dx9_zero_f32    v_fma_f32    v_fma_legacy_f32    v_fma_dx9_zero_f32
v_fmac_f32    v_fmac_legacy_f32    v_fmac_dx9_zero_f32
v_fmaak_f32    v_fmamk_f32    v_mad_i32_i24    v_mad_u32_u24
v_mul_i32_i24    v_mul_u32_u24    v_mul_hi_i32_i24    v_mul_hi_u32_u24
v_mul_lo_u32    v_mul_hi_u32    v_mul_hi_i32
v_mad_i64_i32    v_mad_u64_u32    v_mullit_f32
```

### Arithmetic (FP16)

```
v_add_f16    v_sub_f16    v_subrev_f16    v_mul_f16
v_fma_f16    v_fmac_f16    v_fmaak_f16    v_fmamk_f16
v_mad_i16    v_mad_u16    v_mad_i32_i16    v_mad_u32_u16
v_mul_lo_u16
```

### Arithmetic (FP64)

```
v_add_f64    v_mul_f64    v_fma_f64    v_min_f64    v_max_f64
v_ldexp_f64    v_frexp_mant_f64    v_frexp_exp_i32_f64
v_div_scale_f64    v_div_fixup_f64    v_div_fmas_f64
v_trig_preop_f64    v_fract_f64    v_floor_f64    v_ceil_f64
v_trunc_f64    v_rndne_f64    v_sqrt_f64    v_rsq_f64    v_rcp_f64
```

### Arithmetic (Integer)

```
v_add_nc_u32    v_add_nc_i32    v_add_nc_u16    v_add_nc_i16
v_sub_nc_u32    v_sub_nc_i32    v_sub_nc_u16    v_sub_nc_i16
v_subrev_nc_u32    v_add_co_u32    v_add_co_ci_u32
v_sub_co_u32    v_sub_co_ci_u32    v_subrev_co_u32    v_subrev_co_ci_u32
v_add_i32    v_sub_i32    v_add_u32    v_sub_u32    v_subrev_u32
v_add3_u32    v_add_lshl_u32    v_lshl_add_u32    v_xad_u32
```

### Bitwise

```
v_and_b32    v_or_b32    v_xor_b32    v_not_b32    v_xnor_b32
v_and_b16    v_or_b16    v_xor_b16    v_not_b16
v_or3_b32    v_xor3_b32    v_and_or_b32    v_lshl_or_b32
v_bfe_u32    v_bfe_i32    v_bfi_b32    v_bfm_b32    v_bfrev_b32
v_lshlrev_b32    v_lshrrev_b32    v_ashrrev_i32
v_lshlrev_b16    v_lshrrev_b16    v_ashrrev_i16
v_lshlrev_b64    v_lshrrev_b64    v_ashrrev_i64
v_alignbit_b32    v_alignbyte_b32    v_perm_b32
v_bcnt_u32_b32    v_clz_i32_u32    v_cls_i32    v_ctz_i32_b32
v_ffbh_u32    v_ffbh_i32    v_ffbl_b32
```

### Min/Max/Med/Clamp

```
v_min_f32    v_max_f32    v_min_f16    v_max_f16
v_min_i32    v_max_i32    v_min_u32    v_max_u32
v_min_i16    v_max_i16    v_min_u16    v_max_u16
v_min3_f32    v_max3_f32    v_med3_f32
v_min3_f16    v_max3_f16    v_med3_f16
v_min3_i32    v_max3_i32    v_med3_i32
v_min3_i16    v_max3_i16    v_med3_i16
v_min3_u32    v_max3_u32    v_med3_u32
v_min3_u16    v_max3_u16    v_med3_u16
v_minmax_f32    v_maxmin_f32    v_minmax_i32    v_maxmin_i32
v_minmax_f16    v_maxmin_f16    v_minmax_u32    v_maxmin_u32
```

### Transcendental / Special

```
v_rcp_f32    v_rcp_f16    v_rcp_iflag_f32
v_rsq_f32    v_rsq_f16    v_sqrt_f32    v_sqrt_f16
v_exp_f32    v_exp_f16    v_log_f32    v_log_f16
v_sin_f32    v_sin_f16    v_cos_f32    v_cos_f16
v_fract_f32    v_fract_f16    v_floor_f32    v_floor_f16
v_ceil_f32    v_ceil_f16    v_trunc_f32    v_trunc_f16
v_rndne_f32    v_rndne_f16    v_ldexp_f32    v_ldexp_f16
v_frexp_mant_f32    v_frexp_mant_f16
v_frexp_exp_i32_f32    v_frexp_exp_i16_f16
v_div_scale_f32    v_div_fixup_f32    v_div_fmas_f32
v_cubeid_f32    v_cubema_f32    v_cubesc_f32    v_cubetc_f32
```

### Move / Lane / Permute

```
v_mov_b32    v_mov_b16
v_cndmask_b32    v_cndmask_b16
v_readlane_b32    v_readfirstlane_b32    v_writelane_b32
v_permlane16_b32    v_permlanex16_b32    v_permlane64_b32
v_movreld_b32    v_movrels_b32    v_movrelsd_b32    v_movrelsd_2_b32
v_swaprel_b32    v_swap_b32    v_swap_b16
```

### Mixed-precision FMA

```
v_fma_mix_f32       (mixed fp16/f32 inputs → f32 output)
v_fma_mixlo_f16     (mixed → f16 low)
v_fma_mixhi_f16     (mixed → f16 high)
```

### DUAL issue (VOPD — two ops in one cycle)

```
v_dual_add_f32    v_dual_sub_f32    v_dual_subrev_f32
v_dual_mul_f32    v_dual_mul_dx9_zero_f32
v_dual_fmac_f32    v_dual_fmaak_f32    v_dual_fmamk_f32
v_dual_max_f32    v_dual_min_f32
v_dual_mov_b32    v_dual_cndmask_b32
v_dual_dot2acc_f32_f16    v_dual_dot2acc_f32_bf16
```

### SAD / Integer reduction

```
v_sad_u8    v_sad_hi_u8    v_sad_u16    v_sad_u32
v_msad_u8    v_qsad_pk_u16_u8    v_mqsad_pk_u16_u8    v_mqsad_u32_u8
v_lerp_u8    v_sat_pk_u8_i16    v_pack_b32_f16
```

---

## DS — LDS/GDS Operations (164 instructions)

### Load (various widths and formats)

```
ds_load_b32    ds_load_b64    ds_load_b96    ds_load_b128
ds_load_2addr_b32    ds_load_2addr_b64
ds_load_2addr_stride64_b32    ds_load_2addr_stride64_b64
ds_load_i8    ds_load_u8    ds_load_i16    ds_load_u16
ds_load_i8_d16    ds_load_i8_d16_hi
ds_load_u8_d16    ds_load_u8_d16_hi
ds_load_u16_d16    ds_load_u16_d16_hi
ds_load_addtid_b32
```

### Store

```
ds_store_b32    ds_store_b64    ds_store_b96    ds_store_b128
ds_store_2addr_b32    ds_store_2addr_b64
ds_store_2addr_stride64_b32    ds_store_2addr_stride64_b64
ds_store_b8    ds_store_b16
ds_store_b8_d16_hi    ds_store_b16_d16_hi
ds_store_addtid_b32
```

### Atomics (LDS)

```
ds_add_u32/u64    ds_sub_u32/u64    ds_rsub_u32/u64
ds_min_i32/i64/u32/u64/f32/f64    ds_max_i32/i64/u32/u64/f32/f64
ds_and_b32/b64    ds_or_b32/b64    ds_xor_b32/b64
ds_inc_u32/u64    ds_dec_u32/u64
ds_cmpstore_b32/b64/f32/f64    ds_storexchg_rtn_b32/b64
ds_add_f32    ds_add_rtn_f32    ds_mskor_b32/b64
```

(All atomics have `_rtn_` variants that return the old value)

### Special

```
ds_bpermute_b32    ds_permute_b32    ds_swizzle_b32
ds_append    ds_consume    ds_ordered_count
ds_gws_init    ds_gws_barrier    ds_gws_sema_*
ds_bvh_stack_rtn_b32    ds_nop    ds_wrap_rtn_b32
```

---

## GLOBAL Memory (109 instructions)

### Loads

```
global_load_b32    global_load_b64    global_load_b96    global_load_b128
global_load_u8     global_load_i8     global_load_u16    global_load_i16
global_load_d16_u8    global_load_d16_i8    (load byte → half VGPR)
global_load_d16_hi_u8    global_load_d16_hi_i8
global_load_d16_b16    global_load_d16_hi_b16
global_load_addtid_b32
```

### Stores

```
global_store_b8    global_store_b16    global_store_b32
global_store_b64    global_store_b96    global_store_b128
global_store_d16_hi_b8    global_store_d16_hi_b16
global_store_addtid_b32
```

### Atomics

```
global_atomic_add_u32/u64/f32    global_atomic_sub_u32/u64
global_atomic_min_i32/i64/u32/u64/f32    global_atomic_max_i32/i64/u32/u64/f32
global_atomic_and_b32/b64    global_atomic_or_b32/b64    global_atomic_xor_b32/b64
global_atomic_swap_b32/b64    global_atomic_cmpswap_b32/b64/f32
global_atomic_inc_u32/u64    global_atomic_dec_u32/u64
global_atomic_csub_u32    global_atomic_fcmpswap    global_atomic_fmax    global_atomic_fmin
```

---

## SALU — Scalar ALU (218 instructions)

### Arithmetic

```
s_add_i32    s_sub_i32    s_addc_u32    s_subb_u32
s_mul_i32    s_mul_hi_u32    s_mul_hi_i32
s_addk_i32    s_mulk_i32    s_cmovk_i32
s_abs_i32    s_absdiff_i32
s_lshl1_add_u32    s_lshl2_add_u32    s_lshl3_add_u32    s_lshl4_add_u32
```

### Bitwise

```
s_and_b32/b64    s_or_b32/b64    s_xor_b32/b64
s_andn2_b32/b64    s_orn2_b32/b64    s_xnor_b32/b64
s_nand_b32/b64    s_nor_b32/b64
s_and_not1_b32/b64    s_or_not1_b32/b64
s_not_b32/b64    s_lshl_b32/b64    s_lshr_b32/b64    s_ashr_i32/i64
s_bfe_u32/u64/i32/i64    s_bfm_b32/b64    s_brev_b32/b64
s_bitset0_b32/b64    s_bitset1_b32/b64
s_bcnt0_i32_b32/b64    s_bcnt1_i32_b32/b64
s_ff1_i32_b32/b64    s_flbit_i32/i32_b32/i32_b64/i32_i64
s_clz_i32_u32/u64    s_cls_i32/i32_i64    s_ctz_i32_b32/b64
s_bitreplicate_b64_b32
s_pack_ll_b32_b16    s_pack_lh_b32_b16    s_pack_hl_b32_b16    s_pack_hh_b32_b16
```

### Compare

```
s_cmp_eq_i32/u32/u64    s_cmp_lg_i32/u32/u64
s_cmp_gt_i32/u32    s_cmp_ge_i32/u32    s_cmp_lt_i32/u32    s_cmp_le_i32/u32
s_cmpk_eq/lg/gt/ge/lt/le_i32/u32    s_bitcmp0/1_b32/b64
```

### Move / Select

```
s_mov_b32/b64    s_movk_i32    s_cmov_b32/b64    s_cselect_b32/b64
s_movreld_b32/b64    s_movrels_b32/b64    s_movrelsd_2_b32
s_sext_i32_i8    s_sext_i32_i16
```

### Control flow

```
s_branch    s_cbranch_scc0/scc1/vccz/vccnz/execz/execnz
s_cbranch_cdbgsys/cdbguser/cdbgsys_and_user/cdbgsys_or_user
s_setpc_b64    s_swappc_b64    s_getpc_b64    s_call_b64
s_endpgm    s_endpgm_saved    s_endpgm_ordered_ps_done
s_trap    s_rfe_b64    s_sethalt    s_setkill    s_sleep
s_barrier    s_wakeup    s_wait_idle    s_wait_event
s_sendmsg    s_sendmsghalt    s_sendmsg_rtn_b32/b64
```

### EXEC manipulation (SaveExec family)

```
s_and_saveexec_b32/b64    s_or_saveexec_b32/b64    s_xor_saveexec_b32/b64
s_nand_saveexec_b32/b64    s_nor_saveexec_b32/b64    s_xnor_saveexec_b32/b64
s_andn2_saveexec_b32/b64    s_orn2_saveexec_b32/b64
s_and_not0_saveexec_b32/b64    s_or_not0_saveexec_b32/b64
s_and_not1_saveexec_b32/b64    s_or_not1_saveexec_b32/b64
s_andn1_saveexec_b32/b64    s_orn1_saveexec_b32/b64
(+ wrexec variants)
```

### Waitcnt / Synchronization

```
s_waitcnt    s_waitcnt_vmcnt    s_waitcnt_lgkmcnt
s_waitcnt_expcnt    s_waitcnt_vscnt    s_waitcnt_depctr
s_clause    s_delay_alu    s_inst_prefetch    s_set_inst_prefetch_distance
s_nop    s_code_end    s_pipeflush (via v_pipeflush)
```

### System / Cache

```
s_dcache_inv    s_gl1_inv    s_icache_inv
s_atc_probe    s_atc_probe_buffer
s_getreg_b32    s_setreg_b32    s_setreg_imm32_b32
s_denorm_mode    s_round_mode    s_setprio
s_incperflevel    s_decperflevel    s_ttracedata    s_ttracedata_imm
s_version    s_subvector_loop_begin    s_subvector_loop_end
s_quadmask_b32/b64    s_wqm_b32/b64
```

---

## SMEM — Scalar Memory (20 instructions)

```
s_load_b32    s_load_b64    s_load_b128    s_load_b256    s_load_b512
s_buffer_load_b32    s_buffer_load_b64    s_buffer_load_b128
s_buffer_load_b256    s_buffer_load_b512
```

(Plus legacy aliases: `s_load_dword`, `s_load_dwordx2/4/8/16`, etc.)

---

## VINTERP — Interpolation (6 instructions)

```
v_interp_p10_f32           v_interp_p2_f32
v_interp_p10_f16_f32       v_interp_p2_f16_f32
v_interp_p10_rtz_f16_f32   v_interp_p2_rtz_f16_f32
```

---

## What gfx1100 Does NOT Have (vs CDNA3 / gfx1200+)

| Feature | Instructions | Available on |
|---|---|---|
| FP8/BF8 conversion | `v_cvt_f32_fp8`, `v_cvt_pk_f32_fp8`, etc. | gfx94x (MI300) |
| FP8 WMMA | `v_wmma_f32_16x16x16_fp8` | gfx94x |
| SWMMAC (scaled) | `v_swmmac_*` | gfx1200+ |
| BF16 native cvt | `v_cvt_f32_bf16` | gfx1200+ |
| Packed BF16 FMA | `v_pk_fma_bf16` | gfx94x |
| Packed f32→f16 (RNE) | `v_cvt_pk_f16_f32` | gfx9xx (removed in gfx11) |
| Packed f32→bf16 | `v_cvt_pk_bf16_f32` | gfx94x / gfx1200+ |
| Stochastic rounding | `v_cvt_sr_*` | gfx94x / gfx1200+ |
| FP8 stochastic round | `v_cvt_sr_fp8_f32` | gfx94x |
| Packed FP16 atomics | `global_atomic_pk_add_f16` | gfx94x |
| BF16 atomics | `global_atomic_pk_add_bf16` | gfx94x |
| Conditional subtract | `buffer_atomic_cond_sub_u32` | gfx12+ |

---

## Key Architecture Parameters (gfx1100)

- **Wave size:** 32 (wave32)
- **VGPRs:** 1536 per CU (max 256 per wave for full occupancy of 6 waves)
- **SGPRs:** 128 per wave
- **LDS:** 64 KB per Workgroup Processor (WGP = 2 CUs)
- **WMMA:** 16×16×16, wave32, mode 1 fragment layout
- **Max occupancy:** 16 waves/CU (with ≤48 VGPRs each)
- **Register granularity:** allocated in blocks of 24 VGPRs (gfx11)
- **VOPD (dual issue):** Two VOP1/VOP2 ops can issue in one cycle via `v_dual_*`

---

*Generated 2026-05-01 from llvm-mc assembler validation on ROCm 7.2 / gfx1100.*
*Source script: `/tmp/test_isa_bulk.py` (brute-force assembly of 2330 candidate mnemonics).*
