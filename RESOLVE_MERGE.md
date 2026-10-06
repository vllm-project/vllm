# Resolución de merge: `main` → `perf/rdna3_full_stack`

**Fecha:** 2026-05-21
**Rama destino:** `perf/rdna3_full_stack` (`dc2612d04`)
**Rama fusionada:** `main` (`2a43b407c`)
**merge-base:** `0a9362d6a`

main aportaba **537 commits** nuevos; la rama RDNA3 tenía **131** propios. El merge
toca código central de cuantización GPTQ y de los kernels de atención Triton, que
ambas ramas habían refactorizado de forma divergente.

## Resumen de la divergencia estructural

main introdujo dos cambios de fondo que generaron la mayoría de los conflictos:

1. **Split de la extensión C++ `_C_stable_libtorch`.** Las ops GPTQ y de
   cuantización (`gptq_gemm`, `gptq_shuffle`, `static/dynamic_scaled_fp8_quant`,
   `static/dynamic_scaled_int8_quant`) se movieron del target `_C` a un target
   nuevo registrado con `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops)` (ABI estable de
   libtorch). **Importante:** se siguen registrando en el namespace
   `torch.ops._C`, solo cambia el `.so` que las provee (`_C_stable_libtorch.abi3.so`).
2. **Movimiento de directorio GPTQ:** `csrc/quantization/gptq/` →
   `csrc/libtorch_stable/quantization/gptq/`.

main, además, ya soportaba de forma independiente `int8_per_token_head` y
`fp8_per_token_head` (KV_QUANT_MODE 2/3) y añadió soporte de *tensor descriptors*
(`USE_TD`, para Intel Xe2/Xe3) en el kernel unificado de atención.

## Conflictos de ubicación de archivo (6)

Los kernels/headers RDNA3 GPTQ se reubicaron junto al resto de GPTQ en el nuevo
directorio de main, **pero permanecen compilados en el target `_C`** (usan la API
completa `torch/all.h`, no son compatibles con el ABI estable):

- `csrc/libtorch_stable/quantization/gptq/q_gemm_rdna3.cu`
- `csrc/libtorch_stable/quantization/gptq/q_gemm_rdna3_wmma.cu`
- `csrc/libtorch_stable/quantization/gptq/qdq_4_rdna3.cuh`
- `.../README_RDNA3.md`, `README_RDNA3_FULL_ISA.md`, `README_RDNA3_CVT_ISA.md`

## Conflictos de contenido (7)

### `CMakeLists.txt`
main quitó de `VLLM_EXT_SRC` las fuentes que pasaron a `VLLM_STABLE_EXT_SRC`
(`q_gemm.cu`, `w8a8/int8/scaled_quant.cu`, `w8a8/fp8/common.cu`). Resolución:
adoptar la estructura de main y añadir nuestros dos kernels rdna3 a
`VLLM_EXT_SRC` con la ruta nueva `csrc/libtorch_stable/quantization/gptq/...`.

### `csrc/ops.h` y `csrc/torch_bindings.cpp`
main eliminó de `_C` las declaraciones/registros de `gptq_gemm`, `gptq_shuffle`
y las quant fp8/int8 (ahora en el binding stable). Resolución: **conservar solo
las declaraciones/registros RDNA3** (`paged_prefill_attn_rdna3_int8/int4`,
`reshape_cache_int4_rdna3`, `rht_rotate_inplace_rdna3`, `pth_decode_int4/int8_rdna3`,
`gptq_gemm_rdna3` + variantes `_wmma`/`_probe`/`_dump`/`_lds_check`) y eliminar las
upstream movidas, evitando doble registro en `torch.ops._C`.

### `docs/design/attention_backends.md`
main añadió la columna **Non-Causal** y eliminó la fila `TREE_ATTN`. Resolución:
adoptar la tabla de main pero conservar nuestros KV dtypes extra en `TRITON_ATTN`
(`int2_per_token_head`, `int4_per_token_head`).

### `vllm/model_executor/layers/attention/attention.py`
Ambos lados resuelven el `kv_cache_scheme` del checkpoint. Resolución combinada:
nuestra detección extendida `get_kv_cache_scheme_dtype` (mapea a los dtypes
per-token-head int2/int4/int8/fp8) **+** la guarda de main de que un dtype
elegido explícitamente por el usuario gana (`scheme_dtype is not None and
kv_cache_dtype == "auto"`).

### `vllm/v1/attention/backends/triton_attn.py`
- Constructor: se conservan **ambos** bloques (flags fast-path RDNA3 INT4/INT8 de
  HEAD + selección de *tensor descriptors* `use_td` de main).
- Path fallback fp8/int8: se adopta la condición amplia de main
  (`is_quantized_kv_cache(...) and key_cache.dtype != fp8_dtype`) y se conserva
  el assert de q_scale==1.0 de HEAD, guardado por `kv_cache_dtype.startswith("fp8")`.

### `vllm/v1/attention/ops/triton_unified_attention.py`
El conflicto más profundo. Se combinaron las dos refactorizaciones:
- **Helpers TD de main** (`_load_q_td`, `_load_kv_tile_td`, `_store_output_td`) y su
  `_cast_kv_tile` local — se usa `_cast_kv_tile` de forma consistente (se eliminó
  el import del helper equivalente `cast_kv_tile`, funcionalmente idéntico).
- **`USE_PER_TOKEN_HEAD_SCALES = KV_QUANT_MODE >= 2`** + `USE_FP8_Q_DESCALE` + assert
  TD de main (equivalente para nuestros modos 2/3; INT4=4/INT2=5 van por factories
  separadas y no llegan a este kernel).
- **Carga KV combinada:** rama `USE_TD` (main) / `else { QK_INT8_WMMA (HEAD) /
  normal }`, preservando el dot int8 WMMA/MFMA en gfx1100.
- **Dispatch INT4/INT2** de HEAD a `triton_quant_kv` factories (se quitó el
  `assert q_descale is None` para no romper la query fp8 de main).
- **Lanzamiento del kernel:** se conservan todos los kwargs de ambos lados
  (`QK_INT8_WMMA` + `launch_kwargs` RDNA3 de HEAD; `Q_IS_FP8`, `USE_TD`,
  `USE_TD_QO` de main).

## Verificación (build + run en contenedor `vllm-vllm1-1`)

Workflow: copiar fuente a `/tmp/vllm_build/vllm_jartx`, `build_ext --inplace`,
copiar `.so` + árbol Python a dist-packages, relanzar serve.

- ✅ Build limpio. Ahora genera **dos** extensiones C (`_C.abi3.so` +
  `_C_stable_libtorch.abi3.so`) más `spinloop.abi3.so` (nueva de main).
- ✅ Ops registradas en `torch.ops._C`: RDNA3 (`gptq_gemm_rdna3`, `pth_decode_*`,
  `paged_prefill_attn_rdna3_*`) y upstream movidas (`gptq_gemm`, `gptq_shuffle`,
  `static_scaled_fp8_quant`).
- ✅ Server arranca con **RDNA3W4A16LinearKernel** + **TRITON_ATTN** +
  `kv_cache_dtype=int8_per_token_head` (Qwen3.6-27B-GPTQ-W4A16, TP2).
- ✅ Inferencia correcta: *17 × 23 = 391*, `finish_reason=stop`, log sin errores,
  ~20 tok/s de generación.

> Nota: el primer arranque tras el merge re-autotunea en frío miles de variantes
> de kernels Triton del GDN/FLA (gated-delta-net de Qwen3.5) → ~21 min CPU-bound
> con GPU idle. Es comportamiento de upstream, no del merge; los arranques
> posteriores usan la caché.
