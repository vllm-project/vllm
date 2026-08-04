// SPDX-License-Identifier: Apache-2.0

#pragma once

// Physical KV-cache memory layout an engine hands to LMCache, plus the
// classification predicates over it. Vendor-header-free, so every backend and
// the Python facade (lmc_ops) share one definition. Detection (raw layout ->
// format) lives in lmcache/v1/gpu_connector/kv_format.

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
CS: content size (per-head content width; 2 * head size when K/V are fused)
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)
*/
enum class EngineKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  /*
  used by:
  - vLLM CROSS_LAYER mode
  */

  NL_X_TWO_NB_BS_NH_HS = 1,
  /*
  used by:
  - vLLM non-MLA flash attention
  */

  NL_X_NB_TWO_BS_NH_HS = 2,
  /*
  used by:
  - vLLM non-MLA flash infer
  */

  NL_X_NB_BS_HS = 3,
  /*
  used by:
  - vLLM MLA
  */

  TWO_X_NL_X_NBBS_NH_HS = 4,
  /*
  used by:
  - SGLang MHA (flash attention and flash infer)
  */

  NL_X_NBBS_ONE_HS = 5,
  /*
  used by:
  - SGLang MLA
  */

  NL_X_TWO_NB_NH_BS_HS = 6,
  /*
  used by:
  - vLLM non-MLA flash attention (HND layout)
  physical shape per layer: [2, num_blocks, num_heads, block_size, head_size]
  */

  NL_X_NB_TWO_NH_BS_HS = 7,
  /*
  used by:
  - vLLM non-MLA flash infer (HND layout)
  physical shape per layer: [num_blocks, 2, num_heads, block_size, head_size]
  */

  NB_NL_TWO_NH_BS_HS = 8,
  /*
  used by:
  - TRT-LLM cross-layer (HND layout)
  physical shape: [num_blocks, num_layers, 2, num_heads, block_size, head_size]
  */

  TWO_X_NL_X_NB_BS_NH_HS = 9,
  /*
  used by:
  - SGLang MHA via the MP daemon path
  physical shape per layer: [num_blocks, block_size, num_heads, head_size]
  */

  NL_X_NB_NH_BS_TWO_HS = 10,
  /*
  DEPRECATED: superseded by NL_X_NB_NH_BS_CS; no longer produced by detection.
  used by:
  - vLLM non-MLA blocks-first attention with K/V fused into the trailing dim
  physical shape per layer: [num_blocks, num_heads, block_size, 2, head_size]
  (recovered by splitting the fused trailing [block_size, 2 * head_size]).
  The device transfer kernels treat it as HND with kv_size == 1 and
  hs == 2 * head_size (the K/V axis stays packed inside each head copy).
  */

  NL_X_NB_BS_NH_TWO_HS = 11,
  /*
  DEPRECATED: superseded by NL_X_NB_BS_NH_CS; no longer produced by detection.
  used by:
  - vLLM non-MLA blocks-first attention (NHD layout) with K/V fused into the
    trailing dim
  physical shape per layer: [num_blocks, block_size, num_heads, 2, head_size]
  (recovered by splitting the fused trailing [num_heads, 2 * head_size]).
  Like NL_X_NB_NH_BS_TWO_HS but tokens before heads; the device transfer
  kernels treat it as NHD with kv_size == 1 and hs == 2 * head_size.
  */

  NL_X_NB_NH_BS_CS = 12,
  /*
  used by:
  - vLLM non-MLA blocks-first attention (HND layout) with K/V fused into the
    trailing content dim (unified KV cache)
  physical shape per layer: [num_blocks, num_heads, block_size, content_size]
  The device transfer kernels treat it as HND with kv_size == 1 and
  hs == content_size.
  */

  NL_X_NB_BS_NH_CS = 13,
  /*
  used by:
  - vLLM non-MLA blocks-first attention (NHD layout) with K/V fused into the
    trailing content dim (unified KV cache)
  physical shape per layer: [num_blocks, block_size, num_heads, content_size]
  Like NL_X_NB_NH_BS_CS but tokens before heads; the device transfer kernels
  treat it as NHD with kv_size == 1 and hs == content_size.
  */

  // vLLM DSA indexer k-cache [NB,BS,132] u8, paged [BSxvals][BSxscales]; kv 1
  NL_X_NB_BSV_BSS = 14,
};

// __host__ __device__ under CUDA/HIP so the kernels can call these; the guard
// keeps the header vendor-runtime-free.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define LMC_KV_FORMAT_HD __host__ __device__
#else
  #define LMC_KV_FORMAT_HD
#endif

// Structural shape of the normalized kv_caches: exactly one is true per format.

// All layers in one fused tensor.
LMC_KV_FORMAT_HD constexpr bool is_cross_layer(EngineKVFormat f) {
  return f == EngineKVFormat::NB_NL_TWO_BS_NH_HS ||
         f == EngineKVFormat::NB_NL_TWO_NH_BS_HS;
}

// Keys and values in two separate top-level lists: [key_layers, value_layers].
LMC_KV_FORMAT_HD constexpr bool is_kv_list(EngineKVFormat f) {
  return f == EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS ||
         f == EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS;
}

// One list entry per layer: kv_caches[layer_idx] is that layer's tensor.
LMC_KV_FORMAT_HD constexpr bool is_layer_list(EngineKVFormat f) {
  return f == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS ||
         f == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS ||
         f == EngineKVFormat::NL_X_NB_BS_HS ||
         f == EngineKVFormat::NL_X_NBBS_ONE_HS ||
         f == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
         f == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS ||
         f == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS ||
         f == EngineKVFormat::NL_X_NB_BS_NH_TWO_HS ||
         f == EngineKVFormat::NL_X_NB_NH_BS_CS ||
         f == EngineKVFormat::NL_X_NB_BS_NH_CS ||
         f == EngineKVFormat::NL_X_NB_BSV_BSS;
}

// Multi-head Latent Attention: a single latent KV head (no separate K/V split).
// The blocked-scale indexer cache transfers like MLA (single plane,
// kv_size == 1); only its paged addressing differs.
LMC_KV_FORMAT_HD constexpr bool is_mla(EngineKVFormat f) {
  return f == EngineKVFormat::NL_X_NB_BS_HS ||     // vLLM MLA
         f == EngineKVFormat::NL_X_NBBS_ONE_HS ||  // SGLang MLA
         f == EngineKVFormat::NL_X_NB_BSV_BSS;     // DSA indexer (blocked)
}

// vLLM fused K/V: K and V packed in the trailing dim (2 * head_size), no
// separate K/V axis — transferred as one k_or_v == 0 pass (like MLA).
LMC_KV_FORMAT_HD constexpr bool is_fused_packed(EngineKVFormat f) {
  return f == EngineKVFormat::NL_X_NB_NH_BS_TWO_HS ||  // fused HND (deprecated)
         f == EngineKVFormat::NL_X_NB_BS_NH_TWO_HS ||  // fused NHD (deprecated)
         f == EngineKVFormat::NL_X_NB_NH_BS_CS ||      // content-size HND
         f == EngineKVFormat::NL_X_NB_BS_NH_CS;        // content-size NHD
}
