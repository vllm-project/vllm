#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

#ifdef VLLM_SKINNY_GEMM_SWEEP
  // FP16/BF16 skinny GEMM sweep: ytile/unrl as runtime args (benchmark only)
  rocm_ops.def(
      "wvSplitK_sweep(Tensor in_a, Tensor in_b, Tensor? in_bias, "
      "int CuCount, int ytile, int unrl) -> Tensor");
  rocm_ops.impl("wvSplitK_sweep", torch::kCUDA, &wvSplitK_sweep);
#endif

  // W8A16 skinny GEMM: int8 weights, fp16/bf16 activations, per-channel scale
  rocm_ops.def(
      "wvSplitK_int8(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_bias, int CuCount) -> Tensor");
  rocm_ops.impl("wvSplitK_int8", torch::kCUDA, &wvSplitK_int8);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points for asymmetric quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

#ifdef VLLM_SKINNY_GEMM_SWEEP
  rocm_ops.def(
      "wvSplitK_int8_sweep(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_bias, int CuCount, int ytile, int unrl, int achunk, "
      "int wvprgrp) -> Tensor");
  rocm_ops.impl("wvSplitK_int8_sweep", torch::kCUDA, &wvSplitK_int8_sweep);

  rocm_ops.def(
      "wvSplitK_int4g_sweep(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "int CuCount, int group_size, int ytile, int unrl, int achunk, "
      "int wvprgrp) -> Tensor");
  rocm_ops.impl("wvSplitK_int4g_sweep", torch::kCUDA, &wvSplitK_int4g_sweep);

  rocm_ops.def(
      "wvSplitK_int4g_hf_sweep(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "int CuCount, int group_size, int ytile, int unrl, int achunk, "
      "int wvprgrp) -> Tensor");
  rocm_ops.impl("wvSplitK_int4g_hf_sweep", torch::kCUDA,
                &wvSplitK_int4g_hf_sweep);
#endif  // VLLM_SKINNY_GEMM_SWEEP

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
