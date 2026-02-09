#pragma once

#include <optional>
#include <torch/library.h>
#include <tuple>

#include "core/scalar_type.hpp"

#include <vector>

torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

#ifndef USE_ROCM
void silu_and_mul_nvfp4_quant(torch::Tensor& out,
                              torch::Tensor& output_block_scale,
                              torch::Tensor& input,
                              torch::Tensor& input_global_scale);
#endif

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);
void swigluoai_and_mul(torch::Tensor& out, torch::Tensor& input,
                       double alpha = 1.702, double limit = 7.0);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale);

#ifndef USE_ROCM

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

#endif

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const& dtype);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_a8_vec(torch::Tensor X, torch::Tensor W,
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);

#ifndef USE_ROCM

bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);

void cutlass_scaled_fp4_mm(torch::Tensor& D, torch::Tensor const& A,
                           torch::Tensor const& B, torch::Tensor const& A_sf,
                           torch::Tensor const& B_sf,
                           torch::Tensor const& alpha);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);

void cutlass_fp4_group_mm(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets);

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets);

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::Tensor& expert_first_token_offset,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    const int64_t n, const int64_t k, const bool swap_ab);

void get_cutlass_pplx_moe_mm_data(torch::Tensor& expert_offsets,
                                  torch::Tensor& problem_sizes1,
                                  torch::Tensor& problem_sizes2,
                                  const torch::Tensor& expert_num_tokens,
                                  const int64_t num_local_experts,
                                  const int64_t padded_m, const int64_t n,
                                  const int64_t k);

void cutlass_scaled_mm_azp(torch::Tensor& out, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias);

bool cutlass_sparse_scaled_mm_supported(int64_t cuda_device_capability);

void cutlass_scaled_sparse_mm(torch::Tensor& out, torch::Tensor const& a,
                              torch::Tensor const& b, torch::Tensor const& e,
                              torch::Tensor const& a_scales,
                              torch::Tensor const& b_scales,
                              std::optional<torch::Tensor> const& bias);

std::vector<torch::Tensor> cutlass_sparse_compress(torch::Tensor const& a);

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_scale,
                      torch::Tensor const& input_scale,
                      bool is_sf_swizzled_layout);

void scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

#endif

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, bool use_v2_format, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void selective_scan_fwd(
    const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A,
    const torch::Tensor& B, const torch::Tensor& C,
    const std::optional<torch::Tensor>& D_,
    const std::optional<torch::Tensor>& z_,
    const std::optional<torch::Tensor>& delta_bias_, bool delta_softplus,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& ssm_states, int64_t pad_slot_id, int64_t block_size,
    const std::optional<torch::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx);

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I,
    int64_t I2, int64_t group_size, bool apply_router_weight_on_input,
    int64_t activation_kind);

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);

torch::Tensor hadacore_transform(torch::Tensor& x, bool inplace);

#ifdef USE_ROCM
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
torch::Tensor qr_get_handle(fptr_t _fa);
void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles);
void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t quant_level, bool cast_bf2half = false);
int64_t qr_max_size();
#endif
