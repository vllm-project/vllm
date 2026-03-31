#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#ifndef USE_ROCM
torch::stable::Tensor permute_cols(torch::stable::Tensor const& A,
                                   torch::stable::Tensor const& perm);

void per_token_group_quant_fp8(const torch::stable::Tensor& input,
                               torch::stable::Tensor& output_q,
                               torch::stable::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned);

// Fused activation quantisation + DeepGEMM-compatible UE8M0-packed scales.
void per_token_group_quant_8bit_packed(const torch::stable::Tensor& input,
                                       torch::stable::Tensor& output_q,
                                       torch::stable::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit);

void per_token_group_quant_int8(const torch::stable::Tensor& input,
                                torch::stable::Tensor& output_q,
                                torch::stable::Tensor& output_s,
                                int64_t group_size, double eps, double int8_min,
                                double int8_max);

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);

void cutlass_scaled_mm(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b,
                       torch::stable::Tensor const& a_scales,
                       torch::stable::Tensor const& b_scales,
                       std::optional<torch::stable::Tensor> const& bias);

void cutlass_moe_mm(torch::stable::Tensor& out_tensors,
                    torch::stable::Tensor const& a_tensors,
                    torch::stable::Tensor const& b_tensors,
                    torch::stable::Tensor const& a_scales,
                    torch::stable::Tensor const& b_scales,
                    torch::stable::Tensor const& expert_offsets,
                    torch::stable::Tensor const& problem_sizes,
                    torch::stable::Tensor const& a_strides,
                    torch::stable::Tensor const& b_strides,
                    torch::stable::Tensor const& c_strides, bool per_act_token,
                    bool per_out_ch);

void cutlass_scaled_mm_azp(torch::stable::Tensor& out,
                           torch::stable::Tensor const& a,
                           torch::stable::Tensor const& b,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b_scales,
                           torch::stable::Tensor const& azp_adj,
                           std::optional<torch::stable::Tensor> const& azp,
                           std::optional<torch::stable::Tensor> const& bias);

void get_cutlass_moe_mm_data(
    const torch::stable::Tensor& topk_ids,
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    torch::stable::Tensor& input_permutation,
    torch::stable::Tensor& output_permutation, const int64_t num_experts,
    const int64_t n, const int64_t k,
    const std::optional<torch::stable::Tensor>& blockscale_offsets,
    const bool is_gated);

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2, const int64_t n, const int64_t k,
    const bool swap_ab);

void get_cutlass_batched_moe_mm_data(
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    const torch::stable::Tensor& expert_num_tokens,
    const int64_t num_local_experts, const int64_t padded_m, const int64_t n,
    const int64_t k);

// FP4/NVFP4 ops
bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);

void cutlass_scaled_fp4_mm(torch::stable::Tensor& D,
                           torch::stable::Tensor const& A,
                           torch::stable::Tensor const& B,
                           torch::stable::Tensor const& A_sf,
                           torch::stable::Tensor const& B_sf,
                           torch::stable::Tensor const& alpha);

void cutlass_fp4_group_mm(torch::stable::Tensor& output,
                          const torch::stable::Tensor& a,
                          const torch::stable::Tensor& b,
                          const torch::stable::Tensor& a_blockscale,
                          const torch::stable::Tensor& b_blockscales,
                          const torch::stable::Tensor& alphas,
                          const torch::stable::Tensor& problem_sizes,
                          const torch::stable::Tensor& expert_offsets,
                          const torch::stable::Tensor& sf_offsets);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> scaled_fp4_quant_func(
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_scale, bool is_sf_swizzled_layout);

void scaled_fp4_quant_out(torch::stable::Tensor const& input,
                          torch::stable::Tensor const& input_scale,
                          bool is_sf_swizzled_layout,
                          torch::stable::Tensor& output,
                          torch::stable::Tensor& output_scale);

void scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_nvfp4_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor& output_block_scale,
                              torch::stable::Tensor& input,
                              torch::stable::Tensor& input_global_scale);

// AWQ ops
torch::stable::Tensor awq_gemm(torch::stable::Tensor _in_feats,
                               torch::stable::Tensor _kernel,
                               torch::stable::Tensor _scaling_factors,
                               torch::stable::Tensor _zeros,
                               int64_t split_k_iters);

torch::stable::Tensor awq_dequantize(torch::stable::Tensor _kernel,
                                     torch::stable::Tensor _scaling_factors,
                                     torch::stable::Tensor _zeros,
                                     int64_t split_k_iters, int64_t thx,
                                     int64_t thy);

// DSV3 fused A GEMM: conditionally compiled so declaration and impl
// registration are in the source file (dsv3_fused_a_gemm.cu)

#endif

torch::stable::Tensor hadacore_transform(torch::stable::Tensor& x,
                                         bool inplace);
