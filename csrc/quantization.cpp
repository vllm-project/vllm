#include <cstdint>
#include <torch/extension.h>

torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

uintptr_t make_q_matrix(
    torch::Tensor q_weight,
    torch::Tensor q_perm,
    torch::Tensor q_invperm,
    torch::Tensor gptq_qzeros,
    torch::Tensor gptq_scales,
    torch::Tensor gptq_g_idx,
    torch::Tensor temp_dq
);

void gemm_half_q_half(
    torch::Tensor a,
    uintptr_t b,
    torch::Tensor c,
    bool force_cuda
);

void gptq_descact_matmul(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx);

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "awq_gemm",
    &awq_gemm,
    "Quantized GEMM for AWQ");
  m.def(
    "make_q_matrix",
    &make_q_matrix,
    "make_q_matrix");
  m.def(
    "gemm_half_q_half",
    &gemm_half_q_half,
    "gemm_half_q_half");
  m.def(
    "gptq_descact_matmul",
    &gptq_descact_matmul,
    "Quantized GEMM for GPTQ for parallelized desc_act layer");
  m.def(
    "squeezellm_gemm",
    &squeezellm_gemm,
    "Quantized GEMM for SqueezeLLM");
}
