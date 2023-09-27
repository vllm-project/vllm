#include <cstdint>
#include <torch/extension.h>

torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

void gptq_set_tuning_params(
  int matmul_recons_thd,
  bool matmul_fused_remap,
  bool matmul_no_half2);

void gptq_prepare_buffers(
  torch::Device device,
  torch::Tensor temp_state,
  torch::Tensor temp_dq);

uintptr_t gptq_make_q4(
  torch::Tensor qweight,
  torch::Tensor qzeros,
  torch::Tensor scales,
  torch::Tensor g_idx,
  int device);

void gptq_q4_matmul(
  torch::Tensor x,
  uintptr_t w,
  torch::Tensor out);

void gptq_descact_matmul(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "awq_gemm",
    &awq_gemm,
    "Quantized GEMM for AWQ");
  m.def(
    "gptq_set_tuning_params",
    &gptq_set_tuning_params,
    "Set tuning params for GPTQ");
  m.def(
    "gptq_prepare_buffers",
    &gptq_prepare_buffers,
    "Prepare buffers for GPTQ");
  m.def(
    "gptq_make_q4",
    &gptq_make_q4,
    "Preprocess weight for GPTQ");
  m.def(
    "gptq_q4_matmul",
    &gptq_q4_matmul,
    "Quantized GEMM for GPTQ");
  m.def(
    "gptq_descact_matmul",
    &gptq_descact_matmul,
    "Quantized GEMM for GPTQ for parallelized desc_act layer");
}