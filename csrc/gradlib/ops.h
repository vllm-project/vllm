#pragma once

#include <torch/all.h>

void hipb_create_extension();
void hipb_destroy_extension();
torch::Tensor hipb_mm(const torch::Tensor& mat1, const torch::Tensor& mat2,
                      const int64_t solution_index,
                      at::optional<torch::Tensor> bias = at::nullopt,
                      at::optional<c10::ScalarType> out_dtype = at::nullopt,
                      at::optional<torch::Tensor> scale1 = at::nullopt,
                      at::optional<torch::Tensor> scale2 = at::nullopt,
                      at::optional<torch::Tensor> scaleOut = at::nullopt);

std::vector<int64_t> hipb_findallsols(const torch::Tensor& mat1,
                                      const torch::Tensor& mat2,
                                      at::optional<torch::Tensor> bias,
                                      at::optional<c10::ScalarType> out_dtype);

void rocb_create_extension();
void rocb_destroy_extension();
torch::Tensor RocSolIdxBlas(const torch::Tensor& mat1,
                            const torch::Tensor& mat2,
                            const int64_t solution_index);

std::vector<int64_t> RocFindAllSolIdxBlas(const torch::Tensor& mat1,
                                          const torch::Tensor& mat2);