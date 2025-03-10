#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "permute_unpermute_kernels/moe_permute_unpermute_kernel.h"
#include "core/registration.h"

void moe_permute(torch::Tensor& input,                  // [n_token, hidden]
                 torch::Tensor& topk_weight,            //[n_token, topk]
                 torch::Tensor& topk_ids,               // [n_token, topk]
                 torch::Tensor& token_expert_indicies,  // [n_token, topk]
                 int64_t n_expert, int64_t topk,
                 torch::Tensor& permuted_input,  // [topk * n_token, hidden]
                 torch::Tensor& expert_first_token_offset,    // [expert + 1]
                 torch::Tensor& src_row_id2dst_row_id_map) {  // [n_token, topk]

  TORCH_CHECK(topk_weight.scalar_type() == at::ScalarType::Float,
              "topk_weight must be float32");
  TORCH_CHECK(expert_first_token_offset.scalar_type() == at::ScalarType::Long,
              "expert_first_token_offset must be int64");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  TORCH_CHECK(token_expert_indicies.scalar_type() == at::ScalarType::Int,
              "token_expert_indicies must be int32");
  TORCH_CHECK(src_row_id2dst_row_id_map.scalar_type() == at::ScalarType::Int,
              "src_row_id2dst_row_id_map must be int32");
  TORCH_CHECK(expert_first_token_offset.size(0) == n_expert + 1,
              "expert_first_token_offset shape != n_expert+1")
  TORCH_CHECK(
      src_row_id2dst_row_id_map.sizes() == token_expert_indicies.sizes(),
      "token_expert_indicies shape must be same as src_row_id2dst_row_id_map");
  auto n_token = input.sizes()[0];
  auto n_hidden = input.sizes()[1];
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const long sorter_size =
      CubKeyValueSorter::getWorkspaceSize(n_token, n_expert);
  auto sort_workspace = torch::empty(
      {sorter_size},
      torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
  auto permuted_experts_id = torch::empty_like(topk_ids);
  auto dst_row_id2src_row_id_map = torch::empty_like(src_row_id2dst_row_id_map);

  CubKeyValueSorter sorter{};
  // sort topk expert id and scan expert id get expert_first_token_offset
  sortAndScanExpert(get_ptr<int>(topk_ids), get_ptr<int>(token_expert_indicies),
                    get_ptr<int>(permuted_experts_id),
                    get_ptr<int>(dst_row_id2src_row_id_map),
                    get_ptr<int64_t>(expert_first_token_offset), n_token,
                    n_expert, n_expert, topk, sorter,
                    get_ptr<int>(sort_workspace), stream);
  // std::cout << "permuted_experts_id" << permuted_experts_id << std::endl;
  // std::cout << "dst_row_id2src_row_id_map" << dst_row_id2src_row_id_map
  //           << std::endl;
  // dispatch expandInputRowsKernelLauncher
  MOE_DISPATCH(input.scalar_type(), [&] {
    expandInputRowsKernelLauncher<scalar_t>(
        get_ptr<scalar_t>(input), get_ptr<scalar_t>(permuted_input),
        get_ptr<float>(topk_weight), nullptr,
        get_ptr<int>(dst_row_id2src_row_id_map),
        get_ptr<int>(src_row_id2dst_row_id_map), n_token, nullptr, n_hidden,
        topk, stream);
  });
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("moe_permute", &moe_permute);
}