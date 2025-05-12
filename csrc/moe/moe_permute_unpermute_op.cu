#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "permute_unpermute_kernels/moe_permute_unpermute_kernel.h"
#include "permute_unpermute_kernels/dispatch.h"
#include "core/registration.h"

void moe_permute(
    const torch::Tensor& input,                      // [n_token, hidden]
    const torch::Tensor& topk_weights,               //[n_token, topk]
    torch::Tensor& topk_ids,                         // [n_token, topk]
    const torch::Tensor& token_expert_indicies,      // [n_token, topk]
    const std::optional<torch::Tensor>& expert_map,  // [n_expert]
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    const std::optional<int64_t>& align_block_size,
    torch::Tensor&
        permuted_input,  // [topk * n_token/align_block_size_m, hidden]
    torch::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
    torch::Tensor& src_row_id2dst_row_id_map,  // [n_token, topk]
    torch::Tensor& m_indices) {                // [align_expand_m]
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
              "topk_weights must be float32");
  TORCH_CHECK(expert_first_token_offset.scalar_type() == at::ScalarType::Long,
              "expert_first_token_offset must be int64");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  TORCH_CHECK(token_expert_indicies.scalar_type() == at::ScalarType::Int,
              "token_expert_indicies must be int32");
  TORCH_CHECK(src_row_id2dst_row_id_map.scalar_type() == at::ScalarType::Int,
              "src_row_id2dst_row_id_map must be int32");
  TORCH_CHECK(expert_first_token_offset.size(0) == n_local_expert + 1,
              "expert_first_token_offset shape != n_local_expert+1")
  TORCH_CHECK(
      src_row_id2dst_row_id_map.sizes() == token_expert_indicies.sizes(),
      "token_expert_indicies shape must be same as src_row_id2dst_row_id_map");
  auto n_token = input.sizes()[0];
  auto n_hidden = input.sizes()[1];
  auto align_block_size_value =
      align_block_size.has_value() ? align_block_size.value() : -1;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const long sorter_size =
      CubKeyValueSorter::getWorkspaceSize(n_token * topk, n_expert);
  auto sort_workspace = torch::empty(
      {sorter_size},
      torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
  auto permuted_experts_id = torch::empty_like(topk_ids);
  auto dst_row_id2src_row_id_map = torch::empty_like(src_row_id2dst_row_id_map);
  auto align_expert_first_token_offset =
      torch::zeros_like(expert_first_token_offset);

  CubKeyValueSorter sorter{};
  int64_t* valid_num_ptr = nullptr;
  // pre-process kernel for expert-parallelism:
  // no local expert id plus "n_expert" offset for priority to local expert
  // map local expert id [n, .., n+n_local_expert-1] to [0, n_local_expert -1]
  // For example, 4 expert with ep_size=2. ep_rank=1 owns global expert id
  // [2,3] with expert_map[-1, -1, 0, 1], preprocess_topk_id  process topk_ids
  // and map global expert id [2, 3] to local_expert id [0, 1] and map global
  // expert id [0, 1] ( not in ep rank=1)  to [4, 5] by plus n_expert. This map
  // operation is to make local expert high priority in following sort topk_ids
  // and scan local expert_first_token_offset for each ep rank for next group
  // gemm.
  if (expert_map.has_value()) {
    const int* expert_map_ptr = get_ptr<int>(expert_map.value());
    valid_num_ptr =
        get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
    preprocessTopkIdLauncher(get_ptr<int>(topk_ids), n_token * topk,
                             expert_map_ptr, n_expert, stream);
  }
  // expert sort topk expert id and scan expert id get expert_first_token_offset
  sortAndScanExpert(get_ptr<int>(topk_ids), get_ptr<int>(token_expert_indicies),
                    get_ptr<int>(permuted_experts_id),
                    get_ptr<int>(dst_row_id2src_row_id_map),
                    get_ptr<int64_t>(expert_first_token_offset), n_token,
                    n_expert, n_local_expert, topk, sorter,
                    get_ptr<int>(sort_workspace), stream);

  // dispatch expandInputRowsKernelLauncher
  MOE_DISPATCH(input.scalar_type(), [&] {
    expandInputRowsKernelLauncher<scalar_t>(
        get_ptr<scalar_t>(input), get_ptr<scalar_t>(permuted_input),
        get_ptr<float>(topk_weights), get_ptr<int>(permuted_experts_id),
        get_ptr<int>(dst_row_id2src_row_id_map),
        get_ptr<int>(src_row_id2dst_row_id_map),
        get_ptr<int64_t>(expert_first_token_offset), n_token, valid_num_ptr,
        n_hidden, topk, n_local_expert, align_block_size_value, stream);
  });

  // get m_indices and update expert_first_token_offset with align block
  getMIndices(get_ptr<int64_t>(expert_first_token_offset),
              get_ptr<int64_t>(align_expert_first_token_offset),
              get_ptr<int>(m_indices), n_local_expert, align_block_size_value,
              stream);
  if (align_block_size.has_value()) {
    // update align_expert_first_token_offset
    expert_first_token_offset.copy_(align_expert_first_token_offset);
  }
}

void moe_unpermute(
    const torch::Tensor& permuted_hidden_states,     // [n_token * topk, hidden]
    const torch::Tensor& topk_weights,               //[n_token, topk]
    const torch::Tensor& topk_ids,                   // [n_token, topk]
    const torch::Tensor& src_row_id2dst_row_id_map,  // [n_token, topk]
    const torch::Tensor& expert_first_token_offset,  // [n_local_expert+1]
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::Tensor& hidden_states  // [n_token, hidden]
) {
  TORCH_CHECK(src_row_id2dst_row_id_map.sizes() == topk_ids.sizes(),
              "topk_ids shape must be same as src_row_id2dst_row_id_map");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  TORCH_CHECK(
      permuted_hidden_states.scalar_type() == hidden_states.scalar_type(),
      "topk_ids dtype must be same as src_row_id2dst_row_id_map");
  auto n_token = hidden_states.size(0);
  auto n_hidden = hidden_states.size(1);
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t* valid_ptr =
      get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
  MOE_DISPATCH(hidden_states.scalar_type(), [&] {
    finalizeMoeRoutingKernelLauncher<scalar_t, scalar_t>(
        get_ptr<scalar_t>(permuted_hidden_states),
        get_ptr<scalar_t>(hidden_states), get_ptr<float>(topk_weights),
        get_ptr<int>(src_row_id2dst_row_id_map), get_ptr<int>(topk_ids),
        n_token, n_hidden, topk, valid_ptr, stream);
  });
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("moe_permute", &moe_permute);
  m.impl("moe_unpermute", &moe_unpermute);
}