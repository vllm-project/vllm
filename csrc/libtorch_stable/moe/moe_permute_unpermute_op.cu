#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "core/registration.h"
#include "libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h"
#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/library.h>

// moe_permute kernels require at least CUDA 12.0
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)

namespace {

int64_t product_integers(torch::headeronly::IntHeaderOnlyArrayRef sizes) {
  int64_t numel = 1;
  for (int64_t s : sizes) {
    numel *= s;
  }
  return numel;
}

torch::stable::Tensor maybe_allocate_tensor(
    const std::optional<torch::stable::Tensor>& maybe_tensor,
    torch::headeronly::IntHeaderOnlyArrayRef expected_sizes,
    torch::headeronly::ScalarType dtype, torch::stable::Device device,
    char const* name) {
  auto expected_numel = product_integers(expected_sizes);
  if (maybe_tensor.has_value()) {
    auto tensor = maybe_tensor.value();
    STD_TORCH_CHECK(tensor.device() == device, name,
                    " must be on the same device");
    STD_TORCH_CHECK(tensor.scalar_type() == dtype, name,
                    " has incorrect dtype");
    STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    STD_TORCH_CHECK(tensor.numel() >= expected_numel, name,
                    " is too small for the requested shape");
    auto flat_tensor = torch::stable::view(tensor, {tensor.numel()});
    return torch::stable::view(
        torch::stable::narrow(flat_tensor, 0, 0, expected_numel),
        expected_sizes);
  }
  return torch::stable::empty(expected_sizes, dtype, std::nullopt, device);
}

}  // namespace

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t n_expert) {
  return static_cast<int64_t>(
      CubKeyValueSorter::getWorkspaceSize(num_expanded_rows, n_expert));
}

void moe_permute_impl(
    const torch::stable::Tensor& input,                 // [n_token, hidden]
    const torch::stable::Tensor& topk_ids,              // [n_token, topk]
    const torch::stable::Tensor& token_expert_indices,  // [n_token, topk]
    const std::optional<torch::stable::Tensor>& expert_map,  // [n_expert]
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::stable::Tensor& permuted_input,  // [permuted_size, hidden]
    torch::stable::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
    torch::stable::Tensor& inv_permuted_idx,           // [n_token, topk]
    torch::stable::Tensor& permuted_idx,               // [permute_size]
    const std::optional<torch::stable::Tensor>& maybe_sort_workspace,
    const std::optional<torch::stable::Tensor>& maybe_permuted_experts_id,
    const std::optional<torch::stable::Tensor>& maybe_sorted_row_idx,
    const std::optional<torch::stable::Tensor>& maybe_topk_ids_for_sort) {
  STD_TORCH_CHECK(expert_first_token_offset.scalar_type() ==
                      torch::headeronly::ScalarType::Long,
                  "expert_first_token_offset must be int64");
  STD_TORCH_CHECK(topk_ids.scalar_type() == torch::headeronly::ScalarType::Int,
                  "topk_ids must be int32");
  STD_TORCH_CHECK(
      token_expert_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "token_expert_indices must be int32");
  STD_TORCH_CHECK(
      inv_permuted_idx.scalar_type() == torch::headeronly::ScalarType::Int,
      "inv_permuted_idx must be int32");
  STD_TORCH_CHECK(expert_first_token_offset.size(0) == n_local_expert + 1,
                  "expert_first_token_offset shape != n_local_expert+1");
  STD_TORCH_CHECK(
      inv_permuted_idx.sizes().equals(token_expert_indices.sizes()),
      "token_expert_indices shape must be same as inv_permuted_idx");

  auto device = input.device();
  auto n_token = input.sizes()[0];
  auto n_hidden = input.sizes()[1];
  auto expanded_rows = n_token * topk;
  auto stream = get_current_cuda_stream(input.get_device_index());

  auto sorter_size = moe_permute_sort_workspace_size(expanded_rows, n_expert);
  auto sort_workspace = maybe_allocate_tensor(
      maybe_sort_workspace, {sorter_size}, torch::headeronly::ScalarType::Char,
      device, "sort_workspace");
  auto permuted_experts_id = maybe_allocate_tensor(
      maybe_permuted_experts_id, topk_ids.sizes(),
      torch::headeronly::ScalarType::Int, device, "permuted_experts_id");
  auto sorted_row_idx = maybe_allocate_tensor(
      maybe_sorted_row_idx, inv_permuted_idx.sizes(),
      torch::headeronly::ScalarType::Int, device, "sorted_row_idx");

  CubKeyValueSorter sorter{};
  int64_t* valid_num_ptr = nullptr;
  torch::stable::Tensor topk_ids_for_sort = topk_ids;

  if (expert_map.has_value()) {
    const int* expert_map_ptr = get_ptr<int>(expert_map.value());
    valid_num_ptr =
        get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
    topk_ids_for_sort = maybe_allocate_tensor(
        maybe_topk_ids_for_sort, topk_ids.sizes(),
        torch::headeronly::ScalarType::Int, device, "topk_ids_for_sort");
    torch::stable::copy_(topk_ids_for_sort, topk_ids);
    preprocessTopkIdLauncher(get_ptr<int>(topk_ids_for_sort), n_token * topk,
                             expert_map_ptr, n_expert, stream);
  }

  sortAndScanExpert(
      get_ptr<const int>(topk_ids_for_sort), get_ptr<int>(token_expert_indices),
      get_ptr<int>(permuted_experts_id), get_ptr<int>(sorted_row_idx),
      get_ptr<int64_t>(expert_first_token_offset), n_token, n_expert,
      n_local_expert, topk, sorter, get_ptr<int>(sort_workspace), stream);

  MOE_DISPATCH(input.scalar_type(), [&] {
    expandInputRowsKernelLauncher<scalar_t>(
        get_ptr<scalar_t>(input), get_ptr<scalar_t>(permuted_input),
        get_ptr<int>(sorted_row_idx), get_ptr<int>(inv_permuted_idx),
        get_ptr<int>(permuted_idx), get_ptr<int64_t>(expert_first_token_offset),
        n_token, valid_num_ptr, n_hidden, topk, n_local_expert, stream);
  });
}

void moe_permute(
    const torch::stable::Tensor& input,                 // [n_token, hidden]
    const torch::stable::Tensor& topk_ids,              // [n_token, topk]
    const torch::stable::Tensor& token_expert_indices,  // [n_token, topk]
    const std::optional<torch::stable::Tensor>& expert_map,  // [n_expert]
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::stable::Tensor& permuted_input,  // [permuted_size, hidden]
    torch::stable::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
    torch::stable::Tensor& inv_permuted_idx,           // [n_token, topk]
    torch::stable::Tensor& permuted_idx) {             // [permute_size]
  moe_permute_impl(input, topk_ids, token_expert_indices, expert_map, n_expert,
                   n_local_expert, topk, permuted_input,
                   expert_first_token_offset, inv_permuted_idx, permuted_idx,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

void moe_permute_with_scratch(
    const torch::stable::Tensor& input, const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& token_expert_indices,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t n_expert,
    int64_t n_local_expert, int64_t topk, torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& inv_permuted_idx,
    torch::stable::Tensor& permuted_idx, torch::stable::Tensor& sort_workspace,
    torch::stable::Tensor& permuted_experts_id,
    torch::stable::Tensor& sorted_row_idx,
    torch::stable::Tensor& topk_ids_for_sort) {
  moe_permute_impl(input, topk_ids, token_expert_indices, expert_map, n_expert,
                   n_local_expert, topk, permuted_input,
                   expert_first_token_offset, inv_permuted_idx, permuted_idx,
                   sort_workspace, permuted_experts_id, sorted_row_idx,
                   topk_ids_for_sort);
}

void moe_unpermute(
    const torch::stable::Tensor&
        permuted_hidden_states,                     // [n_token * topk, hidden]
    const torch::stable::Tensor& topk_weights,      // [n_token, topk]
    const torch::stable::Tensor& inv_permuted_idx,  // [n_token, topk]
    const std::optional<torch::stable::Tensor>&
        expert_first_token_offset,  // [n_local_expert+1]
    int64_t topk,
    torch::stable::Tensor& hidden_states) {  // [n_token, hidden]
  STD_TORCH_CHECK(
      permuted_hidden_states.scalar_type() == hidden_states.scalar_type(),
      "permuted_hidden_states dtype must be same as hidden_states");

  auto n_token = hidden_states.size(0);
  auto n_hidden = hidden_states.size(1);
  auto stream = get_current_cuda_stream(hidden_states.get_device_index());

  int64_t const* valid_ptr = nullptr;
  if (expert_first_token_offset.has_value()) {
    int n_local_expert = expert_first_token_offset.value().size(0) - 1;
    valid_ptr =
        get_ptr<int64_t>(expert_first_token_offset.value()) + n_local_expert;
  }

  MOE_DISPATCH(hidden_states.scalar_type(), [&] {
    finalizeMoeRoutingKernelLauncher<scalar_t, scalar_t>(
        get_ptr<scalar_t>(permuted_hidden_states),
        get_ptr<scalar_t>(hidden_states), get_ptr<float>(topk_weights),
        get_ptr<int>(inv_permuted_idx), n_token, n_hidden, topk, valid_ptr,
        stream);
  });
}

template <typename T>
__global__ void shuffleInputRowsKernel(const T* input,
                                       const int32_t* dst2src_map, T* output,
                                       int64_t num_src_rows,
                                       int64_t num_dst_rows, int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (blockIdx.x < num_dst_rows) {
    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    auto const* source_row_ptr =
        reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr =
        reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = blockDim.x;
    int64_t const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    for (int elem_index = start_offset; elem_index < num_elems_in_col;
         elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

void shuffle_rows(const torch::stable::Tensor& input_tensor,
                  const torch::stable::Tensor& dst2src_map,
                  torch::stable::Tensor& output_tensor) {
  STD_TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(),
                  "Input and output tensors must have the same data type");

  auto stream = get_current_cuda_stream(output_tensor.get_device_index());
  const int64_t blocks = output_tensor.size(0);
  const int64_t threads = 256;
  const int64_t num_dest_rows = output_tensor.size(0);
  const int64_t num_src_rows = input_tensor.size(0);
  const int64_t num_cols = input_tensor.size(1);

  STD_TORCH_CHECK(!(num_cols % (128 / input_tensor.element_size() / 8)),
                  "num_cols must be divisible by 128 / "
                  "input_tensor.element_size() / 8");

  MOE_DISPATCH(input_tensor.scalar_type(), [&] {
    shuffleInputRowsKernel<scalar_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(input_tensor.const_data_ptr()),
        reinterpret_cast<const int32_t*>(dst2src_map.const_data_ptr()),
        reinterpret_cast<scalar_t*>(output_tensor.mutable_data_ptr()),
        num_src_rows, num_dest_rows, num_cols);
  });
}

#else

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t n_expert) {
  STD_TORCH_CHECK(
      false, "moe_permute_sort_workspace_size is not supported on CUDA < 12.0");
}

void moe_permute(const torch::stable::Tensor& input,
                 const torch::stable::Tensor& topk_ids,
                 const torch::stable::Tensor& token_expert_indices,
                 const std::optional<torch::stable::Tensor>& expert_map,
                 int64_t n_expert, int64_t n_local_expert, int64_t topk,
                 torch::stable::Tensor& permuted_input,
                 torch::stable::Tensor& expert_first_token_offset,
                 torch::stable::Tensor& inv_permuted_idx,
                 torch::stable::Tensor& permuted_idx) {
  STD_TORCH_CHECK(false, "moe_permute is not supported on CUDA < 12.0");
}

void moe_permute_with_scratch(
    const torch::stable::Tensor& input, const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& token_expert_indices,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t n_expert,
    int64_t n_local_expert, int64_t topk, torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& inv_permuted_idx,
    torch::stable::Tensor& permuted_idx, torch::stable::Tensor& sort_workspace,
    torch::stable::Tensor& permuted_experts_id,
    torch::stable::Tensor& sorted_row_idx,
    torch::stable::Tensor& topk_ids_for_sort) {
  STD_TORCH_CHECK(false,
                  "moe_permute_with_scratch is not supported on CUDA < 12.0");
}

void moe_unpermute(
    const torch::stable::Tensor& permuted_hidden_states,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& inv_permuted_idx,
    const std::optional<torch::stable::Tensor>& expert_first_token_offset,
    int64_t topk, torch::stable::Tensor& hidden_states) {
  STD_TORCH_CHECK(false, "moe_unpermute is not supported on CUDA < 12.0");
}

#endif

bool moe_permute_unpermute_supported() {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
  return true;
#else
  return false;
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("moe_permute", TORCH_BOX(&moe_permute));
  m.impl("moe_permute_with_scratch", TORCH_BOX(&moe_permute_with_scratch));
  m.impl("moe_unpermute", TORCH_BOX(&moe_unpermute));
}