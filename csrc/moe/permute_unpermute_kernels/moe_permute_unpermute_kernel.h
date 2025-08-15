#pragma once
// reference from tensorrt_llm moe kernel implementation archive in
// https://github.com/BBuf/tensorrt-llm-moe/tree/master

#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include "dispatch.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "cutlass/numeric_size.h"
#include "cutlass/array.h"

template <typename T>
inline T* get_ptr(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

template <typename T>
inline const T* get_ptr(const torch::Tensor& t) {
  return reinterpret_cast<const T*>(t.data_ptr());
}

class CubKeyValueSorter {
 public:
  CubKeyValueSorter();

  CubKeyValueSorter(int const num_experts);

  void updateNumExperts(int const num_experts);

  static size_t getWorkspaceSize(size_t const num_key_value_pairs,
                                 int const num_experts);

  void run(void* workspace, size_t const workspace_size, int const* keys_in,
           int* keys_out, int const* values_in, int* values_out,
           size_t const num_key_value_pairs, cudaStream_t stream);

 private:
  static int expertsToBits(int experts);
  int num_experts_;
  int num_bits_;
};

void computeExpertFirstTokenOffset(int const* sorted_indices,
                                   int const total_indices,
                                   int const num_experts,
                                   int64_t* expert_first_token_offset,
                                   cudaStream_t stream);

void sortAndScanExpert(int* expert_for_source_row, const int* source_rows,
                       int* permuted_experts, int* permuted_rows,
                       int64_t* expert_first_token_offset, int num_rows,
                       int num_experts, int num_experts_per_node, int k,
                       CubKeyValueSorter& sorter, void* sorter_ws,
                       cudaStream_t stream);

template <typename T>
void expandInputRowsKernelLauncher(
    T const* unpermuted_input, T* permuted_output, int* sorted_experts,
    int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int* permuted_idx,
    int64_t* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    int num_local_experts, const int& align_block_size, cudaStream_t stream);

template <class T, class OutputType>
void finalizeMoeRoutingKernelLauncher(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const cols, int64_t const k,
    int64_t const* num_valid_ptr, cudaStream_t stream);

void preprocessTopkIdLauncher(int* topk_id_ptr, int size,
                              const int* expert_map_ptr, int num_experts,
                              cudaStream_t stream);

void getMIndices(int64_t* expert_first_token_offset,
                 int64_t* align_expert_first_token_offset, int* m_indices,
                 int num_local_expert, const int align_block_size,
                 cudaStream_t stream);

#include "moe_permute_unpermute_kernel.inl"
