#pragma once
'''   reference from tensorrt_llm moe kernel implementation archive in https
    :  // github.com/BBuf/tensorrt-llm-moe/tree/master
'''
#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include "dispatch.h"
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include "cutlass/numeric_size.h"
#include "cutlass/array.h"

       template <at::ScalarType type>
       struct ScalarType2CudaType;

template <>
struct ScalarType2CudaType<at::ScalarType::Float> {
  using type = float;
};
template <>
struct ScalarType2CudaType<at::ScalarType::Half> {
  using type = half;
};
template <>
struct ScalarType2CudaType<at::ScalarType::BFloat16> {
  using type = __nv_bfloat16;
};

// #if __CUDA_ARCH__ >= 890
// fp8
template <>
struct ScalarType2CudaType<at::ScalarType::Float8_e5m2> {
  using type = __nv_fp8_e5m2;
};
template <>
struct ScalarType2CudaType<at::ScalarType::Float8_e4m3fn> {
  using type = __nv_fp8_e4m3;
};
// #endif

template <typename T>
inline T* get_ptr(torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
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

static inline size_t pad_to_multiple_of_16(size_t const& input) {
  static constexpr int ALIGNMENT = 16;
  return ALIGNMENT * ((input + ALIGNMENT - 1) / ALIGNMENT);
}

// CubKeyValueSorter definition begin
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0), num_bits_(sizeof(int) * 8) {}

int CubKeyValueSorter::expertsToBits(int num_experts) {
  // Max value we represent is V = num_experts + (num_experts - 1) = 2 *
  // num_experts - 1 The maximum number of bits is therefore floor(log2(V)) + 1
  return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts), num_bits_(expertsToBits(num_experts)) {}

void CubKeyValueSorter::updateNumExperts(int const num_experts) {
  num_experts_ = num_experts;
  num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs,
                                           int const num_experts) {
  int num_bits = expertsToBits(num_experts);
  size_t required_storage = 0;
  int* null_int = nullptr;
  cub::DeviceRadixSort::SortPairs(nullptr, required_storage, null_int, null_int,
                                  null_int, null_int, num_key_value_pairs, 0,
                                  num_bits);

  //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64,
  //   4, 3, 0 The required_storage seems to vary between 0 and 1 for the same
  //   inputs
  if (required_storage == 0) {
    required_storage = 1;
  }
  return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size,
                            int const* keys_in, int* keys_out,
                            int const* values_in, int* values_out,
                            size_t const num_key_value_pairs,
                            cudaStream_t stream) {
  size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
  size_t actual_ws_size = workspace_size;

  TORCH_CHECK(expected_ws_size <= workspace_size,
              "[CubKeyValueSorter::run] The allocated workspace is too small "
              "to run this problem.");
  cub::DeviceRadixSort::SortPairs(workspace, actual_ws_size, keys_in, keys_out,
                                  values_in, values_out, num_key_value_pairs, 0,
                                  num_bits_, stream);
}
// CubKeyValueSorter definition end

template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices,
                                                      int64_t const arr_length,
                                                      T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

// Calculates the start offset of the tokens for a given expert. The last
// element is the total number of valid tokens
__global__ void computeExpertFirstTokenOffsetKernel(
    int const* sorted_experts, int64_t const sorted_experts_len,
    int64_t const num_experts, int64_t* expert_first_token_offset) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;

  // Note that expert goes [0, num_experts] (inclusive) because we want a count
  // for the total number of active tokens at the end of the scan.
  if (expert >= num_experts + 1) {
    return;
  }
  expert_first_token_offset[expert] =
      findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
}

void computeExpertFirstTokenOffset(int const* sorted_indices,
                                   int const total_indices,
                                   int const num_experts,
                                   int64_t* expert_first_token_offset,
                                   cudaStream_t stream) {
  int const num_entries = num_experts + 1;
  int const threads = std::min(1024, num_entries);
  int const blocks = (num_entries + threads - 1) / threads;

  computeExpertFirstTokenOffsetKernel<<<blocks, threads, 0, stream>>>(
      sorted_indices, total_indices, num_experts, expert_first_token_offset);
}

//
void sortAndScanExpert(int* expert_for_source_row, int* source_rows,
                       int* permuted_experts, int* permuted_rows,
                       int64_t* expert_first_token_offset, int64_t num_rows,
                       int64_t num_experts, int64_t num_experts_per_node,
                       int64_t k, CubKeyValueSorter& sorter, void* sorter_ws,
                       cudaStream_t stream) {
  int64_t const expanded_num_rows = k * num_rows;
  // We need to use the full num_experts because that is the sentinel value used
  // by topk for disabled experts
  sorter.updateNumExperts(num_experts);
  size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(
      sorter.getWorkspaceSize(expanded_num_rows, num_experts));
  sorter.run((void*)sorter_ws, sorter_ws_size_bytes, expert_for_source_row,
             permuted_experts, source_rows, permuted_rows, expanded_num_rows,
             stream);
  computeExpertFirstTokenOffset(permuted_experts, expanded_num_rows,
                                num_experts_per_node, expert_first_token_offset,
                                stream);
}

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(
    T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows,
    int64_t const* num_dest_rows, int64_t const cols, int64_t k) {
  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  int64_t const expanded_dest_row = blockIdx.x;
  int64_t const expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  if (threadIdx.x == 0) {
    assert(expanded_dest_row <= INT32_MAX);
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        static_cast<int>(expanded_dest_row);
  }

  if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows) {
    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    int64_t const source_k_rank = expanded_source_row / num_rows;
    int64_t const source_row = expanded_source_row % num_rows;

    auto const* source_row_ptr =
        reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols);
    auto* dest_row_ptr =
        reinterpret_cast<DataElem*>(permuted_output + expanded_dest_row * cols);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = blockDim.x;
    int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

    for (int elem_index = start_offset; elem_index < num_elems_in_col;
         elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }

    if (permuted_scales && threadIdx.x == 0) {
      int64_t const source_k_idx = source_row * k + source_k_rank;
      permuted_scales[expanded_dest_row] = unpermuted_scales[source_k_idx];
    }
  }
}
template <typename T>
void expandInputRowsKernelLauncher(
    T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows,
    int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    cudaStream_t stream) {
  int64_t const blocks = num_rows * k;
  int64_t const threads = 256;
  auto func = (num_valid_tokens_ptr != nullptr)
                  ? expandInputRowsKernel<T, true>
                  : expandInputRowsKernel<T, false>;
  func<<<blocks, threads, 0, stream>>>(unpermuted_input, permuted_output,
                                       unpermuted_scales, permuted_scales,
                                       expanded_dest_row_to_expanded_source_row,
                                       expanded_source_row_to_expanded_dest_row,
                                       num_rows, num_valid_tokens_ptr, cols, k);
}
