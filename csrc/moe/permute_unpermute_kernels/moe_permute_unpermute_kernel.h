#pragma once
// reference from tensorrt_llm moe kernel implementation archive in
// https://github.com/BBuf/tensorrt-llm-moe/tree/master

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
    int const num_experts, int64_t* expert_first_token_offset) {
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
                       int64_t* expert_first_token_offset, int num_rows,
                       int num_experts, int num_experts_per_node, int k,
                       CubKeyValueSorter& sorter, void* sorter_ws,
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

template <typename T, bool CHECK_SKIPPED, bool ALIGN_BLOCK_SIZE>
__global__ void expandInputRowsKernel(
    T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    int* sorted_experts, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    int64_t* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_dest_rows, int64_t const cols, int64_t k,
    int num_local_experts, int align_block_size) {
  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  int64_t expanded_dest_row = blockIdx.x;
  int64_t const expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  int expert_id = sorted_experts[expanded_dest_row];

  extern __shared__ int64_t smem_expert_first_token_offset[];
  int64_t align_expanded_row_accumulate = 0;
  if constexpr (ALIGN_BLOCK_SIZE) {
    // load g2s
    for (int idx = threadIdx.x; idx < num_local_experts + 1;
         idx += blockDim.x) {
      smem_expert_first_token_offset[idx] =
          __ldg(expert_first_token_offset + idx);
    }
    __syncthreads();
    int lane_idx = threadIdx.x & 31;

    if (lane_idx == 0) {
      // set token_offset_in_expert = 0 if this expert is not local expert
      int token_offset_in_expert =
          expert_id >= num_local_experts
              ? 0
              : expanded_dest_row - smem_expert_first_token_offset[expert_id];
      int64_t accumulate_align_offset = 0;
#pragma unroll 1
      for (int eidx = 1; eidx <= min(expert_id, num_local_experts); eidx++) {
        auto n_token_in_expert = smem_expert_first_token_offset[eidx] -
                                 smem_expert_first_token_offset[eidx - 1];
        accumulate_align_offset += (n_token_in_expert + align_block_size - 1) /
                                   align_block_size * align_block_size;
      }
      expanded_dest_row = accumulate_align_offset + token_offset_in_expert;
    }
    // lane0 shuffle broadcast align_expanded_dest_row
    expanded_dest_row = __shfl_sync(0xffffffff, expanded_dest_row, 0);
  }

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
  }
}
template <typename T>
void expandInputRowsKernelLauncher(
    T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    int* sorted_experts, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    int64_t* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    int num_local_experts, const int& align_block_size, cudaStream_t stream) {
  int64_t const blocks = num_rows * k;
  int64_t const threads = 256;
  using FuncPtr = decltype(&expandInputRowsKernel<T, true, true>);
  FuncPtr func_map[2][2] = {
      {&expandInputRowsKernel<T, false, false>,
       &expandInputRowsKernel<T, false, true>},
      {&expandInputRowsKernel<T, true, false>,
       &expandInputRowsKernel<T, true, true>},
  };
  bool is_check_skip = num_valid_tokens_ptr != nullptr;
  bool is_align_block_size = align_block_size != -1;
  auto func = func_map[is_check_skip][is_align_block_size];

  int64_t smem_size = sizeof(int64_t) * (num_local_experts + 1);

  func<<<blocks, threads, smem_size, stream>>>(
      unpermuted_input, permuted_output, unpermuted_scales, sorted_experts,
      expanded_dest_row_to_expanded_source_row,
      expanded_source_row_to_expanded_dest_row, expert_first_token_offset,
      num_rows, num_valid_tokens_ptr, cols, k, num_local_experts,
      align_block_size);
}

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  using Type = typename U::Element;
  static_assert(T::kElements == U::kElements);
  U u;
#pragma unroll
  for (int i = 0; i < U::kElements; i++) {
    u[i] = static_cast<Type>(input[i]);
  }
  return u;
}
// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and
// performs the final skip connection.
template <typename T, typename OutputType, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int const* expert_for_source_row, int64_t const orig_cols, int64_t const k,
    int64_t const* num_valid_ptr) {
  assert(orig_cols % 4 == 0);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * orig_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
  int64_t const num_valid = *num_valid_ptr;

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(cutlass::sizeof_bits<OutputType>::value,
                     cutlass::sizeof_bits<T>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = blockDim.x;
  int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

  // using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  // auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v =
      reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_col;
       elem_index += stride) {
    bool has_valid = false;
    ComputeElem thread_output;
    thread_output.fill(0);
    float row_rescale{0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      int64_t const expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      int64_t const k_offset = original_row * k + k_idx;
      float const row_scale = scales[k_offset];

      // Check after row_rescale has accumulated
      if (CHECK_SKIPPED && expanded_permuted_row >= num_valid) {
        continue;
      }

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      int64_t const expert_idx = expert_for_source_row[k_offset];

      ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(
          expanded_permuted_rows_row_ptr[elem_index]);
      thread_output = thread_output + row_scale * (expert_result);
      has_valid = true;
    }

    OutputElem output_elem =
        arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
}

template <class T, class OutputType>
void finalizeMoeRoutingKernelLauncher(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int const* expert_for_source_row, int64_t const num_rows,
    int64_t const cols, int64_t const k, int64_t const* num_valid_ptr,
    cudaStream_t stream) {
  int64_t const blocks = num_rows;
  int64_t const threads = 256;
  bool const check_finished = num_valid_ptr != nullptr;
  using FuncPtr = decltype(&finalizeMoeRoutingKernel<T, OutputType, false>);
  FuncPtr func_map[2] = {&finalizeMoeRoutingKernel<T, OutputType, false>,
                         &finalizeMoeRoutingKernel<T, OutputType, true>};
  auto* const kernel = func_map[check_finished];
  kernel<<<blocks, threads, 0, stream>>>(
      expanded_permuted_rows, reduced_unpermuted_output, scales,
      expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k,
      num_valid_ptr);
}

__global__ void preprocessTopkIdKernel(int* topk_id_ptr, int size,
                                       int* expert_map_ptr, int num_experts) {
  auto tidx = threadIdx.x;
  auto bidx = blockIdx.x;
  auto lidx = tidx & 31;
  auto widx = tidx >> 5;
  auto warp_count = (blockDim.x + 31) >> 5;
  auto offset = bidx * blockDim.x;
  auto bound = min(offset + blockDim.x, size);
  extern __shared__ int smem_expert_map[];
  // store expert_map in smem
  for (int i = tidx; i < num_experts; i += blockDim.x) {
    smem_expert_map[i] = expert_map_ptr[i];
  }
  __syncthreads();

  // query global expert id in expert map.
  // if global expert id = -1 in exert map, plus n_expert
  // else set global expert id = exert map[global expert id]
  if (offset + tidx < bound) {
    auto topk_id = topk_id_ptr[offset + tidx];
    auto local_expert_idx = smem_expert_map[topk_id];
    if (local_expert_idx == -1) {
      topk_id += num_experts;
    } else {
      topk_id = local_expert_idx;
    }
    __syncwarp();
    topk_id_ptr[offset + tidx] = topk_id;
  }
}

void preprocessTopkIdLauncher(int* topk_id_ptr, int size, int* expert_map_ptr,
                              int num_experts, cudaStream_t stream) {
  int block = std::min(size, 1024);
  int grid = (size + block - 1) / block;
  int smem_size = (num_experts) * sizeof(int);
  preprocessTopkIdKernel<<<grid, block, smem_size, stream>>>(
      topk_id_ptr, size, expert_map_ptr, num_experts);
}

template <bool ALIGN_BLOCK_SIZE>
__global__ void getMIndicesKernel(int64_t* expert_first_token_offset,
                                  int64_t* align_expert_first_token_offset,
                                  int* m_indices, const int num_local_expert,
                                  const int align_block_size) {
  int eidx = blockIdx.x;
  int tidx = threadIdx.x;
  extern __shared__ int64_t smem_expert_first_token_offset[];
  for (int i = tidx; i <= num_local_expert; i += blockDim.x) {
    smem_expert_first_token_offset[tidx] = __ldg(expert_first_token_offset + i);
  }
  __syncthreads();
  auto last_token_offset = smem_expert_first_token_offset[eidx + 1];
  auto first_token_offset = smem_expert_first_token_offset[eidx];
  int n_token_in_expert = last_token_offset - first_token_offset;

  if constexpr (ALIGN_BLOCK_SIZE) {
    // round up to ALIGN_BLOCK_SIZE
    int64_t accumulate_align_offset = 0;
    for (int i = 1; i <= eidx + 1; i++) {
      int n_token = smem_expert_first_token_offset[i] -
                    smem_expert_first_token_offset[i - 1];
      accumulate_align_offset =
          accumulate_align_offset + (n_token + align_block_size - 1) /
                                        align_block_size * align_block_size;
      if (i == eidx) {
        first_token_offset = accumulate_align_offset;
      }
      // last block store align_expert_first_token_offset
      if (eidx == num_local_expert - 1 && threadIdx.x == 0) {
        align_expert_first_token_offset[i] = accumulate_align_offset;
      }
    }
  }
  for (int idx = tidx; idx < n_token_in_expert; idx += blockDim.x) {
    // update m_indice with expert id
    m_indices[first_token_offset + idx] = eidx;
  }
}

void getMIndices(int64_t* expert_first_token_offset,
                 int64_t* align_expert_first_token_offset, int* m_indices,
                 int num_local_expert, const int align_block_size,
                 cudaStream_t stream) {
  int block = 256;
  int grid = num_local_expert;
  int smem_size = sizeof(int64_t) * (num_local_expert + 1);
  if (align_block_size == -1) {
    getMIndicesKernel<false><<<grid, block, smem_size, stream>>>(
        expert_first_token_offset, align_expert_first_token_offset, m_indices,
        num_local_expert, align_block_size);
  } else {
    getMIndicesKernel<true><<<grid, block, smem_size, stream>>>(
        expert_first_token_offset, align_expert_first_token_offset, m_indices,
        num_local_expert, align_block_size);
  }
}
