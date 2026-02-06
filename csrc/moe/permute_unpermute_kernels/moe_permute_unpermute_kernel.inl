#pragma once

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(
    T const* unpermuted_input, T* permuted_output, int* sorted_experts,
    int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int* permuted_idx,
    int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_dest_rows, int64_t const cols, int64_t k,
    int num_local_experts) {
  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  int64_t expanded_dest_row = blockIdx.x;
  int64_t const expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  int expert_id = sorted_experts[expanded_dest_row];

  if (threadIdx.x == 0) {
    assert(expanded_dest_row <= INT32_MAX);
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        static_cast<int>(expanded_dest_row);
    // skip non local expert token
    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows) {
      permuted_idx[expanded_dest_row] = expanded_source_row;
    }
  }

  if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows) {
    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    int64_t const source_row = expanded_source_row / k;

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
    T const* unpermuted_input, T* permuted_output, int* sorted_experts,
    int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int* permuted_idx,
    int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k,
    int num_local_experts, cudaStream_t stream) {
  int64_t const blocks = num_rows * k;
  int64_t const threads = 256;
  using FuncPtr = decltype(&expandInputRowsKernel<T, true>);
  FuncPtr func_map[2] = {
      &expandInputRowsKernel<T, false>,
      &expandInputRowsKernel<T, true>,
  };
  bool is_check_skip = num_valid_tokens_ptr != nullptr;
  auto func = func_map[is_check_skip];

  func<<<blocks, threads, 0, stream>>>(
      unpermuted_input, permuted_output, sorted_experts,
      expanded_dest_row_to_expanded_source_row,
      expanded_source_row_to_expanded_dest_row, permuted_idx,
      expert_first_token_offset, num_rows, num_valid_tokens_ptr, cols, k,
      num_local_experts);
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

template <typename T, typename OutputType, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(
    T const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    float const* scales, int const* expanded_source_row_to_expanded_dest_row,
    int64_t const orig_cols, int64_t const k, int64_t const* num_valid_ptr) {
  assert(orig_cols % 4 == 0);
  int64_t const original_row = blockIdx.x;
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

  using InputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* expanded_permuted_rows_v =
      reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_col;
       elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      int64_t const expanded_original_row = original_row * k + k_idx;
      int64_t const expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      int64_t const k_offset = original_row * k + k_idx;
      float const row_scale = scales[k_offset];

      if (CHECK_SKIPPED && expanded_permuted_row >= num_valid) {
        continue;
      }

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(
          expanded_permuted_rows_row_ptr[elem_index]);
      thread_output = thread_output + row_scale * (expert_result);
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
    int64_t const num_rows, int64_t const cols, int64_t const k,
    int64_t const* num_valid_ptr, cudaStream_t stream) {
  int64_t const blocks = num_rows;
  int64_t const threads = 256;
  bool const check_finished = num_valid_ptr != nullptr;
  using FuncPtr = decltype(&finalizeMoeRoutingKernel<T, OutputType, false>);
  FuncPtr func_map[2] = {&finalizeMoeRoutingKernel<T, OutputType, false>,
                         &finalizeMoeRoutingKernel<T, OutputType, true>};
  auto* const kernel = func_map[check_finished];
  kernel<<<blocks, threads, 0, stream>>>(
      expanded_permuted_rows, reduced_unpermuted_output, scales,
      expanded_source_row_to_expanded_dest_row, cols, k, num_valid_ptr);
}
