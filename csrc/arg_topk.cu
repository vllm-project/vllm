/******************************************************************************
 * Copyright (c) 2025, Tri Dao, Samsung SDSA.
 ******************************************************************************/
// This file contains code adapted from TensorFlow:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op_gpu.h

#include <cassert>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

int const CUDA_NUM_THREADS = 512;
enum class HeapType { kMinHeap, kMaxHeap };
enum class PreferIndices { kLower, kHigher };

template <typename T>
struct Entry {
  int index;
  T value;
};

template <typename T>
struct LinearData {
  typedef Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const { return data[index]; }

  __device__ int get_index(int i) const { return data[i].index; }
  __device__ T get_value(int i) const { return data[i].value; }

  Entry* const data;
};

template <typename T>
struct IndirectLinearData {
  typedef Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const { return data[index]; }

  __device__ int get_index(int i) const {
    return backing_data[data[i].index].index;
  }
  __device__ T get_value(int i) const { return data[i].value; }

  Entry* const data;
  Entry* const backing_data;
};

template <typename T>
struct StridedData {
  typedef Entry<T> Entry;

  __device__ Entry& operator[](std::size_t index) const {
    return data[index * blockDim.x + threadIdx.x];
  }

  __device__ int get_index(int i) const { return (*this)[i].index; }
  __device__ T get_value(int i) const { return (*this)[i].value; }

  Entry* const data;
};

// A heap of Entry<T> that can either work as a min-heap or as a max-heap.
template <HeapType heapType, PreferIndices preferIndices,
          template <typename> class Data, typename T>
struct IndexedHeap {
  typedef typename Data<T>::Entry Entry;
  Data<T> const data;
  __device__ IndexedHeap(Data<T> const& d) : data(d) {}

  __device__ bool is_above(int left, int right) {
    T left_value = data.get_value(left);
    T right_value = data.get_value(right);
    if (left_value == right_value) {
      if (preferIndices == PreferIndices::kLower) {
        return data.get_index(left) < data.get_index(right);
      } else {
        return data.get_index(left) > data.get_index(right);
      }
    }
    if (heapType == HeapType::kMinHeap) {
      return left_value < right_value;
    } else {
      return left_value > right_value;
    }
  }

  __device__ void assign(int i, Entry const& entry) { data[i] = entry; }

  __device__ void push_up(int i) {
    int child = i;
    int parent;
    for (; child > 0; child = parent) {
      parent = (child - 1) / 2;
      if (!is_above(child, parent)) {
        // Heap property satisfied.
        break;
      }
      swap(child, parent);
    }
  }

  __device__ void swap(int a, int b) {
    auto tmp = data[b];
    data[b] = data[a];
    data[a] = tmp;
  }

  __device__ void push_root_down(int k) { push_down(0, k); }

  // MAX-HEAPIFY in Cormen
  __device__ void push_down(int node, int k) {
    while (true) {
      int const left = 2 * node + 1;
      int const right = left + 1;
      int smallest = node;
      if (left < k && is_above(left, smallest)) {
        smallest = left;
      }
      if (right < k && is_above(right, smallest)) {
        smallest = right;
      }
      if (smallest == node) {
        break;
      }
      swap(smallest, node);
      node = smallest;
    }
  }

  // BUILD-MAX-HEAPIFY in Cormen
  __device__ void build(int k) {
    for (int node = (k - 1) / 2; node >= 0; node--) {
      push_down(node, k);
    }
  }

  // HEAP-EXTRACT-MAX in Cormen
  __device__ void remove_root(int k) {
    data[0] = data[k - 1];
    push_root_down(k - 1);
  }

  // in-place HEAPSORT in Cormen
  // This method destroys the heap property.
  __device__ void sort(int k) {
    for (int slot = k - 1; slot > 0; slot--) {
      // This is like remove_root but we insert the element at the end.
      swap(slot, 0);
      // Heap is now an element smaller.
      push_root_down(/*k=*/slot);
    }
  }

  __device__ void replace_root(Entry const& entry, int k) {
    data[0] = entry;
    push_root_down(k);
  }

  __device__ Entry const& root() { return data[0]; }
};

template <HeapType heapType, PreferIndices preferIndices,
          template <typename> class Data, typename T>
__device__ IndexedHeap<heapType, preferIndices, Data, T> make_indexed_heap(
    typename Data<T>::Entry* data) {
  return IndexedHeap<heapType, preferIndices, Data, T>{Data<T>{data}};
}

template <typename T, template <typename> class Data = LinearData>
__device__ void heapArgTopK(T const* __restrict__ input, int length, int k,
                            Entry<T>* __restrict__ heap_entries,
                            bool sorted = false, int start_index = 0,
                            int step_size = 1) {
  assert(k <= length);

  auto heap =
      make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher, Data, T>(
          heap_entries);

  int heap_end_index = start_index + k * step_size;
  if (heap_end_index > length) {
    heap_end_index = length;
  }
  // Initialize the min-heap.
  for (int index = start_index, slot = 0; index < heap_end_index;
       index += step_size, slot++) {
    heap.assign(slot, {index, input[index]});
  }

  heap.build(k);

  // Now iterate over the remaining items.
  // If an item is smaller than the min element, it is not amongst the top k.
  // Otherwise, replace the min element with it and push upwards.
  for (int index = heap_end_index; index < length; index += step_size) {
    // We prefer elements with lower indices. This is given here.
    // Later elements automatically have higher indices, so can be discarded.
    if (input[index] > heap.root().value) {
      // This element should replace the min.
      heap.replace_root({index, input[index]}, k);
    }
  }

  // Sort if wanted.
  if (sorted) {
    heap.sort(k);
  }
}

template <typename T>
__device__ void mergeShards(int num_shards, int k,
                            Entry<T>* __restrict__ entries,
                            Entry<T>* __restrict__ top_k_heap,
                            float* top_k_values, int* top_k_indices) {
  // If k < num_shards, we can use a min-heap with k elements to get the top k
  // of the sorted blocks.
  // If k > num_shards, we can initialize a min-heap with the top element from
  // each sorted block.
  int const heap_size = k < num_shards ? k : num_shards;

  // Min-heap part.
  {
    auto min_heap = IndexedHeap<HeapType::kMinHeap, PreferIndices::kHigher,
                                IndirectLinearData, T>{
        IndirectLinearData<T>{top_k_heap, entries}};
    // Initialize the heap as a min-heap.
    for (int slot = 0; slot < heap_size; slot++) {
      min_heap.assign(slot, {slot, entries[slot].value});
    }
    min_heap.build(heap_size);

    // Now perform top k with the remaining shards (if num_shards > heap_size).
    for (int shard = heap_size; shard < num_shards; shard++) {
      auto const entry = entries[shard];
      auto const root = min_heap.root();
      if (entry.value < root.value) {
        continue;
      }
      if (entry.value == root.value &&
          entry.index > entries[root.index].index) {
        continue;
      }
      // This element should replace the min.
      min_heap.replace_root({shard, entry.value}, heap_size);
    }
  }

  // Max-part.
  {
    // Turn the min-heap into a max-heap in-place.
    auto max_heap = IndexedHeap<HeapType::kMaxHeap, PreferIndices::kLower,
                                IndirectLinearData, T>{
        IndirectLinearData<T>{top_k_heap, entries}};
    // Heapify into a max heap.
    max_heap.build(heap_size);

    // Now extract the minimum k-1 times.
    // k is treated specially.
    int const last_k = k - 1;
    for (int rank = 0; rank < last_k; rank++) {
      Entry<T> const& max_element = max_heap.root();

      assert(top_k_values != nullptr);
      top_k_values[rank] = static_cast<float>(max_element.value);

      int shard_index = max_element.index;
      top_k_indices[rank] = entries[shard_index].index;
      int next_shard_index = shard_index + num_shards;
      // For rank < k-1, each top k heap still contains at least 1 element,
      // so we can draw a replacement.
      max_heap.replace_root({next_shard_index, entries[next_shard_index].value},
                            heap_size);
    }

    // rank == last_k.
    Entry<T> const& max_element = max_heap.root();
    // top_k_values[last_k] = max_element.value;
    int shard_index = max_element.index;
    top_k_indices[last_k] = entries[shard_index].index;
    top_k_values[last_k] = static_cast<float>(max_element.value);
  }
}

template <typename T>
__global__ void arg_topk_forward_kernel(T const* __restrict__ input,
                                        size_t shared_memory_size, int length,
                                        int k, bool sorted,
                                        float* __restrict__ output,
                                        int* __restrict__ indices) {
  __shared__ char shared_memory[48 << 10];
  int const batch_index = blockIdx.x;
  T const* batch_input = input + batch_index * length;
  int const thread_index = threadIdx.x;
  int const thread_count = blockDim.x;
  Entry<T>* shared_entries = (Entry<T>*)shared_memory;
  heapArgTopK<T, StridedData>(batch_input, length, k, shared_entries, true,
                              thread_index, thread_count);
  __syncthreads();
  if (thread_index == 0) {
    int const offset = batch_index * k;
    auto batch_output = output + offset;
    auto batch_indices = indices + offset;
    Entry<T>* top_k_heap = shared_entries + thread_count * k;
    mergeShards(thread_count, k, shared_entries, top_k_heap, batch_output,
                batch_indices);
  }
}

template <typename DT>
void forward_kernel(DT const* input_ptr, float* output_ptr, int* indices_ptr,
                    size_t batch_size, int length, int k, bool sorted,
                    cudaStream_t stream) {
  // Adopted from TensorFlow's ArgTopK implementation
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/topk_op_gpu.h
  int num_shards = 0;
  {
    constexpr auto shared_memory_size = 48 << 10;
    auto const heap_size = k * sizeof(Entry<DT>);
    // shared_memory_size = (num_shards + 1) * heap_size <=>
    num_shards = shared_memory_size / heap_size - 1;
    assert(num_shards > 0);
    if (num_shards > CUDA_NUM_THREADS) {
      num_shards = CUDA_NUM_THREADS;
    }
  }
  // We are limited by the amount of shared memory we have per block.
  size_t shared_memory_size = (num_shards + 1) * k * sizeof(Entry<DT>);
  // size_t num_blocks = (batch_size + num_shards - 1) / num_shards;
  size_t num_blocks = batch_size;

  int beam_size = k;
  assert(num_shards >= (size_t)beam_size);
  num_shards = 600;
  arg_topk_forward_kernel<<<num_blocks, num_shards, 0, stream>>>(
      input_ptr, shared_memory_size, length, beam_size, sorted, output_ptr,
      indices_ptr);
}

std::tuple<torch::Tensor, torch::Tensor> arg_topk_impl(torch::Tensor& input,
                                                       int64_t k, bool sorted) {
  // Ensure input is valid
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be 2-dimensional");
  TORCH_CHECK(k > 0 && k <= input.size(1),
              "k must be between 1 and input.size(1)");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  const auto batch_size = input.size(0);
  const auto length = input.size(1);

  input = input.to(torch::kFloat32);

  // Allocate output tensors
  auto options_values =
      torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
  auto options_indices =
      torch::TensorOptions().dtype(torch::kInt32).device(input.device());

  auto values = torch::empty({batch_size, k}, options_values);
  auto indices = torch::empty({batch_size, k}, options_indices);

  // Get current CUDA stream
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  at::cuda::CUDAGuard device_guard(input.device());
  int beam_width = static_cast<int>(k);

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "arg_topk_cuda", ([&] {
                               forward_kernel<scalar_t>(
                                   input.data_ptr<scalar_t>(),
                                   values.data_ptr<float>(),
                                   indices.data_ptr<int>(), batch_size,
                                   length, beam_width, sorted, stream);
                             }));

  return std::make_tuple(values, indices);
}
