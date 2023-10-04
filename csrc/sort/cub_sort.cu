#include <cuda_runtime.h>
#include <cub/cub.cuh>
void vllm_sort_cuda(const float *src, float *dst, int batch_size, int len,
                    bool desending) {
  int num_items = batch_size * len;
  int num_segments = batch_size;
  const float *d_keys_in = src;
  float *d_keys_out = dst;

  int *h_offsets = new int[num_segments + 1];
  int *d_offsets = nullptr;

  size_t temp_storage_bytes = 0;
  void *d_temp_storage = nullptr;

  for (int i = 0; i <= num_segments; ++i) {
    h_offsets[i] = i * len;
  }

  cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int));
  cudaMemcpy(d_offsets, h_offsets, (num_segments + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  if (!desending) {
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
        num_segments, d_offsets, d_offsets + 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
        num_segments, d_offsets, d_offsets + 1);
  } else {
    cub::DeviceSegmentedRadixSort::SortKeysDescending(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
        num_segments, d_offsets, d_offsets + 1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedRadixSort::SortKeysDescending(
        d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
        num_segments, d_offsets, d_offsets + 1);
  }

  delete[] h_offsets;
  cudaFree(d_offsets);
  cudaFree(d_temp_storage);
}
