// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// GPU-side weight decompression (and compression) for CompressedModelLoader.
//
// Supported algorithms:
//   lz4      — nvCOMP batched LZ4 block decompression + compression (GPU)
//              Falls back to CPU zlib if nvCOMP is not available.
//   gdeflate — nvCOMP batched GDeflate decompression + compression (GPU only)
//              Errors if nvCOMP is not available (no CPU fallback exists).
//
// On-disk/in-memory format for each compressed tensor (both algorithms):
//   [4 bytes: uint32_t  n_chunks]
//   [n_chunks × 4 bytes: uint32_t comp_size[i]]
//   [chunk_0 compressed bytes]
//   [chunk_1 compressed bytes]
//   ...
//
// CHUNK_SIZE = 65536 bytes (uncompressed, except last chunk).
// This constant MUST match _LZ4_CHUNK_SIZE in compress_weights.py and
// compressed_loader.py.
//
// Build with nvCOMP:
//   cmake .. -DVLLM_NVCOMP_PATH=/path/to/nvcomp/install

#include "ops.h"

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <mutex>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstring>

// ---------------------------------------------------------------------------
// zlib fallback (always available — zlib ships with every Linux distro)
// ---------------------------------------------------------------------------
#include <zlib.h>

namespace {

// Decompress DEFLATE-compressed bytes (zlib format) on CPU.
std::vector<uint8_t> zlib_decompress(const uint8_t* src, size_t src_len,
                                      size_t expected_out_len) {
  std::vector<uint8_t> dst(expected_out_len);
  uLongf dest_len = static_cast<uLongf>(expected_out_len);
  int ret = uncompress(reinterpret_cast<Bytef*>(dst.data()), &dest_len,
                       reinterpret_cast<const Bytef*>(src), src_len);
  if (ret != Z_OK) {
    throw std::runtime_error(
        std::string("zlib decompress failed: ") + std::to_string(ret));
  }
  dst.resize(dest_len);
  return dst;
}

// Map dtype string to torch ScalarType
at::ScalarType dtype_str_to_scalar(const std::string& dtype) {
  if (dtype == "float32")        return at::kFloat;
  if (dtype == "float16")        return at::kHalf;
  if (dtype == "bfloat16")       return at::kBFloat16;
  if (dtype == "int32")          return at::kInt;
  if (dtype == "int8")           return at::kChar;
  if (dtype == "uint8")          return at::kByte;
  if (dtype == "int64")          return at::kLong;
  if (dtype == "float8_e4m3fn")  return at::kFloat8_e4m3fn;
  if (dtype == "float8_e5m2")    return at::kFloat8_e5m2;
  throw std::invalid_argument("Unsupported dtype string: " + dtype);
}

// Uncompressed chunk size (must match _LZ4_CHUNK_SIZE in compress_weights.py).
constexpr size_t DECOMP_CHUNK_SIZE = 65536;

// Maximum number of chunks we pre-allocate in the pool.
// 32768 x 65536 = 2 GB — covers embedding tables and other large weight matrices.
constexpr size_t MAX_POOL_CHUNKS = 32768;

}  // namespace

// ---------------------------------------------------------------------------
// nvCOMP-based GPU paths (LZ4 decompression + GDeflate compress/decompress)
// ---------------------------------------------------------------------------
#ifdef HAVE_NVCOMP

#include <nvcomp/lz4.h>
#include <nvcomp/gdeflate.h>

// Persistent per-process pool of device-side metadata buffers shared by
// both the LZ4 and GDeflate decompression paths.
//
// Pre-allocating device arrays for up to MAX_POOL_CHUNKS chunks eliminates
// per-call cudaMalloc/cudaFree overhead.  The temp workspace is grown lazily
// to the maximum needed by any algorithm seen so far.
//
// Thread-safety: one-time initialisation is guarded by mutex + bool flag.
struct DecompressPool {
  void**            d_comp_ptrs    = nullptr;  // MAX_POOL_CHUNKS x void*
  size_t*           d_comp_sizes   = nullptr;  // MAX_POOL_CHUNKS x size_t
  size_t*           d_uncomp_sizes = nullptr;  // MAX_POOL_CHUNKS x size_t
  size_t*           d_actual_uncomp= nullptr;  // MAX_POOL_CHUNKS x size_t
  void**            d_out_ptrs     = nullptr;  // MAX_POOL_CHUNKS x void*
  nvcompStatus_t*   d_statuses     = nullptr;  // MAX_POOL_CHUNKS x nvcompStatus_t
  void*             d_temp         = nullptr;
  size_t            d_temp_capacity= 0;
  std::mutex        mu;
  bool              ready          = false;

  void init() {
    std::lock_guard<std::mutex> lock(mu);
    if (ready) return;
    cudaMalloc(&d_comp_ptrs,    MAX_POOL_CHUNKS * sizeof(void*));
    cudaMalloc(&d_comp_sizes,   MAX_POOL_CHUNKS * sizeof(size_t));
    cudaMalloc(&d_uncomp_sizes, MAX_POOL_CHUNKS * sizeof(size_t));
    cudaMalloc(&d_actual_uncomp,MAX_POOL_CHUNKS * sizeof(size_t));
    cudaMalloc(&d_out_ptrs,     MAX_POOL_CHUNKS * sizeof(void*));
    cudaMalloc(&d_statuses,     MAX_POOL_CHUNKS * sizeof(nvcompStatus_t));
    ready = true;
  }

  // Ensure the temp workspace is at least `needed` bytes.
  void ensure_temp(size_t needed) {
    if (needed <= d_temp_capacity) return;
    std::lock_guard<std::mutex> lock(mu);
    if (needed <= d_temp_capacity) return;  // double-checked
    if (d_temp) cudaFree(d_temp);
    cudaMalloc(&d_temp, needed);
    d_temp_capacity = needed;
  }
};

static DecompressPool g_decomp_pool;  // one pool per process

// ---------------------------------------------------------------------------
// LZ4 decompression
// ---------------------------------------------------------------------------

torch::Tensor decompress_tensor_gpu_lz4(
    const torch::Tensor& compressed_gpu,
    const std::vector<int64_t>& shape,
    const std::string& dtype,
    int64_t original_size) {

  at::cuda::CUDAGuard device_guard(compressed_gpu.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::ScalarType scalar_type = dtype_str_to_scalar(dtype);
  auto output = torch::empty(shape,
                             torch::TensorOptions()
                                 .dtype(scalar_type)
                                 .device(compressed_gpu.device()));

  const uint8_t* comp_data = reinterpret_cast<const uint8_t*>(
      compressed_gpu.data_ptr());

  // 1. Copy n_chunks from device header
  uint32_t n_chunks;
  cudaMemcpy(&n_chunks, comp_data, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  TORCH_CHECK(n_chunks > 0 && n_chunks <= MAX_POOL_CHUNKS,
              "decompress_tensor: n_chunks out of range: ", n_chunks);

  // 2. Copy per-chunk compressed sizes from device header
  std::vector<uint32_t> h_comp_sizes32(n_chunks);
  cudaMemcpy(h_comp_sizes32.data(), comp_data + sizeof(uint32_t),
            n_chunks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  // 3. Build host-side pointer/size arrays
  const size_t header_bytes = sizeof(uint32_t) + n_chunks * sizeof(uint32_t);

  std::vector<const void*> h_comp_ptrs(n_chunks);
  std::vector<size_t>      h_comp_bytes(n_chunks);
  std::vector<void*>       h_out_ptrs(n_chunks);
  std::vector<size_t>      h_uncomp_bytes(n_chunks);

  size_t data_offset = header_bytes;
  uint8_t* out_data  = reinterpret_cast<uint8_t*>(output.data_ptr());

  for (uint32_t i = 0; i < n_chunks; i++) {
    h_comp_ptrs[i]  = comp_data + data_offset;
    h_comp_bytes[i] = static_cast<size_t>(h_comp_sizes32[i]);
    h_out_ptrs[i]   = out_data + static_cast<size_t>(i) * DECOMP_CHUNK_SIZE;
    size_t remaining = static_cast<size_t>(original_size) -
                       static_cast<size_t>(i) * DECOMP_CHUNK_SIZE;
    h_uncomp_bytes[i] = std::min(remaining, DECOMP_CHUNK_SIZE);
    data_offset += h_comp_sizes32[i];
  }

  // 4. Ensure pool ready, grow temp if needed
  g_decomp_pool.init();
  size_t lz4_temp_needed = 0;
  nvcompBatchedLZ4DecompressGetTempSize(n_chunks, DECOMP_CHUNK_SIZE,
                                         &lz4_temp_needed);
  g_decomp_pool.ensure_temp(lz4_temp_needed);

  // 5. Upload pointer/size arrays to pre-allocated device buffers
  cudaMemcpyAsync(g_decomp_pool.d_comp_ptrs,    h_comp_ptrs.data(),
                 n_chunks * sizeof(void*),   cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_decomp_pool.d_comp_sizes,   h_comp_bytes.data(),
                 n_chunks * sizeof(size_t),  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_decomp_pool.d_uncomp_sizes, h_uncomp_bytes.data(),
                 n_chunks * sizeof(size_t),  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_decomp_pool.d_out_ptrs,     h_out_ptrs.data(),
                 n_chunks * sizeof(void*),   cudaMemcpyHostToDevice, stream);

  // 6. Launch batched LZ4 decompression
  nvcompStatus_t status = nvcompBatchedLZ4DecompressAsync(
      reinterpret_cast<const void* const*>(g_decomp_pool.d_comp_ptrs),
      reinterpret_cast<const size_t*>    (g_decomp_pool.d_comp_sizes),
      reinterpret_cast<const size_t*>    (g_decomp_pool.d_uncomp_sizes),
      reinterpret_cast<size_t*>          (g_decomp_pool.d_actual_uncomp),
      static_cast<size_t>(n_chunks),
      g_decomp_pool.d_temp,
      g_decomp_pool.d_temp_capacity,
      reinterpret_cast<void* const*>     (g_decomp_pool.d_out_ptrs),
      g_decomp_pool.d_statuses,
      stream);

  cudaStreamSynchronize(stream);

  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "nvCOMP LZ4 decompression failed with status: " +
        std::to_string(status));
  }

  return output;
}

// ---------------------------------------------------------------------------
// LZ4 compression (GPU-side; same chunked format as CPU lz4.block compressor)
// ---------------------------------------------------------------------------
torch::Tensor compress_tensor_gpu_lz4(
    const torch::Tensor& raw_gpu) {

  TORCH_CHECK(raw_gpu.is_cuda(),
              "compress_tensor: raw_gpu must be a CUDA tensor");
  TORCH_CHECK(raw_gpu.scalar_type() == at::kByte,
              "compress_tensor: raw_gpu must have dtype uint8");
  TORCH_CHECK(raw_gpu.is_contiguous(),
              "compress_tensor: raw_gpu must be contiguous");

  at::cuda::CUDAGuard device_guard(raw_gpu.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const size_t total_bytes = static_cast<size_t>(raw_gpu.numel());
  const size_t n_chunks =
      (total_bytes + DECOMP_CHUNK_SIZE - 1) / DECOMP_CHUNK_SIZE;

  TORCH_CHECK(n_chunks > 0 && n_chunks <= MAX_POOL_CHUNKS,
              "compress_tensor: tensor too large (", n_chunks, " chunks, max ",
              MAX_POOL_CHUNKS, ")");

  // LZ4 opts: use byte (uint8) data type
  nvcompBatchedLZ4Opts_t opts = {NVCOMP_TYPE_UCHAR};

  // Query workspace and max output sizes
  size_t temp_size = 0;
  nvcompBatchedLZ4CompressGetTempSize(n_chunks, DECOMP_CHUNK_SIZE,
                                       opts, &temp_size);
  size_t max_out_chunk = 0;
  nvcompBatchedLZ4CompressGetMaxOutputChunkSize(DECOMP_CHUNK_SIZE,
                                                 opts, &max_out_chunk);

  // Build host-side input pointer/size arrays
  const uint8_t* raw_ptr = reinterpret_cast<const uint8_t*>(raw_gpu.data_ptr());

  std::vector<const void*> h_in_ptrs(n_chunks);
  std::vector<size_t>      h_in_sizes(n_chunks);
  for (size_t i = 0; i < n_chunks; i++) {
    h_in_ptrs[i]  = raw_ptr + i * DECOMP_CHUNK_SIZE;
    h_in_sizes[i] = std::min(DECOMP_CHUNK_SIZE,
                             total_bytes - i * DECOMP_CHUNK_SIZE);
  }

  // Allocate device buffers (per-call; compression is offline, not hot-path)
  void*   d_temp    = nullptr;  cudaMalloc(&d_temp,    temp_size ? temp_size : 1);
  void*   d_out_buf = nullptr;  cudaMalloc(&d_out_buf, n_chunks * max_out_chunk);

  const void** d_in_ptrs  = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_in_ptrs),  n_chunks * sizeof(const void*));
  size_t* d_in_sizes  = nullptr;  cudaMalloc(&d_in_sizes,  n_chunks * sizeof(size_t));
  void**  d_out_ptrs  = nullptr;  cudaMalloc(&d_out_ptrs,  n_chunks * sizeof(void*));
  size_t* d_out_sizes = nullptr;  cudaMalloc(&d_out_sizes, n_chunks * sizeof(size_t));

  // Build and upload strided output pointer array
  std::vector<void*> h_out_ptrs(n_chunks);
  uint8_t* out_buf_ptr = static_cast<uint8_t*>(d_out_buf);
  for (size_t i = 0; i < n_chunks; i++) {
    h_out_ptrs[i] = out_buf_ptr + i * max_out_chunk;
  }

  cudaMemcpyAsync(d_in_ptrs,  h_in_ptrs.data(),  n_chunks * sizeof(const void*),
                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_in_sizes, h_in_sizes.data(), n_chunks * sizeof(size_t),
                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_out_ptrs, h_out_ptrs.data(), n_chunks * sizeof(void*),
                 cudaMemcpyHostToDevice, stream);

  // Launch batched LZ4 compression
  nvcompStatus_t status = nvcompBatchedLZ4CompressAsync(
      reinterpret_cast<const void* const*>(d_in_ptrs),
      reinterpret_cast<const size_t*>(d_in_sizes),
      DECOMP_CHUNK_SIZE,
      n_chunks,
      d_temp,
      temp_size,
      reinterpret_cast<void* const*>(d_out_ptrs),
      d_out_sizes,
      opts,
      stream);

  cudaStreamSynchronize(stream);

  if (status != nvcompSuccess) {
    cudaFree(d_temp); cudaFree(d_out_buf);
    cudaFree(d_in_ptrs); cudaFree(d_in_sizes);
    cudaFree(d_out_ptrs); cudaFree(d_out_sizes);
    throw std::runtime_error(
        "nvCOMP LZ4 compression failed with status: " +
        std::to_string(status));
  }

  // D2H: actual compressed sizes per chunk
  std::vector<size_t> h_out_sizes(n_chunks);
  cudaMemcpy(h_out_sizes.data(), d_out_sizes,
            n_chunks * sizeof(size_t), cudaMemcpyDeviceToHost);

  // D2H: full strided output buffer (compacted on CPU below)
  const size_t d_out_buf_bytes = n_chunks * max_out_chunk;
  std::vector<uint8_t> h_out_buf(d_out_buf_bytes);
  cudaMemcpy(h_out_buf.data(), d_out_buf, d_out_buf_bytes, cudaMemcpyDeviceToHost);

  // Free device allocations
  cudaFree(d_temp); cudaFree(d_out_buf);
  cudaFree(d_in_ptrs); cudaFree(d_in_sizes);
  cudaFree(d_out_ptrs); cudaFree(d_out_sizes);

  // Build output in our chunked format on CPU:
  //   [4-byte n_chunks][n_chunks x 4-byte comp_sizes][chunk_0][chunk_1]...
  const size_t header_size = sizeof(uint32_t) + n_chunks * sizeof(uint32_t);
  size_t total_comp = header_size;
  for (size_t i = 0; i < n_chunks; i++) total_comp += h_out_sizes[i];

  auto output = torch::empty(
      {static_cast<int64_t>(total_comp)},
      torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  uint8_t* out_data = reinterpret_cast<uint8_t*>(output.data_ptr());

  // Write header
  uint32_t n32 = static_cast<uint32_t>(n_chunks);
  std::memcpy(out_data, &n32, sizeof(uint32_t));
  for (size_t i = 0; i < n_chunks; i++) {
    uint32_t sz32 = static_cast<uint32_t>(h_out_sizes[i]);
    std::memcpy(out_data + sizeof(uint32_t) + i * sizeof(uint32_t),
                &sz32, sizeof(uint32_t));
  }

  // Write compacted compressed chunks (skip stride padding in h_out_buf)
  size_t dst = header_size;
  for (size_t i = 0; i < n_chunks; i++) {
    std::memcpy(out_data + dst,
                h_out_buf.data() + i * max_out_chunk,
                h_out_sizes[i]);
    dst += h_out_sizes[i];
  }

  return output;  // CPU tensor — compatible with decompress_tensor_gpu_lz4()
}

// ---------------------------------------------------------------------------
// GDeflate decompression
// ---------------------------------------------------------------------------

torch::Tensor decompress_tensor_gpu_gdeflate(
    const torch::Tensor& compressed_gpu,
    const std::vector<int64_t>& shape,
    const std::string& dtype,
    int64_t original_size) {

  at::cuda::CUDAGuard device_guard(compressed_gpu.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::ScalarType scalar_type = dtype_str_to_scalar(dtype);
  auto output = torch::empty(shape,
                             torch::TensorOptions()
                                 .dtype(scalar_type)
                                 .device(compressed_gpu.device()));

  const uint8_t* comp_data = reinterpret_cast<const uint8_t*>(
      compressed_gpu.data_ptr());

  // 1. Parse header (identical format to LZ4)
  uint32_t n_chunks;
  cudaMemcpy(&n_chunks, comp_data, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  TORCH_CHECK(n_chunks > 0 && n_chunks <= MAX_POOL_CHUNKS,
              "decompress_tensor (gdeflate): n_chunks out of range: ", n_chunks);

  std::vector<uint32_t> h_comp_sizes32(n_chunks);
  cudaMemcpy(h_comp_sizes32.data(), comp_data + sizeof(uint32_t),
            n_chunks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  const size_t header_bytes = sizeof(uint32_t) + n_chunks * sizeof(uint32_t);

  // 2. Build host-side pointer/size arrays
  std::vector<const void*> h_comp_ptrs(n_chunks);
  std::vector<size_t>      h_comp_bytes(n_chunks);
  std::vector<void*>       h_out_ptrs(n_chunks);
  std::vector<size_t>      h_uncomp_bytes(n_chunks);

  size_t data_offset = header_bytes;
  uint8_t* out_data  = reinterpret_cast<uint8_t*>(output.data_ptr());

  for (uint32_t i = 0; i < n_chunks; i++) {
    h_comp_ptrs[i]  = comp_data + data_offset;
    h_comp_bytes[i] = static_cast<size_t>(h_comp_sizes32[i]);
    h_out_ptrs[i]   = out_data + static_cast<size_t>(i) * DECOMP_CHUNK_SIZE;
    size_t remaining = static_cast<size_t>(original_size) -
                       static_cast<size_t>(i) * DECOMP_CHUNK_SIZE;
    h_uncomp_bytes[i] = std::min(remaining, DECOMP_CHUNK_SIZE);
    data_offset += h_comp_sizes32[i];
  }

  // 3. Ensure shared pool is ready; grow temp for GDeflate if needed
  g_decomp_pool.init();
  size_t gdeflate_temp_needed = 0;
  nvcompBatchedGdeflateDecompressGetTempSize(n_chunks, DECOMP_CHUNK_SIZE,
                                              &gdeflate_temp_needed);
  g_decomp_pool.ensure_temp(gdeflate_temp_needed);

  // 4. Upload pointer/size arrays to pre-allocated device buffers
  cudaMemcpyAsync(g_decomp_pool.d_comp_ptrs,    h_comp_ptrs.data(),
                 n_chunks * sizeof(void*),   cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_decomp_pool.d_comp_sizes,   h_comp_bytes.data(),
                 n_chunks * sizeof(size_t),  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_decomp_pool.d_uncomp_sizes, h_uncomp_bytes.data(),
                 n_chunks * sizeof(size_t),  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(g_decomp_pool.d_out_ptrs,     h_out_ptrs.data(),
                 n_chunks * sizeof(void*),   cudaMemcpyHostToDevice, stream);

  // 5. Launch batched GDeflate decompression
  nvcompStatus_t status = nvcompBatchedGdeflateDecompressAsync(
      reinterpret_cast<const void* const*>(g_decomp_pool.d_comp_ptrs),
      reinterpret_cast<const size_t*>    (g_decomp_pool.d_comp_sizes),
      reinterpret_cast<const size_t*>    (g_decomp_pool.d_uncomp_sizes),
      reinterpret_cast<size_t*>          (g_decomp_pool.d_actual_uncomp),
      static_cast<size_t>(n_chunks),
      g_decomp_pool.d_temp,
      g_decomp_pool.d_temp_capacity,
      reinterpret_cast<void* const*>     (g_decomp_pool.d_out_ptrs),
      g_decomp_pool.d_statuses,
      stream);

  cudaStreamSynchronize(stream);

  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "nvCOMP GDeflate decompression failed with status: " +
        std::to_string(status));
  }

  return output;
}

// ---------------------------------------------------------------------------
// GDeflate compression (offline / setup-time use)
// ---------------------------------------------------------------------------
torch::Tensor compress_tensor_gpu_gdeflate(
    const torch::Tensor& raw_gpu,
    int64_t algo_level) {

  TORCH_CHECK(raw_gpu.is_cuda(),
              "compress_tensor: raw_gpu must be a CUDA tensor");
  TORCH_CHECK(raw_gpu.scalar_type() == at::kByte,
              "compress_tensor: raw_gpu must have dtype uint8");
  TORCH_CHECK(raw_gpu.is_contiguous(),
              "compress_tensor: raw_gpu must be contiguous");

  at::cuda::CUDAGuard device_guard(raw_gpu.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const size_t total_bytes = static_cast<size_t>(raw_gpu.numel());
  const size_t n_chunks =
      (total_bytes + DECOMP_CHUNK_SIZE - 1) / DECOMP_CHUNK_SIZE;

  TORCH_CHECK(n_chunks > 0 && n_chunks <= MAX_POOL_CHUNKS,
              "compress_tensor: tensor too large (", n_chunks, " chunks, max ",
              MAX_POOL_CHUNKS, ")");

  nvcompBatchedGdeflateOpts_t opts = {static_cast<int>(algo_level)};

  // Query workspace and max output sizes
  size_t temp_size = 0;
  nvcompBatchedGdeflateCompressGetTempSize(n_chunks, DECOMP_CHUNK_SIZE,
                                            opts, &temp_size);
  size_t max_out_chunk = 0;
  nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(DECOMP_CHUNK_SIZE,
                                                      opts, &max_out_chunk);

  // Build host-side input pointer/size arrays
  const uint8_t* raw_ptr = reinterpret_cast<const uint8_t*>(raw_gpu.data_ptr());

  std::vector<const void*> h_in_ptrs(n_chunks);
  std::vector<size_t>      h_in_sizes(n_chunks);
  for (size_t i = 0; i < n_chunks; i++) {
    h_in_ptrs[i]  = raw_ptr + i * DECOMP_CHUNK_SIZE;
    h_in_sizes[i] = std::min(DECOMP_CHUNK_SIZE,
                             total_bytes - i * DECOMP_CHUNK_SIZE);
  }

  // Allocate device buffers (per-call; compression is offline, not hot-path)
  void*   d_temp    = nullptr;  cudaMalloc(&d_temp,    temp_size);
  void*   d_out_buf = nullptr;  cudaMalloc(&d_out_buf, n_chunks * max_out_chunk);

  std::vector<void*> h_out_ptrs(n_chunks);
  uint8_t* out_buf_ptr = static_cast<uint8_t*>(d_out_buf);
  for (size_t i = 0; i < n_chunks; i++) {
    h_out_ptrs[i] = out_buf_ptr + i * max_out_chunk;
  }

  const void** d_in_ptrs  = nullptr;
  cudaMalloc(reinterpret_cast<void**>(&d_in_ptrs),  n_chunks * sizeof(const void*));
  size_t* d_in_sizes  = nullptr;  cudaMalloc(&d_in_sizes,  n_chunks * sizeof(size_t));
  void**  d_out_ptrs  = nullptr;  cudaMalloc(&d_out_ptrs,  n_chunks * sizeof(void*));
  size_t* d_out_sizes = nullptr;  cudaMalloc(&d_out_sizes, n_chunks * sizeof(size_t));

  // Upload pointer/size arrays to device
  cudaMemcpyAsync(d_in_ptrs,  h_in_ptrs.data(),  n_chunks * sizeof(const void*),
                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_in_sizes, h_in_sizes.data(), n_chunks * sizeof(size_t),
                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_out_ptrs, h_out_ptrs.data(), n_chunks * sizeof(void*),
                 cudaMemcpyHostToDevice, stream);

  // Launch batched GDeflate compression
  nvcompStatus_t status = nvcompBatchedGdeflateCompressAsync(
      reinterpret_cast<const void* const*>(d_in_ptrs),
      reinterpret_cast<const size_t*>(d_in_sizes),
      DECOMP_CHUNK_SIZE,
      n_chunks,
      d_temp,
      temp_size,
      reinterpret_cast<void* const*>(d_out_ptrs),
      d_out_sizes,
      opts,
      stream);

  cudaStreamSynchronize(stream);

  if (status != nvcompSuccess) {
    cudaFree(d_temp); cudaFree(d_out_buf);
    cudaFree(d_in_ptrs); cudaFree(d_in_sizes);
    cudaFree(d_out_ptrs); cudaFree(d_out_sizes);
    throw std::runtime_error(
        "nvCOMP GDeflate compression failed with status: " +
        std::to_string(status));
  }

  // D2H: actual compressed sizes per chunk
  std::vector<size_t> h_out_sizes(n_chunks);
  cudaMemcpy(h_out_sizes.data(), d_out_sizes,
            n_chunks * sizeof(size_t), cudaMemcpyDeviceToHost);

  // D2H: full strided output buffer (compacted on CPU below)
  const size_t d_out_buf_bytes = n_chunks * max_out_chunk;
  std::vector<uint8_t> h_out_buf(d_out_buf_bytes);
  cudaMemcpy(h_out_buf.data(), d_out_buf, d_out_buf_bytes, cudaMemcpyDeviceToHost);

  // Free device allocations
  cudaFree(d_temp); cudaFree(d_out_buf);
  cudaFree(d_in_ptrs); cudaFree(d_in_sizes);
  cudaFree(d_out_ptrs); cudaFree(d_out_sizes);

  // Build output in our chunked format on CPU
  const size_t header_size = sizeof(uint32_t) + n_chunks * sizeof(uint32_t);
  size_t total_comp = header_size;
  for (size_t i = 0; i < n_chunks; i++) total_comp += h_out_sizes[i];

  auto output = torch::empty(
      {static_cast<int64_t>(total_comp)},
      torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  uint8_t* out_data_ptr = reinterpret_cast<uint8_t*>(output.data_ptr());

  // Write header
  uint32_t n32 = static_cast<uint32_t>(n_chunks);
  std::memcpy(out_data_ptr, &n32, sizeof(uint32_t));
  for (size_t i = 0; i < n_chunks; i++) {
    uint32_t sz32 = static_cast<uint32_t>(h_out_sizes[i]);
    std::memcpy(out_data_ptr + sizeof(uint32_t) + i * sizeof(uint32_t),
                &sz32, sizeof(uint32_t));
  }

  // Write compacted compressed chunks
  size_t dst_off = header_size;
  for (size_t i = 0; i < n_chunks; i++) {
    std::memcpy(out_data_ptr + dst_off,
                h_out_buf.data() + i * max_out_chunk,
                h_out_sizes[i]);
    dst_off += h_out_sizes[i];
  }

  return output;
}

#endif  // HAVE_NVCOMP

// ---------------------------------------------------------------------------
// CPU fallback: parse chunked header, decompress each chunk on CPU with zlib,
// then async H2D copy.
// ---------------------------------------------------------------------------
torch::Tensor decompress_tensor_gpu_cpu_fallback(
    const torch::Tensor& compressed_gpu,
    const std::vector<int64_t>& shape,
    const std::string& dtype,
    int64_t original_size,
    const std::string& algorithm) {

  if (algorithm == "gdeflate") {
    throw std::runtime_error(
        "GDeflate decompression requires nvCOMP. "
        "This build does not include nvCOMP support. "
        "Rebuild vLLM with -DVLLM_NVCOMP_PATH=... or use --algorithm lz4.");
  }

  at::cuda::CUDAGuard device_guard(compressed_gpu.device());

  at::ScalarType scalar_type = dtype_str_to_scalar(dtype);

  // 1. Move compressed bytes from GPU to CPU
  torch::Tensor compressed_cpu = compressed_gpu.to(at::kCPU);
  const uint8_t* comp_data = reinterpret_cast<const uint8_t*>(
      compressed_cpu.data_ptr());

  // 2. Parse chunked LZ4 header
  uint32_t n_chunks;
  std::memcpy(&n_chunks, comp_data, sizeof(uint32_t));
  TORCH_CHECK(n_chunks > 0 && n_chunks <= MAX_POOL_CHUNKS,
              "decompress_tensor: n_chunks out of range: ", n_chunks);

  std::vector<uint32_t> comp_sizes32(n_chunks);
  std::memcpy(comp_sizes32.data(), comp_data + sizeof(uint32_t),
              n_chunks * sizeof(uint32_t));

  const size_t header_bytes = sizeof(uint32_t) + n_chunks * sizeof(uint32_t);

  // 3. Decompress each chunk on CPU with zlib
  std::vector<uint8_t> raw(static_cast<size_t>(original_size));
  size_t src_offset = header_bytes;
  size_t dst_offset = 0;

  for (uint32_t i = 0; i < n_chunks; i++) {
    size_t cs = comp_sizes32[i];
    size_t remaining = static_cast<size_t>(original_size) - dst_offset;
    size_t uncomp_size = std::min(remaining, DECOMP_CHUNK_SIZE);

    auto chunk = zlib_decompress(comp_data + src_offset, cs, uncomp_size);
    std::memcpy(raw.data() + dst_offset, chunk.data(), chunk.size());

    src_offset += cs;
    dst_offset += chunk.size();
  }

  // 4. Wrap raw bytes in a typed CPU tensor (no copy)
  torch::Tensor cpu_tensor = torch::from_blob(
      raw.data(),
      {static_cast<int64_t>(raw.size())},
      torch::TensorOptions().dtype(at::kByte).device(at::kCPU));
  torch::Tensor typed_cpu = cpu_tensor.view(scalar_type).reshape(shape);

  // 5. Async H2D copy to GPU
  auto output = torch::empty(shape,
                             torch::TensorOptions()
                                 .dtype(scalar_type)
                                 .device(compressed_gpu.device()));
  output.copy_(typed_cpu, /*non_blocking=*/true);

  return output;
}

// ---------------------------------------------------------------------------
// Public API: decompress_tensor
// ---------------------------------------------------------------------------
torch::Tensor decompress_tensor(
    const torch::Tensor& compressed_gpu,
    const std::vector<int64_t>& shape,
    const std::string& dtype,
    int64_t original_size,
    const std::string& algorithm) {

  TORCH_CHECK(compressed_gpu.is_cuda(),
              "decompress_tensor: compressed_gpu must be a CUDA tensor");
  TORCH_CHECK(compressed_gpu.scalar_type() == at::kByte,
              "decompress_tensor: compressed_gpu must have dtype uint8");
  TORCH_CHECK(compressed_gpu.is_contiguous(),
              "decompress_tensor: compressed_gpu must be contiguous");
  TORCH_CHECK(original_size > 0,
              "decompress_tensor: original_size must be > 0");

#ifdef HAVE_NVCOMP
  if (algorithm == "lz4") {
    return decompress_tensor_gpu_lz4(compressed_gpu, shape, dtype, original_size);
  } else if (algorithm == "gdeflate") {
    return decompress_tensor_gpu_gdeflate(compressed_gpu, shape, dtype, original_size);
  }
  throw std::invalid_argument(
      "decompress_tensor: unsupported algorithm: " + algorithm);
#else
  return decompress_tensor_gpu_cpu_fallback(
      compressed_gpu, shape, dtype, original_size, algorithm);
#endif
}

// ---------------------------------------------------------------------------
// Public API: compress_tensor
// ---------------------------------------------------------------------------
torch::Tensor compress_tensor(
    const torch::Tensor& raw_gpu,
    const std::string& algorithm,
    int64_t level) {

#ifdef HAVE_NVCOMP
  if (algorithm == "lz4") {
    return compress_tensor_gpu_lz4(raw_gpu);
  }
  if (algorithm == "gdeflate") {
    return compress_tensor_gpu_gdeflate(raw_gpu, level);
  }
  throw std::invalid_argument(
      "compress_tensor: unsupported algorithm: " + algorithm +
      ". Supported: lz4, gdeflate");
#else
  throw std::runtime_error(
      "compress_tensor requires nvCOMP. "
      "Rebuild vLLM with -DVLLM_NVCOMP_PATH=... .");
#endif
}

// ---------------------------------------------------------------------------
// Query whether GPU decompression (nvCOMP) is available
// ---------------------------------------------------------------------------
bool is_gpu_decompress_available() {
#ifdef HAVE_NVCOMP
  return true;
#else
  return false;
#endif
}

// ---------------------------------------------------------------------------
// Query whether GPU compression (nvCOMP LZ4 or GDeflate) is available
// ---------------------------------------------------------------------------
bool is_gpu_compress_available() {
#ifdef HAVE_NVCOMP
  return true;
#else
  return false;
#endif
}
