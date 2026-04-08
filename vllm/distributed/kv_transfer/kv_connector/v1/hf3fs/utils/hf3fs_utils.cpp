#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstring>
#include <limits>
#include <vector>

// Compute the byte size of a tensor with explicit overflow protection.
// Directly computing numel() * element_size() can silently overflow; this
// helper guards against integer overflow and validates the result.
static inline size_t safe_tensor_bytes(const torch::Tensor& t,
                                       const char* context) {
  TORCH_CHECK(t.numel() >= 0,
              context, ": tensor numel() must be non-negative, got ", t.numel());
  const size_t numel = static_cast<size_t>(t.numel());
  const size_t elem_size = static_cast<size_t>(t.element_size());
  TORCH_CHECK(elem_size == 0 || numel <= std::numeric_limits<size_t>::max() / elem_size,
              context, ": tensor size calculation would overflow (numel=",
              numel, ", element_size=", elem_size, ")");
  return numel * elem_size;
}

// Bounds-checked memcpy: validates that dst has sufficient capacity and that
// neither pointer is null before performing the copy. This prevents
// out-of-bounds writes that corrupt adjacent memory structures.
static inline void checked_memcpy(void* dst, const void* src, size_t copy_size,
                                   size_t dst_capacity, const char* context) {
  TORCH_CHECK(dst != nullptr, context, ": destination pointer must not be null");
  TORCH_CHECK(src != nullptr, context, ": source pointer must not be null");
  TORCH_CHECK(copy_size <= dst_capacity,
              context, ": copy size (", copy_size,
              ") exceeds destination buffer capacity (", dst_capacity, ")");
  std::memcpy(dst, src, copy_size);
}

void read_shm(const torch::Tensor& shm, const torch::Tensor& pin,
              std::vector<torch::Tensor> dst, uint64_t stream_ptr) {
  py::gil_scoped_release release;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // Compute sizes with overflow protection.
  const size_t shm_bytes = safe_tensor_bytes(shm, "read_shm");
  const size_t pin_bytes = safe_tensor_bytes(pin, "read_shm");

  // Copy from shared memory to pinned memory.
  // checked_memcpy enforces that shm_bytes <= pin_bytes (dst capacity) before
  // performing the copy, preventing out-of-bounds writes into the pin buffer.
  checked_memcpy(pin.data_ptr(), shm.data_ptr(), shm_bytes, pin_bytes,
                 "read_shm memcpy(pin, shm)");

  // Copy from pinned memory to GPU tensors.
  const char* src_ptr = static_cast<const char*>(pin.data_ptr());
  size_t current = 0;
  for (size_t i = 0; i < dst.size(); ++i) {
    auto& t = dst[i];
    const size_t t_bytes = safe_tensor_bytes(t, "read_shm");
    // Subtraction-form check avoids unsigned integer overflow in the sum.
    TORCH_CHECK(t_bytes <= shm_bytes - current,
                "read_shm: tensor[", i, "] copy would exceed shared memory "
                "buffer bounds (current=", current, ", t_bytes=", t_bytes,
                ", shm_bytes=", shm_bytes, ")");
    char* dst_ptr = static_cast<char*>(t.data_ptr());
    cudaMemcpyAsync(dst_ptr, src_ptr + current, t_bytes, cudaMemcpyHostToDevice,
                    stream);
    current += t_bytes;
  }
  cudaStreamSynchronize(stream);
}

void write_shm(const std::vector<torch::Tensor> src, torch::Tensor& shm,
               const torch::Tensor& pin, uint64_t stream_ptr) {
  py::gil_scoped_release release;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // Compute sizes with overflow protection.
  const size_t shm_bytes = safe_tensor_bytes(shm, "write_shm");
  const size_t pin_bytes = safe_tensor_bytes(pin, "write_shm");

  // Copy from GPU tensors to pinned memory.
  char* dst_ptr = static_cast<char*>(pin.data_ptr());
  size_t current = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    auto& t = src[i];
    const size_t t_bytes = safe_tensor_bytes(t, "write_shm");
    // Subtraction-form check avoids unsigned integer overflow in the sum.
    TORCH_CHECK(t_bytes <= pin_bytes - current,
                "write_shm: tensor[", i, "] copy would exceed pinned memory "
                "buffer bounds (current=", current, ", t_bytes=", t_bytes,
                ", pin_bytes=", pin_bytes, ")");
    char* src_ptr = static_cast<char*>(t.data_ptr());
    cudaMemcpyAsync(dst_ptr + current, src_ptr, t_bytes, cudaMemcpyDeviceToHost,
                    stream);
    current += t_bytes;
  }
  cudaStreamSynchronize(stream);

  // Copy from pinned memory to shared memory.
  // Validate that the pin (source) buffer contains at least shm_bytes of
  // data before copying, preventing an out-of-bounds read from pin.
  TORCH_CHECK(pin_bytes >= shm_bytes,
              "write_shm: pinned memory buffer (", pin_bytes,
              " bytes) is smaller than shared memory buffer (", shm_bytes, " bytes)");
  checked_memcpy(shm.data_ptr(), pin.data_ptr(), shm_bytes, shm_bytes,
                 "write_shm memcpy(shm, pin)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("read_shm", &read_shm, "Read tensors from shared memory");
  m.def("write_shm", &write_shm, "Write tensors to shared memory");
}