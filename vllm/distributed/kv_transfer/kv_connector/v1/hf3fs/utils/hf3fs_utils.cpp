#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstring>
#include <vector>

void read_shm(const torch::Tensor &shm, const torch::Tensor &pin,
              std::vector<torch::Tensor> dst, uint64_t stream_ptr) {
    py::gil_scoped_release release;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Copy from shared memory to pinned memory
    char *shm_ptr = static_cast<char *>(shm.data_ptr());
    char *src_ptr = static_cast<char *>(pin.data_ptr());
    std::memcpy(src_ptr, shm_ptr, shm.numel() * shm.element_size());

    // Copy from pinned memory to GPU tensors
    size_t current = 0;
    for (size_t i = 0; i < dst.size(); ++i) {
        auto &t = dst[i];
        size_t t_bytes = t.numel() * t.element_size();
        char *dst_ptr = static_cast<char *>(t.data_ptr());
        cudaMemcpyAsync(dst_ptr, src_ptr + current, t_bytes, cudaMemcpyHostToDevice, stream);
        current += t_bytes;
    }
    cudaStreamSynchronize(stream);
}

void write_shm(const std::vector<torch::Tensor> src, torch::Tensor &shm,
               const torch::Tensor &pin, uint64_t stream_ptr) {
    py::gil_scoped_release release;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Copy from GPU tensors to pinned memory
    char *dst_ptr = static_cast<char *>(pin.data_ptr());
    size_t current = 0;
    for (size_t i = 0; i < src.size(); ++i) {
        auto &t = src[i];
        size_t t_bytes = t.numel() * t.element_size();
        char *src_ptr = static_cast<char *>(t.data_ptr());
        cudaMemcpyAsync(dst_ptr + current, src_ptr, t_bytes, cudaMemcpyDeviceToHost, stream);
        current += t_bytes;
    }
    cudaStreamSynchronize(stream);

    // Copy from pinned memory to shared memory
    char *shm_ptr = static_cast<char *>(shm.data_ptr());
    std::memcpy(shm_ptr, dst_ptr, shm.numel() * shm.element_size());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_shm", &read_shm, "Read tensors from shared memory");
    m.def("write_shm", &write_shm, "Write tensors to shared memory");
}