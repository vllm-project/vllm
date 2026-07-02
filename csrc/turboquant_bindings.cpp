/*
 * TurboQuant PyTorch Bindings
 * 
 * C++ interface for PyTorch's ops registration system, providing access
 * to CUDA kernels with proper error handling and type checking.
 */

#include <torch/extension.h>

namespace turboquant {
namespace cuda {

// Forward declarations of CUDA kernel wrapper functions
at::Tensor pack_lowbit_cuda(at::Tensor values, int bits);
at::Tensor unpack_lowbit_cuda(at::Tensor packed, int bits, int length);

}  // namespace cuda
}  // namespace turboquant


/*
 * PyTorch dispatching functions
 */

at::Tensor pack_lowbit(at::Tensor values, int bits) {
    TORCH_CHECK(values.numel() > 0, "values tensor must not be empty");
    TORCH_CHECK(bits > 0, "bits must be positive");
    TORCH_CHECK(bits <= 32, "bits must be <= 32");
    
    // Convert to uint32 for bit operations
    auto values_uint = values.to(at::kInt);
    
    if (values_uint.is_cuda()) {
        return turboquant::cuda::pack_lowbit_cuda(values_uint, bits);
    } else {
        TORCH_CHECK(false, "pack_lowbit only supports CUDA tensors currently");
    }
}


at::Tensor unpack_lowbit(at::Tensor packed, int bits, int length) {
    TORCH_CHECK(packed.numel() > 0, "packed tensor must not be empty");
    TORCH_CHECK(bits > 0, "bits must be positive");
    TORCH_CHECK(bits <= 32, "bits must be <= 32");
    TORCH_CHECK(length > 0, "length must be positive");
    
    // Ensure correct type
    auto packed_uint = packed.to(at::kInt);
    
    if (packed_uint.is_cuda()) {
        return turboquant::cuda::unpack_lowbit_cuda(packed_uint, bits, length);
    } else {
        TORCH_CHECK(false, "unpack_lowbit only supports CUDA tensors currently");
    }
}


/*
 * Module initialization - register ops with PyTorch
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_lowbit", &pack_lowbit, 
          "Pack low-bit integers into 32-bit words (CUDA)");
    m.def("unpack_lowbit", &unpack_lowbit,
          "Unpack low-bit integers from 32-bit words (CUDA)");
}
