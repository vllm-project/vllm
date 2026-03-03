/**
 * axpby_binding.cpp
 *
 * PyTorch tensor wrapper for the axpby CUDA kernel.
 * Registered automatically via REGISTER_BINDING — do not add it to binding.cpp.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Forward declaration of the CUDA launcher in axpby.cu
extern void axpby_launcher(const float* a,
                            const float* b,
                            float*       out,
                            float        alpha,
                            int          n,
                            int          config,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// PyTorch wrapper
// ---------------------------------------------------------------------------

/**
 * axpby_forward(a, b, alpha, config=0) -> Tensor
 *
 * Computes out = alpha * a + b using a custom CUDA kernel.
 *
 * Args:
 *   a      : float32 CUDA tensor (contiguous)
 *   b      : float32 CUDA tensor (contiguous, same shape as a)
 *   alpha  : scalar multiplier
 *   config : thread-count selector (0→256, 1→128, 2→512)
 *
 * Returns:
 *   out    : float32 CUDA tensor with the same shape as a
 */
torch::Tensor axpby_forward(torch::Tensor a,
                             torch::Tensor b,
                             float         alpha,
                             int           config = 0) {
    TORCH_CHECK(a.is_cuda(),        "Tensor 'a' must be on a CUDA device");
    TORCH_CHECK(b.is_cuda(),        "Tensor 'b' must be on a CUDA device");
    TORCH_CHECK(a.is_contiguous(),  "Tensor 'a' must be contiguous");
    TORCH_CHECK(b.is_contiguous(),  "Tensor 'b' must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor 'a' must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor 'b' must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(),
                "Tensors 'a' and 'b' must have matching shapes");

    auto out = torch::empty_like(a);

    axpby_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        static_cast<int>(a.numel()),
        config,
        at::cuda::getCurrentCUDAStream()
    );

    return out;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

REGISTER_BINDING(axpby_forward,
    [](pybind11::module& m) {
        m.def("axpby_forward",
              &axpby_forward,
              pybind11::arg("a"),
              pybind11::arg("b"),
              pybind11::arg("alpha"),
              pybind11::arg("config") = 0,
              "Compute alpha * a + b via a custom CUDA kernel.");
    }
);
