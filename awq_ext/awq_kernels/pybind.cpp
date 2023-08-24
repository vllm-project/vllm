// adapted from llm-awq: https://github.com/mit-han-lab/llm-awq

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemm_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
}
