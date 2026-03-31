"""JIT-compile cache_kernels.cu with kernel-side NaN/Inf detection.

Compiles only cache_kernels.cu as a standalone extension via nvcc,
avoiding a full cmake rebuild. Exposes concat_and_cache_mla with the
nan_flag parameter.

Usage (from /opt/vllm-source):
    python tools/jit_cache_nan.py
    sed -i '1s/^/import cache_nan_ext\n/' vllm/_custom_ops.py
    sed -i 's/torch\.ops\._C_cache_ops\.concat_and_cache_mla/cache_nan_ext.concat_and_cache_mla/g' vllm/_custom_ops.py
"""
import os
import textwrap

from torch.utils.cpp_extension import load

SRC_DIR = "/opt/vllm-source/csrc"
BUILD_DIR = "/tmp/jit_build/cache_nan"
os.makedirs(BUILD_DIR, exist_ok=True)

# Normalize arch list — torch JIT doesn't understand "10.0f" (cmake-only syntax)
arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
if arch:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"

BINDING_SRC = textwrap.dedent("""\
    #include <torch/extension.h>
    #include <c10/util/Optional.h>

    void concat_and_cache_mla(torch::Tensor& kv_c, torch::Tensor& k_pe,
                              torch::Tensor& kv_cache, torch::Tensor& slot_mapping,
                              const std::string& kv_cache_dtype, torch::Tensor& scale,
                              std::optional<torch::Tensor> nan_flag);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("concat_and_cache_mla", &concat_and_cache_mla,
              "Concat and cache MLA with NaN/Inf detection",
              py::arg("kv_c"), py::arg("k_pe"), py::arg("kv_cache"),
              py::arg("slot_mapping"), py::arg("kv_cache_dtype"), py::arg("scale"),
              py::arg("nan_flag") = py::none());
    }
""")

binding_path = os.path.join(SRC_DIR, "cache_nan_binding.cpp")
with open(binding_path, "w") as f:
    f.write(BINDING_SRC)

mod = load(
    name="cache_nan_ext",
    sources=[
        binding_path,
        os.path.join(SRC_DIR, "cache_kernels.cu"),
    ],
    extra_include_paths=[SRC_DIR],
    build_directory=BUILD_DIR,
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    verbose=True,
)

print(f"cache_nan_ext built: {mod}")
