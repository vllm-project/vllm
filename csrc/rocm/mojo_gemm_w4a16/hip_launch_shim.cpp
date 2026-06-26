#include <torch/extension.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <dlfcn.h>

namespace {

using CreateFn = void* (*)(std::intptr_t);
using LaunchFn = void (*)(void*, std::intptr_t, std::intptr_t, std::intptr_t,
                          std::intptr_t, std::intptr_t, std::intptr_t,
                          std::intptr_t, std::intptr_t, std::intptr_t,
                          std::intptr_t, std::intptr_t);

struct MojoSymbols {
  void* handle = nullptr;
  CreateFn create = nullptr;
  LaunchFn launch = nullptr;
};

std::mutex symbols_mutex;
std::unordered_map<std::string, MojoSymbols> symbols_cache;

MojoSymbols& symbols_for(const std::string& so_path) {
  std::lock_guard<std::mutex> guard(symbols_mutex);
  auto it = symbols_cache.find(so_path);
  if (it != symbols_cache.end()) {
    return it->second;
  }

  void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    throw std::runtime_error(std::string("dlopen failed for ") + so_path +
                             ": " + dlerror());
  }

  auto create =
      reinterpret_cast<CreateFn>(dlsym(handle, "mojo_w4a16_runner_create"));
  auto launch =
      reinterpret_cast<LaunchFn>(dlsym(handle, "mojo_w4a16_runner_launch"));
  if (create == nullptr || launch == nullptr) {
    std::string err = dlerror();
    dlclose(handle);
    throw std::runtime_error("missing Mojo W4A16 native symbols in " + so_path +
                             (err.empty() ? "" : ": " + err));
  }

  auto [inserted, _] =
      symbols_cache.emplace(so_path, MojoSymbols{handle, create, launch});
  return inserted->second;
}

std::intptr_t data_addr(const torch::Tensor& tensor) {
  return reinterpret_cast<std::intptr_t>(tensor.data_ptr());
}

void check_common(const torch::Tensor& out, const torch::Tensor& a,
                  const torch::Tensor& qweight,
                  const torch::Tensor& qweight_kpacked,
                  const torch::Tensor& qzeros, const torch::Tensor& scales,
                  const torch::Tensor& partial) {
  TORCH_CHECK(out.is_cuda() && a.is_cuda() && qweight.is_cuda() &&
                  qweight_kpacked.is_cuda() && qzeros.is_cuda() &&
                  scales.is_cuda() && partial.is_cuda(),
              "Mojo W4A16 HIP shim expects CUDA/ROCm tensors");
  TORCH_CHECK(out.is_contiguous() && a.is_contiguous() &&
                  qweight.is_contiguous() && qweight_kpacked.is_contiguous() &&
                  qzeros.is_contiguous() && scales.is_contiguous() &&
                  partial.is_contiguous(),
              "Mojo W4A16 HIP shim expects contiguous tensors");
}

}  // namespace

std::uintptr_t create_runner(const std::string& so_path,
                             std::uintptr_t stream_addr) {
  auto& symbols = symbols_for(so_path);
  void* runner = symbols.create(static_cast<std::intptr_t>(stream_addr));
  TORCH_CHECK(runner != nullptr, "Mojo W4A16 native runner creation failed");
  return reinterpret_cast<std::uintptr_t>(runner);
}

void launch(const std::string& so_path, std::uintptr_t runner_addr,
            std::uintptr_t stream_addr, const torch::Tensor& out,
            const torch::Tensor& a, const torch::Tensor& qweight,
            const torch::Tensor& qweight_kpacked, const torch::Tensor& qzeros,
            const torch::Tensor& scales, const torch::Tensor& partial) {
  TORCH_CHECK(runner_addr != 0, "Mojo W4A16 native runner is null");
  check_common(out, a, qweight, qweight_kpacked, qzeros, scales, partial);
  TORCH_CHECK(a.dim() == 2 && qweight.dim() == 2 && scales.dim() == 2,
              "Mojo W4A16 HIP shim expects rank-2 tensors");

  auto& symbols = symbols_for(so_path);
  const auto m = static_cast<std::intptr_t>(a.size(0));
  const auto k = static_cast<std::intptr_t>(a.size(1));
  const auto n = static_cast<std::intptr_t>(scales.size(1));

  symbols.launch(reinterpret_cast<void*>(runner_addr),
                 static_cast<std::intptr_t>(stream_addr), data_addr(out),
                 data_addr(a), data_addr(qweight), data_addr(qweight_kpacked),
                 data_addr(qzeros), data_addr(scales), data_addr(partial), m, n,
                 k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("create_runner", &create_runner);
  m.def("launch", &launch);
}
