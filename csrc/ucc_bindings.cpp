// Python bindings for PyTorch's ProcessGroupUCC (vllm._ucc_C), whose sources
// are fetched by cmake/external_projects/torch_ucc.cmake.
//
// Mirrors the ProcessGroupUCC binding in PyTorch's
// torch/csrc/distributed/c10d/init.cpp, so the backend can be registered with
// torch.distributed.Backend.register_backend() by a torch that was built
// without UCC (e.g. the PyPI wheels).

#include <chrono>

#include <pybind11/chrono.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/utils/pybind.h>

#include <torch/csrc/distributed/c10d/ProcessGroupUCC.hpp>

namespace py = pybind11;

namespace {

// Same holder as torch uses for c10d::Backend and its subclasses
// (IntrusivePtrNoGilDestructor in torch/csrc/distributed/c10d/init.cpp): the
// GIL is released while the backend is destroyed, because the destructor joins
// the UCC progress thread.
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(
      IntrusivePtrNoGilDestructor&&) noexcept = default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // See https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        py::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept { return *impl_; }
  T* operator->() const noexcept { return impl_.get(); }
  [[nodiscard]] T* get() const noexcept { return impl_.get(); }
  void reset() noexcept { impl_.reset(); }
  operator bool() const noexcept { return impl_; }
};

}  // namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true)

PYBIND11_MODULE(_ucc_C, m) {
  // c10d::Backend and c10d::Store must already be registered with pybind.
  py::module_::import("torch._C._distributed_c10d");

  using c10d::ProcessGroupUCC;
  py::class_<ProcessGroupUCC, c10d::Backend,
             IntrusivePtrNoGilDestructor<ProcessGroupUCC>>(m, "ProcessGroupUCC")
      .def(py::init([](const c10::intrusive_ptr<c10d::Store>& store, int rank,
                       int size, const std::chrono::milliseconds& timeout) {
             // gil_scoped_release is not safe as a call_guard in init.
             // https://github.com/pybind/pybind11/issues/5473
             py::gil_scoped_release nogil{};
             return c10::make_intrusive<ProcessGroupUCC>(store, rank, size,
                                                         timeout);
           }),
           py::arg("store"), py::arg("rank"), py::arg("size"),
           py::arg("timeout"));
}
