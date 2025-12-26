#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "vmm_pool_client.h"

namespace py = pybind11;

static at::Tensor tensor_from_device_ptr(uint64_t ptr, int64_t nbytes, int device) {
  void* p = reinterpret_cast<void*>(ptr);
  auto options = at::device(at::kCUDA).dtype(at::kByte).device_index(device);
  auto deleter = [](void*){};
  return at::from_blob(p, {nbytes}, deleter, options);
}

PYBIND11_MODULE(vmm_pool_py, m) {
  py::class_<VmmPoolClient>(m, "VmmPoolClient")
    .def(py::init<const std::string&, int>())
    .def("allocate", &VmmPoolClient::Allocate)
    .def("free", &VmmPoolClient::Free)
    .def("stats", &VmmPoolClient::Stats)
    .def("map", &VmmPoolClient::Map)
    .def("unmap", &VmmPoolClient::Unmap);
  m.def("tensor_from_device_ptr", &tensor_from_device_ptr,
        py::arg("ptr"), py::arg("nbytes"), py::arg("device"));
}
