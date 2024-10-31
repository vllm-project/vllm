#include <arm_neon.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <memory>

#include "../../csrc/cpu/cpu_types_arm.hpp"

namespace py = pybind11;
using namespace vec_op;

PYBIND11_MODULE(cputypes, m) {
  /****************************************/
  /*               BF16Vec8               */
  /****************************************/
  py::module bf16vec8_mod = m.def_submodule("bf16vec8");
  py::class_<BF16Vec8>(bf16vec8_mod, "BF16Vec8")
    .def(py::init([](py::array_t<uint16_t> arr) {
      py::buffer_info buff = arr.request();
      return new BF16Vec8(static_cast<const void *>(buff.ptr));
    }))

    .def(py::init<const FP32Vec8 &>())

    .def("save", [](BF16Vec8 &self, py::array_t<uint16_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<void *>(buff.ptr));
    });

  /****************************************/
  /*               BF16Vec16              */
  /****************************************/
  py::module bf16vec16_mod = m.def_submodule("bf16vec16");
  py::class_<BF16Vec16>(bf16vec16_mod, "BF16Vec16")
    .def(py::init([](py::array_t<uint16_t> arr) {
      py::buffer_info buff = arr.request();
      return new BF16Vec16(static_cast<const void *>(buff.ptr));
    }))

    .def(py::init<const FP32Vec16 &>())

    .def("save", [](BF16Vec16 &self, py::array_t<uint16_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<void *>(buff.ptr));
    })

    .def("save_n", [](BF16Vec16 &self, py::array_t<uint16_t> arr, uint8_t elem_num) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<void *>(buff.ptr), elem_num);
    });

  /****************************************/
  /*               BF16Vec32              */
  /****************************************/
  py::module bf16vec32_mod = m.def_submodule("bf16vec32");
  py::class_<BF16Vec32>(bf16vec32_mod, "BF16Vec32")
    .def(py::init([](py::array_t<uint16_t> arr) {
      py::buffer_info buff = arr.request();
      return new BF16Vec32(static_cast<const void *>(buff.ptr));
    }))

    .def(py::init<const BF16Vec8 &>())

    .def("save", [](BF16Vec32 &self, py::array_t<uint16_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<void *>(buff.ptr));
    });

  /****************************************/
  /*               FP32Vec4               */
  /****************************************/
  py::module fp32vec4_mod = m.def_submodule("fp32vec4");
  py::class_<FP32Vec4>(fp32vec4_mod, "FP32Vec4")
    .def(py::init([](py::array_t<float32_t> arr) {
      py::buffer_info buff = arr.request();
      return new FP32Vec4(static_cast<const float32_t *>(buff.ptr));
    }))

    .def(py::init<const FP32Vec4 &>())

    .def(py::init<const float32_t &>())

    .def(py::init<>())

    .def("save", [](FP32Vec4 &self, py::array_t<float32_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<void *>(buff.ptr));
    });

  /****************************************/
  /*               FP32Vec8               */
  /****************************************/
  py::module fp32vec8_mod = m.def_submodule("fp32vec8");
  py::class_<FP32Vec8>(fp32vec8_mod, "FP32Vec8")
    .def(py::init([](py::array_t<float32_t> arr) {
      py::buffer_info buff = arr.request();
      return new FP32Vec8(static_cast<const float32_t *>(buff.ptr));
    }))

    .def(py::init<const float32_t &>())

    .def(py::init<>())

    .def(py::init<const FP32Vec4 &, const FP32Vec4 &>())

    .def(py::init<const FP32Vec8 &>())

    .def(py::init<const BF16Vec8 &>())

    .def("reduce_sum", &FP32Vec8::reduce_sum)

    .def("exp", &FP32Vec8::exp)

    .def("tanh", &FP32Vec8::tanh)

    .def("er", &FP32Vec8::er)

    .def("__mul__", [](const FP32Vec8 &a, const FP32Vec8 &b) { return a * b; })

    .def("__add__", [](const FP32Vec8 &a, const FP32Vec8 &b) { return a + b; })

    .def("__sub__", [](const FP32Vec8 &a, const FP32Vec8 &b) { return a - b; })

    .def("__truediv__", [](const FP32Vec8 &a, const FP32Vec8 &b) { return a / b; })

    .def("save", [](FP32Vec8 &self, py::array_t<float32_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<float32_t *>(buff.ptr));
    });

  /****************************************/
  /*               INT32Vec16             */
  /****************************************/
  py::module int32vec16_mod = m.def_submodule("int32vec16");
  py::class_<INT32Vec16>(int32vec16_mod, "INT32Vec16")
    .def(py::init([](py::array_t<int32_t> arr) {
      py::buffer_info buff = arr.request();
      return new INT32Vec16(static_cast<const void *>(buff.ptr));
    }))

    .def("save", [](INT32Vec16 &self, py::array_t<int32_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<int32_t *>(buff.ptr));
    })

    .def("save_n", [](INT32Vec16 &self, py::array_t<int32_t> arr, int8_t elem_num) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<int32_t *>(buff.ptr), elem_num);
    });

  /****************************************/
  /*               FP32Vec16              */
  /****************************************/
  py::module fp32vec16_mod = m.def_submodule("fp32vec16");
  py::class_<FP32Vec16>(fp32vec16_mod, "FP32Vec16")
    .def(py::init([](py::array_t<float32_t> arr) {
      py::buffer_info buff = arr.request();
      return new FP32Vec16(static_cast<const float32_t *>(buff.ptr));
    }))

    .def(py::init<const float32_t &>())

    .def(py::init<>())

    .def(py::init<const FP32Vec4 &, const FP32Vec4 &,
                  const FP32Vec4 &, const FP32Vec4 &>())

    .def(py::init<const FP32Vec16 &>())

    .def(py::init<const FP32Vec8 &>())

    .def(py::init<const FP32Vec4 &>())

    .def("reduce_sum", &FP32Vec16::reduce_sum)

    .def("__mul__", [](const FP32Vec16 &a, const FP32Vec16 &b) { return a * b; })

    .def("__add__", [](const FP32Vec16 &a, const FP32Vec16 &b) { return a + b; })

    .def("__sub__", [](const FP32Vec16 &a, const FP32Vec16 &b) { return a - b; })

    .def("__truediv__", [](const FP32Vec16 &a, const FP32Vec16 &b) { return a / b; })

    .def("save", [](FP32Vec16 &self, py::array_t<float32_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<float32_t *>(buff.ptr));
    });

  /****************************************/
  /*               INT8Vec16              */
  /****************************************/
  py::module int8vec16_mod = m.def_submodule("int8vec16");
  py::class_<INT8Vec16>(int8vec16_mod, "INT8Vec16")
    .def(py::init([](py::array_t<int8_t> arr) {
      py::buffer_info buff = arr.request();
      return new INT8Vec16(static_cast<const int8_t *>(buff.ptr));
    }))

    .def(py::init<const FP32Vec16 &>())

    .def("save", [](INT8Vec16 &self, py::array_t<int8_t> arr) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<int8_t *>(buff.ptr));
    })

    .def("save_n", [](INT8Vec16 &self, py::array_t<int8_t> arr, int8_t elem_num) {
      py::buffer_info buff = arr.request();
      self.save(static_cast<int8_t *>(buff.ptr), elem_num);
    });
}
