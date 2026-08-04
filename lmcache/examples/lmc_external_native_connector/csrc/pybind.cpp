// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include "connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "Example native connector plugin for LMCache";

  py::class_<example_connector::ExampleFSConnector>(m, "ExampleFSConnector")
      .def(py::init<std::string, int>(), py::arg("base_path"),
           py::arg("num_workers"))
          LMCACHE_BIND_CONNECTOR_METHODS(example_connector::ExampleFSConnector);

  py::class_<example_connector::ExampleMemoryConnector>(
      m, "ExampleMemoryConnector")
      .def(py::init<int>(), py::arg("num_workers"))
          LMCACHE_BIND_CONNECTOR_METHODS(
              example_connector::ExampleMemoryConnector);
}
