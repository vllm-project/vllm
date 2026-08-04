// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_aerospike, m) {
  m.doc() = "Native Aerospike connector for LMCache";

  py::class_<lmcache::connector::AerospikeNativeConnector>(
      m, "LMCacheAerospikeClient")
      .def(py::init<std::string, std::string, std::string, int, uint32_t,
                    uint32_t, uint32_t, size_t, size_t, std::string,
                    std::string>(),
           py::arg("hosts"), py::arg("namespace"), py::arg("set_name"),
           py::arg("num_workers"), py::arg("read_timeout_ms") = 1000,
           py::arg("write_timeout_ms") = 2000,
           py::arg("default_ttl_seconds") = 86400,
           py::arg("target_segment_bytes") = 0, py::arg("max_record_bytes") = 0,
           py::arg("username") = "", py::arg("password") = "")
          LMCACHE_BIND_CONNECTOR_METHODS(
              lmcache::connector::AerospikeNativeConnector);
}
