// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// pybind11 bindings for the CUDA compression module.

#include <torch/extension.h>

#include "kvs_compress.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // All entry points are pure C++/CUDA and must not hold the GIL: a single
  // encode launches hundreds of kernels, and holding the GIL for that long
  // stalls the engine main thread of this TP rank (and with it every other
  // rank's NCCL all-reduce).
  m.def("kvs_ping", &kvs_ping, "Build integration smoke test",
        py::call_guard<py::gil_scoped_release>());
  m.def("kvs_encode_flat", &kvs_encode_flat,
        "Encode one chunk from a flat metadata blob",
        py::call_guard<py::gil_scoped_release>());
  m.def("kvs_decode_flat", &kvs_decode_flat,
        "Decode one chunk from a flat metadata blob",
        py::call_guard<py::gil_scoped_release>());
  m.def("kvs_test_gather_fwht", &kvs_test_gather_fwht,
        "Run gather (+sign) and FWHT for a single part (validation)",
        py::call_guard<py::gil_scoped_release>());
}
