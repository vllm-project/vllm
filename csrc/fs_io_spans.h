// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <Python.h>

extern "C" {

// read_spans_init(threads: int = 16, queue_depth: int = 16,
//                 use_io_uring: bool = True) -> str
PyObject* read_spans_init(PyObject* self, PyObject* args);

// batch_read_spans(fds, offsets, length: int, out) -> None
PyObject* batch_read_spans(PyObject* self, PyObject* args);

}  // extern "C"
