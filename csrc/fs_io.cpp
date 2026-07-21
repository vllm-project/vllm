// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <sys/stat.h>

#include <vector>

extern "C" {

static void _batch_lookup(const std::vector<const char*>& paths,
                          std::vector<long long>& sizes) {
  struct stat st;
  for (size_t i = 0; i < paths.size(); i++) {
    // -1 on any error (missing, permission, race-unlink): the caller
    // compares against the expected block size, so a negative sentinel
    // is an unconditional miss and never a spurious hit.
    sizes[i] =
        (stat(paths[i], &st) == 0) ? static_cast<long long>(st.st_size) : -1;
  }
}

/// @brief Stat a batch of paths, returning each file's size.
/// @param paths list[str] – absolute paths to stat.
/// @return list[int] – st_size for each path, or -1 if the stat failed
///         (missing file, permission error, or a concurrent unlink).
/// @note Releases the GIL for the entire batch. Size via stat(2); the caller
///       validates each size against the expected block size, so lookup
///       agrees with what load will accept for roughly one stat per file.
static PyObject* batch_lookup(PyObject* /*self*/, PyObject* args) {
  PyObject* path_list;
  if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &path_list)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(path_list);
  std::vector<const char*> paths(n);
  for (Py_ssize_t i = 0; i < n; i++) {
    paths[i] = PyUnicode_AsUTF8AndSize(PyList_GetItem(path_list, i), nullptr);
    if (paths[i] == nullptr) {
      return nullptr;
    }
  }

  std::vector<long long> sizes(n);
  {
    Py_BEGIN_ALLOW_THREADS _batch_lookup(paths, sizes);
    Py_END_ALLOW_THREADS
  }

  PyObject* result = PyList_New(n);
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    PyObject* item = PyLong_FromLongLong(sizes[i]);
    if (item == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SetItem(result, i, item);
  }
  return result;
}

static PyMethodDef fs_io_C_methods[] = {
    {"batch_lookup", batch_lookup, METH_VARARGS,
     "batch_lookup(paths: list[str]) -> list[int]\n"
     "\n"
     "Return each path's size, or -1 if it could not be stat'd."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef fs_io_C_module = {
    PyModuleDef_HEAD_INIT, "fs_io_C", "Filesystem helpers for KV offload", -1,
    fs_io_C_methods,
};

PyMODINIT_FUNC PyInit_fs_io_C(void) { return PyModule_Create(&fs_io_C_module); }

}  // extern "C"
