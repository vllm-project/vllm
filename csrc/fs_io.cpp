// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

#if defined(O_DIRECT)
constexpr int kODirectFlag = O_DIRECT;
#else
constexpr int kODirectFlag = 0;
#endif

extern "C" {

namespace {

struct IOResult {
  std::string error_message;
  int errnum = 0;

  void clear() {
    error_message.clear();
    errnum = 0;
  }

  bool has_error() const { return !error_message.empty() || errnum != 0; }

  void maybe_set_errnum() {
    if (!has_error()) {
      errnum = errno;
    }
  }

  void maybe_set_error_message(std::string msg) {
    if (!has_error()) {
      error_message = std::move(msg);
    }
  }

  PyObject* to_py_error(const char* path_for_errno) const {
    if (!error_message.empty()) {
      return PyErr_Format(PyExc_IOError, "%s", error_message.c_str());
    }
    if (errnum != 0) {
      errno = errnum;
      return PyErr_SetFromErrnoWithFilename(PyExc_OSError, path_for_errno);
    }
    return nullptr;
  }
};

inline bool ensure_parent_dirs(const std::string& path, IOResult* io_result) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return true;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  if (ec) {
    io_result->maybe_set_error_message("Failed to create parent dirs for " +
                                       path + ": " + ec.message());
    return false;
  }
  return true;
}

inline int safe_open(const char* path, int flags, mode_t mode,
                     IOResult* result) {
  const int fd = open(path, flags, mode);
  if (fd < 0) {
    result->maybe_set_errnum();
  }
  return fd;
}

inline ssize_t safe_write(int fd, const char* src, size_t size,
                          IOResult* result) {
  const ssize_t written = write(fd, src, size);
  if (written < 0) {
    result->maybe_set_errnum();
  } else if (static_cast<size_t>(written) != size) {
    result->maybe_set_error_message("Short write: expected " +
                                    std::to_string(size) + " bytes, wrote " +
                                    std::to_string(written));
  }
  return written;
}

inline ssize_t safe_read(int fd, char* dst, size_t size, IOResult* result) {
  const ssize_t bytes_read = read(fd, dst, size);
  if (bytes_read < 0) {
    result->maybe_set_errnum();
  } else if (static_cast<size_t>(bytes_read) != size) {
    result->maybe_set_error_message("Short read: expected " +
                                    std::to_string(size) + " bytes, read " +
                                    std::to_string(bytes_read));
  }
  return bytes_read;
}

inline bool safe_close(int fd, IOResult* result) {
  if (close(fd) != 0) {
    result->maybe_set_errnum();
    return false;
  }
  return true;
}

inline bool safe_rename(const char* src, const char* dst, IOResult* result) {
  if (rename(src, dst) != 0) {
    result->maybe_set_errnum();
    return false;
  }
  return true;
}

inline void safe_unlink(const char* path, IOResult* result) {
  if (unlink(path) != 0) {
    result->maybe_set_errnum();
  }
}

// Core single-block store: src/size are raw pointer + byte count.
inline bool _store_block(const char* tmp_path, const char* dest_path,
                         const char* src, size_t size, IOResult* result) {
  result->clear();

  if (access(dest_path, F_OK) == 0) {
    // Already present
    return true;
  }

  if (!ensure_parent_dirs(dest_path, result)) {
    return false;
  }

  const int fd =
      safe_open(tmp_path, O_CREAT | O_EXCL | O_WRONLY | O_TRUNC | kODirectFlag,
                0644, result);
  if (fd < 0) {
    return false;
  }

  safe_write(fd, src, size, result);
  safe_close(fd, result);

  if (!result->has_error()) {
    safe_rename(tmp_path, dest_path, result);
  }

  if (result->has_error()) {
    safe_unlink(tmp_path, result);
    return false;
  }

  return true;
}

// Core single-block load: dst/size are raw pointer + byte count.
inline bool _load_block(const char* source_path, char* dst, size_t size,
                        IOResult* result) {
  result->clear();

  int fd = safe_open(source_path, O_RDONLY | kODirectFlag, 0, result);
  if (fd < 0) {
    unlink(source_path);
    return false;
  }

  safe_read(fd, dst, size, result);
  safe_close(fd, result);

  if (result->has_error()) {
    unlink(source_path);
    return false;
  }

  return true;
}

inline void _batch_lookup(const std::vector<const char*>& paths,
                          std::vector<int>& exists_flags) {
  for (size_t i = 0; i < paths.size(); i++) {
    exists_flags[i] = (access(paths[i], F_OK) == 0) ? 1 : 0;
  }
}

// Helper: extract a list[str] of length n into a vector<const char*>.
// Returns false and sets a Python exception on error.
inline bool extract_str_list(PyObject* list, Py_ssize_t n,
                             std::vector<const char*>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    out[i] = PyUnicode_AsUTF8AndSize(PyList_GetItem(list, i), nullptr);
    if (out[i] == nullptr) {
      return false;
    }
  }
  return true;
}

// Helper: extract a Py_buffer per element of a list[bytes-like] of length n.
// On success, `out` holds n acquired buffers (caller must PyBuffer_Release
// each). On failure, any buffers already acquired are released before
// returning false, and a Python exception is set.
inline bool extract_buffer_list(PyObject* list, Py_ssize_t n, int flags,
                                std::vector<Py_buffer>& out) {
  for (Py_ssize_t i = 0; i < n; i++) {
    if (PyObject_GetBuffer(PyList_GetItem(list, i), &out[i], flags) != 0) {
      for (Py_ssize_t j = 0; j < i; j++) {
        PyBuffer_Release(&out[j]);
      }
      return false;
    }
  }
  return true;
}

inline void release_buffer_list(std::vector<Py_buffer>& buffers) {
  for (auto& buf : buffers) {
    PyBuffer_Release(&buf);
  }
}

}  // namespace

/// @brief Check file existence for a batch of paths.
/// @param paths list[str] – absolute paths to check.
/// @return list[bool] – True if the corresponding path exists, False otherwise.
/// @note Releases the GIL for the entire batch. File existence via access(2).
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

  std::vector<int> exists_flags(n);
  {
    Py_BEGIN_ALLOW_THREADS _batch_lookup(paths, exists_flags);
    Py_END_ALLOW_THREADS
  }

  PyObject* result = PyList_New(n);
  if (result == nullptr) {
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < n; i++) {
    PyList_SetItem(result, i, PyBool_FromLong(exists_flags[i]));
  }
  return result;
}

/// @brief Store a batch of blocks, each from its own buffer, to disk.
/// @param tmp_paths  list[str] – one temp path per block.
/// @param dest_paths list[str] – one destination path per block.
/// @param buffers    list[bytes-like] – one source buffer per block.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_store_block(PyObject* /*self*/, PyObject* args) {
  PyObject* tmp_paths_obj = nullptr;
  PyObject* dest_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;

  if (!PyArg_ParseTuple(args, "O!O!O!", &PyList_Type, &tmp_paths_obj,
                        &PyList_Type, &dest_paths_obj, &PyList_Type,
                        &buffers_obj)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(tmp_paths_obj);
  if (PyList_Size(dest_paths_obj) != n || PyList_Size(buffers_obj) != n) {
    PyErr_SetString(
        PyExc_ValueError,
        "tmp_paths, dest_paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> tmp_paths(n);
  std::vector<const char*> dest_paths(n);

  if (!extract_str_list(tmp_paths_obj, n, tmp_paths)) return nullptr;
  if (!extract_str_list(dest_paths_obj, n, dest_paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_SIMPLE, buffers)) {
    return nullptr;
  }

  IOResult io_result;
  const char* failed_path = nullptr;

  {
    Py_BEGIN_ALLOW_THREADS for (size_t i = 0; i < static_cast<size_t>(n); i++) {
      const char* buf = static_cast<const char*>(buffers[i].buf);
      if (!_store_block(tmp_paths[i], dest_paths[i], buf,
                        static_cast<size_t>(buffers[i].len), &io_result)) {
        failed_path = dest_paths[i];
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_path) {
    return io_result.to_py_error(failed_path);
  }

  Py_RETURN_NONE;
}

/// @brief Load a batch of blocks from disk, each into its own buffer.
/// @param source_paths list[str] – one source path per block.
/// @param buffers      list[writable bytes-like] – one destination buffer
///                     per block.
/// @note Releases the GIL for the entire batch. Raises on first error.
static PyObject* batch_load_block(PyObject* /*self*/, PyObject* args) {
  PyObject* source_paths_obj = nullptr;
  PyObject* buffers_obj = nullptr;

  if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &source_paths_obj,
                        &PyList_Type, &buffers_obj)) {
    return nullptr;
  }

  const Py_ssize_t n = PyList_Size(source_paths_obj);
  if (PyList_Size(buffers_obj) != n) {
    PyErr_SetString(PyExc_ValueError,
                    "source_paths and buffers must have the same length");
    return nullptr;
  }

  std::vector<const char*> source_paths(n);
  if (!extract_str_list(source_paths_obj, n, source_paths)) return nullptr;

  std::vector<Py_buffer> buffers(n);
  if (!extract_buffer_list(buffers_obj, n, PyBUF_WRITABLE, buffers)) {
    return nullptr;
  }

  IOResult io_result;
  const char* failed_path = nullptr;

  {
    Py_BEGIN_ALLOW_THREADS for (size_t i = 0; i < static_cast<size_t>(n); i++) {
      char* buf = static_cast<char*>(buffers[i].buf);
      if (!_load_block(source_paths[i], buf,
                       static_cast<size_t>(buffers[i].len), &io_result)) {
        failed_path = source_paths[i];
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_path) {
    return io_result.to_py_error(failed_path);
  }

  Py_RETURN_NONE;
}

static PyMethodDef fs_io_C_methods[] = {
    {"batch_lookup", batch_lookup, METH_VARARGS,
     "batch_lookup(paths: list[str]) -> list[bool]\n"
     "\n"
     "Check file existence for a batch of paths."},
    {"batch_store_block", batch_store_block, METH_VARARGS,
     "batch_store_block(tmp_paths: list[str], dest_paths: list[str],\n"
     "                  buffers: list[bytes-like]) -> None\n"
     "\n"
     "Store a batch of blocks, each from its own buffer, to disk. Raises on "
     "first error."},
    {"batch_load_block", batch_load_block, METH_VARARGS,
     "batch_load_block(source_paths: list[str],\n"
     "                 buffers: list[writable bytes-like]) -> None\n"
     "\n"
     "Load a batch of blocks from disk into corresponding buffers. "
     "Raises on first error."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef fs_io_C_module = {
    PyModuleDef_HEAD_INIT, "fs_io_C", "Filesystem helpers for KV offload", -1,
    fs_io_C_methods,
};

PyMODINIT_FUNC PyInit_fs_io_C(void) { return PyModule_Create(&fs_io_C_module); }

}  // extern "C"
