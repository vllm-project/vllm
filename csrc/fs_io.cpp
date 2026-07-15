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

// Returns 0 on success, or the std::error_code's POSIX-compatible value on
// failure, mirroring the errno convention used by the syscalls below.
inline int ensure_parent_dirs(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) {
    return 0;
  }
  std::error_code ec;
  std::filesystem::create_directories(parent, ec);
  return ec ? ec.value() : 0;
}

// Core single-block store: src/size are raw pointer + byte count. Returns 0
// on success, or the errno of the failing step on failure -- captured
// before any subsequent cleanup call can overwrite it. On failure, the temp
// file is removed.
inline int _store_block(const char* tmp_path, const char* dest_path,
                        const char* src, size_t size) {
  if (access(dest_path, F_OK) == 0) {
    return 0;  // Already present.
  }

  if (const int err = ensure_parent_dirs(dest_path); err != 0) {
    return err;
  }

  const int fd = open(
      tmp_path, O_CREAT | O_EXCL | O_WRONLY | O_TRUNC | kODirectFlag, 0644);
  if (fd < 0) {
    return errno;
  }

  const ssize_t written = write(fd, src, size);
  if (written < 0 || static_cast<size_t>(written) != size) {
    const int err = written < 0 ? errno : EIO;
    close(fd);  // Best-effort cleanup; the real error is already captured.
    unlink(tmp_path);
    return err;
  }

  if (close(fd) != 0) {
    const int err = errno;
    unlink(tmp_path);
    return err;
  }

  if (rename(tmp_path, dest_path) != 0) {
    const int err = errno;
    unlink(tmp_path);
    return err;
  }

  return 0;
}

// Core single-block load: dst/size are raw pointer + byte count. Returns 0
// on success, or the errno of the failing step on failure. On failure, the
// source file is removed since a partially-read block should not be reused.
inline int _load_block(const char* source_path, char* dst, size_t size) {
  const int fd = open(source_path, O_RDONLY | kODirectFlag, 0);
  if (fd < 0) {
    const int err = errno;
    unlink(source_path);
    return err;
  }

  const ssize_t bytes_read = read(fd, dst, size);
  if (bytes_read < 0 || static_cast<size_t>(bytes_read) != size) {
    const int err = bytes_read < 0 ? errno : EIO;
    close(fd);
    unlink(source_path);
    return err;
  }

  if (close(fd) != 0) {
    const int err = errno;
    unlink(source_path);
    return err;
  }

  return 0;
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

  Py_ssize_t failed_index = -1;
  int failure_errno = 0;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      const char* buf = static_cast<const char*>(buffers[i].buf);
      const int err = _store_block(tmp_paths[i], dest_paths[i], buf,
                                   static_cast<size_t>(buffers[i].len));
      if (err != 0) {
        failed_index = i;
        failure_errno = err;
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_index >= 0) {
    // PyErr_SetFromErrnoWithFilename() reads the errno to format exception.
    errno = failure_errno;
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,
                                          dest_paths[failed_index]);
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

  Py_ssize_t failed_index = -1;
  int failure_errno = 0;

  {
    Py_BEGIN_ALLOW_THREADS for (Py_ssize_t i = 0; i < n; i++) {
      char* buf = static_cast<char*>(buffers[i].buf);
      const int err = _load_block(source_paths[i], buf,
                                  static_cast<size_t>(buffers[i].len));
      if (err != 0) {
        failed_index = i;
        failure_errno = err;
        break;
      }
    }
    Py_END_ALLOW_THREADS
  }

  release_buffer_list(buffers);

  if (failed_index >= 0) {
    // PyErr_SetFromErrnoWithFilename() reads the errno to format exception.
    errno = failure_errno;
    return PyErr_SetFromErrnoWithFilename(PyExc_OSError,
                                          source_paths[failed_index]);
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
