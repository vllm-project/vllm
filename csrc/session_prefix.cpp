// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <array>
#include <limits>
#include <new>
#include <vector>

namespace {
constexpr const char* kCapsule = "vllm.session_prefix.markers";
struct Marker {
  PyObject* text;
  long id;
  Py_ssize_t length;
};
struct Markers {
  std::vector<Marker> values;
  std::array<int, 4096> lookup;
  long first = std::numeric_limits<long>::max();
  long last = 0;
  ~Markers() {
    for (auto& marker : values) Py_DECREF(marker.text);
  }
};
void destroy_markers(PyObject* capsule) {
  delete static_cast<Markers*>(PyCapsule_GetPointer(capsule, kCapsule));
}
PyObject* make_markers(PyObject*, PyObject* mapping) {
  if (!PyDict_Check(mapping) || PyDict_Size(mapping) < 1 ||
      PyDict_Size(mapping) > 128) {
    PyErr_SetString(PyExc_ValueError, "expected 1..128 audited markers");
    return nullptr;
  }
  auto* state = new (std::nothrow) Markers;
  if (!state) return PyErr_NoMemory();
  state->lookup.fill(-1);
  try {
    state->values.reserve(PyDict_Size(mapping));
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(mapping, &pos, &key, &value)) {
      long id = PyLong_AsLong(key);
      if (PyErr_Occurred()) {
        delete state;
        return nullptr;
      }
      if (id < 0 || !PyUnicode_CheckExact(value) ||
          PyUnicode_GetLength(value) == 0 ||
          PyUnicode_ReadChar(value, 0) != '<') {
        delete state;
        PyErr_SetString(PyExc_ValueError, "invalid marker");
        return nullptr;
      }
      state->values.push_back(
          {Py_NewRef(value), id, PyUnicode_GetLength(value)});
      if (id < state->first) state->first = id;
      if (id > state->last) state->last = id;
    }
  } catch (const std::bad_alloc&) {
    delete state;
    return PyErr_NoMemory();
  }
  if (state->last - state->first >= 4096) {
    delete state;
    PyErr_SetString(PyExc_ValueError, "unsupported marker ID range");
    return nullptr;
  }
  for (size_t i = 0; i < state->values.size(); ++i)
    state->lookup[state->values[i].id - state->first] = static_cast<int>(i);
  PyObject* result = PyCapsule_New(state, kCapsule, destroy_markers);
  if (!result) delete state;
  return result;
}
PyObject* lcp(PyObject*, PyObject* args) {
  PyObject *a, *b;
  if (!PyArg_ParseTuple(args, "UU", &a, &b)) return nullptr;
  Py_ssize_t n = PyUnicode_GetLength(a), m = PyUnicode_GetLength(b), i = 0;
  if (n < 0 || m < 0) return nullptr;
  if (m < n) n = m;
  for (; i < n; ++i)
    if (PyUnicode_ReadChar(a, i) != PyUnicode_ReadChar(b, i)) break;
  return PyLong_FromSsize_t(i);
}
PyObject* token_lcp(PyObject*, PyObject* args) {
  PyObject *a, *b;
  Py_ssize_t n;
  if (!PyArg_ParseTuple(args, "OOn", &a, &b, &n)) return nullptr;
  if (!PyList_CheckExact(a) || !PyList_CheckExact(b) || n < 0 ||
      n > PyList_Size(a) || n > PyList_Size(b)) {
    PyErr_SetString(PyExc_ValueError,
                    "expected exact lists and bounded length");
    return nullptr;
  }
  Py_ssize_t i = 0;
  for (; i < n; ++i) {
    PyObject *x = PyList_GetItem(a, i), *y = PyList_GetItem(b, i);
    if (!PyLong_CheckExact(x) || !PyLong_CheckExact(y)) {
      PyErr_SetString(PyExc_ValueError, "expected integer token IDs");
      return nullptr;
    }
    unsigned long u = PyLong_AsUnsignedLong(x);
    if (PyErr_Occurred()) return nullptr;
    unsigned long v = PyLong_AsUnsignedLong(y);
    if (PyErr_Occurred()) return nullptr;
    if (u != v) break;
  }
  return PyLong_FromSsize_t(i);
}
PyObject* max_token_id(PyObject*, PyObject* ids) {
  if (!PyList_CheckExact(ids)) {
    PyErr_SetString(PyExc_ValueError, "expected an exact list");
    return nullptr;
  }
  unsigned long long maximum = 0;
  for (Py_ssize_t i = 0; i < PyList_Size(ids); ++i) {
    PyObject* item = PyList_GetItem(ids, i);
    if (!PyLong_CheckExact(item)) Py_RETURN_NONE;
    unsigned long long value = PyLong_AsUnsignedLongLong(item);
    if (PyErr_Occurred()) {
      PyErr_Clear();
      Py_RETURN_NONE;
    }
    if (value > 0xffffffffULL) Py_RETURN_NONE;
    if (value > maximum) maximum = value;
  }
  return PyLong_FromUnsignedLongLong(maximum);
}
int next_marker(const Markers& state, PyObject* text, Py_ssize_t n,
                Py_ssize_t& position, Py_ssize_t& end) {
  while (position < n) {
    auto found = PyUnicode_FindChar(text, '<', position, n, 1);
    if (found < 0) {
      position = n;
      return static_cast<int>(found);
    }
    for (size_t i = 0; i < state.values.size(); ++i) {
      const auto& marker = state.values[i];
      if (marker.length > n - found) continue;
      auto match = PyUnicode_Tailmatch(text, marker.text, found,
                                       found + marker.length, -1);
      if (match < 0) return -2;
      if (match) {
        end = found + marker.length;
        position = end;
        return static_cast<int>(i);
      }
    }
    position = found + 1;
  }
  return -1;
}
PyObject* boundaries(PyObject*, PyObject* args) {
  PyObject *capsule, *text, *ids;
  Py_ssize_t cb = 0, tb = 0;
  if (!PyArg_ParseTuple(args, "OUO!|nn", &capsule, &text, &PyList_Type, &ids,
                        &cb, &tb))
    return nullptr;
  auto* state = static_cast<Markers*>(PyCapsule_GetPointer(capsule, kCapsule));
  if (!state) return nullptr;
  Py_ssize_t length = PyUnicode_GetLength(text), n = PyList_Size(ids);
  if (cb < 0 || tb < 0 || cb > PY_SSIZE_T_MAX - length ||
      tb > PY_SSIZE_T_MAX - n) {
    PyErr_SetString(PyExc_ValueError, "invalid boundary base");
    return nullptr;
  }
  PyObject* result = PyList_New(0);
  if (!result) return nullptr;
  Py_ssize_t position = 0, end = 0;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PyList_GetItem(ids, i);
    if (!PyLong_CheckExact(item)) {
      PyErr_SetString(PyExc_ValueError, "expected integer token IDs");
      Py_DECREF(result);
      return nullptr;
    }
    long id = PyLong_AsLong(item);
    if (PyErr_Occurred()) {
      Py_DECREF(result);
      return nullptr;
    }
    if (id < state->first || id > state->last) continue;
    int marker = state->lookup[id - state->first];
    if (marker < 0) continue;
    int source = next_marker(*state, text, length, position, end);
    if (source == -2) {
      Py_DECREF(result);
      return nullptr;
    }
    if (source != marker) goto mismatch;
    {
      PyObject* pair = Py_BuildValue("nn", cb + end, tb + i + 1);
      if (!pair) {
        Py_DECREF(result);
        return nullptr;
      }
      int status = PyList_Append(result, pair);
      Py_DECREF(pair);
      if (status < 0) {
        Py_DECREF(result);
        return nullptr;
      }
    }
  }
  {
    int remaining = next_marker(*state, text, length, position, end);
    if (remaining == -2) {
      Py_DECREF(result);
      return nullptr;
    }
    if (remaining >= 0) goto mismatch;
  }
  return result;
mismatch:
  Py_DECREF(result);
  PyErr_SetString(PyExc_ValueError, "source markers differ from token IDs");
  return nullptr;
}
PyMethodDef methods[] = {
    {"make_markers", make_markers, METH_O,
     "Create immutable marker configuration."},
    {"lcp", lcp, METH_VARARGS, "Exact Unicode prefix length."},
    {"token_lcp", token_lcp, METH_VARARGS,
     "Exact token prefix length under the GIL."},
    {"max_token_id", max_token_id, METH_O,
     "Maximum exact uint32 token ID, or None for an unsupported element."},
    {"boundaries", boundaries, METH_VARARGS,
     "Sparse character/token boundary pairs."},
    {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_session_prefix", nullptr, -1,
                      methods};
}  // namespace
PyMODINIT_FUNC PyInit__session_prefix() { return PyModule_Create(&module); }
