// A CUDAPluggableAllocator based on cumem* APIs.
// Important: allocation size, CUdeviceptr and CUmemGenericAllocationHandle*
// need to be unsigned long long
#include <iostream>

#include "cumem_allocator_compat.h"

#ifndef USE_ROCM
static const char* PYARGS_PARSE = "KKKK";
#else
  #define MEMCREATE_CHUNK_SIZE (2 * 1024 * 1024)
  #define MIN(a, b) (a < b ? a : b)

static const char* PYARGS_PARSE = "KKKO";
#endif

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>

char error_msg[10240];  // 10KB buffer to store error messages
CUresult no_error = CUresult(0);
CUresult error_code = no_error;  // store error code

#define CUDA_CHECK(condition)                                           \
  do {                                                                  \
    CUresult error = condition;                                         \
    if (error != 0) {                                                   \
      error_code = error;                                               \
      char* error_string;                                               \
      cuGetErrorString(error, (const char**)&error_string);             \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d", \
               error_string, __FILE__, __LINE__);                       \
      std::cerr << error_msg << std::endl;                              \
    }                                                                   \
  } while (0)

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
// This brings the limitation that the allocator needs to be singleton.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

// ---------------------------------------------------------------------------
// Helper functions:

void ensure_context(unsigned long long device) {
  CUcontext pctx;
  CUDA_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    // Ensure device context.
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&pctx, device));
    CUDA_CHECK(cuCtxSetCurrent(pctx));
  }
}

void create_and_map(unsigned long long device, ssize_t size, CUdeviceptr d_mem,
#ifndef USE_ROCM
                    CUmemGenericAllocationHandle* p_memHandle) {
#else
                    CUmemGenericAllocationHandle** p_memHandle,
                    unsigned long long* chunk_sizes, size_t num_chunks) {
#endif
  ensure_context(device);
  // Define memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

#ifndef USE_ROCM
  // Allocate memory using cuMemCreate
  CUDA_CHECK(cuMemCreate(p_memHandle, size, &prop, 0));
  if (error_code != 0) {
    return;
  }
  CUDA_CHECK(cuMemMap(d_mem, size, 0, *p_memHandle, 0));
  if (error_code != 0) {
    return;
  }
#else
  for (auto i = 0; i < num_chunks; ++i) {
    CUDA_CHECK(cuMemCreate(p_memHandle[i], chunk_sizes[i], &prop, 0));
    if (error_code != 0) {
      return;
    }
  }
  unsigned long long allocated_size = 0;
  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUDA_CHECK(cuMemMap(map_addr, chunk_sizes[i], 0, *(p_memHandle[i]), 0));
    if (error_code != 0) {
      return;
    }
    allocated_size += chunk_sizes[i];
  }
#endif

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUDA_CHECK(cuMemSetAccess(d_mem, size, &accessDesc, 1));
  if (error_code != 0) {
    return;
  }
  // std::cout << "create_and_map: device=" << device << ", size=" << size << ",
  // d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
}

void unmap_and_release(unsigned long long device, ssize_t size,
                       CUdeviceptr d_mem,
#ifndef USE_ROCM
                       CUmemGenericAllocationHandle* p_memHandle) {
#else
                       CUmemGenericAllocationHandle** p_memHandle,
                       unsigned long long* chunk_sizes, size_t num_chunks) {
#endif
  // std::cout << "unmap_and_release: device=" << device << ", size=" << size <<
  // ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
  ensure_context(device);
#ifndef USE_ROCM
  CUDA_CHECK(cuMemUnmap(d_mem, size));
  if (error_code != 0) {
    return;
  }
  CUDA_CHECK(cuMemRelease(*p_memHandle));
  if (error_code != 0) {
    return;
  }
#else
  unsigned long long allocated_size = 0;
  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUDA_CHECK(cuMemUnmap(map_addr, chunk_sizes[i]));
    if (error_code != 0) {
      return;
    }
    allocated_size += chunk_sizes[i];
  }
  for (auto i = 0; i < num_chunks; ++i) {
    CUDA_CHECK(cuMemRelease(*(p_memHandle[i])));
    if (error_code != 0) {
      return;
    }
  }
#endif
}

PyObject* create_tuple_from_c_integers(unsigned long long a,
                                       unsigned long long b,
                                       unsigned long long c,
                                       unsigned long long d) {
  // Create a new tuple of size 4
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) {
    return NULL;  // Return NULL on failure
  }

  // Convert integers to Python objects and set them in the tuple
  PyTuple_SetItem(
      tuple, 0,
      PyLong_FromUnsignedLongLong(a));  // Steals reference to the PyLong
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));

  // Note: PyTuple_SetItem "steals" a reference to each object,
  // so we do not need to Py_DECREF the PyLong objects explicitly.

  return tuple;  // Return the created tuple
}

PyObject* create_tuple_from_c_mixed(unsigned long long a, unsigned long long b,
                                    unsigned long long c,
                                    CUmemGenericAllocationHandle** vec,
                                    unsigned long long* chunk_sizes,
                                    size_t num_chunks) {
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) {
    return NULL;
  }

  // PyObject* list = PyList_New(vec.size());
  PyObject* list = PyList_New(num_chunks);
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* addr_size_pair = PyTuple_New(2);
    PyObject* addr = PyLong_FromUnsignedLongLong((unsigned long long)(vec[i]));
    PyObject* size =
        PyLong_FromUnsignedLongLong((unsigned long long)(chunk_sizes[i]));
    PyTuple_SetItem(addr_size_pair, 0, addr);
    PyTuple_SetItem(addr_size_pair, 1, size);
    PyList_SetItem(list, i, addr_size_pair);
  }

  PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLongLong(a));
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, list);

  return tuple;
}

// ---------------------------------------------------------------------------
// Our exported C functions that call Python:

// use CUstream instead of cudaStream_t, to avoid including cuda_runtime_api.h
void* my_malloc(ssize_t size, int device, CUstream stream) {
  ensure_context(device);

  // first allocation, align the size, and reserve an address, and also allocate
  // a CUmemGenericAllocationHandle

  // Define memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  // Check if the allocation is supported
  size_t granularity;
  CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (error_code != 0) {
    return nullptr;
  }
  size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

  CUdeviceptr d_mem;
#ifndef USE_ROCM
  CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, 0, 0, 0));
  if (error_code != 0) {
    return nullptr;
  }
#else
  CUDA_CHECK(cuMemAddressReserve(&d_mem, alignedSize, granularity, 0, 0));
  if (error_code != 0) {
    return nullptr;
  }
#endif

#ifndef USE_ROCM
  // allocate the CUmemGenericAllocationHandle
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)malloc(
          sizeof(CUmemGenericAllocationHandle));
#else
  // Make sure chunk size is aligned with hardware granularity
  size_t aligned_chunk_size =
      ((MEMCREATE_CHUNK_SIZE + granularity - 1) / granularity) * granularity;
  size_t num_chunks =
      (alignedSize + aligned_chunk_size - 1) / aligned_chunk_size;
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  for (auto i = 0; i < num_chunks; ++i) {
    p_memHandle[i] = (CUmemGenericAllocationHandle*)malloc(
        sizeof(CUmemGenericAllocationHandle));
    chunk_sizes[i] =
        MIN(alignedSize - i * aligned_chunk_size, aligned_chunk_size);
  }
#endif

  if (!g_python_malloc_callback) {
    std::cerr << "ERROR: g_python_malloc_callback not set.\n";
    return nullptr;
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

#ifndef USE_ROCM
  PyObject* arg_tuple = create_tuple_from_c_integers(
      (unsigned long long)device, (unsigned long long)alignedSize,
      (unsigned long long)d_mem, (unsigned long long)p_memHandle);
#else
  PyObject* arg_tuple = create_tuple_from_c_mixed(
      (unsigned long long)device, (unsigned long long)alignedSize,
      (unsigned long long)d_mem, p_memHandle, chunk_sizes, num_chunks);
#endif

  // Call g_python_malloc_callback
  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
  Py_DECREF(arg_tuple);

  if (!py_result) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return nullptr;
  }

  PyGILState_Release(gstate);

  // do the final mapping
#ifndef USE_ROCM
  create_and_map(device, alignedSize, d_mem, p_memHandle);
#else
  create_and_map(device, alignedSize, d_mem, p_memHandle, chunk_sizes,
                 num_chunks);

  free(p_memHandle);
  free(chunk_sizes);
#endif

  return (void*)d_mem;
}

// use CUstream instead of cudaStream_t, to avoid including cuda_runtime_api.h
void my_free(void* ptr, ssize_t size, int device, CUstream stream) {
  // get memory handle from the pointer
  if (!g_python_free_callback) {
    std::cerr << "ERROR: g_python_free_callback not set.\n";
    return;
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);

  if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem;
#ifndef USE_ROCM
  unsigned long long recv_p_memHandle;
#else
  PyObject* recv_p_memHandle;
#endif
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(py_result, PYARGS_PARSE, &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return;
  }

  PyGILState_Release(gstate);

  // recv_size == size
  // recv_device == device

  // Free memory

  CUdeviceptr d_mem = (CUdeviceptr)recv_d_mem;
#ifndef USE_ROCM
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;

  unmap_and_release(device, size, d_mem, p_memHandle);
#else
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = (unsigned long long)PyLong_AsUnsignedLongLong(size_py);
  }

  unmap_and_release(device, size, d_mem, p_memHandle, chunk_sizes, num_chunks);
#endif

  // free address and the handle
  CUDA_CHECK(cuMemAddressFree(d_mem, size));
  if (error_code != 0) {
    return;
  }
#ifndef USE_ROCM
  free(p_memHandle);
#else
  for (auto i = 0; i < num_chunks; ++i) {
    free(p_memHandle[i]);
  }
  free(p_memHandle);
  free(chunk_sizes);
#endif
}

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args) {
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }

  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // Save the Python callables
  // This module does not handle GC of these objects, so they must be kept alive
  // outside of this module.
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;

  Py_RETURN_NONE;
}

static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem;
#ifndef USE_ROCM
  unsigned long long recv_p_memHandle;
#else
  PyObject* recv_p_memHandle;
#endif
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, PYARGS_PARSE, &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
#ifndef USE_ROCM
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;

  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);
#else
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = (unsigned long long)PyLong_AsUnsignedLongLong(size_py);
  }

  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle, chunk_sizes,
                    num_chunks);

  free(p_memHandle);
  free(chunk_sizes);
#endif

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem;
#ifndef USE_ROCM
  unsigned long long recv_p_memHandle;
#else
  PyObject* recv_p_memHandle;
#endif
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, PYARGS_PARSE, &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  CUdeviceptr d_mem_ptr = (CUdeviceptr)recv_d_mem;
#ifndef USE_ROCM
  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;

  create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);
#else
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = PyLong_AsUnsignedLongLong(size_py);
  }

  create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle, chunk_sizes,
                 num_chunks);

  free(p_memHandle);
  free(chunk_sizes);
#endif

  if (error_code != 0) {
    error_code = no_error;
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_create_and_map", (PyCFunction)python_create_and_map, METH_VARARGS,
     "Create and map memory on the device."},
    {"python_unmap_and_release", (PyCFunction)python_unmap_and_release,
     METH_VARARGS, "Unmap and release memory on the device."},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef cumem_allocator_module = {
    PyModuleDef_HEAD_INIT, "cumem_allocator",
    "cumem-based allocator for CUDAPluggableAllocator", -1, module_methods};

PyMODINIT_FUNC PyInit_cumem_allocator(void) {
  // Initialize the module
  PyObject* module = PyModule_Create(&cumem_allocator_module);
  if (!module) {
    return NULL;
  }
  return module;
}
}  // extern "C"
