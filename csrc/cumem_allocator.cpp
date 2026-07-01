// A CUDAPluggableAllocator based on cumem* APIs.
// Important: allocation size, CUdeviceptr and CUmemGenericAllocationHandle*
// need to be unsigned long long
#include <atomic>
#include <iostream>

#include "cumem_allocator_compat.h"

#ifndef USE_ROCM
static const char* PYARGS_PARSE = "KKKK";
#else
  #include <cstdlib>
  #include <cstdint>
  #include <cerrno>
  #include <climits>

// Default chunk size 256MB for ROCm. Can be overridden at runtime by the
// environment variable VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE, specified in megabytes
// (MB). The env value is parsed with strtoull as an integer number of MB
// (decimal or 0x hex). The parsed MB value is converted to bytes. If
// parsing fails, the value is 0, or the multiplication would overflow,
// the default (256MB) is used.
static const unsigned long long DEFAULT_MEMCREATE_CHUNK_SIZE =
    (256ULL * 1024ULL * 1024ULL);

static unsigned long long get_memcreate_chunk_size() {
  const char* env = getenv("VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE");
  if (!env) return DEFAULT_MEMCREATE_CHUNK_SIZE;
  char* endptr = nullptr;
  errno = 0;
  unsigned long long val_mb = strtoull(env, &endptr, 0);
  if (endptr == env || errno != 0) {
    // parsing failed, fallback to default
    return DEFAULT_MEMCREATE_CHUNK_SIZE;
  }
  if (val_mb == 0) return DEFAULT_MEMCREATE_CHUNK_SIZE;

  const unsigned long long MB = 1024ULL * 1024ULL;
  // guard against overflow when converting MB -> bytes
  if (val_mb > (ULLONG_MAX / MB)) {
    return DEFAULT_MEMCREATE_CHUNK_SIZE;
  }
  return val_mb * MB;
}

static inline unsigned long long my_min(unsigned long long a,
                                        unsigned long long b) {
  return a < b ? a : b;
}

static CUresult reserve_rocm_address(CUdeviceptr* d_mem, size_t size,
                                     size_t alignment, CUdeviceptr addr = 0) {
  CUresult status = cuMemAddressReserve(d_mem, size, alignment, addr, 0);
  if (status == CUresult(0) || alignment == 0) {
    return status;
  }

  // Some ROCm stacks can report OOM while reserving VA with an explicit
  // alignment even when physical VRAM is free. Let HIP choose the default
  // alignment, then verify that the returned address still satisfies the
  // requested alignment before accepting it.
  status = cuMemAddressReserve(d_mem, size, 0, addr, 0);
  if (status != CUresult(0)) {
    return status;
  }
  if (((std::uintptr_t)(*d_mem) % alignment) == 0) {
    return status;
  }

  (void)cuMemAddressFree(*d_mem, size);
  return hipErrorNotSupported;
}

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

// ---------------------------------------------------------------------------
// Cached fabric handle probe (CUDA 12.4+, NVIDIA only):

#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12040
// Per-device cache: 0 = not probed, 1 = supported, 2 = not supported
static constexpr int MAX_DEVICES = 32;
static std::atomic<int> fabric_support[MAX_DEVICES] = {};

static bool probe_fabric_support(unsigned long long device) {
  if (device >= MAX_DEVICES) return false;
  int cached = fabric_support[device].load(std::memory_order_acquire);
  if (cached != 0) return cached == 1;

  int fab_flag = 0;
  CUresult r = cuDeviceGetAttribute(
      &fab_flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
  if (r != CUDA_SUCCESS || !fab_flag) {
    fabric_support[device].store(2, std::memory_order_release);
    return false;
  }

  // Attribute says supported — verify with a real allocation.
  // cuDeviceGetAttribute can report supported even when IMEX is not
  // configured, so we need a real probe.
  CUmemAllocationProp probe_prop = {};
  probe_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  probe_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  probe_prop.location.id = device;
  probe_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity;
  r = cuMemGetAllocationGranularity(&granularity, &probe_prop,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (r != CUDA_SUCCESS) {
    fabric_support[device].store(2, std::memory_order_release);
    return false;
  }

  CUmemGenericAllocationHandle test_handle;
  r = cuMemCreate(&test_handle, granularity, &probe_prop, 0);
  if (r == CUDA_SUCCESS) {
    cuMemRelease(test_handle);
    fabric_support[device].store(1, std::memory_order_release);
    return true;
  }

  fabric_support[device].store(2, std::memory_order_release);
  return false;
}
#endif

// ---------------------------------------------------------------------------

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
  int flag = 0;
  CUresult rdma_result = cuDeviceGetAttribute(
      &flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      device);
  if (rdma_result == CUDA_SUCCESS && flag) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  #if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
  if (probe_fabric_support(device)) {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  } else {
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
  #else
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  #endif
#endif

#ifndef USE_ROCM
  // Allocate memory using cuMemCreate
  CUresult ret = (CUresult)cuMemCreate(p_memHandle, size, &prop, 0);
  if (ret) {
  #if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
    // Safety net: if fabric was probed as available but this allocation
    // still fails, fall back to POSIX FD and update the cache.
    if (device < MAX_DEVICES &&
        fabric_support[device].load(std::memory_order_acquire) == 1 &&
        (ret == CUDA_ERROR_NOT_PERMITTED || ret == CUDA_ERROR_NOT_SUPPORTED)) {
      fabric_support[device].store(2, std::memory_order_release);
      prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
      CUDA_CHECK(cuMemCreate(p_memHandle, size, &prop, 0));
    } else {
      CUDA_CHECK(ret);
    }
  #else
    CUDA_CHECK(ret);
  #endif
  }
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
      // Clean up previously created handles
      for (auto j = 0; j < i; ++j) {
        cuMemRelease(*(p_memHandle[j]));
      }
      return;
    }
  }
  unsigned long long allocated_size = 0;
  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUDA_CHECK(cuMemMap(map_addr, chunk_sizes[i], 0, *(p_memHandle[i]), 0));
    if (error_code != 0) {
      // unmap previously mapped chunks
      unsigned long long unmapped_size = 0;
      for (auto j = 0; j < i; ++j) {
        void* unmap_addr = (void*)((uintptr_t)d_mem + unmapped_size);
        cuMemUnmap(unmap_addr, chunk_sizes[j]);
        unmapped_size += chunk_sizes[j];
      }
      // release all created handles
      for (auto j = 0; j < num_chunks; ++j) {
        cuMemRelease(*(p_memHandle[j]));
      }
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
  CUresult first_error = no_error;

  for (auto i = 0; i < num_chunks; ++i) {
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    CUresult status = cuMemUnmap(map_addr, chunk_sizes[i]);
    if (status != no_error && first_error == no_error) {
      first_error = status;
    }
    allocated_size += chunk_sizes[i];
  }

  for (auto i = 0; i < num_chunks; ++i) {
    CUresult status = cuMemRelease(*(p_memHandle[i]));
    if (status != no_error && first_error == no_error) {
      first_error = status;
    }
  }

  if (first_error != no_error) {
    CUDA_CHECK(first_error);
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

  // No handle type here; create_and_map sets fabric/POSIX as needed.
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

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
  CUDA_CHECK(reserve_rocm_address(&d_mem, alignedSize, granularity));
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
  // Make sure chunk size is aligned with hardware granularity. The base
  // chunk size can be configured via environment variable
  // ``VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE``; otherwise
  // DEFAULT_MEMCREATE_CHUNK_SIZE is used.
  size_t base_chunk = (size_t)get_memcreate_chunk_size();
  size_t aligned_chunk_size =
      ((base_chunk + granularity - 1) / granularity) * granularity;
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
    if (p_memHandle[i] == nullptr) {
      std::cerr << "ERROR: malloc failed for p_memHandle[" << i << "].\n";
      for (auto j = 0; j < i; ++j) {
        free(p_memHandle[j]);
      }
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;
    }
    chunk_sizes[i] = (unsigned long long)my_min(
        (unsigned long long)(alignedSize - i * aligned_chunk_size),
        (unsigned long long)aligned_chunk_size);
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
  free(chunk_sizes);
#endif

  if (error_code != 0) {
    // free address and the handle
    CUDA_CHECK(cuMemAddressFree(d_mem, alignedSize));
#ifndef USE_ROCM
    free(p_memHandle);
#else
    for (size_t i = 0; i < num_chunks; ++i) {
      free(p_memHandle[i]);
    }
    free(p_memHandle);
#endif
    return nullptr;
  }

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
    Py_XDECREF(py_result);
    Py_XDECREF(py_ptr);
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
    Py_XDECREF(py_result);
    Py_XDECREF(py_ptr);
    return;
  }

  // For ROCm, copy the Python list of (addr,size) pairs into C arrays while
  // holding the GIL. Then release the GIL and call the unmap/release helper
  // using the copied arrays. This avoids calling PyList_* APIs without the
  // GIL (which is undefined behavior and can crash when called from other
  // threads).
  CUdeviceptr d_mem = (CUdeviceptr)recv_d_mem;
#ifdef USE_ROCM
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  if (p_memHandle == nullptr) {
    Py_DECREF(py_ptr);
    Py_DECREF(py_result);
    PyGILState_Release(gstate);
    std::cerr << "ERROR: malloc failed for p_memHandle in my_free."
              << std::endl;
    return;
  }
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  if (chunk_sizes == nullptr) {
    free(p_memHandle);
    Py_DECREF(py_ptr);
    Py_DECREF(py_result);
    PyGILState_Release(gstate);
    std::cerr << "ERROR: malloc failed for chunk_sizes in my_free."
              << std::endl;
    return;
  }
  for (Py_ssize_t i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = (unsigned long long)PyLong_AsUnsignedLongLong(size_py);
  }

  // Drop temporary Python refs, then release the GIL before calling into
  // non-Python APIs.
  Py_DECREF(py_ptr);
  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  // An empty chunk list means this allocation is asleep: its physical chunks
  // were already unmapped and released by sleep(), but the virtual address is
  // still held as a placeholder reservation. Skip unmap/release (freeing the
  // placeholder address happens below).
  if (num_chunks > 0) {
    unmap_and_release(device, size, d_mem, p_memHandle, chunk_sizes,
                      num_chunks);
  }
#else
  // Non-ROCm path: simple integer handle already extracted; drop temporary
  // Python refs while still holding the GIL, then release it.
  Py_DECREF(py_ptr);
  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  CUmemGenericAllocationHandle* p_memHandle =
      (CUmemGenericAllocationHandle*)recv_p_memHandle;
  unmap_and_release(device, size, d_mem, p_memHandle);
#endif

  // Free the virtual address. On ROCm this also covers an asleep allocation,
  // whose placeholder reservation made by sleep() is still held here.
  CUDA_CHECK(cuMemAddressFree(d_mem, size));
#ifndef USE_ROCM
  free(p_memHandle);
#else
  // Only awake allocations have per-chunk handles to free.
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
  if (!PyList_Check(recv_p_memHandle)) {
    PyErr_SetString(PyExc_TypeError,
                    "Expected a list for the 4th argument on ROCm");
    return nullptr;
  }
  Py_ssize_t num_chunks = PyList_Size(recv_p_memHandle);
  if (num_chunks < 0) {
    return nullptr;  // PyList_Size sets an exception on error.
  }
  CUmemGenericAllocationHandle** p_memHandle =
      (CUmemGenericAllocationHandle**)malloc(
          num_chunks * sizeof(CUmemGenericAllocationHandle*));
  if (p_memHandle == nullptr) {
    PyErr_SetString(PyExc_MemoryError, "malloc failed for p_memHandle");
    return nullptr;
  }
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  if (chunk_sizes == nullptr) {
    free(p_memHandle);
    PyErr_SetString(PyExc_MemoryError, "malloc failed for chunk_sizes");
    return nullptr;
  }
  for (Py_ssize_t i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    if (item == nullptr || !PyTuple_Check(item) || PyTuple_Size(item) != 2) {
      free(p_memHandle);
      free(chunk_sizes);
      PyErr_SetString(
          PyExc_TypeError,
          "List items must be tuples of size 2 (handle_addr, size)");
      return nullptr;
    }
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    if (addr_py == nullptr || size_py == nullptr) {
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;  // PyTuple_GetItem sets an exception
    }
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    if (PyErr_Occurred()) {
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;
    }
    chunk_sizes[i] = (unsigned long long)PyLong_AsUnsignedLongLong(size_py);
    if (PyErr_Occurred()) {
      free(p_memHandle);
      free(chunk_sizes);
      return nullptr;
    }
  }

  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle, chunk_sizes,
                    num_chunks);

  // On ROCm/Linux, physical VRAM is only reclaimed once the virtual address
  // range is freed; hipMemUnmap + hipMemRelease alone leave the memory
  // resident (see ROCm#6021). Free the address to release physical memory,
  // then immediately re-reserve the SAME address as an empty placeholder so
  // the regular allocator cannot hand it out while we sleep. wake_up remaps
  // physical chunks into this placeholder.
  if (error_code == no_error) {
    CUDA_CHECK(cuMemAddressFree(d_mem_ptr, recv_size));
    if (error_code == no_error) {
      CUdeviceptr reserved = 0;
      CUDA_CHECK(reserve_rocm_address(&reserved, recv_size, /*alignment=*/0,
                                      d_mem_ptr));
      if (error_code == no_error && reserved != d_mem_ptr) {
        (void)cuMemAddressFree(reserved, recv_size);
        snprintf(error_msg, sizeof(error_msg),
                 "failed to re-reserve placeholder address on sleep "
                 "(requested %#llx, got %#llx)",
                 (unsigned long long)d_mem_ptr, (unsigned long long)reserved);
        error_code = CUresult(1);
      }
    }
  }

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
  if (p_memHandle == nullptr) {
    PyErr_SetString(PyExc_MemoryError, "malloc failed for p_memHandle");
    return nullptr;
  }
  unsigned long long* chunk_sizes =
      (unsigned long long*)malloc(num_chunks * sizeof(unsigned long long));
  if (chunk_sizes == nullptr) {
    free(p_memHandle);
    PyErr_SetString(PyExc_MemoryError, "malloc failed for chunk_sizes");
    return nullptr;
  }
  for (auto i = 0; i < num_chunks; ++i) {
    PyObject* item = PyList_GetItem(recv_p_memHandle, i);
    PyObject* addr_py = PyTuple_GetItem(item, 0);
    PyObject* size_py = PyTuple_GetItem(item, 1);
    p_memHandle[i] =
        (CUmemGenericAllocationHandle*)PyLong_AsUnsignedLongLong(addr_py);
    chunk_sizes[i] = PyLong_AsUnsignedLongLong(size_py);
  }

  // Address already reserved as a placeholder by sleep(); just remap chunks.
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
