#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cassert>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mman.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <cstring>            // for strerror
#include <linux/mempolicy.h>  // for MPOL_BIND, MPOL_MF_MOVE, MPOL_MF_STRICT
#include "mem_alloc.h"

static constexpr size_t HUGEPAGE_SIZE = 2UL * 1024 * 1024;  // MAP_HUGE_2MB

static inline size_t _align_hugepage(size_t size) {
  return (size + HUGEPAGE_SIZE - 1) & ~(HUGEPAGE_SIZE - 1);
}

static void* _mmap_anon(size_t size, bool hugepages) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS;
  if (hugepages) {
    flags |= MAP_HUGETLB | MAP_HUGE_2MB;
  }
  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }
  return ptr;
}

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&ptr, size, flags);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc failed: " + std::to_string(err));
  }
  return reinterpret_cast<uintptr_t>(ptr);
}

void free_pinned_ptr(uintptr_t ptr) {
  cudaError_t err = cudaFreeHost(reinterpret_cast<void*>(ptr));
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaFreeHost failed: " + std::to_string(err));
  }
}

uintptr_t alloc_hugepage_pinned_ptr(size_t size, unsigned int flags) {
  size = _align_hugepage(size);
  void* ptr = _mmap_anon(size, true);

  cudaError_t st = cudaHostRegister(ptr, size, flags);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_hugepage_pinned_ptr(uintptr_t ptr, size_t size) {
  size = _align_hugepage(size);
  void* p = reinterpret_cast<void*>(ptr);

  // Unpin first, then unmap.
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

void batched_memcpy(const std::vector<uintptr_t>& src_ptrs,
                    const std::vector<uintptr_t>& dst_ptrs,
                    const std::vector<size_t>& sizes) {
  if (src_ptrs.size() != dst_ptrs.size() || src_ptrs.size() != sizes.size()) {
    throw std::invalid_argument(
        "batched_memcpy expects equally sized src_ptrs, dst_ptrs, and sizes");
  }

  for (size_t i = 0; i < src_ptrs.size(); ++i) {
    if (sizes[i] == 0) {
      continue;
    }
    std::memmove(reinterpret_cast<void*>(dst_ptrs[i]),
                 reinterpret_cast<const void*>(src_ptrs[i]), sizes[i]);
  }
}

static void first_touch(void* p, size_t size, bool hugepages) {
  const size_t ps =
      hugepages ? HUGEPAGE_SIZE : static_cast<size_t>(sysconf(_SC_PAGESIZE));
  for (size_t off = 0; off < size; off += ps) {
    volatile char* c = static_cast<volatile char*>(p) + off;
    *c = 0;
  }
}

static inline int mbind_sys(void* addr, unsigned long len, int mode,
                            const unsigned long* nodemask,
                            unsigned long maxnode, unsigned int flags) {
  long rc = syscall(SYS_mbind, addr, len, mode, nodemask, maxnode, flags);
  return (rc == -1) ? -errno : 0;
}

static uintptr_t _alloc_numa_impl(size_t size, int node, bool hugepages) {
  if (hugepages) {
    assert(size % HUGEPAGE_SIZE == 0);
  }

  void* ptr = _mmap_anon(size, hugepages);

  // Maximum of 64 numa nodes
  unsigned long mask = 1UL << node;
  long maxnode = 8 * sizeof(mask);
  if (mbind_sys(ptr, size, MPOL_BIND, &mask, maxnode,
                MPOL_MF_MOVE | MPOL_MF_STRICT) != 0) {
    int err = errno;
    munmap(ptr, size);
    throw std::runtime_error(std::string("mbind failed: ") + strerror(err));
  }

  first_touch(ptr, size, hugepages);

  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t alloc_numa_ptr(size_t size, int node) {
  return _alloc_numa_impl(size, node, false);
}

void free_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

static uintptr_t _alloc_pinned_numa_impl(size_t size, int node,
                                         bool hugepages) {
  void* ptr = reinterpret_cast<void*>(_alloc_numa_impl(size, node, hugepages));

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t alloc_pinned_numa_ptr(size_t size, int node) {
  return _alloc_pinned_numa_impl(size, node, false);
}

uintptr_t alloc_hugepage_pinned_numa_ptr(size_t size, int node) {
  size = _align_hugepage(size);
  return _alloc_pinned_numa_impl(size, node, true);
}

void free_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  void* p = reinterpret_cast<void*>(ptr);
  // Unpin first, then unmap.
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
}

void free_hugepage_pinned_numa_ptr(uintptr_t ptr, size_t size) {
  size = _align_hugepage(size);
  free_pinned_numa_ptr(ptr, size);
}

uintptr_t alloc_shm_pinned_ptr(size_t size, const std::string& shm_name) {
  int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
  if (fd < 0)
    throw std::runtime_error(std::string("shm_open failed: ") +
                             strerror(errno));

  if (ftruncate(fd, size) != 0) {
    int err = errno;
    close(fd);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("ftruncate failed: ") + strerror(err));
  }

  void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("mmap failed: ") + strerror(errno));
  }

  first_touch(ptr, size, false);

  cudaError_t st = cudaHostRegister(ptr, size, 0);
  if (st != cudaSuccess) {
    munmap(ptr, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                             cudaGetErrorString(st));
  }

  return reinterpret_cast<uintptr_t>(ptr);
}

void free_shm_pinned_ptr(uintptr_t ptr, size_t size,
                         const std::string& shm_name) {
  void* p = reinterpret_cast<void*>(ptr);
  cudaError_t st = cudaHostUnregister(p);
  if (st != cudaSuccess) {
    munmap(p, size);
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("cudaHostUnregister failed: ") +
                             cudaGetErrorString(st));
  }
  if (munmap(p, size) != 0) {
    shm_unlink(shm_name.c_str());
    throw std::runtime_error(std::string("munmap failed: ") + strerror(errno));
  }
  shm_unlink(shm_name.c_str());
}
