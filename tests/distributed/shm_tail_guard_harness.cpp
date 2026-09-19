// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANON
  #define MAP_ANON MAP_ANONYMOUS
#endif

#ifndef MAP_POPULATE
  #define MAP_POPULATE 0
#endif

#include "csrc/cpu/shm.cpp"

struct GuardedBuffer {
  void* mapping = MAP_FAILED;
  size_t page_size = 0;
  int8_t* data = nullptr;

  explicit GuardedBuffer(int64_t bytes) {
    page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (bytes <= 0 || static_cast<size_t>(bytes) > page_size) {
      std::cerr << "bytes must fit in one page\n";
      std::exit(2);
    }
    mapping = mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mapping == MAP_FAILED) {
      perror("mmap");
      std::exit(2);
    }
    if (mprotect(static_cast<int8_t*>(mapping) + page_size, page_size,
                 PROT_NONE) != 0) {
      perror("mprotect");
      std::exit(2);
    }
    data = static_cast<int8_t*>(mapping) + page_size - bytes;
  }

  ~GuardedBuffer() {
    if (mapping != MAP_FAILED) {
      munmap(mapping, page_size * 2);
    }
  }
};

int run_copy(int8_t* dst, int8_t* src, int64_t bytes) {
  for (int64_t i = 0; i < bytes; ++i) {
    src[i] = static_cast<int8_t>(i % 127);
    dst[i] = 0;
  }

  shm_cc_ops::memcpy_to_shm(dst, src, bytes);

  for (int64_t i = 0; i < bytes; ++i) {
    if (dst[i] != src[i]) {
      std::cerr << "copy mismatch at index " << i << "\n";
      return 1;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  const int64_t bytes = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 65;
  const std::string_view mode = argc > 2 ? argv[2] : "guard-src";
  std::cout << "bytes=" << bytes << " mode=" << mode << std::endl;

  if (mode == "guard-src") {
    GuardedBuffer src(bytes);
    std::vector<int8_t> dst(static_cast<size_t>(bytes) + 64);
    return run_copy(dst.data(), src.data, bytes);
  }
  if (mode == "guard-dst") {
    std::vector<int8_t> src(static_cast<size_t>(bytes) + 64);
    GuardedBuffer dst(bytes);
    return run_copy(dst.data, src.data(), bytes);
  }

  std::cerr << "mode must be guard-src or guard-dst\n";
  return 2;
}
