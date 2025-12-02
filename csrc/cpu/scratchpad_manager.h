#ifndef SCRATCHPAD_MANAGER_H
#define SCRATCHPAD_MANAGER_H

#include <cstddef>
#include <cstdio>

class DNNLScratchPadManager {
 public:
  static constexpr size_t allocation_unit = 4 * 1024;  // 4KB

  static DNNLScratchPadManager* get_dnnl_scratchpad_manager();

  DNNLScratchPadManager();

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(ptr_);
  }

  static size_t round(size_t size) {
    return ((size + allocation_unit - 1) / allocation_unit) * allocation_unit;
  }

  void realloc(size_t new_size);

 private:
  size_t size_;
  void* ptr_;
};

#endif
