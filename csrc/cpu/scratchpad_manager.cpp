#include <cstdlib>

#include "scratchpad_manager.h"

DNNLScratchPadManager::DNNLScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

void DNNLScratchPadManager::realloc(size_t new_size) {
  new_size = round(new_size);
  if (new_size > size_) {
    if (ptr_ != nullptr) {
      std::free(ptr_);
    }
    ptr_ = std::aligned_alloc(64, new_size);
    size_ = new_size;
  }
}

DNNLScratchPadManager* DNNLScratchPadManager::get_dnnl_scratchpad_manager() {
  static DNNLScratchPadManager manager;
  return &manager;
}
