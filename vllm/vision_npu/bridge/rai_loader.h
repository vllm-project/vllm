// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Cross-platform memory-mapped RAI file loader (from flexmlRT test/generic).

#ifndef VLLM_VISION_NPU_RAI_LOADER_H
#define VLLM_VISION_NPU_RAI_LOADER_H

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace vaiml_run {

class RaiLoader {
 public:
  RaiLoader() = default;
  ~RaiLoader() { unload(); }

  RaiLoader(const RaiLoader&) = delete;
  RaiLoader& operator=(const RaiLoader&) = delete;

  RaiLoader(RaiLoader&& other) noexcept
      : mapped_data_(other.mapped_data_),
        file_size_(other.file_size_)
#ifdef _WIN32
        ,
        file_handle_(other.file_handle_),
        mapping_handle_(other.mapping_handle_)
#else
        ,
        fd_(other.fd_)
#endif
  {
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
#ifdef _WIN32
    other.file_handle_ = INVALID_HANDLE_VALUE;
    other.mapping_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
  }

  RaiLoader& operator=(RaiLoader&& other) noexcept {
    if (this != &other) {
      unload();
      mapped_data_ = other.mapped_data_;
      file_size_ = other.file_size_;
#ifdef _WIN32
      file_handle_ = other.file_handle_;
      mapping_handle_ = other.mapping_handle_;
      other.file_handle_ = INVALID_HANDLE_VALUE;
      other.mapping_handle_ = nullptr;
#else
      fd_ = other.fd_;
      other.fd_ = -1;
#endif
      other.mapped_data_ = nullptr;
      other.file_size_ = 0;
    }
    return *this;
  }

  bool load(const fs::path& rai_path) {
    if (!fs::exists(rai_path)) {
      std::cerr << "ERROR: RAI file does not exist: " << rai_path << std::endl;
      return false;
    }
    rai_path_ = rai_path;
#ifdef _WIN32
    return loadWindows(rai_path);
#else
    return loadUnix(rai_path);
#endif
  }

  void unload() {
#ifdef _WIN32
    if (mapped_data_) {
      UnmapViewOfFile(mapped_data_);
      mapped_data_ = nullptr;
    }
    if (mapping_handle_) {
      CloseHandle(mapping_handle_);
      mapping_handle_ = nullptr;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_handle_);
      file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (mapped_data_ && mapped_data_ != MAP_FAILED) {
      munmap(mapped_data_, file_size_);
      mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
#endif
    file_size_ = 0;
  }

  void* data() const { return mapped_data_; }
  size_t size() const { return file_size_; }
  bool isLoaded() const { return mapped_data_ != nullptr && file_size_ > 0; }
  const fs::path& path() const { return rai_path_; }

 private:
#ifdef _WIN32
  bool loadWindows(const fs::path& rai_path);
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_handle_ = nullptr;
#else
  bool loadUnix(const fs::path& rai_path);
  int fd_ = -1;
#endif

  void* mapped_data_ = nullptr;
  size_t file_size_ = 0;
  fs::path rai_path_;
};

#ifdef _WIN32
inline bool RaiLoader::loadWindows(const fs::path& rai_path) {
  file_handle_ =
      CreateFileW(rai_path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file_handle_ == INVALID_HANDLE_VALUE) {
    std::cerr << "ERROR: Cannot open RAI file: " << rai_path << std::endl;
    return false;
  }
  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(file_handle_, &file_size)) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }
  file_size_ = static_cast<size_t>(file_size.QuadPart);
  mapping_handle_ =
      CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping_handle_) {
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }
  mapped_data_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
  if (!mapped_data_) {
    CloseHandle(mapping_handle_);
    mapping_handle_ = nullptr;
    CloseHandle(file_handle_);
    file_handle_ = INVALID_HANDLE_VALUE;
    return false;
  }
  return true;
}
#else
inline bool RaiLoader::loadUnix(const fs::path& rai_path) {
  fd_ = open(rai_path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    std::cerr << "ERROR: Cannot open RAI file: " << rai_path << std::endl;
    return false;
  }
  struct stat sb;
  if (fstat(fd_, &sb) < 0) {
    close(fd_);
    fd_ = -1;
    return false;
  }
  file_size_ = static_cast<size_t>(sb.st_size);
  mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped_data_ == MAP_FAILED) {
    close(fd_);
    fd_ = -1;
    mapped_data_ = nullptr;
    return false;
  }
  return true;
}
#endif

}  // namespace vaiml_run

#endif  // VLLM_VISION_NPU_RAI_LOADER_H
