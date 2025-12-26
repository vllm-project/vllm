#pragma once
#include <cuda.h>
#include <string>
#include <cstdint>

struct ImportedPool {
  CUmemGenericAllocationHandle handle{0};
  CUdeviceptr base_va{0};
  uint64_t size{0};
  uint64_t gran{0};
  int device{0};
};

class VmmPoolClient {
public:
  VmmPoolClient(const std::string& uds_path, int device);
  ~VmmPoolClient();

  ImportedPool& pool() { return pool_; }

  bool Hello();
  std::pair<uint64_t,uint64_t> Allocate(uint64_t nbytes);
  bool Free(uint64_t off, uint64_t len);
  std::tuple<uint64_t,uint64_t,uint64_t> Stats();

  CUdeviceptr Map(uint64_t off, uint64_t len);
  void Unmap(uint64_t off, uint64_t len);

private:
  std::string uds_;
  int dev_;
  CUdevice cu_dev_{};
  CUcontext ctx_{};
  ImportedPool pool_;
  int hello_fd_{-1};
};
