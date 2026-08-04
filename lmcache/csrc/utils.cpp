#include <cuda_runtime.h>
#include <stdexcept>
#include "utils.h"

std::string get_gpu_pci_bus_id(int device) {
  char pciBusId[13];  // 13 bytes per CUDA doc
  cudaError_t err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaDeviceGetPCIBusId failed: ") +
                             cudaGetErrorString(err));
  }
  return std::string(pciBusId);
}
