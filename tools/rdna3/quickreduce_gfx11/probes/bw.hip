// Where does the push all-reduce's 2.8 GB/s go? Time each ingredient on its
// own: the peer write over PCIe, the local read out of uncached memory, and
// the same local read out of ordinary cached memory as a reference point.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                              \
  do {                                                                     \
    hipError_t e = (x);                                                    \
    if (e != hipSuccess) {                                                 \
      printf("FATAL %s:%d %s -> %s\n", __FILE__, __LINE__, #x,             \
             hipGetErrorString(e));                                        \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

__global__ void k_write(float4* dst, const float4* src, uint32_t n4) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += gridDim.x * blockDim.x)
    dst[i] = src[i];
}
__global__ void k_read(const float4* src, uint32_t n4, float4* sink) {
  float4 acc = {0, 0, 0, 0};
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += gridDim.x * blockDim.x) {
    float4 v = src[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x == 1e30f) *sink = acc;  // impide que el compilador lo borre
}

static float timed(int dev, hipStream_t s, void (*)(void), int) { return 0; }

int main(int argc, char** argv) {
  uint32_t n = (argc > 1) ? (uint32_t)atoi(argv[1]) : (1u << 20);
  int reps = (argc > 2) ? atoi(argv[2]) : 100;
  size_t bytes = (size_t)n * 4;
  uint32_t n4 = n / 4;
  printf("%zu KiB x %d repeticiones, bloques=16x256\n", bytes >> 10, reps);

  CK(hipSetDevice(0));
  CK(hipDeviceEnablePeerAccess(1, 0));
  CK(hipSetDevice(1));
  CK(hipDeviceEnablePeerAccess(0, 0));

  void *p;
  CK(hipSetDevice(0));
  CK(hipMalloc(&p, bytes)); float4* src0 = (float4*)p;
  CK(hipMalloc(&p, bytes)); float4* sink0 = (float4*)p;
  CK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached)); float4* unc0 = (float4*)p;
  CK(hipMalloc(&p, bytes)); float4* cac0 = (float4*)p;
  CK(hipMemset(src0, 1, bytes)); CK(hipMemset(unc0, 1, bytes)); CK(hipMemset(cac0, 1, bytes));
  CK(hipSetDevice(1));
  CK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached)); float4* unc1 = (float4*)p;
  CK(hipMalloc(&p, bytes)); float4* cac1 = (float4*)p;
  CK(hipMemset(unc1, 0, bytes)); CK(hipMemset(cac1, 0, bytes));
  CK(hipDeviceSynchronize());

  hipEvent_t a, b;
  CK(hipSetDevice(0));
  CK(hipEventCreate(&a)); CK(hipEventCreate(&b));

  auto bench = [&](const char* name, int mode, float4* dst, const float4* s) {
    CK(hipSetDevice(0));
    // calentar
    if (mode == 0) k_write<<<16, 256>>>(dst, s, n4);
    else k_read<<<16, 256>>>(s, n4, sink0);
    CK(hipGetLastError()); CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int r = 0; r < reps; ++r) {
      if (mode == 0) k_write<<<16, 256>>>(dst, s, n4);
      else k_read<<<16, 256>>>(s, n4, sink0);
    }
    CK(hipEventRecord(b));
    CK(hipEventSynchronize(b));
    float ms = 0; CK(hipEventElapsedTime(&ms, a, b));
    printf("  %-52s %8.2f GB/s   %7.1f us/rep\n", name,
           (double)bytes * reps / 1e9 / (ms / 1e3), ms * 1000.0 / reps);
  };

  bench("ESCRITURA al peer, destino uncached (PCIe)", 0, unc1, src0);
  bench("ESCRITURA al peer, destino cacheado (PCIe)", 0, cac1, src0);
  bench("escritura local, destino uncached", 0, unc0, src0);
  bench("escritura local, destino cacheado  (referencia VRAM)", 0, cac0, src0);
  bench("LECTURA local de memoria uncached", 1, nullptr, unc0);
  bench("LECTURA local de memoria cacheada  (referencia VRAM)", 1, nullptr, cac0);
  return 0;
}
