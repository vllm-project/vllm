// Same copy, different mover. If the engine that performs the transfer is the
// one that OWNS THE SOURCE it only has to write across the link (push). If it
// is the one that owns the destination it has to read across (pull).
// rocm-bandwidth-test says the path is fine; hipMemcpyPeer said zeros. This
// tells us which of the two the difference is.

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

__device__ __forceinline__ uint32_t pat(uint32_t it, uint32_t i) {
  return (it * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
}
__global__ void fill(uint32_t* p, uint32_t n, uint32_t it) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    p[i] = pat(it, i);
}
__global__ void check(const uint32_t* p, uint32_t n, uint32_t it, uint32_t* c) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    uint32_t v = p[i];
    if (v != pat(it, i)) {
      atomicAdd(&c[0], 1u);
      if (v == 0u) atomicAdd(&c[1], 1u);
      if (atomicAdd(&c[2], 1u) == 0) c[3] = v;
    }
  }
}

static uint32_t n;
static size_t bytes;
static uint32_t* cn[2];

static void say(const char* s, int dev, const uint32_t* p, uint32_t it) {
  CK(hipSetDevice(dev));
  CK(hipMemset(cn[dev], 0, 16));
  check<<<16, 256>>>(p, n, it, cn[dev]);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  uint32_t h[4];
  CK(hipMemcpy(h, cn[dev], 16, hipMemcpyDeviceToHost));
  printf("  %-56s %s", s, h[0] ? "ROTO " : "ok");
  if (h[0]) printf(" %u/%u ceros=%u primero=0x%08x", h[0], n, h[1], h[3]);
  printf("\n");
}

int main(int argc, char** argv) {
  n = (argc > 1) ? (uint32_t)atoi(argv[1]) : (1u << 18);
  bytes = (size_t)n * 4;
  printf("buffer=%zu KiB\n", bytes >> 10);

  CK(hipSetDevice(0));
  CK(hipDeviceEnablePeerAccess(1, 0));
  CK(hipSetDevice(1));
  CK(hipDeviceEnablePeerAccess(0, 0));

  void* p;
  CK(hipSetDevice(0));
  CK(hipMalloc(&p, bytes));
  uint32_t* d0 = (uint32_t*)p;
  CK(hipMalloc(&cn[0], 16));
  CK(hipSetDevice(1));
  CK(hipMalloc(&p, bytes));
  uint32_t* d1 = (uint32_t*)p;
  CK(hipMalloc(&cn[1], 16));

  hipStream_t s0, s1;
  CK(hipSetDevice(0));
  CK(hipStreamCreate(&s0));
  CK(hipSetDevice(1));
  CK(hipStreamCreate(&s1));

  // llenar d1 en dev1 (origen) y comprobarlo alli
  CK(hipSetDevice(1));
  fill<<<16, 256>>>(d1, n, 31u);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  say("control: dev1 lee d1", 1, d1, 31u);

  // A. copia agendada en el stream del DESTINO (dev0) -> el motor de dev0 LEE
  CK(hipSetDevice(0));
  CK(hipMemset(d0, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipMemcpyPeerAsync(d0, 0, d1, 1, bytes, s0));
  CK(hipStreamSynchronize(s0));
  say("A copia en el stream del DESTINO  (dev0 tira: pull)", 0, d0, 31u);

  // B. copia agendada en el stream del ORIGEN (dev1) -> el motor de dev1 ESCRIBE
  CK(hipSetDevice(0));
  CK(hipMemset(d0, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipSetDevice(1));
  CK(hipMemcpyPeerAsync(d0, 0, d1, 1, bytes, s1));
  CK(hipStreamSynchronize(s1));
  say("B copia en el stream del ORIGEN   (dev1 empuja: push)", 0, d0, 31u);

  // C. kernel del ORIGEN escribiendo en el destino (push explicito, sin DMA)
  CK(hipSetDevice(0));
  CK(hipMemset(d0, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipSetDevice(1));
  fill<<<16, 256>>>(d0, n, 31u);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  say("C kernel del ORIGEN escribe en el destino (push)", 0, d0, 31u);

  // D. y el mismo par por el otro sentido, para descartar asimetria de enlace
  CK(hipSetDevice(0));
  fill<<<16, 256>>>(d0, n, 37u);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
  CK(hipSetDevice(1));
  CK(hipMemset(d1, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipMemcpyPeerAsync(d1, 1, d0, 0, bytes, s1));
  CK(hipStreamSynchronize(s1));
  say("D 0->1 en el stream del DESTINO   (dev1 tira: pull)", 1, d1, 37u);

  CK(hipSetDevice(1));
  CK(hipMemset(d1, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipSetDevice(0));
  CK(hipMemcpyPeerAsync(d1, 1, d0, 0, bytes, s0));
  CK(hipStreamSynchronize(s0));
  say("E 0->1 en el stream del ORIGEN    (dev0 empuja: push)", 1, d1, 37u);
  return 0;
}
