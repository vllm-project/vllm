// Minimal, paranoid direction test: does a peer READ work, does a peer WRITE
// work, and is the source buffer actually filled in the first place?
// Every launch is error-checked and every buffer is verified LOCALLY first,
// so "peer read returns zeros" cannot be confused with "nobody ever wrote".

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
#define LAUNCHED()                                                         \
  do {                                                                     \
    CK(hipGetLastError());                                                 \
    CK(hipDeviceSynchronize());                                            \
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

static uint32_t* cbuf(int dev) {
  CK(hipSetDevice(dev));
  uint32_t* c;
  CK(hipMalloc(&c, 16));
  CK(hipMemset(c, 0, 16));
  return c;
}
struct R {
  uint32_t bad, zero, first;
};
static R run_check(int dev, const uint32_t* p, uint32_t n, uint32_t it,
                   uint32_t* c) {
  CK(hipSetDevice(dev));
  CK(hipMemset(c, 0, 16));
  check<<<16, 256>>>(p, n, it, c);
  LAUNCHED();
  uint32_t h[4];
  CK(hipMemcpy(h, c, 16, hipMemcpyDeviceToHost));
  return {h[0], h[1], h[3]};
}
static void say(const char* s, R r, uint32_t n) {
  printf("  %-52s %s", s, r.bad ? "ROTO " : "ok");
  if (r.bad)
    printf(" %u/%u  ceros=%u  primer_valor_malo=0x%08x", r.bad, n, r.zero,
           r.first);
  printf("\n");
}

int main(int argc, char** argv) {
  uint32_t n = (argc > 1) ? (uint32_t)atoi(argv[1]) : (1u << 18);
  int uncached = (argc > 2) ? atoi(argv[2]) : 1;
  size_t bytes = (size_t)n * 4;
  printf("buffer=%zu KiB  uncached=%d\n", bytes >> 10, uncached);

  CK(hipSetDevice(0));
  CK(hipDeviceEnablePeerAccess(1, 0));
  CK(hipSetDevice(1));
  CK(hipDeviceEnablePeerAccess(0, 0));

  auto alloc = [&](int dev) {
    CK(hipSetDevice(dev));
    void* p;
    if (uncached)
      CK(hipExtMallocWithFlags(&p, bytes, hipDeviceMallocUncached));
    else
      CK(hipMalloc(&p, bytes));
    CK(hipMemset(p, 0, bytes));
    CK(hipDeviceSynchronize());
    return (uint32_t*)p;
  };
  uint32_t* d0 = alloc(0);
  uint32_t* d1 = alloc(1);
  uint32_t* c0 = cbuf(0);
  uint32_t* c1 = cbuf(1);

  // 1. dev1 llena d1 y lo verifica EN dev1. Sin esto, todo lo demas es humo.
  CK(hipSetDevice(1));
  fill<<<16, 256>>>(d1, n, 13u);
  LAUNCHED();
  say("A  dev1 llena d1 y dev1 lo lee   (control)", run_check(1, d1, n, 13u, c1),
      n);

  // 2. dev0 lee d1 por el puntero peer. Misma memoria, mismo instante.
  say("B  dev0 LEE d1 por puntero peer", run_check(0, d1, n, 13u, c0), n);

  // 3. hipMemcpyPeer d1 -> d0, y dev0 lee d0 localmente.
  CK(hipSetDevice(0));
  CK(hipMemset(d0, 0, bytes));
  CK(hipMemcpyPeer(d0, 0, d1, 1, bytes));
  CK(hipDeviceSynchronize());
  say("C  hipMemcpyPeer d1->d0, luego lectura local", run_check(0, d0, n, 13u, c0),
      n);

  // 4. dev0 escribe EN d1 por el puntero peer; dev1 lo verifica localmente.
  CK(hipSetDevice(1));
  CK(hipMemset(d1, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipSetDevice(0));
  fill<<<16, 256>>>(d1, n, 17u);
  LAUNCHED();
  say("D  dev0 ESCRIBE en d1 por peer, dev1 lo lee", run_check(1, d1, n, 17u, c1),
      n);

  // 5. el sentido contrario, por si solo falla una direccion del enlace.
  CK(hipSetDevice(0));
  fill<<<16, 256>>>(d0, n, 19u);
  LAUNCHED();
  say("E  dev0 llena d0 y dev0 lo lee   (control)", run_check(0, d0, n, 19u, c0),
      n);
  say("F  dev1 LEE d0 por puntero peer", run_check(1, d0, n, 19u, c1), n);

  CK(hipSetDevice(1));
  CK(hipMemset(d0, 0, bytes));
  CK(hipDeviceSynchronize());
  CK(hipSetDevice(1));
  fill<<<16, 256>>>(d0, n, 23u);
  LAUNCHED();
  say("G  dev1 ESCRIBE en d0 por peer, dev0 lo lee", run_check(0, d0, n, 23u, c0),
      n);
  return 0;
}
