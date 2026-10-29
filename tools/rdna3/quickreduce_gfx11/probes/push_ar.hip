// A PUSH all-reduce for a box whose PCIe root complex forwards peer writes but
// not peer reads.
//
// vLLM's cross_device_reduce_1stage is a PULL: every rank reads every peer's
// buffer. Here each rank instead WRITES its contribution into a slot the peer
// owns, and then reduces reading only its OWN memory. Same bytes on the wire,
// opposite direction. Correctness rests on PCIe keeping posted writes in
// order, which is what NCCL's PCIe protocol relies on too.
//
// Self-checking: CTRL runs the same reduction against a deliberately wrong
// expected value, so a harness that cannot fail is visible as one that never
// fails.

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

static constexpr int NBLK = 16;
static constexpr int NTHR = 256;
static constexpr int NGPUS = 2;

struct Signal {
  alignas(128) uint32_t start[NBLK][NGPUS];
  alignas(128) uint32_t end[NBLK][NGPUS];
};

__device__ __forceinline__ bool barrier(uint32_t (*self)[NGPUS],
                                        uint32_t (*peer)[NGPUS], int rank,
                                        uint32_t flag, uint64_t budget) {
  bool ok = true;
  if (threadIdx.x < NGPUS) {
    uint32_t* slots[NGPUS];
    slots[rank] = &self[blockIdx.x][rank];
    slots[1 - rank] = &peer[blockIdx.x][rank];
    __scoped_atomic_store_n(slots[threadIdx.x], flag, __ATOMIC_RELAXED,
                            __MEMORY_SCOPE_SYSTEM);
    uint64_t t0 = wall_clock64();
    while (__scoped_atomic_load_n(&self[blockIdx.x][threadIdx.x],
                                  __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) <
           flag) {
      if (wall_clock64() - t0 > budget) {
        ok = false;
        break;
      }
    }
  }
  __syncthreads();
  return ok;
}

// inbox layout on every rank: [NGPUS][n], slot s holds rank s's contribution.
__global__ void k_push_allreduce(const float* in, float* out, uint32_t n,
                                 float* self_inbox, float* peer_inbox,
                                 Signal* self_sig, Signal* peer_sig, int rank,
                                 uint32_t iters, uint64_t budget,
                                 uint32_t* timeout) {
  float* my_slot_here = self_inbox + (size_t)rank * n;
  float* my_slot_there = peer_inbox + (size_t)rank * n;
  for (uint32_t it = 1; it <= iters; ++it) {
    // 1. publicar: una escritura local y una escritura AL PEER.
    // 16-byte stores: a 4-byte store per float turns into one tiny TLP each
    // and wastes most of the link.
    const uint32_t n4 = n / 4;
    const float4* in4 = (const float4*)in;
    float4* here4 = (float4*)my_slot_here;
    float4* there4 = (float4*)my_slot_there;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x) {
      float4 v = in4[i];
      here4[i] = v;
      there4[i] = v;
    }
    // 2. que los datos salgan antes que la bandera.
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    if (!barrier(self_sig->start, peer_sig->start, rank, it, budget)) {
      if (threadIdx.x == 0) atomicAdd(timeout, 1u);
      return;
    }
    // 2b. si el buzon es cacheado, invalidar antes de leer lo que llego por
    // PCIe; si no, una linea vieja de L2 nos da el valor de la iteracion
    // anterior.
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    // 3. reducir leyendo SOLO memoria propia.
    const float4* box4 = (const float4*)self_inbox;
    float4* out4 = (float4*)out;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += gridDim.x * blockDim.x) {
      float4 acc = box4[i];
#pragma unroll
      for (int s = 1; s < NGPUS; ++s) {
        float4 o = box4[(size_t)s * n4 + i];
        acc.x += o.x; acc.y += o.y; acc.z += o.z; acc.w += o.w;
      }
      out4[i] = acc;
    }
    if (!barrier(self_sig->end, peer_sig->end, rank, it, budget)) {
      if (threadIdx.x == 0) atomicAdd(timeout, 1u);
      return;
    }
  }
}

__global__ void k_seed(float* p, uint32_t n, int rank) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    p[i] = (float)(rank + 1) * 1000.0f + (float)(i % 997);
}
// esperado = suma sobre ranks de la semilla, + sesgo para el control que falla
__global__ void k_verify(const float* out, uint32_t n, float bias, uint32_t* c) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float exp = 0.f;
    for (int r = 0; r < NGPUS; ++r) exp += (float)(r + 1) * 1000.0f + (float)(i % 997);
    exp += bias;
    float v = out[i];
    if (v != exp) {
      atomicAdd(&c[0], 1u);
      if (v == 0.f) atomicAdd(&c[1], 1u);
      if (atomicAdd(&c[2], 1u) == 0) {
        ((float*)c)[3] = v;
      }
    }
  }
}

int main(int argc, char** argv) {
  uint32_t n = (argc > 1) ? (uint32_t)atoi(argv[1]) : (1u << 20);  // floats
  uint32_t iters = (argc > 2) ? (uint32_t)atoi(argv[2]) : 200;
  int cached_inbox = (argc > 3) ? atoi(argv[3]) : 0;
  size_t bytes = (size_t)n * 4;
  printf("all-reduce de empuje: %u floats (%zu KiB) x %u iteraciones, buzon %s\n",
         n, bytes >> 10, iters, cached_inbox ? "CACHEADO" : "uncached");

  CK(hipSetDevice(0));
  CK(hipDeviceEnablePeerAccess(1, 0));
  CK(hipSetDevice(1));
  CK(hipDeviceEnablePeerAccess(0, 0));
  int wcr = 0;
  CK(hipDeviceGetAttribute(&wcr, hipDeviceAttributeWallClockRate, 0));
  if (wcr <= 0) wcr = 100000;
  uint64_t budget = (uint64_t)wcr * 1000ull * 3ull;

  float *in[2], *out[2], *inbox[2];
  Signal* sig[2];
  uint32_t *cn[2], *to[2];
  hipStream_t st[2];
  for (int d = 0; d < 2; ++d) {
    CK(hipSetDevice(d));
    void* p;
    CK(hipMalloc(&p, bytes));
    in[d] = (float*)p;
    CK(hipMalloc(&p, bytes));
    out[d] = (float*)p;
    if (cached_inbox)
      CK(hipMalloc(&p, bytes * NGPUS));
    else
      CK(hipExtMallocWithFlags(&p, bytes * NGPUS, hipDeviceMallocUncached));
    inbox[d] = (float*)p;
    CK(hipMemset(inbox[d], 0, bytes * NGPUS));
    CK(hipExtMallocWithFlags(&p, sizeof(Signal), hipDeviceMallocUncached));
    sig[d] = (Signal*)p;
    CK(hipMemset(sig[d], 0, sizeof(Signal)));
    CK(hipMalloc(&cn[d], 16));
    CK(hipMalloc(&to[d], 4));
    CK(hipMemset(to[d], 0, 4));
    CK(hipStreamCreate(&st[d]));
    k_seed<<<NBLK, NTHR>>>(in[d], n, d);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
  }

  auto once = [&](const char* name, float bias) {
    for (int d = 0; d < 2; ++d) {
      CK(hipSetDevice(d));
      CK(hipMemset(sig[d], 0, sizeof(Signal)));
      CK(hipMemset(to[d], 0, 4));
      CK(hipMemset(out[d], 0, bytes));
      CK(hipDeviceSynchronize());
    }
    hipEvent_t e0, e1;
    CK(hipSetDevice(0));
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0, st[0]));
    for (int d = 0; d < 2; ++d) {
      CK(hipSetDevice(d));
      k_push_allreduce<<<NBLK, NTHR, 0, st[d]>>>(
          in[d], out[d], n, inbox[d], inbox[1 - d], sig[d], sig[1 - d], d,
          iters, budget, to[d]);
      CK(hipGetLastError());
    }
    CK(hipSetDevice(0));
    CK(hipEventRecord(e1, st[0]));
    for (int d = 0; d < 2; ++d) {
      CK(hipSetDevice(d));
      CK(hipStreamSynchronize(st[d]));
    }
    float ms = 0;
    CK(hipSetDevice(0));
    CK(hipEventElapsedTime(&ms, e0, e1));

    uint32_t tsum = 0, h[4];
    for (int d = 0; d < 2; ++d) {
      uint32_t t;
      CK(hipSetDevice(d));
      CK(hipMemcpy(&t, to[d], 4, hipMemcpyDeviceToHost));
      tsum += t;
    }
    printf("  %-44s ", name);
    if (tsum) {
      printf("TIMEOUT (la barrera no progresa)\n");
      return;
    }
    uint32_t bad_total = 0;
    for (int d = 0; d < 2; ++d) {
      CK(hipSetDevice(d));
      CK(hipMemset(cn[d], 0, 16));
      k_verify<<<NBLK, NTHR>>>(out[d], n, bias, cn[d]);
      CK(hipGetLastError());
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(h, cn[d], 16, hipMemcpyDeviceToHost));
      bad_total += h[0];
      if (h[0])
        printf("[dev%d ROTO %u/%u ceros=%u primero=%.1f] ", d, h[0], n, h[1],
               ((float*)h)[3]);
    }
    if (!bad_total) {
      double gb = (double)bytes * iters / 1e9;
      printf("ok   %.2f ms  (%.2f GB/s de carga util por rank)", ms,
             gb / (ms / 1e3));
    }
    printf("\n");
  };

  once("all-reduce de empuje (bias 0 = correcto)", 0.f);
  once("CTRL el arnes SABE fallar (bias 1.0)", 1.f);
  return 0;
}
