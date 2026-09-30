// Pin the semantics of ds_read_b64_tr_b4 (gfx950 4-bit LDS transpose load).
//
// Each lane supplies an 8-byte-aligned LDS address and gets 64 bits back. The
// hypothesis is a 16 x 16 nibble transpose within each 16-lane group: lane i
// receives nibble i of each of the group's 16 loaded 64-bit words. Rather than
// trust that, the probe fills LDS with random bytes over several trials and
// solves, for every (lane, output nibble), which (source lane, source nibble)
// it always equals.
//
// Lane l loads 8 bytes at row l of a [64][16]-byte LDS array.
//
// Build: hipcc --offload-arch=gfx950 -O2 hip_ds_read_tr4_probe.hip -o probe
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

constexpr int TRIALS = 10, ROWB = 16;
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int v2i;

__global__ void probe(const uint8_t* src, uint32_t* out) {
  __shared__ __attribute__((aligned(16))) uint8_t lds[64 * ROWB];
  const int l = threadIdx.x;
  for (int t = 0; t < TRIALS; ++t) {
    for (int b = 0; b < ROWB; ++b) lds[l * ROWB + b] = src[(t * 64 + l) * ROWB + b];
    __syncthreads();
    v2i v = __builtin_amdgcn_ds_read_tr4_b64_v2i32(
        (__attribute__((address_space(3))) v2i*)(lds + l * ROWB));
    out[(t * 64 + l) * 2 + 0] = v[0];
    out[(t * 64 + l) * 2 + 1] = v[1];
    __syncthreads();
  }
}

int main() {
  static uint8_t src[TRIALS * 64 * ROWB];
  static uint32_t out[TRIALS * 64 * 2];
  srand(3);
  for (auto& x : src) x = rand() & 0xFF;
  uint8_t* ds; uint32_t* dout;
  (void)hipMalloc(&ds, sizeof src); (void)hipMalloc(&dout, sizeof out);
  (void)hipMemcpy(ds, src, sizeof src, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, ds, dout);
  if (hipDeviceSynchronize() != hipSuccess) { printf("launch failed\n"); return 2; }
  (void)hipMemcpy(out, dout, sizeof out, hipMemcpyDeviceToHost);

  // Source nibble q (0..15) of lane k: the 8 bytes it loaded, low nibble first.
  auto srcnib = [&](int t, int k, int q) {
    uint8_t b = src[(t * 64 + k) * ROWB + q / 2];
    return (q & 1) ? (b >> 4) : (b & 15);
  };
  auto outnib = [&](int t, int l, int p) {
    return (out[(t * 64 + l) * 2 + p / 8] >> (4 * (p % 8))) & 15u;
  };
  int unique = 0, ambiguous = 0, none = 0, hyp_ok = 0;
  for (int l = 0; l < 64; ++l)
    for (int p = 0; p < 16; ++p) {
      int found = -1, count = 0;
      for (int k = 0; k < 64; ++k)
        for (int q = 0; q < 16; ++q) {
          bool all = true;
          for (int t = 0; t < TRIALS && all; ++t) all = srcnib(t, k, q) == outnib(t, l, p);
          if (all) { found = k * 16 + q; ++count; }
        }
      if (count == 1) {
        ++unique;
        const int k = found / 16, q = found % 16;
        // Hypothesis: lane l, output nibble p <- lane 16*(l/16) + p, nibble l%16.
        if (k == 16 * (l / 16) + p && q == l % 16) ++hyp_ok;
        if (l == 0 || l == 1 || l == 17) printf("lane %2d nibble %2d <- lane %2d nibble %2d\n", l, p, k, q);
      } else if (count == 0) ++none; else ++ambiguous;
    }
  printf("mappings: %d unique, %d ambiguous, %d none (of 1024)\n", unique, ambiguous, none);
  printf("16x16 nibble-transpose hypothesis holds for %d / 1024\n", hyp_ok);
  return hyp_ok == 1024 ? 0 : 1;
}
