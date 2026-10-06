// QuickReduce builds its buffer resource descriptor with word3 = 0x00020000,
// which is the gfx9 encoding (DATA_FORMAT=32). RDNA replaced those fields.
// Load the same array through both encodings on gfx1100 and see which one
// returns the data.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

typedef int __attribute__((ext_vector_type(4))) int32x4_t;

__device__ static int32x4_t buffer_load_dwordx4(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

union BufferResource {
  __device__ constexpr BufferResource(void* a, uint32_t r, uint32_t cfg)
      : address(a), range(r), config(cfg) {}
  int32x4_t descriptor;
  struct {
    void* address;
    uint32_t range;
    uint32_t config;
  };
};

__global__ void k(const int* src, int* dst, uint32_t nbytes, uint32_t cfg) {
  BufferResource b(const_cast<int*>(src), nbytes, cfg);
  uint32_t off = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
  if (off < nbytes) {
    int32x4_t v = buffer_load_dwordx4(b.descriptor, off, 0, 0);
    int4* o = (int4*)(dst + off / 4);
    *o = make_int4(v.x, v.y, v.z, v.w);
  }
}

int main() {
  const uint32_t n = 4096;
  size_t bytes = n * 4;
  int* h = (int*)malloc(bytes);
  for (uint32_t i = 0; i < n; ++i) h[i] = (int)(i * 2654435761u);
  int *d, *o;
  hipMalloc(&d, bytes);
  hipMalloc(&o, bytes);
  hipMemcpy(d, h, bytes, hipMemcpyHostToDevice);

  hipDeviceProp_t p;
  hipGetDeviceProperties(&p, 0);
  printf("arch=%s  warpSize=%d\n", p.gcnArchName, p.warpSize);

  struct {
    const char* name;
    uint32_t cfg;
  } cases[] = {
      {"0x00020000  (lo que usa QuickReduce: gfx9)", 0x00020000u},
      {"0x31014000  (encoding RDNA gfx10/11)", 0x31014000u},
      {"0x00027000  (variante gfx9 con DST_SEL)", 0x00027000u},
  };
  int* back = (int*)malloc(bytes);
  for (auto& c : cases) {
    hipMemset(o, 0, bytes);
    k<<<n / 4 / 64, 64>>>(d, o, bytes, c.cfg);
    hipError_t e = hipDeviceSynchronize();
    hipMemcpy(back, o, bytes, hipMemcpyDeviceToHost);
    uint32_t bad = 0, zero = 0;
    for (uint32_t i = 0; i < n; ++i) {
      if (back[i] != h[i]) {
        ++bad;
        if (back[i] == 0) ++zero;
      }
    }
    printf("  %-44s %s", c.name, bad ? "ROTO " : "ok");
    if (bad) printf(" %u/%u  ceros=%u  err=%s", bad, n, zero, hipGetErrorString(e));
    printf("\n");
  }
  return 0;
}
