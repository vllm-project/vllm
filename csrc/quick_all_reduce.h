#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <vector>
#include <torch/torch.h>

typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

#define __device_inline__ __device__ __forceinline__
#define __quickreduce_launch_bounds__ __launch_bounds__(256, 4)

// Setup acquire-release semantics for vector memory reads (mubuf instruction)
// as per architecture.
#if defined(__gfx942__)
  // CDNA3: Scope bits sc0, sc1
  #define MUBUF_ACQUIRE 16
  #define MUBUF_RELEASE 16
#elif (defined(__gfx908__) || defined(__gfx90a__))
  // CDNA1 and CDNA2 - glc bit
  #define MUBUF_ACQUIRE 1
  #define MUBUF_RELEASE 0
#endif

namespace quickreduce {

// Vector types
using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

using fp8_t = uint8_t;
using fp8x8_t = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;

using fp16x4_t = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using fp16x8_t = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using fp16x16_t = __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16;

using fp32x2_t = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using fp32x4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using fp32x8_t = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using fp32x16_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;

// Standard CDNA wavefront size.
static int constexpr kWavefront = 64;

// 256 thread, 4 wavefronts.
static dim3 constexpr kBlock = {64, 4, 1};

// Methods
__device_inline__ __host__ int divceil(int x, int y) {
  return ((x + y - 1) / y);
}

__device_inline__ __host__ constexpr int divceil_constexpr(int const x,
                                                           int const y) {
  return ((x + y - 1) / y);
}

/*
===============================================================
Desc:
    Utility container to describe the Buffer Resource used in VMEM operations.

Operation:
    BufferResource can be initialized to tensor base address and range/size (in
bytes). The range is used for OOB checks. For example the range for a MxK
dtype=fp16 tensor would have a range of [M * K * sizeof(half)].

    The last dword of the buffer resource description is to a default config
with DFMT=32b.

    Instructions that used the buffer resource (buffer_load/store_dwordx4) wait
on the `vmcnt`.
*/

union BufferResource {
  __device_inline__ constexpr BufferResource() : config(0x00020000U) {}

  __device_inline__ constexpr BufferResource(void* buffer_address,
                                             uint32_t buffer_size)
      : address(buffer_address), range(buffer_size), config(0x00020000U) {}

  int32x4_t descriptor;
  struct {
    void* address;   // 8B, out of which first 48b is address, and 16b is stride
                     // (unused)
    uint32_t range;  // Byte range for the buffer resource
    uint32_t config;  // Constant, DFMT=32b
  };
};

__device_inline__ static int32x4_t buffer_load_dwordx4(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device_inline__ static void buffer_store_dwordx4(
    int32x4_t data, int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device_inline__ static void set_fp16_ovfl(bool const value) {
  // short size = 0b00001;    // Specifies the bit size to modify
  // const short offset = 0b10111;  // Corrected offset to 23, which is the bit
  // position of FP16_OVFL const short hwRegId = 0b000001; // HW register ID for
  // MODE const short simm16 = (size << 11) | (offset << 6) | hwRegId; simm16 =
  // 0xdc1

#if defined(__gfx942__)
  if (value) {
    asm volatile("s_setreg_imm32_b32 0xdc1, 1;" ::);
  } else {
    asm volatile("s_setreg_imm32_b32 0xdc1, 0;" ::);
  }
#endif
}

#define HIP_CHECK(err)                                                  \
  do {                                                                  \
    hipError_t err_ = (err);                                            \
    if (err_ != hipSuccess) {                                           \
      std::printf("HIP error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("HIP error");                            \
    }                                                                   \
  } while (0)

enum struct QuickReduceProfile {
  ONESHOT_F16 = 0,
  TWOSHOT_F16 = 1,
  TWOSHOT_FP8 = 2,
  TWOSHOT_Q8 = 3,
  TWOSHOT_Q6 = 4,
  TWOSHOT_Q4 = 5,
};

/*
===============================================================
Desc:
    Device Comms Handle
*/
struct DeviceComms {
  // Workgroup scope = Tile = (256 threads x 16B x 8 atoms)
  static long constexpr kTileSize = 256 * 16 * 8;

  // Max problem size is 512MB (in bytes)
  static long constexpr kMaxProblemSize = 536870912;
  static long constexpr kMaxTiles = kMaxProblemSize / kTileSize;

  // Max TP-8
  static int constexpr kMaxWorldSize = 8;

  bool initialized = false;
  int flag_color = 1;
  int world_size;
  int rank;

  uint8_t* dbuffer;
  uint8_t** dbuffer_list;
  hipIpcMemHandle_t buffer_ipc_handle;
  std::vector<hipIpcMemHandle_t> all_buffer_ipc_handles;
  std::vector<uint8_t*> buffer_list;

  long data_offset;

  DeviceComms() : initialized(false), world_size(1), rank(0) {}
  ~DeviceComms() { destroy(); }

  void init(int world_size, int rank);
  int get_world_size() { return world_size; }
  int get_rank() { return rank; }
  bool status() { return initialized; }
  void destroy();

  hipIpcMemHandle_t const get_handle() { return buffer_ipc_handle; }
  void open_ipc_handles(std::vector<hipIpcMemHandle_t> const& ipc_handles);
  template <typename T>
  void allreduce(int profile, hipStream_t stream, T const* A, T* B, int N);
  torch::Tensor qr_get_comm_handle();
};

// Function Template for two dtypes
union bf162_int_union {
  int i;
  nv_bfloat162 bf2;
};

// packed add
template <typename T>
__device_inline__ int pk_add(int a, int b);

template <>
__device_inline__ int pk_add<half>(int a, int b) {
  int res;
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(res) : "v"(a), "v"(b));
  return res;
}

template <>
__device_inline__ int pk_add<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hadd2(A.bf2, B.bf2);
  return R.i;
}

// packed max
template <typename T>
__device_inline__ int pk_max(int a, int b);

template <>
__device_inline__ int pk_max<half>(int a, int b) {
  int res;
  asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(res) : "v"(a), "v"(b));
  return res;
}

template <>
__device_inline__ int pk_max<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hmax2(A.bf2, B.bf2);
  return R.i;
}

// packed min
template <typename T>
__device_inline__ int pk_min(int a, int b);

template <>
__device_inline__ int pk_min<half>(int a, int b) {
  int res;
  asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(res) : "v"(a), "v"(b));
  return res;
}

template <>
__device_inline__ int pk_min<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2 = __hmin2(A.bf2, B.bf2);
  return R.i;
}

// pk_max_abs
template <typename T>
__device_inline__ int pk_max_abs(int a, int b);

template <>
__device_inline__ int pk_max_abs<half>(int a, int b) {
  half2 wmaxh2 = __builtin_bit_cast(half2, a);
  half2 wminh2 = __builtin_bit_cast(half2, b);
  half2 wblockmaxh2;

  wblockmaxh2.x =
      __half2float(__habs(wmaxh2.x)) > __half2float(__habs(wminh2.x))
          ? wmaxh2.x
          : wminh2.x;
  wblockmaxh2.y =
      __half2float(__habs(wmaxh2.y)) > __half2float(__habs(wminh2.y))
          ? wmaxh2.y
          : wminh2.y;
  return __builtin_bit_cast(int, wblockmaxh2);
}

template <>
__device_inline__ int pk_max_abs<nv_bfloat16>(int a, int b) {
  bf162_int_union A, B, R;
  A.i = a;
  B.i = b;
  R.bf2.x =
      __bfloat162float(__habs(A.bf2.x)) > __bfloat162float(__habs(B.bf2.x))
          ? A.bf2.x
          : B.bf2.x;
  R.bf2.y =
      __bfloat162float(__habs(A.bf2.y)) > __bfloat162float(__habs(B.bf2.y))
          ? A.bf2.y
          : B.bf2.y;
  return R.i;
}

// pk_mul
template <typename T>
__device_inline__ int pk_mul(int a, int b);

template <>
__device_inline__ int pk_mul<half>(int a, int b) {
  int res;
  asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(res) : "v"(a), "v"(b));
  return res;
}

template <>
__device_inline__ int pk_mul<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hmul2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

// pk_hcp
template <typename T>
__device_inline__ int pk_hcp(int a);

template <>
__device_inline__ int pk_hcp<half>(int a) {
  return __builtin_bit_cast(int, h2rcp(__builtin_bit_cast(half2, a)));
}

template <>
__device_inline__ int pk_hcp<nv_bfloat16>(int a) {
  bf162_int_union A, R;
  A.i = a;
  R.bf2 = h2rcp(A.bf2);
  return R.i;
}

// changes dtype
__device_inline__ float T2float(half a) { return __half2float(a); }

__device_inline__ float T2float(nv_bfloat16 a) { return __bfloat162float(a); }

// const Q8
template <typename T>
struct Quant8Const;

template <>
struct Quant8Const<half> {
  static constexpr int kScaleFactor =
      0xA000A000;  // {-1/128.0h, -1/128.0h}, fp16x2_t
  static constexpr int kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static constexpr int kRangeMin = 0xD800D800;      // {-128, -128}, fp16x2_t
  static constexpr int kRangeMax = 0x57F057F0;      // {+127, +127}, fp16x2_t
};

template <>
struct Quant8Const<nv_bfloat16> {
  static constexpr int kScaleFactor =
      0xBC00BC00;  // {-1/128.0h, -1/128.0h}, fp16x2_t
  static constexpr int kScaleEpsilon = 0x33D733D7;  // {1e-7, 1e-7}, fp16x2_t
  static constexpr int kRangeMin = 0xC300C300;      // {-128, -128}, fp16x2_t
  static constexpr int kRangeMax = 0x42FE42FE;      // {+127, +127}, fp16x2_t
};
// const Q6
template <typename T>
struct Quant6Const;

template <>
struct Quant6Const<half> {
  static int constexpr kScaleFactor =
      0xA800A800;  // {-1/32.0h, -1/32.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xD000D000;      // {-32, -32}, fp16x2_t
  static int constexpr kRangeMax = 0x4FC04FC0;      // {+31, +31}, fp16x2_t
};

template <>
struct Quant6Const<nv_bfloat16> {
  static int constexpr kScaleFactor =
      0xBD00BD00;  // {-1/32.0h, -1/32.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x33D733D7;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xC200C200;      // {-32, -32}, fp16x2_t
  static int constexpr kRangeMax = 0x41F841F8;      // {+31, +31}, fp16x2_t
};

// const Q4
template <typename T>
struct Quant4Const;

template <>
struct Quant4Const<half> {
  static int constexpr kScaleFactor =
      0xB000B000;  // {-1/8.0h, -1/8.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xC800C800;      // {-8, -8}, fp16x2_t
  static int constexpr kRangeMax = 0x47004700;      // {+7, +7}, fp16x2_t
};

template <>
struct Quant4Const<nv_bfloat16> {
  static int constexpr kScaleFactor =
      0xBE00BE00;  // {-1/8.0h, -1/8.0h}, fp16x2_t
  static int constexpr kScaleEpsilon = 0x33D733D7;  // {1e-7, 1e-7}, fp16x2_t
  static int constexpr kRangeMin = 0xC100C100;      // {-8, -8}, fp16x2_t
  static int constexpr kRangeMax = 0x40E040E0;      // {+7, +7}, fp16x2_t
};

// const fp8
template <typename T>
struct Quantfp8Const;

template <>
struct Quantfp8Const<half> {
  static int constexpr kScaleFactor = 0x1C441C44;   // {1/240.0h, 1/240.0h}
  static int constexpr kScaleEpsilon = 0x00010001;  // {1e-7, 1e-7}
};

template <>
struct Quantfp8Const<nv_bfloat16> {
  static int constexpr kScaleFactor = 0x3B883B88;   // {1/240.0h, 1/240.0h}  bf
  static int constexpr kScaleEpsilon = 0x33D733D7;  // {1e-7, 1e-7} bf
};
}  // namespace quickreduce

// /*
// ===============================================================
// API
// */
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));
fptr_t init_quick_ar(int64_t world_size, int64_t rank);
torch::Tensor qr_get_comm_handle(fptr_t _fa);
void qr_set_comm_handles(fptr_t _fa,
                         std::vector<torch::Tensor> const& comm_handles);
void qr_all_reduce(fptr_t _fa, int64_t profile, torch::Tensor const& inp,
                   torch::Tensor& out);
void qr_destroy(fptr_t _fa);
void is_quickreduce_available();
