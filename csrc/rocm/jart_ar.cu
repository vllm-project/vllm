// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// All-reduce TP4 para gfx1100 sin barreras: protocolo LL (dato + etiqueta en el
// mismo paquete de 64 bits: 3 valores de 16 bits y una etiqueta de 16).
//
// Cada GPU ESCRIBE en la memoria (sin cache) de las demas y SONDEA la suya: no hay
// lecturas remotas (ida y vuelta por PCIe), ni handshakes de banderas, ni fences de
// sistema. El receptor sabe que el paquete ha llegado porque la etiqueta coincide.
//
//   algo 0 = oneshot: cada rank manda su vector a los 3; cada uno suma los 4.
//   algo 1 = rsag:    reduce-scatter + all-gather. El dueno de cada paquete recibe 3,
//                     suma y reenvia el resultado a los 3. Mitad de bytes, 2 latencias.
//   algo 3 = rsag LL128-lite (16 B: 7 valores + etiqueta) · algo 5 = oneshot LL128-lite
//   algo 6 = auto (1 hasta 12k elementos, 3 por encima) · 8 = ping · 9 = kernel vacio
//   algo 2 = rsag_simple: lo mismo con datos EN CRUDO (1,5x el tamano, el minimo de
//                     un all-reduce) y UNA bandera por bloque y fase en vez de una
//                     etiqueta por paquete. En la caja local cada GPU cuelga a Gen4 x4
//                     (7,1 GB/s): ahi manda cuantos bytes cruzan el cable.
//
// Suma siempre en fp32 y en orden de rank 0..3 => resultado IDENTICO en los 4 ranks.
//
// Buffer por rank (uncached, IPC): [cabecera: epoch por bloque][datos]
// datos = paridad(2) x fase(2) x origen(4) x kMaxPk paquetes de 8 bytes.
// Paridad por llamada: un rank no puede ir 2 llamadas por delante de otro, porque
// para terminar la llamada i+1 necesita los paquetes i+1 del otro, que este solo
// escribe tras terminar la i. Cada paquete consumido se pone a 0: tras dar la vuelta
// la etiqueta de 16 bits, un hueco no escrito en una llamada pequena no puede
// casar con un resto viejo.

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <tuple>
#include <vector>

#define HIP_CHECK(cmd)                                                   \
  do {                                                                   \
    hipError_t e = (cmd);                                                \
    TORCH_CHECK(e == hipSuccess, "HIP: ", #cmd, " -> ", hipGetErrorString(e)); \
  } while (0)

namespace jart_ar {

using fptr_t = int64_t;
constexpr int kWorld = 4;
constexpr int kMaxBlocks = 128;
constexpr int kThreads = 256;
constexpr int64_t kMaxNumel = 256 * 1024;
constexpr int64_t kMaxPk = ((kMaxNumel + 2) / 3 + 255) / 256 * 256;
constexpr int64_t kHeader = 16384;
constexpr int64_t kBytes = kHeader + 2LL * 2 * kWorld * kMaxPk * 8;

struct Ptrs {
  char* p[kWorld];
  uint32_t* ep;   // epoch por bloque: LOCAL y en memoria normal (con cache), no en la IPC
  float* parc;    // ar_rms7g: suma de cuadrados parcial por onda de paquetes [onda][2]
  uint32_t* bar;  // ar_rms7g: barrera de grid [cuenta, generacion]
};

struct Ctx {
  int rank;
  Ptrs ptrs;
};

__device__ __forceinline__ uint64_t* slot(char* base, int par, int fase, int src) {
  return reinterpret_cast<uint64_t*>(base + kHeader) +
         ((static_cast<int64_t>(par) * 2 + fase) * kWorld + src) * kMaxPk;
}

__device__ __forceinline__ void put(uint64_t* dst, uint64_t v) {
  __builtin_nontemporal_store(v, dst);
}

__device__ __forceinline__ uint64_t get(uint64_t* src, uint32_t tag) {
  uint64_t v;
  do {
    asm volatile(
        "global_load_b64 %0, %1, off glc slc dlc\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v)
        : "v"(src)
        : "memory");
  } while (static_cast<uint32_t>(v >> 48) != tag);
  *reinterpret_cast<volatile uint64_t*>(src) = 0;
  return v & 0xffffffffffffULL;
}

// Sondea los paquetes de los 3 pares A LA VEZ: una espera de memoria por ronda,
// no una por par. Devuelve los 3 datos (sin etiqueta) en orden de rank.
__device__ __forceinline__ void get_otros(uint64_t* base0, uint64_t* base1, uint64_t* base2,
                                          uint32_t tag, uint64_t* v) {
  uint64_t a, b, c;
  bool fa = false, fb = false, fc = false;
  while (true) {
    asm volatile(
        "global_load_b64 %0, %3, off glc slc dlc\n"
        "global_load_b64 %1, %4, off glc slc dlc\n"
        "global_load_b64 %2, %5, off glc slc dlc\n"
        "s_waitcnt vmcnt(0)\n"
        : "=&v"(a), "=&v"(b), "=&v"(c)
        : "v"(base0), "v"(base1), "v"(base2)
        : "memory");
    if (!fa && static_cast<uint32_t>(a >> 48) == tag) { fa = true; v[0] = a; }
    if (!fb && static_cast<uint32_t>(b >> 48) == tag) { fb = true; v[1] = b; }
    if (!fc && static_cast<uint32_t>(c >> 48) == tag) { fc = true; v[2] = c; }
    if (fa && fb && fc) break;
  }
  *reinterpret_cast<volatile uint64_t*>(base0) = 0;
  *reinterpret_cast<volatile uint64_t*>(base1) = 0;
  *reinterpret_cast<volatile uint64_t*>(base2) = 0;
#pragma unroll
  for (int k = 0; k < 3; ++k) v[k] &= 0xffffffffffffULL;
}

template <typename T>
__device__ __forceinline__ uint64_t load3(const T* in, int i, int numel) {
  uint16_t h[3] = {0, 0, 0};
#pragma unroll
  for (int e = 0; e < 3; ++e)
    if (3 * i + e < numel) h[e] = reinterpret_cast<const uint16_t*>(in)[3 * i + e];
  return uint64_t(h[0]) | (uint64_t(h[1]) << 16) | (uint64_t(h[2]) << 32);
}

__device__ __forceinline__ float f(__half x) { return __half2float(x); }
__device__ __forceinline__ float f(__hip_bfloat16 x) { return __bfloat162float(x); }
template <typename T>
__device__ __forceinline__ T t(float x);
template <>
__device__ __forceinline__ __half t<__half>(float x) { return __float2half(x); }
template <>
__device__ __forceinline__ __hip_bfloat16 t<__hip_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <typename T>
__device__ __forceinline__ void acc3(float* a, uint64_t b) {
#pragma unroll
  for (int e = 0; e < 3; ++e) {
    uint16_t h = static_cast<uint16_t>(b >> (16 * e));
    a[e] += f(*reinterpret_cast<T*>(&h));
  }
}

template <typename T>
__device__ __forceinline__ uint64_t pack3(const float* a) {
  uint64_t r = 0;
#pragma unroll
  for (int e = 0; e < 3; ++e) {
    T v = t<T>(a[e]);
    r |= uint64_t(*reinterpret_cast<uint16_t*>(&v)) << (16 * e);
  }
  return r;
}

template <typename T>
__device__ __forceinline__ void store3(T* out, int i, int numel, uint64_t b) {
#pragma unroll
  for (int e = 0; e < 3; ++e)
    if (3 * i + e < numel)
      reinterpret_cast<uint16_t*>(out)[3 * i + e] = static_cast<uint16_t>(b >> (16 * e));
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    oneshot(Ptrs P, const T* __restrict__ in, T* __restrict__ out, int rank, int numel) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const uint64_t tg = uint64_t(tag) << 48;
  const int par = epoch & 1;
  const int pk = (numel + 2) / 3;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int i = first; i < pk; i += stride) {
    const uint64_t v = tg | load3(in, i, numel);
#pragma unroll
    for (int d = 1; d < kWorld; ++d) put(slot(P.p[(rank + d) & 3], par, 0, rank) + i, v);
  }
  for (int i = first; i < pk; i += stride) {
    float a[3] = {0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      acc3<T>(a, s == rank ? load3(in, i, numel) : get(slot(P.p[rank], par, 0, s) + i, tag));
    store3(out, i, numel, pack3<T>(a));
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    rsag(Ptrs P, const T* __restrict__ in, T* __restrict__ out, int rank, int numel) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const uint64_t tg = uint64_t(tag) << 48;
  const int par = epoch & 1;
  const int pk = (numel + 2) / 3;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  // dueno del paquete i: por hilo global, asi cada hilo tiene 1/4 de sus paquetes
  // como dueno y el reparto no depende de la forma
  for (int i = first; i < pk; i += stride) {
    const int own = (i / kThreads) & 3;
    if (own != rank) put(slot(P.p[own], par, 0, rank) + i, tg | load3(in, i, numel));
  }
  for (int i = first; i < pk; i += stride) {
    if (((i / kThreads) & 3) != rank) continue;
    uint64_t otros[3];
    get_otros(slot(P.p[rank], par, 0, (rank + 1) & 3) + i,
              slot(P.p[rank], par, 0, (rank + 2) & 3) + i,
              slot(P.p[rank], par, 0, (rank + 3) & 3) + i, tag, otros);
    float a[3] = {0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      acc3<T>(a, s == rank ? load3(in, i, numel) : otros[(s - rank + 3) & 3]);
    const uint64_t r = pack3<T>(a);
    store3(out, i, numel, r);
#pragma unroll
    for (int d = 1; d < kWorld; ++d) put(slot(P.p[(rank + d) & 3], par, 1, rank) + i, tg | r);
  }
  for (int i = first; i < pk; i += stride) {
    const int own = (i / kThreads) & 3;
    if (own != rank) store3(out, i, numel, get(slot(P.p[rank], par, 1, own) + i, tag));
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

using v4 = __attribute__((__vector_size__(4 * sizeof(int)))) int;

__device__ __forceinline__ uint32_t* bandera(char* base, int fase, int src, int blk) {
  return reinterpret_cast<uint32_t*>(base + 4096) + (fase * kWorld + src) * kMaxBlocks + blk;
}

template <typename T>
__device__ __forceinline__ void suma8(float* a, v4 b) {
  const T* h = reinterpret_cast<const T*>(&b);
#pragma unroll
  for (int e = 0; e < 8; ++e) a[e] += f(h[e]);
}

template <typename T>
__device__ __forceinline__ v4 empaca8(const float* a) {
  v4 r;
  T* h = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int e = 0; e < 8; ++e) h[e] = t<T>(a[e]);
  return r;
}

// Publica las escrituras de TODO el bloque y luego la bandera en los destinos.
__device__ __forceinline__ void publica(Ptrs& P, int rank, int fase, uint32_t valor,
                                        int solo) {
  asm volatile("s_waitcnt_vscnt null, 0" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int d = 1; d < kWorld; ++d) {
      const int dst = (rank + d) & 3;
      if (solo >= 0 && dst != solo) continue;
      __scoped_atomic_store_n(bandera(P.p[dst], fase, rank, blockIdx.x), valor,
                              __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }
  }
}

__device__ __forceinline__ void espera(char* self, int fase, int src, uint32_t valor) {
  if (threadIdx.x == 0) {
    while (__scoped_atomic_load_n(bandera(self, fase, src, blockIdx.x), __ATOMIC_ACQUIRE,
                                  __MEMORY_SCOPE_SYSTEM) < valor) {
    }
  }
  __syncthreads();
}

// Unidades de 16 B (8 valores). Cuarto q = unidades del dueno q; dentro de cada cuarto
// el bloque b lleva su tramo contiguo. Mismo reparto en los 4 ranks.
template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    rsag_simple(Ptrs P, const T* __restrict__ in, T* __restrict__ out, int rank, int numel) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t valor = epoch + 1;
  const int par = epoch & 1;
  const int U = numel / 8;
  const int cuarto = (U + 3) / 4;
  const int tramo = (cuarto + gridDim.x - 1) / gridDim.x;
  const v4* vin = reinterpret_cast<const v4*>(in);
  v4* vout = reinterpret_cast<v4*>(out);
  auto rango = [&](int q, int& a, int& b) {
    a = min(q * cuarto + blockIdx.x * tramo, min((q + 1) * cuarto, U));
    b = min(a + tramo, min((q + 1) * cuarto, U));
  };
  // fase 0: mi aportacion al cuarto de cada dueno, en crudo
  for (int d = 1; d < kWorld; ++d) {
    const int q = (rank + d) & 3;
    int a, b;
    rango(q, a, b);
    v4* dst = reinterpret_cast<v4*>(slot(P.p[q], par, 0, rank));
    for (int i = a + threadIdx.x; i < b; i += kThreads)
      __builtin_nontemporal_store(vin[i], dst + i);
  }
  publica(P, rank, 0, valor, -1);
  // mi cuarto: sumar las 4 aportaciones en orden de rank y repartir el resultado
  int a, b;
  rango(rank, a, b);
  for (int s = 1; s < kWorld; ++s) espera(P.p[rank], 0, (rank + s) & 3, valor);
  for (int i = a + threadIdx.x; i < b; i += kThreads) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      suma8<T>(acc, s == rank ? vin[i]
                              : __builtin_nontemporal_load(
                                    reinterpret_cast<v4*>(slot(P.p[rank], par, 0, s)) + i));
    const v4 r = empaca8<T>(acc);
    vout[i] = r;
#pragma unroll
    for (int d = 1; d < kWorld; ++d)
      __builtin_nontemporal_store(
          r, reinterpret_cast<v4*>(slot(P.p[(rank + d) & 3], par, 1, rank)) + i);
  }
  publica(P, rank, 1, valor, -1);
  // los otros 3 cuartos llegan ya reducidos de su dueno
  for (int d = 1; d < kWorld; ++d) {
    const int q = (rank + d) & 3;
    rango(q, a, b);
    espera(P.p[rank], 1, q, valor);
    const v4* src = reinterpret_cast<const v4*>(slot(P.p[rank], par, 1, q));
    for (int i = a + threadIdx.x; i < b; i += kThreads) vout[i] = __builtin_nontemporal_load(src + i);
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = valor;
}

// ---------- algo 3: rsag LL128-lite: paquete de 16 B = 7 valores + etiqueta de 16 bits
// Depende de que un store de 16 B alineado llegue ENTERO al par (un solo TLP). Si se
// partiera, la etiqueta nueva podria llegar con datos viejos: el banco de estres lo mira.
__device__ __forceinline__ bool tag7(v4 v, uint32_t tag) {
  return (static_cast<uint32_t>(v[3]) >> 16) == tag;
}

__device__ __forceinline__ void get7_otros(v4* b0, v4* b1, v4* b2, uint32_t tag, v4* o) {
  v4 a, b, c;
  bool fa = false, fb = false, fc = false;
  while (true) {
    asm volatile(
        "global_load_b128 %0, %3, off glc slc dlc\n"
        "global_load_b128 %1, %4, off glc slc dlc\n"
        "global_load_b128 %2, %5, off glc slc dlc\n"
        "s_waitcnt vmcnt(0)\n"
        : "=&v"(a), "=&v"(b), "=&v"(c)
        : "v"(b0), "v"(b1), "v"(b2)
        : "memory");
    if (!fa && tag7(a, tag)) { fa = true; o[0] = a; }
    if (!fb && tag7(b, tag)) { fb = true; o[1] = b; }
    if (!fc && tag7(c, tag)) { fc = true; o[2] = c; }
    if (fa && fb && fc) break;
  }
  const v4 z = {0, 0, 0, 0};
  *reinterpret_cast<volatile v4*>(b0) = z;
  *reinterpret_cast<volatile v4*>(b1) = z;
  *reinterpret_cast<volatile v4*>(b2) = z;
}

__device__ __forceinline__ v4 get7(v4* src, uint32_t tag) {
  v4 a;
  do {
    asm volatile("global_load_b128 %0, %1, off glc slc dlc\n s_waitcnt vmcnt(0)\n"
                 : "=v"(a) : "v"(src) : "memory");
  } while (!tag7(a, tag));
  const v4 z = {0, 0, 0, 0};
  *reinterpret_cast<volatile v4*>(src) = z;
  return a;
}

template <typename T>
__device__ __forceinline__ v4 load7(const T* in, int i, int numel, uint32_t tag) {
  v4 r = {0, 0, 0, 0};
  uint16_t* h = reinterpret_cast<uint16_t*>(&r);
#pragma unroll
  for (int e = 0; e < 7; ++e)
    if (7 * i + e < numel) h[e] = reinterpret_cast<const uint16_t*>(in)[7 * i + e];
  h[7] = static_cast<uint16_t>(tag);
  return r;
}

template <typename T>
__device__ __forceinline__ void acc7(float* a, v4 b) {
  const T* h = reinterpret_cast<const T*>(&b);
#pragma unroll
  for (int e = 0; e < 7; ++e) a[e] += f(h[e]);
}

template <typename T>
__device__ __forceinline__ v4 pack7(const float* a, uint32_t tag) {
  v4 r;
  T* h = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int e = 0; e < 7; ++e) h[e] = t<T>(a[e]);
  reinterpret_cast<uint16_t*>(&r)[7] = static_cast<uint16_t>(tag);
  return r;
}

template <typename T>
__device__ __forceinline__ void store7(T* out, int i, int numel, v4 b) {
  const uint16_t* h = reinterpret_cast<const uint16_t*>(&b);
#pragma unroll
  for (int e = 0; e < 7; ++e)
    if (7 * i + e < numel) reinterpret_cast<uint16_t*>(out)[7 * i + e] = h[e];
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    rsag7(Ptrs P, const T* __restrict__ in, T* __restrict__ out, int rank, int numel) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const int par = epoch & 1;
  const int pk = (numel + 6) / 7;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  auto s7 = [&](char* base, int fase, int src) {
    return reinterpret_cast<v4*>(slot(base, par, fase, src));
  };
  for (int i = first; i < pk; i += stride) {
    const int own = (i / kThreads) & 3;
    if (own != rank) __builtin_nontemporal_store(load7(in, i, numel, tag), s7(P.p[own], 0, rank) + i);
  }
  for (int i = first; i < pk; i += stride) {
    if (((i / kThreads) & 3) != rank) continue;
    v4 o[3];
    get7_otros(s7(P.p[rank], 0, (rank + 1) & 3) + i, s7(P.p[rank], 0, (rank + 2) & 3) + i,
               s7(P.p[rank], 0, (rank + 3) & 3) + i, tag, o);
    float a[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      acc7<T>(a, s == rank ? load7(in, i, numel, tag) : o[(s - rank + 3) & 3]);
    const v4 r = pack7<T>(a, tag);
    store7(out, i, numel, r);
#pragma unroll
    for (int d = 1; d < kWorld; ++d) __builtin_nontemporal_store(r, s7(P.p[(rank + d) & 3], 1, rank) + i);
  }
  for (int i = first; i < pk; i += stride) {
    const int own = (i / kThreads) & 3;
    if (own != rank) store7(out, i, numel, get7(s7(P.p[rank], 1, own) + i, tag));
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

// ---------- algo 5: oneshot LL128-lite: cada rank manda su vector a los 3 (un salto),
// cada uno suma los 4. 3x los bytes del rsag: gana cuando manda la latencia.
template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    oneshot7(Ptrs P, const T* __restrict__ in, T* __restrict__ out, int rank, int numel) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const int par = epoch & 1;
  const int pk = (numel + 6) / 7;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  auto s7 = [&](char* base, int src) { return reinterpret_cast<v4*>(slot(base, par, 0, src)); };
  for (int i = first; i < pk; i += stride) {
    const v4 v = load7(in, i, numel, tag);
#pragma unroll
    for (int d = 1; d < kWorld; ++d) __builtin_nontemporal_store(v, s7(P.p[(rank + d) & 3], rank) + i);
  }
  for (int i = first; i < pk; i += stride) {
    v4 o[3];
    get7_otros(s7(P.p[rank], (rank + 1) & 3) + i, s7(P.p[rank], (rank + 2) & 3) + i,
               s7(P.p[rank], (rank + 3) & 3) + i, tag, o);
    float a[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < kWorld; ++s)
      acc7<T>(a, s == rank ? load7(in, i, numel, tag) : o[(s - rank + 3) & 3]);
    store7(out, i, numel, pack7<T>(a, tag));
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

// ---------- algo 4: rsag en crudo con banderas SIN fence de sistema: las banderas se
// escriben tras s_waitcnt_vscnt + barrera del bloque y se sondean con glc slc dlc,
// los 3 pares a la vez (hilos 0..2).
__device__ __forceinline__ void publica2(Ptrs& P, int rank, int fase, uint32_t valor) {
  asm volatile("s_waitcnt_vscnt null, 0" ::: "memory");
  __syncthreads();
  if (threadIdx.x < 3) {
    const int dst = (rank + threadIdx.x + 1) & 3;
    *reinterpret_cast<volatile uint32_t*>(bandera(P.p[dst], fase, rank, blockIdx.x)) = valor;
  }
}

__device__ __forceinline__ void espera2(char* self, int fase, int rank, uint32_t valor, int solo) {
  if (threadIdx.x < 3) {
    const int src = (rank + threadIdx.x + 1) & 3;
    if (solo < 0 || src == solo) {
      uint32_t* b = bandera(self, fase, src, blockIdx.x);
      uint32_t v;
      do {
        asm volatile("global_load_b32 %0, %1, off glc slc dlc\n s_waitcnt vmcnt(0)\n"
                     : "=v"(v) : "v"(b) : "memory");
      } while (v < valor);
    }
  }
  __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    rsag_simple2(Ptrs P, const T* __restrict__ in, T* __restrict__ out, int rank, int numel) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t valor = epoch + 1;
  const int par = epoch & 1;
  const int U = numel / 8;
  const int cuarto = (U + 3) / 4;
  const int tramo = (cuarto + gridDim.x - 1) / gridDim.x;
  const v4* vin = reinterpret_cast<const v4*>(in);
  v4* vout = reinterpret_cast<v4*>(out);
  auto rango = [&](int q, int& a, int& b) {
    a = min(q * cuarto + blockIdx.x * tramo, min((q + 1) * cuarto, U));
    b = min(a + tramo, min((q + 1) * cuarto, U));
  };
  for (int d = 1; d < kWorld; ++d) {
    const int q = (rank + d) & 3;
    int a, b;
    rango(q, a, b);
    v4* dst = reinterpret_cast<v4*>(slot(P.p[q], par, 0, rank));
    for (int i = a + threadIdx.x; i < b; i += kThreads) __builtin_nontemporal_store(vin[i], dst + i);
  }
  publica2(P, rank, 0, valor);
  int a, b;
  rango(rank, a, b);
  espera2(P.p[rank], 0, rank, valor, -1);
  for (int i = a + threadIdx.x; i < b; i += kThreads) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < kWorld; ++s) {
      v4 v;
      if (s == rank) v = vin[i];
      else {
        const v4* src = reinterpret_cast<const v4*>(slot(P.p[rank], par, 0, s)) + i;
        asm volatile("global_load_b128 %0, %1, off glc slc dlc\n s_waitcnt vmcnt(0)\n"
                     : "=v"(v) : "v"(src) : "memory");
      }
      suma8<T>(acc, v);
    }
    const v4 r = empaca8<T>(acc);
    vout[i] = r;
#pragma unroll
    for (int d = 1; d < kWorld; ++d)
      __builtin_nontemporal_store(r, reinterpret_cast<v4*>(slot(P.p[(rank + d) & 3], par, 1, rank)) + i);
  }
  publica2(P, rank, 1, valor);
  espera2(P.p[rank], 1, rank, valor, -1);
  for (int d = 1; d < kWorld; ++d) {
    const int q = (rank + d) & 3;
    rango(q, a, b);
    const v4* src = reinterpret_cast<const v4*>(slot(P.p[rank], par, 1, q));
    for (int i = a + threadIdx.x; i < b; i += kThreads) {
      v4 v;
      asm volatile("global_load_b128 %0, %1, off glc slc dlc\n s_waitcnt vmcnt(0)\n"
                   : "=v"(v) : "v"(src + i) : "memory");
      vout[i] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = valor;
}

// ---------- all-reduce + suma residual + Gemma-RMSNorm FUSIONADOS (LL128-lite)
// Una fila por bloque (1024 hilos, un paquete de 7 valores por hilo: H <= 7168).
// Semantica de vllm.ir.ops.fused_add_rms_norm(ar, res, w.float() + 1, eps):
//   x = float(ar_fp16) + float(res); res_out = T(x); var = mean(x^2) en fp32;
//   out = T((x * rsqrt(var + eps)) * (float(w) + 1))
// Dueno de cada paquete por ONDA ((j / 32) & 3): cada onda escribe 512 B contiguos.
// Grid FIJO (kFilasBloques): los contadores de epoca de todos los bloques avanzan
// juntos y la paridad es la misma para todos en cada llamada.
constexpr int kRmsThreads = 1024;
constexpr int kFilasBloques = 32;

template <typename T>
__device__ __forceinline__ v4 load7r(const T* row, int j, int H, uint32_t tag) {
  v4 r = {0, 0, 0, 0};
  uint16_t* h = reinterpret_cast<uint16_t*>(&r);
#pragma unroll
  for (int e = 0; e < 7; ++e)
    if (7 * j + e < H) h[e] = reinterpret_cast<const uint16_t*>(row)[7 * j + e];
  h[7] = static_cast<uint16_t>(tag);
  return r;
}

template <typename T>
__global__ void __launch_bounds__(kRmsThreads, 1)
    ar_rms7(Ptrs P, const T* __restrict__ in, const T* __restrict__ res,
            const T* __restrict__ w, T* __restrict__ out, T* __restrict__ res_out,
            float eps, int rank, int M, int H) {
  __shared__ float red_s[kRmsThreads / 32];
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const int par = epoch & 1;
  const int pkrow = (H + 6) / 7;
  const int j = threadIdx.x;
  const bool act = j < pkrow;
  const int own = (j >> 5) & 3;
  auto s7 = [&](char* base, int fase, int src) {
    return reinterpret_cast<v4*>(slot(base, par, fase, src));
  };
  for (int r = blockIdx.x; r < M; r += gridDim.x) {
    const int i = r * pkrow + j;
    const T* rin = in + static_cast<int64_t>(r) * H;
    if (act && own != rank)
      __builtin_nontemporal_store(load7r(rin, j, H, tag), s7(P.p[own], 0, rank) + i);
    v4 red;
    if (act && own == rank) {
      v4 o[3];
      get7_otros(s7(P.p[rank], 0, (rank + 1) & 3) + i, s7(P.p[rank], 0, (rank + 2) & 3) + i,
                 s7(P.p[rank], 0, (rank + 3) & 3) + i, tag, o);
      float a[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < kWorld; ++s)
        acc7<T>(a, s == rank ? load7r(rin, j, H, tag) : o[(s - rank + 3) & 3]);
      red = pack7<T>(a, tag);
#pragma unroll
      for (int d = 1; d < kWorld; ++d)
        __builtin_nontemporal_store(red, s7(P.p[(rank + d) & 3], 1, rank) + i);
    }
    if (act && own != rank) red = get7(s7(P.p[rank], 1, own) + i, tag);
    // suma residual + norma
    float x[7];
    float ss = 0.f;
    const int64_t base = static_cast<int64_t>(r) * H + 7 * j;
    if (act) {
      const T* rv = reinterpret_cast<const T*>(&red);
#pragma unroll
      for (int e = 0; e < 7; ++e) {
        x[e] = 0.f;
        if (7 * j + e < H) {
          x[e] = f(rv[e]) + f(res[base + e]);
          res_out[base + e] = t<T>(x[e]);
          ss += x[e] * x[e];
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if ((j & 31) == 0) red_s[j >> 5] = ss;
    __syncthreads();
    if (j < 32) {
      float v = j < kRmsThreads / 32 ? red_s[j] : 0.f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
      if (j == 0) red_s[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(red_s[0] / H + eps);
    if (act) {
#pragma unroll
      for (int e = 0; e < 7; ++e)
        if (7 * j + e < H) out[base + e] = t<T>((x[e] * inv) * (f(w[7 * j + e]) + 1.0f));
    }
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

// Suma residual + Gemma-RMSNorm SOLA (sin all-reduce), una fila por bloque: para
// comparar con el fusionado y como camino a M grande.
template <typename T>
__global__ void __launch_bounds__(kRmsThreads, 1)
    add_rms(const T* __restrict__ ar, const T* __restrict__ res, const T* __restrict__ w,
            T* __restrict__ out, T* __restrict__ res_out, float eps, int H) {
  __shared__ float red_s[kRmsThreads / 32];
  const int r = blockIdx.x, j = threadIdx.x;
  const int64_t base = static_cast<int64_t>(r) * H + 7 * j;
  float x[7];
  float ss = 0.f;
#pragma unroll
  for (int e = 0; e < 7; ++e) {
    x[e] = 0.f;
    if (7 * j + e < H) {
      x[e] = f(ar[base + e]) + f(res[base + e]);
      res_out[base + e] = t<T>(x[e]);
      ss += x[e] * x[e];
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
  if ((j & 31) == 0) red_s[j >> 5] = ss;
  __syncthreads();
  if (j < 32) {
    float v = red_s[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    if (j == 0) red_s[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(red_s[0] / H + eps);
#pragma unroll
  for (int e = 0; e < 7; ++e)
    if (7 * j + e < H) out[base + e] = t<T>((x[e] * inv) * (f(w[7 * j + e]) + 1.0f));
}

// ---------- all-reduce + residual + Gemma-RMSNorm con el reparto del rsag7 (32x256):
// cada paquete lo termina UN hilo (dueno: lo reduce; resto: lo recibe). Cada onda escribe
// su suma de cuadrados parcial para las (a lo sumo) 2 filas que toca; UNA barrera de grid
// (los 32 bloques son co-residentes); cada hilo suma los parciales de sus filas EN ORDEN
// DE ONDA (determinista e identico en los 4 ranks) y escribe la salida.
constexpr int kPkHilo = 4;
constexpr int kMaxFilasG = 64;

__device__ __forceinline__ void barrera_grid(uint32_t* bar) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const uint32_t g = __hip_atomic_load(bar + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    if (atomicAdd(bar, 1u) == gridDim.x - 1) {
      __hip_atomic_store(bar, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(bar + 1, g + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      while (__hip_atomic_load(bar + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == g) {
      }
    }
  }
  __syncthreads();
}

template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    ar_rms7g(Ptrs P, const T* __restrict__ in, const T* __restrict__ res,
             const T* __restrict__ w, T* __restrict__ out, T* __restrict__ res_out,
             float eps, int rank, int numel, int H) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const int par = epoch & 1;
  const int pk = (numel + 6) / 7;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  auto s7 = [&](char* base, int fase, int src) {
    return reinterpret_cast<v4*>(slot(base, par, fase, src));
  };
  for (int i = first; i < pk; i += stride) {
    const int own = (i / kThreads) & 3;
    if (own != rank) __builtin_nontemporal_store(load7(in, i, numel, tag), s7(P.p[own], 0, rank) + i);
  }
  v4 red[kPkHilo];
#pragma unroll
  for (int k = 0; k < kPkHilo; ++k) {
    const int i = first + k * stride;
    if (i < pk && ((i / kThreads) & 3) == rank) {
      v4 o[3];
      get7_otros(s7(P.p[rank], 0, (rank + 1) & 3) + i, s7(P.p[rank], 0, (rank + 2) & 3) + i,
                 s7(P.p[rank], 0, (rank + 3) & 3) + i, tag, o);
      float a[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s = 0; s < kWorld; ++s)
        acc7<T>(a, s == rank ? load7(in, i, numel, tag) : o[(s - rank + 3) & 3]);
      red[k] = pack7<T>(a, tag);
#pragma unroll
      for (int d = 1; d < kWorld; ++d)
        __builtin_nontemporal_store(red[k], s7(P.p[(rank + d) & 3], 1, rank) + i);
    }
  }
  float x[kPkHilo][7];
#pragma unroll
  for (int k = 0; k < kPkHilo; ++k) {
    const int i = first + k * stride;
    const int own = (i / kThreads) & 3;
    if (i < pk && own != rank) red[k] = get7(s7(P.p[rank], 1, own) + i, tag);
    // suma residual y parciales de la onda: sus 32 paquetes (224 elementos) tocan
    // como mucho 2 filas, r0 y r0 + 1
    const int w0 = (i >> 5) << 5;
    const int r0 = (7 * w0) / H;
    float ss0 = 0.f, ss1 = 0.f;
    const T* rv = reinterpret_cast<const T*>(&red[k]);
#pragma unroll
    for (int e = 0; e < 7; ++e) {
      x[k][e] = 0.f;
      const int el = 7 * i + e;
      if (i < pk && el < numel) {
        x[k][e] = f(rv[e]) + f(res[el]);
        res_out[el] = t<T>(x[k][e]);
        if (el / H == r0) ss0 += x[k][e] * x[k][e];
        else ss1 += x[k][e] * x[k][e];
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss0 += __shfl_xor(ss0, o, 32);
      ss1 += __shfl_xor(ss1, o, 32);
    }
    if ((threadIdx.x & 31) == 0 && w0 < pk) {
      P.parc[2 * (w0 >> 5)] = ss0;
      P.parc[2 * (w0 >> 5) + 1] = ss1;
    }
  }
  barrera_grid(P.bar);
  // inv de cada fila una vez por bloque: una onda por fila, un parcial por carril
  __shared__ float inv_s[kMaxFilasG];
  const int M = (numel + H - 1) / H;
  const int onda = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int r = onda; r < M; r += kThreads / 32) {
    const int wa = (r * H) / 224, wb = min(((r + 1) * H - 1) / 224, (pk - 1) >> 5);
    float v = 0.f;
    for (int wv = wa + lane; wv <= wb; wv += 32) {
      const int rw = (224 * wv) / H;
      const float* pp = P.parc + 2 * wv;
      if (rw == r) v += __hip_atomic_load(pp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      else if (rw + 1 == r) v += __hip_atomic_load(pp + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    // orden fijo de reduccion (xor 16..1): identico en los 4 ranks
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    if (lane == 0) inv_s[r] = rsqrtf(v / H + eps);
  }
  __syncthreads();
#pragma unroll
  for (int k = 0; k < kPkHilo; ++k) {
    const int i = first + k * stride;
    if (i >= pk) continue;
    const int ra = (7 * i) / H;
#pragma unroll
    for (int e = 0; e < 7; ++e) {
      const int el = 7 * i + e;
      if (el < numel) {
        const int r = el / H;
        out[el] = t<T>((x[k][e] * inv_s[r]) * (f(w[el - r * H]) + 1.0f));
      }
    }
    (void)ra;
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

void add_rms_solo(at::Tensor& ar, at::Tensor& res, at::Tensor& w, double eps, at::Tensor& out,
                  at::Tensor& res_out) {
  const int M = static_cast<int>(ar.size(0)), H = static_cast<int>(ar.size(1));
  TORCH_CHECK((H + 6) / 7 <= kRmsThreads);
  const c10::DeviceGuard g(ar.device());
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  if (ar.scalar_type() == at::kHalf)
    add_rms<__half><<<M, kRmsThreads, 0, stream>>>(
        (const __half*)ar.data_ptr(), (const __half*)res.data_ptr(), (const __half*)w.data_ptr(),
        (__half*)out.data_ptr(), (__half*)res_out.data_ptr(), static_cast<float>(eps), H);
  else
    add_rms<__hip_bfloat16><<<M, kRmsThreads, 0, stream>>>(
        (const __hip_bfloat16*)ar.data_ptr(), (const __hip_bfloat16*)res.data_ptr(),
        (const __hip_bfloat16*)w.data_ptr(), (__hip_bfloat16*)out.data_ptr(),
        (__hip_bfloat16*)res_out.data_ptr(), static_cast<float>(eps), H);
  HIP_CHECK(hipGetLastError());
}

static int blocks_env();

void ar_add_rms(fptr_t ctx, at::Tensor& inp, at::Tensor& res, at::Tensor& w, double eps,
                at::Tensor& out, at::Tensor& res_out) {
  auto* c = reinterpret_cast<Ctx*>(ctx);
  TORCH_CHECK(inp.dim() == 2 && inp.is_contiguous() && res.is_contiguous() &&
              w.is_contiguous() && out.is_contiguous() && res_out.is_contiguous());
  const int M = static_cast<int>(inp.size(0)), H = static_cast<int>(inp.size(1));
  TORCH_CHECK((H + 6) / 7 <= kRmsThreads, "ar_add_rms: H > 7168");
  TORCH_CHECK(static_cast<int64_t>(M) * ((H + 6) / 7) * 2 <= kMaxPk, "ar_add_rms: M*H grande");
  TORCH_CHECK(w.scalar_type() == inp.scalar_type() && res.scalar_type() == inp.scalar_type());
  const c10::DeviceGuard g(inp.device());
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  const int bl = blocks_env();
  // Seleccion medida (caja local, grafo, TP4, H=5120, us):
  //   M      AR solo  AR + norma aparte  fusion por fila  fusion con barrera de grid
  //   4       15,6         19,3              18,0               19,8
  //   16      46,0         49,9              51,1               49,9
  // => fila hasta 8 filas; por encima, AR LL128 + norma aparte (dos kernels). La de
  // grid (JART_RMS_GRID=1) queda como registro: la barrera se come lo que ahorra.
  if (M > 8 && std::getenv("JART_RMS_GRID") == nullptr) {
    at::Tensor ar = at::empty_like(inp);
    const int n = M * H;
    if (inp.scalar_type() == at::kHalf) {
      rsag7<__half><<<bl, kThreads, 0, stream>>>(c->ptrs, (const __half*)inp.data_ptr(),
                                                  (__half*)ar.data_ptr(), c->rank, n);
    } else {
      rsag7<__hip_bfloat16><<<bl, kThreads, 0, stream>>>(
          c->ptrs, (const __hip_bfloat16*)inp.data_ptr(), (__hip_bfloat16*)ar.data_ptr(),
          c->rank, n);
    }
    HIP_CHECK(hipGetLastError());
    add_rms_solo(ar, res, w, eps, out, res_out);
    return;
  }
  if (M > 8 && H >= 224 && M <= kMaxFilasG &&
      (static_cast<int64_t>(M) * H + 6) / 7 <= static_cast<int64_t>(kPkHilo) * bl * kThreads) {
#define LANZA_G(T)                                                                    \
  ar_rms7g<T><<<bl, kThreads, 0, stream>>>(                                           \
      c->ptrs, (const T*)inp.data_ptr(), (const T*)res.data_ptr(), (const T*)w.data_ptr(), \
      (T*)out.data_ptr(), (T*)res_out.data_ptr(), static_cast<float>(eps), c->rank, M * H, H);
    if (inp.scalar_type() == at::kHalf) {
      LANZA_G(__half)
    } else {
      LANZA_G(__hip_bfloat16)
    }
#undef LANZA_G
    HIP_CHECK(hipGetLastError());
    return;
  }
#define LANZA_RMS(T)                                                                  \
  ar_rms7<T><<<kFilasBloques, kRmsThreads, 0, stream>>>(                              \
      c->ptrs, (const T*)inp.data_ptr(), (const T*)res.data_ptr(), (const T*)w.data_ptr(), \
      (T*)out.data_ptr(), (T*)res_out.data_ptr(), static_cast<float>(eps), c->rank, M, H);
  if (inp.scalar_type() == at::kHalf) {
    LANZA_RMS(__half)
  } else {
    TORCH_CHECK(inp.scalar_type() == at::kBFloat16);
    LANZA_RMS(__hip_bfloat16)
  }
#undef LANZA_RMS
  HIP_CHECK(hipGetLastError());
}

__global__ void nop(Ptrs P, int rank) {
  if (threadIdx.x == 0) {
    uint32_t* ep = P.ep;
    ep[blockIdx.x] = ep[blockIdx.x] + 1;
  }
}

// Ping LL: cada rank escribe UN paquete a cada par y espera los 3. Suelo de latencia.
__global__ void ping(Ptrs P, int rank) {
  uint32_t* ep = P.ep;
  const uint32_t epoch = ep[blockIdx.x];
  const uint32_t tag = epoch % 65535u + 1;
  const int par = epoch & 1;
  if (threadIdx.x < 3) {
    const int d = threadIdx.x + 1;
    put(slot(P.p[(rank + d) & 3], par, 0, rank) + blockIdx.x, uint64_t(tag) << 48);
  }
  if (threadIdx.x == 0) {
    uint64_t o[3];
    get_otros(slot(P.p[rank], par, 0, (rank + 1) & 3) + blockIdx.x,
              slot(P.p[rank], par, 0, (rank + 2) & 3) + blockIdx.x,
              slot(P.p[rank], par, 0, (rank + 3) & 3) + blockIdx.x, tag, o);
  }
  __syncthreads();
  if (threadIdx.x == 0) ep[blockIdx.x] = epoch + 1;
}

std::tuple<fptr_t, at::Tensor> alloc_shared() {
  void* buf = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(&buf, kBytes, hipDeviceMallocUncached));
  HIP_CHECK(hipMemset(buf, 0, kBytes));
  HIP_CHECK(hipDeviceSynchronize());
  auto h = at::empty({static_cast<int64_t>(sizeof(hipIpcMemHandle_t))},
                     at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  HIP_CHECK(hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t*>(h.data_ptr()), buf));
  return {reinterpret_cast<fptr_t>(buf), h};
}

fptr_t open_handle(at::Tensor& h) {
  TORCH_CHECK(h.device().is_cpu() && h.numel() == sizeof(hipIpcMemHandle_t));
  hipIpcMemHandle_t ih;
  std::memcpy(&ih, h.data_ptr(), sizeof(ih));
  void* p = nullptr;
  HIP_CHECK(hipIpcOpenMemHandle(&p, ih, hipIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(p);
}

fptr_t init(std::vector<int64_t> ptrs, int64_t rank) {
  TORCH_CHECK(ptrs.size() == kWorld && rank >= 0 && rank < kWorld);
  auto* c = new Ctx;
  c->rank = static_cast<int>(rank);
  for (int i = 0; i < kWorld; ++i) c->ptrs.p[i] = reinterpret_cast<char*>(ptrs[i]);
  HIP_CHECK(hipMalloc(&c->ptrs.ep, kMaxBlocks * sizeof(uint32_t)));
  HIP_CHECK(hipMalloc(&c->ptrs.parc, 2 * (kMaxPk / 32 + 64) * sizeof(float)));
  HIP_CHECK(hipMalloc(&c->ptrs.bar, 64 * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(c->ptrs.bar, 0, 64 * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(c->ptrs.ep, 0, kMaxBlocks * sizeof(uint32_t)));
  HIP_CHECK(hipDeviceSynchronize());
  return reinterpret_cast<fptr_t>(c);
}

static int blocks_env() {
  static const int b = [] {
    const char* s = std::getenv("JART_AR_BLOCKS");
    return s ? std::atoi(s) : 32;
  }();
  return b;
}

void all_reduce(fptr_t ctx, at::Tensor& inp, at::Tensor& out, int64_t algo) {
  auto* c = reinterpret_cast<Ctx*>(ctx);
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous() && inp.numel() == out.numel());
  TORCH_CHECK(inp.numel() <= kMaxNumel, "jart_ar: numel > ", kMaxNumel);
  const c10::DeviceGuard g(inp.device());
  auto stream = c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream();
  const int n = static_cast<int>(inp.numel());
  // algo 6 = auto (medido en la caja local, grafo, TP4): rsag LL de 8 B hasta ~12k
  // elementos (1-2 filas de 5120), LL128-lite por encima
  if (algo == 6) algo = n <= 12288 ? 1 : 3;
  const int blocks = blocks_env();
  TORCH_CHECK(blocks > 0 && blocks <= kMaxBlocks);
#define LANZA(T)                                                                   \
  if (algo == 2) {                                                                 \
    TORCH_CHECK(n % 8 == 0, "rsag_simple: numel % 8");                             \
    rsag_simple<T><<<blocks, kThreads, 0, stream>>>(                               \
        c->ptrs, (const T*)inp.data_ptr(), (T*)out.data_ptr(), c->rank, n);        \
  } else if (algo == 3) {                                                          \
    rsag7<T><<<blocks, kThreads, 0, stream>>>(c->ptrs, (const T*)inp.data_ptr(),   \
                                              (T*)out.data_ptr(), c->rank, n);     \
  } else if (algo == 5) {                                                          \
    oneshot7<T><<<blocks, kThreads, 0, stream>>>(c->ptrs, (const T*)inp.data_ptr(), \
                                                 (T*)out.data_ptr(), c->rank, n);  \
  } else if (algo == 4) {                                                          \
    TORCH_CHECK(n % 8 == 0, "rsag_simple2: numel % 8");                            \
    rsag_simple2<T><<<blocks, kThreads, 0, stream>>>(                              \
        c->ptrs, (const T*)inp.data_ptr(), (T*)out.data_ptr(), c->rank, n);        \
  } else if (algo == 0)                                                            \
    oneshot<T><<<blocks, kThreads, 0, stream>>>(c->ptrs, (const T*)inp.data_ptr(), \
                                                (T*)out.data_ptr(), c->rank, n);   \
  else                                                                             \
    rsag<T><<<blocks, kThreads, 0, stream>>>(c->ptrs, (const T*)inp.data_ptr(),    \
                                             (T*)out.data_ptr(), c->rank, n);
  if (algo == 9) {
    nop<<<blocks, kThreads, 0, stream>>>(c->ptrs, c->rank);
  } else if (algo == 8) {
    ping<<<blocks, kThreads, 0, stream>>>(c->ptrs, c->rank);
  } else if (inp.scalar_type() == at::kHalf) {
    LANZA(__half)
  } else {
    TORCH_CHECK(inp.scalar_type() == at::kBFloat16);
    LANZA(__hip_bfloat16)
  }
#undef LANZA
  HIP_CHECK(hipGetLastError());
}

}  // namespace jart_ar

TORCH_LIBRARY(_jart_ar, m) {
  m.def("alloc_shared() -> (int, Tensor)", &jart_ar::alloc_shared);
  m.def("open_handle(Tensor h) -> int", &jart_ar::open_handle);
  m.def("init(int[] ptrs, int rank) -> int", &jart_ar::init);
  m.def("all_reduce(int ctx, Tensor inp, Tensor(a!) out, int algo) -> ()",
        &jart_ar::all_reduce);
  m.def("add_rms_solo(Tensor ar, Tensor res, Tensor w, float eps, Tensor(a!) out, "
        "Tensor(b!) res_out) -> ()", &jart_ar::add_rms_solo);
  m.def("ar_add_rms(int ctx, Tensor inp, Tensor res, Tensor w, float eps, Tensor(a!) out, "
        "Tensor(b!) res_out) -> ()", &jart_ar::ar_add_rms);
}
