// Pieces shared by the decode megakernels in this directory (gfx1201, wave32; also builds for gfx11).
#pragma once

#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#define CHECK(x)                                                                 \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

#define AGENT __HIP_MEMORY_SCOPE_AGENT

// The few instructions written as inline asm, in each ISA's syntax. On gfx11 a device-scope scalar
// load is s_load ... glc, and other blocks' stores become visible only after invalidating both the
// CU's L0 and the shader array's GL1 (gfx12 has no GL1: SCOPE_SE invalidates the L0).
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define MK_SLOAD_DEV(op) op " %0, %1, 0x0 glc dlc\n\ts_waitcnt lgkmcnt(0)"
#define MK_WAIT_STORES_LDS "s_waitcnt_vscnt null, 0x0\n\ts_waitcnt lgkmcnt(0)"
#define MK_INV_BARRIER "buffer_gl1_inv\n\tbuffer_gl0_inv"
#define MK_INV_DEV "buffer_gl1_inv\n\tbuffer_gl0_inv"
#define MK_DCACHE_INV "buffer_gl1_inv\n\tbuffer_gl0_inv\n\ts_dcache_inv\n\ts_waitcnt lgkmcnt(0)"
#else
#define MK_SLOAD_DEV(op) op " %0, %1, 0x0 scope:SCOPE_DEV\n\ts_wait_kmcnt 0x0"
#define MK_WAIT_STORES_LDS "s_wait_storecnt 0x0\n\ts_wait_dscnt 0x0"
#define MK_INV_BARRIER "global_inv scope:SCOPE_SE"
#define MK_INV_DEV "global_inv scope:SCOPE_DEV"
#define MK_DCACHE_INV "s_dcache_inv\n\ts_wait_kmcnt 0x0"
#endif

typedef unsigned u4v __attribute__((ext_vector_type(4)));

// ------------------------------------------------------------------ device
__device__ __forceinline__ float f16_to_f32(uint16_t u) { return (float)__builtin_bit_cast(_Float16, u); }
__device__ __forceinline__ uint16_t f32_to_f16(float f) { return __builtin_bit_cast(uint16_t, (_Float16)f); }
__device__ __forceinline__ float half_lo(unsigned bits) { return f16_to_f32((uint16_t)bits); }
__device__ __forceinline__ float half_hi(unsigned bits) { return f16_to_f32((uint16_t)(bits >> 16)); }
__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __builtin_bit_cast(float, (unsigned)u << 16); }
__device__ __forceinline__ float bf16_lo(unsigned bits) { return __builtin_bit_cast(float, bits << 16); }
__device__ __forceinline__ float bf16_hi(unsigned bits) { return __builtin_bit_cast(float, bits & 0xFFFF0000u); }

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
  return v;
}
__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
  return v;
}

// Workgroup barrier that only orders LDS. __syncthreads also waits for every
// outstanding global load and invalidates L0 (global_inv scope:SCOPE_SE).
__device__ __forceinline__ void lds_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ unsigned load_dev_scalar(const unsigned* p) {
  unsigned v;
  asm volatile(MK_SLOAD_DEV("s_load_b32") : "=s"(v) : "s"(p) : "memory");
  return v;
}

// Grid barrier on a monotonic arrival counter count[0]. Each block drains its stores, then
// thread 0 adds one and spins with a scalar device-scope load (kmcnt, not loadcnt),
// so vector loads still in flight do not hold the barrier. Afterwards every wave
// invalidates L0 so it sees the other blocks' stores.
//
// The counter wraps after 2^32 arrivals (about 470,000 tokens), so arrival and target are
// compared by their signed difference. Watchdog: a block that waits longer than
// kBarrierTimeout (only possible if the grid is not all resident) sets count[1], and from then
// on every barrier of every block falls through; the step's output is garbage and the host,
// which reads count[1] through mk_status, reports an error instead of hanging.
constexpr unsigned long long kBarrierTimeout = 100000000ull;  // 1 s of wall clock (100 MHz)

__device__ __forceinline__ void grid_barrier(unsigned* count, unsigned& my_gen) {
  asm volatile(MK_WAIT_STORES_LDS ::: "memory");
  __builtin_amdgcn_s_barrier();
  my_gen++;
  if (threadIdx.x == 0) {
    __hip_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, AGENT);
    const unsigned target = my_gen * gridDim.x;
    unsigned long long t0 = 0;
    for (unsigned spin = 1; (int)(load_dev_scalar(count) - target) < 0; spin++) {
      if ((spin & 63) == 0) {
        if (load_dev_scalar(count + 1)) break;
        const unsigned long long now = wall_clock64();
        if (!t0) {
          t0 = now;
        } else if (now - t0 > kBarrierTimeout) {
          __hip_atomic_store(count + 1, 1u, __ATOMIC_RELAXED, AGENT);
          break;
        }
      }
    }
  }
  __builtin_amdgcn_s_barrier();
  asm volatile(MK_INV_BARRIER ::: "memory");
}

// Copies the watchdog flag to host memory (one thread, after the step's last barrier).
__device__ __forceinline__ void mk_status(unsigned* status, unsigned* count) {
  if (status && blockIdx.x == 0 && threadIdx.x == 0)
    __hip_atomic_store(status, load_dev_scalar(count + 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

constexpr int kMaxStamps = 512;

__device__ __forceinline__ void stamp(unsigned long long* stamps, int& n) {
  if (stamps && blockIdx.x == 0 && threadIdx.x == 0 && n < kMaxStamps) stamps[n] = wall_clock64();
  n++;
}

__device__ __forceinline__ int wave_index() { return (blockIdx.x * blockDim.x + threadIdx.x) >> 5; }
__device__ __forceinline__ int wave_count() { return (gridDim.x * blockDim.x) >> 5; }

// --------------------------------------------------------------- PCIe keepalive
// The GPU's PCIe link enters ASPM L1 after about a millisecond without traffic, and the first
// host access after that (a kernel's completion signal, a mailbox poll, logits written to host
// memory) waits about 280 us for it to wake. A decode step reads only device memory for over a
// millisecond, so it pays that once per token. One posted write to host memory per layer keeps
// the link awake.
__device__ __forceinline__ void mk_keepalive(unsigned* p, int v) {
  if (p && blockIdx.x == 0 && threadIdx.x == 0) __hip_atomic_store(p, (unsigned)v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Cooperative launch: the runtime starts the grid only if every block can be resident at once and
// returns an error otherwise, instead of a grid barrier that waits forever. About 10 us more than a
// plain launch.
template <typename... P, typename... A>
inline hipError_t mk_launch(void (*kernel)(P...), int grid, int block, hipStream_t stream, A&... args) {
  void* params[] = {(void*)&args...};
  return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), dim3(grid), dim3(block), params, 0, stream);
}

// ------------------------------------------------------------- resident mode
// A resident kernel stays launched across tokens and takes them from a mailbox in pinned host
// memory, so a token costs no kernel launch. Launch boundaries are expensive once a kernel has
// touched more than about 600 MB (a decode step reads all the weights): about 290 us between
// the end of one launch and the start of the next.
//
// Block 0 polls the mailbox and relays each command to the other blocks through device memory.
// With no command for idle_ticks (wall clock, 100 MHz) the kernel exits by itself, so the GPU is
// never held for long and a host call that waits for the whole device still returns; the host
// relaunches it for the next token.
constexpr int kMkArgs = 64;

struct MkMailbox {           // pinned, fine-grained host memory
  unsigned seq;              // host: last posted command
  unsigned done;             // kernel: last finished command
  unsigned stop;             // host: exit now
  unsigned gen;              // kernel, on exit: grid barrier generation to continue from
  int args[kMkArgs];         // host: the command
};

struct MkRelay {             // device memory, written by block 0
  unsigned long long word;   // launch id << 32 | command number, bit 31 set for exit
  int args[kMkArgs];
};

constexpr unsigned kMkExit = 0x80000000u;

__device__ __forceinline__ unsigned long long load_dev_scalar64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile(MK_SLOAD_DEV("s_load_b64") : "=s"(v) : "s"(p) : "memory");
  return v;
}

// Waits for the next command after `last` and copies its first n_args arguments to args (LDS,
// every thread sees them on return). Returns false when the kernel should exit.
__device__ __forceinline__ bool mk_next_command(MkMailbox* mb, MkRelay* rl, unsigned launch_id, unsigned& last,
                                                int* args, int n_args, unsigned long long idle_ticks) {
  __shared__ unsigned s_cmd;
  if (threadIdx.x == 0) {
    unsigned cmd;
    if (blockIdx.x == 0) {
      const unsigned long long t0 = wall_clock64();
      for (;;) {
        cmd = __hip_atomic_load(&mb->seq, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (cmd != last) break;
        if (__hip_atomic_load(&mb->stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) ||
            wall_clock64() - t0 > idle_ticks) {
          cmd = last | kMkExit;
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
      if (!(cmd & kMkExit))
        for (int i = 0; i < n_args; i++) rl->args[i] = args[i] = __hip_atomic_load(&mb->args[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __hip_atomic_store(&rl->word, (unsigned long long)launch_id << 32 | cmd, __ATOMIC_RELEASE, AGENT);
    } else {
      unsigned long long w;
      while (((w = load_dev_scalar64(&rl->word)) >> 32) != launch_id || (unsigned)w == last) __builtin_amdgcn_s_sleep(1);
      cmd = (unsigned)w;
      asm volatile(MK_INV_DEV ::: "memory");
      if (!(cmd & kMkExit))
        for (int i = 0; i < n_args; i++) args[i] = __hip_atomic_load(&rl->args[i], __ATOMIC_RELAXED, AGENT);
    }
    s_cmd = cmd;
  }
  __syncthreads();
  const unsigned cmd = s_cmd;
  __syncthreads();
  if (cmd & kMkExit) return false;
  last = cmd;
  return true;
}

// After a step that wrote host-visible output: every block makes its writes visible to the host
// with a system-scope release (about 10 us: it writes back L2), all blocks meet, then block 0
// reports the command done. Without the release the host occasionally reads logits that are still
// in L2: completed stores to "coherent" host memory are not yet visible to the CPU.
__device__ __forceinline__ void mk_report_done(MkMailbox* mb, unsigned* bar_count, unsigned& my_gen, unsigned cmd) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
  grid_barrier(bar_count, my_gen);
  if (blockIdx.x == 0 && threadIdx.x == 0) __hip_atomic_store(&mb->done, cmd, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Host side of one resident kernel: the mailbox, its stream, and the post / wait protocol.
struct MkResident {
  MkMailbox* mb = nullptr;      // pinned host memory, also the device's view (unified addressing)
  MkRelay* rl = nullptr;        // device memory
  hipStream_t stream = nullptr;
  bool running = false;
  unsigned launch_id = 0;
  unsigned long long idle_ticks = 200000;  // 2 ms

  bool init() {
    if (hipHostMalloc((void**)&mb, sizeof(MkMailbox), hipHostMallocCoherent) != hipSuccess) return false;
    memset(mb, 0, sizeof(MkMailbox));
    if (hipMalloc(&rl, sizeof(MkRelay)) != hipSuccess) return false;
    if (hipMemset(rl, 0, sizeof(MkRelay)) != hipSuccess) return false;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) return false;
    if (const char* e = getenv("LLAMA_MEGAKERNEL_IDLE_US")) idle_ticks = 100ull * strtoull(e, nullptr, 10);
    return true;
  }
  // Asks a running kernel to exit and waits for it.
  void stop_kernel() {
    if (!running) return;
    __atomic_store_n(&mb->stop, 1u, __ATOMIC_RELEASE);
    (void)hipStreamSynchronize(stream);
    __atomic_store_n(&mb->stop, 0u, __ATOMIC_RELEASE);
    running = false;
  }
  // Posts the command already written to mb->args, (re)launching the kernel through `launch`
  // (called as launch(launch_id, last, gen)) whenever it is not running, and waits until it is
  // done. Returns false on a launch error.
  template <class Launch>
  bool run(Launch&& launch) {
    const unsigned cmd = (__atomic_load_n(&mb->seq, __ATOMIC_RELAXED) + 1) & ~kMkExit;
    __atomic_store_n(&mb->seq, cmd, __ATOMIC_RELEASE);
    for (;;) {
      if (!running) {
        if (!launch(++launch_id, __atomic_load_n(&mb->done, __ATOMIC_ACQUIRE), __atomic_load_n(&mb->gen, __ATOMIC_ACQUIRE)))
          return false;
        running = true;
      }
      for (int spin = 0; spin < 1 << 14; spin++)
        if (__atomic_load_n(&mb->done, __ATOMIC_ACQUIRE) == cmd) return true;
      // the kernel may have timed out just before the command arrived
      if (hipStreamQuery(stream) == hipSuccess) running = false;
    }
  }
  void free_all();
};

// One megakernel at a time per process: two would each wait in their grid barriers for
// workgroups the other holds. Every decode call holds g_mk_mutex, and taking it stops another
// context's resident kernel, which would otherwise keep the GPU until it idles out.
inline std::mutex g_mk_mutex;
inline MkResident* g_mk_active = nullptr;

struct MkGpuLock {
  std::lock_guard<std::mutex> g;
  explicit MkGpuLock(MkResident* self) : g(g_mk_mutex) {
    if (g_mk_active && g_mk_active != self) {
      g_mk_active->stop_kernel();
      g_mk_active = nullptr;
    }
    if (self && self->mb) g_mk_active = self;
  }
};

inline void MkResident::free_all() {
  {
    std::lock_guard<std::mutex> g(g_mk_mutex);
    stop_kernel();
    if (g_mk_active == this) g_mk_active = nullptr;
  }
  if (stream) (void)hipStreamDestroy(stream);
  if (rl) (void)hipFree(rl);
  if (mb) (void)hipHostFree(mb);
}

// Context split for decode attention: work items (pairs of query heads x chunks)
// spread over `items_target` blocks, chunks a multiple of 16 tokens.
__host__ __device__ inline void attn_split(int ntok, int groups, int items_target, int max_chunk, int& nchunk,
                                           int& chunk) {
  int per = items_target / groups;
  if (per < 1) per = 1;
  chunk = (ntok + per - 1) / per;
  chunk = (chunk + 15) / 16 * 16;
  if (chunk > max_chunk) chunk = max_chunk;
  nchunk = (ntok + chunk - 1) / chunk;
}

// -------------------------------------------------------------------- host
static inline uint16_t f2h(float f) {
  _Float16 h = (_Float16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}
static inline float h2f(uint16_t u) {
  _Float16 h;
  memcpy(&h, &u, 2);
  return (float)h;
}
static inline float bf2f(uint16_t u) {
  const uint32_t b = (uint32_t)u << 16;
  float f;
  memcpy(&f, &b, 4);
  return f;
}
static inline uint16_t f2bf(float f) {  // round to nearest even
  uint32_t b;
  memcpy(&b, &f, 4);
  b += 0x7FFF + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float uniform(float lo, float hi) { return lo + (hi - lo) * (float)((next() >> 40) * (1.0 / (1ull << 24))); }
};

static float* to_dev(const std::vector<float>& h) {
  float* p;
  CHECK(hipMalloc(&p, h.size() * 4));
  CHECK(hipMemcpy(p, h.data(), h.size() * 4, hipMemcpyHostToDevice));
  return p;
}

struct Err {
  double rel, ref_max;
  int at, nonfinite;
};
static Err max_err(const float* got, const float* ref, size_t n) {
  double num = 0, den = 0;
  int at = 0, bad = 0;
  for (size_t i = 0; i < n; i++) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) {
      bad++;
      continue;
    }
    const double e = fabs((double)got[i] - ref[i]);
    if (e > num) {
      num = e;
      at = (int)i;
    }
    den = std::max(den, fabs((double)ref[i]));
  }
  return {num / (den + 1e-30), den, at, bad};
}

// Butterfly sum over 32 lanes, the same order as wave_sum on the device.
static float wave_tree_sum(const float* v32) {
  float cur[32];
  memcpy(cur, v32, sizeof(cur));
  for (int off = 16; off > 0; off >>= 1) {
    float nxt[32];
    for (int l = 0; l < 32; l++) nxt[l] = cur[l] + cur[l ^ off];
    memcpy(cur, nxt, sizeof(cur));
  }
  return cur[0];
}

// ------------------------------------------------------------------- GGUF
enum GgmlType : uint32_t { kF32 = 0, kF16 = 1, kQ8_0 = 8, kQ4_K = 12, kQ6_K = 14, kBF16 = 30, kMXFP4 = 39 };

struct GgufTensor {
  std::vector<uint64_t> dims;
  uint32_t type;
  uint64_t offset;
};

// The file is mapped, not read: tensor data pointers stay valid while the Gguf lives.
struct Gguf {
  const uint8_t* base = nullptr;
  size_t size = 0;
  std::map<std::string, GgufTensor> tensors;
  std::map<std::string, double> nums;
  std::map<std::string, std::string> strs;
  size_t data_start = 0;

  Gguf() = default;
  Gguf(const Gguf&) = delete;
  ~Gguf() {
    if (base) munmap((void*)base, size);
  }
  const GgufTensor& tensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      fprintf(stderr, "gguf: missing tensor %s\n", name.c_str());
      exit(1);
    }
    return it->second;
  }
  bool has(const std::string& name) const { return tensors.count(name) != 0; }
  const uint8_t* data(const GgufTensor& t) const { return base + data_start + t.offset; }
  double num(const std::string& k) const {
    auto it = nums.find(k);
    if (it == nums.end()) {
      fprintf(stderr, "gguf: missing %s\n", k.c_str());
      exit(1);
    }
    return it->second;
  }
  void expect(const std::string& k, double v) const {
    if (fabs(num(k) - v) > 1e-6 * fabs(v)) {
      fprintf(stderr, "gguf: %s = %g, the kernel is built for %g\n", k.c_str(), num(k), v);
      exit(1);
    }
  }
};

static void read_gguf(Gguf& g, const char* path) {
  const int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(1);
  }
  g.size = st.st_size;
  g.base = (const uint8_t*)mmap(nullptr, g.size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (g.base == MAP_FAILED) {
    fprintf(stderr, "cannot map %s\n", path);
    exit(1);
  }
  size_t p = 0;
  auto take = [&](void* dst, size_t n) {
    if (p + n > g.size) {
      fprintf(stderr, "gguf: truncated\n");
      exit(1);
    }
    memcpy(dst, g.base + p, n);
    p += n;
  };
  auto u32 = [&] { uint32_t v; take(&v, 4); return v; };
  auto u64 = [&] { uint64_t v; take(&v, 8); return v; };
  auto str = [&] {
    const uint64_t n = u64();
    std::string s((const char*)g.base + p, n);
    p += n;
    return s;
  };
  // scalar sizes by GGUF value type; 8 = string, 9 = array
  const int scalar[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  auto scalar_value = [&](uint32_t type) -> double {
    double v = 0;
    switch (type) {
      case 0: { uint8_t x; take(&x, 1); v = x; } break;
      case 1: { int8_t x; take(&x, 1); v = x; } break;
      case 2: { uint16_t x; take(&x, 2); v = x; } break;
      case 3: { int16_t x; take(&x, 2); v = x; } break;
      case 4: { uint32_t x; take(&x, 4); v = x; } break;
      case 5: { int32_t x; take(&x, 4); v = x; } break;
      case 6: { float x; take(&x, 4); v = x; } break;
      case 7: { uint8_t x; take(&x, 1); v = x; } break;
      case 10: { uint64_t x; take(&x, 8); v = (double)x; } break;
      case 11: { int64_t x; take(&x, 8); v = (double)x; } break;
      case 12: { double x; take(&x, 8); v = x; } break;
      default: fprintf(stderr, "gguf: bad type %u\n", type); exit(1);
    }
    return v;
  };

  char magic[4];
  take(magic, 4);
  if (memcmp(magic, "GGUF", 4) != 0 || u32() != 3) {
    fprintf(stderr, "%s: not a GGUF v3 file\n", path);
    exit(1);
  }
  const uint64_t n_tensors = u64(), n_kv = u64();
  for (uint64_t i = 0; i < n_kv; i++) {
    const std::string key = str();
    const uint32_t type = u32();
    if (type == 8) {
      g.strs[key] = str();
    } else if (type == 9) {
      const uint32_t et = u32();
      const uint64_t n = u64();
      for (uint64_t j = 0; j < n; j++) {
        if (et == 8) str();
        else if (et < 13 && scalar[et]) p += scalar[et];
        else {
          fprintf(stderr, "gguf: nested array in %s\n", key.c_str());
          exit(1);
        }
      }
    } else {
      g.nums[key] = scalar_value(type);
    }
  }
  for (uint64_t i = 0; i < n_tensors; i++) {
    const std::string name = str();
    GgufTensor t;
    t.dims.resize(u32());
    for (auto& d : t.dims) d = u64();
    t.type = u32();
    t.offset = u64();
    g.tensors[name] = t;
  }
  const size_t align = g.nums.count("general.alignment") ? (size_t)g.nums["general.alignment"] : 32;
  g.data_start = (p + align - 1) / align * align;
}

static std::vector<int> read_ids(const std::string& line, const char* tag) {
  std::vector<int> ids;
  const size_t c = line.find(':');
  if (line.compare(0, strlen(tag), tag) != 0 || c == std::string::npos) {
    fprintf(stderr, "reference: expected a '%s' line\n", tag);
    exit(1);
  }
  const char* s = line.c_str() + c + 1;
  char* end;
  for (long v = strtol(s, &end, 10); end != s; v = strtol(s, &end, 10)) {
    ids.push_back((int)v);
    s = end;
  }
  return ids;
}

// tools/llama_ref.cpp output: prompt ids, generated ids, logits [n_gen][vocab].
struct Reference {
  std::vector<int> prompt, gen;
  std::vector<float> logits;
};

static bool read_reference(const std::string& prefix, int vocab, Reference& r) {
  FILE* f = fopen((prefix + ".txt").c_str(), "r");
  if (!f) return false;
  std::string l1, l2;
  {
    std::vector<char> buf(1 << 20);
    l1 = fgets(buf.data(), (int)buf.size(), f) ? buf.data() : "";
    l2 = fgets(buf.data(), (int)buf.size(), f) ? buf.data() : "";
  }
  fclose(f);
  r.prompt = read_ids(l1, "prompt");
  r.gen = read_ids(l2, "generated");
  r.logits.resize(r.gen.size() * (size_t)vocab);
  f = fopen((prefix + ".logits").c_str(), "rb");
  if (!f) return false;
  const bool ok = fread(r.logits.data(), 4, r.logits.size(), f) == r.logits.size();
  fclose(f);
  return ok;
}

// Teacher-forced comparison of [n][vocab] logits against the reference.
static void print_logit_comparison(const float* ours, const Reference& r, int vocab, int& agree) {
  const int G = (int)r.gen.size();
  int in_top5 = 0;
  double worst_abs = 0, sum_cos = 0, min_cos = 1;
  agree = 0;
  for (int i = 0; i < G; i++) {
    const float* a = ours + (size_t)i * vocab;
    const float* b = &r.logits[(size_t)i * vocab];
    int am = 0;
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (int v = 0; v < vocab; v++) {
      if (a[v] > a[am]) am = v;
      dot += (double)a[v] * b[v];
      na += (double)a[v] * a[v];
      nb += (double)b[v] * b[v];
      mx = std::max(mx, (double)fabsf(a[v] - b[v]));
    }
    int rank = 0;
    for (int v = 0; v < vocab; v++)
      if (b[v] > b[am]) rank++;
    const double cs = dot / sqrt(na * nb);
    agree += am == r.gen[i];
    in_top5 += rank < 5;
    worst_abs = std::max(worst_abs, mx);
    sum_cos += cs;
    min_cos = std::min(min_cos, cs);
  }
  printf("{\"compare\":\"teacher_forced\",\"prompt\":%zu,\"steps\":%d,\"argmax_agree\":%d,\"ours_in_ref_top5\":%d,"
         "\"cosine_mean\":%.6f,\"cosine_min\":%.6f,\"max_abs_logit_diff\":%.3f}\n",
         r.prompt.size(), G, agree, in_top5, sum_cos / G, min_cos, worst_abs);
}
