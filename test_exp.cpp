#include <unistd.h>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>
#include <immintrin.h>
#include <sys/syscall.h>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

#define ALWAYS_INLINE __attribute__((always_inline)) inline
// #define ALWAYS_INLINE __attribute__((noinline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

#include <stdio.h>
#include <unistd.h>  // Required for sysconf

int test_L2_fetch() {
  long l2_cache_size = sysconf(_SC_LEVEL2_CACHE_SIZE);

  if (l2_cache_size != -1) {
    printf("L2 Cache Size: %ld bytes\n", l2_cache_size);
  } else {
    perror("Failed to get L2 cache size");
    return 1;
  }

  return 0;
}

int64_t calcu_default_tile_size(int64_t cache_size, int64_t head_dim,
                                int64_t elem_size, int64_t max_num_q_per_iter,
                                int64_t round_size) {
  // For CPU, different from CUDA, Q@K^T results should also be hold in cache,
  // using float32. Intermediate outputs should be float32 to be compatible with
  // AMX Then the cache includes:
  //  - Q: q_tile_size * head_dim * sizeof(qkv_compute_t)
  //  - K, V: 2 * k_tile_size * head_dim * sizeof(qkv_compute_t)
  //  - Q@K^T: max_num_q_per_iter * k_tile_size * 4
  //  - Intermediate outputs: q_tile_size * head_dim * 4, stored as float
  // By default, let tile_size = q_tile_size = k_tile_size

  int64_t tile_size = cache_size / (3 * head_dim * elem_size +
                                    4 * max_num_q_per_iter + 4 * head_dim);
  int64_t rounded_tile_size = (tile_size / round_size) * round_size;
  return std::max(rounded_tile_size, round_size);
}

static int64_t calcu_tile_size_with_constant_q(
    int64_t cache_size, int64_t head_dim, int64_t elem_size,
    int64_t max_num_q_per_iter, int64_t round_size, int64_t q_tile_size,
    bool one_round) {
  // calculate tile_size with known q_tile_size
  // If one_round is True, the outer Q tile loop time is 1, then the K,V will
  // not be included in the cache
  int64_t tile_size;
  if (one_round) {
    tile_size = (cache_size - q_tile_size * head_dim * (4 + elem_size)) /
                (4 * max_num_q_per_iter);
  } else {
    tile_size = (cache_size - q_tile_size * head_dim * (4 + elem_size)) /
                (4 * max_num_q_per_iter + 2 * head_dim * elem_size);
  }
  int64_t rounded_tile_size = (tile_size / round_size) * round_size;
  return std::max(rounded_tile_size, round_size);
}

void test_default_tile_size_calcu() {
  std::printf(
      "round_size, \telem_size, \tcache_size, \thidden_size, \ttile_size\n");
  for (int64_t round_size : {16, 32}) {
    for (int64_t elem_size : {2, 4}) {
      for (int64_t cache_size : {128, 256, 512, 1024, 2048}) {
        cache_size >>= 1;
        cache_size *= 1024;
        for (int64_t hidden_size = 32; hidden_size < 298; hidden_size += 32) {
          int64_t tile_size = calcu_default_tile_size(
              cache_size, hidden_size, elem_size, round_size, round_size);
          std::printf("%ld, \t%ld, \t%ld, \t%ld, \t%ld\n", round_size,
                      elem_size, cache_size, hidden_size, tile_size);
        }
      }
    }
  }
}

void test_calcu_tile_size_with_constant_q() {
  std::printf(
      "round_size, \telem_size, \tcache_size, \thidden_size, \tq_len, "
      "\ttile_size\n");
  for (int64_t round_size : {16, 32}) {
    for (int64_t elem_size : {2, 4}) {
      for (int64_t cache_size : {128, 256, 512, 1024, 2048}) {
        cache_size >>= 1;
        cache_size *= 1024;
        for (int64_t hidden_size = 32; hidden_size < 298; hidden_size += 32) {
          for (int64_t q_len : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
            bool one_round = q_len <= round_size;
            int64_t tile_size = calcu_tile_size_with_constant_q(
                cache_size, hidden_size, elem_size, round_size, round_size,
                q_len, one_round);
            std::printf("%ld, \t%ld, \t%ld, \t%ld, \t%ld, \t%ld\n", round_size,
                        elem_size, cache_size, hidden_size, q_len, tile_size);
          }
        }
      }
    }
  }
}

typedef struct __tile_config {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};
} __tilecfg;

// Function to get the timestamp using RDTSCP
ALWAYS_INLINE uint64_t bench_timestamp() {
  unsigned int cycles_low, cycles_high;
  asm volatile(
      ".intel_syntax noprefix\n\t"
      "CPUID\n\t"        // Serialize instruction stream to ensure previous
                         // instructions complete
      "RDTSCP\n\t"       // Read TSC and core ID
      "mov %0, edx\n\t"  // Store high 32 bits of TSC
      "mov %1, eax\n\t"  // Store low 32 bits of TSC
      ".att_syntax"
      : "=r"(cycles_high), "=r"(cycles_low)::"rax", "rbx", "rcx",
        "rdx"  // Clobbered registers
  );
  return (uint64_t)cycles_high << 32 | cycles_low;
}

bool set_tiledata_use() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
    return false;
  } else {
    return true;
  }

  return true;
}

template <typename T>
void print_tile(const char* name, const T* tile_data) {
  constexpr size_t elems_per_row = 64 / sizeof(T);
  std::stringstream ss;
  ss << name << ":[\n";
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < elems_per_row; ++j) {
      ss << std::to_string(tile_data[i * elems_per_row + j]) << ',';
    }
    ss << '\n';
  }
  ss << "]\n";
  printf("%s\n", ss.str().c_str());
}

void test_ldconfig(size_t times) {
  __tile_config config;
  for (size_t i = 0; i < 1; ++i) {
    config.rows[i] = 16;
    config.colsb[i] = 64;
  }

  for (size_t i = 0; i < times; ++i) {
    _tile_loadconfig(&config);
  }
  _tile_release();
}

void test_ldconfig_change(size_t times) {
  int8_t data[1024];
  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 64; ++j) {
      data[i * 64 + j] = i + 1;
    }
  }

  __tile_config config;

  for (size_t i = 0; i < times; ++i) {
    {
      config.rows[0] = 1;
      config.colsb[0] = 64;
      _tile_loadconfig(&config);
      _tile_stream_loadd(0, data, 64);
      int8_t output[1024] = {0};
      _tile_stored(0, output, 64);
      print_tile("tile0-1", (int8_t*)output);
    }

    {
      config.rows[0] = 2;
      config.colsb[0] = 64;
      _tile_loadconfig(&config);
      _tile_stream_loadd(0, data, 64);
      int8_t output[1024] = {0};
      _tile_stored(0, output, 64);
      //   print_tile("tile0-2", (int8_t*)output);
    }
  }
  _tile_release();
}

void test_tile_load_dynamic_m(size_t times, int8_t m) {
  int8_t data[1024];
  __tile_config config;
  config.rows[0] = m;
  config.colsb[0] = 64;
  _tile_loadconfig(&config);

  for (size_t i = 0; i < times; ++i) {
    _tile_stream_loadd(0, data, 64);
  }
  _tile_release();
}

void test_tile_dot(size_t times) {
  int8_t a[1024], b[1024];
  int32_t ref_c[256] = {0};

  for (size_t i = 0; i < 16; ++i) {
    for (size_t j = 0; j < 64; ++j) {
      a[i * 64 + j] = i;
      b[i * 64 + j] = i;
    }
  }
  // print_tile("a", a);
  // print_tile("b", b);

  for (size_t k = 0; k < 64; ++k) {
    for (size_t m = 0; m < 16; ++m) {
      int8_t mv = a[m * 64 + k];
      for (size_t n = 0; n < 16; ++n) {
        int8_t nv = b[n * 64 + k];
        ref_c[m * 16 + n] += (int32_t)(mv) * (int32_t)(nv);
      }
    }
  }
  // print_tile("ref_c", ref_c);

  int8_t packed_b[1024];
  int32_t* origin_b_32 = reinterpret_cast<int32_t*>(b);
  int32_t* packed_b_32 = reinterpret_cast<int32_t*>(packed_b);
  for (size_t m = 0; m < 16; ++m) {
    for (size_t n = 0; n < 16; ++n) {
      packed_b_32[n * 16 + m] = origin_b_32[m * 16 + n];
    }
  }
  // print_tile("packed_b", packed_b);

  __tile_config config;
  config.rows[0] = 16;
  config.colsb[0] = 64;
  config.rows[1] = 16;
  config.colsb[1] = 64;
  config.rows[2] = 16;
  config.colsb[2] = 64;
  _tile_loadconfig(&config);
  int32_t amx_c[1024];

  for (size_t i = 0; i < times; ++i) {
    _tile_zero(2);
    _tile_stream_loadd(0, a, 64);
    _tile_stream_loadd(1, packed_b, 64);
    _tile_dpbssd(2, 0, 1);
    _tile_stored(2, amx_c, 64);
  }
  // print_tile("amx_c", amx_c);

  _tile_release();
}

void test_tile_dot_dynamic_m(size_t times, int8_t m) {
  int8_t a[1024], b[1024];

  __tile_config config;
  config.rows[0] = m;
  config.colsb[0] = 64;
  config.rows[1] = 16;
  config.colsb[1] = 64;
  config.rows[2] = m;
  config.colsb[2] = 64;
  _tile_loadconfig(&config);
  int32_t amx_c[1024];
  _tile_zero(2);
  _tile_stream_loadd(0, a, 64);
  _tile_stream_loadd(1, b, 64);
  for (size_t i = 0; i < times; ++i) {
    _tile_dpbssd(2, 0, 1);
  }
  _tile_stored(2, amx_c, 64);
  // print_tile("amx_c", amx_c);

  _tile_release();
}

void test_contigous_load(size_t times) {
  alignas(64) int8_t a[1024 * 1024 * 4];

  __tile_config config;
  config.rows[0] = 16;
  config.colsb[0] = 64;
  _tile_loadconfig(&config);

  for (size_t i = 0; i < times; ++i) {
    _tile_loadd(0, a, 64);
  }

  _tile_release();
}

void test_tile_dot_m1(size_t times, int32_t* amx_c) {
  int32_t a[1024], b[1024];

  __tile_config config;
  config.rows[0] = 1;
  config.colsb[0] = 64;
  config.rows[1] = 16;
  config.colsb[1] = 64;
  config.rows[2] = 1;
  config.colsb[2] = 64;
  _tile_loadconfig(&config);
  _tile_zero(2);

  for (size_t i = 0; i < times; ++i) {
    _tile_loadd(0, a, 64);
    _tile_stream_loadd(1, b, 64);
    _tile_dpbf16ps(2, 0, 1);
    _tile_stored(2, amx_c, 64);
  }
  // print_tile("amx_c", amx_c);

  _tile_release();
}

void test_avx_dot_m1(size_t times, int32_t* amx_c) {
  int32_t a[1024], b[1024];
  __m512 c_vec[8];

  for (size_t i = 0; i < times; ++i) {
    __m512i a_vec_bf16[8];
    a_vec_bf16[0] = _mm512_loadu_epi16((void*)a);
// a_vec_bf16[1] = _mm512_loadu_epi16((void*)((int16_t*)a + 32));
// a_vec_bf16[2] = _mm512_loadu_epi16((void*)((int16_t*)a + 64));
// a_vec_bf16[3] = _mm512_loadu_epi16((void*)((int16_t*)a + 96));
// a_vec_bf16[4] = _mm512_loadu_epi16((void*)((int16_t*)a + 128));
// a_vec_bf16[5] = _mm512_loadu_epi16((void*)((int16_t*)a + 160));
// a_vec_bf16[6] = _mm512_loadu_epi16((void*)((int16_t*)a + 192));
// a_vec_bf16[7] = _mm512_loadu_epi16((void*)((int16_t*)a + 224));
#pragma GCC unroll 16
    for (size_t j = 0; j < 512; j += 32) {
      __m512i b_vec_bf16 = _mm512_loadu_epi16((void*)((int16_t*)(b) + j));
      c_vec[0] = _mm512_dpbf16_ps(c_vec[0], (__m512bh)a_vec_bf16[0],
                                  (__m512bh)b_vec_bf16);
      // c_vec[1] = _mm512_dpbf16_ps(c_vec[1], (__m512bh)a_vec_bf16[1],
      // (__m512bh)b_vec_bf16); c_vec[2] = _mm512_dpbf16_ps(c_vec[2],
      // (__m512bh)a_vec_bf16[2], (__m512bh)b_vec_bf16); c_vec[3] =
      // _mm512_dpbf16_ps(c_vec[3], (__m512bh)a_vec_bf16[2],
      // (__m512bh)b_vec_bf16); c_vec[4] = _mm512_dpbf16_ps(c_vec[4],
      // (__m512bh)a_vec_bf16[2], (__m512bh)b_vec_bf16); c_vec[5] =
      // _mm512_dpbf16_ps(c_vec[5], (__m512bh)a_vec_bf16[2],
      // (__m512bh)b_vec_bf16); c_vec[6] = _mm512_dpbf16_ps(c_vec[6],
      // (__m512bh)a_vec_bf16[2], (__m512bh)b_vec_bf16); c_vec[7] =
      // _mm512_dpbf16_ps(c_vec[7], (__m512bh)a_vec_bf16[2],
      // (__m512bh)b_vec_bf16);
    }
    _mm512_storeu_ps((void*)amx_c, c_vec[0]);
    // _mm512_storeu_ps((void*)(amx_c+1), c_vec[1]);
    // _mm512_storeu_ps((void*)(amx_c+2), c_vec[2]);
    // _mm512_storeu_ps((void*)(amx_c+3), c_vec[3]);
    // _mm512_storeu_ps((void*)(amx_c+4), c_vec[4]);
    // _mm512_storeu_ps((void*)(amx_c+5), c_vec[5]);
    // _mm512_storeu_ps((void*)(amx_c+6), c_vec[6]);
    // _mm512_storeu_ps((void*)(amx_c+7), c_vec[7]);
  }
  // print_tile("amx_c", amx_c);
}

void print_execute_time(const char* name, size_t start, size_t end,
                        size_t times) {
  std::printf("%s execute cycles: %f\n", name, (float)(end - start) / times);
}

struct BF16Vec32 {
  __m512i reg;

  explicit BF16Vec32() : reg(_mm512_setzero_si512()) {}

  explicit BF16Vec32(const void* ptr) : reg((__m512i)_mm512_loadu_si512(ptr)) {}

  explicit BF16Vec32(__m512i data) : reg(data) {}

  void save(void* ptr) const { *reinterpret_cast<__m512i*>(ptr) = reg; }
};

struct BF16Vec16 {
  __m256i reg;

  // normal load
  explicit BF16Vec16(const void* ptr)
      : reg((__m256i)_mm256_loadu_si256((__m256i*)ptr)) {}

  // non-temporal load
  explicit BF16Vec16(bool, void* ptr)
      : reg(_mm256_stream_load_si256((__m256i*)ptr)) {}

  void save(void* ptr) const { _mm256_storeu_si256((__m256i*)ptr, reg); }

  void save(void* ptr, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    _mm256_mask_storeu_epi16(ptr, mask, reg);
  }
};

struct FP32Vec16 {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    __m512 reg;
    float values[VEC_ELEM_NUM];
  };

  __m512 reg;

  explicit FP32Vec16(float v) : reg(_mm512_set1_ps(v)) {}

  explicit FP32Vec16() : reg(_mm512_set1_ps(0.0)) {}

  // normal load
  explicit FP32Vec16(const float* ptr) : reg(_mm512_loadu_ps(ptr)) {}

  // non-temporal load
  explicit FP32Vec16(bool, void* ptr)
      : reg((__m512)_mm512_stream_load_si512(ptr)) {}

  explicit FP32Vec16(__m512 data) : reg(data) {}

  explicit FP32Vec16(const BF16Vec16& v)
      : reg(_mm512_castsi512_ps(
            _mm512_bslli_epi128(_mm512_cvtepu16_epi32(v.reg), 2))) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_mul_ps(reg, b.reg));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_add_ps(reg, b.reg));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_sub_ps(reg, b.reg));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_div_ps(reg, b.reg));
  }

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(_mm512_min_ps(max.reg, _mm512_max_ps(min.reg, reg)));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_max_ps(reg, b.reg));
  }

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    return FP32Vec16(_mm512_mask_max_ps(reg, mask, reg, b.reg));
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(_mm512_min_ps(reg, b.reg));
  }

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    return FP32Vec16(_mm512_mask_min_ps(reg, mask, reg, b.reg));
  }

  FP32Vec16 abs() const { return FP32Vec16(_mm512_abs_ps(reg)); }

  float reduce_sum() const { return _mm512_reduce_add_ps(reg); }

  float reduce_max() const { return _mm512_reduce_max_ps(reg); }

  float reduce_min() const { return _mm512_reduce_min_ps(reg); }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr uint32_t base_mask = (0xFFFF >> (16 - group_size));
    __mmask16 mask = _cvtu32_mask16(base_mask << (idx * group_size));
    return _mm512_mask_reduce_add_ps(mask, reg);
  }

  void save(float* ptr) const { _mm512_storeu_ps(ptr, reg); }

  void save(float* ptr, const int elem_num) const {
    constexpr uint32_t M = 0xFFFFFFFF;
    __mmask16 mask = _cvtu32_mask16(M >> (32 - elem_num));
    _mm512_mask_storeu_ps(ptr, mask, reg);
  }
};

std::vector<float> generate_input(size_t num) {
  std::default_random_engine generator(11451);
  std::uniform_real_distribution<float> distribution_0_to_1(0.0f, 1.0f);
  std::vector<float> data(num);
  for (size_t i = 0; i < num; ++i) {
    float random_float_0_to_1 = distribution_0_to_1(generator);
    data[i] = random_float_0_to_1 * 30 - 30;
  }
  return data;
}

void generate_input(float* ptr, size_t num) {
  std::default_random_engine generator(11451);
  std::uniform_real_distribution<float> distribution_0_to_1(0.0f, 1.0f);
  for (size_t i = 0; i < num; ++i) {
    float random_float_0_to_1 = distribution_0_to_1(generator);
    ptr[i] = random_float_0_to_1;
  }
}

template <typename T>
void print_vector(const std::vector<T>& data) {
  std::stringstream ss;
  ss << '[';
  for (auto v : data) {
    ss << v << ',';
  }
  ss << ']';
  std::printf("%s\n", ss.str().c_str());
}

ALWAYS_INLINE void naive_exp(const std::vector<float>& input,
                             std::vector<float>& output) {
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = expf(input[i]);
  }
}

ALWAYS_INLINE void naive_exp2(const std::vector<float>& input,
                              std::vector<float>& output) {
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = exp2f(input[i]);
  }
}

ALWAYS_INLINE __m512 exp_u20(__m512 values) {
  // A faster version of exp with ULP=20
  const __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f);  // 1/factorial(1)
  const __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f);  // 1/factorial(2)
  const __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f);  // 1/factorial(3)
  const __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f);  // 1/factorial(4)
  const __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f);  // 1/factorial(5)
  const __m512 vec_exp_log2ef =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));  // log2(e)
  const __m512 vec_half = _mm512_set1_ps(0.5f);
  const __m512 vec_one = _mm512_set1_ps(1.f);
  const __m512 vec_zero = _mm512_set1_ps(0.f);
  const __m512 vec_two = _mm512_set1_ps(2.f);
  const __m512 vec_ln2f =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));  // ln(2)
  const __m512 vec_ln_flt_min =
      _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
  const __m512 vec_ln_flt_max =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
  const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  const int n_mantissa_bits = 23;

  // exp(x) =
  // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
  // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
  vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundps_epi32(
      vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
  vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
  vec_two_pow_n =
      _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ps(vec_res, vec_two);
  return vec_res;
}

ALWAYS_INLINE void torch_exp(const std::vector<float>& input,
                             std::vector<float>& output) {
  const float* data = input.data();
  float* output_ptr = output.data();
  for (size_t i = 0; i < input.size(); i += 16) {
    FP32Vec16 vec(data + i);
    FP32Vec16 res(exp_u20(vec.reg));
    res.save(output_ptr + i);
  }
}

ALWAYS_INLINE void torch_unroll_exp(const std::vector<float>& input,
                                    std::vector<float>& output) {
  const float* data = input.data();
  float* output_ptr = output.data();
  // A faster version of exp with ULP=20
  const __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f);  // 1/factorial(1)
  const __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f);  // 1/factorial(2)
  const __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f);  // 1/factorial(3)
  const __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f);  // 1/factorial(4)
  const __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f);  // 1/factorial(5)
  const __m512 vec_exp_log2ef =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));  // log2(e)
  const __m512 vec_half = _mm512_set1_ps(0.5f);
  const __m512 vec_one = _mm512_set1_ps(1.f);
  const __m512 vec_zero = _mm512_set1_ps(0.f);
  const __m512 vec_two = _mm512_set1_ps(2.f);
  const __m512 vec_ln2f =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));  // ln(2)
  const __m512 vec_ln_flt_min =
      _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
  const __m512 vec_ln_flt_max =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
  const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  const int n_mantissa_bits = 23;
  for (size_t i = 0; i < input.size(); i += 16) {
    FP32Vec16 vec(data + i);

    __m512 values = vec.reg;
    auto less_ln_flt_min_mask =
        _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
    auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
    vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

    // fx = floorf(x * log2ef + 0.5)
    auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
    auto vec_fx_i = _mm512_cvt_roundps_epi32(
        vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

    // x = x - fx * ln2
    auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

    // compute polynomial
    auto vec_res =
        _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
    vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

    // compute 2^(n-1)
    auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
    auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
    auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
    vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
    auto vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
    vec_two_pow_n =
        _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

    // y = y * 2^n
    vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
    vec_res = _mm512_mul_ps(vec_res, vec_two);

    FP32Vec16 res(vec_res);
    res.save(output_ptr + i);
  }
}

ALWAYS_INLINE void naive_tanh(const std::vector<float>& input,
                              std::vector<float>& output) {
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = tanhf(input[i]);
  }
}

ALWAYS_INLINE __m512 tanh_u20(__m512 values) {
  const __m512 vec_factorial_1 =
      _mm512_set1_ps(0.999999701f);  // 1/factorial(1)
  const __m512 vec_factorial_2 =
      _mm512_set1_ps(0.499991506f);  // 1/factorial(2)
  const __m512 vec_factorial_3 =
      _mm512_set1_ps(0.166676521f);  // 1/factorial(3)
  const __m512 vec_factorial_4 =
      _mm512_set1_ps(0.0418978221f);  // 1/factorial(4)
  const __m512 vec_factorial_5 =
      _mm512_set1_ps(0.00828929059f);  // 1/factorial(5)
  const __m512 vec_exp_log2ef =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));  // log2(e)
  const __m512 vec_half = _mm512_set1_ps(0.5f);
  const __m512 vec_one = _mm512_set1_ps(1.f);
  const __m512 vec_zero = _mm512_set1_ps(0.f);
  const __m512 vec_two = _mm512_set1_ps(2.f);
  const __m512 vec_ln2f =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));  // ln(2)
  const __m512 vec_ln_flt_min =
      _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
  const __m512 vec_ln_flt_max =
      _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
  const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
  const int n_mantissa_bits = 23;

  values = _mm512_mul_ps(values, vec_two);

  auto less_ln_flt_min_mask =
      _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);
  auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);
  vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

  // fx = floorf(x * log2ef + 0.5)
  auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
  auto vec_fx_i = _mm512_cvt_roundps_epi32(
      vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

  // x = x - fx * ln2
  auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

  // compute polynomial
  auto vec_res =
      _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
  vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

  // compute 2^(n-1)
  auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
  auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
  auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
  vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
  auto vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
  vec_two_pow_n =
      _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

  // y = y * 2^n
  vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
  vec_res = _mm512_mul_ps(vec_res, vec_two);

  auto vec_down = _mm512_add_ps(vec_res, vec_one);
  auto vec_upper = _mm512_sub_ps(vec_res, vec_one);
  vec_res = _mm512_div_ps(vec_upper, vec_down);

  return vec_res;
}

ALWAYS_INLINE void torch_tanh(const std::vector<float>& input,
                              std::vector<float>& output) {
  const float* data = input.data();
  float* output_ptr = output.data();
  for (size_t i = 0; i < input.size(); i += 16) {
    FP32Vec16 vec(data + i);
    FP32Vec16 res(tanh_u20(vec.reg));
    res.save(output_ptr + i);
  }
}

template <typename T, typename F>
ALWAYS_INLINE void execute(size_t times, F&& func, const std::vector<T>& input,
                           std::vector<T>& output) {
  for (size_t i = 0; i < times; ++i) {
    func(input, output);
  }
}

void print_float_matrix(float* ptr, int32_t m, int32_t n, const char* name) {
  std::stringstream ss;
  ss << name << ": [";
  for (int i = 0; i < m; ++i) {
    float* curr_ptr = ptr + n * i;
    for (int j = 0; j < n; ++j) {
      ss << curr_ptr[j] << ',';
    }
    ss << '\n';
  }
  ss << ']';
  std::printf("%s\n", ss.str().c_str());
}

void print_bf16_matrix(uint16_t* ptr, int32_t m, int32_t n, const char* name) {
  std::stringstream ss;
  ss << name << ": [";
  for (int i = 0; i < m; ++i) {
    uint16_t* curr_ptr = ptr + n * i;
    for (int j = 0; j < n; ++j) {
      uint16_t val = curr_ptr[j];
      ss << _mm_cvtsbh_ss(val) << ',';
    }
    ss << '\n';
  }
  ss << ']';
  std::printf("%s\n", ss.str().c_str());
}

void check_c(float* __restrict__ ref_c, float* __restrict__ c, int32_t num) {
  for (int32_t i = 0; i < num; ++i) {
    if (abs(ref_c[i] - c[i]) > 1e-4) {
      std::printf("abnormal value: %f, %f\n", ref_c[i], c[i]);
    }
  }
}

// Benchmark AMXBF16 performance
// a: [m, k], row-major
// b: [k, n], row-major
// c: [m, n], row-major
template <int32_t M, int32_t K, int32_t N>
class AMXBF16Benchmark {
  static_assert(16 < M <= 32);
  static_assert(K % 32 == 0);
  static_assert(N % 32 == 0);

  // AMX specific
  constexpr static int64_t AMX_TILE_ROW_BYTES = 64;
  constexpr static int64_t AMX_TILE_ROW_NUM = 16;
  constexpr static int64_t AMX_TILE_BYTES =
      AMX_TILE_ROW_BYTES * AMX_TILE_ROW_NUM;

 public:
  template <int32_t M_tile>
  void gemm_avxbf16_micro(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                          float* __restrict__ c, int32_t n_group_num) {
    static_assert(0 < M_tile <= 8);

    uint16_t* __restrict__ curr_b_0 = b;
    uint16_t* __restrict__ curr_b_1 = b + 32;
    float* __restrict__ curr_c_0 = c;
    float* __restrict__ curr_c_1 = c + 16;
    for (int32_t n_group = 0; n_group < n_group_num; ++n_group) {
      FP32Vec16 c_regs[M_tile * 2];
      uint32_t* __restrict__ curr_a = reinterpret_cast<uint32_t*>(a);
      uint16_t* __restrict__ curr_k_b_0 = curr_b_0;
      uint16_t* __restrict__ curr_k_b_1 = curr_b_1;
      for (int32_t k = 0; k < K / 2; ++k) {
        BF16Vec32 b_0_reg(curr_k_b_0);
        BF16Vec32 b_1_reg(curr_k_b_1);

        uint32_t* __restrict__ curr_m_a = curr_a;
        unroll_loop<int32_t, M_tile>([&](int32_t i) {
          uint32_t v = *curr_m_a;
          __m512i a_reg = _mm512_set1_epi32(v);
          c_regs[i * 2].reg = _mm512_dpbf16_ps(
              c_regs[i * 2].reg, (__m512bh)a_reg, (__m512bh)b_0_reg.reg);
          c_regs[i * 2 + 1].reg = _mm512_dpbf16_ps(
              c_regs[i * 2 + 1].reg, (__m512bh)a_reg, (__m512bh)b_1_reg.reg);

          // update
          curr_m_a += K / 2;
        });

        // update
        curr_a += 1;
        curr_k_b_0 += 2 * N;
        curr_k_b_1 += 2 * N;
      }

      float* __restrict__ curr_m_c_0 = curr_c_0;
      float* __restrict__ curr_m_c_1 = curr_c_1;
      unroll_loop<int32_t, M_tile>([&](int32_t i) {
        c_regs[i * 2].save(curr_m_c_0);
        c_regs[i * 2 + 1].save(curr_m_c_1);

        // update
        curr_m_c_0 += N;
        curr_m_c_1 += N;
      });

      // update
      curr_b_0 += 64;
      curr_b_1 += 64;
      curr_c_0 += 32;
      curr_c_1 += 32;
    }
  }

  void gemm_avxbf16(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                    float* __restrict__ c) {
    constexpr int32_t n_group_num = N / 32;

    uint16_t* __restrict__ curr_a = a;
    float* __restrict__ curr_c = c;
    // M is splited as tiles with at most 8
    for (int32_t m = 0; m < M; m += 8) {
      int32_t actual_m = std::min(8, M - m);
      switch (actual_m) {
        case 1:
          gemm_avxbf16_micro<1>(curr_a, b, curr_c, n_group_num);
          break;
        case 2:
          gemm_avxbf16_micro<2>(curr_a, b, curr_c, n_group_num);
          break;
        case 3:
        case 4:
          gemm_avxbf16_micro<4>(curr_a, b, curr_c, n_group_num);
          break;
        case 5:
        case 6:
        case 7:
        case 8:
          gemm_avxbf16_micro<8>(curr_a, b, curr_c, n_group_num);
          break;
      }

      // update
      curr_a += 8 * K;
      curr_c += 8 * N;
    }
  }

  template <int32_t M_tile>
  void gemm_avx_micro(float* __restrict__ a, uint16_t* __restrict__ b,
                      float* __restrict__ c, int32_t n_group_num) {
    static_assert(0 < M_tile <= 8);

    uint16_t* __restrict__ curr_b_0 = b;
    uint16_t* __restrict__ curr_b_1 = b + 16;
    float* __restrict__ curr_c_0 = c;
    float* __restrict__ curr_c_1 = c + 16;
    for (int32_t n_group = 0; n_group < n_group_num; ++n_group) {
      FP32Vec16 c_regs[M_tile * 2];
      float* __restrict__ curr_a = a;
      uint16_t* __restrict__ curr_k_b_0 = curr_b_0;
      uint16_t* __restrict__ curr_k_b_1 = curr_b_1;
      for (int32_t k = 0; k < K; ++k) {
        BF16Vec16 b_0_reg(curr_k_b_0);
        FP32Vec16 fp32_b_0_reg(b_0_reg);
        BF16Vec16 b_1_reg(curr_k_b_1);
        FP32Vec16 fp32_b_1_reg(b_1_reg);

        float* __restrict__ curr_m_a = curr_a;
        unroll_loop<int32_t, M_tile>([&](int32_t i) {
          float v = *curr_m_a;
          FP32Vec16 a_reg(v);
          c_regs[i * 2] = c_regs[i * 2] + a_reg * fp32_b_0_reg;
          c_regs[i * 2 + 1] = c_regs[i * 2 + 1] + a_reg * fp32_b_1_reg;

          // update
          curr_m_a += K;
        });

        // update
        curr_a += 1;
        curr_k_b_0 += N;
        curr_k_b_1 += N;
      }

      float* __restrict__ curr_m_c_0 = curr_c_0;
      float* __restrict__ curr_m_c_1 = curr_c_1;
      unroll_loop<int32_t, M_tile>([&](int32_t i) {
        c_regs[i * 2].save(curr_m_c_0);
        c_regs[i * 2 + 1].save(curr_m_c_1);

        // update
        curr_m_c_0 += N;
        curr_m_c_1 += N;
      });

      // update
      curr_b_0 += 32;
      curr_b_1 += 32;
      curr_c_0 += 32;
      curr_c_1 += 32;
    }
  }

  void gemm_avx(float* __restrict__ a, uint16_t* __restrict__ b,
                float* __restrict__ c) {
    constexpr int32_t n_group_num = N / 32;

    float* __restrict__ curr_a = a;
    float* __restrict__ curr_c = c;
    // M is splited as tiles with at most 8
    for (int32_t m = 0; m < M; m += 8) {
      int32_t actual_m = std::min(8, M - m);
      switch (actual_m) {
        case 1:
          gemm_avx_micro<1>(curr_a, b, curr_c, n_group_num);
          break;
        case 2:
          gemm_avx_micro<2>(curr_a, b, curr_c, n_group_num);
          break;
        case 3:
        case 4:
          gemm_avx_micro<4>(curr_a, b, curr_c, n_group_num);
          break;
        case 5:
        case 6:
        case 7:
        case 8:
          gemm_avx_micro<8>(curr_a, b, curr_c, n_group_num);
          break;
      }

      // update
      curr_a += 8 * K;
      curr_c += 8 * N;
    }
  }

  void gemm_v0_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                    float* __restrict__ c) {
    if constexpr (M > AMX_TILE_ROW_NUM) {
      gemm224_v0_pack(a, b, c);
    } else {
      gemm122_v0_pack(a, b, c);
    }
  }

  void gemm_v1_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                    float* __restrict__ c) {
    if constexpr (M > AMX_TILE_ROW_NUM) {
      gemm224_v1_pack(a, b, c);
    } else {
      gemm122_v1_pack(a, b, c);
    }
  }

  void gemm224_v0_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                       float* __restrict__ c) {
    uint16_t* __restrict__ a_tile_0 = a;
    uint16_t* __restrict__ a_tile_1 = a + AMX_TILE_ROW_NUM * K;
    int64_t a_stride = K * sizeof(uint16_t);

    uint16_t* __restrict__ b_tile_2 = b;
    uint16_t* __restrict__ b_tile_3 = b + AMX_TILE_ROW_BYTES / sizeof(uint16_t);
    int64_t b_stride = N * 4;

    float* __restrict__ c_tile_4 = c;
    float* __restrict__ c_tile_5 = c + AMX_TILE_ROW_BYTES / sizeof(float);
    float* __restrict__ c_tile_6 = c + AMX_TILE_ROW_NUM * N;
    float* __restrict__ c_tile_7 =
        c + AMX_TILE_ROW_NUM * N + AMX_TILE_ROW_BYTES / sizeof(float);
    int64_t c_stride = N * sizeof(float);

    constexpr int32_t n_times = N / (2 * AMX_TILE_ROW_BYTES / 4);
    constexpr int32_t k_times = K / (AMX_TILE_ROW_NUM * 4 / sizeof(uint16_t));
    for (int32_t n = 0; n < n_times; ++n) {
      uint16_t* __restrict__ curr_a_tile_0 = a_tile_0;
      uint16_t* __restrict__ curr_a_tile_1 = a_tile_1;
      uint16_t* __restrict__ curr_b_tile_2 = b_tile_2;
      uint16_t* __restrict__ curr_b_tile_3 = b_tile_3;

      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6);
      _tile_zero(7);

      for (int32_t k = 0; k < k_times; ++k) {
        _tile_loadd(0, curr_a_tile_0, a_stride);
        _tile_stream_loadd(2, curr_b_tile_2, b_stride);
        _tile_dpbf16ps(4, 0, 2);
        _tile_stream_loadd(3, curr_b_tile_3, b_stride);
        _tile_dpbf16ps(5, 0, 3);
        _tile_loadd(1, curr_a_tile_1, a_stride);
        _tile_dpbf16ps(6, 1, 2);
        _tile_dpbf16ps(7, 1, 3);

        // {
        //     float tmp[AMX_TILE_ROW_NUM][16] = {0};
        //     _tile_stored(4, tmp, 64);
        //     print_float_matrix((float*)tmp, AMX_TILE_ROW_NUM, 16, "tile_4");
        // }

        // update ptrs
        curr_a_tile_0 += AMX_TILE_ROW_BYTES / sizeof(uint16_t);
        curr_a_tile_1 += AMX_TILE_ROW_BYTES / sizeof(uint16_t);
        curr_b_tile_2 += AMX_TILE_ROW_NUM * 2 * N;
        curr_b_tile_3 += AMX_TILE_ROW_NUM * 2 * N;
      }

      _tile_stored(4, c_tile_4, c_stride);
      _tile_stored(5, c_tile_5, c_stride);
      _tile_stored(6, c_tile_6, c_stride);
      _tile_stored(7, c_tile_7, c_stride);

      // update ptrs
      b_tile_2 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
      b_tile_3 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
      c_tile_4 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_5 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_6 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_7 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
    }
  }

  void gemm122_v0_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                       float* __restrict__ c) {
    uint16_t* __restrict__ a_tile_0 = a;
    uint16_t* __restrict__ a_tile_1 = a + AMX_TILE_ROW_BYTES / sizeof(uint16_t);
    int64_t a_stride = K * sizeof(uint16_t);

    uint16_t* __restrict__ b_tile_2 = b;
    uint16_t* __restrict__ b_tile_3 = b + AMX_TILE_ROW_BYTES / sizeof(uint16_t);
    uint16_t* __restrict__ b_tile_4 = b_tile_2 + AMX_TILE_ROW_NUM * 2 * N;
    uint16_t* __restrict__ b_tile_5 = b_tile_3 + AMX_TILE_ROW_NUM * 2 * N;
    int64_t b_stride = N * 4;

    float* __restrict__ c_tile_6 = c;
    float* __restrict__ c_tile_7 = c + AMX_TILE_ROW_BYTES / sizeof(float);
    int64_t c_stride = N * sizeof(float);

    constexpr int32_t n_times = N / (2 * AMX_TILE_ROW_BYTES / 4);
    constexpr int32_t k_times = K / (AMX_TILE_ROW_NUM * 4 / sizeof(uint16_t));
    constexpr int32_t k_group_times = k_times / 2;
    constexpr bool has_tile_loop = (k_times % 2 == 1);
    for (int32_t n = 0; n < n_times; ++n) {
      uint16_t* __restrict__ curr_a_tile_0 = a_tile_0;
      uint16_t* __restrict__ curr_a_tile_1 = a_tile_1;
      uint16_t* __restrict__ curr_b_tile_2 = b_tile_2;
      uint16_t* __restrict__ curr_b_tile_3 = b_tile_3;
      uint16_t* __restrict__ curr_b_tile_4 = b_tile_4;
      uint16_t* __restrict__ curr_b_tile_5 = b_tile_5;

      _tile_zero(6);
      _tile_zero(7);

      for (int32_t k = 0; k < k_group_times; ++k) {
        _tile_loadd(0, curr_a_tile_0, a_stride);
        _tile_stream_loadd(2, curr_b_tile_2, b_stride);
        _tile_dpbf16ps(6, 0, 2);
        _tile_stream_loadd(3, curr_b_tile_3, b_stride);
        _tile_dpbf16ps(7, 0, 3);
        _tile_loadd(1, curr_a_tile_1, a_stride);
        _tile_stream_loadd(4, curr_b_tile_4, b_stride);
        _tile_dpbf16ps(6, 1, 4);
        _tile_stream_loadd(5, curr_b_tile_5, b_stride);
        _tile_dpbf16ps(7, 1, 5);

        // {
        //     float tmp[AMX_TILE_ROW_NUM][16] = {0};
        //     _tile_stored(4, tmp, 64);
        //     print_float_matrix((float*)tmp, AMX_TILE_ROW_NUM, 16, "tile_4");
        // }

        // update ptrs
        curr_a_tile_0 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
        curr_a_tile_1 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
        curr_b_tile_2 += 2 * AMX_TILE_ROW_NUM * 2 * N;
        curr_b_tile_3 += 2 * AMX_TILE_ROW_NUM * 2 * N;
        curr_b_tile_4 += 2 * AMX_TILE_ROW_NUM * 2 * N;
        curr_b_tile_5 += 2 * AMX_TILE_ROW_NUM * 2 * N;
      }

      if constexpr (has_tile_loop) {
        _tile_loadd(0, curr_a_tile_0, a_stride);
        _tile_stream_loadd(2, curr_b_tile_2, b_stride);
        _tile_dpbf16ps(6, 0, 2);
        _tile_stream_loadd(3, curr_b_tile_3, b_stride);
        _tile_dpbf16ps(7, 0, 3);
      }

      _tile_stored(6, c_tile_6, c_stride);
      _tile_stored(7, c_tile_7, c_stride);

      // update ptrs
      b_tile_2 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
      b_tile_3 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
      b_tile_4 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
      b_tile_5 += 2 * AMX_TILE_ROW_BYTES / sizeof(uint16_t);
      c_tile_6 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_7 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
    }
  }

  void gemm224_v1_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                       float* __restrict__ c) {
    uint16_t* __restrict__ a_tile_0 = a;
    uint16_t* __restrict__ a_tile_1 = a + AMX_TILE_ROW_NUM * K;
    int64_t a_stride = 64;

    uint16_t* __restrict__ b_tile_2 = b;
    uint16_t* __restrict__ b_tile_3 = b + (AMX_TILE_ROW_BYTES / 4) * K;
    int64_t b_stride = 64;

    float* __restrict__ c_tile_4 = c;
    float* __restrict__ c_tile_5 = c + AMX_TILE_ROW_BYTES / sizeof(float);
    float* __restrict__ c_tile_6 = c + AMX_TILE_ROW_NUM * N;
    float* __restrict__ c_tile_7 =
        c + AMX_TILE_ROW_NUM * N + AMX_TILE_ROW_BYTES / sizeof(float);
    int64_t c_stride = N * sizeof(float);

    constexpr int32_t n_times = N / (2 * AMX_TILE_ROW_BYTES / 4);
    constexpr int32_t k_times = K / (AMX_TILE_ROW_NUM * 4 / sizeof(uint16_t));
    for (int32_t n = 0; n < n_times; ++n) {
      uint16_t* __restrict__ curr_a_tile_0 = a_tile_0;
      uint16_t* __restrict__ curr_a_tile_1 = a_tile_1;
      uint16_t* __restrict__ curr_b_tile_2 = b_tile_2;
      uint16_t* __restrict__ curr_b_tile_3 = b_tile_3;

      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6);
      _tile_zero(7);

      for (int32_t k = 0; k < k_times; ++k) {
        _tile_loadd(0, curr_a_tile_0, a_stride);
        _tile_stream_loadd(2, curr_b_tile_2, b_stride);
        _tile_dpbf16ps(4, 0, 2);
        _tile_stream_loadd(3, curr_b_tile_3, b_stride);
        _tile_dpbf16ps(5, 0, 3);
        _tile_loadd(1, curr_a_tile_1, a_stride);
        _tile_dpbf16ps(6, 1, 2);
        _tile_dpbf16ps(7, 1, 3);

        // {
        //     float tmp[AMX_TILE_ROW_NUM][16] = {0};
        //     _tile_stored(4, tmp, 64);
        //     print_float_matrix((float*)tmp, AMX_TILE_ROW_NUM, 16, "tile_4");
        // }

        // update ptrs
        curr_a_tile_0 += AMX_TILE_BYTES / sizeof(uint16_t);
        curr_a_tile_1 += AMX_TILE_BYTES / sizeof(uint16_t);
        curr_b_tile_2 += AMX_TILE_BYTES / sizeof(uint16_t);
        curr_b_tile_3 += AMX_TILE_BYTES / sizeof(uint16_t);
      }

      _tile_stored(4, c_tile_4, c_stride);
      _tile_stored(5, c_tile_5, c_stride);
      _tile_stored(6, c_tile_6, c_stride);
      _tile_stored(7, c_tile_7, c_stride);

      // update ptrs
      b_tile_2 += 2 * (AMX_TILE_ROW_BYTES / 4) * K;
      b_tile_3 += 2 * (AMX_TILE_ROW_BYTES / 4) * K;
      c_tile_4 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_5 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_6 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_7 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
    }
  }

  void gemm122_v1_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ b,
                       float* __restrict__ c) {
    uint16_t* __restrict__ a_tile_0 = a;
    uint16_t* __restrict__ a_tile_1 = a + AMX_TILE_BYTES / sizeof(uint16_t);
    int64_t a_stride = 64;

    uint16_t* __restrict__ b_tile_2 = b;
    uint16_t* __restrict__ b_tile_3 = b + (AMX_TILE_ROW_BYTES / 4) * K;
    uint16_t* __restrict__ b_tile_4 =
        b_tile_2 + AMX_TILE_BYTES / sizeof(uint16_t);
    uint16_t* __restrict__ b_tile_5 =
        b_tile_3 + AMX_TILE_BYTES / sizeof(uint16_t);
    int64_t b_stride = 64;

    float* __restrict__ c_tile_6 = c;
    float* __restrict__ c_tile_7 = c + AMX_TILE_ROW_BYTES / sizeof(float);
    int64_t c_stride = N * sizeof(float);

    constexpr int32_t n_times = N / (2 * AMX_TILE_ROW_BYTES / 4);
    constexpr int32_t k_times = K / (AMX_TILE_ROW_NUM * 4 / sizeof(uint16_t));
    constexpr int32_t k_group_times = k_times / 2;
    constexpr bool has_tile_loop = (k_times % 2 == 1);
    for (int32_t n = 0; n < n_times; ++n) {
      uint16_t* __restrict__ curr_a_tile_0 = a_tile_0;
      uint16_t* __restrict__ curr_a_tile_1 = a_tile_1;
      uint16_t* __restrict__ curr_b_tile_2 = b_tile_2;
      uint16_t* __restrict__ curr_b_tile_3 = b_tile_3;
      uint16_t* __restrict__ curr_b_tile_4 = b_tile_4;
      uint16_t* __restrict__ curr_b_tile_5 = b_tile_5;

      _tile_zero(6);
      _tile_zero(7);

      for (int32_t k = 0; k < k_group_times; ++k) {
        _tile_loadd(0, curr_a_tile_0, a_stride);
        _tile_stream_loadd(2, curr_b_tile_2, b_stride);
        _tile_dpbf16ps(6, 0, 2);
        _tile_stream_loadd(3, curr_b_tile_3, b_stride);
        _tile_dpbf16ps(7, 0, 3);
        _tile_loadd(1, curr_a_tile_1, a_stride);
        _tile_stream_loadd(4, curr_b_tile_4, b_stride);
        _tile_dpbf16ps(6, 1, 4);
        _tile_stream_loadd(5, curr_b_tile_5, b_stride);
        _tile_dpbf16ps(7, 1, 5);

        // {
        //     float tmp[AMX_TILE_ROW_NUM][16] = {0};
        //     _tile_stored(4, tmp, 64);
        //     print_float_matrix((float*)tmp, AMX_TILE_ROW_NUM, 16, "tile_4");
        // }

        // update ptrs
        curr_a_tile_0 += 2 * AMX_TILE_BYTES / sizeof(uint16_t);
        curr_a_tile_1 += 2 * AMX_TILE_BYTES / sizeof(uint16_t);
        curr_b_tile_2 += 2 * AMX_TILE_BYTES / sizeof(uint16_t);
        curr_b_tile_3 += 2 * AMX_TILE_BYTES / sizeof(uint16_t);
        curr_b_tile_4 += 2 * AMX_TILE_BYTES / sizeof(uint16_t);
        curr_b_tile_5 += 2 * AMX_TILE_BYTES / sizeof(uint16_t);
      }

      if constexpr (has_tile_loop) {
        _tile_loadd(0, curr_a_tile_0, a_stride);
        _tile_stream_loadd(2, curr_b_tile_2, b_stride);
        _tile_dpbf16ps(6, 0, 2);
        _tile_stream_loadd(3, curr_b_tile_3, b_stride);
        _tile_dpbf16ps(7, 0, 3);
      }

      _tile_stored(6, c_tile_6, c_stride);
      _tile_stored(7, c_tile_7, c_stride);

      // update ptrs
      b_tile_2 += 2 * (AMX_TILE_ROW_BYTES / 4) * K;
      b_tile_3 += 2 * (AMX_TILE_ROW_BYTES / 4) * K;
      b_tile_4 += 2 * (AMX_TILE_ROW_BYTES / 4) * K;
      b_tile_5 += 2 * (AMX_TILE_ROW_BYTES / 4) * K;
      c_tile_6 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
      c_tile_7 += 2 * AMX_TILE_ROW_BYTES / sizeof(float);
    }
  }

  ~AMXBF16Benchmark() { _tile_release(); }

  void init_tile_config() {
    for (int32_t i = 0; i < 8; ++i) {
      amx_tile_config_.colsb[i] = 64;
    }

    if constexpr (M > AMX_TILE_ROW_NUM) {
      const int32_t m_0 = AMX_TILE_ROW_NUM;
      const int32_t m_1 = M - AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[0] = m_0;
      amx_tile_config_.rows[1] = m_1;
      amx_tile_config_.rows[2] = AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[3] = AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[4] = m_0;
      amx_tile_config_.rows[5] = m_0;
      amx_tile_config_.rows[6] = m_1;
      amx_tile_config_.rows[7] = m_1;
    } else {
      amx_tile_config_.rows[0] = M;
      amx_tile_config_.rows[1] = M;
      amx_tile_config_.rows[2] = AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[3] = AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[4] = AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[5] = AMX_TILE_ROW_NUM;
      amx_tile_config_.rows[6] = M;
      amx_tile_config_.rows[7] = M;
    }

    _tile_loadconfig(&amx_tile_config_);
  }

  void v0_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ packed_a,
               uint16_t* __restrict__ b, uint16_t* __restrict__ packed_b) {
    // v0 will not pack a
    for (size_t i = 0; i < M * K; ++i) {
      packed_a[i] = a[i];
    }

    // b will be packed as a whole matrix
    uint16_t* curr_b = b;
    uint16_t* curr_packed_b = packed_b;
    for (int32_t k = 0; k < K; k += 2) {
      for (int32_t n = 0; n < N; ++n) {
        curr_packed_b[2 * n] = curr_b[n];
        curr_packed_b[2 * n + 1] = curr_b[n + N];
      }
      curr_b += 2 * N;
      curr_packed_b += 2 * N;
    }
  }

  void v1_pack(uint16_t* __restrict__ a, uint16_t* __restrict__ packed_a,
               uint16_t* __restrict__ b, uint16_t* __restrict__ packed_b) {
    uint16_t* __restrict__ curr_a = a;
    uint16_t* __restrict__ curr_packed_a = packed_a;
    for (int32_t m = 0; m < M; m += AMX_TILE_ROW_NUM) {
      int32_t m_num = std::min((int32_t)AMX_TILE_ROW_NUM, M - m);
      uint16_t* __restrict__ curr_m_a = curr_a;
      uint16_t* __restrict__ curr_m_packed_a = curr_packed_a;
      for (int32_t m_offset = 0; m_offset < m_num; ++m_offset) {
        constexpr int32_t k_num_per_block =
            AMX_TILE_ROW_BYTES / sizeof(uint16_t);
        uint16_t* __restrict__ curr_m_packed_a_iter = curr_m_packed_a;
        for (int32_t k = 0; k < K; k += k_num_per_block) {
          for (int32_t j = 0; j < k_num_per_block; ++j) {
            curr_m_packed_a_iter[j] = curr_m_a[j];
          }
          curr_m_packed_a_iter += AMX_TILE_BYTES / sizeof(uint16_t);
          curr_m_a += k_num_per_block;
        }
        curr_m_packed_a += k_num_per_block;
      }
      curr_a += AMX_TILE_ROW_NUM * K;
      curr_packed_a += AMX_TILE_ROW_NUM * K;
    }

    // b will be packed per AMX_TILE_ROW_NUM rows
    uint16_t* curr_b = b;
    uint16_t* curr_packed_b = packed_b;
    constexpr int32_t n_num_per_group = AMX_TILE_ROW_BYTES / 4;
    constexpr int32_t n_group_num = N / n_num_per_group;
    constexpr int32_t elem_num_per_n_group = n_num_per_group * K;
    constexpr int32_t k_num_per_group = AMX_TILE_ROW_NUM * 4 / sizeof(uint16_t);
    constexpr int32_t k_group_num = K / k_num_per_group;
    constexpr int32_t k_num_per_sub_group = 4 / sizeof(uint16_t);
    constexpr int32_t k_sub_group_num_per_group =
        k_num_per_group / k_num_per_sub_group;
    for (int32_t n_group_idx = 0; n_group_idx < n_group_num; ++n_group_idx) {
      uint16_t* curr_k_group = curr_b;
      uint16_t* curr_packed_k_group = curr_packed_b;
      for (int32_t k_group_idx = 0; k_group_idx < k_group_num; ++k_group_idx) {
        uint16_t* curr_sub_k_group = curr_k_group;
        uint16_t* curr_packed_sub_k_group = curr_packed_k_group;
        for (int32_t k_sub_group_idx = 0;
             k_sub_group_idx < k_sub_group_num_per_group; ++k_sub_group_idx) {
          uint16_t* curr_n_b = curr_sub_k_group;
          uint16_t* curr_n_packed_b = curr_packed_sub_k_group;
          for (int32_t n = 0; n < n_num_per_group; ++n) {
            uint16_t* curr_n_k_b = curr_n_b;
            uint16_t* curr_n_k_packed_b = curr_n_packed_b;
            for (int32_t k = 0; k < k_num_per_sub_group; ++k) {
              *curr_n_k_packed_b = *curr_n_k_b;

              curr_n_k_b += N;
              curr_n_k_packed_b += 1;
            }

            curr_n_b += 1;
            curr_n_packed_b += k_num_per_sub_group;
          }

          curr_sub_k_group += k_num_per_sub_group * N;
          curr_packed_sub_k_group += k_num_per_sub_group * n_num_per_group;
        }

        curr_k_group += k_num_per_group * N;
        curr_packed_k_group += k_num_per_group * n_num_per_group;
      }

      curr_b += n_num_per_group;
      curr_packed_b += elem_num_per_n_group;
    }
  }

  void float_to_bf16(float* __restrict__ src, uint16_t* __restrict__ dst,
                     size_t num) {
    for (int32_t i = 0; i < num; ++i) {
      dst[i] = _mm_cvtness_sbh(src[i]);
    }
  }

  void bf16_to_float(uint16_t* __restrict__ src, float* __restrict__ dst,
                     size_t num) {
    for (int32_t i = 0; i < num; ++i) {
      dst[i] = _mm_cvtsbh_ss(src[i]);
    }
  }

  void reference_fp32_gemm(float* __restrict__ a, float* __restrict__ b,
                           float* __restrict__ c) {
    // reset c buffer
    {
      float* curr_c = c;
      for (int32_t m = 0; m < M; ++m) {
        for (int32_t n = 0; n < N; ++n) {
          curr_c[n] = 0;
        }
        curr_c += N;
      }
    }

    {
      float* curr_k_a = a;
      float* curr_k_b = b;
      for (int32_t k = 0; k < K; ++k) {
        float* curr_k_m_a = curr_k_a;
        float* curr_m_c = c;
        for (int32_t m = 0; m < M; ++m) {
          float a_val = *curr_k_m_a;
          for (int32_t n = 0; n < N; ++n) {
            curr_m_c[n] += a_val * curr_k_b[n];
          }
          curr_k_m_a += K;
          curr_m_c += N;
        }
        curr_k_a += 1;
        curr_k_b += N;
      }
    }
  }

 private:
  alignas(64) __tilecfg amx_tile_config_;
};

int main() {
  if (!set_tiledata_use()) exit(-1);
  const size_t times = 10000000;

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_ldconfig(times);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("ld_config with init", start_time, end_time, times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_ldconfig_change(1);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("ldconfig_change", start_time, end_time, times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_tile_load_dynamic_m(times, 1);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("tile_load_dynamic_m-1", start_time, end_time,
  //     times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_tile_load_dynamic_m(times, 16);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("tile_load_dynamic_m-16", start_time, end_time,
  //     times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_tile_dot(times);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("tile_dot", start_time, end_time, times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_tile_dot_dynamic_m(times, 1);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("tile_dot_dynamic_m-1", start_time, end_time,
  //     times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_tile_dot_dynamic_m(times, 16);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("tile_dot_dynamic_m-16", start_time, end_time,
  //     times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     test_contigous_load(times);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("contigous_load", start_time, end_time, times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     int32_t amx_c[1024];
  //     test_tile_dot_m1(times, amx_c);
  //     size_t end_time = bench_timestamp();
  //     std::printf("%d\n", amx_c[0]);
  //     print_execute_time("test_tile_dot_m1", start_time, end_time, times);
  //   }

  //   {
  //     size_t start_time = bench_timestamp();
  //     int32_t amx_c[1024];
  //     test_avx_dot_m1(times, amx_c);
  //     size_t end_time = bench_timestamp();
  //     std::printf("%d\n", amx_c[0]);
  //     print_execute_time("test_avx_dot_m1", start_time, end_time, times);
  //   }

  //   {
  //     for (int i = 0; i < 128; ++i) {
  //         std::printf("(%d, %d), ", i, ((i + 63) >> 6) << 6);
  //     }
  //     std::printf("\n");
  //   }

  //   test_L2_fetch();

  //   test_default_tile_size_calcu();

  //   test_calcu_tile_size_with_constant_q();

  // const size_t elem_num = 1024;
  // auto input = generate_input(elem_num);
  // std::vector<float> naive_output, torch_output;

  // {
  //     std::vector<float> output(elem_num);
  //     size_t start_time = bench_timestamp();
  //     execute(times, naive_exp, input, output);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("naive_exp", start_time, end_time, times);
  // }

  // {
  //     std::vector<float> output(elem_num);
  //     size_t start_time = bench_timestamp();
  //     execute(times, naive_exp2, input, output);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("naive_exp2", start_time, end_time, times);
  // }

  // {
  //     std::vector<float> output(elem_num);
  //     size_t start_time = bench_timestamp();
  //     execute(times, torch_exp, input, output);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("torch_exp", start_time, end_time, times);
  // }

  // {
  //     std::vector<float> output(elem_num);
  //     size_t start_time = bench_timestamp();
  //     execute(times, torch_unroll_exp, input, output);
  //     size_t end_time = bench_timestamp();
  //     print_execute_time("torch_unroll_exp", start_time, end_time, times);
  // }

  // {
  //     std::vector<float> output(elem_num);
  //     size_t start_time = bench_timestamp();
  //     execute(times, naive_tanh, input, output);
  //     size_t end_time = bench_timestamp();
  //     naive_output = output;
  //     print_execute_time("naive_tanh", start_time, end_time, times);
  // }

  // {
  //     std::vector<float> output(elem_num);
  //     size_t start_time = bench_timestamp();
  //     execute(times, torch_tanh, input, output);
  //     size_t end_time = bench_timestamp();
  //     torch_output = output;
  //     print_execute_time("torch_tanh", start_time, end_time, times);
  // }

  // for (size_t i = 0; i < naive_output.size(); ++i) {
  //     if (abs(naive_output[i] - torch_output[i]) > 1e-5) {
  //         std::printf("abnormal value: %f, %f\n", naive_output[i],
  //         torch_output[i]);
  //     }
  // }

  // AMXBF16 benchmark

#define BF16GEMM_benchmark(m_size, k_size, n_size)                          \
  {                                                                         \
    constexpr int32_t m = m_size;                                           \
    constexpr int32_t k = k_size;                                           \
    constexpr int32_t n = n_size;                                           \
    constexpr int32_t rounded_m = ((m + 15) / 16) * 16;                     \
    std::printf("case M=%d, N=%d, K=%d\n", m, n, k);                        \
    float* a = reinterpret_cast<float*>(                                    \
        std::aligned_alloc(64, rounded_m * k * sizeof(float)));             \
    float* b = reinterpret_cast<float*>(                                    \
        std::aligned_alloc(64, k * n * sizeof(float)));                     \
    float* c_v0_pack = reinterpret_cast<float*>(                            \
        std::aligned_alloc(64, m * n * sizeof(float)));                     \
    float* c_v1_pack = reinterpret_cast<float*>(                            \
        std::aligned_alloc(64, m * n * sizeof(float)));                     \
    float* c_avxbf16 = reinterpret_cast<float*>(                            \
        std::aligned_alloc(64, rounded_m * n * sizeof(float)));             \
    float* c_avx = reinterpret_cast<float*>(                                \
        std::aligned_alloc(64, rounded_m * n * sizeof(float)));             \
    float* ref_c = reinterpret_cast<float*>(                                \
        std::aligned_alloc(64, m * n * sizeof(float)));                     \
    uint16_t* bf16_a = reinterpret_cast<uint16_t*>(                         \
        std::aligned_alloc(64, rounded_m * k * sizeof(uint16_t)));          \
    uint16_t* bf16_a_v0_pack = reinterpret_cast<uint16_t*>(                 \
        std::aligned_alloc(64, m * k * sizeof(uint16_t)));                  \
    uint16_t* bf16_a_v1_pack = reinterpret_cast<uint16_t*>(                 \
        std::aligned_alloc(64, rounded_m * k * sizeof(uint16_t)));          \
    uint16_t* bf16_b = reinterpret_cast<uint16_t*>(                         \
        std::aligned_alloc(64, k * n * sizeof(uint16_t)));                  \
    uint16_t* bf16_b_v0_pack = reinterpret_cast<uint16_t*>(                 \
        std::aligned_alloc(64, k * n * sizeof(uint16_t)));                  \
    uint16_t* bf16_b_v1_pack = reinterpret_cast<uint16_t*>(                 \
        std::aligned_alloc(64, k * n * sizeof(uint16_t)));                  \
    generate_input(a, m * k);                                               \
    generate_input(b, k * n);                                               \
    AMXBF16Benchmark<m, k, n> benchmark;                                    \
    benchmark.float_to_bf16(a, bf16_a, m * k);                              \
    benchmark.bf16_to_float(bf16_a, a, m * k);                              \
    benchmark.float_to_bf16(b, bf16_b, k * n);                              \
    benchmark.bf16_to_float(bf16_b, b, k * n);                              \
    benchmark.v0_pack(bf16_a, bf16_a_v0_pack, bf16_b, bf16_b_v0_pack);      \
    benchmark.v1_pack(bf16_a, bf16_a_v1_pack, bf16_b, bf16_b_v1_pack);      \
    benchmark.reference_fp32_gemm(a, b, ref_c);                             \
    {                                                                       \
      benchmark.gemm_avxbf16(bf16_a, bf16_b_v0_pack, c_avxbf16);            \
      size_t start_time = bench_timestamp();                                \
      for (int32_t i = 0; i < gemm_times; ++i) {                            \
        benchmark.gemm_avxbf16(bf16_a, bf16_b_v0_pack, c_avxbf16);          \
      }                                                                     \
      size_t end_time = bench_timestamp();                                  \
      print_execute_time("gemm_avxbf16", start_time, end_time, gemm_times); \
      check_c(ref_c, c_avxbf16, m * n);                                     \
    }                                                                       \
    {                                                                       \
      benchmark.gemm_avx(a, bf16_b, c_avx);                                 \
      size_t start_time = bench_timestamp();                                \
      for (int32_t i = 0; i < gemm_times; ++i) {                            \
        benchmark.gemm_avx(a, bf16_b, c_avx);                               \
      }                                                                     \
      size_t end_time = bench_timestamp();                                  \
      print_execute_time("gemm_avx", start_time, end_time, gemm_times);     \
      check_c(ref_c, c_avx, m * n);                                         \
    }                                                                       \
    {                                                                       \
      benchmark.init_tile_config();                                         \
      benchmark.gemm_v0_pack(bf16_a_v0_pack, bf16_b_v0_pack, c_v0_pack);    \
      size_t start_time = bench_timestamp();                                \
      for (int32_t i = 0; i < gemm_times; ++i) {                            \
        benchmark.gemm_v0_pack(bf16_a_v0_pack, bf16_b_v0_pack, c_v0_pack);  \
      }                                                                     \
      size_t end_time = bench_timestamp();                                  \
      print_execute_time("gemm_v0_pack", start_time, end_time, gemm_times); \
      check_c(ref_c, c_v0_pack, m * n);                                     \
    }                                                                       \
    {                                                                       \
      benchmark.init_tile_config();                                         \
      benchmark.gemm_v1_pack(bf16_a_v1_pack, bf16_b_v1_pack, c_v1_pack);    \
      size_t start_time = bench_timestamp();                                \
      for (int32_t i = 0; i < gemm_times; ++i) {                            \
        benchmark.gemm_v1_pack(bf16_a_v1_pack, bf16_b_v1_pack, c_v1_pack);  \
      }                                                                     \
      size_t end_time = bench_timestamp();                                  \
      print_execute_time("gemm_v1_pack", start_time, end_time, gemm_times); \
      check_c(ref_c, c_v1_pack, m * n);                                     \
    }                                                                       \
    std::free(a);                                                           \
    std::free(b);                                                           \
    std::free(c_v0_pack);                                                   \
    std::free(c_v1_pack);                                                   \
    std::free(c_avxbf16);                                                   \
    std::free(c_avx);                                                       \
    std::free(ref_c);                                                       \
    std::free(bf16_a);                                                      \
    std::free(bf16_a_v0_pack);                                              \
    std::free(bf16_a_v1_pack);                                              \
    std::free(bf16_b);                                                      \
    std::free(bf16_b_v0_pack);                                              \
    std::free(bf16_b_v1_pack);                                              \
    std::printf("-------------------------\n");                             \
  }

  {
    const int32_t gemm_times = 1000;
    std::printf("BF16 GEMM benchmarks:\n");
    std::printf("-------------------------\n");
    BF16GEMM_benchmark(1, 128, 128) BF16GEMM_benchmark(2, 128, 128)
        BF16GEMM_benchmark(4, 128, 128) BF16GEMM_benchmark(8, 128, 128)
            BF16GEMM_benchmark(16, 128, 128) BF16GEMM_benchmark(32, 128, 128)
  }
}