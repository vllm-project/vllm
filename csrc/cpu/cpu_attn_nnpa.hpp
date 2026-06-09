
#ifndef CPU_ATTN_NNPA_HPP
#define CPU_ATTN_NNPA_HPP

#include "cpu_attn_impl.hpp"
#include "/zDNN/zdnn/zdnn.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace cpu_attention {

namespace {

#define NNPA_BLOCK_SIZE_ALIGNMENT    32
#define NNPA_HEAD_SIZE_ALIGNMENT     32
#define NNPA_MAX_Q_HEAD_NUM_PER_ITER 16

// ─────────────────────────────────────────────────────────────────────────────
// ZDNN_CHECK macro
// Every zDNN API returns zdnn_status. ZDNN_OK (0) = success.
// Anything else = hardware or input error — print and abort.
// ─────────────────────────────────────────────────────────────────────────────
#define ZDNN_CHECK(call, msg)                                          \
  do {                                                                 \
    zdnn_status _st = (call);                                          \
    if (_st != ZDNN_OK) {                                              \
      fprintf(stderr, "[NNPA] zDNN error at %s: status=%d\n",         \
              (msg), (int)_st);                                        \
      abort();                                                         \
    }                                                                  \
  } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// NNPATensor
//
// Wraps the full zdnn_ztensor lifecycle:
//   1. zdnn_init_pre_transformed_desc  — describe your float array shape
//   2. zdnn_generate_transformed_desc  — generate Telum II internal layout
//   3. zdnn_init_ztensor_with_malloc   — allocate 4KB-aligned buffer
//   4. zdnn_transform_ztensor          — convert FP32 → DLFLOAT16
//
// After computation:
//   5. zdnn_transform_origtensor       — convert result back → FP32
//   6. zdnn_free_ztensor_buffer        — release internal buffer
//
// ─────────────────────────────────────────────────────────────────────────────
struct NNPATensor {
  zdnn_ztensor     zt;
  zdnn_tensor_desc pre_desc;
  zdnn_tensor_desc tfrmd_desc;
  bool             allocated = false;

  // Initialize a 3DS tensor — shape [s, rows, cols]
  void init(uint32_t s, uint32_t rows, uint32_t cols,
            const float* src = nullptr, bool use_2d = false) {
    if (use_2d) {
      // ZDNN_2D layout for B matrix and bias
      zdnn_init_pre_transformed_desc(ZDNN_2D, FP32,
                                     &pre_desc, rows, cols);
    } else {
      // ZDNN_3DS layout for A, output
      zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32,
                                     &pre_desc, s, rows, cols);
    }
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc),
               "generate_transformed_desc");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt),
               "init_ztensor_with_malloc");
    allocated = true;

    if (src != nullptr)
      ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "transform_ztensor");
  }

  // Initialize a 1D tensor — shape [cols] for bias
  void init_1d(uint32_t cols, const float* src = nullptr) {
    zdnn_init_pre_transformed_desc(ZDNN_1D, FP32, &pre_desc, cols);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &tfrmd_desc),
               "generate_transformed_desc_1d");
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_desc, &tfrmd_desc, &zt),
               "init_ztensor_with_malloc_1d");
    allocated = true;
    if (src != nullptr)
      ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "transform_ztensor_1d");
  }

  void load(const float* src) {
    ZDNN_CHECK(zdnn_transform_ztensor(&zt, src), "transform_ztensor (load)");
  }

  void store(float* dst) const {
    ZDNN_CHECK(zdnn_transform_origtensor(
                   const_cast<zdnn_ztensor*>(&zt), dst),
               "transform_origtensor (store)");
  }

  void free_buf() {
    if (allocated) {
      zdnn_free_ztensor_buffer(&zt);
      allocated = false;
    }
  }

  ~NNPATensor() { free_buf(); }
};

// ─────────────────────────────────────────────────────────────────────────────
static void nnpa_matmul(const float* A, const float* B, float* C,
                        uint32_t s, uint32_t m, uint32_t k, uint32_t n,
                        int64_t lda=0, int64_t ldb=0, int64_t ldc=0) {
  NNPATensor tA, tB, tBias, tC;

  // Copy A with correct row strides into contiguous buffer
  float* a_buf = nullptr;
  if (lda > 0 && (int64_t)k != lda) {
    a_buf = (float*)malloc((size_t)m * k * sizeof(float));
    assert(a_buf != nullptr);
    for (uint32_t i = 0; i < m; i++)
      for (uint32_t j = 0; j < k; j++)
        a_buf[i * k + j] = A[i * lda + j];
    A = a_buf;
  }

  // Copy B into contiguous buffer
  // K cache layout: b_tile[dim * ldb + token] = K[dim, token]
  // We need b_buf[dim * n + token] = K[dim, token]
  float* b_buf = (float*)malloc((size_t)k * n * sizeof(float));
  assert(b_buf != nullptr);
  bool b_needs_free = true;
  if (ldb > 0 && (int64_t)n != ldb) {
    // Strided: copy row by row with stride ldb
    for (uint32_t i = 0; i < k; i++)
      for (uint32_t j = 0; j < n; j++)
        b_buf[i * n + j] = static_cast<float>(B[i * ldb + j]);
  } else {
    // Contiguous: simple copy
    for (size_t i = 0; i < (size_t)k * n; i++)
      b_buf[i] = static_cast<float>(B[i]);
  }
  B = b_buf;


  // Debug: verify A and B before zdnn
  {FILE* dbg=fopen("/tmp/b_layout.txt","a"); if(dbg){
    fprintf(dbg,"nnpa_matmul: s=%d m=%d k=%d n=%d\n",s,m,k,n);
    fprintf(dbg,"  A[head0,0:8]=");
    for(int _j=0;_j<8;_j++) fprintf(dbg,"%.4f ",A[_j]);
    fprintf(dbg,"\n");
    fprintf(dbg,"  B[0:8] linear=");
    for(int _j=0;_j<8;_j++) fprintf(dbg,"%.4f ",B[_j]);
    fprintf(dbg,"\n");
    fprintf(dbg,"  B col0 (B[i*n+0] for i=0..7)=");
    for(int _i=0;_i<8;_i++) fprintf(dbg,"%.4f ",B[_i*(int)n+0]);
    fprintf(dbg,"\n");
    fclose(dbg);
  }}

  // Debug: dump all A heads before zdnn
  {FILE* dbg=fopen("/tmp/b_layout.txt","a"); if(dbg){
    fprintf(dbg,"A all heads [m=%d,k=%d]:\n",m,k);
    for(uint32_t _h=0;_h<m;_h++){
      fprintf(dbg,"  head%d[0:8]=",_h);
      for(int _d=0;_d<8;_d++) fprintf(dbg,"%.4f ",A[_h*k+_d]);
      fprintf(dbg,"\n");
    }
    fclose(dbg);}}
  tA.init(s, m, k, A);
  tB.init(s, k, n, B, true);
  tC.init(s, m, n);

  // Bias shape: [n] — zero values (ZDNN_1D)
  float* zero_bias = (float*)calloc(n, sizeof(float));
  assert(zero_bias != nullptr);
  tBias.init_1d(n, zero_bias);
  ::free(zero_bias);

  {
    static int call_cnt = 0;
    call_cnt++;
    FILE* dbg=fopen("/tmp/b_layout.txt","a");
    if(dbg){
      fprintf(dbg,"[call %d] s=%d m=%d k=%d n=%d ldc=%ld\n", call_cnt,s,m,k,n,(long)ldc);
      fclose(dbg);
    }
  }

  ZDNN_CHECK(
    zdnn_matmul_bcast_op(&tA.zt, &tB.zt, &tBias.zt,
                         MATMUL_BCAST_OP_ADDITION, &tC.zt),
    "zdnn_matmul_bcast_op"
  );

  // Extract result
  if (ldc > 0 && (int64_t)n != ldc) {
    float* c_tmp = (float*)malloc((size_t)m * n * sizeof(float));
    assert(c_tmp != nullptr);
    tC.store(c_tmp);
    for (uint32_t i = 0; i < m; i++)
      for (uint32_t j = 0; j < n; j++)
        C[i * ldc + j] = c_tmp[i * n + j];
    ::free(c_tmp);
  } else {
    tC.store(C);
  }

  {FILE* dbg=fopen("/tmp/b_layout.txt","a"); if(dbg){
    fprintf(dbg,"After zdnn m=%d n=%d\n",m,n); for(uint32_t _i=0;_i<m&&_i<6;_i++){fprintf(dbg,"row%d: ",_i);for(uint32_t _j=0;_j<6;_j++)fprintf(dbg,"%.4f ",C[_i*n+_j]);fprintf(dbg,"\n");}
    fclose(dbg);}}
  if (a_buf) ::free(a_buf);
  if (b_buf) ::free(b_buf);

}

// ─────────────────────────────────────────────────────────────────────────────
static void nnpa_softmax(const float* input, float* output,
                         uint32_t rows, uint32_t cols) {
  NNPATensor tIn, tOut;
  tIn.init(1, rows, cols, input);
  tOut.init(1, rows, cols);

  // Fire softmax on Telum II
  // SOFTMAX_ACT_NONE = plain softmax (no extra activation)
  ZDNN_CHECK(
    zdnn_softmax(&tIn.zt, nullptr, SOFTMAX_ACT_NONE, &tOut.zt),
    "zdnn_softmax"
  );

  tOut.store(output);
}

// ─────────────────────────────────────────────────────────────────────────────
template <typename kv_cache_t>
class TileGemmNNPA {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile,
                                const int64_t lda,
                                const int64_t ldb,
                                const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    // Resolve actual K at runtime
    const int32_t K = (k_size > 0) ? k_size : dynamic_k_size;
    {
      FILE* f = fopen("/tmp/nnpa_debug.txt", "a");
      if (f) {
        fprintf(f, "[NNPA] gemm: phase=%s m=%d K=%d N_block=%d lda=%ld ldb=%ld ldc=%ld accum=%d a_tile=%p\n",
                (phase==AttentionGemmPhase::QK)?"QK":"PV",
                m_size, K, block_size, (long)lda, (long)ldb, (long)ldc, (int)accum_c, (void*)a_tile);
        fprintf(f, "  A_strided[rows@lda]:\n");
        for(int _r=0;_r<m_size&&_r<6;_r++){fprintf(f,"    r%d: ",_r);
          for(int _c=0;_c<8;_c++) fprintf(f,"%.4f ",a_tile[_r*(int)lda+_c]); fprintf(f,"\n");}
        fprintf(f, "  A[0:8]=");
        for(int i=0;i<8&&i<m_size*(int)K;i++) fprintf(f, "%.4f ", a_tile[i]);
        fprintf(f, "\n");
        fprintf(f, "  B[0:8]=");
        for(int i=0;i<8;i++) fprintf(f, "%.4f ", (float)b_tile[i]);
        fprintf(f, "\n");
        fclose(f);
      }
    }

    const int32_t N = (phase == AttentionGemmPhase::QK)
                          ? block_size
                          : NNPA_HEAD_SIZE_ALIGNMENT;

    // ── Convert B from kv_cache_t → float ─────────────────────────────────
    // zDNN always takes FP32. If KV cache is BFloat16 or Half, convert first.
    const size_t b_elems = (size_t)K * N;
    float* b_fp32 = nullptr;
    bool   b_needs_free = false;

    // Always copy B with correct strides — b_tile may not be contiguous
    b_fp32 = (float*)malloc(b_elems * sizeof(float));
    assert(b_fp32 != nullptr);
    b_needs_free = true;
    if constexpr (phase == AttentionGemmPhase::QK) {
      // K cache layout: b_tile[dim * ldb + token] = K[dim, token]
      // Copy to contiguous: b_fp32[dim * N + token] = K[dim, token]
      {static int dc=0; dc++;
      if(dc<=3){FILE* dbg=fopen("/tmp/nnpa_bdump.txt","a"); if(dbg){
        fprintf(dbg,"NNPA QK B K=%d N=%d ldb=%ld m=%d b_tile[0:8]: ",K,N,(long)ldb,m_size);
        for(int i=0;i<8;i++) fprintf(dbg,"%.4f ",(float)b_tile[i]);
        fprintf(dbg,"\n");
        fclose(dbg);}}}
      for (int32_t i = 0; i < K; i++)
        for (int32_t j = 0; j < N; j++)
          b_fp32[i * N + j] = static_cast<float>(b_tile[i * ldb + j]);
      {FILE* dbg=fopen("/tmp/b_layout.txt","a"); if(dbg){
        fprintf(dbg,"QK B after  transpose: ");
        for(int i=0;i<8;i++) fprintf(dbg,"%.4f ",b_fp32[i]);
        fprintf(dbg,"\n");
        fclose(dbg);}}
    } else {
      // V cache: b_tile[token*ldb + dim] = V[token, dim]
      {static int pc=0;pc++;if(pc<=2){FILE*f=fopen("/tmp/pv_debug.txt","a");if(f){
        fprintf(f,"PV: K=%d N=%d ldb=%ld b_tile[0:8]=",K,N,(long)ldb);
        for(int i=0;i<8;i++) fprintf(f,"%.4f ",(float)b_tile[i]);
        fprintf(f,"\n");fclose(f);}}}
      for (int32_t i = 0; i < K; i++)
        for (int32_t j = 0; j < N; j++)
          b_fp32[i * N + j] = static_cast<float>(b_tile[i * ldb + j]);
    }

    // ── Scratch buffer for output (before accumulate) ──────────────────────
    // For PV: N=32 (group size) but ldc=64 (output stride) — use ldc for buffer
    const int64_t out_stride = (ldc > 0) ? ldc : N;
    const size_t c_elems = (size_t)m_size * out_stride;
    float* c_out = c_tile;
    float* c_tmp = nullptr;

    if (accum_c) {
      c_tmp = (float*)malloc(c_elems * sizeof(float));
      assert(c_tmp != nullptr);
      std::memset(c_tmp, 0, c_elems * sizeof(float));
      c_out = c_tmp;
    }

    if constexpr (phase == AttentionGemmPhase::QK) {
      // QK: use zdnn — K cache stored [dim,token] matches zdnn col-major
      nnpa_matmul(a_tile, b_fp32, c_out, 1,
                  (uint32_t)m_size, (uint32_t)K, (uint32_t)N, lda, N, ldc);
    } else {
      // PV: CPU fallback — V stored [token,dim] row-major
      // N=32 (head_dim group), ldc=64 (output stride in partial_q_buffer)
      uint32_t _m=(uint32_t)m_size, _k=(uint32_t)K, _n=(uint32_t)N;
      int64_t _lda=lda>0?lda:_k;
      int64_t _ldc=ldc>0?ldc:_n;
      for(uint32_t _i=0;_i<_m;_i++)
        for(uint32_t _j=0;_j<_n;_j++) {
          float _s=0;
          for(uint32_t _d=0;_d<_k;_d++)
            _s += a_tile[_i*_lda+_d] * b_fp32[_d*_n+_j];
          c_out[_i*_ldc+_j] = _s;
        }
    }

    // Debug: dump QK scores and A matrix
    if constexpr (phase == AttentionGemmPhase::QK) {
      FILE* dbg=fopen("/tmp/b_layout.txt","a");
      if(dbg){
        fprintf(dbg,"QK m=%d K=%d N=%d\n", m_size, K, N);
        fprintf(dbg,"A rows (Q heads):\n");
        for(int _i=0;_i<m_size&&_i<6;_i++){
          fprintf(dbg,"  Qhead%d: ",_i);
          for(int _j=0;_j<6;_j++) fprintf(dbg,"%.4f ",a_tile[_i*(int)lda+_j]);
          fprintf(dbg,"\n");
        }
        fprintf(dbg,"QK scores c_out rows:\n");
        for(int _i=0;_i<m_size&&_i<6;_i++){
          fprintf(dbg,"  head%d: ",_i);
          for(int _j=0;_j<6;_j++) fprintf(dbg,"%.4f ",c_out[_i*N+_j]);
          fprintf(dbg,"\n");
        }
        fclose(dbg);
      }
    }

    // ── Accumulate if needed (C += C_new) ─────────────────────────────────
    if (accum_c) {
      for (size_t i = 0; i < c_elems; ++i)
        c_tile[i] += c_out[i];
      ::free(c_tmp);
    }

    if (b_needs_free)
      ::free(b_fp32);
  }
};

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// AttentionImpl<ISA::NNPA>
//
// Full attention implementation for Telum II NNPA.
// Plugs into the vLLM AttentionMainLoop<> framework.
//
// Attention formula (Flash Attention style, tiled):
//   For each KV tile:
//     scores  = Q_tile × K^T_tile     → zdnn_matmul_op (NNPA)
//     scores *= scale                  → CPU scalar multiply
//     weights = softmax(scores)        → zdnn_softmax (NNPA)
//     output += weights × V_tile       → zdnn_matmul_op (NNPA)
// ─────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::NNPA, scalar_t, head_dim> {
 public:
  // ── Type aliases (required by AttentionMainLoop) ─────────────────────────
  using query_t                 = scalar_t;
  using q_buffer_t              = float;
  using kv_cache_t              = scalar_t;
  using logits_buffer_t         = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t           = float;

  // ── Constants (required by AttentionMainLoop) ────────────────────────────
  constexpr static int64_t BlockSizeAlignment      = NNPA_BLOCK_SIZE_ALIGNMENT;
  constexpr static int64_t HeadDimAlignment        = NNPA_HEAD_SIZE_ALIGNMENT;
  constexpr static int64_t MaxQHeadNumPerIteration = NNPA_MAX_Q_HEAD_NUM_PER_ITER;
  constexpr static int64_t HeadDim                 = head_dim;
  constexpr static ISA     ISAType                 = ISA::NNPA;
  // Scale is applied to Q before matmul (not on logits after)
  constexpr static bool    scale_on_logits         = false;

 public:
  AttentionImpl() {}

  // ── execute_attention ─────────────────────────────────────────────────────
  // Plug TileGemmNNPA into AttentionMainLoop<>.
  // VXE uses TileGemmS390X — we use TileGemmNNPA.
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    // Confirm NNPA is active
    {static bool _printed=false; if(!_printed){
      fprintf(stderr,"[NNPA] Attention running on Telum II NNPA via zdnn\n");
      _printed=true;}}
    attention<TileGemmNNPA<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // ── KV cache stride helpers (same layout as VXE) ─────────────────────────
  constexpr static int64_t k_cache_token_group_stride(const int32_t) {
    return BlockSizeAlignment;
  }

  constexpr static int64_t v_cache_token_group_stride(const int32_t) {
    return head_dim * BlockSizeAlignment;
  }

  constexpr static int64_t v_cache_head_group_stride(const int32_t) {
    return HeadDimAlignment;
  }

  // ── copy_q_heads_tile ─────────────────────────────────────────────────────
  // Copy Q tile from model input → float buffer with scale applied.
  // VXE uses SIMD (vec_xl, vec_mul). We use a plain loop — correct and
  // portable. The Q copy is not on the critical path vs matmul.
  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                float*    __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride,
                                float scale) {
    {static int gc=0; if(gc<3){FILE*rf=fopen("/tmp/q_rawsrc.txt","a"); if(rf){
      fprintf(rf,"RAW src=%p num_stride=%ld head_stride=%ld q_num=%d\n",(void*)src,(long)q_num_stride,(long)q_head_stride,q_num);
      for(int ii=0;ii<q_num&&ii<6;ii++){fprintf(rf,"  src+%d*ns [off=%ld]: ",ii,(long)(ii*q_num_stride));
        for(int d=0;d<8;d++) fprintf(rf,"%.4f ",(float)src[ii*q_num_stride+d]); fprintf(rf,"\n");}
      fclose(rf);} gc++;}}
    {static int gc=0; if(gc<3){FILE*rf=fopen("/tmp/q_rawsrc.txt","a"); if(rf){
      fprintf(rf,"RAW src=%p num_stride=%ld head_stride=%ld q_num=%d\n",(void*)src,(long)q_num_stride,(long)q_head_stride,q_num);
      for(int ii=0;ii<q_num&&ii<6;ii++){fprintf(rf,"  src+%d*ns [off=%ld]: ",ii,(long)(ii*q_num_stride));
        for(int d=0;d<8;d++) fprintf(rf,"%.4f ",(float)src[ii*q_num_stride+d]); fprintf(rf,"\n");}
      fclose(rf);} gc++;}}
    for (int32_t i = 0; i < q_num; ++i) {
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        const scalar_t* curr_src =
            src + i * q_num_stride + h * q_head_stride;
        float* curr_dst =
            q_buffer + i * q_heads_per_kv * head_dim + h * head_dim;

        for (int64_t d = 0; d < head_dim; ++d)
          curr_dst[d] = static_cast<float>(curr_src[d]) * scale;
      }
    }
    {FILE* gf=fopen("/tmp/q_gather.txt","a"); if(gf){
      fprintf(gf,"GATHER q_num=%d hpk=%d num_stride=%ld head_stride=%ld scale=%.4f buf=%p src=%p\n",
              q_num,q_heads_per_kv,(long)q_num_stride,(long)q_head_stride,scale,(void*)q_buffer,(void*)src);
      for(int32_t r=0;r<q_num&&r<6;r++){fprintf(gf,"  row%d: ",r);
        for(int d=0;d<8;d++) fprintf(gf,"%.4f ",q_buffer[r*q_heads_per_kv*head_dim+d]);
        fprintf(gf,"\n");} fclose(gf);}}
  }

  // ── reshape_and_cache ─────────────────────────────────────────────────────
  // Store K/V into paged KV cache.
  // Layout is identical to VXE — copied exactly.
  static void reshape_and_cache(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      scalar_t*       __restrict__ key_cache,
      scalar_t*       __restrict__ value_cache,
      const int64_t*  __restrict__ slot_mapping,
      const int64_t token_num,
      const int64_t key_token_num_stride,
      const int64_t value_token_num_stride,
      const int64_t head_num,
      const int64_t key_head_num_stride,
      const int64_t value_head_num_stride,
      const int64_t num_blocks,
      const int64_t num_blocks_stride,
      const int64_t cache_head_num_stride,
      const int64_t block_size,
      const int64_t block_size_stride) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;

        const int64_t block_idx    = pos / block_size;
        const int64_t block_offset = pos % block_size;

        // Key cache
        {
          const scalar_t* key_src =
              key + token_idx * key_token_num_stride +
              head_idx * key_head_num_stride;
          scalar_t* key_dst =
              key_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset;

          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size)
            key_dst[j] = key_src[i];
          // Debug: dump first 8 K values being stored
          if (token_idx == 0 && head_idx == 0) {
            FILE* f = fopen("/tmp/kcache_debug.txt", "a");
            if (f) {
              fprintf(f, "K stored[0:8]=");
              for (int i = 0; i < 8; i++)
                fprintf(f, "%.4f ", (float)key_src[i]);
              fprintf(f, "\n");
              fclose(f);
            }
          }
        }

        // Value cache
        {
          const scalar_t* val_src =
              value + token_idx * value_token_num_stride +
              head_idx * value_head_num_stride;
          scalar_t* val_dst =
              value_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride +
              block_offset * head_dim;

          std::memcpy(val_dst, val_src, sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};

}  // namespace cpu_attention

#undef NNPA_BLOCK_SIZE_ALIGNMENT
#undef NNPA_HEAD_SIZE_ALIGNMENT
#undef NNPA_MAX_Q_HEAD_NUM_PER_ITER

#endif  // CPU_ATTN_NNPA_HPP
