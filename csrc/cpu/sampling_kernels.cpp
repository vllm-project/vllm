#include "cpu_types.hpp"

#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>

namespace {

// IEEE-754 bit-trick log approximation (~5 cycles vs ~30 for libm logf).
// Max relative error ≈ 0.5 % — sufficient for Gumbel noise where the
// stochastic nature dominates any approximation error.
static inline float fast_logf(float x) {
  union {
    float f;
    uint32_t i;
  } u = {x};
  float log2 = static_cast<float>(static_cast<int32_t>(u.i) - 0x3F800000) *
               5.9604645e-8f;
  u.i = (u.i & 0x007FFFFF) | 0x3F800000;
  log2 += -0.3447207f * u.f * u.f + 1.3447207f * u.f - 1.0f;
  return log2 * 0.6931472f;  // log2 → ln
}

constexpr int GUMBEL_TABLE_SIZE = 1 << 20;
constexpr int GUMBEL_TABLE_MASK = GUMBEL_TABLE_SIZE - 1;

static float g_gumbel_table[GUMBEL_TABLE_SIZE];
static std::once_flag g_gumbel_init_flag;

static void init_gumbel_table() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(
      std::numeric_limits<float>::min(), 1.0f);
  for (int i = 0; i < GUMBEL_TABLE_SIZE; ++i) {
    float u = uniform(rng);
    g_gumbel_table[i] = -std::log(-std::log(u));
  }
}

static inline void ensure_gumbel_table() {
  std::call_once(g_gumbel_init_flag, init_gumbel_table);
}

// Fused Gumbel-max kernel

static void fused_gumbel_argmax_kernel(int64_t* __restrict__ output,
                                       const float* __restrict__ logits,
                                       const int64_t* __restrict__ seeds,
                                       const int64_t batch_size,
                                       const int64_t vocab_size) {
  ensure_gumbel_table();
  constexpr int VEC_ELEM_NUM = vec_op::FP32Vec8::VEC_ELEM_NUM;
  const int64_t vec_end = vocab_size - (vocab_size % VEC_ELEM_NUM);

#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* row = logits + b * vocab_size;
    const uint32_t seed = static_cast<uint32_t>(seeds[b]);

    float best_score = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;

    for (int64_t i = 0; i < vec_end; i += VEC_ELEM_NUM) {
      vec_op::FP32Vec8 logit_vec(row + i);

      float gumbel_buf[VEC_ELEM_NUM];
      for (int k = 0; k < VEC_ELEM_NUM; ++k)
        gumbel_buf[k] = g_gumbel_table[(seed + i + k) & GUMBEL_TABLE_MASK];
      vec_op::FP32Vec8 gumbel_vec(gumbel_buf);

      vec_op::FP32Vec8 score = logit_vec + gumbel_vec;

      float score_buf[VEC_ELEM_NUM];
      score.save(score_buf);
      for (int k = 0; k < VEC_ELEM_NUM; ++k) {
        if (score_buf[k] > best_score) {
          best_score = score_buf[k];
          best_idx = i + k;
        }
      }
    }

    for (int64_t i = vec_end; i < vocab_size; ++i) {
      float s = row[i] + g_gumbel_table[(seed + i) & GUMBEL_TABLE_MASK];
      if (s > best_score) {
        best_score = s;
        best_idx = i;
      }
    }

    output[b] = best_idx;
  }
}

static void greedy_argmax_kernel(int64_t* __restrict__ output,
                                 const float* __restrict__ logits,
                                 const int64_t batch_size,
                                 const int64_t vocab_size) {
  constexpr int VEC_ELEM_NUM = vec_op::FP32Vec8::VEC_ELEM_NUM;
  const int64_t vec_end = vocab_size - (vocab_size % VEC_ELEM_NUM);

#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* row = logits + b * vocab_size;
    float best_val = -std::numeric_limits<float>::infinity();
    int64_t best_idx = 0;

    for (int64_t i = 0; i < vec_end; i += VEC_ELEM_NUM) {
      vec_op::FP32Vec8 v(row + i);
      float buf[VEC_ELEM_NUM];
      v.save(buf);
      for (int k = 0; k < VEC_ELEM_NUM; ++k) {
        if (buf[k] > best_val) {
          best_val = buf[k];
          best_idx = i + k;
        }
      }
    }

    for (int64_t i = vec_end; i < vocab_size; ++i) {
      if (row[i] > best_val) {
        best_val = row[i];
        best_idx = i;
      }
    }

    output[b] = best_idx;
  }
}

}  // namespace

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(seeds.dim() == 1 && seeds.size(0) == logits.size(0),
              "seeds must be 1-D with batch_size elements");

  auto logits_contig = logits.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  fused_gumbel_argmax_kernel(
      output.data_ptr<int64_t>(), logits_contig.data_ptr<float>(),
      seeds.data_ptr<int64_t>(), logits_contig.size(0), logits_contig.size(1));
  return output;
}

torch::Tensor greedy_argmax(const torch::Tensor& logits) {
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");

  auto logits_contig = logits.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  greedy_argmax_kernel(output.data_ptr<int64_t>(),
                       logits_contig.data_ptr<float>(), logits_contig.size(0),
                       logits_contig.size(1));
  return output;
}
