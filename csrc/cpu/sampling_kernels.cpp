#include "cpu_types.hpp"

#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Gumbel-max: argmax_i(logit_i + g_i) with g_i ~ Gumbel(0, 1) matches
// softmax. Noise is SplitMix64(seed, i), so tokens are independent and
// consecutive seeds are not a 1-token table shift (vllm#59786).
constexpr uint64_t SPLITMIX_GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr double GUMBEL_MAX = 37.5;  // 53-bit u in (0,1) => -log(-log u) < 37.5

static inline uint64_t splitmix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void fused_gumbel_argmax_row(int64_t* __restrict__ output,
                                    const float* __restrict__ row,
                                    uint64_t seed, int64_t vocab_size) {
  const uint64_t key = splitmix64(seed);

  // Compare in (logit, noise) space: x + g > best_x + best_g is
  // (x - best_x) + g > best_g. Adding g into a 1e38 logit loses it in
  // double ULP, so ties at the float max would always keep the lower index.
  double best_logit = 0.0;
  double best_g = 0.0;
  int64_t best_idx = 0;
  bool have_best = false;
  for (int64_t i = 0; i < vocab_size; ++i) {
    const double x = row[i];
    if (x == -std::numeric_limits<double>::infinity()) continue;
    if (have_best && (x - best_logit) + GUMBEL_MAX <= best_g) continue;
    const uint64_t bits =
        splitmix64(key + uint64_t(i + 1) * SPLITMIX_GAMMA) >> 11;
    const double one_minus_u =
        (double((1ULL << 53) - 1 - bits) + 0.5) * 0x1.0p-53;
    if (have_best && one_minus_u >= std::exp((x - best_logit) - best_g))
      continue;
    const double u = (double(bits) + 0.5) * 0x1.0p-53;
    const double g = -std::log(-std::log(u));
    if (!have_best || (x - best_logit) + g > best_g) {
      have_best = true;
      best_logit = x;
      best_g = g;
      best_idx = i;
    }
  }
  *output = best_idx;
}

static void fused_gumbel_argmax_kernel(int64_t* __restrict__ output,
                                       const float* __restrict__ logits,
                                       const int64_t* __restrict__ seeds,
                                       const int64_t batch_size,
                                       const int64_t vocab_size) {
  if (batch_size <= 0) return;
  if (batch_size == 1) {
    fused_gumbel_argmax_row(output, logits, uint64_t(seeds[0]), vocab_size);
    return;
  }
#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    fused_gumbel_argmax_row(output + b, logits + b * vocab_size,
                            uint64_t(seeds[b]), vocab_size);
  }
}

static void greedy_argmax_kernel(int64_t* __restrict__ output,
                                 const float* __restrict__ logits,
                                 const int64_t batch_size,
                                 const int64_t vocab_size) {
  constexpr int VEC_ELEM_NUM = vec_op::FP32Vec16::VEC_ELEM_NUM;
  const int64_t vec_end = vocab_size - (vocab_size % VEC_ELEM_NUM);

#pragma omp parallel for schedule(static)
  for (int64_t b = 0; b < batch_size; ++b) {
    const float* row = logits + b * vocab_size;

    // Vector max has no consistent NaN rule, so NaNs are caught via the sum:
    // any NaN (or inf + -inf, a harmless false positive) makes it NaN.
    vec_op::FP32Vec16 vmax(-std::numeric_limits<float>::infinity());
    vec_op::FP32Vec16 vsum(0.0f);
    for (int64_t i = 0; i < vec_end; i += VEC_ELEM_NUM) {
      vec_op::FP32Vec16 v(row + i);
      vmax = vmax.max(v);
      vsum = vsum + v;
    }
    float best_val = vmax.reduce_max();
    bool maybe_nan = std::isnan(vsum.reduce_sum());
    for (int64_t i = vec_end; i < vocab_size; ++i) {
      maybe_nan |= std::isnan(row[i]);
      if (row[i] > best_val) {
        best_val = row[i];
      }
    }

    int64_t best_idx = -1;
    if (maybe_nan) {
      // torch.argmax treats NaN as the maximum: return the first NaN.
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (std::isnan(row[i])) {
          best_idx = i;
          break;
        }
      }
    }
    if (best_idx < 0) {
      best_idx = 0;
      for (int64_t i = 0; i < vocab_size; ++i) {
        if (row[i] == best_val) {
          best_idx = i;
          break;
        }
      }
    }
    output[b] = best_idx;
  }
}

}  // namespace

torch::Tensor fused_gumbel_argmax(const torch::Tensor& logits,
                                  const torch::Tensor& seeds) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be a CPU tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.size(1) > 0, "vocab_size must be positive");
  TORCH_CHECK(seeds.device().is_cpu(), "seeds must be a CPU tensor");
  TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
  TORCH_CHECK(seeds.dim() == 1 && seeds.size(0) == logits.size(0),
              "seeds must be 1-D with batch_size elements");

  auto logits_contig = logits.contiguous();
  auto seeds_contig = seeds.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  fused_gumbel_argmax_kernel(output.data_ptr<int64_t>(),
                             logits_contig.data_ptr<float>(),
                             seeds_contig.data_ptr<int64_t>(),
                             logits_contig.size(0), logits_contig.size(1));
  return output;
}

torch::Tensor greedy_argmax(const torch::Tensor& logits) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be a CPU tensor");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");

  // A single row is faster in ATen: OpenMP fork/join dominates the scan
  // (~1 ms vs ~50 µs on s390x at vocab=32k).
  if (logits.size(0) <= 1) {
    return logits.argmax(/*dim=*/-1);
  }

  auto logits_contig = logits.contiguous();
  auto output = torch::empty({logits_contig.size(0)}, torch::kInt64);
  greedy_argmax_kernel(output.data_ptr<int64_t>(),
                       logits_contig.data_ptr<float>(), logits_contig.size(0),
                       logits_contig.size(1));
  return output;
}
