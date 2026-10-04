#include "cpu_types.hpp"

#include <ATen/Parallel.h>
#include <ATen/core/PhiloxRNGEngine.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using BucketMass = unsigned __int128;
constexpr int TAIL_BUCKET = 64;
constexpr int NUM_BUCKETS = TAIL_BUCKET + 1;
constexpr double LN2_HI = 0x1.62e42fefa39efp-1;
constexpr double LN2_LO = 0x1.abc9e3b39803fp-56;
constexpr double INV_LN2 = 0x1.71547652b82fep+0;

class SamplingRng {
 public:
  explicit SamplingRng(uint64_t seed) : engine_(seed) {}

  uint64_t next64() {
    const uint64_t low = engine_();
    return low | (uint64_t(engine_()) << 32);
  }

  uint64_t bounded64(uint64_t bound) {
    const uint64_t threshold = (uint64_t(0) - bound) % bound;
    uint64_t value;
    do {
      value = next64();
    } while (value < threshold);
    return value % bound;
  }

  BucketMass bounded_mass(BucketMass bound) {
    const BucketMass last = bound - 1;
    if (last == 0) return 0;
    const uint64_t high = uint64_t(last >> 64);
    const unsigned bits = high ? 128 - __builtin_clzll(high)
                               : 64 - __builtin_clzll(uint64_t(last));
    // vocab_size fits int64, so the total mass is strictly below 2^127.
    const BucketMass mask = (BucketMass(1) << bits) - 1;
    BucketMass value;
    do {
      value = next64();
      if (bits > 64) value |= BucketMass(next64()) << 64;
      value &= mask;
    } while (value >= bound);
    return value;
  }

  bool bernoulli(double probability) {
    TORCH_INTERNAL_ASSERT(probability >= 0.0 && probability <= 1.0);
    if (probability == 0.0) return false;
    if (probability == 1.0) return true;
    int exponent;
    const double fraction = std::frexp(probability, &exponent);
    const uint64_t mantissa = uint64_t(std::ldexp(fraction, 53));
    // Sample the exact binary64 probability, including subnormals, without
    // quantizing tiny probabilities onto a fixed 53-bit uniform grid.
    for (int remaining = -exponent; remaining > 0;) {
      const int take = std::min(remaining, 64);
      const uint64_t word = next64();
      if ((take == 64 ? word : word >> (64 - take)) != 0) return false;
      remaining -= take;
    }
    return (next64() >> 11) < mantissa;
  }

 private:
  at::Philox4_32 engine_;
};

const std::array<double, NUM_BUCKETS> BUCKET_BOUNDARIES = [] {
  std::array<double, NUM_BUCKETS> boundaries{};
  for (int k = 1; k < NUM_BUCKETS; ++k) {
    double value = std::fma(double(k), LN2_HI, double(k) * LN2_LO);
    // Two upward ULPs cover boundary construction and logit subtraction,
    // keeping 2^-k an upper bound even immediately beside a boundary.
    for (int j = 0; j < 2; ++j)
      value = std::nextafter(value, std::numeric_limits<double>::infinity());
    boundaries[k] = value;
  }
  return boundaries;
}();

int bucket_for(double difference) {
  if (difference >= BUCKET_BOUNDARIES[TAIL_BUCKET]) return TAIL_BUCKET;
  int bucket = std::min(TAIL_BUCKET - 1, int(difference * INV_LN2));
  while (bucket > 0 && difference < BUCKET_BOUNDARIES[bucket]) --bucket;
  while (bucket < TAIL_BUCKET && difference >= BUCKET_BOUNDARIES[bucket + 1])
    ++bucket;
  return bucket;
}

int64_t bucketed_rejection_sample_row(const float* row, int64_t vocab_size,
                                      uint64_t seed) {
  double max_logit = -std::numeric_limits<double>::infinity();
  for (int64_t i = 0; i < vocab_size; ++i) {
    const double value = row[i];
    // Invalid rows are reported to the caller without interrupting other rows.
    if (std::isnan(value) || value == std::numeric_limits<double>::infinity())
      return -1;
    max_logit = std::max(max_logit, value);
  }
  if (!std::isfinite(max_logit)) return -1;  // All tokens are masked.

  std::array<uint64_t, NUM_BUCKETS> counts{}, offsets{};
  std::array<BucketMass, NUM_BUCKETS> cumulative{};
  std::vector<uint8_t> buckets(vocab_size, 255);
  for (int64_t i = 0; i < vocab_size; ++i) {
    if (!std::isfinite(row[i])) continue;  // -inf mask
    const int bucket = bucket_for(max_logit - double(row[i]));
    buckets[i] = uint8_t(bucket);
    ++counts[bucket];
  }
  uint64_t size = 0;
  BucketMass mass = 0;
  for (int k = 0; k < NUM_BUCKETS; ++k) {
    offsets[k] = size;
    size += counts[k];
    mass += BucketMass(counts[k]) << (TAIL_BUCKET - k);
    cumulative[k] = mass;
  }
  std::vector<int64_t> indices(size);
  auto cursor = offsets;
  for (int64_t i = 0; i < vocab_size; ++i)
    if (buckets[i] != 255) indices[cursor[buckets[i]]++] = i;

  SamplingRng rng(seed);
  for (;;) {
    const auto proposal = rng.bounded_mass(mass);
    const int bucket =
        std::upper_bound(cumulative.begin(), cumulative.end(), proposal) -
        cumulative.begin();
    const int64_t index =
        indices[offsets[bucket] + rng.bounded64(counts[bucket])];
    const double difference = max_logit - double(row[index]);
    const double log_acceptance =
        std::fma(double(bucket), LN2_HI, -difference) + double(bucket) * LN2_LO;
    TORCH_INTERNAL_ASSERT(log_acceptance <= 0.0);
    // Proposal mass h_i=2^-bucket and acceptance exp(logit_i-max)/h_i
    // give accepted mass proportional to exp(logit_i-max). Restart globally.
    // The capped tail retains every finite token; extreme exp may underflow.
    if (rng.bernoulli(std::exp(log_acceptance))) return index;
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

    vec_op::FP32Vec16 vmax(-std::numeric_limits<float>::infinity());
    for (int64_t i = 0; i < vec_end; i += VEC_ELEM_NUM) {
      vmax = vmax.max(vec_op::FP32Vec16(row + i));
    }
    float best_val = vmax.reduce_max();
    for (int64_t i = vec_end; i < vocab_size; ++i) {
      if (row[i] > best_val) {
        best_val = row[i];
      }
    }

    int64_t best_idx = 0;
    for (int64_t i = 0; i < vocab_size; ++i) {
      if (row[i] == best_val) {
        best_idx = i;
        break;
      }
    }
    output[b] = best_idx;
  }
}

}  // namespace

torch::Tensor bucketed_rejection_sample(const torch::Tensor& logits,
                                        const torch::Tensor& seeds) {
  TORCH_CHECK(logits.device().is_cpu() && seeds.device().is_cpu(),
              "logits and seeds must be CPU tensors");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2-D [batch, vocab]");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "logits must be float32");
  TORCH_CHECK(logits.size(1) > 0, "vocab_size must be positive");
  TORCH_CHECK(seeds.scalar_type() == torch::kInt64, "seeds must be int64");
  TORCH_CHECK(seeds.dim() == 1 && seeds.size(0) == logits.size(0),
              "seeds must be 1-D with batch_size elements");

  auto logits_contig = logits.contiguous();
  auto seeds_contig = seeds.contiguous();
  const int64_t batch_size = logits.size(0), vocab_size = logits.size(1);
  auto output =
      torch::empty({batch_size}, logits.options().dtype(torch::kInt64));
  const auto* logits_ptr = logits_contig.data_ptr<float>();
  const auto* seeds_ptr = seeds_contig.data_ptr<int64_t>();
  auto* output_ptr = output.data_ptr<int64_t>();
  at::parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
    for (int64_t b = begin; b < end; ++b)
      output_ptr[b] = bucketed_rejection_sample_row(
          logits_ptr + b * vocab_size, vocab_size, uint64_t(seeds_ptr[b]));
  });
  return output;
}

torch::Tensor greedy_argmax(const torch::Tensor& logits) {
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
