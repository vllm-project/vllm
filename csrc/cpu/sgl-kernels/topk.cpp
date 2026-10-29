// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu:
// `biased_topk_kernel_impl`/`biased_topk_cpu` (flat, non-grouped DeepSeek-V4
// biased top-k) and `hash_topk_kernel_impl`/`hash_topk_cpu` (hash-routed
// layers, expert IDs from a precomputed `tid2eid` lookup table), plus their
// `sigmoid`/`softmax`/`apply_bias`/`sqrtsoftplus` scoring helpers. See
// `biased_topk_cpu`/`hash_topk_cpu` below for the one deliberate deviation
// from the upstream kernel (routed_scaling_factor handling when no shared
// expert is fused into routing, vLLM's CPU DeepSeek-V4 calling convention).
//
// clang-format off

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t, int SIZE, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void sigmoid(float* __restrict__ out, const scalar_t* __restrict__ input) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  const fVec one = fVec(1.f);

  constexpr int kVecSize = bVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x_fvec0, x_fvec1;
    std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

    x_fvec0 = one / (one + x_fvec0.neg().exp_u20());
    x_fvec1 = one / (one + x_fvec1.neg().exp_u20());

    x_fvec0.store(out + d);
    x_fvec1.store(out + d + fVec::size());
  }
}

template <typename scalar_t, int SIZE, std::enable_if_t<std::is_same_v<scalar_t, float>, int> = 0>
inline void sigmoid(float* __restrict__ out, const float* __restrict__ input) {
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  constexpr int kVecSize = fVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    fVec in_fvec = fVec::loadu(input + d);
    in_fvec = one / (one + in_fvec.neg().exp_u20());
    in_fvec.store(out + d);
  }
}

template <typename scalar_t, int SIZE, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void softmax(float* __restrict__ out, const scalar_t* __restrict__ input) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = bVec::size();

  // step 1: get max
  fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
  for (int d = 0; d < SIZE; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x_fvec0, x_fvec1;
    std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

    max_fvec = at::vec::maximum(max_fvec, x_fvec0);
    max_fvec = at::vec::maximum(max_fvec, x_fvec1);
    x_fvec0.store(out + d);
    x_fvec1.store(out + d + fVec::size());
  }
  float max_val = vec_reduce_max(max_fvec);
  max_fvec = fVec(max_val);

  // step 2: sum of (x - max).exp()
  fVec sum_fvec = fVec(float(0));
  for (int d = 0; d < SIZE; d += fVec::size()) {
    fVec x_fvec = (fVec::loadu(out + d) - max_fvec).exp_u20();
    sum_fvec += x_fvec;
    x_fvec.store(out + d);
  }
  float sum_val = vec_reduce_sum(sum_fvec);

  // step 3: x * (1 / sum)
  sum_fvec = fVec(1.f / sum_val);
  for (int d = 0; d < SIZE; d += fVec::size()) {
    fVec out_fvec = fVec::loadu(out + d) * sum_fvec;
    out_fvec.store(out + d);
  }
}

template <typename scalar_t, int SIZE, std::enable_if_t<std::is_same_v<scalar_t, float>, int> = 0>
inline void softmax(float* __restrict__ out, const float* __restrict__ input) {
  using fVec = at::vec::Vectorized<float>;

  constexpr int kVecSize = fVec::size();

  // step 1: get max
  fVec max_fvec = fVec(-std::numeric_limits<float>::infinity());
  for (int d = 0; d < SIZE; d += kVecSize) {
    fVec x_fvec = fVec::loadu(input + d);
    max_fvec = at::vec::maximum(max_fvec, x_fvec);
    x_fvec.store(out + d);
  }
  float max_val = vec_reduce_max(max_fvec);
  max_fvec = fVec(max_val);

  // step 2: sum of (x - max).exp()
  fVec sum_fvec = fVec(float(0));
  for (int d = 0; d < SIZE; d += kVecSize) {
    fVec x_fvec = (fVec::loadu(out + d) - max_fvec).exp_u20();
    sum_fvec += x_fvec;
    x_fvec.store(out + d);
  }
  float sum_val = vec_reduce_sum(sum_fvec);

  // step 3: x * (1 / sum)
  sum_fvec = fVec(1.f / sum_val);
  for (int d = 0; d < SIZE; d += kVecSize) {
    fVec out_fvec = fVec::loadu(out + d) * sum_fvec;
    out_fvec.store(out + d);
  }
}

template <typename param_t, int SIZE>
inline void
apply_bias(float* __restrict__ scores2, const float* __restrict__ scores, const param_t* __restrict__ bias) {
  using fVec = at::vec::Vectorized<float>;
  auto vec_size = fVec::size() * 2;
  int d = 0;
  for (; d <= SIZE - vec_size; d += vec_size) {
    fVec bias0, bias1, x0, x1;
    std::tie(bias0, bias1) = load_float_vec2(bias + d);
    std::tie(x0, x1) = load_float_vec2(scores + d);
    x0 = x0 + bias0;
    x1 = x1 + bias1;
    x0.store(scores2 + d);
    x1.store(scores2 + d + fVec::size());
  }
  for (; d < SIZE; d++) {
    scores2[d] = scores[d] + (float)bias[d];
  }
}

// sqrtsoftplus: sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
// For numerical stability: when x > threshold, softplus(x) ≈ x
// When x < -threshold, softplus(x) ≈ exp(x), so sqrt(softplus(x)) ≈ sqrt(exp(x)) = exp(x/2)
template <typename scalar_t, int SIZE, std::enable_if_t<!std::is_same_v<scalar_t, float>, int> = 0>
inline void sqrtsoftplus(float* __restrict__ out, const scalar_t* __restrict__ input) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  const fVec one = fVec(1.f);
  const fVec half = fVec(0.5f);
  const fVec threshold = fVec(20.f);
  const fVec neg_threshold = fVec(-15.f);  // below this, 1+exp(x) loses precision in float32

  constexpr int kVecSize = bVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    bVec x_bvec = bVec::loadu(input + d);
    fVec x_fvec0, x_fvec1;
    std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

    // default: softplus(x) = log(1 + exp(x))
    fVec sp0 = (one + x_fvec0.exp_u20()).log();
    fVec sp1 = (one + x_fvec1.exp_u20()).log();
    // x > 20: softplus(x) ≈ x
    sp0 = fVec::blendv(sp0, x_fvec0, x_fvec0 > threshold);
    sp1 = fVec::blendv(sp1, x_fvec1, x_fvec1 > threshold);
    // x < -15: softplus(x) ≈ exp(x), sqrt(exp(x)) = exp(x/2)
    fVec exp_half0 = (x_fvec0 * half).exp_u20();
    fVec exp_half1 = (x_fvec1 * half).exp_u20();
    sp0 = fVec::blendv(sp0.sqrt(), exp_half0, x_fvec0 < neg_threshold);
    sp1 = fVec::blendv(sp1.sqrt(), exp_half1, x_fvec1 < neg_threshold);

    sp0.store(out + d);
    sp1.store(out + d + fVec::size());
  }
}

template <typename scalar_t, int SIZE, std::enable_if_t<std::is_same_v<scalar_t, float>, int> = 0>
inline void sqrtsoftplus(float* __restrict__ out, const float* __restrict__ input) {
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  const fVec half = fVec(0.5f);
  const fVec threshold = fVec(20.f);
  const fVec neg_threshold = fVec(-15.f);
  constexpr int kVecSize = fVec::size();
  for (int d = 0; d < SIZE; d += kVecSize) {
    fVec x = fVec::loadu(input + d);
    fVec sp = (one + x.exp_u20()).log();
    sp = fVec::blendv(sp, x, x > threshold);
    fVec exp_half = (x * half).exp_u20();
    sp = fVec::blendv(sp.sqrt(), exp_half, x < neg_threshold);
    sp.store(out + d);
  }
}

// biased_topk: flat (non-grouped) biased topk for DeepSeek V4
// scoring_func: 0 = sigmoid, 1 = sqrtsoftplus
template <typename scalar_t, typename param_t, int NUM_EXPERTS, int TOPK>
void biased_topk_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    const param_t* __restrict__ bias,
    int64_t num_tokens,
    int64_t topk,
    bool renormalize,
    int scoring_func,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scores[NUM_EXPERTS];
    alignas(64) float scores_biased[NUM_EXPERTS];

    using elem_t = std::pair<float, int32_t>;
    std::array<elem_t, NUM_EXPERTS> queue;

    // simple RNG for fused shared expert random ID
    uint64_t rng_state = begin * 6364136223846793005ULL + 1442695040888963407ULL;

    for (int64_t i = begin; i < end; ++i) {
      // compute scores
      if (scoring_func == 0) {
        sigmoid<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);
      } else {
        sqrtsoftplus<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);
      }

      // add bias for selection
      apply_bias<param_t, NUM_EXPERTS>(scores_biased, scores, bias);

      // build queue and partial sort to find top-k
      for (int64_t e = 0; e < NUM_EXPERTS; ++e) {
        queue[e] = {scores_biased[e], static_cast<int32_t>(e)};
      }

      std::partial_sort(
          queue.begin(), queue.begin() + topk, queue.end(), [](const elem_t& x, const elem_t& y) -> bool {
            return x.first > y.first;
          });

      // gather original scores (without bias) as weights
      for (int64_t j = 0; j < topk; ++j) {
        int32_t idx = queue[j].second;
        topk_ids[i * topk + j] = idx;
        topk_weights[i * topk + j] = scores[idx];
      }

      // handle fused shared experts
      if (num_fused_shared_experts > 0) {
        // replace last slot with random shared expert ID
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        int32_t shared_id =
            NUM_EXPERTS + static_cast<int32_t>((rng_state >> 33) % static_cast<uint64_t>(num_fused_shared_experts));
        topk_ids[i * topk + topk - 1] = shared_id;

        // shared expert weight = sum of routed weights / scaling_factor
        if (routed_scaling_factor != 0.0f) {
          float routed_sum = 0.f;
          for (int64_t j = 0; j < topk - 1; ++j) {
            routed_sum += topk_weights[i * topk + j];
          }
          topk_weights[i * topk + topk - 1] = routed_sum / routed_scaling_factor;
        }
      }

      // renormalize
      if (renormalize) {
        float sum = 0.f;
        int64_t norm_end = (num_fused_shared_experts == 0) ? topk : topk - 1;
        for (int64_t j = 0; j < norm_end; ++j) {
          sum += topk_weights[i * topk + j];
        }
        float scale = 1.f / sum;
        if (apply_routed_scaling_factor_on_output) {
          scale *= routed_scaling_factor;
        }
        for (int64_t j = 0; j < norm_end; ++j) {
          topk_weights[i * topk + j] *= scale;
        }
        // also scale the shared expert weight if present
        if (num_fused_shared_experts > 0) {
          topk_weights[i * topk + topk - 1] *= scale;
        }
      }
    }
  });
}

// hash_topk: expert IDs come from a precomputed lookup table tid2eid[input_ids]
// scoring_func: 0 = softmax, 1 = sigmoid, 2 = sqrtsoftplus
template <typename scalar_t, int NUM_EXPERTS, int TOPK>
void hash_topk_kernel_impl(
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ gating_output,
    const int32_t* __restrict__ tid2eid,  // [num_tokens, routed_topk]
    int64_t num_tokens,
    int scoring_func,
    int64_t num_fused_shared_experts,
    int64_t num_experts,
    float routed_scaling_factor,
    int64_t topk) {
  const int64_t routed_topk = topk - num_fused_shared_experts;
  const bool need_renormalize = (scoring_func != 0);  // renormalize for non-softmax

  at::parallel_for(0, num_tokens, 0, [&](int64_t begin, int64_t end) {
    alignas(64) float scores[NUM_EXPERTS];

    // simple RNG for fused shared expert random ID
    uint64_t rng_state = begin * 6364136223846793005ULL + 1442695040888963407ULL;

    for (int64_t i = begin; i < end; ++i) {
      // compute scores over all experts
      if (scoring_func == 0) {
        softmax<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);
      } else if (scoring_func == 1) {
        sigmoid<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);
      } else {
        sqrtsoftplus<scalar_t, NUM_EXPERTS>(scores, gating_output + i * NUM_EXPERTS);
      }

      // gather expert IDs from lookup table
      const int32_t* eid_row = tid2eid + i * routed_topk;
      for (int64_t j = 0; j < routed_topk; ++j) {
        int32_t eid = eid_row[j];
        topk_ids[i * topk + j] = eid;
        topk_weights[i * topk + j] = scores[eid];
      }

      // renormalize routed weights (for non-softmax scoring). Unlike the
      // upstream kernel, always fold routed_scaling_factor in here rather
      // than only applying it to the fused-shared-expert slot below: vLLM's
      // DeepSeek-V4 CPU hash-routed layers never fuse a shared expert into
      // this routing table (num_fused_shared_experts is always 0 on this
      // call path), so this is the only place the scaling factor can be
      // applied. Matches the eager reference (`sqrtsoftplus_bias_topk` in
      // cpu_moe.py) and the CUDA kernel (`dsv4HashTopkSoftplusSqrt`).
      if (need_renormalize) {
        float sum = 0.f;
        for (int64_t j = 0; j < routed_topk; ++j) {
          sum += topk_weights[i * topk + j];
        }
        if (sum > 0.f) {
          float scale = 1.f / sum;
          if (num_fused_shared_experts == 0) {
            scale *= routed_scaling_factor;
          }
          for (int64_t j = 0; j < routed_topk; ++j) {
            topk_weights[i * topk + j] *= scale;
          }
        }
      }

      // handle fused shared expert
      if (num_fused_shared_experts > 0) {
        // random shared expert ID in [num_experts, num_experts + num_fused_shared_experts)
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        int32_t shared_id =
            num_experts + static_cast<int32_t>((rng_state >> 33) % static_cast<uint64_t>(num_fused_shared_experts));
        topk_ids[i * topk + topk - 1] = shared_id;

        // shared expert weight = sum of routed weights / scaling_factor
        float routed_sum = 0.f;
        for (int64_t j = 0; j < routed_topk; ++j) {
          routed_sum += topk_weights[i * topk + j];
        }
        topk_weights[i * topk + topk - 1] = routed_sum / routed_scaling_factor;
      }
    }
  });
}

#define LAUNCH_HASH_TOPK_KERNEL(NE, NTOPK)    \
  hash_topk_kernel_impl<scalar_t, NE, NTOPK>( \
      topk_weights.data_ptr<float>(),         \
      topk_ids.data_ptr<int32_t>(),           \
      gating_output.data_ptr<scalar_t>(),     \
      tid2eid_flat.data_ptr<int32_t>(),       \
      num_tokens,                             \
      scoring_func_id,                        \
      num_fused_shared_experts,               \
      num_experts,                            \
      routed_scaling_factor_value,            \
      topk);

#define LAUNCH_BIASED_TOPK_KERNEL(NE, NTOPK)             \
  biased_topk_kernel_impl<scalar_t, param_t, NE, NTOPK>( \
      topk_weights.data_ptr<float>(),                    \
      topk_ids.data_ptr<int32_t>(),                      \
      gating_output.data_ptr<scalar_t>(),                \
      correction_bias.data_ptr<param_t>(),               \
      num_tokens,                                        \
      topk,                                              \
      renormalize,                                       \
      scoring_func_id,                                   \
      num_fused_shared_experts,                          \
      routed_scaling_factor_value,                       \
      apply_routed_scaling_factor_on_output);

}  // namespace

// biased topk for DeepSeek V4 (flat, non-grouped)
std::tuple<at::Tensor, at::Tensor> biased_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    at::Tensor& correction_bias,
    int64_t topk,
    bool renormalize,
    std::string scoring_func,
    int64_t num_fused_shared_experts,
    std::optional<double> routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  CHECK_INPUT(gating_output);
  CHECK_INPUT(correction_bias);

  const auto st = gating_output.scalar_type();
  int64_t num_tokens = hidden_states.size(0);
  int64_t num_experts = gating_output.size(1);
  TORCH_CHECK(gating_output.size(0) == num_tokens, "Number of tokens mismatch");
  TORCH_CHECK(correction_bias.numel() == num_experts, "Bias shape mismatch");

  int scoring_func_id = 0;
  if (scoring_func == "sigmoid") {
    scoring_func_id = 0;
  } else if (scoring_func == "sqrtsoftplus") {
    scoring_func_id = 1;
  } else {
    TORCH_CHECK(false, "Unsupported scoring_func: ", scoring_func);
  }

  float routed_scaling_factor_value = routed_scaling_factor.has_value() ? routed_scaling_factor.value() : 0.0f;

  at::Tensor topk_weights = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, hidden_states.options().dtype(at::kInt));

  // The actual routed topk (excluding fused shared experts)
  int64_t routed_topk = topk - num_fused_shared_experts;

  CPU_DISPATCH_FLOATING_TYPES_EXT(st, correction_bias.scalar_type(), "biased_topk_kernel", [&] {
    // dispatch on num_experts and routed_topk. For DeepSeek V4: 256 experts,
    // routed_topk=6 (num_fused_shared_experts is always 0 on this CPU call
    // path -- see cpu_moe.py). Anything outside this table isn't a real
    // model shape, so fail loudly instead of silently degrading to a slow
    // generic path.
    switch (num_experts) {
      case 64:
        switch (routed_topk) {
          case 6:
            LAUNCH_BIASED_TOPK_KERNEL(64, 6);
            break;
          case 8:
            LAUNCH_BIASED_TOPK_KERNEL(64, 8);
            break;
          default:
            TORCH_CHECK(false, "biased_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=64");
        }
        break;
      case 128:
        switch (routed_topk) {
          case 6:
            LAUNCH_BIASED_TOPK_KERNEL(128, 6);
            break;
          case 8:
            LAUNCH_BIASED_TOPK_KERNEL(128, 8);
            break;
          default:
            TORCH_CHECK(false, "biased_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=128");
        }
        break;
      case 256:
        switch (routed_topk) {
          case 6:
            LAUNCH_BIASED_TOPK_KERNEL(256, 6);
            break;
          case 8:
            LAUNCH_BIASED_TOPK_KERNEL(256, 8);
            break;
          case 9:
            LAUNCH_BIASED_TOPK_KERNEL(256, 9);
            break;
          default:
            TORCH_CHECK(false, "biased_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=256");
        }
        break;
      case 384:
        switch (routed_topk) {
          case 6:
            LAUNCH_BIASED_TOPK_KERNEL(384, 6);
            break;
          case 8:
            LAUNCH_BIASED_TOPK_KERNEL(384, 8);
            break;
          default:
            TORCH_CHECK(false, "biased_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=384");
        }
        break;
      default:
        TORCH_CHECK(false, "biased_topk_cpu: unsupported num_experts ", num_experts);
    }
  });
  return std::make_tuple(topk_weights, topk_ids);
}

// hash topk for DeepSeek V4 (expert IDs from precomputed lookup table)
std::tuple<at::Tensor, at::Tensor> hash_topk_cpu(
    at::Tensor& gating_output,
    at::Tensor& tid2eid,
    int64_t topk,
    std::string scoring_func,
    int64_t num_fused_shared_experts,
    int64_t num_experts,
    double routed_scaling_factor) {
  CHECK_INPUT(gating_output);
  CHECK_INPUT(tid2eid);

  const auto st = gating_output.scalar_type();
  int64_t num_tokens = gating_output.size(0);
  int64_t num_experts_gating = gating_output.size(1);
  int64_t routed_topk = topk - num_fused_shared_experts;

  TORCH_CHECK(tid2eid.size(0) == num_tokens, "tid2eid row count must match num_tokens");
  TORCH_CHECK(tid2eid.size(1) == routed_topk, "tid2eid column count must match routed_topk");
  TORCH_CHECK(tid2eid.scalar_type() == at::kInt, "tid2eid must be int32");
  TORCH_CHECK(num_experts_gating == num_experts, "num_experts mismatch");

  int scoring_func_id = 0;
  if (scoring_func == "softmax") {
    scoring_func_id = 0;
  } else if (scoring_func == "sigmoid") {
    scoring_func_id = 1;
  } else if (scoring_func == "sqrtsoftplus") {
    scoring_func_id = 2;
  } else {
    TORCH_CHECK(false, "Unsupported scoring_func: ", scoring_func);
  }

  float routed_scaling_factor_value = static_cast<float>(routed_scaling_factor);

  at::Tensor topk_weights = at::empty({num_tokens, topk}, gating_output.options().dtype(at::kFloat));
  at::Tensor topk_ids = at::empty({num_tokens, topk}, gating_output.options().dtype(at::kInt));

  // tid2eid is [num_tokens, routed_topk], already indexed by input_ids in
  // Python; CHECK_INPUT above already guarantees it is contiguous.
  const at::Tensor& tid2eid_flat = tid2eid;

  // Dispatch for bf16, fp16, and float32
  // Note: cannot use AT_DISPATCH_FLOATING_TYPES_AND2 since it includes double which lacks convert_to_float
  auto dispatch_fn = [&]<typename scalar_t>() {
    // For DeepSeek V4: 256 experts, routed_topk in {6,7,8} depending on
    // layer (num_fused_shared_experts is always 0 on this CPU call path --
    // see cpu_moe.py). Anything outside this table isn't a real model
    // shape, so fail loudly instead of silently degrading to a slow
    // generic path.
    switch (num_experts) {
      case 64:
        switch (routed_topk) {
          case 6:
            LAUNCH_HASH_TOPK_KERNEL(64, 6);
            break;
          case 7:
            LAUNCH_HASH_TOPK_KERNEL(64, 7);
            break;
          case 8:
            LAUNCH_HASH_TOPK_KERNEL(64, 8);
            break;
          default:
            TORCH_CHECK(false, "hash_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=64");
        }
        break;
      case 128:
        switch (routed_topk) {
          case 6:
            LAUNCH_HASH_TOPK_KERNEL(128, 6);
            break;
          case 7:
            LAUNCH_HASH_TOPK_KERNEL(128, 7);
            break;
          case 8:
            LAUNCH_HASH_TOPK_KERNEL(128, 8);
            break;
          default:
            TORCH_CHECK(false, "hash_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=128");
        }
        break;
      case 256:
        switch (routed_topk) {
          case 6:
            LAUNCH_HASH_TOPK_KERNEL(256, 6);
            break;
          case 7:
            LAUNCH_HASH_TOPK_KERNEL(256, 7);
            break;
          case 8:
            LAUNCH_HASH_TOPK_KERNEL(256, 8);
            break;
          default:
            TORCH_CHECK(false, "hash_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=256");
        }
        break;
      case 384:
        switch (routed_topk) {
          case 6:
            LAUNCH_HASH_TOPK_KERNEL(384, 6);
            break;
          case 7:
            LAUNCH_HASH_TOPK_KERNEL(384, 7);
            break;
          case 8:
            LAUNCH_HASH_TOPK_KERNEL(384, 8);
            break;
          default:
            TORCH_CHECK(false, "hash_topk_cpu: unsupported routed_topk ", routed_topk,
                        " for num_experts=384");
        }
        break;
      default:
        TORCH_CHECK(false, "hash_topk_cpu: unsupported num_experts ", num_experts);
    }
  };

  if (st == at::ScalarType::BFloat16) {
    dispatch_fn.template operator()<at::BFloat16>();
  } else if (st == at::ScalarType::Half) {
    dispatch_fn.template operator()<at::Half>();
  } else if (st == at::ScalarType::Float) {
    dispatch_fn.template operator()<float>();
  } else {
    TORCH_CHECK(false, "Unsupported dtype for hash_topk_cpu: ", st);
  }

  return std::make_tuple(topk_weights, topk_ids);
}

// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu:
// `topk_transform_512_cpu`/`topk_transform_512_cpu_kernel_impl`. Near-
// verbatim port for the sparse indexer's DECODE path: hardcodes top-512
// (the only `index_topk` DeepSeek-V4-Flash uses) and streams a min-heap over
// the tail of the row instead of sorting the whole row.
//
// Differences from upstream:
// - `out_page_indices` is still computed (required by the kernel contract)
//   but vLLM's caller only consumes `out_raw_indices` (local/compressed-
//   context positions -- `DeepseekV4CPUAttention.forward_mqa` resolves those
//   to physical slots itself via `map_local_to_global_slots_cpu`).
// - `page_size` was already a runtime argument upstream, unlike
//   `fp8_paged_mqa_logits_cpu` (paged_mqa_logits.cpp), which needed one.
namespace {

constexpr int64_t kC4Topk = 512;

template <typename T>
inline int64_t load_index_value(const T* __restrict__ ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

struct TopKTransformElem {
  float score;
  int32_t index;
};

struct TopKTransformMinHeapCmp {
  bool operator()(const TopKTransformElem& lhs, const TopKTransformElem& rhs) const {
    if (lhs.score == rhs.score) {
      return lhs.index > rhs.index;
    }
    return lhs.score > rhs.score;
  }
};

template <typename seq_t, typename page_t>
void topk_transform_512_cpu_kernel_impl(
    const float* __restrict__ scores,
    const seq_t* __restrict__ seq_lens,
    const page_t* __restrict__ page_tables,
    int32_t* __restrict__ out_page_indices,
    int32_t* __restrict__ out_raw_indices,
    int64_t batch_size,
    int64_t max_seq_len,
    int64_t page_table_stride,
    int64_t out_stride,
    int64_t page_size) {
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK((page_size & (page_size - 1)) == 0, "page_size must be a power of 2");
  const int page_bits = page_size > 1 ? static_cast<int>(std::log2(static_cast<double>(page_size))) : 0;
  const int64_t page_mask = page_size - 1;

  at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
    std::array<TopKTransformElem, kC4Topk> heap;

    for (int64_t b = begin; b < end; ++b) {
      const float* __restrict__ scores_row = scores + b * max_seq_len;
      const page_t* __restrict__ page_table_row = page_tables + b * page_table_stride;
      int32_t* __restrict__ out_page_row = out_page_indices + b * out_stride;
      int32_t* __restrict__ out_raw_row = out_raw_indices == nullptr ? nullptr : out_raw_indices + b * out_stride;

      int64_t seq_len = load_index_value(seq_lens, b);
      seq_len = std::max<int64_t>(0, std::min<int64_t>(seq_len, max_seq_len));
      const int64_t valid_topk = std::min<int64_t>(seq_len, kC4Topk);

      auto store_slot = [&](int64_t slot, int32_t raw_index) {
        if (raw_index < 0) {
          out_page_row[slot] = -1;
          if (out_raw_row != nullptr) {
            out_raw_row[slot] = -1;
          }
          return;
        }

        const int64_t page_idx = static_cast<int64_t>(raw_index) >> page_bits;
        const int64_t offset_in_page = static_cast<int64_t>(raw_index) & page_mask;
        const int64_t physical_page = load_index_value(page_table_row, page_idx);
        out_page_row[slot] = static_cast<int32_t>((physical_page << page_bits) | offset_in_page);
        if (out_raw_row != nullptr) {
          out_raw_row[slot] = raw_index;
        }
      };

      if (seq_len <= kC4Topk) {
        for (int64_t i = 0; i < valid_topk; ++i) {
          store_slot(i, static_cast<int32_t>(i));
        }
        for (int64_t i = valid_topk; i < kC4Topk; ++i) {
          store_slot(i, -1);
        }
        continue;
      }

      for (int64_t i = 0; i < kC4Topk; ++i) {
        heap[i] = {scores_row[i], static_cast<int32_t>(i)};
      }
      std::make_heap(heap.begin(), heap.end(), TopKTransformMinHeapCmp());

      for (int64_t i = kC4Topk; i < seq_len; ++i) {
        const float score = scores_row[i];
        const TopKTransformElem& current_min = heap.front();
        if (score > current_min.score ||
            (score == current_min.score && static_cast<int32_t>(i) < current_min.index)) {
          std::pop_heap(heap.begin(), heap.end(), TopKTransformMinHeapCmp());
          heap.back() = {score, static_cast<int32_t>(i)};
          std::push_heap(heap.begin(), heap.end(), TopKTransformMinHeapCmp());
        }
      }

      for (int64_t i = 0; i < kC4Topk; ++i) {
        store_slot(i, heap[i].index);
      }
    }
  });
}

}  // namespace

void topk_transform_512_cpu(
    at::Tensor& scores,
    at::Tensor& seq_lens,
    at::Tensor& page_tables,
    at::Tensor& out_page_indices,
    int64_t page_size,
    const std::optional<at::Tensor>& out_raw_indices) {
  CHECK_INPUT(scores);
  CHECK_INPUT(seq_lens);
  CHECK_INPUT(page_tables);
  CHECK_INPUT(out_page_indices);

  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(out_page_indices.scalar_type() == at::kInt, "out_page_indices must be int32");
  TORCH_CHECK(scores.dim() == 2, "scores must be a 2D tensor");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be a 1D tensor");
  TORCH_CHECK(page_tables.dim() == 2, "page_tables must be a 2D tensor");
  TORCH_CHECK(out_page_indices.dim() == 2, "out_page_indices must be a 2D tensor");

  const int64_t batch_size = scores.size(0);
  const int64_t max_seq_len = scores.size(1);
  TORCH_CHECK(seq_lens.size(0) == batch_size, "seq_lens row count must match scores");
  TORCH_CHECK(page_tables.size(0) == batch_size, "page_tables row count must match scores");
  TORCH_CHECK(out_page_indices.size(0) == batch_size, "out_page_indices row count must match scores");
  TORCH_CHECK(out_page_indices.size(1) >= kC4Topk, "out_page_indices must have at least 512 columns");

  int32_t* raw_ptr = nullptr;
  if (out_raw_indices.has_value()) {
    at::Tensor raw = out_raw_indices.value();
    CHECK_INPUT(raw);
    TORCH_CHECK(raw.scalar_type() == at::kInt, "out_raw_indices must be int32");
    TORCH_CHECK(raw.dim() == 2, "out_raw_indices must be a 2D tensor");
    TORCH_CHECK(raw.sizes() == out_page_indices.sizes(), "out_raw_indices shape must match out_page_indices");
    raw_ptr = raw.data_ptr<int32_t>();
  }

  if (batch_size == 0) {
    return;
  }

  auto launch_with_seq_type = [&](auto seq_zero) {
    using seq_t = decltype(seq_zero);
    if (page_tables.scalar_type() == at::kInt) {
      topk_transform_512_cpu_kernel_impl<seq_t, int32_t>(
          scores.data_ptr<float>(), seq_lens.data_ptr<seq_t>(), page_tables.data_ptr<int32_t>(),
          out_page_indices.data_ptr<int32_t>(), raw_ptr, batch_size, max_seq_len, page_tables.stride(0),
          out_page_indices.stride(0), page_size);
    } else if (page_tables.scalar_type() == at::kLong) {
      topk_transform_512_cpu_kernel_impl<seq_t, int64_t>(
          scores.data_ptr<float>(), seq_lens.data_ptr<seq_t>(), page_tables.data_ptr<int64_t>(),
          out_page_indices.data_ptr<int32_t>(), raw_ptr, batch_size, max_seq_len, page_tables.stride(0),
          out_page_indices.stride(0), page_size);
    } else {
      TORCH_CHECK(false, "page_tables must be int32 or int64");
    }
  };

  if (seq_lens.scalar_type() == at::kInt) {
    launch_with_seq_type(int32_t{0});
  } else if (seq_lens.scalar_type() == at::kLong) {
    launch_with_seq_type(int64_t{0});
  } else {
    TORCH_CHECK(false, "seq_lens must be int32 or int64");
  }
}
