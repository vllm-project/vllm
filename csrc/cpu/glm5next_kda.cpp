#include <ATen/ATen.h>

#include <cmath>
#include <immintrin.h>
#include <tuple>
#include <vector>

namespace {

std::tuple<at::Tensor, at::Tensor> glm5next_kda_recurrent_cpu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g, const at::Tensor& beta,
    const at::Tensor& initial_state, double scale, bool sigmoid_beta,
    const at::Tensor& a_log, const at::Tensor& g_bias, bool compute_gate,
    double lower_bound) {
  TORCH_CHECK(q.device().is_cpu() && k.device().is_cpu() && v.device().is_cpu(),
              "GLM5Next KDA CPU op expects CPU inputs");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && g.dim() == 4,
              "q, k, v and g must be [1, T, H, D]");
  TORCH_CHECK(
      q.size(0) == 1 && k.size(0) == 1 && v.size(0) == 1 && g.size(0) == 1,
      "first native KDA path supports batch size 1");
  const auto t_count = q.size(1);
  const auto q_heads = q.size(2);
  const auto key_dim = q.size(3);
  const auto value_heads = v.size(2);
  const auto value_dim = v.size(3);
  TORCH_CHECK(k.size(1) == t_count && k.size(2) == q_heads &&
                  k.size(3) == key_dim && v.size(1) == t_count &&
                  g.size(1) == t_count && g.size(2) == value_heads &&
                  g.size(3) == key_dim,
              "incompatible KDA tensor shapes");
  TORCH_CHECK(value_heads % q_heads == 0,
              "value heads must be a multiple of query heads");
  TORCH_CHECK(initial_state.dim() == 3 &&
                  initial_state.size(0) == value_heads &&
                  initial_state.size(1) == value_dim &&
                  initial_state.size(2) == key_dim,
              "initial_state must be [value_heads, value_dim, key_dim]");
  TORCH_CHECK(beta.numel() == t_count * value_heads,
              "native first path supports per-head beta");
  TORCH_CHECK(!compute_gate || (a_log.numel() == value_heads &&
                                g_bias.numel() == value_heads * key_dim),
              "invalid KDA gate parameter shapes");

  auto qf = q.to(at::kFloat).contiguous();
  auto kf = k.to(at::kFloat).contiguous();
  auto vf = v.to(at::kFloat).contiguous();
  auto gf = g.to(at::kFloat).contiguous();
  auto bf = beta.to(at::kFloat).contiguous();
  auto af = a_log.to(at::kFloat).contiguous();
  auto biasf = g_bias.to(at::kFloat).contiguous();
  auto state = initial_state.to(at::kFloat).contiguous().clone();
  auto out = at::zeros({1, t_count, value_heads, value_dim},
                       q.options().dtype(at::kFloat));

  const auto* qp = qf.data_ptr<float>();
  const auto* kp = kf.data_ptr<float>();
  const auto* vp = vf.data_ptr<float>();
  const auto* gp = gf.data_ptr<float>();
  const auto* bp = bf.data_ptr<float>();
  const auto* ap = af.data_ptr<float>();
  const auto* biasp = biasf.data_ptr<float>();
  auto* sp = state.data_ptr<float>();
  auto* op = out.data_ptr<float>();
  const auto groups = value_heads / q_heads;

  for (int64_t t = 0; t < t_count; ++t) {
    for (int64_t hv = 0; hv < value_heads; ++hv) {
      const auto hq = hv / groups;
      auto* sh = sp + hv * value_dim * key_dim;
      const auto* qh = qp + (t * q_heads + hq) * key_dim;
      const auto* kh = kp + (t * q_heads + hq) * key_dim;
      const auto* vh = vp + (t * value_heads + hv) * value_dim;
      const auto* gh = gp + (t * value_heads + hv) * key_dim;
      const auto qnorm = [&]() {
        float sum = 0.0f;
        for (int64_t d = 0; d < key_dim; ++d) sum += qh[d] * qh[d];
        return std::sqrt(sum + 1e-6f);
      }();
      const auto knorm = [&]() {
        float sum = 0.0f;
        for (int64_t d = 0; d < key_dim; ++d) sum += kh[d] * kh[d];
        return std::sqrt(sum + 1e-6f);
      }();
      const auto beta_value =
          sigmoid_beta ? 1.0f / (1.0f + std::exp(-bp[t * value_heads + hv]))
                       : bp[t * value_heads + hv];
      std::vector<float> q_normalized(key_dim);
      std::vector<float> k_normalized(key_dim);
      for (int64_t d = 0; d < key_dim; ++d) {
        q_normalized[d] = qh[d] / qnorm * scale;
        k_normalized[d] = kh[d] / knorm;
      }
      for (int64_t d = 0; d < key_dim; ++d) {
        auto gate = gh[d];
        if (compute_gate) {
          gate = static_cast<float>(lower_bound) *
                 (1.0f / (1.0f + std::exp(-std::exp(ap[hv]) *
                                          (gate + biasp[hv * key_dim + d]))));
        }
        gf.data_ptr<float>()[((t * value_heads + hv) * key_dim) + d] =
            std::exp(gate);
      }
      const auto* decay =
          gf.data_ptr<float>() + (t * value_heads + hv) * key_dim;
      for (int64_t row = 0; row < value_dim; ++row) {
        int64_t d = 0;
        for (; d + 16 <= key_dim; d += 16) {
          const auto state_vec = _mm512_loadu_ps(sh + row * key_dim + d);
          const auto decay_vec = _mm512_loadu_ps(decay + d);
          _mm512_storeu_ps(sh + row * key_dim + d,
                           _mm512_mul_ps(state_vec, decay_vec));
        }
        for (; d < key_dim; ++d) sh[row * key_dim + d] *= decay[d];
      }
      for (int64_t row = 0; row < value_dim; ++row) {
        auto projected_vec = _mm512_setzero_ps();
        int64_t d = 0;
        for (; d + 16 <= key_dim; d += 16) {
          projected_vec = _mm512_fmadd_ps(
              _mm512_loadu_ps(sh + row * key_dim + d),
              _mm512_loadu_ps(k_normalized.data() + d), projected_vec);
        }
        float projected = _mm512_reduce_add_ps(projected_vec);
        for (; d < key_dim; ++d)
          projected += sh[row * key_dim + d] * k_normalized[d];
        const auto delta = (vh[row] - projected) * beta_value;
        const auto delta_vec = _mm512_set1_ps(delta);
        d = 0;
        for (; d + 16 <= key_dim; d += 16) {
          const auto state_vec = _mm512_loadu_ps(sh + row * key_dim + d);
          _mm512_storeu_ps(
              sh + row * key_dim + d,
              _mm512_fmadd_ps(delta_vec,
                              _mm512_loadu_ps(k_normalized.data() + d),
                              state_vec));
        }
        for (; d < key_dim; ++d)
          sh[row * key_dim + d] += delta * k_normalized[d];
        auto result_vec = _mm512_setzero_ps();
        d = 0;
        for (; d + 16 <= key_dim; d += 16) {
          result_vec = _mm512_fmadd_ps(_mm512_loadu_ps(sh + row * key_dim + d),
                                       _mm512_loadu_ps(q_normalized.data() + d),
                                       result_vec);
        }
        float result = _mm512_reduce_add_ps(result_vec);
        for (; d < key_dim; ++d)
          result += sh[row * key_dim + d] * q_normalized[d];
        op[(t * value_heads + hv) * value_dim + row] = result;
      }
    }
  }
  return {out.to(v.scalar_type()), state.to(initial_state.scalar_type())};
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> glm5next_kda_recurrent_cpu_binding(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g, const at::Tensor& beta,
    const at::Tensor& initial_state, double scale, bool sigmoid_beta,
    const at::Tensor& a_log, const at::Tensor& g_bias, bool compute_gate,
    double lower_bound) {
  return glm5next_kda_recurrent_cpu(q, k, v, g, beta, initial_state, scale,
                                    sigmoid_beta, a_log, g_bias, compute_gate,
                                    lower_bound);
}
