#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/all.h>

// _dyn_quant_matmul_4bit is only available on AArch64.
#if defined(__aarch64__)
  #include <ATen/ops/_dyn_quant_matmul_4bit.h>
#endif

inline torch::Tensor mm(const torch::Tensor& a, const torch::Tensor& packed_w,
                        int64_t group_size_eff, int64_t in_features,
                        int64_t out_features) {
#if defined(__aarch64__)
  return at::_ops::_dyn_quant_matmul_4bit::call(a, packed_w, group_size_eff,
                                                in_features, out_features);
#else
  TORCH_CHECK(false,
              "dynamic 4-bit int MoE path requires AArch64 (ARM64); "
              "_dyn_quant_matmul_4bit is unavailable on this architecture");
  return {};
#endif
}

enum ActivationKind : int64_t {
  SwiGLU_Gu = 0,  // act = SiLU(g) * u
  SwiGLUOAI = 1,  // act = SiLU(u) * g
  SiLU = 2        // SiLU
};

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I,
    int64_t I2, int64_t group_size, bool apply_router_weight_on_input,
    int64_t activation_kind) {
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.dim() == 2,
              "topk tensors must be [T, K]");
  TORCH_CHECK(
      w13_packed.size(0) == w2_packed.size(0),
      "w13_packed and w2_packed must have same number of experts in dim 0");
  TORCH_CHECK(I2 == 2 * I, "I2 must equal 2*I");

  const int64_t T = x.size(0);
  const int64_t K = topk_ids.size(1);
  const int64_t E = w13_packed.size(0);
  const int64_t N = T * K;

  auto x_c = x.contiguous();
  auto ids_c = topk_ids.contiguous();
  auto gates_c = topk_weights.to(at::kFloat).contiguous();

  // bucketing tokens -> experts
  c10::SmallVector<int64_t, 64> counts(
      E, 0);  // Small vector uses stack allocation
  {
    const auto* ids_ptr = ids_c.data_ptr<int64_t>();
    for (int64_t i = 0; i < N; ++i) {
      const int64_t e_id = ids_ptr[i];
      TORCH_CHECK(0 <= e_id && e_id < E, "expert id out of range");
      counts[e_id]++;
    }
  }
  c10::SmallVector<int64_t, 65> offsets(E + 1, 0);  // ( E +1 )
  for (int64_t e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + counts[e];

  auto expert_tokens = at::empty({offsets[E]}, ids_c.options());
  auto expert_gates = at::empty({offsets[E]}, gates_c.options());
  {
    c10::SmallVector<int64_t, 64> cursor(E, 0);
    const auto* ids_ptr = ids_c.data_ptr<int64_t>();
    const auto* gts_ptr = gates_c.data_ptr<float>();
    auto* tok_ptr = expert_tokens.data_ptr<int64_t>();
    auto* gate_ptr = expert_gates.data_ptr<float>();

    for (int64_t t = 0; t < T; ++t) {
      const int64_t base = t * K;
      for (int64_t k = 0; k < K; ++k) {
        const int64_t idx = base + k;
        const int64_t e = ids_ptr[idx];
        const int64_t p = offsets[e] + (cursor[e]++);
        tok_ptr[p] = t;
        gate_ptr[p] = gts_ptr[idx];
      }
    }
  }

  const int64_t g_eff_13 = (group_size != -1) ? group_size : H;
  const int64_t g_eff_2 = (group_size != -1) ? group_size : I;

  auto X_all = x_c.index_select(/*dim=*/0, expert_tokens);
  if (apply_router_weight_on_input) {
    X_all = X_all.mul(expert_gates.unsqueeze(1));
  }
  auto Y_all = at::empty({offsets[E], H}, x_c.options());

  at::parallel_for(0, E, 1, [&](int64_t e_begin, int64_t e_end) {
    c10::InferenceMode guard;
    for (int64_t e = e_begin; e < e_end; ++e) {
      const int64_t te = counts[e];
      if (te == 0) {
        continue;
      }

      const int64_t start = offsets[e];

      auto x_e = X_all.narrow(/*dim=*/0, /*start=*/start, /*length=*/te);

      auto w13_e = w13_packed.select(/*dim=*/0, e);
      auto w2_e = w2_packed.select(/*dim=*/0, e);

      // W13
      auto y13 =
          mm(x_e, w13_e, g_eff_13, /*in_features=*/H, /*out_features=*/I2);

      auto g_part = y13.narrow(/*dim=*/1, /*start=*/0, /*length=*/I);
      auto u_part = y13.narrow(/*dim=*/1, /*start=*/I, /*length=*/I);

      torch::Tensor act;
      if (activation_kind == ActivationKind::SwiGLUOAI) {  // SwiGLUOAI
        constexpr double kAlpha = 1.702;                   // GPT-OSS default
        constexpr double kLimit = 7.0;                     // GPT-OSS default
        auto gate_c = at::clamp_max(g_part, kLimit);
        auto up_c = at::clamp(u_part, -kLimit, kLimit);
        auto glu = gate_c.mul(at::sigmoid(gate_c.mul(kAlpha)));
        act = up_c.add(1.0).mul(glu);
      } else {  // SiLU , SwiGLU_GU, vLLM maps silu to SiluAndMul()
        act = at::silu(g_part).mul(u_part);
      }

      // W2
      auto y = mm(act, w2_e, g_eff_2, /*in_features=*/I, /*out_features=*/H);

      // Store per-expert result
      Y_all.narrow(/*dim=*/0, /*start=*/start, /*length=*/te).copy_(y);
    }
  });

  if (!apply_router_weight_on_input) {
    Y_all = Y_all.mul(expert_gates.unsqueeze(1));
  }

  auto out = at::zeros({T, H}, x.options());
  out =
      at::index_add(out, /*dim=*/0, /*index=*/expert_tokens, /*source=*/Y_all);

  return out;
}
