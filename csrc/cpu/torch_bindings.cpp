#include "cache.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>

std::string init_cpu_threads_env(const std::string& cpu_ids);

void release_dnnl_matmul_handler(int64_t handler);

int64_t create_onednn_scaled_mm_handler(const torch::Tensor& b,
                                        const torch::Tensor& b_scales,
                                        at::ScalarType output_type,
                                        bool dynamic_act_quant, bool use_azp,
                                        int64_t primitive_cache_size);

void onednn_scaled_mm(torch::Tensor& c, const torch::Tensor& a,
                      const torch::Tensor& a_scales,
                      const std::optional<torch::Tensor>& azp,
                      const std::optional<torch::Tensor>& azp_adj,
                      const std::optional<torch::Tensor>& bias,
                      int64_t handler);

int64_t create_onednn_mm_handler(const torch::Tensor& b,
                                 int64_t primitive_cache_size);

void onednn_mm(torch::Tensor& c, const torch::Tensor& a,
               const std::optional<torch::Tensor>& bias, int64_t handler);

bool is_onednn_acl_supported();

void mla_decode_kvcache(torch::Tensor& out, torch::Tensor& query,
                        torch::Tensor& kv_cache, double scale,
                        torch::Tensor& block_tables, torch::Tensor& seq_lens);

int64_t init_shm_manager(const std::string& name, const int64_t group_size,
                         const int64_t rank);

std::string join_shm_manager(int64_t handle, const std::string& name);

void shm_allreduce(int64_t handle, torch::Tensor& data);

void shm_gather(int64_t handle, torch::Tensor& data,
                const std::optional<std::vector<torch::Tensor>>& outputs,
                int64_t dst);

void shm_all_gather(int64_t handle, const torch::Tensor& data,
                    torch::Tensor& output);

void shm_send_tensor_list(int64_t handle,
                          const std::vector<torch::Tensor>& tensor_list,
                          int64_t dst);

std::vector<torch::Tensor> shm_recv_tensor_list(int64_t handle, int64_t src);

at::Tensor weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2,
                                const std::optional<at::Tensor>& bias,
                                bool is_vnni);

at::Tensor convert_weight_packed(at::Tensor& weight);

at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states, at::Tensor& w1, at::Tensor& w2,
    at::Tensor& topk_weights, at::Tensor& topk_ids, bool inplace,
    bool use_int8_w8a8, bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& a1_scale,
    const std::optional<at::Tensor>& a2_scale, bool is_vnni);

at::Tensor int8_scaled_mm_with_quant(at::Tensor& mat1, at::Tensor& mat2,
                                     at::Tensor& scales2,
                                     const std::optional<at::Tensor>& bias,
                                     at::ScalarType out_dtype, bool is_vnni);

torch::Tensor get_scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q,
    const int64_t num_heads_kv, const int64_t head_dim,
    const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool casual,
    const int64_t window_size, const std::string& isa_hint,
    const bool enable_kv_split);

void cpu_attn_reshape_and_cache(const torch::Tensor& key,
                                const torch::Tensor& value,
                                torch::Tensor& key_cache,
                                torch::Tensor& value_cache,
                                const torch::Tensor& slot_mapping,
                                const std::string& isa);

void cpu_attention_with_kv_cache(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, torch::Tensor& output,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const double scale, const bool causal,
    const std::optional<torch::Tensor>& alibi_slopes,
    const int64_t sliding_window_left, const int64_t sliding_window_right,
    const torch::Tensor& block_table, const double softcap,
    const torch::Tensor& scheduler_metadata,
    const std::optional<torch::Tensor>& s_aux);

// Note: just for avoiding importing errors
void placeholder_op() { TORCH_CHECK(false, "Unimplemented"); }

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  ops.def(
      "dynamic_4bit_int_moe("
      "Tensor x, Tensor topk_ids, Tensor topk_weights,"
      "Tensor w13_packed, Tensor w2_packed, int H, int I, int I2,"
      "int group_size, bool apply_router_weight_on_input, int activation_kind"
      ") -> Tensor");

  ops.impl("dynamic_4bit_int_moe", torch::kCPU, &dynamic_4bit_int_moe_cpu);

  // Activation ops

  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCPU, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCPU, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCPU, &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCPU, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCPU, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kCPU, &gelu_quick);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCPU, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCPU, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCPU, &rotary_embedding);

  // Quantization
#if defined(__AVX512F__) || (defined(__aarch64__) && !defined(__APPLE__)) || \
    defined(__powerpc64__)
  at::Tag stride_tag = at::Tag::needs_fixed_stride_order;
  // Helper function to release oneDNN handlers
  ops.def("release_dnnl_matmul_handler(int handler) -> ()",
          &release_dnnl_matmul_handler);

  // Create oneDNN GEMM handler
  ops.def(
      "create_onednn_mm_handler(Tensor b, int "
      "primitive_cache_size) -> int",
      &create_onednn_mm_handler);

  // oneDNN GEMM
  ops.def(
      "onednn_mm(Tensor! c, Tensor a, Tensor? bias, "
      "int handler) -> ()");
  ops.impl("onednn_mm", torch::kCPU, &onednn_mm);

  // Check if oneDNN was built with ACL backend
  ops.def("is_onednn_acl_supported() -> bool", &is_onednn_acl_supported);

  // Create oneDNN W8A8 handler
  ops.def(
      "create_onednn_scaled_mm_handler(Tensor b, Tensor b_scales, ScalarType "
      "output_type, bool dynamic_act_quant, bool use_azp, int "
      "primitive_cache_size) -> int",
      &create_onednn_scaled_mm_handler);

  // oneDNN scaled_mm for W8A8 with static per-tensor activation quantization
  ops.def(
      "onednn_scaled_mm(Tensor! c, Tensor a, Tensor a_scales, Tensor? azp, "
      "Tensor? azp_adj, Tensor? bias, int handler) -> ()");
  ops.impl("onednn_scaled_mm", torch::kCPU, &onednn_scaled_mm);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! out, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()",
      {stride_tag});
  ops.impl("static_scaled_int8_quant", torch::kCPU, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! out, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()",
      {stride_tag});
  ops.impl("dynamic_scaled_int8_quant", torch::kCPU,
           &dynamic_scaled_int8_quant);
#endif

// SHM CCL
#ifdef __AVX512F__
  ops.def("init_shm_manager(str name, int group_size, int rank) -> int",
          &init_shm_manager);
  ops.def("join_shm_manager(int handle, str name) -> str", &join_shm_manager);
  ops.def("shm_allreduce(int handle, Tensor! data) -> ()");
  ops.impl("shm_allreduce", torch::kCPU, &shm_allreduce);
  ops.def(
      "shm_gather(int handle, Tensor data, Tensor[](a!)? outputs, int dst) -> "
      "()");
  ops.impl("shm_gather", torch::kCPU, &shm_gather);
  ops.def(
      "shm_all_gather(int handle, Tensor data, Tensor! output) -> "
      "()");
  ops.impl("shm_all_gather", torch::kCPU, &shm_all_gather);
  ops.def(
      "shm_send_tensor_list(int handle, Tensor[](a) tensor_list, int dst) -> "
      "()");
  ops.impl("shm_send_tensor_list", torch::kCPU, &shm_send_tensor_list);
  ops.def("shm_recv_tensor_list(int handle, int src) -> Tensor[](a)",
          &shm_recv_tensor_list);
#endif

  // sgl-kernels
#if defined(__AVX512BF16__) && defined(__AVX512F__) && defined(__AVX512VNNI__)
  ops.def(
      "weight_packed_linear(Tensor(a0!) mat1, Tensor(a1!) mat2, Tensor(a2!)? "
      "bias, bool is_vnni) -> Tensor");
  ops.impl("weight_packed_linear", torch::kCPU, &weight_packed_linear);
  ops.def("convert_weight_packed(Tensor! weight) -> Tensor");
  ops.impl("convert_weight_packed", torch::kCPU, &convert_weight_packed);
  ops.def(
      "fused_experts_cpu(Tensor! hidden_states, Tensor w1, Tensor w2, Tensor "
      "topk_weights, Tensor topk_ids, bool inplace, bool use_int8_w8a8, bool "
      "use_fp8_w8a16, Tensor? w1_scale, Tensor? w2_scale, SymInt[]? "
      "block_size, Tensor? a1_scale, Tensor? a2_scale, bool is_vnni) -> "
      "Tensor");
  ops.impl("fused_experts_cpu", torch::kCPU, &fused_experts_cpu);
  ops.def(
      "int8_scaled_mm_with_quant(Tensor mat1, Tensor mat2, Tensor scales2, "
      "Tensor? bias, ScalarType out_dtype, bool is_vnni) -> Tensor");
  ops.impl("int8_scaled_mm_with_quant", torch::kCPU,
           &int8_scaled_mm_with_quant);
#endif

  // CPU attention kernels
  ops.def(
      "get_scheduler_metadata(int num_req, int num_heads_q, int num_heads_kv, "
      "int head_dim, Tensor seq_lens, ScalarType dtype, Tensor "
      "query_start_loc, bool casual, int window_size, str isa_hint, bool "
      "enable_kv_split) -> Tensor",
      &get_scheduler_metadata);
  ops.def(
      "cpu_attn_reshape_and_cache(Tensor key, Tensor value, Tensor(a2!) "
      "key_cache, Tensor(a3!) value_cache, Tensor slot_mapping, str "
      "isa) -> ()",
      &cpu_attn_reshape_and_cache);
  ops.def(
      "cpu_attention_with_kv_cache(Tensor query, Tensor key_cache, Tensor "
      "value_cache, Tensor(a3!) output, Tensor query_start_loc, Tensor "
      "seq_lens, float scale, bool causal, Tensor? alibi_slopes, SymInt "
      "sliding_window_left, SymInt sliding_window_right, Tensor block_table, "
      "float softcap, Tensor sheduler_metadata, Tensor? s_aux) -> ()",
      &cpu_attention_with_kv_cache);

  // placeholders
  ops.def("static_scaled_fp8_quant() -> ()", placeholder_op);
  ops.def("dynamic_scaled_fp8_quant() -> ()", placeholder_op);
  ops.def("dynamic_per_token_scaled_fp8_quant() -> ()", placeholder_op);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _utils), utils) {
  // CPU utils
  utils.def("init_cpu_threads_env(str cpu_ids) -> str", &init_cpu_threads_env);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cpu), cpu_ops) {
  cpu_ops.def(
      "mla_decode_kvcache("
      "   Tensor! out, Tensor query, Tensor kv_cache,"
      "   float scale, Tensor block_tables, Tensor seq_lens) -> ()");
  cpu_ops.impl("mla_decode_kvcache", torch::kCPU, &mla_decode_kvcache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
