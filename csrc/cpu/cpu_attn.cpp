#include "cpu_attn_vec.hpp"
#include "cpu_attn_vec16.hpp"

#ifdef CPU_CAPABILITY_AMXBF16
  #include "cpu_attn_amx.hpp"
  #define AMX_DISPATCH(...)                                                   \
    case cpu_attention::ISA::AMX: {                                           \
      using attn_impl = cpu_attention::AttentionImpl<cpu_attention::ISA::AMX, \
                                                     scalar_t, head_dim>;     \
      return __VA_ARGS__();                                                   \
    }
#else
  #define AMX_DISPATCH(...) case cpu_attention::ISA::AMX:
#endif

#ifdef __aarch64__
  #include "cpu_attn_neon.hpp"
  // NEON requires head_dim to be a multiple of 32
  #define NEON_DISPATCH(...)                                                   \
    case cpu_attention::ISA::NEON: {                                           \
      using attn_impl = cpu_attention::AttentionImpl<cpu_attention::ISA::NEON, \
                                                     scalar_t, head_dim>;      \
      return __VA_ARGS__();                                                    \
    }
#else
  #define NEON_DISPATCH(...) case cpu_attention::ISA::NEON:
#endif  // #ifdef __aarch64__

#define CPU_ATTN_DISPATCH_CASE(HEAD_DIM, ...) \
  case HEAD_DIM: {                            \
    constexpr size_t head_dim = HEAD_DIM;     \
    return __VA_ARGS__();                     \
  }

#define CPU_ATTN_DISPATCH_CASE_HEADDIM(HEAD_DIM, ...)           \
  [&] {                                                         \
    switch (HEAD_DIM) {                                         \
      CPU_ATTN_DISPATCH_CASE(32, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(64, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(80, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(96, __VA_ARGS__)                   \
      CPU_ATTN_DISPATCH_CASE(112, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(128, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(160, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(192, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(224, __VA_ARGS__)                  \
      CPU_ATTN_DISPATCH_CASE(256, __VA_ARGS__)                  \
      default: {                                                \
        TORCH_CHECK(false, "Invalid CPU attention head_dim: " + \
                               std::to_string(HEAD_DIM));       \
      }                                                         \
    }                                                           \
  }()

#define CPU_ATTN_DISPATCH_IMPL(ISA_TYPE, ...)                                 \
  [&] {                                                                       \
    switch (ISA_TYPE) {                                                       \
      AMX_DISPATCH(__VA_ARGS__)                                               \
      NEON_DISPATCH(__VA_ARGS__)                                              \
      case cpu_attention::ISA::VEC: {                                         \
        using attn_impl =                                                     \
            cpu_attention::AttentionImpl<cpu_attention::ISA::VEC, scalar_t,   \
                                         head_dim>;                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      case cpu_attention::ISA::VEC16: {                                       \
        using attn_impl =                                                     \
            cpu_attention::AttentionImpl<cpu_attention::ISA::VEC16, scalar_t, \
                                         head_dim>;                           \
        return __VA_ARGS__();                                                 \
      }                                                                       \
      default: {                                                              \
        TORCH_CHECK(false, "Invalid CPU attention ISA type.");                \
      }                                                                       \
    }                                                                         \
  }()

torch::Tensor get_scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q,
    const int64_t num_heads_kv, const int64_t head_dim,
    const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool casual,
    const int64_t window_size, const std::string& isa_hint,
    const bool enable_kv_split) {
  cpu_attention::ISA isa;
  if (isa_hint == "amx") {
    isa = cpu_attention::ISA::AMX;
  } else if (isa_hint == "vec") {
    isa = cpu_attention::ISA::VEC;
  } else if (isa_hint == "vec16") {
    isa = cpu_attention::ISA::VEC16;
  } else if (isa_hint == "neon") {
    isa = cpu_attention::ISA::NEON;
  } else {
    TORCH_CHECK(false, "Unsupported CPU attention ISA hint: " + isa_hint);
  }

  cpu_attention::AttentionScheduler::ScheduleInput input;
  input.num_reqs = num_req;
  input.num_heads_q = num_heads_q;
  input.num_heads_kv = num_heads_kv;
  input.head_dim = head_dim;
  input.query_start_loc = query_start_loc.data_ptr<int32_t>();
  input.seq_lens = seq_lens.data_ptr<int32_t>();
  if (window_size != -1) {
    input.left_sliding_window_size = window_size - 1;
    if (casual) {
      input.right_sliding_window_size = 0;
    } else {
      input.right_sliding_window_size = window_size - 1;
    }
  } else {
    input.left_sliding_window_size = -1;
    if (casual) {
      input.right_sliding_window_size = 0;
    } else {
      input.right_sliding_window_size = -1;
    }
  }
  input.casual = casual;
  input.isa = isa;
  input.enable_kv_split = enable_kv_split;

  VLLM_DISPATCH_FLOATING_TYPES(dtype, "get_scheduler_metadata", [&]() {
    CPU_ATTN_DISPATCH_CASE_HEADDIM(head_dim, [&] {
      CPU_ATTN_DISPATCH_IMPL(isa, [&]() {
        input.elem_size = sizeof(scalar_t);
        input.q_buffer_elem_size = sizeof(attn_impl::q_buffer_t);
        input.logits_buffer_elem_size = sizeof(attn_impl::logits_buffer_t);
        input.output_buffer_elem_size =
            sizeof(attn_impl::partial_output_buffer_t);
        input.max_num_q_per_iter = attn_impl::MaxQHeadNumPerIteration;
        input.kv_block_alignment = attn_impl::BlockSizeAlignment;
      });
    });
  });

  cpu_attention::AttentionScheduler scheduler;
  torch::Tensor metadata = scheduler.schedule(input);
  return metadata;
}

void cpu_attn_reshape_and_cache(
    const torch::Tensor& key,    // [token_num, head_num, head_size]
    const torch::Tensor& value,  // [token_num, head_num, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    const torch::Tensor& slot_mapping, const std::string& isa) {
  TORCH_CHECK_EQ(key.dim(), 3);
  TORCH_CHECK_EQ(value.dim(), 3);
  TORCH_CHECK_EQ(key_cache.dim(), 4);
  TORCH_CHECK_EQ(value_cache.dim(), 4);
  TORCH_CHECK_EQ(key.stride(2), 1);
  TORCH_CHECK_EQ(value.stride(2), 1);

  const int64_t token_num = key.size(0);
  const int64_t key_token_num_stride = key.stride(0);
  const int64_t value_token_num_stride = value.stride(0);
  const int64_t head_num = value.size(1);
  const int64_t key_head_num_stride = key.stride(1);
  const int64_t value_head_num_stride = value.stride(1);
  const int64_t num_blocks = key_cache.size(0);
  const int64_t num_blocks_stride = key_cache.stride(0);
  const int64_t cache_head_num_stride = key_cache.stride(1);
  const int64_t block_size = key_cache.size(2);
  const int64_t block_size_stride = key_cache.stride(2);
  const int64_t head_dim = key.size(-1);

  cpu_attention::ISA isa_tag = [&]() {
    if (isa == "amx") {
      return cpu_attention::ISA::AMX;
    } else if (isa == "vec") {
      return cpu_attention::ISA::VEC;
    } else if (isa == "vec16") {
      return cpu_attention::ISA::VEC16;
    } else if (isa == "neon") {
      return cpu_attention::ISA::NEON;
    } else {
      TORCH_CHECK(false, "Invalid ISA type: " + isa);
    }
  }();

  VLLM_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "cpu_attn_reshape_and_cache", [&]() {
        CPU_ATTN_DISPATCH_CASE_HEADDIM(head_dim, [&] {
          CPU_ATTN_DISPATCH_IMPL(isa_tag, [&]() {
            attn_impl::reshape_and_cache(
                key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(), token_num,
                key_token_num_stride, value_token_num_stride, head_num,
                key_head_num_stride, value_head_num_stride, num_blocks,
                num_blocks_stride, cache_head_num_stride, block_size,
                block_size_stride);
          });
        });
      });
}

void cpu_attention_with_kv_cache(
    const torch::Tensor& query,  // [num_tokens, num_heads, head_size]
    const torch::Tensor&
        key_cache,  // [num_blocks, num_kv_heads, block_size, head_size]
    const torch::Tensor&
        value_cache,        // [num_blocks, num_kv_heads, block_size, head_size]
    torch::Tensor& output,  // [num_tokens, num_heads, head_size]
    const torch::Tensor& query_start_loc,  // [num_tokens + 1]
    const torch::Tensor& seq_lens,         // [num_tokens]
    const double scale, const bool causal,
    const std::optional<torch::Tensor>& alibi_slopes,  // [num_heads]
    const int64_t sliding_window_left, const int64_t sliding_window_right,
    const torch::Tensor& block_table,  // [num_tokens, max_block_num]
    const double softcap, const torch::Tensor& scheduler_metadata,
    const std::optional<torch::Tensor>& s_aux  // [num_heads]
) {
  TORCH_CHECK_EQ(query.dim(), 3);
  TORCH_CHECK_EQ(query.stride(2), 1);
  TORCH_CHECK_EQ(key_cache.dim(), 4);
  TORCH_CHECK_EQ(value_cache.dim(), 4);

  cpu_attention::AttentionInput input;
  input.metadata = reinterpret_cast<cpu_attention::AttentionMetadata*>(
      scheduler_metadata.data_ptr());
  input.num_tokens = query.size(0);
  input.num_heads = query.size(1);
  input.num_kv_heads = key_cache.size(1);
  input.block_size = key_cache.size(2);
  input.query = query.data_ptr();
  input.query_num_tokens_stride = query.stride(0);
  input.query_num_heads_stride = query.stride(1);
  input.cache_num_blocks_stride = key_cache.stride(0);
  input.cache_num_kv_heads_stride = key_cache.stride(1);
  input.blt_num_tokens_stride = block_table.stride(0);
  input.key_cache = key_cache.data_ptr();
  input.value_cache = value_cache.data_ptr();
  input.output = output.data_ptr();
  input.query_start_loc = query_start_loc.data_ptr<int32_t>();
  input.seq_lens = seq_lens.data_ptr<int32_t>();
  input.block_table = block_table.data_ptr<int32_t>();
  input.alibi_slopes =
      alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr;
  // For now sink must be bf16
  input.s_aux = s_aux.has_value() ? s_aux->data_ptr<c10::BFloat16>() : nullptr;
  input.scale = scale;
  input.causal = causal;
  input.sliding_window_left = sliding_window_left;
  input.sliding_window_right = sliding_window_right;
  if (input.causal) {
    // to make boundary calculation easier
    input.sliding_window_right = 0;
  }
  float softcap_fp32 = softcap;
  input.softcap = softcap_fp32;

  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "cpu_attention_with_kv_cache", [&]() {
        CPU_ATTN_DISPATCH_CASE_HEADDIM(query.size(2), [&] {
          CPU_ATTN_DISPATCH_IMPL(input.metadata->isa, [&]() {
            TORCH_CHECK_EQ(input.block_size % attn_impl::BlockSizeAlignment, 0);
            cpu_attention::AttentionMainLoop<attn_impl> mainloop;
            mainloop(&input);
          });
        });
      });
}
