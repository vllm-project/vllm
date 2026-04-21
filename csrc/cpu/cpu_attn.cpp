#include "cpu_attn_dispatch_generated.h"
#include "cpu/cpu_attn_fp8.hpp"

// Maps kv_cache_dtype string to Fp8KVCacheDataType enum.
// Mirrors DISPATCH_BY_KV_CACHE_DTYPE in
// csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:
//   "fp8" / "fp8_e4m3" -> kFp8E4M3; "fp8_e5m2" -> kFp8E5M2.
static inline bool is_fp8_kv_dtype(const std::string& kv_cache_dtype) {
  return kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2" ||
         kv_cache_dtype == "fp8";
}

static inline cpu_attention::Fp8KVCacheDataType parse_fp8_kv_dtype(
    const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "fp8_e5m2")
    return cpu_attention::Fp8KVCacheDataType::kFp8E5M2;
  return cpu_attention::Fp8KVCacheDataType::kFp8E4M3;
}

// Dispatch E4M3 vs E5M2 at runtime; ISA / scalar_t / head_dim are
// compile-time constants supplied by the surrounding dispatch macros.
template <cpu_attention::ISA isa, typename scalar_t, int64_t head_dim>
static void run_fp8_attn(cpu_attention::Fp8KVCacheDataType fp8_dtype,
                         cpu_attention::AttentionInput& input) {
  if (fp8_dtype == cpu_attention::Fp8KVCacheDataType::kFp8E5M2) {
    using impl_t =
        cpu_attention::AttentionImpl<isa, scalar_t, head_dim, c10::Float8_e5m2>;
    TORCH_CHECK_EQ(input.block_size % impl_t::BlockSizeAlignment, 0);
    cpu_attention::AttentionMainLoop<impl_t> mainloop;
    mainloop(&input);
  } else {
    using impl_t = cpu_attention::AttentionImpl<isa, scalar_t, head_dim,
                                                c10::Float8_e4m3fn>;
    TORCH_CHECK_EQ(input.block_size % impl_t::BlockSizeAlignment, 0);
    cpu_attention::AttentionMainLoop<impl_t> mainloop;
    mainloop(&input);
  }
}

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
  } else if (isa_hint == "vxe") {
    isa = cpu_attention::ISA::VXE;
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
    CPU_ATTN_DISPATCH(head_dim, isa, [&]() {
      input.elem_size = sizeof(scalar_t);
      input.q_buffer_elem_size = sizeof(attn_impl::q_buffer_t);
      input.logits_buffer_elem_size = sizeof(attn_impl::logits_buffer_t);
      input.output_buffer_elem_size =
          sizeof(attn_impl::partial_output_buffer_t);
      input.max_num_q_per_iter = attn_impl::MaxQHeadNumPerIteration;
      input.kv_block_alignment = attn_impl::BlockSizeAlignment;
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
    const torch::Tensor& slot_mapping, const std::string& isa,
    const double k_scale = 1.0, const double v_scale = 1.0,
    const std::string& kv_cache_dtype = "auto") {
  TORCH_CHECK_EQ(key.dim(), 3);
  TORCH_CHECK_EQ(value.dim(), 3);
  TORCH_CHECK_EQ(key_cache.dim(), 4);
  TORCH_CHECK_EQ(value_cache.dim(), 4);

  const bool is_fp8 = is_fp8_kv_dtype(kv_cache_dtype);

  if (is_fp8) {
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Byte,
                "key_cache must be uint8 for FP8 path");
    TORCH_CHECK(value_cache.scalar_type() == at::ScalarType::Byte,
                "value_cache must be uint8 for FP8 path");

    const float k_inv = 1.0f / static_cast<float>(k_scale);
    const float v_inv = 1.0f / static_cast<float>(v_scale);
    const auto fp8_dtype = parse_fp8_kv_dtype(kv_cache_dtype);
    const bool use_e5m2 =
        (fp8_dtype == cpu_attention::Fp8KVCacheDataType::kFp8E5M2);

    const int64_t token_num = key.size(0);
    const int64_t head_num = key.size(1);
    const int64_t head_dim = key.size(2);
    const int64_t block_size = key_cache.size(2);

    VLLM_DISPATCH_FLOATING_TYPES(
        key.scalar_type(), "cpu_attn_reshape_and_cache", [&]() {
#if defined(CPU_CAPABILITY_AMXBF16)
          if (isa == "amx") {
            if (use_e5m2)
              reshape_and_cache_fp8_amx_e5m2_typed<scalar_t>(
                  key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                  key_cache.data_ptr<uint8_t>(),
                  value_cache.data_ptr<uint8_t>(),
                  slot_mapping.data_ptr<int64_t>(), token_num, head_num,
                  head_dim, block_size, key.stride(0), key.stride(1),
                  value.stride(0), value.stride(1), key_cache.stride(0),
                  key_cache.stride(1), value_cache.stride(0),
                  value_cache.stride(1), k_inv, v_inv);
            else
              reshape_and_cache_fp8_amx_typed<scalar_t>(
                  key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                  key_cache.data_ptr<uint8_t>(),
                  value_cache.data_ptr<uint8_t>(),
                  slot_mapping.data_ptr<int64_t>(), token_num, head_num,
                  head_dim, block_size, key.stride(0), key.stride(1),
                  value.stride(0), value.stride(1), key_cache.stride(0),
                  key_cache.stride(1), value_cache.stride(0),
                  value_cache.stride(1), k_inv, v_inv);
            return;
          }
#endif
          if (use_e5m2)
            reshape_and_cache_fp8_e5m2_typed<scalar_t>(
                key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
                slot_mapping.data_ptr<int64_t>(), token_num, head_num, head_dim,
                block_size, key.stride(0), key.stride(1), value.stride(0),
                value.stride(1), key_cache.stride(0), key_cache.stride(1),
                value_cache.stride(0), value_cache.stride(1), k_inv, v_inv);
          else
            reshape_and_cache_fp8_typed<scalar_t>(
                key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
                key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
                slot_mapping.data_ptr<int64_t>(), token_num, head_num, head_dim,
                block_size, key.stride(0), key.stride(1), value.stride(0),
                value.stride(1), key_cache.stride(0), key_cache.stride(1),
                value_cache.stride(0), value_cache.stride(1), k_inv, v_inv);
        });
    return;
  }

  // Non-FP8 path
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
    } else if (isa == "vxe") {
      return cpu_attention::ISA::VXE;
    } else {
      TORCH_CHECK(false, "Invalid ISA type: " + isa);
    }
  }();

  VLLM_DISPATCH_FLOATING_TYPES(
      key.scalar_type(), "cpu_attn_reshape_and_cache", [&]() {
        CPU_ATTN_DISPATCH(head_dim, isa_tag, [&]() {
          attn_impl::reshape_and_cache(
              key.data_ptr<scalar_t>(), value.data_ptr<scalar_t>(),
              key_cache.data_ptr<scalar_t>(), value_cache.data_ptr<scalar_t>(),
              slot_mapping.data_ptr<int64_t>(), token_num, key_token_num_stride,
              value_token_num_stride, head_num, key_head_num_stride,
              value_head_num_stride, num_blocks, num_blocks_stride,
              cache_head_num_stride, block_size, block_size_stride);
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
    const std::optional<torch::Tensor>& s_aux,  // [num_heads]
    const double k_scale = 1.0, const double v_scale = 1.0,
    const std::string& kv_cache_dtype = "auto") {
  TORCH_CHECK_EQ(query.dim(), 3);
  TORCH_CHECK_EQ(query.stride(2), 1);
  TORCH_CHECK_EQ(key_cache.dim(), 4);
  TORCH_CHECK_EQ(value_cache.dim(), 4);

  const bool is_fp8 = is_fp8_kv_dtype(kv_cache_dtype);
  if (is_fp8) {
    TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::Byte,
                "key_cache must be uint8 for FP8 path");
    TORCH_CHECK(value_cache.scalar_type() == at::ScalarType::Byte,
                "value_cache must be uint8 for FP8 path");
  }

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
  input.s_aux = s_aux.has_value() ? s_aux->data_ptr<c10::BFloat16>() : nullptr;
  input.scale = scale;
  input.causal = causal;
  input.sliding_window_left = sliding_window_left;
  input.sliding_window_right = sliding_window_right;
  if (input.causal) {
    input.sliding_window_right = 0;
  }
  input.softcap = static_cast<float>(softcap);

  if (is_fp8) {
    input.k_scale_fp8 = static_cast<float>(k_scale);
    input.v_scale_fp8 = static_cast<float>(v_scale);
    const auto fp8_dtype = parse_fp8_kv_dtype(kv_cache_dtype);

#if !defined(__AVX2__) && !defined(__AVX512F__)
    TORCH_CHECK(false,
                "cpu_attention_with_kv_cache requires AVX2 or AVX-512 for "
                "FP8 KV cache; not supported on this platform.");
#endif

#if defined(CPU_CAPABILITY_AMXBF16)
    if (input.metadata->isa == cpu_attention::ISA::AMX) {
      TORCH_CHECK(query.scalar_type() == at::ScalarType::BFloat16,
                  "FP8 KV cache AMX path requires BFloat16 query dtype");
      using scalar_t = c10::BFloat16;
      CPU_ATTN_DISPATCH(query.size(2), cpu_attention::ISA::AMX, [&]() {
        if constexpr (attn_impl::ISAType == cpu_attention::ISA::AMX) {
          run_fp8_attn<cpu_attention::ISA::AMX, scalar_t, attn_impl::HeadDim>(
              fp8_dtype, input);
        } else {
          TORCH_CHECK(false, "FP8 AMX is not supported for head_dim=",
                      attn_impl::HeadDim, "; use ISA vec instead");
        }
      });
      return;
    }
#endif

    VLLM_DISPATCH_FLOATING_TYPES(
        query.scalar_type(), "cpu_attention_with_kv_cache", [&]() {
          CPU_ATTN_DISPATCH(query.size(2), cpu_attention::ISA::VEC, [&]() {
            run_fp8_attn<cpu_attention::ISA::VEC, scalar_t, attn_impl::HeadDim>(
                fp8_dtype, input);
          });
        });
    return;
  }

  // Non-FP8 path
  VLLM_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "cpu_attention_with_kv_cache", [&]() {
        CPU_ATTN_DISPATCH(query.size(2), input.metadata->isa, [&]() {
          TORCH_CHECK_EQ(input.block_size % attn_impl::BlockSizeAlignment, 0);
          cpu_attention::AttentionMainLoop<attn_impl> mainloop;
          mainloop(&input);
        });
      });
}
