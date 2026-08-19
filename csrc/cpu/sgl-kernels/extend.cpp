// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#include "common.h"
#include "flash_attn.h"
#include "gemm.h"

namespace {

// [NOTE]: extend attention for CPU
//   1. BLOCK_M and BLOCK_N tuned for various seq lengths
//   2. can handle non-contiguous k_extend and v_extend
//   3. computes attention for prefix and extend separately
//   4. TODO: apply head dimension blocking to optimize GQA
//   5. optional tree mask for speculative decoding TARGET_VERIFY (EAGLE topk > 1):
//      `tree_mask` is a flat [batches * qlen * qlen] bool tensor in
//      TreeMaskMode::QLEN_ONLY layout, where qlen == extend_seq_lens[bs] ==
//      max_len_extend (uniform across the batch, equal to draft_token_num).
//      Row i = query draft token, column j = key draft token; true means query i
//      may attend key j (each row marks self + ancestors + root). The committed
//      prefix (stage 1) is implicitly fully visible to every draft token, which
//      is why the mask only covers the qlen x qlen new-token block; the GPU
//      FULL_MASK layout carries the prefix columns explicitly but they are
//      all-true for EAGLE. When tree_mask is absent, stage 2 falls back to the
//      plain causal mask (correct for non-spec extend and topk == 1 chains).
//

template <typename scalar_t, typename index_t, int BLOCK_M, int BLOCK_N>
void extend_attention_kernel_impl(
    scalar_t* __restrict__ o_extend,
    const scalar_t* __restrict__ q_extend,
    const scalar_t* __restrict__ k_extend,
    const scalar_t* __restrict__ v_extend,
    const scalar_t* __restrict__ k_buffer,
    const scalar_t* __restrict__ v_buffer,
    const index_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int64_t* __restrict__ seq_lens,
    const int64_t* __restrict__ encoder_lens,
    const index_t* __restrict__ extend_seq_lens,
    const index_t* __restrict__ extend_start_loc,
    const void* __restrict__ buffer,
    const scalar_t* __restrict__ sinks,
    const bool* __restrict__ tree_mask,
    int batches,
    int num_heads,
    int num_heads_kv,
    int head_size,
    int head_size_v,
    int q_strideM,
    int q_strideH,
    int ke_strideN,
    int ke_strideH,
    int ve_strideN,
    int ve_strideH,
    int k_strideN,
    int k_strideH,
    int v_strideN,
    int v_strideH,
    float sm_scale,
    int max_num_reqs,
    int max_context_len,
    int max_total_num_tokens,
    int max_len_extend,
    int buffer_size_per_thread,
    int64_t sliding_window_size,
    bool is_prefix_skipped,
    bool is_cross_attn,
    bool has_encoder_lens,
    bool has_sink) {
  // strides
  const int o_strideM = num_heads * head_size_v;
  const int o_strideH = head_size_v;

  // we use same buffer for packed key and value
  const int ldb_tmp = std::max(head_size, head_size_v);

  const int num_groups = num_heads / num_heads_kv;
  TORCH_CHECK(num_groups * num_heads_kv == num_heads);

  // number of blocks along M
  int MB = div_up(max_len_extend, BLOCK_M);

  // parallel on [batches, num_heads, BM]
  at::parallel_for(0, batches * num_heads * MB, 0, [&](int begin, int end) {
    int bs{0}, head_id{0}, mb{0};
    data_index_init(begin, bs, batches, head_id, num_heads, mb, MB);

    int tid = at::get_thread_num();
    // s_i: [BLOCK_M, BLOCK_N]
    float* __restrict__ s_i = reinterpret_cast<float*>((char*)(buffer) + tid * buffer_size_per_thread);

    // v_prime: [BLOCK_M, head_size_v]
    float* __restrict__ v_prime = s_i + BLOCK_M * BLOCK_N;

    // s_delta: [BLOCK_M, BLOCK_N]
    scalar_t* __restrict__ s_delta = reinterpret_cast<scalar_t*>(v_prime + BLOCK_M * head_size_v);

    // Btmp: [BLOCK_N, max(head_size, head_size_v)]
    scalar_t* __restrict__ Btmp = reinterpret_cast<scalar_t*>(s_delta + BLOCK_M * BLOCK_N);

    // init Btmp just once for each thread to prevent NaN
    fill_stub(Btmp, 0.f, BLOCK_N * ldb_tmp);
    fill_stub(s_delta, 0.f, BLOCK_M * BLOCK_N);

    alignas(64) float s_prime[BLOCK_M];
    alignas(64) float m_prime[BLOCK_M];

    for (int i = begin; i < end; ++i) {
      // seq_len = prefix + extend
      int head_kv_id = head_id / num_groups;
      int seq_len = seq_lens[bs];
      int seq_len_extend = extend_seq_lens[bs];
      int seq_len_prefix = seq_len - seq_len_extend;
      int seq_extend_start_loc = extend_start_loc[bs];

      int req_pool_id = req_pool_indices[bs];
      int kv_offset = (has_encoder_lens && (!is_cross_attn)) ? encoder_lens[bs] : 0;
      TORCH_CHECK(seq_len_prefix >= 0, "prefix len < 0!");
      TORCH_CHECK(seq_len <= max_context_len, "seq_len out of scope!");
      TORCH_CHECK(req_pool_id < max_num_reqs, "req_pool_id out of scope!");

      if (is_prefix_skipped) {
        TORCH_CHECK(seq_len_prefix == 0, "extend attention: expect seq_len_prefix to be 0, got ", seq_len_prefix);
      }

      if (tree_mask != nullptr) {
        // QLEN_ONLY layout assumes a uniform qlen across the batch (TARGET_VERIFY)
        TORCH_CHECK(
            seq_len_extend == max_len_extend,
            "extend attention: tree_mask requires uniform extend_seq_lens, got ",
            seq_len_extend,
            " vs ",
            max_len_extend);
      }

      // offset and size in MB
      int m = mb * BLOCK_M;
      int m_size = std::min(BLOCK_M, seq_len_extend - m);

      if (m_size <= 0) {
        data_index_step(bs, batches, head_id, num_heads, mb, MB);
        continue;
      }

      // get query
      const scalar_t* __restrict__ q_ptr = q_extend + (seq_extend_start_loc + m) * q_strideM + head_id * q_strideH;

      // init v', s' and m'
      fill_stub(v_prime, 0.f, m_size * head_size_v);
      fill_stub(s_prime, 0.f, m_size);
      fill_stub(m_prime, -std::numeric_limits<scalar_t>::infinity(), m_size);
      // stage 1: compute scores with prefix
      int kv_start = 0;
      int kv_end = is_cross_attn ? encoder_lens[bs] : seq_len_prefix;
      for (int n = kv_start; n < kv_end; n += BLOCK_N) {
        int n_size = std::min(BLOCK_N, kv_end - n);

        // `n_size` is K in 2nd gemm, pad to TILE_K;
        const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

        // get key and pack
        pack_vnni<scalar_t, index_t>(
            /*    dst */ Btmp,
            /*    src */ k_buffer + head_kv_id * k_strideH,
            /*    ind */ req_to_token + req_pool_id * max_context_len + n + kv_offset,
            /*     N  */ n_size,
            /*     K  */ head_size,
            /* ld_src */ k_strideN,
            /* ld_dst */ BLOCK_N);

        // calculate s_i <- Q @ K
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ n_size,
            /* K     */ head_size,
            /* lda   */ q_strideM,
            /* ldb   */ BLOCK_N,
            /* ldc   */ BLOCK_N,
            /* add_C */ false,
            /* A     */ q_ptr,
            /* B     */ Btmp,
            /* C     */ s_i);

        for (int row = 0; row < m_size; ++row) {
          if (sliding_window_size > 0) {
            int last_col = seq_len_prefix + row + m - sliding_window_size + 1;
            if (last_col >= n + n_size) {
              continue;
            }
            fill_stub(s_i + row * BLOCK_N, -std::numeric_limits<float>::infinity(), last_col - n);
          }
          flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
              s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, sm_scale, row);
        }

        // get value and pack
        pack_vnni2<scalar_t>(
            /*    dst */ Btmp,
            /*    src */ v_buffer + head_kv_id * v_strideH,
            /*    ind */ req_to_token + req_pool_id * max_context_len + n + kv_offset,
            /*     K  */ n_size,
            /*     N  */ head_size_v,
            /* ld_src */ v_strideN,
            /* ld_dst */ head_size_v);

        // calculate V' <- s_delta @ V + V'
        at::native::cpublas::brgemm(
            /* M     */ m_size,
            /* N     */ head_size_v,
            /* K     */ padded_n_size,  // n_size
            /* lda   */ BLOCK_N,
            /* ldb   */ head_size_v,
            /* ldc   */ head_size_v,
            /* add_C */ true,
            /* A     */ s_delta,
            /* B     */ Btmp,
            /* C     */ v_prime);
      }  // loop with seq_len_prefix
      if (!is_cross_attn) {
        // stage 2: compute the triangle part
        int num_keys = std::min(seq_len_extend, m + BLOCK_M);
        for (int n = 0; n < num_keys; n += BLOCK_N) {
          int n_size = std::min(BLOCK_N, num_keys - n);

          // `n_size` is K in 2nd gemm, pad to TILE_K;
          const int padded_n_size = div_up(n_size, TILE_K) * TILE_K;

          // get key and pack
          pack_vnni<scalar_t>(
              /*    dst */ Btmp,
              /*    src */ k_extend + (seq_extend_start_loc + n) * ke_strideN + head_kv_id * ke_strideH,
              /*     N  */ n_size,
              /*     K  */ head_size,
              /* ld_src */ ke_strideN,
              /* ld_dst */ BLOCK_N);

          // calculate s_i <- Q @ K
          at::native::cpublas::brgemm(
              /* M     */ m_size,
              /* N     */ n_size,
              /* K     */ head_size,
              /* lda   */ q_strideM,
              /* ldb   */ BLOCK_N,
              /* ldc   */ BLOCK_N,
              /* add_C */ false,
              /* A     */ q_ptr,
              /* B     */ Btmp,
              /* C     */ s_i);

          // apply tree mask (speculative TARGET_VERIFY) or causal mask
          if (tree_mask != nullptr) {
            // [Note] tree mask for EAGLE topk > 1 (TreeMaskMode::QLEN_ONLY).
            // mask[bs][m + row][n + col] == false -> query draft token (m + row)
            // may not attend key draft token (n + col); set the score to -inf
            // before softmax. The tree mask subsumes the causal constraint:
            // ancestors always precede descendants in the draft token ordering,
            // so permitted keys satisfy j <= i and the causal `num_keys` bound
            // above remains valid.
            const bool* __restrict__ mask_base =
                tree_mask + (static_cast<int64_t>(bs) * seq_len_extend + m) * seq_len_extend + n;
            for (int row = 0; row < m_size; ++row) {
              float* __restrict__ row_ptr = s_i + row * BLOCK_N;
              const bool* __restrict__ mask_ptr = mask_base + static_cast<int64_t>(row) * seq_len_extend;
              for (int col = 0; col < n_size; ++col) {
                if (!mask_ptr[col]) {
                  row_ptr[col] = -std::numeric_limits<float>::infinity();
                }
              }
            }
          } else if (n + n_size - 1 > m) {
            // apply causal mask
            // [Note] condition to apply causal mask.
            // Mask any block whose last key (n + n_size - 1) is strictly after the first query position (m), i.e. n +
            // n_size - 1 > m. The original condition was `num_keys - n <= BLOCK_N` (last n-block only). That was
            // correct when BLOCK_M <= BLOCK_N/2 because earlier n-blocks were guaranteed to contain only past keys.
            // With BLOCK_M=512, BLOCK_N=768:
            //   BLOCK_M > BLOCK_N/2, so the first n-block can contain future keys.
            //   Example: m=512 (mb=1), num_keys=1024, first n-block covers keys [0, 768).
            //   Query row=0 is at position 512, so keys 513..767 are future and must be
            //   masked — but `num_keys - 0 = 1024 > BLOCK_N` skips masking entirely,
            //   producing wrong (non-causal) attention for rows 0..254 of this m-block.
            for (int row = 0; row < m_size; ++row) {
              int last_col = m + row - n;
              // [Note] mask the entire row if last_col < 0.
              // Clamp to -1: when n > m + row every key in this block is a future
              // key, so the entire row should be masked.  Without this clamp,
              // last_col+1 <= 0 and fill_stub would write before row_ptr.
              last_col = std::max(last_col, -1);
              // fill [last_col + 1, n_size) to -inf
              float* row_ptr = s_i + row * BLOCK_N;
              fill_stub(row_ptr + last_col + 1, -std::numeric_limits<float>::infinity(), n_size - last_col - 1);
            }
          }

          for (int row = 0; row < m_size; ++row) {
            if (sliding_window_size > 0 && row + m + 1 >= n + sliding_window_size - 1 &&
                row + m + 1 < n + sliding_window_size + n_size) {
              fill_stub(
                  s_i + row * BLOCK_N, -std::numeric_limits<float>::infinity(), row + m - n - sliding_window_size + 1);
            } else if (sliding_window_size > 0 && row + m + 1 >= n + sliding_window_size) {
              continue;
            }
            flash_attn_softmax<scalar_t, BLOCK_M, BLOCK_N>::apply(
                s_i, s_delta, v_prime, s_prime, m_prime, m_size, n_size, padded_n_size, head_size_v, sm_scale, row);
          }

          // get value and pack
          pack_vnni2<scalar_t>(
              /*    dst */ Btmp,
              /*    src */ v_extend + (seq_extend_start_loc + n) * ve_strideN + head_kv_id * ve_strideH,
              /*     K  */ n_size,
              /*     N  */ head_size_v,
              /* ld_src */ ve_strideN,
              /* ld_dst */ head_size_v);

          // calculate V' <- s_delta @ V + V'
          at::native::cpublas::brgemm(
              /* M     */ m_size,
              /* N     */ head_size_v,
              /* K     */ padded_n_size,  // n_size
              /* lda   */ BLOCK_N,
              /* ldb   */ head_size_v,
              /* ldc   */ head_size_v,
              /* add_C */ true,
              /* A     */ s_delta,
              /* B     */ Btmp,
              /* C     */ v_prime);
        }  // loop with seq_len_extend
      }
      scalar_t* __restrict__ out_ptr = o_extend + (seq_extend_start_loc + m) * o_strideM + head_id * o_strideH;
      for (int row = 0; row < m_size; ++row) {
        if (has_sink) {
          s_prime[row] += std::exp(sinks[head_id] - m_prime[row]);
        }
        float s = 1 / s_prime[row];
        copy_stub<scalar_t>(out_ptr + row * o_strideM, v_prime + row * head_size_v, s, head_size_v);
      }

      // move to the next index
      data_index_step(bs, batches, head_id, num_heads, mb, MB);
    }
    at::native::cpublas::brgemm_release();
  });
}

}  // anonymous namespace

template <int BLOCK_M, int BLOCK_N>
inline int resize_buffer(at::Tensor& buffer, int num_threads, int head_size, int head_size_v) {
  static_assert(BLOCK_M <= BLOCK_N, "Make sure BLOCK_M <= BLOCK_N to prevent buffer overflows during causal masking");
  const int size_per_thread =
      /* s_i     */ BLOCK_M * BLOCK_N * sizeof(float) +
      /* v_prime */ BLOCK_M * head_size_v * sizeof(float) +
      /* s_delta */ BLOCK_M * BLOCK_N * sizeof(uint16_t) +
      /* Btmp    */ BLOCK_N * std::max(head_size, head_size_v) * sizeof(uint16_t);

  buffer.resize_({num_threads, size_per_thread});
  return size_per_thread;
}

#define LAUNCH_EXTEND_ATTENTION_KERNEL(BLOCK_M, BLOCK_N)                                   \
  do {                                                                                     \
    int sz = resize_buffer<BLOCK_M, BLOCK_N>(buffer, num_threads, head_size, head_size_v); \
                                                                                           \
    extend_attention_kernel_impl<scalar_t, index_t, BLOCK_M, BLOCK_N>(                     \
        o_extend.data_ptr<scalar_t>(),                                                     \
        q_extend.data_ptr<scalar_t>(),                                                     \
        k_extend.data_ptr<scalar_t>(),                                                     \
        v_extend.data_ptr<scalar_t>(),                                                     \
        k_buffer.data_ptr<scalar_t>(),                                                     \
        v_buffer.data_ptr<scalar_t>(),                                                     \
        req_to_token.data_ptr<index_t>(),                                                  \
        req_pool_indices.data_ptr<int64_t>(),                                              \
        seq_lens.data_ptr<int64_t>(),                                                      \
        encoder_lens_t.data_ptr<int64_t>(),                                                \
        extend_seq_lens.data_ptr<index_t>(),                                               \
        extend_start_loc.data_ptr<index_t>(),                                              \
        buffer.data_ptr(),                                                                 \
        sinks_tensor.data_ptr<scalar_t>(),                                                 \
        tree_mask_ptr,                                                                     \
        num_seqs,                                                                          \
        num_heads,                                                                         \
        num_heads_kv,                                                                      \
        head_size,                                                                         \
        head_size_v,                                                                       \
        q_strideM,                                                                         \
        q_strideH,                                                                         \
        ke_strideN,                                                                        \
        ke_strideH,                                                                        \
        ve_strideN,                                                                        \
        ve_strideH,                                                                        \
        k_strideN,                                                                         \
        k_strideH,                                                                         \
        v_strideN,                                                                         \
        v_strideH,                                                                         \
        sm_scale,                                                                          \
        max_num_reqs,                                                                      \
        max_context_len,                                                                   \
        max_total_num_tokens,                                                              \
        max_len_extend,                                                                    \
        sz,                                                                                \
        sliding_window_size,                                                               \
        is_prefix_skipped,                                                                 \
        is_cross_attn,                                                                     \
        has_encoder_lens,                                                                  \
        has_sink);                                                                         \
  } while (0)

// q_extend, k_extend, v_extend, o_extend: contiguous tensors
// k_buffer, v_buffer: (prefix + extend) tensors in mem_manager
//
// q_extend: [num_tokens, num_heads, head_size]
// k_extend: [num_extend_tokens, num_heads, head_size]
// v_extend: [num_extend_tokens, num_heads, head_size]
// o_extend: [num_tokens, num_heads, head_size]
// k_buffer: [max_total_num_tokens, num_heads, head_size]
// v_buffer: [max_total_num_tokens, num_heads, head_size]
// req_to_token: [max_num_reqs, max_context_len] int32 or int64
// req_pool_indices: [num_seqs] int64
// seq_lens: [num_seqs] int64
// extend_seq_lens: [num_seqs]
// extend_start_loc: [num_seqs]
// encoder_lens: [num_seqs] int64 or None
// sinks: [num_heads] or None
// tree_mask: [num_seqs * max_len_extend * max_len_extend] bool or None
//   TreeMaskMode::QLEN_ONLY tree mask for speculative TARGET_VERIFY; see [NOTE] 5 above.
void extend_attention_cpu(
    at::Tensor& q_extend,
    const std::optional<at::Tensor>& k_extend_opt,
    const std::optional<at::Tensor>& v_extend_opt,
    at::Tensor& o_extend,
    at::Tensor& k_buffer,
    at::Tensor& v_buffer,
    at::Tensor& req_to_token,
    at::Tensor& req_pool_indices,
    at::Tensor& seq_lens,
    at::Tensor& extend_seq_lens,
    at::Tensor& extend_start_loc,
    int64_t max_len_extend,
    double sm_scale,
    double logit_cap,
    bool is_cross_attn,
    int64_t sliding_window_size,
    std::optional<at::Tensor> encoder_lens,
    std::optional<at::Tensor> sinks,
    std::optional<at::Tensor> tree_mask) {
  if (!is_cross_attn) {
    TORCH_CHECK(
        k_extend_opt.has_value() && v_extend_opt.has_value(),
        "k_extend and v_extend are required for non-cross attention");
  }
  // Since k_extend and v_extend are not used for cross attention, they can be initialized as k_buffer and v_buffer
  // here.
  auto k_extend = k_extend_opt.has_value() ? k_extend_opt.value() : k_buffer;
  auto v_extend = v_extend_opt.has_value() ? v_extend_opt.value() : v_buffer;

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_extend);
  CHECK_INPUT(o_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_extend);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_buffer);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(v_buffer);

  int num_seqs = seq_lens.size(0);
  int max_num_reqs = req_to_token.size(0);
  int max_context_len = req_to_token.size(1);
  int max_total_num_tokens = k_buffer.size(0);

  int num_heads = q_extend.size(1);
  int num_heads_kv = k_extend.size(1);
  int head_size = q_extend.size(2);
  int head_size_v = v_extend.size(2);

  // strides for q_extend, k_extend and v_extend
  int q_strideM = q_extend.stride(0);
  int q_strideH = q_extend.stride(1);
  int ke_strideN = k_extend.stride(0);
  int ke_strideH = k_extend.stride(1);
  int ve_strideN = v_extend.stride(0);
  int ve_strideH = v_extend.stride(1);

  // strides for k_buffer and v_buffer
  int k_strideN = k_buffer.stride(0);
  int k_strideH = k_buffer.stride(1);
  int v_strideN = v_buffer.stride(0);
  int v_strideH = v_buffer.stride(1);

  // check sizes
  CHECK_EQ(req_pool_indices.size(0), num_seqs);
  CHECK_EQ(extend_seq_lens.size(0), num_seqs);
  CHECK_EQ(extend_start_loc.size(0), num_seqs);
  CHECK_EQ(v_extend.size(1), num_heads_kv);
  CHECK_EQ(k_buffer.size(1), v_buffer.size(1));

  // MLA will skip prefix part
  const bool is_prefix_skipped = k_buffer.size(1) != num_heads_kv;

  // check index data types
  const auto index_dtype = req_to_token.scalar_type();
  TORCH_CHECK(
      index_dtype == at::kInt || index_dtype == at::kLong,
      "extend: expect req_to_token to be int32 or int64, got ",
      index_dtype);
  TORCH_CHECK(seq_lens.scalar_type() == at::kLong, "extend: expect req_lens to be int64, got ", seq_lens.scalar_type());
  TORCH_CHECK(
      req_pool_indices.scalar_type() == at::kLong,
      "extend: expect req_pool_indices to be int64, got ",
      req_pool_indices.scalar_type());
  TORCH_CHECK(
      extend_seq_lens.scalar_type() == index_dtype && extend_start_loc.scalar_type() == index_dtype,
      "extend: expect extend_seq_lens and extend_start_loc to have same dtype as req_to_token.");

  // D and DV need to be 32x as we transpose by 512-bit
  TORCH_CHECK(head_size % 32 == 0, "invalid head_size ", head_size);
  TORCH_CHECK(head_size_v % 32 == 0, "invalid head_size_v ", head_size_v);

  int num_threads = at::get_num_threads();
  auto buffer = at::empty({}, q_extend.options().dtype(at::kChar));

  bool has_encoder_lens = encoder_lens.has_value();
  // Since encoder_lens is not used when it is None, encoder_lens_t can be initialized as any tensor of int64_t dtype.
  at::Tensor encoder_lens_t = seq_lens;
  if (has_encoder_lens) {
    encoder_lens_t = encoder_lens.value();
    CHECK_EQ(encoder_lens_t.size(0), num_seqs);
  }
  bool has_sink = sinks.has_value();
  at::Tensor sinks_tensor = has_sink ? sinks.value() : at::empty({num_heads}, q_extend.options());
  CHECK_DIM(1, sinks_tensor);
  CHECK_EQ(sinks_tensor.size(0), num_heads);

  const bool* tree_mask_ptr = nullptr;
  if (tree_mask.has_value()) {
    const at::Tensor& tree_mask_t = tree_mask.value();
    CHECK_INPUT(tree_mask_t);
    TORCH_CHECK(
        tree_mask_t.scalar_type() == at::kBool, "extend: expect tree_mask to be bool, got ", tree_mask_t.scalar_type());
    TORCH_CHECK(
        tree_mask_t.numel() == static_cast<int64_t>(num_seqs) * max_len_extend * max_len_extend,
        "extend: expect tree_mask numel to be num_seqs * max_len_extend^2 = ",
        static_cast<int64_t>(num_seqs) * max_len_extend * max_len_extend,
        ", got ",
        tree_mask_t.numel());
    TORCH_CHECK(!is_cross_attn, "extend: tree_mask is not supported for cross attention");
    // The window mask derives query positions from the row index
    // (seq_len_prefix + m + row), but tree-mask rows sit at their tree depth,
    // which is <= the row index; combining the two would over-mask the prefix.
    TORCH_CHECK(sliding_window_size <= 0, "extend: tree_mask is not supported with sliding window attention");
    tree_mask_ptr = tree_mask_t.data_ptr<bool>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_extend.scalar_type(), "extend_attention_kernel", [&] {
    AT_DISPATCH_INDEX_TYPES(index_dtype, "extend_attention_indices", [&] {
      if (max_len_extend <= 256) {
        LAUNCH_EXTEND_ATTENTION_KERNEL(32, 64);
      } else if (max_len_extend <= 1024) {
        LAUNCH_EXTEND_ATTENTION_KERNEL(128, 256);
      } else if (max_len_extend <= 4096) {
        LAUNCH_EXTEND_ATTENTION_KERNEL(256, 768);
      } else {  // max_len_extend > 4096
        LAUNCH_EXTEND_ATTENTION_KERNEL(512, 768);
      }
    });
  });
}
