#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
namespace kda_replayssm {
struct Dims {
  int64_t N, T, H, D, P, W, S, R;
  int64_t M, MO;  // token rows (at least query_start_loc[N]), out rows
};
struct Args {
  Dims dims;
  const __nv_bfloat16* x;           // in [M, P, H, D] view
  int64_t x_stride[4];              // in elements; x_stride[3] == 1
  const __nv_bfloat16* beta;        // in [M, H] view
  int64_t beta_stride[2];           // in elements; beta_stride[1] == 1
  const __nv_bfloat16* f_a;         // in [M, D] view
  int64_t f_a_stride[2];            // in elements; f_a_stride[1] == 1
  const __nv_bfloat16* g_a;         // in [M, D] view
  int64_t g_a_stride[2];            // in elements; g_a_stride[1] == 1
  const float* conv_weight;         // in [P, W, H, D]
  const float* A_log;               // in [H]
  const float* dt_bias;             // in [H, D]
  const __nv_bfloat16* f_b_weight;  // in [H, D, D]
  const __nv_bfloat16* g_b_weight;  // in [H, D, D]
  const float* norm_weight;         // in [D]
  const int32_t* query_start_loc;   // in [N+1]
  const int32_t* slot;              // in [N]
  const int32_t* ring_slot;         // in [N]: the slot of request n's rings
  const int32_t* num_accepted;      // in [N]
  const int32_t* ring_start;        // in [N]
  const int32_t* num_committed;     // in [N]
  __nv_bfloat16* conv_state;        // inout [S, T+2, P, H, D] view
  int64_t conv_state_stride[5];     // in elements; conv_state_stride[4] == 1
  float* checkpoint;                // inout [S, H, D, D] view
  int64_t checkpoint_stride[4];     // in elements; checkpoint_stride[3] == 1
  float* u_ring;                    // inout [S, H, R, D] view
  int64_t u_ring_stride0;           // in elements; the other dims contiguous
  __nv_bfloat16* k_ring;            // inout [S, H, R, D] view
  int64_t k_ring_stride0;
  float* g_ring;  // inout [S, H, R, D] view
  int64_t g_ring_stride0;
  __nv_bfloat16* out;  // out [MO, H, D]
  float lower_bound;
  float scale;
  float norm_eps;
};
struct Ctx {
  cudaStream_t stream;  // the only stream the call may use
  int device, sm_count;
  void* scratch;
  size_t scratch_bytes;  // uninitialized at entry
};
void kernel_workspace(const Dims&, size_t* scratch_bytes);
// enqueue on ctx.stream; no sync, no allocation
void kernel_run(const Args&, const Ctx&);
}  // namespace kda_replayssm
