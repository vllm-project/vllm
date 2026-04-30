// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Gemma4 MoE Expert GEMV Kernels for Decode
//
// Optimized CUDA GEMV kernels for Gemma4 MoE expert computation during the
// decode phase (small batch sizes, T <= 8). During decode, each token
// independently activates top-k experts via routing, so each expert invocation
// is a GEMV (matrix-vector multiply) rather than a batched GEMM.
//
// Architecture: Each (assignment, column_group) pair gets its own thread block.
// For Gemma4 with E=128 experts, top_k=8, and intermediate_size=352 per TP
// shard, a single token generates ~1400 blocks for gate_up + ~560 for down.
// This high block count saturates all 132 SMs on H200, achieving much higher
// utilization than the generic Triton fused_experts kernel which uses fewer,
// larger blocks.
//
// Three-phase pipeline:
//   Phase 1: gate_up GEMV  - [2*N, H] x [H, 1] per expert assignment
//   Phase 2: GELU*mul      - elementwise activation
//   Phase 3: down GEMV     - [H, N] x [N, 1] per expert assignment, with
//                            atomic accumulation weighted by routing weights
//
// Performance (isolated MoE forward, H200):
//   T=1: 5.3x faster than Triton fused_experts
//   T=4: 2.2x faster
//   T=8: 1.4x faster
//   T>16: Triton fused_experts is faster (amortizes weight loads across tokens)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;

// GELU tanh approximation: gelu(x) =
// 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
__device__ __forceinline__ float gelu_tanh_approx(float x) {
  constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
  constexpr float COEFF = 0.044715f;
  float x3 = x * x * x;
  float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// Warp-level sum reduction via shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Number of output columns each block computes
constexpr int COLS_PER_BLOCK = 4;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / WARP_SIZE;  // 8

// ---------------------------------------------------------------------------
// Phase 1: Gate+Up GEMV
// Each block computes COLS_PER_BLOCK columns of the gate_up output for one
// (token, expert_slot) assignment. All 256 threads collaborate on the
// H-dimension reduction for each column.
// ---------------------------------------------------------------------------
__global__ void gemma4_gate_up_gemv(
    const __nv_bfloat16* __restrict__ hidden_states,  // [T, H]
    const __nv_bfloat16* __restrict__ w13,            // [E, 2*N, H]
    const int* __restrict__ topk_ids,                 // [T, K]
    float* __restrict__ gate_up_out,                  // [T*K, 2*N]
    int T, int H, int N, int E, int K) {
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // Map block to (assignment, column_group)
  const int total_col_groups = (2 * N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
  const int assignment = blockIdx.x / total_col_groups;
  const int col_group = blockIdx.x % total_col_groups;

  if (assignment >= T * K) return;

  const int token_id = assignment / K;
  const int slot = assignment % K;
  const int expert_id = topk_ids[token_id * K + slot];
  if (expert_id < 0 || expert_id >= E) return;

  const __nv_bfloat16* x = hidden_states + (long long)token_id * H;
  const __nv_bfloat16* w13_e = w13 + (long long)expert_id * (2 * N) * H;

  // Shared memory for cross-warp partial sum reduction
  __shared__ float partial_sums[WARPS][COLS_PER_BLOCK];

  int col_start = col_group * COLS_PER_BLOCK;

  // Partition H dimension across all threads
  const int h_per_thread = (H + THREADS - 1) / THREADS;

  for (int c = 0; c < COLS_PER_BLOCK; c++) {
    int col = col_start + c;
    if (col >= 2 * N) break;

    const __nv_bfloat16* w_row = w13_e + (long long)col * H;

    // Each thread computes partial dot product over its H-chunk
    float dot = 0.0f;
    int h_start = tid * h_per_thread;
    int h_end = min(h_start + h_per_thread, H);

    // Vectorized loads: process 2 bf16 elements at a time
    int h = h_start;
    if (h % 2 != 0 && h < h_end) {
      dot += __bfloat162float(x[h]) * __bfloat162float(w_row[h]);
      h++;
    }
    for (; h + 1 < h_end; h += 2) {
      __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(&x[h]);
      __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162*>(&w_row[h]);
      dot += __bfloat162float(xv.x) * __bfloat162float(wv.x);
      dot += __bfloat162float(xv.y) * __bfloat162float(wv.y);
    }
    if (h < h_end) {
      dot += __bfloat162float(x[h]) * __bfloat162float(w_row[h]);
    }

    // Warp-level reduction
    dot = warp_reduce_sum(dot);

    if (lane == 0) {
      partial_sums[warp_id][c] = dot;
    }
  }

  __syncthreads();

  // First warp reduces across all warps and writes final result
  if (warp_id == 0) {
    for (int c = lane; c < COLS_PER_BLOCK; c += WARP_SIZE) {
      int col = col_start + c;
      if (col < 2 * N) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; w++) {
          sum += partial_sums[w][c];
        }
        gate_up_out[assignment * (2 * N) + col] = sum;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 2: GELU activation + element-wise multiply
// gate_up layout: [total_assignments, 2*N] where first N columns are gate,
// second N columns are up. Output: hidden[i] = gelu(gate[i]) * up[i]
// ---------------------------------------------------------------------------
__global__ void gemma4_gelu_mul_kernel(
    float* __restrict__ gate_up,     // [total_assignments, 2*N]
    float* __restrict__ hidden_out,  // [total_assignments, N]
    int total_assignments, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int assignment = idx / N;
  const int n = idx % N;

  if (assignment >= total_assignments || n >= N) return;

  float gate_val = gate_up[assignment * (2 * N) + n];
  float up_val = gate_up[assignment * (2 * N) + N + n];
  hidden_out[assignment * N + n] = gelu_tanh_approx(gate_val) * up_val;
}

// ---------------------------------------------------------------------------
// Phase 3: Down-projection GEMV
// Each block computes COLS_PER_BLOCK rows of the output for one assignment.
// Results are accumulated into the token output with atomic adds, weighted
// by the routing weight for this (token, expert) assignment.
// ---------------------------------------------------------------------------
__global__ void gemma4_down_gemv(
    const float* __restrict__ hidden,        // [total_assignments, N]
    const __nv_bfloat16* __restrict__ w2,    // [E, H, N]
    const int* __restrict__ topk_ids,        // [T, K]
    const float* __restrict__ topk_weights,  // [T, K]
    float* __restrict__ output,              // [T, H] fp32
    int T, int H, int N, int E, int K) {
  const int tid = threadIdx.x;
  const int lane = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int total_h_groups = (H + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
  const int assignment = blockIdx.x / total_h_groups;
  const int h_group = blockIdx.x % total_h_groups;

  if (assignment >= T * K) return;

  const int token_id = assignment / K;
  const int slot = assignment % K;
  const int expert_id = topk_ids[token_id * K + slot];
  const float routing_weight = topk_weights[token_id * K + slot];
  if (expert_id < 0 || expert_id >= E) return;

  const float* hid = hidden + (long long)assignment * N;
  const __nv_bfloat16* w2_e = w2 + (long long)expert_id * H * N;

  __shared__ float partial_sums[WARPS][COLS_PER_BLOCK];

  int h_start = h_group * COLS_PER_BLOCK;
  const int n_per_thread = (N + THREADS - 1) / THREADS;

  for (int c = 0; c < COLS_PER_BLOCK; c++) {
    int h = h_start + c;
    if (h >= H) break;

    const __nv_bfloat16* w2_row = w2_e + (long long)h * N;

    float dot = 0.0f;
    int n_start_t = tid * n_per_thread;
    int n_end_t = min(n_start_t + n_per_thread, N);

    for (int n = n_start_t; n < n_end_t; n++) {
      dot += hid[n] * __bfloat162float(w2_row[n]);
    }

    dot = warp_reduce_sum(dot);

    if (lane == 0) {
      partial_sums[warp_id][c] = dot;
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    for (int c = lane; c < COLS_PER_BLOCK; c += WARP_SIZE) {
      int h = h_start + c;
      if (h < H) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; w++) {
          sum += partial_sums[w][c];
        }
        atomicAdd(&output[token_id * H + h], sum * routing_weight);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Python binding: runs all three phases and returns bf16 output
// ---------------------------------------------------------------------------
torch::Tensor gemma4_moe_decode_forward(
    torch::Tensor hidden_states,  // [T, H] bf16
    torch::Tensor w13,            // [E, 2*N, H] bf16
    torch::Tensor w2,             // [E, H, N] bf16
    torch::Tensor topk_ids,       // [T, K] int32
    torch::Tensor topk_weights,   // [T, K] fp32
    int intermediate_size         // N (per TP shard)
) {
  TORCH_CHECK(hidden_states.is_cuda());
  TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16);
  TORCH_CHECK(w13.dtype() == torch::kBFloat16);
  TORCH_CHECK(w2.dtype() == torch::kBFloat16);

  const int T = hidden_states.size(0);
  const int H = hidden_states.size(1);
  const int E = w13.size(0);
  const int N = intermediate_size;
  const int K = topk_ids.size(1);
  const int total_assignments = T * K;

  // Intermediate buffers in fp32 for numerical accuracy
  auto gate_up = torch::empty(
      {total_assignments, 2 * N},
      torch::dtype(torch::kFloat32).device(hidden_states.device()));
  auto hidden_buf = torch::empty(
      {total_assignments, N},
      torch::dtype(torch::kFloat32).device(hidden_states.device()));
  auto output_fp32 = torch::zeros(
      {T, H}, torch::dtype(torch::kFloat32).device(hidden_states.device()));

  // Phase 1: Gate+Up GEMV
  {
    const int total_col_groups = (2 * N + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int blocks = total_assignments * total_col_groups;
    gemma4_gate_up_gemv<<<blocks, THREADS>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w13.data_ptr()),
        topk_ids.data_ptr<int>(), gate_up.data_ptr<float>(), T, H, N, E, K);
  }

  // Phase 2: GELU activation
  {
    const int total_elems = total_assignments * N;
    const int threads = 256;
    const int blocks = (total_elems + threads - 1) / threads;
    gemma4_gelu_mul_kernel<<<blocks, threads>>>(gate_up.data_ptr<float>(),
                                                hidden_buf.data_ptr<float>(),
                                                total_assignments, N);
  }

  // Phase 3: Down GEMV
  {
    const int total_h_groups = (H + COLS_PER_BLOCK - 1) / COLS_PER_BLOCK;
    const int blocks = total_assignments * total_h_groups;
    gemma4_down_gemv<<<blocks, THREADS>>>(
        hidden_buf.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(w2.data_ptr()),
        topk_ids.data_ptr<int>(), topk_weights.data_ptr<float>(),
        output_fp32.data_ptr<float>(), T, H, N, E, K);
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "gemma4_moe_decode_forward failed: ", cudaGetErrorString(err));

  return output_fp32.to(torch::kBFloat16);
}

// When built via JIT (torch.utils.cpp_extension.load), TORCH_EXTENSION_NAME
// is defined and we need the pybind11 module. When built via CMake as part
// of _moe_C, the ops are registered in torch_bindings.cpp instead.
#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gemma4_moe_decode_forward,
        "Gemma4 MoE decode GEMV forward (gate_up + GELU + down)",
        py::arg("hidden_states"), py::arg("w13"), py::arg("w2"),
        py::arg("topk_ids"), py::arg("topk_weights"),
        py::arg("intermediate_size"));
}
#endif
