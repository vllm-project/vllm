#include <torch/all.h>
#include <hip/hip_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "fast_hadamard_transform_common.h"

#include "FHT.h"
#include <cmath>

using input_t = __hip_bfloat16;

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

void set_hadamard_params(HadamardParamsBase& params,
                         // sizes
                         const size_t batch, const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x, const at::Tensor out,
                         float scale) {
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  params.batch = batch;
  params.dim = dim;
  params.log_N = int(ceil(std::log2(dim / multiple)));

  // Set the pointers and strides.
  params.x_ptr = x.data_ptr();
  params.out_ptr = out.data_ptr();
  // All stride are in elements, not bytes.
  params.x_batch_stride = x.stride(0);
  params.out_batch_stride = out.stride(0);

  params.scale = scale;
}

template <int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_kernel_traits {
  using input_t = input_t_;
  static constexpr int kNThreads = kNThreads_;
  static constexpr int kLogN = kLogN_;
  static constexpr int N = 1 << kLogN;
  static constexpr int kNBytes = sizeof(input_t);
  static_assert(kNBytes == 2 || kNBytes == 4);
  static constexpr int kNElts = kLogN == 9 ? 8 : 16;  // kNBytes == 4 ? 4 : 8;
  // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
  // (since then we'd have 8 values of float, and each round we can exchange 4
  // floats).
  static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
  using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
  static constexpr int kNChunks = N / (kNElts * kNThreads);
  // We don't want to use more than 32 KB of shared memory.
  static constexpr int kSmemExchangeSize = N * 4;  // std::min(N * 4, 32 *
                                                   // 1024);
  static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
  static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
  static constexpr int kSmemSize = kSmemExchangeSize;
};

template <typename Ktraits>
__global__
__launch_bounds__(Ktraits::kNThreads) void fast_hadamard_transform_kernel(
    HadamardParamsBase params) {
  constexpr int kNThreads = Ktraits::kNThreads;
  constexpr int kNElts = Ktraits::kNElts;
  constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
  constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
  constexpr int kNChunks = Ktraits::kNChunks;
  using input_t = typename Ktraits::input_t;
  using vec_t = typename Ktraits::vec_t;

  constexpr int kLogNElts = cilog2(Ktraits::kNElts);
  static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
  constexpr int kWarpSize = 64;  // std::min(kNThreads, 32);
  constexpr int kLogWarpSize = cilog2(kWarpSize);
  static_assert(1 << kLogWarpSize == kWarpSize,
                "Warp size must be a power of 2");
  constexpr int kNWarps = kNThreads / kWarpSize;
  constexpr int kLogNWarps = cilog2(kNWarps);
  static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
  constexpr int kLoadsPerExchange =
      Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
  static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads ==
                    Ktraits::kSmemExchangeSize,
                "kSmemExchangeSize should be a power of 2");
  static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) ==
                kNChunks * kNElts * sizeof(float));

  constexpr int kChunksPerExchange =
      Ktraits::kSmemExchangeSize /
      (sizeof(vec_t) * kNExchangePerVec * kNThreads);
  static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec *
                    kNThreads ==
                Ktraits::kSmemExchangeSize);
  constexpr int kNExchanges = kNChunks / kChunksPerExchange;
  static_assert(kNExchanges * kChunksPerExchange == kNChunks);

  // Shared memory.
  extern __shared__ char smem_[];
  vec_t* smem_exchange = reinterpret_cast<vec_t*>(smem_);

  const int batch_id = blockIdx.x;
  input_t* x = reinterpret_cast<input_t*>(params.x_ptr) +
               batch_id * params.x_batch_stride;
  input_t* out = reinterpret_cast<input_t*>(params.out_ptr) +
                 batch_id * params.out_batch_stride;

  float x_vals[kNChunks][kNElts];
  load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);

  hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
  hadamard_mult_warp<kWarpSize, kLogWarpSize, 0, kNChunks, kNElts>(
      x_vals);  //<- this is the problem - works for <5,0,2,8> but not for
                //<5,0,28,4>

  if constexpr (kNWarps > 1) {
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps,
                      true, vec_t>(x_vals, smem_exchange);
    hadamard_mult_warp<kWarpSize, kLogNWarps, 0, kNChunks, kNElts>(x_vals);
    exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps,
                      false, vec_t>(x_vals, smem_exchange);
  }

  if constexpr (kNChunks > 1) {
    float x_vals_transposed[kNElts][kNChunks];
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals_transposed[i][c] = x_vals[c][i];
      }
    }

    if constexpr (kNChunks == 28) {
      hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
    } else {
      constexpr int kLogNChunks = cilog2(kNChunks);
      static_assert(1 << kLogNChunks == kNChunks,
                    "kNChunks must be a power of 2");
      hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
    }

#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNElts; ++i) {
        x_vals[c][i] = x_vals_transposed[i][c];
      }
    }
  }

  store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim,
                                          params.scale);
}

template <int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_launch(HadamardParamsBase& params,
                                    cudaStream_t stream) {
  using Ktraits =
      fast_hadamard_transform_kernel_traits<kNThreads, kLogN, input_t>;
  constexpr int kSmemSize = Ktraits::kSmemSize;

  dim3 grid(params.batch);
  auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
  kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor fast_hadamard_transform(at::Tensor& x, double scale) {
  // NOTE: reshape and get batch_size
  const auto shapes_og = x.sizes();
  const int dim_og = x.size(-1);
  x = x.reshape({-1, dim_og});
  if (x.stride(-1) != 1) {
    x = x.contiguous();
  }
  const auto sizes = x.sizes();
  const int batch_size = sizes[0];

  // NOTE: mystery checks
  CHECK_SHAPE(x, batch_size, dim_og);
  TORCH_CHECK(x.stride(1) == 1);

  // NOTE: get dim
  const int dim = x.size(1);

  // NOTE: get output tensor
  at::Tensor out = torch::empty_like(x);

  // NOTE: construct params
  HadamardParamsBase params;
  set_hadamard_params(params, batch_size, dim, 1, x, out,
                      static_cast<float>(scale));

  auto stream = at::cuda::getCurrentCUDAStream();

  if (dim_og == 512) {
    fast_hadamard_transform_launch<64, 9, input_t>(params, stream);
  } else if (dim_og == 1024) {
    fast_hadamard_transform_launch<64, 10, input_t>(params, stream);
  }

  return out.reshape(shapes_og);
}