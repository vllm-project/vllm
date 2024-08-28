// clang-format off
// adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_fwd.cu 
// and https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d_update.cu
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "causal_conv1d.h"
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "static_switch.h"



#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)              \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        using weight_t = at::Half;                                                  \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        using weight_t = at::BFloat16;                                              \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        using weight_t = float;                                                     \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }


template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);
template <typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream);

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream);

void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor weight,
                         const at::Tensor out,
                         void* bias_ptr,
                         bool silu_activation) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.weight_ptr = weight.data_ptr();
    params.bias_ptr = bias_ptr;
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.x_c_stride = x.stride(1);
    params.x_l_stride = x.stride(-1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(0);
    params.out_c_stride = out.stride(1);
    params.out_l_stride = out.stride(-1);
}


at::Tensor
causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  const c10::optional<at::Tensor> &seq_idx_,
                  const c10::optional<at::Tensor> &initial_states_,
                  const c10::optional<at::Tensor> &final_states_out_,
                  bool silu_activation) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(x.stride(2) == 1 || x.stride(1) == 1);
    const bool is_channel_last = x.stride(1) == 1 && x.stride(2) > 1;

    if (is_channel_last) {
        TORCH_CHECK(dim % 8 == 0, "causal_conv1d only supports channel dimension divisible by 8 for now");
        TORCH_CHECK(x.stride(2) % 8 == 0 and x.stride(0) % 8 == 0, "causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8");
    }
    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    if (seq_idx_.has_value()) {
        TORCH_CHECK(is_channel_last, "seq_idx is only supported for channel last layout");
        auto seq_idx = seq_idx_.value();
        TORCH_CHECK(seq_idx.scalar_type() == torch::kInt32);
        TORCH_CHECK(seq_idx.is_cuda());
        TORCH_CHECK(seq_idx.is_contiguous());
        CHECK_SHAPE(seq_idx, batch_size, seqlen);
    }

    at::Tensor out = torch::empty_like(x);

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);

    if (seq_idx_.has_value()) {
        params.seq_idx_ptr = seq_idx_.value().data_ptr();
    } else {
        params.seq_idx_ptr = nullptr;
    }

    if (initial_states_.has_value()) {
        TORCH_CHECK(is_channel_last, "initial_states is only supported for channel last layout");
        auto initial_states = initial_states_.value();
        TORCH_CHECK(initial_states.scalar_type() == input_type);
        TORCH_CHECK(initial_states.is_cuda());
        CHECK_SHAPE(initial_states, batch_size, dim, width - 1);
        TORCH_CHECK(initial_states.stride(1) == 1);
        params.initial_states_ptr = initial_states.data_ptr();
        params.initial_states_batch_stride = initial_states.stride(0);
        params.initial_states_c_stride = initial_states.stride(1);
        params.initial_states_l_stride = initial_states.stride(2);
    } else {
        params.initial_states_ptr = nullptr;
    }

    if (final_states_out_.has_value()) {
        TORCH_CHECK(is_channel_last, "final_states is only supported for channel last layout");
        auto final_states = final_states_out_.value();
        TORCH_CHECK(final_states.scalar_type() == input_type);
        TORCH_CHECK(final_states.is_cuda());
        CHECK_SHAPE(final_states, batch_size, dim, width - 1);
        TORCH_CHECK(final_states.stride(1) == 1);
        params.final_states_ptr = final_states.data_ptr();
        params.final_states_batch_stride = final_states.stride(0);
        params.final_states_c_stride = final_states.stride(1);
        params.final_states_l_stride = final_states.stride(2);
    } else {
        params.final_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fwd", [&] {
            if (!is_channel_last) {
                causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
            } else {
                causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
            }
    });
    return out;
}


at::Tensor
causal_conv1d_update(const at::Tensor &x,
                     const at::Tensor &conv_state,
                     const at::Tensor &weight,
                     const c10::optional<at::Tensor> &bias_,
                     bool silu_activation) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == input_type, "weight type must equal to input type, other variations are disabled due to binary size limitations");
    TORCH_CHECK(conv_state.scalar_type() == input_type);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(conv_state.is_cuda());
    TORCH_CHECK(weight.is_cuda());

    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int width = weight.size(-1);

    CHECK_SHAPE(x, batch_size, dim);
    CHECK_SHAPE(conv_state, batch_size, dim, width);
    CHECK_SHAPE(weight, dim, width);

    TORCH_CHECK(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }

    at::Tensor out = torch::empty_like(x);

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, /*seqlen=*/1, width, x, weight, out,
                        bias_.has_value() ? bias_.value().data_ptr() : nullptr,
                        silu_activation);
    params.conv_state_ptr = conv_state.data_ptr();
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_update", [&] {
            causal_conv1d_update_cuda<input_t, weight_t>(params, stream);
    });
    return out;
}

template<int kNThreads_, int kWidth_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_fwd_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static_assert(kWidth <= kNElts);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNElts, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNElts, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, 1, cub::BLOCK_STORE_DIRECT>;
    static constexpr int kSmemIOSize = kIsVecLoad
        ? 0
        : custom_max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockStoreT::TempStorage)});
    static constexpr int kSmemExchangeSize = kNThreads * kNBytes * kNElts;
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize;
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    static constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    extern __shared__ char smem_[];
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_);
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_);
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_ + Ktraits::kSmemIOSize);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
    if (tidx == 0) {
        input_t zeros[kNElts] = {0};
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(zeros)[0];
    }

    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (params.seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), params.seqlen - chunk * kChunkSize);
        }
        x += kChunkSize;
        __syncthreads();
        // Thread kNThreads - 1 don't write yet, so that thread 0 can read
        // the last elements of the previous chunk.
        if (tidx < kNThreads - 1) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }
        __syncthreads();
        reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[tidx > 0 ? tidx - 1 : kNThreads - 1];
        __syncthreads();
        // Now thread kNThreads - 1 can write the last elements of the current chunk.
        if (tidx == kNThreads - 1) { smem_exchange[tidx] = reinterpret_cast<vec_t *>(x_vals_load)[1]; }

        float x_vals[2 * kNElts];
        #pragma unroll
        for (int i = 0; i < 2 * kNElts; ++i) { x_vals[i] = float(x_vals_load[i]); }

        float out_vals[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            out_vals[i] = bias_val;
            #pragma unroll
            for (int w = 0; w < kWidth; ++w) {
                out_vals[i] += weight_vals[w] * x_vals[kNElts + i - (kWidth - w - 1)];
            }
        }

        if (params.silu_activation) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) {
                out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i]));
            }
        }

        input_t out_vals_store[kNElts];
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { out_vals_store[i] = out_vals[i]; }
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(out), reinterpret_cast<vec_t (&)[1]>(out_vals_store), (params.seqlen - chunk * kChunkSize) / kNElts);
        } else {
            typename Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, params.seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;
    }
}


template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    BOOL_SWITCH(params.seqlen % kNElts == 0, kIsVecLoad, [&] {
        using Ktraits = Causal_conv1d_fwd_kernel_traits<kNThreads, kWidth, kIsVecLoad, input_t, weight_t>;
        constexpr int kSmemSize = Ktraits::kSmemSize;
        dim3 grid(params.batch, params.dim);

        auto kernel = &causal_conv1d_fwd_kernel<Ktraits>;

        if (kSmemSize >= 48 * 1024) {
            #ifndef USE_ROCM
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            #else
            // There is a slight signature discrepancy in HIP and CUDA "FuncSetAttribute" function.
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            std::cerr << "Warning (causal_conv1d fwd launch): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
            #endif
        }
        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_fwd_cuda(ConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_fwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_fwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_fwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

template<int kNThreads_, int kWidth_, int kChunkSizeL_, bool kIsVecLoad_, typename input_t_, typename weight_t_>
struct Causal_conv1d_channellast_fwd_kernel_traits {
    // The cache line is 128 bytes, and we try to read 16 bytes per thread.
    // So we have 8 threads per "row", so 32 or 64 elements in the channel dimension.
    // That leaves 4 columns per warp, and so 16 columns per block (assuming each block has 128
    // threads). Each each load is 16 x 32|64 elements in the L x C dimensions.
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static_assert(kNThreads % 32 == 0);
    static constexpr int kNWarps = kNThreads / 32;
    static constexpr int kWidth = kWidth_;
    static constexpr int kChunkSizeL = kChunkSizeL_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    static constexpr int kNEltsPerRow = 128 / kNBytes;
    static constexpr int kNThreadsPerRow = kNEltsPerRow / kNElts;  // Always 8 for now
    static_assert(kNThreadsPerRow * kNBytes * kNElts == 128);
    static constexpr int kNColsPerWarp = 32 / kNThreadsPerRow;  // Always 4 for now
    static_assert(kNColsPerWarp * kNThreadsPerRow == 32);
    static constexpr int kNColsPerLoad = kNColsPerWarp * kNWarps;
    static constexpr int kNLoads = kChunkSizeL / kNColsPerLoad;
    static_assert(kNLoads * kNColsPerLoad == kChunkSizeL);
    static constexpr bool kIsVecLoad = kIsVecLoad_;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    // using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    // using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // static constexpr int kSmemSize = std::max({sizeof(typename BlockLoadT::TempStorage),
    //                                            sizeof(typename BlockStoreT::TempStorage)});
    // static constexpr int kSmemSize = kChunkSizeL * kNEltsPerRow * kNBytes;
};

template<typename Ktraits, bool kHasSeqIdx>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_channellast_fwd_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNThreadsPerC = Ktraits::kNThreadsPerRow;
    constexpr int kLPerLoad = Ktraits::kNColsPerLoad;
    constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
    constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;
    using weight_t = typename Ktraits::weight_t;

    // Shared memory.
    __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL][kChunkSizeC + kNElts];

    const int batch_id = blockIdx.x;
    const int chunk_l_id = blockIdx.y;
    const int chunk_c_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.x_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr)
        + chunk_c_id * kChunkSizeC * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + (chunk_l_id * kChunkSizeL + l_idx) * params.out_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    int *seq_idx = !kHasSeqIdx ? nullptr : reinterpret_cast<int *>(params.seq_idx_ptr)
        + batch_id * params.seqlen + chunk_l_id * kChunkSizeL;
    input_t *initial_states = params.initial_states_ptr == nullptr || chunk_l_id > 0 ? nullptr
        : reinterpret_cast<input_t *>(params.initial_states_ptr) + batch_id * params.initial_states_batch_stride + l_idx * params.initial_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    // The last L-chunk will also have enough info to write to final states, since it also contain a few x values
    // from the previous L-chunk.
    input_t *final_states = params.final_states_ptr == nullptr || chunk_l_id < gridDim.y - 1 ? nullptr
        : reinterpret_cast<input_t *>(params.final_states_ptr) + batch_id * params.final_states_batch_stride + l_idx * params.final_states_l_stride + chunk_c_id * kChunkSizeC + c_idx * kNElts;

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x + l * kLPerLoad * params.x_l_stride);
        }
        reinterpret_cast<vec_t *>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }
    // Load the elements from the previous chunk that are needed for convolution.
    if (l_idx < kWidth - 1) {
        input_t x_vals_load[kNElts] = {0};
        if (chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) >= 0
            && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(x - (kWidth - 1) * params.x_l_stride);
        } else if (initial_states != nullptr
                   && chunk_l_id * kChunkSizeL + l_idx - (kWidth - 1) < 0
                   && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            reinterpret_cast<vec_t *>(x_vals_load)[0] = *reinterpret_cast<vec_t *>(initial_states);
        }
        reinterpret_cast<vec_t *>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t *>(x_vals_load)[0];
    }

    __syncthreads();

    if (final_states != nullptr
        && l_idx < kWidth - 1
        && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
        // x_smem[0] contains element at index chunk_l_id * kChunkSizeL - (kWidth - 1)
        // So last few elements (index params.seqlen - kWidth + 1 + l_idx) are stored in x_smem[params.seqlen - kWidth + 1 + l_idx - (chunk_l_id * kChunkSizeL - kWidth + 1)][c_idx]
        *reinterpret_cast<vec_t *>(final_states) = reinterpret_cast<vec_t *>(x_smem[params.seqlen + l_idx - chunk_l_id * kChunkSizeL])[c_idx];
    }

    constexpr int kLPerThread = constexpr_min(kChunkSizeL * kChunkSizeC / kNThreads, kChunkSizeL);
    static_assert(kLPerThread * kNThreads == kChunkSizeL * kChunkSizeC);
    constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
    static_assert(kNThreadsPerRow * kLPerThread == kChunkSizeL);
    // kChunkSizeL, kLPerThread, kNThreadsPerRow should be powers of 2 for simplicity
    static_assert((kChunkSizeL & (kChunkSizeL - 1)) == 0);
    static_assert((kLPerThread & (kLPerThread - 1)) == 0);
    static_assert((kNThreadsPerRow & (kNThreadsPerRow - 1)) == 0);
    static_assert(kNThreadsPerRow <= 32);

    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;

    float bias_val = params.bias_ptr == nullptr || chunk_c_id * kChunkSizeC + row_idx >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[chunk_c_id * kChunkSizeC + row_idx]);
    float weight_vals[kWidth] = {0};
    if (chunk_c_id * kChunkSizeC + row_idx < params.dim) {
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            weight_vals[w] = weight[row_idx * params.weight_c_stride + w * params.weight_width_stride];
        }
    }
    float x_vals[kWidth - 1 + kLPerThread];
    #pragma unroll
    for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
        x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
    }
    int seq_idx_thread[kWidth - 1 + kLPerThread];
    if constexpr (kHasSeqIdx) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            seq_idx_thread[i] = chunk_l_id * kChunkSizeL + col_idx * kLPerThread + i - (kWidth - 1) >= 0 ? seq_idx[col_idx * kLPerThread + i - (kWidth - 1)] : -1;
        }
    }

    float out_vals[kLPerThread];
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) {
        out_vals[i] = bias_val;
        const int seq_idx_cur = !kHasSeqIdx ? 0 : seq_idx_thread[i + kWidth - 1];
        #pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            if constexpr (!kHasSeqIdx) {
                out_vals[i] += weight_vals[w] * x_vals[i + w];
            } else {
                out_vals[i] += seq_idx_thread[i + w] == seq_idx_cur ? weight_vals[w] * x_vals[i + w] : 0.f;
            }
        }
        if (params.silu_activation) {out_vals[i] = out_vals[i] / (1 + expf(-out_vals[i])); }
    }

    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kLPerThread; ++i) { x_smem[col_idx * kLPerThread + i][row_idx] = out_vals[i]; }
    __syncthreads();

    #pragma unroll
    for (int l = 0; l < Ktraits::kNLoads; ++l) {
        input_t out_vals_store[kNElts];
        reinterpret_cast<vec_t *>(out_vals_store)[0] = reinterpret_cast<vec_t *>(x_smem[l * kLPerLoad + l_idx])[c_idx];
        if (chunk_l_id * kChunkSizeL + l * kLPerLoad + l_idx < params.seqlen
            && chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim) {
            *reinterpret_cast<vec_t *>(out + l * kLPerLoad * params.out_l_stride) = reinterpret_cast<vec_t *>(out_vals_store)[0];
        }
    }

}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seq_idx_ptr != nullptr, kHasSeqIdx, [&] {
        using Ktraits = Causal_conv1d_channellast_fwd_kernel_traits<kNThreads, kWidth, 64, true, input_t, weight_t>;
        // constexpr int kSmemSize = Ktraits::kSmemSize;
        constexpr int kChunkSizeL = Ktraits::kChunkSizeL;
        constexpr int kChunkSizeC = Ktraits::kNEltsPerRow;
        const int n_chunks_L = (params.seqlen + kChunkSizeL - 1) / kChunkSizeL;
        const int n_chunks_C = (params.dim + kChunkSizeC - 1) / kChunkSizeC;
        dim3 grid(params.batch, n_chunks_L, n_chunks_C);
        dim3 block(Ktraits::kNThreads);
        auto kernel = &causal_conv1d_channellast_fwd_kernel<Ktraits, kHasSeqIdx>;
        // if (kSmemSize >= 48 * 1024) {
        //     C10_CUDA_CHECK(cudaFuncSetAttribute(
        //         kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        //     }
        // kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
        kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template<typename input_t, typename weight_t>
void causal_conv1d_channellast_fwd_cuda(ConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_channellast_fwd_launch<128, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_channellast_fwd_launch<128, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_channellast_fwd_launch<128, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);

template void causal_conv1d_channellast_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_channellast_fwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
///////




template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_update_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr) + batch_id * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr || channel_id >= params.dim ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    float weight_vals[kWidth] = {0};
    if (channel_id < params.dim) {
        #pragma unroll
        for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }
    }

    float x_vals[kWidth] = {0};
    if (channel_id < params.dim) {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = float(conv_state[(i + 1) * params.conv_state_l_stride]); }
        x_vals[kWidth - 1] = float(x[0]);
        #pragma unroll
        for (int i = 0; i < kWidth; ++i) { conv_state[i * params.conv_state_l_stride] = input_t(x_vals[i]); }
    }

    float out_val = bias_val;
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { out_val += weight_vals[i] * x_vals[i]; }
    if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
    if (channel_id < params.dim) { out[0] = input_t(out_val); }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBase &params, cudaStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    auto kernel = &causal_conv1d_update_kernel<Ktraits>;
    kernel<<<grid, Ktraits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void causal_conv1d_update_cuda(ConvParamsBase &params, cudaStream_t stream) {
    if (params.width == 2) {
        causal_conv1d_update_launch<64, 2, input_t, weight_t>(params, stream);
    } else if (params.width == 3) {
        causal_conv1d_update_launch<64, 3, input_t, weight_t>(params, stream);
    } else if (params.width == 4) {
        causal_conv1d_update_launch<64, 4, input_t, weight_t>(params, stream);
    }
}

template void causal_conv1d_update_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_update_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_update_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);
