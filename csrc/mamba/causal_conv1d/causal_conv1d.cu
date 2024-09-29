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
                         const c10::optional<at::Tensor>& bias,
                         bool silu_activation,
                         const c10::optional<at::Tensor>& query_start_loc = std::nullopt,
                         const c10::optional<at::Tensor>& cache_indices = std::nullopt,
                         const c10::optional<at::Tensor>& has_initial_state = std::nullopt) {

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
    params.bias_ptr = bias.has_value() ? bias.value().data_ptr() : nullptr;
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.query_start_loc_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr() : nullptr;
    params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
    params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr() : nullptr;
    const bool varlen = params.query_start_loc_ptr != nullptr;
    params.x_batch_stride = x.stride(varlen ? 1 : 0);
    params.x_c_stride = x.stride(varlen ? 0 : 1);
    params.x_l_stride = x.stride(varlen ? 1 : -1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.out_batch_stride = out.stride(varlen ? 1 : 0);
    params.out_c_stride = out.stride(varlen ? 0 : 1);
    params.out_l_stride = out.stride(varlen ? 1 : -1);
}


at::Tensor
causal_conv1d_fwd(const at::Tensor &x, const at::Tensor &weight,
                  const c10::optional<at::Tensor> &bias_,
                  const c10::optional<at::Tensor> &conv_states,
                  const c10::optional<at::Tensor> &query_start_loc,
                  const c10::optional<at::Tensor> &cache_indices,
                  const c10::optional<at::Tensor> &has_initial_state,
                  bool silu_activation) {
    auto input_type = x.scalar_type();
    auto weight_type = weight.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float || weight_type == at::ScalarType::Half || weight_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(weight.is_cuda());
    
    const bool varlen = query_start_loc.has_value() ? true : false;
    const auto sizes = x.sizes();
    const int batch_size = varlen ? query_start_loc.value().sizes()[0] - 1 : sizes[0];
    const int dim = varlen ? sizes[0] : sizes[1];
    const int seqlen = varlen ? sizes[1] : sizes[2];
    const int width = weight.size(-1);
    if (varlen){
        CHECK_SHAPE(x, dim, seqlen);
    }
    else {
        CHECK_SHAPE(x, batch_size, dim, seqlen);
    }
    CHECK_SHAPE(weight, dim, width);



    if (bias_.has_value()) {
        auto bias = bias_.value();
        TORCH_CHECK(bias.scalar_type() == weight_type);
        TORCH_CHECK(bias.is_cuda());
        TORCH_CHECK(bias.stride(-1) == 1);
        CHECK_SHAPE(bias, dim);
    }


    if (has_initial_state.has_value()) {
        auto has_initial_state_ = has_initial_state.value();
        TORCH_CHECK(has_initial_state_.scalar_type() == at::ScalarType::Bool);
        TORCH_CHECK(has_initial_state_.is_cuda());
        CHECK_SHAPE(has_initial_state_, batch_size);
    }


    if (query_start_loc.has_value()) {
        auto query_start_loc_ = query_start_loc.value();
        TORCH_CHECK(query_start_loc_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(query_start_loc_.is_cuda());
    }


    if (cache_indices.has_value()) {
        auto cache_indices_ = cache_indices.value();
        TORCH_CHECK(cache_indices_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(cache_indices_.is_cuda());
        CHECK_SHAPE(cache_indices_, batch_size);
    }

    at::Tensor out = torch::empty_like(x);

    ConvParamsBase params;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_,
                        silu_activation, 
                        query_start_loc,
                        cache_indices,
                        has_initial_state
                        );

    if (conv_states.has_value()) {
        auto conv_states_ = conv_states.value();
        TORCH_CHECK(conv_states_.scalar_type() == input_type);
        TORCH_CHECK(conv_states_.is_cuda());
        params.conv_states_ptr = conv_states_.data_ptr();
        params.conv_states_batch_stride = conv_states_.stride(0);
        params.conv_states_c_stride = conv_states_.stride(1);
        params.conv_states_l_stride = conv_states_.stride(2);
    } else {
        params.conv_states_ptr = nullptr;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "causal_conv1d_fwd", [&] {
            causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
    });
    return out;
}


at::Tensor
causal_conv1d_update(const at::Tensor &x,
                     const at::Tensor &conv_state,
                     const at::Tensor &weight,
                     const c10::optional<at::Tensor> &bias_,
                     bool silu_activation,
                     const c10::optional<at::Tensor> &cache_seqlens_,
                     const c10::optional<at::Tensor> &conv_state_indices_) {
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
    const int seqlen = sizes[2];
    const int width = weight.size(-1);
    const int conv_state_len = conv_state.size(2);
    TORCH_CHECK(conv_state_len >= width - 1);

    CHECK_SHAPE(x, batch_size, dim, seqlen);
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
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_,
                        silu_activation);
    params.conv_state_ptr = conv_state.data_ptr();
    params.conv_state_len = conv_state_len;
    // All stride are in elements, not bytes.
    params.conv_state_batch_stride = conv_state.stride(0);
    params.conv_state_c_stride = conv_state.stride(1);
    params.conv_state_l_stride = conv_state.stride(2);

    if (cache_seqlens_.has_value()) {
        auto cache_seqlens = cache_seqlens_.value();
        TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32);
        TORCH_CHECK(cache_seqlens.is_cuda());
        TORCH_CHECK(cache_seqlens.stride(-1) == 1);
        CHECK_SHAPE(cache_seqlens, batch_size);
        params.cache_seqlens = cache_seqlens.data_ptr<int32_t>();
    } else {
        params.cache_seqlens = nullptr;
    }

    if (conv_state_indices_.has_value()) {
        auto conv_state_indices = conv_state_indices_.value();
        TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32)
        TORCH_CHECK(conv_state_indices.is_cuda());
        TORCH_CHECK(conv_state_indices.stride(0) == 1)
        CHECK_SHAPE(conv_state_indices, batch_size);

        int conv_state_entries = conv_state.size(0);
        CHECK_SHAPE(conv_state, conv_state_entries, dim, conv_state_len);

        params.conv_state_indices_ptr = conv_state_indices.data_ptr<int32_t>();
    } else {
        CHECK_SHAPE(conv_state, batch_size, dim, conv_state_len);
        params.conv_state_indices_ptr = nullptr;
    }

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
    constexpr bool kIsVecLoad = Ktraits::kIsVecLoad;
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

    const bool kVarlen = params.query_start_loc_ptr != nullptr;
    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    const int *query_start_loc = kVarlen ? reinterpret_cast<int *>(params.query_start_loc_ptr) : nullptr;
    const int sequence_start_index = kVarlen ? query_start_loc[batch_id] : batch_id;
    const int seqlen = kVarlen ? query_start_loc[batch_id + 1] - sequence_start_index : params.seqlen;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + sequence_start_index * params.x_batch_stride
        + channel_id * params.x_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + sequence_start_index * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    bool has_initial_state = params.has_initial_state_ptr == nullptr ? false
        : reinterpret_cast<bool *>(params.has_initial_state_ptr)[batch_id];

    int* cache_indices = params.cache_indices_ptr == nullptr ? nullptr
        : reinterpret_cast<int *>(params.cache_indices_ptr);
    int cache_index = cache_indices == nullptr ? batch_id : cache_indices[batch_id];

    input_t *conv_states = params.conv_states_ptr == nullptr ? nullptr
        : reinterpret_cast<input_t *>(params.conv_states_ptr) + cache_index * params.conv_states_batch_stride + channel_id * params.conv_states_c_stride;

    // Thread 0 will load the last elements of the previous chunk, so we initialize those to 0.
    if (tidx == 0) {
        input_t initial_state[kNElts] = {0};
        if (has_initial_state) {
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){ initial_state[kNElts - 1 - (kWidth - 2) + w ] = conv_states[w]; }
        }
        smem_exchange[kNThreads - 1] = reinterpret_cast<vec_t *>(initial_state)[0];
    }

    float weight_vals[kWidth];
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    constexpr int kChunkSize = kNThreads * kNElts;
    const int n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t x_vals_load[2 * kNElts] = {0};
        if constexpr(kIsVecLoad) {
            typename Ktraits::BlockLoadVecT(smem_load_vec).Load(reinterpret_cast<vec_t*>(x), *reinterpret_cast<vec_t (*)[1]>(&x_vals_load[kNElts]), (seqlen - chunk * kChunkSize) / kNElts);
        } else {
            __syncthreads();
            typename Ktraits::BlockLoadT(smem_load).Load(x, *reinterpret_cast<input_t (*)[kNElts]>(&x_vals_load[kNElts]), seqlen - chunk * kChunkSize);
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
            typename Ktraits::BlockStoreVecT(smem_store_vec).Store(reinterpret_cast<vec_t*>(out), reinterpret_cast<vec_t (&)[1]>(out_vals_store), (seqlen - chunk * kChunkSize) / kNElts);
        } else {
            typename Ktraits::BlockStoreT(smem_store).Store(out, out_vals_store, seqlen - chunk * kChunkSize);
        }
        out += kChunkSize;
    }
    // Final state is stored in the smem_exchange last token slot,
    // in case seqlen < kWidth, we would need to take the final state from the 
    // initial state which is stored in conv_states
    // in case seqlen > kWidth, we would need to load the last kWidth - 1 data
    // and load it into conv_state accordingly
    int last_thread =  ((seqlen - (kWidth - 1)) - (n_chunks - 1) * kChunkSize) / kNElts;
    if (conv_states != nullptr && tidx == last_thread) { 
        input_t x_vals_load[kNElts * 2] = {0};
        // in case we are on the first kWidth tokens
        if (last_thread == 0 && seqlen < kWidth){
            // Need to take the initial state
            reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[0];
            const int offset = seqlen - (kWidth - 1);
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){
                // pad the existing state
                if ((w - seqlen) >= 0 && has_initial_state) { conv_states[w - seqlen] = conv_states[w]; }
                else if ((w - seqlen) >= 0 && !has_initial_state) { conv_states[w - seqlen] = input_t(0.0f); }
            }
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){
                if (offset + w >= 0) 
                    conv_states[w] = x_vals_load[offset + w ];
            }
        }
        else {
            // in case the final state is in between the threads data
            reinterpret_cast<vec_t *>(x_vals_load)[1] = smem_exchange[last_thread + 1];
            reinterpret_cast<vec_t *>(x_vals_load)[0] = smem_exchange[last_thread];
            const int offset = ((seqlen - (kWidth - 1)) % (kNElts));
            #pragma unroll
            for (int w = 0; w < kWidth - 1; ++w){
                conv_states[w] = x_vals_load[offset + w ];
            }
        }
        
    }
}


template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_fwd_launch(ConvParamsBase &params, cudaStream_t stream) {
    static constexpr int kNElts = sizeof(input_t) == 4 ? 4 : 8;
    const bool kVarlen = params.query_start_loc_ptr != nullptr;
    BOOL_SWITCH(params.seqlen % kNElts == 0 && !kVarlen, kIsVecLoad, [&] {
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


template void causal_conv1d_fwd_cuda<float, float>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::Half, at::Half>(ConvParamsBase &params, cudaStream_t stream);
template void causal_conv1d_fwd_cuda<at::BFloat16, at::BFloat16>(ConvParamsBase &params, cudaStream_t stream);




template<int kNThreads_, int kWidth_, typename input_t_, typename weight_t_>
struct Causal_conv1d_update_kernel_traits {
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kWidth = kWidth_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
};

template<typename Ktraits, bool kIsCircularBuffer>
__global__ __launch_bounds__(Ktraits::kNThreads)
void causal_conv1d_update_kernel(ConvParamsBase params) {
    constexpr int kWidth = Ktraits::kWidth;
    constexpr int kNThreads = Ktraits::kNThreads;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y * kNThreads + tidx;
    if (channel_id >= params.dim) return;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride
        + channel_id * params.x_c_stride;

    // If params.conv_state_batch_indices is set, then the conv state is gathered from the conv state tensor
    // along the batch axis. Otherwise, the conv state coordinate is the same as the batch id.
    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr) 
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;

    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;
    float bias_val = params.bias_ptr == nullptr ? 0.f : float(reinterpret_cast<weight_t *>(params.bias_ptr)[channel_id]);

    int state_len = params.conv_state_len;
    int advance_len = params.seqlen;
    int cache_seqlen = kIsCircularBuffer ? params.cache_seqlens[batch_id] % state_len : 0;
    int update_idx = cache_seqlen - (kWidth - 1);
    update_idx = update_idx < 0 ? update_idx + state_len : update_idx;

    float weight_vals[kWidth] = {0};
    #pragma unroll
    for (int i = 0; i < kWidth; ++i) { weight_vals[i] = float(weight[i * params.weight_width_stride]); }

    float x_vals[kWidth] = {0};
    if constexpr (!kIsCircularBuffer) {
        #pragma unroll 2
        for (int i = 0; i < state_len - advance_len - (kWidth - 1); ++i) {
            conv_state[i * params.conv_state_l_stride] = conv_state[(i + advance_len) * params.conv_state_l_stride];
        }
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) {
            input_t state_val = conv_state[(state_len - (kWidth - 1) + i) * params.conv_state_l_stride];
            if (i < advance_len + (kWidth - 1) && state_len - advance_len - (kWidth - 1) + i >= 0) {
                conv_state[(state_len - advance_len - (kWidth - 1) + i) * params.conv_state_l_stride] = state_val;
            }
            x_vals[i] = float(state_val);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i, update_idx = update_idx + 1 >= state_len ? update_idx + 1 - state_len : update_idx + 1) {
            input_t state_val = conv_state[update_idx * params.conv_state_l_stride];
            x_vals[i] = float(state_val);
        }
    }
    #pragma unroll 2
    for (int i = 0; i < params.seqlen; ++i) {
        input_t x_val = x[i * params.x_l_stride];
        if constexpr (!kIsCircularBuffer) {
            if (i < advance_len && state_len - advance_len + i >= 0) {
                conv_state[(state_len - advance_len + i) * params.conv_state_l_stride] = x_val;
            }
        } else {
            conv_state[update_idx * params.conv_state_l_stride] = x_val;
            ++update_idx;
            update_idx = update_idx >= state_len ? update_idx - state_len : update_idx;
        }
        x_vals[kWidth - 1] = float(x_val);
        float out_val = bias_val;
        #pragma unroll
        for (int j = 0; j < kWidth; ++j) { out_val += weight_vals[j] * x_vals[j]; }
        if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
        out[i * params.out_l_stride] = input_t(out_val);
        // Shift the input buffer by 1
        #pragma unroll
        for (int i = 0; i < kWidth - 1; ++i) { x_vals[i] = x_vals[i + 1]; }
    }
}

template<int kNThreads, int kWidth, typename input_t, typename weight_t>
void causal_conv1d_update_launch(ConvParamsBase &params, cudaStream_t stream) {
    using Ktraits = Causal_conv1d_update_kernel_traits<kNThreads, kWidth, input_t, weight_t>;
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    auto kernel = params.cache_seqlens == nullptr
        ? &causal_conv1d_update_kernel<Ktraits, false>
        : &causal_conv1d_update_kernel<Ktraits, true>;
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
