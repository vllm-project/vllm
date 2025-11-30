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

#include "../static_switch.h"

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

// (APC writeback is implemented in Python wrapper for functional equivalence)
void set_conv_params_fwd(ConvParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t seqlen,
                         const size_t width,
                         // device tensors
                         const at::Tensor& x,
                         const at::Tensor& weight,
                         const at::Tensor& out,
                         // optional pointers (can be nullptr)
                         void* bias_ptr,
                         bool silu_activation,
                         int64_t pad_slot_id,
                         void* query_start_loc_ptr,
                         void* cache_indices_ptr,
                         void* has_initial_state_ptr) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.width = width;
    params.pad_slot_id = pad_slot_id;

    params.silu_activation = silu_activation;

    // Set the pointers and strides.
    params.x_ptr = const_cast<void*>(x.const_data_ptr());
    params.weight_ptr = const_cast<void*>(weight.const_data_ptr());
    params.bias_ptr = bias_ptr;
    params.out_ptr = const_cast<void*>(out.const_data_ptr());
    // All stride are in elements, not bytes.
    params.query_start_loc_ptr = query_start_loc_ptr;
    params.cache_indices_ptr = cache_indices_ptr;
    params.has_initial_state_ptr = has_initial_state_ptr;
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





void causal_conv1d_update(const at::Tensor &x,
                     const at::Tensor &conv_state,
                     const at::Tensor &weight,
                     const std::optional<at::Tensor> &bias_,
                     bool silu_activation,
                     const std::optional<at::Tensor> &cache_seqlens_,
                     const std::optional<at::Tensor> &conv_state_indices_,
                     // used to identify padding entries if cache_indices provided
                     // in case of padding, the kernel will return early
                     int64_t pad_slot_id) {
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

    at::Tensor out = x;

    ConvParamsBase params;
    void* bias_ptr = bias_.has_value() ? bias_.value().data_ptr() : nullptr;
    set_conv_params_fwd(params, batch_size, dim, seqlen, width, x, weight, out,
                        bias_ptr,
                        silu_activation,
                        pad_slot_id,
                        /*query_start_loc_ptr*/ nullptr,
                        /*cache_indices_ptr*/ nullptr,
                        /*has_initial_state_ptr*/ nullptr);
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
        TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32);
        TORCH_CHECK(conv_state_indices.is_cuda());
        TORCH_CHECK(conv_state_indices.stride(0) == 1);
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
}


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
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
        + channel_id * params.out_c_stride;

    // If params.conv_state_batch_indices is set, then the conv state is gathered from the conv state tensor
    // along the batch axis. Otherwise, the conv state coordinate is the same as the batch id.
    const int conv_state_batch_coord = params.conv_state_indices_ptr == nullptr
        ? batch_id
        : params.conv_state_indices_ptr[batch_id];

    // Skip padding tokens when the selected coord equals pad_slot_id.
    if (conv_state_batch_coord == params.pad_slot_id) {
        #pragma unroll 2
        for (int i = 0; i < params.seqlen; ++i) {
            out[i * params.out_l_stride] = input_t(0.f);
        }
        return;
    }
    input_t *conv_state = reinterpret_cast<input_t *>(params.conv_state_ptr)
        + conv_state_batch_coord * params.conv_state_batch_stride
        + channel_id * params.conv_state_c_stride;
    weight_t *weight = reinterpret_cast<weight_t *>(params.weight_ptr) + channel_id * params.weight_c_stride;
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