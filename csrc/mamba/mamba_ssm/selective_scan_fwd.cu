// clang-format off
// adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "selective_scan.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, bool kUseIndex_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;
    static constexpr bool kUseIndex = kUseIndex_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;
    static constexpr int kNLoadsIndex = kNItems / 4;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadIndexT = cub::BlockLoad<int, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadIndexVecT = cub::BlockLoad<uint4, kNThreads, kNLoadsIndex,
        !(kIsEvenLen && kNLoadsIndex == 1) ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems , cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads ,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 sizeof(typename BlockLoadIndexT::TempStorage),
                                                 sizeof(typename BlockLoadIndexVecT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr bool kUseIndex = Ktraits::kUseIndex;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_index = reinterpret_cast<typename Ktraits::BlockLoadIndexT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;
    int *index = !kUseIndex ? nullptr :reinterpret_cast<int *>(params.index_ptr) + batch_id * params.seqlen;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }


    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
    // }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        int index_vals_load[kNRows][kNItems];

        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (kUseIndex) {
                load_index<Ktraits>(index + r * params.delta_d_stride, index_vals_load[r], smem_load_index, params.seqlen - chunk * kChunkSize);
            }
        }
        if constexpr (kUseIndex) {
            index += kChunkSize;
        }
        u += kChunkSize;
        delta += kChunkSize;
    
        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                A_val[r] *= kLog2e;
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
                        if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (1));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (1 ));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                                                 !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                    
                    // Reset A bar for cumulative sequences (Real)
                    if constexpr (kUseIndex) {
                        if (index_vals_load[r][i] == 0) {
                            thread_data[i].x = 0.f;
                        }
                    }

                    if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                        if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                    // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                running_prefix = chunk == 0 ? x[(r * params.n_chunks) * params.dstate + state_idx] : ( threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f));
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    out_vals[r][i] += thread_data[i].y * C_val;
                }
            }
        }
        
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * 1;
        Cvar += kChunkSize * 1;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    // kIsVariableB, kIsVariableC and kHasZ are all set to True to reduce binary size
    constexpr bool kIsVariableB = true;
    constexpr bool kIsVariableC = true;
    constexpr bool kHasZ = true;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.index_ptr != nullptr , kUseIndex, [&] {
            using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ,  kUseIndex, input_t, weight_t>;
            constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
            dim3 grid(params.batch, params.dim / kNRows);
            auto kernel = &selective_scan_fwd_kernel<Ktraits>;
            if (kSmemSize >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
            }
            kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {           
            selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}

template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)              \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        using weight_t = float;                                                     \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        using weight_t = float;                                                     \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        using weight_t = float;                                                     \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }


template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const size_t n_chunks,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const torch::Tensor u,
                        const torch::Tensor delta,
                        const torch::Tensor A,
                        const torch::Tensor B,
                        const torch::Tensor C,
                        const torch::Tensor out,
                        const torch::Tensor z,
                        const torch::Tensor out_z,
                        void* D_ptr,
                        void* delta_bias_ptr,
                        void* x_ptr,
                        bool has_z, 
                        bool delta_softplus,
                        void* index_ptr) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;

    params.index_ptr = index_ptr;

    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        params.out_z_batch_stride = out_z.stride(0);
        params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

std::vector<torch::Tensor>
selective_scan_fwd(const torch::Tensor &u, const torch::Tensor &delta,
                  const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C,
                  const c10::optional<torch::Tensor> &D_,
                  const c10::optional<torch::Tensor> &z_,
                  const c10::optional<torch::Tensor> &delta_bias_,
                  bool delta_softplus,
                  const c10::optional<torch::Tensor> &index_,
                  const c10::optional<torch::Tensor> &x) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    TORCH_CHECK(is_variable_B, "is_variable_B = False is disabled in favor of reduced binary size")
    CHECK_SHAPE(B, batch_size, n_groups, dstate, seqlen );
    TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);

    TORCH_CHECK(is_variable_C, "is_variable_C = False is disabled in favor of reduced binary size")
    CHECK_SHAPE(C, batch_size, n_groups, dstate, seqlen);
    TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }
    if (index_.has_value()) {
        auto index = index_.value();
        TORCH_CHECK(index.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(index.is_cuda());
        CHECK_SHAPE(index, batch_size, seqlen);
    }

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    TORCH_CHECK(has_z, "has_z = False is disabled in favor of reduced binary size")
    z = z_.value();
    TORCH_CHECK(z.scalar_type() == input_type);
    TORCH_CHECK(z.is_cuda());
    TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
    CHECK_SHAPE(z, batch_size, dim, seqlen);
    out_z = torch::empty_like(z);

    const int n_chunks = (seqlen + 2048 - 1) / 2048;
    // const int n_chunks = (seqlen + 1024 - 1) / 1024;
    // at::Tensor out = torch::empty_like(u);
    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    at::Tensor out = torch::empty_like(delta);
    if (x.has_value()){
        auto _x = x.value();
        TORCH_CHECK(_x.scalar_type() == weight_type);
        TORCH_CHECK(_x.is_cuda());
        TORCH_CHECK(_x.stride(-1) == 1);
        CHECK_SHAPE(_x, batch_size, dim, n_chunks, dstate * 2);
    }

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_.has_value() ? D_.value().data_ptr() : nullptr,
                       delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr,
                       x.value().data_ptr(),
                       has_z,
                       delta_softplus,
                       index_.has_value() ? index_.value().data_ptr() : nullptr);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_fwd", [&] {
        selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
    });
    std::vector<at::Tensor> result = {out, x.value()};
    if (has_z) { result.push_back(out_z); }
    return result;
}

