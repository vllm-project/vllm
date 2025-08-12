/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
// clang-format off
// adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.h

#pragma once

#ifndef USE_ROCM
    #include <cuda_bf16.h>
#else
    #include <hip/hip_bf16.h>
#endif
#include <cuda_fp16.h>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SSMParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, dstate, n_groups, n_chunks;
    int dim_ngroups_ratio;
    bool is_variable_B;
    bool is_variable_C;

    bool delta_softplus;

    index_t A_d_stride;
    index_t A_dstate_stride;
    index_t B_batch_stride;
    index_t B_d_stride;
    index_t B_dstate_stride;
    index_t B_group_stride;
    index_t C_batch_stride;
    index_t C_d_stride;
    index_t C_dstate_stride;
    index_t C_group_stride;
    index_t u_batch_stride;
    index_t u_d_stride;
    index_t delta_batch_stride;
    index_t delta_d_stride;
    index_t z_batch_stride;
    index_t z_d_stride;
    index_t out_batch_stride;
    index_t out_d_stride;
    index_t out_z_batch_stride;
    index_t out_z_d_stride;

    // Common data pointers.
    void *__restrict__ A_ptr;
    void *__restrict__ B_ptr;
    void *__restrict__ C_ptr;
    void *__restrict__ D_ptr;
    void *__restrict__ u_ptr;
    void *__restrict__ delta_ptr;
    void *__restrict__ delta_bias_ptr;
    void *__restrict__ out_ptr;
    void *__restrict__ x_ptr;
    void *__restrict__ z_ptr;
    void *__restrict__ out_z_ptr;
    void *__restrict__ index_ptr;
};




#ifndef USE_ROCM

    constexpr size_t custom_max(std::initializer_list<size_t> ilist) 
    {
        return std::max(ilist);
    }

    template<typename T>
    constexpr T constexpr_min(T a, T b) {
        return std::min(a, b);
    }

#else
    constexpr size_t custom_max(std::initializer_list<size_t> ilist) 
    {
        return *std::max_element(ilist.begin(), ilist.end());
    }

    template<typename T>
    constexpr T constexpr_min(T a, T b) {
        return a < b ? a : b;
    }
#endif


#define MAX_DSTATE 256


inline __device__ float2 operator+(const float2 & a, const float2 & b){
    return {a.x + b.x, a.y + b.y};
}

inline __device__ float3 operator+(const float3 &a, const float3 &b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ float4 operator+(const float4 & a, const float4 & b){
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES> struct BytesToType {};

template<> struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<> struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<> struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename scalar_t, int N>
struct Converter{
    static inline __device__ void to_float(const scalar_t (&src)[N], float (&dst)[N]) {
        #pragma unroll
        for (int i = 0; i < N; ++i) { dst[i] = src[i]; }
    }
};

template<int N>
struct Converter<at::Half, N>{
    static inline __device__ void to_float(const at::Half (&src)[N], float (&dst)[N]) {
        static_assert(N % 2 == 0);
        auto &src2 = reinterpret_cast<const half2 (&)[N / 2]>(src);
        auto &dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) { dst2[i] = __half22float2(src2[i]); }
    }
};

#if __CUDA_ARCH__ >= 800
template<int N>
struct Converter<at::BFloat16, N>{
    static inline __device__ void to_float(const at::BFloat16 (&src)[N], float (&dst)[N]) {
        static_assert(N % 2 == 0);
        auto &src2 = reinterpret_cast<const nv_bfloat162 (&)[N / 2]>(src);
        auto &dst2 = reinterpret_cast<float2 (&)[N / 2]>(dst);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i) { dst2[i] = __bfloat1622float2(src2[i]); }
    }
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename scalar_t> struct SSMScanOp;

template<>
struct SSMScanOp<float> {
    __device__ __forceinline__ float2 operator()(const float2 &ab0, const float2 &ab1) const {
        return make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y);
    }
};

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
template <typename scalar_t> struct SSMScanPrefixCallbackOp {
    using scan_t = std::conditional_t<std::is_same_v<scalar_t, float>, float2, float4>;
    scan_t running_prefix;
    // Constructor
    __device__ SSMScanPrefixCallbackOp(scan_t running_prefix_) : running_prefix(running_prefix_) {}
    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ scan_t operator()(scan_t block_aggregate) {
        scan_t old_prefix = running_prefix;
        running_prefix = SSMScanOp<scalar_t>()(running_prefix, block_aggregate);
        return old_prefix;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Ktraits>
inline __device__ void load_input(typename Ktraits::input_t *u,
                                  typename Ktraits::input_t (&u_vals)[Ktraits::kNItems],
                                  typename Ktraits::BlockLoadT::TempStorage &smem_load,
                                  int seqlen) {
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_vec = reinterpret_cast<typename Ktraits::BlockLoadVecT::TempStorage&>(smem_load);
        using vec_t = typename Ktraits::vec_t;
        typename Ktraits::BlockLoadVecT(smem_load_vec).Load(
            reinterpret_cast<vec_t*>(u),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(u_vals)
            #ifdef USE_ROCM
                , Ktraits::kNThreads * Ktraits::kNLoads
            #endif
            
       );
    } else {
        typename Ktraits::BlockLoadT(smem_load).Load(u, u_vals, seqlen, 0.f);
    }
}

template<typename Ktraits>
inline __device__ void load_index(int *u,
                                  int (&u_vals)[Ktraits::kNItems],
                                  typename Ktraits::BlockLoadIndexT::TempStorage &smem_load_index,
                                  int seqlen) {
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_index_vec = reinterpret_cast<typename Ktraits::BlockLoadIndexVecT::TempStorage&>(smem_load_index);
        Ktraits::BlockLoadIndexVecT(smem_load_index_vec).Load(
            reinterpret_cast<uint4*>(u),
            reinterpret_cast<uint4(&)[Ktraits::kNLoadsIndex]>(u_vals)
       );
    } else {
        Ktraits::BlockLoadIndexT(smem_load_index).Load(u, u_vals, seqlen, 0);
    }
}

template<typename Ktraits>
inline __device__ void load_weight(typename Ktraits::input_t *Bvar,
                                   typename Ktraits::weight_t (&B_vals)[Ktraits::kNItems],
                                   typename Ktraits::BlockLoadWeightT::TempStorage &smem_load_weight,
                                   int seqlen) {
    constexpr int kNItems = Ktraits::kNItems;
    typename Ktraits::input_t B_vals_load[kNItems];
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_load_weight_vec = reinterpret_cast<typename Ktraits::BlockLoadWeightVecT::TempStorage&>(smem_load_weight);
        using vec_t = typename Ktraits::vec_t;
        typename Ktraits::BlockLoadWeightVecT(smem_load_weight_vec).Load(
            reinterpret_cast<vec_t*>(Bvar),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(B_vals_load)
      );
    } else {
        typename Ktraits::BlockLoadWeightT(smem_load_weight).Load(Bvar, B_vals_load, seqlen, 0.f);
    }
    // #pragma unroll
    // for (int i = 0; i < kNItems; ++i) { B_vals[i] = B_vals_load[i]; }
    Converter<typename Ktraits::input_t, kNItems>::to_float(B_vals_load, B_vals);
}

template<typename Ktraits>
inline __device__ void store_output(typename Ktraits::input_t *out,
                                    const float (&out_vals)[Ktraits::kNItems],
                                    typename Ktraits::BlockStoreT::TempStorage &smem_store,
                                    int seqlen) {
    typename Ktraits::input_t write_vals[Ktraits::kNItems];
    #pragma unroll
    for (int i = 0; i < Ktraits::kNItems; ++i) { write_vals[i] = out_vals[i]; }
    if constexpr (Ktraits::kIsEvenLen) {
        auto& smem_store_vec = reinterpret_cast<typename Ktraits::BlockStoreVecT::TempStorage&>(smem_store);
        using vec_t = typename Ktraits::vec_t;
        typename Ktraits::BlockStoreVecT(smem_store_vec).Store(
            reinterpret_cast<vec_t*>(out),
            reinterpret_cast<vec_t(&)[Ktraits::kNLoads]>(write_vals)
       );
    } else {
        typename Ktraits::BlockStoreT(smem_store).Store(out, write_vals, seqlen);
    }
}
