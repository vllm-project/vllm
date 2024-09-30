/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/
// clang-format off
// adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/csrc/causal_conv1d.h
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct ConvParamsBase {
    using index_t = uint32_t;

    int batch, dim, seqlen, width;
    bool silu_activation;

    index_t x_batch_stride;
    index_t x_c_stride;
    index_t x_l_stride;
    index_t weight_c_stride;
    index_t weight_width_stride;
    index_t out_batch_stride;
    index_t out_c_stride;
    index_t out_l_stride;

    int conv_state_len;
    index_t conv_state_batch_stride;
    index_t conv_state_c_stride;
    index_t conv_state_l_stride;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ weight_ptr;
    void *__restrict__ bias_ptr;
    void *__restrict__ out_ptr;

    void *__restrict__ conv_state_ptr;
    void *__restrict__ query_start_loc_ptr;
    void *__restrict__ has_initial_state_ptr;
    void *__restrict__ cache_indices_ptr;
    int32_t *__restrict__ cache_seqlens;

    // For the continuous batching case. Makes it so that the mamba state for 
    // the current batch doesn't need to be a contiguous tensor.
    int32_t *__restrict__ conv_state_indices_ptr;

    void *__restrict__ seq_idx_ptr;

    // No __restrict__ since initial_states could be the same as final_states.
    void * initial_states_ptr;
    index_t initial_states_batch_stride;
    index_t initial_states_l_stride;
    index_t initial_states_c_stride;

    void * final_states_ptr;
    index_t final_states_batch_stride;
    index_t final_states_l_stride;
    index_t final_states_c_stride;

    void *  conv_states_ptr;
    index_t conv_states_batch_stride;
    index_t conv_states_l_stride;
    index_t conv_states_c_stride;
};


#ifndef USE_ROCM
    #include <cuda_bf16.h>

    template<typename T>
    __device__ inline T shuffle_xor(T val, int offset) {
        return __shfl_xor_sync(uint32_t(-1), val, offset);
    }

    constexpr size_t custom_max(std::initializer_list<size_t> ilist) 
    {
        return std::max(ilist);
    }

    template<typename T>
    constexpr T constexpr_min(T a, T b) {
        return std::min(a, b);
    }

#else
    #include <hip/hip_bf16.h>

    template<typename T>
    __device__ inline T shuffle_xor(T val, int offset) {
        return __shfl_xor(val, offset);
    }
    constexpr size_t custom_max(std::initializer_list<size_t> ilist) 
    {
        return *std::max_element(ilist.begin(), ilist.end());
    }

    template<typename T>
    constexpr T constexpr_min(T a, T b) {
        return a < b ? a : b;
    }
#endif

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

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};
