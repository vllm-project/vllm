// clang-format off
// Adapted from: https://github.com/meta-pytorch/applied-ai/blob/main/kernels/cuda/inference/hadamard_transform/hadamard_transform_cuda.cu

/***********
Copyright 2024 Meta

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
***********/

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>

#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/annotated_ptr>

#include "../../../torch_utils.h"
#include "../../../dispatch_utils.h"

namespace hadacore {

#ifndef __CUDACC__
#define __launch_bounds__(x,y)
#endif

#define MAX_WARPS_PER_SM 48

#define MIN(a, b) ((a) < (b) ? (a) : (b))

using b16 = uint16_t;
using b32 = uint32_t;

constexpr int launch_configs_big[7][3] = {
    // default
    {2, 1, 24},
    {2, 2, 16},
    {2, 4, 8},
    {2, 8, 4},
    {2, 16, 3},
    {4, 16, 2},
    {8, 16, 1}
    // // extra coalescing
    // {2, 1, 24},
    // {2, 2, 16},
    // {2, 4, 8},
    // {2, 8, 4},
    // {4, 8, 3},
    // {8, 8, 2},
    // {16, 8, 1}
    // // less coalescing
    // {2, 1, 24},
    // {2, 2, 16},
    // {2, 4, 8},
    // {2, 8, 4},
    // {1, 32, 1},
    // {2, 32, 1},
    // {4, 32, 1}
};

// a 4x2, b 2x2, c 2x2
template <torch::headeronly::ScalarType dtype>
__device__ __forceinline__ void mma_m16_n8_k16_b16_b16_b16_noacc(b32 a0, b32 a1, b32 a2, b32 a3, b32 b0, b32 b1, b32& c0, b32& c1){
    static_assert(dtype == torch::headeronly::ScalarType::Half || dtype == torch::headeronly::ScalarType::BFloat16);
    // d, a, b, c
    b32 zero = 0;
    if constexpr(dtype == torch::headeronly::ScalarType::Half) {
        asm (
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n\t"
            : "=r"(c0), "=r"(c1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero), "r"(zero)
        );
    } else {
        b32 temp0, temp1, temp2, temp3;
        asm (
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n\t"
            : "=r"(temp0), "=r"(temp1), "=r"(temp2), "=r"(temp3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(zero), "r"(zero), "r"(zero), "r"(zero)
        );
        asm ("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(temp1), "r"(temp0));
        asm ("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(temp3), "r"(temp2));
    }
}

// a 4x2, b 4x2, c 4x2
template <torch::headeronly::ScalarType dtype>
__device__ __forceinline__ void mma_m16_n16_k16_b16_b16_b16_noacc(b32 a0, b32 a1, b32 a2, b32 a3, b32 b0, b32 b1, b32 b2, b32 b3, b32& c0, b32& c1, b32& c2, b32& c3){
    mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a0, a1, a2, a3, b0, b1, c0, c1);
    mma_m16_n8_k16_b16_b16_b16_noacc<dtype>(a0, a1, a2, a3, b2, b3, c2, c3);
}

__device__ __forceinline__ void matrix_transpose_m8_n8_b16_inplace(b32& a0) {
    asm (
        "movmatrix.sync.aligned.m8n8.trans.b16 "
        "%0, %1;\n\t"
        : "=r"(a0) : "r"(a0)
    );
}

#define p_p(i) ((val_1p[i] & 0x0000FFFF) | val_1p[i] << 16)
#define p_n(i) ((val_1p[i] & 0x0000FFFF) | val_1n[i] << 16)
#define n_p(i) ((val_1n[i] & 0x0000FFFF) | val_1p[i] << 16)
#define n_n(i) ((val_1n[i] & 0x0000FFFF) | val_1n[i] << 16)

template<int64_t num_chunks, int64_t warps_per_block, int64_t log_had_size, int64_t blocks_per_sm, bool enable_mask, torch::headeronly::ScalarType dtype>
__global__ void __launch_bounds__(32 * warps_per_block, blocks_per_sm)
// a is column major, b is row major
hadamard_transform_kernel(b16* a, b16* out, int total_num_chunks) {
    static_assert(dtype == torch::headeronly::ScalarType::Half || dtype == torch::headeronly::ScalarType::BFloat16, "Only fp16 and bf16 supported currently");

    b32 b_frag_all[num_chunks][4]; // for all chunks, holds matrix fragment (which takes 4 regs of b16x2 * 32 threads)

    int64_t blockid = blockIdx.x * warps_per_block + threadIdx.x / 32;
    int64_t threadid = threadIdx.x % 32;
    extern __shared__ b32 bfrag_arr[]; // num_chunks * warps_per_block * 128
    int64_t real_num_chunks = ((blockid + 1) * num_chunks) > total_num_chunks ? (total_num_chunks - (blockid * num_chunks)) : num_chunks;
    int64_t diff_num_chunks = real_num_chunks - num_chunks;

    b32* a_start_ptr = (b32*) (a + blockid * num_chunks * 256); // offset a to where this warp starts
    b32* out_start_ptr = (b32*) (out + blockid * num_chunks * 256);
    b32* a_ptr = a_start_ptr + threadid * 4;
    b32* b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128 + threadid * 4;

    #if (__CUDA_ARCH__ < 900) // SM80, SM89
    uint64_t cache_policy;
    asm volatile(
        "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n"
        : "=l"(cache_policy)
    );
    #endif

    #pragma unroll
    for (int64_t k = 0; k < num_chunks; k++) {
        size_t shared_ptr = __cvta_generic_to_shared(b_frag_ptr);
        #if (__CUDA_ARCH__ >= 900) // SM90
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], 16;\n"
                "cp.async.commit_group;\n"
                :: "l"(shared_ptr), "l"(a_ptr)
            );
        #else // SM80, SM89
            asm volatile(
                "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2;\n"
                "cp.async.commit_group;\n"
                :: "l"(shared_ptr), "l"(a_ptr), "l"(cache_policy)
            );
        #endif

        a_ptr += 128;
        b_frag_ptr += 128;
    }

    // generate hadamard 16x16 (up to 2 of them)
    constexpr b16 fp16_1p[4] = {0b0011100110101000, 0b0011100000000000, 0b0011010110101000, 0b0011010000000000};
    constexpr b16 fp16_1n[4] = {0b1011100110101000, 0b1011100000000000, 0b1011010110101000, 0b1011010000000000};
    constexpr b16 bf16_1p[4] = {0b0011111100110101, 0b0011111100000000, 0b0011111010110101, 0b0011111010000000};
    constexpr b16 bf16_1n[4] = {0b1011111100110101, 0b1011111100000000, 0b1011111010110101, 0b1011111010000000};

    #define val_type_1p(i) (((dtype) == torch::headeronly::ScalarType::Half) ? (fp16_1p[i]) : (bf16_1p[i]))
    #define val_type_1n(i) (((dtype) == torch::headeronly::ScalarType::Half) ? (fp16_1n[i]) : (bf16_1n[i]))
    constexpr b16 val_1p[4] = {val_type_1p(0), val_type_1p(1), val_type_1p(2), val_type_1p(3)};
    constexpr b16 val_1n[4] = {val_type_1n(0), val_type_1n(1), val_type_1n(2), val_type_1n(3)};

    constexpr b32 p_p[4] = {p_p(0), p_p(1), p_p(2), p_p(3)};
    constexpr b32 p_n[4] = {p_n(0), p_n(1), p_n(2), p_n(3)};
    constexpr b32 n_p[4] = {n_p(0), n_p(1), n_p(2), n_p(3)};
    constexpr b32 n_n[4] = {n_n(0), n_n(1), n_n(2), n_n(3)};
    const b32 had_16_p1[4][4] = {
        {
            0b10001000010001000010001000010001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10001000010001000010001000010001
        },
        {
            0b11001100100010000011001100100010,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11001100100010000011001100100010
        },
        {
            0b11111111101010101100110010011001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11111111101010101100110010011001
        },
        {
            0b11111111101010101100110010011001,
            0b11111111101010101100110010011001,
            0b11111111101010101100110010011001,
            0b00000000010101010011001101100110
        }
    };
    const b32 had_16_p2[4][4] = {
        {
            0b10000000010000000010000000010000,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10000000010000000010000000010000
        },
        {
            0b11000000100001000011000000100001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11000000100001000011000000100001
        },
        {
            0b11110000101001011100001110010110,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11110000101001011100001110010110
        },
        {
            0b11110000101001011100001110010110,
            0b11110000101001011100001110010110,
            0b11110000101001011100001110010110,
            0b00001111010110100011110001101001
        }
    };
    const b32 had_16_mask[3][4] = {
        {
            0b10001000010001000010001000010001,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b10001000010001000010001000010001
        },
        {
            0b11001100110011000011001100110011,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11001100110011000011001100110011
        },
        {
            0b11111111111111111111111111111111,
            0b00000000000000000000000000000000,
            0b00000000000000000000000000000000,
            0b11111111111111111111111111111111
        }
    };
    b32 had_frag[8];
    #pragma unroll
    for (int64_t i = 0; i < 2; i++) {
        int64_t c_log_h = (i == 0) ? MIN(4, log_had_size) : log_had_size % 4;
        #pragma unroll
        for (int64_t j = 0; j < 4; j++) {
            if (c_log_h < 4) {
                bool mask = had_16_mask[c_log_h - 1][j] & (1 << (31 - threadid));
                if (!mask) {
                    had_frag[i * 4 + j] = 0;
                    continue;
                }
            }
            bool pred1 = had_16_p1[c_log_h - 1][j] & (1 << (31 - threadid));
            bool pred2 = had_16_p2[c_log_h - 1][j] & (1 << (31 - threadid));
            b32 val = pred1 ? (pred2 ? p_p[c_log_h - 1] : p_n[c_log_h - 1]) : (pred2 ? n_p[c_log_h - 1] : n_n[c_log_h - 1]);
            had_frag[i * 4 + j] = val;
        }
        if constexpr(log_had_size <= 4 || log_had_size % 4 == 0) break;
    }

    // log had size above 8, only used for above 2^8 = 256 size
    constexpr int64_t part8_log_had_size = log_had_size - 8;

    b32* a_chunk_ptr = a_start_ptr; // first chunk starts at this warp's data starts
    b32* out_chunk_ptr = out_start_ptr;

    #pragma unroll
    for (int64_t l = 0; l < 2; l++) {
        if constexpr(log_had_size <= 8) { // l == 0 guaranteed, redundant simplified version of else body, to help compiler warnings
            b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * 128;
        } else {
            b_frag_ptr = bfrag_arr + (blockid % warps_per_block) * num_chunks * (l == 0 ? 128 : (128 >> part8_log_had_size));
        }

        if (l == 1) {
            if constexpr(log_had_size > 8) {
                __syncthreads(); // sync between first and second iterations if above size 256

                if constexpr(log_had_size >= 12) {
                    // sizes 4k and above

                    // a + threadblock offset + warp offset
                    // can then index into all chunks owned by this warp
                    b32* store = bfrag_arr + (128 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block));

                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int64_t k = 0; k < num_chunks; k++) {
                            // here, j represents register, and k represents 8-offset/chunk
                            uint64_t real_chunk_num = (num_chunks - (threadid % num_chunks) + k) % num_chunks; // chunk at which you have target thread #'s data

                            int64_t real_thread_id = (threadid / num_chunks) * num_chunks + k; // target thread #
                            int64_t chunk_idx = 128 * real_chunk_num; // index due to fetching from another chunk (chunk in which this thread has the target thread's original data)
                            int64_t thread_group_idx = (real_thread_id / 4) * 16; // index due to fetching from another group of num_chunk threads (since shuffle is between num_chunk threads)
                            int64_t thread_idx = (real_thread_id % 4) * 2; // index due to original thread's position within the group of num_chunk threads
                            int64_t reg_idx = (j / 2) * 8 + (j % 2); // index due to target register
                            int64_t idx = chunk_idx + thread_group_idx + thread_idx + reg_idx; // final index

                            // fix idx for majorness
                            int64_t rowidx = idx % (1 << part8_log_had_size);
                            int64_t colidx = idx >> part8_log_had_size;

                            // store[rowidx * 128 + colidx] = data;
                            b32 data = store[rowidx * 128 + colidx];

                            // compiler generates excessive instructions, so we manually do the if statement
                            #pragma unroll
                            for (uint64_t i = 0; i < num_chunks; i++) {
                                asm volatile (
                                    "{\n\t"
                                    "  .reg .pred p0;\n\t"
                                    "  setp.eq.s64 p0, %1, %2;\n\t"
                                    "  @p0 mov.b32 %0, %3;\n\t"
                                    "}\n\t"
                                    : "+r"(b_frag_all[i][j]) // Output operand %0
                                    : "l"(real_chunk_num), "l"(i), "r"(data) // Input operands %1, %2, %3
                                );
                            }
                        }
                    }

                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        #pragma unroll
                        for (int64_t k = 1; k < num_chunks; k++) {
                            int64_t threadid_contig = threadid % num_chunks;
                            int64_t threadid_mul = threadid / num_chunks;
                            int64_t threadid2 = (threadid_contig + num_chunks - k) % num_chunks + threadid_mul * num_chunks; // thread to give your data to
                            b_frag_all[k][j] = __shfl_sync(0xFFFFFFFF, b_frag_all[k][j], threadid2);
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int64_t k = 0; k < num_chunks; k++) {
            if constexpr(enable_mask) {
                if (k >= real_num_chunks)
                    break;
            }
            if (l == 0) {
                // bad fix for k not being recognized as a constexpr by compiler
                // asm("cp.async.wait_group %0;\n" :: "n"(num_chunks - k - 1));
                #define SWITCH_WAIT_ASYNC_LOAD_GROUP(i) case i: asm volatile("cp.async.wait_group %0;\n" :: "n"(num_chunks - i - 1)); break;
                if constexpr(enable_mask) {
                    switch(k + diff_num_chunks) {
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(0)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(1)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(2)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(3)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(4)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(5)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(6)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(7)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(8)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(9)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(10)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(11)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(12)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(13)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(14)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(15)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(16)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(17)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(18)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(19)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(20)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(21)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(22)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(23)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(24)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(25)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(26)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(27)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(28)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(29)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(30)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(31)
                    }
                } else {
                    switch(k) {
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(0)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(1)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(2)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(3)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(4)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(5)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(6)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(7)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(8)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(9)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(10)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(11)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(12)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(13)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(14)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(15)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(16)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(17)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(18)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(19)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(20)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(21)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(22)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(23)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(24)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(25)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(26)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(27)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(28)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(29)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(30)
                        SWITCH_WAIT_ASYNC_LOAD_GROUP(31)
                    }
                }
            }

            if (l == 0) {
                // loading for the first iteration

                // thread 0 loads  [t0r0, t16r1, t0r2, t16r3]
                // thread 16 loads [t0r1, t16r0, t0r3, t16r2]
                // allows full coalescing, same for t1/t17, t2/t18, etc.
                #pragma unroll
                for (int64_t j = 0; j < 4; j++) {
                    int64_t reg = ((threadid & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
                    int64_t real_thread_id = (reg == 0 || reg == 2) ? threadid : (threadid ^ 16);
                    int64_t real_row = real_thread_id % 4;
                    int64_t real_col = real_thread_id / 4;
                    b_frag_all[k][j] = b_frag_ptr[(real_row + (reg % 2) * 4) + (real_col + (j / 2) * 8) * 8];
                }

                // for t16 swap r0/r1 and r2/r3 to have [t16r0, t0r1, t16r2, t0r3]
                // so registers are in right order, same for t17, t18, etc.
                if ((threadid & 16) != 0) {
                    b32 temp = b_frag_all[k][0];
                    b_frag_all[k][0] = b_frag_all[k][1];
                    b_frag_all[k][1] = temp;

                    temp = b_frag_all[k][2];
                    b_frag_all[k][2] = b_frag_all[k][3];
                    b_frag_all[k][3] = temp;
                }

                // t0 and t16 swap r1 and r3 to have their own data,
                // same for t1/t17, t2/18, etc.
                #pragma unroll
                for (int64_t j = 1; j < 4; j += 2) {
                    b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], 16);
                }
            } else if constexpr(log_had_size > 8) { // condition is redundant to help compiler warnings
                if constexpr(log_had_size < 12) {
                    // sizes 512, 1k, and 2k

                    // for 512:
                    //     thread 0 loads  [t0r0, t0r1, t16r2, t16r3]
                    //     thread 16 loads [t0r2, t0r3, t16r0, t16r1]
                    //     same for t1/t17, t2/t18, etc.
                    // for 1k and 2k:
                    //     thread 0 loads [t0r0, t0r1, t1r2, t1r3]
                    //     thread 1 loads [t0r2, t0r3, t1r0, t1r1]
                    //     same for t2/t3, t4/t5, etc.
                    // allows full coalescing for 512 and 1k, 16x coalescing for 2k
                    constexpr int64_t xor_val = log_had_size == 9 ? 16 : 1;

                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        int64_t reg = ((threadid & xor_val) == 0) ? j : (j + 2) % 4;
                        int64_t real_thread_id = reg < 2 ? threadid : (threadid ^ xor_val);
                        int64_t idx = (real_thread_id / 4 * 16) + (real_thread_id % 4 * 2) + (reg / 2 * 8) + (reg % 2);
                        int64_t rowidx = idx % (1 << part8_log_had_size);
                        int64_t colidx = idx >> part8_log_had_size;
                        b_frag_all[k][j] = b_frag_ptr[rowidx * 128 + colidx];
                    }

                    if ((threadid & xor_val) != 0) {
                        b32 temp = b_frag_all[k][0];
                        b_frag_all[k][0] = b_frag_all[k][2];
                        b_frag_all[k][2] = temp;

                        temp = b_frag_all[k][1];
                        b_frag_all[k][1] = b_frag_all[k][3];
                        b_frag_all[k][3] = temp;
                    }

                    #pragma unroll
                    for (int64_t j = 2; j < 4; j++) {
                        b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], xor_val);
                    }
                }
            }

            if (l == 1) {
                // for second iteration, we load 2 consecutive b16s (1 b32) per register,
                // but tensor core register layout requires 2 b16s that are in the
                // same column/consecutive rows to be in the same register, so do the swap
                b32 f0 = ((b_frag_all[k][1] & 0xFFFF) << 16) | (b_frag_all[k][0] & 0xFFFF);
                b32 f1 = ((b_frag_all[k][3] & 0xFFFF) << 16) | (b_frag_all[k][2] & 0xFFFF);
                b32 f2 = (b_frag_all[k][1] & 0xFFFF0000) | (b_frag_all[k][0] >> 16);
                b32 f3 = (b_frag_all[k][3] & 0xFFFF0000) | (b_frag_all[k][2] >> 16);
                b_frag_all[k][0] = f0;
                b_frag_all[k][1] = f1;
                b_frag_all[k][2] = f2;
                b_frag_all[k][3] = f3;
            }

            #pragma unroll
            for(int64_t i = 0, remaining_log_had_size = log_had_size - l * 8; i < 2 && remaining_log_had_size > 0; i++) {
                int64_t had_off = ((remaining_log_had_size < 4) && !(log_had_size <= 4 || log_had_size % 4 == 0)) ? 4 : 0;
                mma_m16_n16_k16_b16_b16_b16_noacc<dtype>(had_frag[had_off + 0], had_frag[had_off + 1], had_frag[had_off + 2], had_frag[had_off + 3], b_frag_all[k][0], b_frag_all[k][1], b_frag_all[k][2], b_frag_all[k][3], b_frag_all[k][0], b_frag_all[k][1], b_frag_all[k][2], b_frag_all[k][3]);

                remaining_log_had_size -= 4;
                if (remaining_log_had_size <= 0 && i == 0) {
                    // TODO: consider different storing so no need for transpose
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][0]);
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][1]);
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][2]);
                    matrix_transpose_m8_n8_b16_inplace(b_frag_all[k][3]);
                } else {
                    // swap and use output directly as b_frag for next iteration as an actually free transpose
                    b32 temp = b_frag_all[k][1];
                    b_frag_all[k][1] = b_frag_all[k][2];
                    b_frag_all[k][2] = temp;
                }
            }

            if (l == 1) {
                // invert swap from above for second iteration
                b32 f0 = ((b_frag_all[k][2] & 0xFFFF) << 16) | (b_frag_all[k][0] & 0xFFFF);
                b32 f1 = (b_frag_all[k][2] & 0xFFFF0000) | (b_frag_all[k][0] >> 16);
                b32 f2 = ((b_frag_all[k][3] & 0xFFFF) << 16) | (b_frag_all[k][1] & 0xFFFF);
                b32 f3 = (b_frag_all[k][3] & 0xFFFF0000) | (b_frag_all[k][1] >> 16);
                b_frag_all[k][0] = f0;
                b_frag_all[k][1] = f1;
                b_frag_all[k][2] = f2;
                b_frag_all[k][3] = f3;
            }

            if (l == 0) {
                // inverse of coalesced load for first iteration to store result
                #pragma unroll
                for (int64_t j = 1; j < 4; j += 2) {
                    b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], 16);
                }

                if ((threadid & 16) != 0) {
                    b32 temp = b_frag_all[k][0];
                    b_frag_all[k][0] = b_frag_all[k][1];
                    b_frag_all[k][1] = temp;

                    temp = b_frag_all[k][2];
                    b_frag_all[k][2] = b_frag_all[k][3];
                    b_frag_all[k][3] = temp;
                }

                // if only going up to 256 size, store directly back to global memory,
                // otherwise store back to shared memory for next iteration
                b32* store = (log_had_size <= 8) ? out_chunk_ptr : b_frag_ptr;

                #pragma unroll
                for (int64_t j = 0; j < 4; j++) {
                    int64_t reg = ((threadid & 16) == 0) ? j : (j / 2 * 2 + (1 - j % 2));
                    int64_t real_thread_id = (reg == 0 || reg == 2) ? threadid : (threadid ^ 16);
                    int64_t real_row = real_thread_id % 4;
                    int64_t real_col = real_thread_id / 4;
                    store[(real_row + (reg % 2) * 4) + (real_col + (reg / 2) * 8) * 8] = b_frag_all[k][j];
                }
            } else if constexpr(log_had_size > 8) { // condition is redundant to help compiler warnings
                if (log_had_size < 12) {
                    // inverse of coalesced load for sizes 512, 1k and 2k to store result
                    constexpr int xor_val = log_had_size == 9 ? 16 : 1;
                    #pragma unroll
                    for (int64_t j = 2; j < 4; j++) {
                        b_frag_all[k][j] = __shfl_xor_sync(0xFFFFFFFF, b_frag_all[k][j], xor_val);
                    }

                    if ((threadid & xor_val) != 0) {
                        b32 temp = b_frag_all[k][0];
                        b_frag_all[k][0] = b_frag_all[k][2];
                        b_frag_all[k][2] = temp;

                        temp = b_frag_all[k][1];
                        b_frag_all[k][1] = b_frag_all[k][3];
                        b_frag_all[k][3] = temp;
                    }

                    b32* store = (b32*)(out + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 256 + (256 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block) + k));
                    #pragma unroll
                    for (int64_t j = 0; j < 4; j++) {
                        int64_t reg = ((threadid & xor_val) == 0) ? j : (j + 2) % 4;
                        b32 data = b_frag_all[k][j];
                        int64_t real_thread_id = reg < 2 ? threadid : (threadid ^ xor_val);
                        int64_t idx = (real_thread_id / 4 * 16) + (real_thread_id % 4 * 2) + (reg / 2 * 8) + (reg % 2);
                        int64_t rowidx = idx % (1 << part8_log_had_size);
                        int64_t colidx = idx >> part8_log_had_size;
                        store[rowidx * 128 + colidx] = data;
                    }
                }
                // for size 4k and above, wait to process all chunks so a final store can be performed coalesced
            }

            a_chunk_ptr += 128; // (only affects first 256 size) move on to next chunk by skipping 256 elements in b16 (= 128 in b32)
            out_chunk_ptr += 128;
            if constexpr(log_had_size > 8) {
                b_frag_ptr += (l == 0 ? 128 : (128 >> part8_log_had_size));
            } else { // else is redundant, simplified version of if body, to help compiler warnings
                b_frag_ptr += 128;
            }
        }
        if (log_had_size <= 8)
            break;
    }

    if constexpr(log_had_size >= 12) {
        // for sizes 4k and above, perform final coalesced store after processing all chunks
        #pragma unroll
        for (int64_t j = 0; j < 4; j++) {
            #pragma unroll
            for (int64_t k = 1; k < num_chunks; k++) {
                int64_t threadid_contig = threadid % num_chunks;
                int64_t threadid_mul = threadid / num_chunks;
                int64_t threadid2 = (threadid_contig + k) % num_chunks + threadid_mul * num_chunks; // thread to give your data to
                b_frag_all[k][j] = __shfl_sync(0xFFFFFFFF, b_frag_all[k][j], threadid2);
            }
        }

        // a + threadblock offset + warp offset
        // can then index into all chunks owned by this warp
        b32* store = bfrag_arr + (128 >> part8_log_had_size) * (num_chunks * (blockid % warps_per_block));

        #pragma unroll
        for (int64_t j = 0; j < 4; j++) {
            #pragma unroll
            for (int64_t k = 0; k < num_chunks; k++) {
                // here, j represents register, and k represents 8-offset/chunk
                int64_t real_chunk_num = (num_chunks - (threadid % num_chunks) + k) % num_chunks; // chunk at which you have target thread #'s data

                // b32 data = b_frag_all[real_chunk_num][j]; // target thread data
                b32 data;
                #pragma unroll
                for (int64_t i = 0; i < num_chunks; i++) {
                    if (real_chunk_num == i) data = b_frag_all[i][j];
                }

                int64_t real_thread_id = (threadid / num_chunks) * num_chunks + k; // target thread #
                int64_t chunk_idx = 128 * real_chunk_num; // index due to fetching from another chunk (chunk in which this thread has the target thread's original data)
                int64_t thread_group_idx = (real_thread_id / 4) * 16; // index due to fetching from another group of num_chunk threads (since shuffle is between num_chunk threads)
                int64_t thread_idx = (real_thread_id % 4) * 2; // index due to original thread's position within the group of num_chunk threads
                int64_t reg_idx = (j / 2) * 8 + (j % 2); // index due to target register
                int64_t idx = chunk_idx + thread_group_idx + thread_idx + reg_idx; // final index

                // fix idx for majorness
                int64_t rowidx = idx % (1 << part8_log_had_size);
                int64_t colidx = idx >> part8_log_had_size;

                store[rowidx * 128 + colidx] = data;
            }
        }

        __syncthreads();
        store = ((b32*) out) + (blockid / warps_per_block) * (num_chunks * warps_per_block) * 128;
        int4* store4 = (int4*) store;
        int4* bfrag_arr4 = (int4*) bfrag_arr;
        // flush smem, simply linearly write to store
        // always divisible by 128*32b, so (32*4)*32b is ok
        #pragma unroll
        for (int64_t warp_off = 0; warp_off < (num_chunks * warps_per_block * 128 / 4); warp_off += 32 * warps_per_block) {
            int64_t total_off = warp_off + threadid + (blockid % warps_per_block) * 32;
            store4[total_off] = bfrag_arr4[total_off];
        }
    }

}

constexpr int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

template <torch::headeronly::ScalarType dtype, int64_t chunks_per_warp, int64_t warps_per_block, int64_t log_had_size, int64_t blocks_per_sm, bool check_masking = false>
void __forceinline__ run_kernel(b16* a_mat, b16* out, int64_t num_chunks, cudaStream_t stream) {
    int64_t shared_size = chunks_per_warp * warps_per_block * 128 * 4;
    dim3 block_size = 32 * warps_per_block;

    #define CHECK_SHARED_LIM() {                                                                              \
        if (shared_size > 48 * 1024) {                                                                        \
            STD_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536)); \
        }                                                                                                     \
    }

    if constexpr(check_masking) {
        if (num_chunks % (chunks_per_warp * warps_per_block) != 0) {
            dim3 grid_size = ceil_div(ceil_div(num_chunks, chunks_per_warp), warps_per_block);
            auto kernel = hadamard_transform_kernel<chunks_per_warp, warps_per_block, log_had_size, blocks_per_sm, true, dtype>;
            CHECK_SHARED_LIM();
            kernel<<<dim3(grid_size), dim3(block_size), shared_size, stream>>>(a_mat, out, num_chunks);
        } else {
            dim3 grid_size = num_chunks / chunks_per_warp / warps_per_block;
            auto kernel = hadamard_transform_kernel<chunks_per_warp, warps_per_block, log_had_size, blocks_per_sm, false, dtype>;
            CHECK_SHARED_LIM();
            kernel<<<dim3(grid_size), dim3(block_size), shared_size, stream>>>(a_mat, out, num_chunks);
        }
    } else {
        dim3 grid_size = num_chunks / chunks_per_warp / warps_per_block;
        auto kernel = hadamard_transform_kernel<chunks_per_warp, warps_per_block, log_had_size, blocks_per_sm, false, dtype>;
        CHECK_SHARED_LIM();
        kernel<<<dim3(grid_size), dim3(block_size), shared_size, stream>>>(a_mat, out, num_chunks);
    }

    STD_CUDA_KERNEL_LAUNCH_CHECK();
}

template <torch::headeronly::ScalarType dtype>
void run_fht(void* a_mat_ptr, void* out_ptr, int64_t numel, int64_t had_size, cudaStream_t stream) {
    int64_t num_chunks = numel / 256; // caller required to ensure divisible by 256
    // for size 256, use (2, 1)
    // for size 32k use (8, 16)
    constexpr int64_t chunks_per_warp_small = 1;// 8;
    constexpr int64_t warps_per_block_small = 1;//2;//16;
    constexpr int64_t blocks_per_sm_small = 24;
    constexpr int64_t chunks_per_warp_large = 2;
    constexpr int64_t warps_per_block_large = 1;
    constexpr int64_t blocks_per_sm_large = 24;

    b16* a_mat = (b16*) a_mat_ptr;
    b16* out = (b16*) out_ptr;

    if (numel <= 256) {
        switch (had_size) {
            case (1<<1): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 1, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<2): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 2, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<3): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 3, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<4): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 4, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<5): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 5, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<6): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 6, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<7): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 7, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
            case (1<<8): run_kernel<dtype, chunks_per_warp_small, warps_per_block_small, 8, blocks_per_sm_small>(a_mat, out, num_chunks, stream); break;
        }
    } else {
        switch (had_size) {
            case (1<<1):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 1, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<2):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 2, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<3):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 3, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<4):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 4, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<5):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 5, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<6):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 6, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<7):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 7, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<8):  run_kernel<dtype, chunks_per_warp_large, warps_per_block_large, 8, blocks_per_sm_large, true>(a_mat, out, num_chunks, stream); break;
            case (1<<9):  run_kernel<dtype, launch_configs_big[0][0], launch_configs_big[0][1], 9 , launch_configs_big[0][2]>(a_mat, out, num_chunks, stream); break;
            case (1<<10): run_kernel<dtype, launch_configs_big[1][0], launch_configs_big[1][1], 10, launch_configs_big[1][2]>(a_mat, out, num_chunks, stream); break;
            case (1<<11): run_kernel<dtype, launch_configs_big[2][0], launch_configs_big[2][1], 11, launch_configs_big[2][2]>(a_mat, out, num_chunks, stream); break;
            case (1<<12): run_kernel<dtype, launch_configs_big[3][0], launch_configs_big[3][1], 12, launch_configs_big[3][2]>(a_mat, out, num_chunks, stream); break;
            case (1<<13): run_kernel<dtype, launch_configs_big[4][0], launch_configs_big[4][1], 13, launch_configs_big[4][2]>(a_mat, out, num_chunks, stream); break;
            case (1<<14): run_kernel<dtype, launch_configs_big[5][0], launch_configs_big[5][1], 14, launch_configs_big[5][2]>(a_mat, out, num_chunks, stream); break;
            case (1<<15): run_kernel<dtype, launch_configs_big[6][0], launch_configs_big[6][1], 15, launch_configs_big[6][2]>(a_mat, out, num_chunks, stream); break;
        }
    }
}

template void run_fht<torch::headeronly::ScalarType::Half>(void* a_mat_ptr, void* out_ptr, int64_t numel, int64_t had_size, cudaStream_t stream);
template void run_fht<torch::headeronly::ScalarType::BFloat16>(void* a_mat_ptr, void* out_ptr, int64_t numel, int64_t had_size, cudaStream_t stream);

}  // namespace hadacore

constexpr bool is_power_of_two(int x) { return x && !(x & (x - 1)); }

torch::stable::Tensor hadacore_transform(torch::stable::Tensor& x, bool inplace) {
    auto dtype = x.scalar_type();
    STD_TORCH_CHECK(dtype == torch::headeronly::ScalarType::Half || dtype == torch::headeronly::ScalarType::BFloat16, "Only fp16 and bf16 supported currently");
    STD_TORCH_CHECK(x.is_cuda());

    const int had_size = x.size(-1);
    STD_TORCH_CHECK(is_power_of_two(had_size) && (had_size <= (1U << 15)),
        "Only power of two Hadamard sizes up to 2^15 are supported, got ", had_size);

    const auto res_shape = x.sizes();
    x = torch::stable::reshape(x, {-1, had_size});

    auto numel = x.numel();
    if (numel % 256 != 0) {
        x = torch::stable::pad(x, {0, 0, 0, (256 - numel % 256) / had_size});
    }

    if (x.stride(-1) != 1) {
        x = torch::stable::contiguous(x);
    }
    torch::stable::Tensor out = inplace ? x : torch::stable::empty_like(x);

    int32_t device_index = x.get_device_index();
    const torch::stable::accelerator::DeviceGuard device_guard(device_index);
    auto stream = get_current_cuda_stream(device_index);

    VLLM_STABLE_DISPATCH_HALF_TYPES(x.scalar_type(), "hadacore_transform_runfht", [&] {
      auto constexpr SCALAR_TYPE = torch::headeronly::CppTypeToScalarType<scalar_t>::value;
      hadacore::run_fht<SCALAR_TYPE>(x.data_ptr(), x.data_ptr(), x.numel(), had_size, stream);
    });

    if (numel % 256 != 0) {
        out = torch::stable::narrow(out, 0, 0, numel / had_size);
    }

    if (inplace && out.data_ptr() != x.data_ptr()) {
        torch::stable::copy_(x, torch::stable::view(out, res_shape));
        return x;
    }
    return torch::stable::reshape(out, res_shape);
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
    m.impl("hadacore_transform", TORCH_BOX(&hadacore_transform));
}
