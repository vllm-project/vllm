/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Boost-like numeric conversion operator for int8 interleaved in a register
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace cutlass
{

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.
template <typename T, typename S, int N>
struct FastInterleavedAndBiasedNumericArrayConverter
{
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, 4>
{
    using result_type = Array<half_t, 4>;
    using source_type = Array<uint8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
        uint32_t* h = reinterpret_cast<uint32_t*>(&result);

        asm volatile("{                                      \n"
                    ".reg .b32 a<2>, b<2>;                  \n"  // if input = 0xf1f2f3f4
                    "prmt.b32 a0, 0, %2, 0x5040;            \n"  // a0 = 0xf300f400
                    "prmt.b32 a1, 0, %2, 0x7060;            \n"  // a1 = 0xf100f200
                    "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n"  // b0 = a0 & 0x7fff7fff
                    "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n"  // (strip sign)
                    "shr.b32  b0, b0, 1;                    \n"  // b0 >>= 1
                    "shr.b32  b1, b1, 1;                    \n"  // shift into fp16 position
                    "add.u32  b0, b0, 0x20002000;           \n"  // b0.exp += 2**4-2**3 // exponent compensate = 8
                    "add.u32  b1, b1, 0x20002000;           \n"  // b1 += 8<<10 | 8<<10<<16
                    "lop3.b32 %0, b0, 0x80008000, a0, 0xf8; \n"  // out0 = b0|(0x80008000&a0)
                    "lop3.b32 %1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
                    "}                                      \n" : "=r"(h[0]) , "=r"(h[1]):"r"(i8s));

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, N>
{
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using result_type = Array<half_t, N>;
    using source_type = Array<uint8_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i)
        {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, 4>
{
    using result_type = Array<bfloat16_t, 4>;
    using source_type = Array<uint8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);
        uint32_t* h = reinterpret_cast<uint32_t*>(&result);

        asm volatile( "{                                      \n"
            ".reg .b32 a<2>, b<2>;                  \n"  // if input = // 0xf1f2f3f4
            "prmt.b32 a0, 0, %2, 0x5040;            \n"  // a0 = 0xf300f400
            "prmt.b32 a1, 0, %2, 0x7060;            \n"  // a1 = 0xf100f200
            "and.b32 b0, a0, 0x7fff7fff;            \n"  // b0 = a0 & 0x7fff7fff
            "and.b32 b1, a1, 0x7fff7fff;            \n"  // (strip sign)
            "shr.b32 b0, b0, 4;                     \n"  // b0 >>= 4
            "shr.b32 b1, b1, 4;                     \n"  // shift into fp16 // position
            "add.u32 b0, b0, 0x3c003c00;            \n"  // b0.exp += 2**7-2**3 // exponent compensate // = 120
            "add.u32 b1, b1, 0x3c003c00;            \n"  // b1 += 120<<7 | // 120<<7<<16
            "lop3.b32 %0, b0, 0x80008000, a0, 0xf8; \n"  // out0 = // b0|(0x80008000&a0)
            "lop3.b32 %1, b1, 0x80008000, a1, 0xf8; \n"  // (restore sign)
            "}                                      \n" : "=r"(h[0]), "=r"(h[1]) : "r"(i8s));

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template <int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, N>
{
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using result_type = Array<bfloat16_t, N>;
    using source_type = Array<uint8_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result* result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i)
        {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
