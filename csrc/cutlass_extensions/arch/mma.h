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
/*! \file
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once
#include "cutlass_extensions/weight_only_quant_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace arch
{

// Tag which triggers MMA which will trigger
struct OpMultiplyAddDequantizeInterleavedBToA;

/*
  Below we have extra tags to signal what kind of dequantization we want to do
  (per col, scale only fine grained, finegrained with zero). This still lets us
  the existing template infrastructure (incl. that in CUTLASS). However, we
  split out the template below into OpMultiplyAddDequantizeInterleavedBToA along
  with the quantization op before instantiating the GEMM pieces.

  Note that this is somewhat of a hack, but it SIGNIFICANTLY reduces the amount of
  code we need to duplicate.
 */
struct OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
struct OpMultiplyAddDequantizeInterleavedBToA_fine_scale;
struct OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias;

// The default just forwards the original operator
template <typename MmaOp, WeightOnlyQuantOp QuantOp_>
struct TagOperator
{
    using TaggedOperator = MmaOp;
};

// Specializations below attach more information to the operator
template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>
{
    using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_percol_scale;
};

template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>
{
    using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_fine_scale;
};

template <>
struct TagOperator<OpMultiplyAddDequantizeInterleavedBToA, WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>
{
    using TaggedOperator = OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias;
};

// Here we instantiate some structs to "detag" the tagged operator. It splits it back to the original
// operator + the extra information. If no extra info was tagged, the dequant op per column scaling
// as a default.
template <typename TaggedMmaOp>
struct DetagOperator
{
    using Operator = TaggedMmaOp;
    static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_percol_scale>
{
    using Operator = OpMultiplyAddDequantizeInterleavedBToA;
    static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_fine_scale>
{
    using Operator = OpMultiplyAddDequantizeInterleavedBToA;
    static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;
};

template <>
struct DetagOperator<OpMultiplyAddDequantizeInterleavedBToA_fine_scalebias>
{
    using Operator = OpMultiplyAddDequantizeInterleavedBToA;
    static constexpr WeightOnlyQuantOp QuantOp = WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS;
};

} // namespace arch
} // namespace cutlass
