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
    \brief Templates for visiting scales to be used when dequantizing the weights for weight-only GEMM
           quantization.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator_params.h"

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace transform
{
namespace threadblock
{

////////////////////////////////////////////////////////////////////////////////

template <typename Shape, typename Element, typename Layout, int AdvanceRank, int Alignment>
class FineGrainedScaleZeroIterator;

template <typename Shape_, typename Element_, int Alignment_>
class FineGrainedScaleZeroIterator<Shape_, Element_, layout::RowMajor, 0, Alignment_>
{
public:
    using Shape = Shape_;
    using Element = Element_;
    using Layout = layout::RowMajor;
    static int const kAdvanceRank = 0;
    static int const kAlignment = Alignment_;

    static int const kAccessesPerVector = 1;

    /// Row index of scales corresponding to the groupsize of 64
    int row_groupsize64_;
    int group_size_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = TensorRef<Element, Layout>;
    using TensorView = TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;
    using Pointer = Element*;
    using NonConstPointer = typename platform::remove_const<Element>::type*;

    using AccessType = AlignedArray<Element, kAlignment>;

    // For compatibility with existing iterator interface
    struct Params
    {
        LongIndex stride_ = 0;

        /// amount (in byte) to increment pointer from first access of current tile
        /// to first access of next tile
        LongIndex inc_advance_ = 0;

        // Default ctor
        CUTLASS_HOST_DEVICE
        Params() {}

        /// Construct the Params object given a pitch-linear tensor's layout
        CUTLASS_HOST_DEVICE
        Params(Layout const& layout)
            : stride_(layout.stride(0))
        {
            inc_advance_ = Shape::kRow * stride_ * sizeof_bits<Element>::value / 8;
        }
    };

private:
    /// Internal pointer type permits fast address arithmetic
    using BytePointer = char*;

private:
    //
    // Data members
    //

    /// Parameters object with precomputed internal state
    Params const params_;

    /// Internal pointer to first access of tile
    BytePointer pointer_scale_;
    BytePointer pointer_zero_;

    bool is_valid_ = false;

public:
    /// Constructs a TileIterator from its precomputed state, threadblock offset,
    /// and thread ID
    CUTLASS_DEVICE
    FineGrainedScaleZeroIterator(
        ///< Precomputed parameters object
        Params const& params,
        ///< Pointer to start of scale tensor
        Pointer pointer_scale,
        ///< Pointer to start of zero tensor
        Pointer pointer_zero,
        ///< Extent of the scale and bias
        TensorCoord extent,
        ///< ID of each participating thread
        int thread_id,
        ///< Initial offset of threadblock
        TensorCoord const& threadblock_offset,
        ///< Group size
        int group_size)
        : params_(params)
        , pointer_scale_(reinterpret_cast<BytePointer>(const_cast<NonConstPointer>(pointer_scale)))
        , pointer_zero_(reinterpret_cast<BytePointer>(const_cast<NonConstPointer>(pointer_zero)))
    {
        row_groupsize64_ = threadblock_offset.row();
        group_size_ = group_size;

        const LongIndex tb_row_byte_offset
            = threadblock_offset.row() / (group_size / 64) * params_.stride_ * sizeof_bits<Element>::value / 8;
        const LongIndex tb_col_byte_offset = threadblock_offset.column() * sizeof_bits<Element>::value / 8;
        pointer_scale_ += (tb_row_byte_offset + tb_col_byte_offset);

        if (pointer_zero_ != nullptr)
        {
            pointer_zero_ += (tb_row_byte_offset + tb_col_byte_offset);
        }

        static constexpr int THREADS_PER_ROW = Shape::kColumn / kAlignment;

        const int thread_row = thread_id / THREADS_PER_ROW;
        const int thread_col = thread_id % THREADS_PER_ROW;

        const LongIndex thread_row_byte_offset = thread_row * params_.stride_ * sizeof_bits<Element>::value / 8;
        const LongIndex thread_col_byte_offset = thread_col * kAlignment * sizeof_bits<Element>::value / 8;
        pointer_scale_ += (thread_row_byte_offset + thread_col_byte_offset);
        if (pointer_zero_ != nullptr)
        {
            pointer_zero_ += (thread_row_byte_offset + thread_col_byte_offset);
        }

        // For the rows, we must check that we are within the extent AND the tile to avoid extra reads on
        // a given iteration. The same threads will be responsible for issues reads since the number of scales
        // read in a given iteration is a constant. Therefore, we should never have to update is_valid_
        // outside of the constructor.
        const int global_row = threadblock_offset.row() + thread_row;
        const int global_col = threadblock_offset.column() + thread_col * kAlignment;

        const bool row_in_bounds = global_row < extent.row() && thread_row < Shape::kRow;
        const bool col_in_bounds = global_col < extent.column();

        is_valid_ = row_in_bounds && col_in_bounds;
    }

    /// Construct a PredicatedTileAccessIterator with zero threadblock offset
    CUTLASS_HOST_DEVICE FineGrainedScaleZeroIterator(Params const& params, ///< Precomputed parameters object
        Pointer pointer_scale,                                             ///< Pointer to start of scale tensor
        Pointer pointer_zero,                                              ///< Pointer to start of zero tensor
        TensorCoord extent,                                                ///< Extent of tensor
        int thread_id,                                                     ///< ID of each participating thread
        int group_size)
        : FineGrainedScaleZeroIterator(
            params, pointer_scale, pointer_zero, extent, thread_id, make_Coord(0, 0), group_size)
    {
    }

    CUTLASS_DEVICE
    void add_tile_offset(TensorCoord const& tile_offset)
    {
        const LongIndex row_byte_offset = tile_offset.row() * params_.inc_advance_;
        const LongIndex col_byte_offset = tile_offset.column() * Shape::kColumn * sizeof_bits<Element>::value / 8;
        pointer_scale_ += row_byte_offset + col_byte_offset;
        if (pointer_zero_ != nullptr)
        {
            pointer_zero_ += row_byte_offset + col_byte_offset;
        }
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE void clear_mask(bool enable = true)
    {
        is_valid_ &= (!enable);
    }

    /// Returns whether access is valid or not
    CUTLASS_HOST_DEVICE
    bool valid() const
    {
        return is_valid_;
    }

    /// Returns a scale pointer
    CUTLASS_HOST_DEVICE
    AccessType* get_scale() const
    {
        return reinterpret_cast<AccessType*>(pointer_scale_);
    }

    /// Returns a zero pointer
    CUTLASS_HOST_DEVICE
    AccessType* get_zero() const
    {
        return reinterpret_cast<AccessType*>(pointer_zero_);
    }
};

} // namespace threadblock
} // namespace transform
} // namespace cutlass
