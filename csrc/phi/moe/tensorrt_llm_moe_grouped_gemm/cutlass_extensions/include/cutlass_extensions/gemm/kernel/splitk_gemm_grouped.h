/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief based on cutlass/include/cutlass/gemm/kernel/gemm_grouped.h
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace kernel
{

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate
    typename Epilogue_,                   ///! Epilogue
    typename ThreadblockSwizzle_,         ///! Threadblock swizzling function
    GroupScheduleMode GroupScheduleMode_, ///! Type of scheduling to perform
    bool Transposed = false>
struct SplitkGemmGrouped
{
public:
    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
    static bool const kTransposed = Transposed;

    // Optional transpose
    using MapArguments = kernel::detail::MapArguments<typename Mma::IteratorA::Element, typename Mma::IteratorA::Layout,
        Mma::kTransformA, Mma::IteratorA::AccessType::kElements, typename Mma::IteratorB::Element,
        typename Mma::IteratorB::Layout, Mma::kTransformB, Mma::IteratorB::AccessType::kElements, typename Mma::LayoutC,
        kTransposed>;

    // Public-facing type definitions related to operand element type, layout, and complex conjugate
    // operation. Must interact with the 'kTransposed' notion.
    using ElementA = typename MapArguments::ElementA;
    using LayoutA = typename MapArguments::LayoutA;
    using ElementB = typename MapArguments::ElementB;
    using LayoutB = typename MapArguments::LayoutB;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename MapArguments::LayoutC;

    using ElementFinalOutput = typename MapArguments::ElementA;

    static ComplexTransform const kTransformA = MapArguments::kTransformA;
    static ComplexTransform const kTransformB = MapArguments::kTransformB;

    // Type definitions about the mainloop.
    using Operator = typename Mma::Operator;
    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;
    static int const kAlignmentA = MapArguments::kAlignmentA;
    static int const kAlignmentB = MapArguments::kAlignmentB;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    using ProblemVisitor
        = GemmGroupedProblemVisitor<ThreadblockShape, kGroupScheduleMode, kThreadCount, kThreadCount, kTransposed>;

    //
    // Structures
    //

    /// Argument structure
    struct Arguments
    {

        //
        // Data members
        //

        GemmCoord* problem_sizes;
        int problem_count;
        int threadblock_count;

        typename EpilogueOutputOp::Params output_op;

        ElementA** ptr_A;
        ElementB** ptr_B;
        ElementFinalOutput** ptr_C;
        ElementFinalOutput** ptr_D;

        typename LayoutA::Stride::LongIndex* lda;
        typename LayoutB::Stride::LongIndex* ldb;
        typename LayoutC::Stride::LongIndex* ldc;
        typename LayoutC::Stride::LongIndex* ldd;

        // Only used by device-level operator
        GemmCoord* host_problem_sizes;

        // splitK
        int split_k_slices;
        int64_t* splitk_buffer_offsets;

        //
        // Methods
        //

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Arguments()
            : problem_count(0)
            , threadblock_count(0)
            , ptr_A(nullptr)
            , ptr_B(nullptr)
            , ptr_C(nullptr)
            , ptr_D(nullptr)
            , lda(nullptr)
            , ldb(nullptr)
            , ldc(nullptr)
            , ldd(nullptr)
            , host_problem_sizes(nullptr)
            , split_k_slices(1)
            , splitk_buffer_offsets(nullptr)
        {
        }

        /// Ctor
        CUTLASS_HOST_DEVICE
        Arguments(GemmCoord* problem_sizes, int problem_count, int threadblock_count,
            typename EpilogueOutputOp::Params output_op, ElementA** ptr_A, ElementB** ptr_B, ElementFinalOutput** ptr_C,
            ElementFinalOutput** ptr_D, typename LayoutA::Stride::LongIndex* lda,
            typename LayoutB::Stride::LongIndex* ldb, typename LayoutC::Stride::LongIndex* ldc,
            typename LayoutC::Stride::LongIndex* ldd, GemmCoord* host_problem_sizes, int split_k_slices,
            int64_t* splitk_buffer_offsets)
            : problem_sizes(problem_sizes)
            , problem_count(problem_count)
            , threadblock_count(threadblock_count)
            , output_op(output_op)
            , ptr_A(ptr_A)
            , ptr_B(ptr_B)
            , ptr_C(ptr_C)
            , ptr_D(ptr_D)
            , lda(lda)
            , ldb(ldb)
            , ldc(ldc)
            , ldd(ldd)
            , host_problem_sizes(host_problem_sizes)
            , split_k_slices(split_k_slices)
            , splitk_buffer_offsets(splitk_buffer_offsets)
        {
        }
    };

    //
    // Structure for precomputing values in host memory and passing to kernels
    //

    /// Parameters structure
    struct Params
    {

        typename ProblemVisitor::Params problem_visitor;
        int threadblock_count;

        typename EpilogueOutputOp::Params output_op;

        ElementA** ptr_A;
        ElementB** ptr_B;
        ElementFinalOutput** ptr_C;
        ElementFinalOutput** ptr_D;
        ElementC* ptr_C_split;
        ElementC* ptr_D_split;

        typename LayoutA::Stride::LongIndex* lda;
        typename LayoutB::Stride::LongIndex* ldb;
        typename LayoutC::Stride::LongIndex* ldc;
        typename LayoutC::Stride::LongIndex* ldd;

        //
        // Methods
        //

        // splitk
        GemmCoord grid_tiled_shape;
        int swizzle_log_tile;
        int gemm_k_size;
        GemmCoord* host_problem_sizes;
        int split_k_slices;
        int64_t* splitk_buffer_offsets;

        CUTLASS_HOST_DEVICE
        Params()
            : ptr_A(nullptr)
            , ptr_B(nullptr)
            , ptr_C(nullptr)
            , ptr_D(nullptr)
            , ptr_C_split(nullptr)
            , ptr_D_split(nullptr)
            , lda(nullptr)
            , ldb(nullptr)
            , ldc(nullptr)
            , ldd(nullptr)
            , swizzle_log_tile(0)
            , gemm_k_size(0)
            , host_problem_sizes(nullptr)
            , split_k_slices(1)
            , splitk_buffer_offsets(nullptr)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
            : problem_visitor(args.problem_sizes, args.problem_count, workspace, tile_count)
            , host_problem_sizes(args.host_problem_sizes)
            , threadblock_count(args.threadblock_count)
            , output_op(args.output_op)
            , ptr_A(args.ptr_A)
            , ptr_B(args.ptr_B)
            , ptr_C(args.ptr_C)
            , ptr_D(args.ptr_D)
            , ptr_C_split((ElementC*) workspace)
            , ptr_D_split((ElementC*) workspace)
            , lda(args.lda)
            , ldb(args.ldb)
            , ldc(args.ldc)
            , ldd(args.ldd)
            , split_k_slices(args.split_k_slices)
            , splitk_buffer_offsets(args.splitk_buffer_offsets)
        {
            // Determine grid shape
            ThreadblockSwizzle threadblock_swizzle;
            grid_tiled_shape = threadblock_swizzle.get_tiled_shape(args.host_problem_sizes[0],
                {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, args.split_k_slices);
            swizzle_log_tile = ThreadblockSwizzle().get_log_tile(grid_tiled_shape);

            // only support same k
            int full_gemm_k_iterations = args.host_problem_sizes[0].k() / Mma::Shape::kK;
            int gemm_k_iterations = full_gemm_k_iterations / grid_tiled_shape.k();

            gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
        }

        CUTLASS_HOST_DEVICE
        void update(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
        {

            problem_visitor =
                typename ProblemVisitor::Params(args.problem_sizes, args.problem_count, workspace, tile_count);
            threadblock_count = args.threadblock_count;
            output_op = args.output_op;
            ptr_A = args.ptr_A;
            ptr_B = args.ptr_B;
            ptr_C = args.ptr_C;
            ptr_D = args.ptr_D;
            ptr_C_split = workspace;
            ptr_D_split = workspace;

            lda = args.lda;
            ldb = args.ldb;
            ldc = args.ldc;
            ldd = args.ldd;
        }
    };

    /// Shared memory storage structure
    struct SharedStorage
    {
        union
        {
            typename Mma::SharedStorage main_loop;
            typename Epilogue::SharedStorage epilogue;
        } kernel;

        // ProblemVisitor shared storage can't be overlapped with others
        typename ProblemVisitor::SharedStorage problem_visitor;
    };

public:
    //
    // Methods
    //

    CUTLASS_DEVICE
    SplitkGemmGrouped() {}

    /// Determines whether kernel satisfies alignment
    static Status can_implement(cutlass::gemm::GemmCoord const& problem_size)
    {
        return Status::kSuccess;
    }

    static Status can_implement(Arguments const& args)
    {
        return Status::kSuccess;
    }

    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {

        //
        // These types shadow the type-level definitions and support the ability to implement
        // a 'transposed' GEMM that computes the transposed problems.
        //
        using ElementA = typename Mma::IteratorA::Element;
        using LayoutA = typename Mma::IteratorA::Layout;
        using ElementB = typename Mma::IteratorB::Element;
        using LayoutB = typename Mma::IteratorB::Layout;
        using ElementC = typename Epilogue::OutputTileIterator::Element;
        using LayoutC = typename Epilogue::OutputTileIterator::Layout;

        //
        // Problem visitor.
        //
        ProblemVisitor problem_visitor(params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

        // Outer 'persistent' loop to iterate over tiles
        while (problem_visitor.next_tile())
        {

            GemmCoord problem_size = problem_visitor.problem_size();
            int32_t problem_idx = problem_visitor.problem_index();
            int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

            GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

            // Load element pointers. Exchange pointers and strides if working on the transpose
            ElementA* ptr_A
                = reinterpret_cast<ElementA*>((kTransposed ? params.ptr_B[problem_idx] : params.ptr_A[problem_idx]));
            typename LayoutA::LongIndex ldm_A = (kTransposed ? params.ldb[problem_idx] : params.lda[problem_idx]);

            ElementB* ptr_B
                = reinterpret_cast<ElementB*>((kTransposed ? params.ptr_A[problem_idx] : params.ptr_B[problem_idx]));
            typename LayoutB::LongIndex ldm_B = (kTransposed ? params.lda[problem_idx] : params.ldb[problem_idx]);

            // Compute threadblock location
            ThreadblockSwizzle threadblock_swizzle;
            GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

            cutlass::gemm::GemmCoord threadblock_offset(int(threadblock_idx / grid_shape.n()) * Mma::Shape::kM,
                int(threadblock_idx % grid_shape.n()) * Mma::Shape::kN, 0);

            // Compute initial location in logical coordinates
            cutlass::MatrixCoord tb_offset_A{
                threadblock_offset.m(),
                threadblock_tile_offset.k() * params.gemm_k_size,
            };

            cutlass::MatrixCoord tb_offset_B{threadblock_tile_offset.k() * params.gemm_k_size, threadblock_offset.n()};

            // Problem size is a function of threadblock index in the K dimension
            int problem_size_k;
            if (threadblock_tile_offset.k() + 1 == params.grid_tiled_shape.k())
            {
                problem_size_k = problem_size.k();
            }
            else
            {
                problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
            }

            // Compute threadblock-scoped matrix multiply-add
            int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

            // Compute position within threadblock
            int thread_idx = threadIdx.x;

            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(
                LayoutA(ldm_A), ptr_A, {problem_size.m(), problem_size_k}, thread_idx, tb_offset_A);

            typename Mma::IteratorB iterator_B(
                LayoutB(ldm_B), ptr_B, {problem_size_k, problem_size.n()}, thread_idx, tb_offset_B);

            typename Mma::FragmentC accumulators;

            accumulators.clear();

            // Broadcast the warp_id computed by lane 0 to ensure dependent code
            // is compiled as warp-uniform.
            int warp_idx = canonical_warp_idx_sync();

            int lane_idx = threadIdx.x % 32;

            //
            // Matrix multiply phase
            //

            // Construct thread-scoped matrix multiply
            Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

            // Wait for all threads to finish their epilogue phases from the previous tile.
            __syncthreads();

            // Compute threadblock-scoped matrix multiply-add
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

            //
            // Epilogue
            //

            EpilogueOutputOp output_op(params.output_op);

            ElementC* ptr_C = params.ptr_C_split;
            ElementC* ptr_D = params.ptr_D_split;

            LayoutC layout_C(params.ldc[problem_idx]);
            LayoutC layout_D(params.ldd[problem_idx]);

            typename Epilogue::OutputTileIterator::Params params_C(layout_C);
            typename Epilogue::OutputTileIterator::Params params_D(layout_D);

            // assume identity swizzle
            MatrixCoord threadblock_offset_C(threadblock_offset.m(), threadblock_offset.n());

            // Tile iterator loading from source tensor.
            typename Epilogue::OutputTileIterator iterator_C(
                params_C, ptr_C, problem_size.mn(), thread_idx, threadblock_offset_C);

            iterator_C.add_pointer_offset(problem_size.m() * problem_size.n() * threadblock_tile_offset.k()
                + gridDim.z * params.splitk_buffer_offsets[problem_idx]);

            // Tile iterator writing to destination tensor.
            typename Epilogue::OutputTileIterator iterator_D(
                params_D, ptr_D, problem_size.mn(), thread_idx, threadblock_offset_C);
            iterator_D.add_pointer_offset(problem_size.m() * problem_size.n() * threadblock_tile_offset.k()
                + gridDim.z * params.splitk_buffer_offsets[problem_idx]);

            Epilogue epilogue(shared_storage.kernel.epilogue, thread_idx, warp_idx, lane_idx);

            // Execute the epilogue operator to update the destination tensor.
            epilogue(output_op, iterator_D, accumulators, iterator_C);

            // Next tile
            problem_visitor.advance(gridDim.x);
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
