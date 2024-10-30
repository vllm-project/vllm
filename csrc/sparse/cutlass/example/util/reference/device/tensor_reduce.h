/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/detail/linear_to_coordinate.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp,
  int kBlockSize = 128
>
__global__ void TensorTransformReducePartial(
  TensorView<Element, Layout> view,     /// View of the tensor to reduce over
  ComputeType identity,                 /// Identity element of the reduction operation
  ReduceOp reduce,                      /// Reduces an accumulated value with a transformed element: f(ComputeType, ComputeType) => ComputeType
  TransformOp transform,                /// Transforms the tensor element to ComputeType: g(Element) => ComputeType
  ComputeType *workspace) {             /// Device-side workspace for accumulating partial results. The reduced element is stored in workspace[0]
  
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t size = view.size();

  __shared__ ComputeType scratchpad[kBlockSize];

  for (; idx < size; idx += blockDim.x * gridDim.x) {

    // Map linear thread ID onto tensor coordinate
    typename Layout::TensorCoord coord;

    cutlass::reference::detail::LinearToCoordinate<Layout::kRank>()(coord, idx, view.extent());

    if (view.contains(coord)) {

      // Fetch element
      Element x = view.at(coord);

      // Transform 
      identity = reduce(identity, transform(x));
    }
  }

  scratchpad[threadIdx.x] = identity;

  __syncthreads();

  // One thread performs the final reduction and stores out. This could be enhanced via
  // a tree reduction and pipelining.
  if (threadIdx.x == 0) {

    for (int i = 1; i < kBlockSize; ++i) {
      identity = reduce(identity, scratchpad[i]);
    }
    
    workspace[blockIdx.x] = identity;
  }
}

template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp,
  int kBlockSize = 128
>
__global__ void TensorTransformReducePartial(
  TensorView<Element, Layout> view_A,   /// View of the tensor to reduce over
  TensorView<Element, Layout> view_B,   /// View of the tensor to reduce over
  ComputeType identity,                 /// Identity element of the reduction operation
  ReduceOp reduce,                      /// Reduces an accumulated value with a transformed element: f(ComputeType, ComputeType) => ComputeType
  TransformOp transform,                /// Transforms the tensor element to ComputeType: g(Element) => ComputeType
  ComputeType *workspace) {             /// Device-side workspace for accumulating partial results. The reduced element is stored in workspace[0]
  
  int64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  auto size = static_cast<int64_t>(view_A.size());

  __shared__ ComputeType scratchpad[kBlockSize];

  for (; idx < size; idx += blockDim.x * gridDim.x) {

    // Map linear thread ID onto tensor coordinate
    typename Layout::TensorCoord coord;

    cutlass::reference::detail::LinearToCoordinate<Layout::kRank>()(coord, idx, view_A.extent());

    if (view_A.contains(coord)) {

      // Fetch element
      Element a = view_A.at(coord);
      Element b = view_B.at(coord);

      // Transform 
      identity = reduce(identity, transform(a, b));
    }
  }

  scratchpad[threadIdx.x] = identity;

  __syncthreads();

  // One thread performs the final reduction and stores out. This could be enhanced via
  // a tree reduction and pipelining.
  if (threadIdx.x == 0) {

    for (int i = 1; i < kBlockSize; ++i) {
      identity = reduce(identity, scratchpad[i]);
    }

    workspace[blockIdx.x] = identity;
  }
}


template <
  typename ComputeType,
  typename ReduceOp,
  int kBlockSize = 32
>
__global__ void TensorTransformReduceFinalize(
  ComputeType *workspace, 
  ComputeType identity,
  int workspace_size,
  ReduceOp reduce) {

  __shared__ ComputeType scratchpad[kBlockSize];

  for (int idx = threadIdx.x; idx < workspace_size; idx += kBlockSize) {
    identity = reduce(identity, workspace[idx]);
  }

  scratchpad[threadIdx.x] = identity;

  __syncthreads();

  if (threadIdx.x == 0) {

    for (int i = 1; i < kBlockSize; ++i) {
      identity = reduce(identity, scratchpad[i]);
    }

    workspace[0] = identity;
  }
}

} // namespace kernel

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Transform-reduce operation over the elements of a tensor
template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorView<Element, Layout> view,     /// View of the tensor to reduce over
  ComputeType identity,                 /// Identity element of the reduction operation
  ReduceOp reduce,                      /// Reduces an accumulated value with a transformed element: f(ComputeType, ComputeType) => ComputeType
  TransformOp transform,                /// Transforms the tensor element to ComputeType: g(Element) => ComputeType
  ComputeType *workspace,               /// Device-side workspace for accumulating partial results. The reduced element is stored in workspace[0]
  int workspace_size,                   /// Number of elements in workspace
  cudaStream_t stream = nullptr,        /// CUDA stream to launch into
  bool copy_out = true                  /// If true, the value of workspace[0] is copied to host and returned. Otherwise, `identity` is returned.
) {

  int const kBlockSize = 128;

  dim3 block(kBlockSize, 1);
  dim3 grid(workspace_size, 1);

  kernel::TensorTransformReducePartial<
    Element, Layout, ComputeType, ReduceOp, TransformOp, kBlockSize
  ><<< grid, block, 0, stream >>>(
    view, identity, reduce, transform, workspace
  );

  int const kFinalizeBlockSize = 32;

  kernel::TensorTransformReduceFinalize<
    ComputeType, ReduceOp, kFinalizeBlockSize
  ><<< dim3(1, 1), dim3(kFinalizeBlockSize, 1), 0, stream >>>(
    workspace, identity, workspace_size, reduce
  );

  if (copy_out) {
    cudaError_t result = cudaMemcpy(&identity, workspace, sizeof(identity), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy() failed");
    }
  }

  return identity;
}

/// Transform-reduce operation over the elements of two tensors, zipped together
template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorView<Element, Layout> view_A,   /// View of the tensor to reduce over
  TensorView<Element, Layout> view_B,   /// View of the tensor to reduce over
  ComputeType identity,                 /// Identity element of the reduction operation
  ReduceOp reduce,                      /// Reduces an accumulated value with a transformed element: f(ComputeType, ComputeType) => ComputeType
  TransformOp transform,                /// Transforms the tensor element to ComputeType: g(Element) => ComputeType
  ComputeType *workspace,               /// Device-side workspace for accumulating partial results. The reduced element is stored in workspace[0]
  int workspace_size,                   /// Number of elements in workspace
  cudaStream_t stream = nullptr,        /// CUDA stream to launch into
  bool copy_out = true                  /// If true, the value of workspace[0] is copied to host and returned. Otherwise, `identity` is returned.
) {

  if (view_A.extent() != view_B.extent()) {
    throw std::runtime_error("Extents must be equal.");
  }

  int const kBlockSize = 128;

  dim3 block(kBlockSize, 1);
  dim3 grid(workspace_size, 1);

  kernel::TensorTransformReducePartial<
    Element, Layout, ComputeType, ReduceOp, TransformOp, kBlockSize
  ><<< grid, block, 0, stream >>>(
    view_A, view_B, identity, reduce, transform, workspace
  );

  int const kFinalizeBlockSize = 32;

  kernel::TensorTransformReduceFinalize<
    ComputeType, ReduceOp, kFinalizeBlockSize
  ><<< dim3(1, 1), dim3(kFinalizeBlockSize, 1), 0, stream >>>(
    workspace, identity, workspace_size, reduce
  );

  if (copy_out) {
    cudaError_t result = cudaMemcpy(&identity, workspace, sizeof(identity), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy() failed");
    }
  }

  return identity;
}

/// Transform-reduce operation over the elements of a tensor. This helper allocates the device-side
/// workspace
template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorView<Element, Layout> view,
  ComputeType identity,            
  ReduceOp reduce,                 
  TransformOp transform,
  cudaStream_t stream = nullptr, 
  int workspace_size = 0           
) {

  // Optionally query for the SM count to size the workspace.
  if (!workspace_size) {

    int device_idx = 0;
    cudaDeviceProp prop;

    cudaError_t result = cudaGetDevice(&device_idx);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() failed");
    }

    result = cudaGetDeviceProperties(&prop, device_idx);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProp() failed");
    }

    workspace_size = int(prop.multiProcessorCount);
  }

  DeviceAllocation<ComputeType> workspace(workspace_size);

  ComputeType output = TensorTransformReduce(
    view, 
    identity, 
    reduce, 
    transform, 
    workspace.get(), 
    workspace_size, 
    stream, 
    true);

  return output;
}


/// Transform-reduce operation over the elements of a tensor. This helper allocates the device-side
/// workspace
template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorView<Element, Layout> view_A,
  TensorView<Element, Layout> view_B,
  ComputeType identity,            
  ReduceOp reduce,                 
  TransformOp transform,
  cudaStream_t stream = nullptr, 
  int workspace_size = 0           
) {

  // Optionally query for the SM count to size the workspace.
  if (!workspace_size) {

    int device_idx = 0;
    cudaDeviceProp prop;

    cudaError_t result = cudaGetDevice(&device_idx);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDevice() failed");
    }

    result = cudaGetDeviceProperties(&prop, device_idx);
    if (result != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProp() failed");
    }

    workspace_size = int(prop.multiProcessorCount);
  }

  DeviceAllocation<ComputeType> workspace(workspace_size);

  ComputeType output = TensorTransformReduce(
    view_A,
    view_B, 
    identity, 
    reduce, 
    transform, 
    workspace.get(), 
    workspace_size, 
    stream, 
    true);

  return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to compute the sum of the elements of a tensor
template <
  typename Element,
  typename Layout,
  typename ComputeType = Element
>
ComputeType TensorSum(
  TensorView<Element, Layout> view,
  ComputeType identity = ComputeType(),
  cudaStream_t stream = nullptr,
  int workspace_size = 0
) {

  plus<ComputeType> reduce;
  NumericConverter<ComputeType, Element> transform;

  return TensorTransformReduce(
    view, identity, reduce, transform, stream, workspace_size);
}

/// Helper to compute the sum of the squares of the elements of a tensor
template <
  typename Element,
  typename Layout,
  typename ComputeType = Element
>
ComputeType TensorSumSq(
  TensorView<Element, Layout> view,
  ComputeType identity = ComputeType(),
  cudaStream_t stream = nullptr,
  int workspace_size = 0
) {

  plus<ComputeType> reduce;
  magnitude_squared<Element, ComputeType> transform;

  return TensorTransformReduce(
    view, identity, reduce, transform, stream, workspace_size);
}

/// Helper to compute the norm of the elements of a tensor.
template <
  typename Element,
  typename Layout,
  typename ComputeType = double
>
ComputeType TensorNorm(
  TensorView<Element, Layout> view,
  ComputeType identity = ComputeType(),
  cudaStream_t stream = nullptr,
  int workspace_size = 0
) {

  return std::sqrt(TensorSumSq(view, identity, stream, workspace_size));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to compute the sum of the squares of the differences of two tensors
template <
  typename Element,
  typename Layout,
  typename ComputeType = double
>
ComputeType TensorSumSqDiff(
  TensorView<Element, Layout> view_A,
  TensorView<Element, Layout> view_B,
  ComputeType identity = ComputeType(),
  cudaStream_t stream = nullptr,
  int workspace_size = 0
) {

  plus<ComputeType> reduce;
  magnitude_squared_difference<Element, ComputeType> transform;

  return TensorTransformReduce(
    view_A, view_B, identity, reduce, transform, stream, workspace_size);
}


/// Helper to compute the norm of the tensor computed as the difference of two tensors in memory
template <
  typename Element,
  typename Layout,
  typename ComputeType = double
>
ComputeType TensorNormDiff(
  TensorView<Element, Layout> view_A,
  TensorView<Element, Layout> view_B,
  ComputeType identity = ComputeType(),
  cudaStream_t stream = nullptr,
  int workspace_size = 0
) {

  return std::sqrt(TensorSumSqDiff(view_A, view_B, identity, stream, workspace_size));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
