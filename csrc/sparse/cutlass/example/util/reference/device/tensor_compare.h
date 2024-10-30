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
/* \file
  \brief Defines host-side elementwise operations on TensorView.
*/

#pragma once
// Standard Library includes
#include <utility>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/relatively_equal.h"

#include "cutlass/util/distribution.h"

#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace device {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

template <typename Element>
__global__ void BlockCompareEqual(
  int *equal, 
  Element const *ptr_A,
  Element const *ptr_B,
  size_t capacity) {

  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  for (; idx < capacity; idx += gridDim.x * blockDim.x) {

    Element a = cutlass::ReferenceFactory<Element>::get(ptr_A, idx);
    Element b = cutlass::ReferenceFactory<Element>::get(ptr_B, idx);

    if (a != b) {
      *equal = 0;

      return;
    }
  }
}

template <typename Element>
__global__ void BlockCompareRelativelyEqual(
  int *equal, 
  Element const *ptr_A,
  Element const *ptr_B,
  size_t capacity,
  Element epsilon,
  Element nonzero_floor) {

  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  for (; idx < capacity; idx += gridDim.x * blockDim.x) {

    Element a = cutlass::ReferenceFactory<Element>::get(ptr_A, idx);
    Element b = cutlass::ReferenceFactory<Element>::get(ptr_B, idx);

    if (!relatively_equal(a, b, epsilon, nonzero_floor)) {
      *equal = 0;
      return;
    }
  }
}

} // namespace kernel


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Performs a bit-level equality check between two blocks
template <typename Element>
bool BlockCompareEqual(
  Element const *ptr_A,
  Element const *ptr_B,
  size_t capacity,
  int grid_size = 0, 
  int block_size = 0) {

  int equal_flag = 1;
  int *device_equal_flag = nullptr;

  if (cudaMalloc((void **)&device_equal_flag, sizeof(int)) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device flag.");
  }

  if (cudaMemcpy(
    device_equal_flag, 
    &equal_flag, 
    sizeof(int), 
    cudaMemcpyHostToDevice) != cudaSuccess) {

    throw std::runtime_error("Failed to copy equality flag to device.");
  }

  if (!grid_size || !block_size) {

    // if grid_size or block_size are zero, query occupancy using the CUDA Occupancy API
    cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
      &grid_size,
      &block_size,
      reinterpret_cast<void const *>(kernel::BlockCompareEqual<Element>));

    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to query occupancy.");
    }

    // Limit block size. This has the effect of increasing the number of items processed by a
    // single thread and reduces the impact of initialization overhead.
    block_size = (block_size < 128 ? block_size : 128);
  }

  dim3 grid(grid_size, 1, 1);
  dim3 block(block_size, 1, 1);

  kernel::BlockCompareEqual<Element><<< grid, block >>>(device_equal_flag, ptr_A, ptr_B, capacity);

  if (cudaMemcpy(
    &equal_flag, 
    device_equal_flag,
    sizeof(int), 
    cudaMemcpyDeviceToHost) != cudaSuccess) {
    
    cudaFree(device_equal_flag);

    throw std::runtime_error("Failed to copy equality flag from device.");
  }

  cudaFree(device_equal_flag);

  return equal_flag;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Performs a bit-level equality check between two blocks
template <typename Element>
bool BlockCompareRelativelyEqual(
  Element const *ptr_A,
  Element const *ptr_B,
  size_t capacity,
  Element epsilon,
  Element nonzero_floor,
  int grid_size = 0, 
  int block_size = 0) {

  int equal_flag = 1;
  int *device_equal_flag = nullptr;

  if (cudaMalloc((void **)&device_equal_flag, sizeof(int)) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device flag.");
  }

  if (cudaMemcpy(
    device_equal_flag, 
    &equal_flag, 
    sizeof(int), 
    cudaMemcpyHostToDevice) != cudaSuccess) {

    throw std::runtime_error("Failed to copy equality flag to device.");
  }

  if (!grid_size || !block_size) {

    // if grid_size or block_size are zero, query occupancy using the CUDA Occupancy API
    cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
      &grid_size,
      &block_size,
      reinterpret_cast<void const *>(kernel::BlockCompareRelativelyEqual<Element>));

    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to query occupancy.");
    }

    // Limit block size. This has the effect of increasing the number of items processed by a
    // single thread and reduces the impact of initialization overhead.
    block_size = (block_size < 128 ? block_size : 128);
  }

  dim3 grid(grid_size, 1, 1);
  dim3 block(block_size, 1, 1);

  kernel::BlockCompareRelativelyEqual<Element><<< grid, block >>>(
    device_equal_flag, 
    ptr_A, 
    ptr_B, 
    capacity, 
    epsilon, 
    nonzero_floor
  );

  if (cudaMemcpy(
    &equal_flag, 
    device_equal_flag,
    sizeof(int), 
    cudaMemcpyDeviceToHost) != cudaSuccess) {
    
    cudaFree(device_equal_flag);

    throw std::runtime_error("Failed to copy equality flag from device.");
  }

  cudaFree(device_equal_flag);

  return equal_flag;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // device
} // reference
} // cutlass
