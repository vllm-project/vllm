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

#include <curand_kernel.h>

#include "cutlass/cutlass.h"

namespace cutlass {
namespace reference {
namespace device {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize tensor to uniform random distribution
template <typename T>
__global__ void TensorInitializeUniform(
    Distribution dist, int64_t seed, int dim_contiguous, int dim_strided, T *tensor, int ldm) {
  __shared__ curandState_t rng_state[1024];

  uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;

  curand_init(seed, gtid, 0, &rng_state[threadIdx.x]);

  int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int s_idx = blockIdx.y * blockDim.x;

  tensor += s_idx * ldm + c_idx;

  for (int s_offset = 0; s_offset < blockDim.x; ++s_offset, ++s_idx) {
    if (s_idx < dim_strided && c_idx < dim_contiguous) {
      double range = dist.uniform.max - dist.uniform.min;

      double rnd = curand_uniform(&rng_state[threadIdx.x]);

      rnd = dist.uniform.min + range * rnd;

      // Random values are cast to integer after scaling by a power of two to facilitate error
      // testing
      if (dist.int_scale >= 0) {
        rnd = double(int(rnd * double(1 << dist.int_scale)));
        *tensor = T(rnd / double(1 << dist.int_scale));
      } else {
        *tensor = T(rnd);
      }

      tensor += ldm;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize tensor to uniform distribution
template <typename T>
__global__ void TensorInitializeGaussian(
    Distribution dist, int64_t seed, int dim_contiguous, int dim_strided, T *tensor, int ldm) {
  __shared__ curandState_t rng_state[1024];

  uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;

  curand_init(seed, gtid, 0, &rng_state[threadIdx.x]);

  int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int s_idx = blockIdx.y * blockDim.x;

  tensor += s_idx * ldm + c_idx;

  for (int s_offset = 0; s_offset < blockDim.x; ++s_offset, ++s_idx) {
    if (s_idx < dim_strided && c_idx < dim_contiguous) {
      // Random values are cast to integer after scaling by a power of two to facilitate error
      // testing

      double rnd = curand_normal(&rng_state[threadIdx.x]);

      rnd = dist.gaussian.mean + dist.gaussian.stddev * rnd;

      if (dist.int_scale >= 0) {
        rnd = double(int(rnd * double(1 << dist.int_scale)));
        *tensor = T(rnd / double(1 << dist.int_scale));
      } else {
        *tensor = T(rnd);
      }
    }
  }
}

/// Kernel to initialize tensor to an identity matrix
template <typename T>
__global__ void TensorInitializeLinear(
    Distribution dist, int64_t seed, int dim_contiguous, int dim_strided, T *tensor, int ldm) {
  __shared__ curandState_t rng_state[1024];

  uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;

  curand_init(seed, gtid, 0, &rng_state[threadIdx.x]);

  int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int s_idx = blockIdx.y * blockDim.x;

  tensor += s_idx * ldm + c_idx;

  for (int s_offset = 0; s_offset < blockDim.x; ++s_offset, ++s_idx) {
    if (s_idx < dim_strided && c_idx < dim_contiguous) {
      *tensor =
          dist.linear.offset + dist.linear.delta_row * c_idx + dist.linear.delta_column * s_idx;
    }
  }
}

/// Kernel to initialize tensor to an identity matrix
template <typename T>
__global__ void TensorInitializeIdentity(
    Distribution dist, int64_t seed, int dim_contiguous, int dim_strided, T *tensor, int ldm) {
  __shared__ curandState_t rng_state[1024];

  uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;

  curand_init(seed, gtid, 0, &rng_state[threadIdx.x]);

  int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int s_idx = blockIdx.y * blockDim.x;

  tensor += s_idx * ldm + c_idx;

  for (int s_offset = 0; s_offset < blockDim.x; ++s_offset, ++s_idx) {
    if (s_idx < dim_strided && c_idx < dim_contiguous) {
      *tensor = (c_idx == s_idx ? T(1) : T(0));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace device
} // namespace reference
} // namespace cutlass
