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
    \brief Implementation of a CTA-wide semaphore for inter-CTA synchronization.
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

// namespace cutlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CTA-wide semaphore for inter-CTA synchronization.
class Semaphore
{
public:
  int *lock;
  bool wait_thread;
  int state;

public:
  /// Implements a semaphore to wait for a flag to reach a given value
  __host__ __device__ Semaphore(int *lock_, int thread_id) : lock(lock_),
                                                             wait_thread(thread_id < 0 || thread_id == 0),
                                                             state(-1)
  {
  }

  /// Permit fetching the synchronization mechanism early
  __device__ void fetch()
  {
    if (wait_thread)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#else
      asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
#endif
    }
  }

  /// Gets the internal state
  __device__ int get_state() const
  {
    return state;
  }

  /// Waits until the semaphore is equal to the given value
  __device__ void wait(int status = 0)
  {
    while (__syncthreads_and(state != status))
    {
      fetch();
    }

    __syncthreads();
  }

  /// Updates the lock with the given result
  __device__ void release(int status = 0)
  {
    __syncthreads();

    if (wait_thread)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#else
      asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
#endif
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// } // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////