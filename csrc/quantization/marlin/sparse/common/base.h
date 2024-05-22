/*
 * Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All
 * Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace marlin_24 {

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

template <int M_, int N_, int K_>
struct ShapeBase {
  static constexpr int M = M_, N = N_, K = K_;
};

using I4 = Vec<int, 4>;

// Matrix fragments for tensor core instructions; their precise layout is
// documented here:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragM = Vec<uint, 1>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;  // quantization scales

}  // namespace marlin_24
