/*
 * Modified by HandH1998
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

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
