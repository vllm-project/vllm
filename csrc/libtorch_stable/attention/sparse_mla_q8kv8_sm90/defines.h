/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
using cutlass::arch::fence_barrier_init;
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
